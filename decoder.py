import torch.nn.functional as F
import torch
import math

class DecDecoder(object):
    def __init__(self, K, conf_thresh, num_classes):
        self.K = K
        self.conf_thresh = conf_thresh
        self.num_classes = num_classes

    def _topk(self, scores):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), self.K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds // width).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), self.K)
        topk_clses = (topk_ind // self.K).int()
        topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, self.K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def _nms(self, heat, kernel=3):
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)
        keep = (hmax == heat).float()
        return heat * keep

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def ctdet_decode(self, pr_decs):
        """
        这里的代码说明自己应该有可能做到这些事：
        1. 根据输出的hm图计算出有效的检测结果
        2. 获取的其它支路输出的在检测到的中心点处的值
        3. 对于2, 例如说可以根据中心点和wh输出的vector计算出bounding box
        """
        heat = pr_decs['hm']
        wh = pr_decs['wh']
        reg = pr_decs['reg']
        cls_theta = pr_decs['cls_theta']
        s = pr_decs['sign']
        
        k = pr_decs['k']

        batch, c, height, width = heat.size()
        heat = self._nms(heat)

        # 超级神秘的代码. decoder能够解答很多问题
        scores, inds, clses, ys, xs = self._topk(heat)
        reg = self._tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, self.K, 2)
        xs = xs.view(batch, self.K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, self.K, 1) + reg[:, :, 1:2]
        clses = clses.view(batch, self.K, 1).float()
        scores = scores.view(batch, self.K, 1)
        # print((clses > 0).sum(), (clses == 0).sum())
        # 这里就获取到了([1, 100, 5])的wh向量
        wh = self._tranpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, self.K, 4)  # 4
        s = self._tranpose_and_gather_feat(s, inds)
        s = s.view(batch, self.K, 1)
        k = self._tranpose_and_gather_feat(k, inds)
        k = k.view(batch, self.K, 1)
        # s = wh[:, :, -1:]
        # wh = wh[:, :, :-1]
        # torch.Size([1, 100, 1]) torch.Size([1, 100, 5])
        # print(s.size(), wh.size())

        """
        现在的输入：
        wh:[Batch, K, 5] 
        其中[:,:,0] 是xtx [:,:,1] 是xty
          [:,:,2] 是k [:,:,3] 是w [:,:,4] 是h
        s: [Batch, K, 1] 用来判断xt yt 是否异号的
        xs: [1, 100, 1] 即为xc
        ys: [1, 100, 1] 即为yc
        """

        # add
        # cls_theta应该就是判断是OBB还是HBB的概率
        # 如果是OBB的话, t r b l就用wh支路前8个通道的值计算出结果即
        # 如果是hbb的话, t r b l就用wh支路后2个通道(wh)的值计算出结果
        '''
        cls_theta = self._tranpose_and_gather_feat(cls_theta, inds)
        cls_theta = cls_theta.view(batch, self.K, 1)
        mask = (cls_theta > 0.8).float().view(batch, self.K, 1)
        #
        tt_x = (xs + wh[..., 0:1]) * mask + (xs) * (1. - mask)
        tt_y = (ys + wh[..., 1:2]) * mask + (ys - wh[..., 9:10] / 2) * (1. - mask)
        rr_x = (xs + wh[..., 2:3]) * mask + (xs + wh[..., 8:9] / 2) * (1. - mask)
        rr_y = (ys + wh[..., 3:4]) * mask + (ys) * (1. - mask)
        bb_x = (xs + wh[..., 4:5]) * mask + (xs) * (1. - mask)
        bb_y = (ys + wh[..., 5:6]) * mask + (ys + wh[..., 9:10] / 2) * (1. - mask)
        ll_x = (xs + wh[..., 6:7]) * mask + (xs - wh[..., 8:9] / 2) * (1. - mask)
        ll_y = (ys + wh[..., 7:8]) * mask + (ys) * (1. - mask)
        '''
        """
        现在的输入：
        wh:[Batch, K, 5] 
        其中[:,:,0] 是xtx [:,:,1] 是xty
          [:,:,2] 是k [:,:,3] 是w [:,:,4] 是h
        s: [Batch, K, 1] 用来判断xt yt 是否异号的
        xs: [1, 100, 1] 即为xc
        ys: [1, 100, 1] 即为yc
        需要的输出:
        ttx
        tty
        bbx
        bby
        rrx
        rry
        llx
        lly
        """
        xt = wh[:, :, 0:1]
        yt = wh[:, :, 1:2]
        w = wh[:, :, 2:3]
        h = wh[:, :, 3:4]
        # print(s.size())
        s = (s > 0.5).float().view(batch, self.K, 1) - (s <= 0.5).float().view(batch, self.K, 1)
        bb_x = xt
        bb_y = yt * s
        tt_x = -1 * xt
        tt_y = -1 * yt * s
        theta = torch.arctan(yt / xt)
        short_len = torch.norm(torch.cat([bb_x, bb_y], dim=2), dim=2).unsqueeze(-1) * k
        rr_x = short_len * torch.cos(math.pi / 2 - theta)
        rr_y = short_len * torch.sin(math.pi / 2 - theta) * (-1) * s
        ll_x = -1 * short_len * torch.cos(math.pi / 2 - theta)
        ll_y = -1 * short_len * torch.sin(math.pi / 2 - theta) * (-1) * s
        #
        # a = [xs,ys,tt_x,tt_y,rr_x,rr_y,bb_x,bb_y,ll_x,ll_y,scores,clses]
        # b = ['xs','ys','tt_x','tt_y','rr_x','rr_y','bb_x','bb_y','ll_x','ll_y','scores','clses']
        # for i,j in zip(a,b):
        #   print(j, 'size:', i.size())

        # cls_theta = self._tranpose_and_gather_feat(cls_theta, inds)
        # cls_theta = cls_theta.view(batch, self.K, 1)
        # mask = (cls_theta > 0.8).float().view(batch, self.K, 1)
        # xt = wh[:, :, 0:1]
        # yt = wh[:, :, 1:2]
        # k = wh[:, :, 2:3]
        # w = wh[:, :, 3:4]
        # h = wh[:, :, 4:5]
        # s = (s > 0.5).float().view(batch, self.K, 1) - (s <= 0.5).float().view(batch, self.K, 1)
        # bb_x = (xt) * mask + (0) * (1. - mask)
        # bb_y = (yt * s) * mask + (h / 2) * (1. - mask)
        # tt_x = (-1 * xt) * mask + (0) * (1. - mask)
        # tt_y = (-1 * yt * s) * mask + (- h / 2) * (1. - mask)
        # theta = torch.arctan(yt / xt)
        # short_len = torch.norm(torch.cat([bb_x, bb_y], dim=2), dim=2).unsqueeze(-1) * k
        # rr_x = (short_len * torch.cos(math.pi / 2 - theta)) * mask + (w / 2) * (1. - mask)
        # rr_y = (short_len * torch.sin(math.pi / 2 - theta) * (-1) * s) * mask + (0) * (1. - mask)
        # ll_x = (-1 * short_len * torch.cos(math.pi / 2 - theta)) * mask + (- w / 2) * (1. - mask)
        # ll_y = (-1 * short_len * torch.sin(math.pi / 2 - theta) * (-1) * s) * mask + (0) * (1. - mask)
        # #
        detections = torch.cat([xs,  # cen_x
                                ys,  # cen_y
                                tt_x,
                                tt_y,
                                rr_x,
                                rr_y,
                                bb_x,
                                bb_y,
                                ll_x,
                                ll_y,
                                scores,
                                clses],
                               dim=2)
        # print(detections.size())
        index = (scores > self.conf_thresh).squeeze(0).squeeze(1)
        detections = detections[:, index, :]
        # print(detections.size())
        return detections.data.cpu().numpy()