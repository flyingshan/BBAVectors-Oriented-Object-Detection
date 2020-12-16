# Created by Yishan 2020/12/14
# Oriented Detection with Polar Vectors 

import numpy as np
import math
import torch.nn.functional as F
import torch
from MBB import MinimumBoundingBox, BoundingBox

###########--------Encode部分--------###########

def arrange_order(tt, rr, bb, ll):
    """
    点顺序定义：
    1. 水平框
    这种情况下，t表示y坐标最大的点，b表示y坐标最小的点，l表示x坐标最大的点，r表示x坐标最小的点
    2. 旋转框
    这种情况下，t表示第一象限点，r表示第二象限点，b表示第三象限点，l表示第四象限点
    """
    pts = np.asarray([tt,rr,bb,ll],np.float32)
    # 如果有零，说明向量垂直
    if (pts == 0).sum() > 0:
        l_ind = np.argmax(pts[:,0])
        r_ind = np.argmin(pts[:,0])
        t_ind = np.argmax(pts[:,1])
        b_ind = np.argmin(pts[:,1])
        tt_new = pts[t_ind,:]
        rr_new = pts[r_ind,:]
        bb_new = pts[b_ind,:]
        ll_new = pts[l_ind,:]
        return tt_new, rr_new, bb_new, ll_new       
    else:
        for v in [tt, rr, bb, ll]:
          if v[0] > 0 and v[1] > 0:
            tt_new = v
          if v[0] < 0 and v[1] > 0:
            rr_new = v
          if v[0] < 0 and v[1] < 0:
            bb_new = v
          if v[0] > 0 and v[1] < 0:
            ll_new = v

    return tt_new, rr_new, bb_new, ll_new

def calculate_point_angle(point):
    """
    返回一个直角坐标系下点的角度，范围是(-pi, pi]
    """
    x = point[0];
    y = point[1];
    z = math.sqrt(x**2 + y**2)
    # y == 0 时，x < 0, 角度为pi，因此只能取到正pi
    if (y >= 0):
        return np.arccos(x / z)
    else:
        return -np.arccos(x / z)


def between_range(pre_angle, nxt_angle, target_angle):
    """
    判断target_angle是否在(pre_angle, nxt_angle]范围内，要考虑(-pi, pi]
    """

    '''
    当a->b->c->d这条角度链可能存在几个情况:
    1. pre < nxt 只要pre和nxt连线不过x负半轴，就不会有问题；
       这种情况就直接判断角度范围即可
    2. pre > nxt:
       pre > 0, nxt < 0: 说明pre在12象限，nxt在34象限，判断这个范围即可，注意x负半轴角度是正pi;
    3. pre == nxt: 不可能发生，如果发生，应该报错

    '''

    if pre_angle < nxt_angle:
        if target_angle > pre_angle and target_angle <= nxt_angle:
            return True
        else:
            return False
    elif pre_angle > nxt_angle:
        assert pre_angle > 0 and nxt_angle < 0
        if (target_angle > pre_angle and target_angle <= np.pi) or (target_angle > - np.pi and target_angle <= nxt_angle):
            return True
        else:
            return False
    else:
        raise("OBB with zero width/height!")

def calculate_distance(t_ag, a1, a2, r1, r2):
    """
    给定极坐标下两点(r1, a1), (r2, a2)，计算角度t_ag的射线与这两点确定的直线的交点距极点的距离
    """
    return r1 * r2 * math.sin(a1 - a2) / (r2 * math.sin(t_ag - a2) + r1 * math.sin(a1 - t_ag))


def calculate_intersection_distance(target_angle, neighbor_pts_angle, corner_pts):
    """
    该函数计算target_angle角度对应的边界盒子的交点与原点（中心点）的距离。
    target_angle: 目标角度，弧度制。
    neighbor_pts_angle: ([a_ag, b_ag], [b_ag, c_ag], [c_ag, d_ag], [d_ag, a_ag])
    corner_pts: [a_pt, b_pt, c_pt, d_pt]

    return: target_distance 即为 目标交点与中心点的距离。

    """

    # 对每一个 [pre_a, nxt_a)的范围，如果判断交点属于这一范围，则计算射线与这条直线的交点
    for i, (pre_a, nxt_a) in enumerate(neighbor_pts_angle):
        if between_range(pre_a, nxt_a, target_angle):
            pt_1 = corner_pts[i]
            pt_2 = corner_pts[(i+1) % 4] # 这个不会有问题吧
            pt1_ag = calculate_point_angle(pt_1)
            pt2_ag = calculate_point_angle(pt_2)
            pt1_r  = math.sqrt(pt_1[0]**2 + pt_1[1]**2)
            pt2_r  = math.sqrt(pt_2[0]**2 + pt_2[1]**2)
            dist = calculate_distance(target_angle, pt1_ag, pt2_ag, pt1_r, pt2_r)
            break
    return dist


            
def polar_encode(points_info, n):
    '''
    该函数只处理一张图中一个目标的标注信息，将之从4点标注转为极坐标距离标注
    points_info: 标注角点标注信息的一个字典。
                 其中'pts_4'键对应角点位置信息；'ct'对应中心点位置信息；
                 可以在具体实现中观察其使用方法。
    n: 将0~180度分为几个部分，生成几个标注点，如每隔45度一点的话，n=4。
    return: polar_pts_n表示极坐标系下每隔180/n角度，矩形框边界点与原点（中心点）产生的距离，是一个n长list。
    '''

    pts_4 = points_info['pts_4']
    ct    = points_info['ct']
    theta = points_info['theta']

    bl = pts_4[0,:]
    tl = pts_4[1,:]
    tr = pts_4[2,:]
    br = pts_4[3,:]

    tt = (np.asarray(tl,np.float32)+np.asarray(tr,np.float32))/2 - ct
    rr = (np.asarray(tr,np.float32)+np.asarray(br,np.float32))/2 - ct
    bb = (np.asarray(bl,np.float32)+np.asarray(br,np.float32))/2 - ct
    ll = (np.asarray(tl,np.float32)+np.asarray(bl,np.float32))/2 - ct

    # 转换为了有序的四个象限坐标
    p1, p2, p3, p4 = arrange_order(tt, rr, bb, ll)
    
    # 得到四个角点
    a_pt, b_pt, c_pt, d_pt = p1 + p2, p2 + p3, p3 + p4, p4 + p1

    # 得到四个角点对应的角度，方便计算交点，注意这里角度范围[0, 2*pi)
    a_ag, b_ag, c_ag, d_ag = calculate_point_angle(a_pt), calculate_point_angle(b_pt), \
                             calculate_point_angle(c_pt), calculate_point_angle(d_pt)

    # 接下来给定一个角度，需要计算这个角度射线与边界上某一条边的交点
    # 获取n个标注点，这些点是对应角度射线与包围盒边界的交点

    # 范围是[0, pi)内取n个点
    delta_angle = np.pi / n
    neighbor_pts_angle = ([a_ag, b_ag], [b_ag, c_ag], [c_ag, d_ag], [d_ag, a_ag])
    corner_pts = [a_pt, b_pt, c_pt, d_pt]

    polar_pts_n = []
    for i in range(n):
        target_angle = delta_angle * i
        target_distance = calculate_intersection_distance(target_angle, neighbor_pts_angle, corner_pts)
        polar_pts_n.append(target_distance)

    return polar_pts_n


###########------Encode部分结束------###########


###########--------Decode部分--------###########

def calculate_boundary_points(corner_pts):
    pt1 = corner_pts.pop()
    pt2 = corner_pts.pop()
    pt3 = corner_pts.pop()
    pt4 = corner_pts.pop()
    
    # 采用角度法计算
    pt1_ag = calculate_point_angle(pt1)
    pt2_ag = calculate_point_angle(pt2)
    pt3_ag = calculate_point_angle(pt3)
    pt4_ag = calculate_point_angle(pt4)
    
    unordered_pts    = np.array([pt1,    pt2,    pt3,    pt4   ])
    unordered_angles = np.array([pt1_ag, pt2_ag, pt3_ag, pt4_ag])
    
    indexes = np.argsort(unordered_angles)
    pt1_new = unordered_pts[indexes[0]]
    pt2_new = unordered_pts[indexes[1]]
    pt3_new = unordered_pts[indexes[2]]
    pt4_new = unordered_pts[indexes[3]]
    
    tt = (np.asarray(pt1_new, np.float32) + np.asarray(pt2_new, np.float32)) / 2
    rr = (np.asarray(pt2_new, np.float32) + np.asarray(pt3_new, np.float32)) / 2
    bb = (np.asarray(pt3_new, np.float32) + np.asarray(pt4_new, np.float32)) / 2
    ll = (np.asarray(pt4_new, np.float32) + np.asarray(pt1_new, np.float32)) / 2
    
    return tt, rr, bb, ll
    
    

def polar_decode(wh, scores, clses, xs, ys, thres, n):
    """
    wh, scores, cls，xs, ys即为ctdet_decode中已经转换好的检测结果相关的信息
        其中wh为预测坐标信息，scores用来阈值判断，cls为类别信息，xsys为中心点信息，thres为阈值
    return: detections即为与ctdet_decode中的返回结果一样的东西，在ctdet_decode中可以这样调用该函数：

        return polar_decode(wh, scores, cls, xs, ys, thres)
    sizes:
    wh: (batch, self.K, 8)
    clses: (batch, self.K, 1)
    scores: (batch, self.K, 1)
    xs: (batch, self.K, 1)
    ys: (batch, self.K, 1)
    thres: (1,)
    """

    '''
    接下来：
    1. 找到筛选过后的wh
    2. 将距离信息转换为直角坐标信息，且补全整个矩形目标框的信息
    3. 根据补全后的信息，计算最小矩形包围盒
    4. 计算四个边中心点，生成tt_x - ll_y
    5. 把wh外的其它信息cat，筛，再与生成的几个向量cat起来。
    '''
    # 1. 找到筛选后的wh
    index = (scores>thres).squeeze(0).squeeze(1)
    wh = wh[:,index,:]
    # print(wh.size())
    
    # 2. 转为直角坐标并补全
    num_targets = wh.size()[1]
    angles = torch.linspace(0, np.pi, n+1)[:-1].unsqueeze(0).unsqueeze(0).repeat(1,num_targets,1).cuda()
    wh_x = torch.mul(wh, torch.cos(angles))
    wh_y = torch.mul(wh, torch.sin(angles))
    wh_x_symmetry = -1 * wh_x
    wh_y_symmetry = -1 * wh_y
    
    # 3. 计算最小包围盒
    
    target_loc_pts = torch.zeros(1, num_targets, 8).cuda() # ttx tty rrx rry bbx bby llx lly
    
    for i in range(num_targets):
        # 每个目标计算一个MMB(tt, bb, ll, rr来表示)
        target_pts = []
        for j in range(n):
            target_pts.append((wh_x[0, i, j].cpu(),          wh_y[0, i, j].cpu()))
            target_pts.append((wh_x_symmetry[0, i, j].cpu(), wh_y_symmetry[0, i, j].cpu()))
        # 根据边界点计算最小包围盒
        mbb = MinimumBoundingBox(target_pts)
        
        # 4. 计算四个边中心点，生成tt_x - ll_y，赋给检测结果
        
        # 获得角点
        corner_pts = mbb.corner_points
        # 根据无序角点计算各边中点（无序）
        tt, rr, bb, ll = calculate_boundary_points(corner_pts)
        # 把无序变有序
        tt, rr, bb, ll = arrange_order(tt, rr, bb, ll)
        # 将计算得到的各中点赋值给检测结果
        target_loc_pts[0, i, 0] = float(tt[0])
        target_loc_pts[0, i, 1] = float(tt[1])
        target_loc_pts[0, i, 2] = float(rr[0])
        target_loc_pts[0, i, 3] = float(rr[1])
        target_loc_pts[0, i, 4] = float(bb[0])
        target_loc_pts[0, i, 5] = float(bb[1])
        target_loc_pts[0, i, 6] = float(ll[0])
        target_loc_pts[0, i, 7] = float(ll[1])
    
    # print(target_loc_pts)
    # 5. 把wh外的其它信息cat，筛，再与生成的几个向量cat起来。
    detections = torch.cat([xs,                      # cen_x
                            ys,                      # cen_y
                            scores,
                            clses],
                            dim=2)
    detections = detections[:,index,:]
    ####################################
    # TODO: 修改func_utils中的索引序号 #
    ####################################
    detections = torch.cat([detections, target_loc_pts], dim=2)
    
    
    return detections.data.cpu().numpy()