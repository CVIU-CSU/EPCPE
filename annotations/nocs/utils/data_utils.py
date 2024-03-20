import numpy as np
import cv2

def gen_visib_bbox(inst_map): #visb_box 输入mask图像 输出对应的visb_box 输出可视的bbox和物体id
    """
    generalize visible 2d bbox
    @param inst_map masked image
    """
    boxes = []
    ids = []
    for id in np.unique(inst_map)[1:]:
        inst = inst_map == int(id)
        lr = inst.max(axis=0)
        td = inst.max(axis=1)
        lr_nz = np.nonzero(lr)
        td_nz = np.nonzero(td)
        if lr_nz[0].sum() == 0 and td_nz[0].sum() == 0:
            return []
        l, r, t, d = float(lr_nz[0][0]), float(lr_nz[0][-1]), float(td_nz[0][0]), float(td_nz[0][-1])
        boxes.append([l, t, r-l, d-t])
        ids.append(id)
    return boxes,ids #[n,4] [n,1]

def reproj_3d_2d(K, pose, pts_3d):
    """ 
    Reproj 3d points to 2d points 
    @param K: [3, 3] or [3, 4]
    @param pose: [3, 4] or [4, 4]
    @param pts_3d: [n, 3] n=8
    """
    assert K.shape == (3, 3) or K.shape == (3, 4)
    assert pose.shape == (3, 4) or pose.shape == (4, 4)

    if K.shape == (3, 3):
        K_homo = np.concatenate([K, np.zeros((3, 1))], axis=1)
    else:
        K_homo = K
    
    if pose.shape == (3, 4):
        pose_homo = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
    else:
        pose_homo = pose
    
    pts_3d = pts_3d.reshape(-1, 3)
    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1))], axis=1)
    pts_3d_homo = pts_3d_homo.T

    reproj_points = K_homo @ pose_homo @ pts_3d_homo
    reproj_points = reproj_points[:] / reproj_points[2:]
    reproj_points = reproj_points[:2, :].T
    return reproj_points # [n, 2]

def get_K(camera_annotations): #获取相机内参K
    rgb_k=camera_annotations['rgb']
    fx=rgb_k['fx']
    fy=rgb_k['fy']
    cx=rgb_k['cx']
    cy=rgb_k['cy']
    #print(fx,fy,cx,cy)

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])
    K_homo = np.array([
        [fx, 0, cx, 0],
        [0, fy, cy, 0],
        [0,  0,  1, 0]
    ])
    return K, K_homo

def get_bbox_conner(bbox_size): #获取3d bbox的角点
    x=bbox_size[0]
    y=bbox_size[1]
    z=bbox_size[2]

    bbox_conner=np.array([
        [-x/2,-y/2,-z/2],
        [-x/2,-y/2, z/2],
        [-x/2, y/2, z/2],
        [-x/2, y/2,-z/2],
        [ x/2,-y/2,-z/2],
        [ x/2,-y/2, z/2],
        [ x/2, y/2, z/2],
        [ x/2, y/2,-z/2],
    ])

    return bbox_conner

def get_max_bbox(pts_2d):
    #print(pts_2d)
    pt1=np.min(pts_2d,axis=0)
    pt2=np.max(pts_2d,axis=0)
    # print(pt1)
    # print(pt2)
    #l,t,r,d=pt1[0],pt1[1],pt2[0],pt2[1]
    l,t,w,h=pt1[0],pt1[1],pt2[0]-pt1[0],pt2[1]-pt1[1]
    # print(l,t,w,h)
    return l,t,w,h

def cal_iou(box1,box2):
    l1,t1,w1,h1=box1
    l2,t2,w2,h2=box2
    s1=w1*h1
    s2=w2*h2
    r1=l1+w1
    d1=t1+h1
    r2=l2+w2
    d2=t2+h2

    # 计算相交矩形
    xmin = max(l1, l2)
    ymin = max(t1, t2)
    xmax = min(r1, r2)
    ymax = min(d1, d2)
 
    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    a1 = w * h  # C∩G的面积
    a2 = s1 + s2 - a1
    iou = a1 / a2 #iou = a1/ (s1 + s2 - a1)
    return iou

    
