__author__= 'fanxiaofeng'
import torch
import numpy as np
from mmengine.logging import print_log
from terminaltables import AsciiTable
from collections.abc import Sequence


def eval_pose(gt_rots,pred_rots,
            gt_poses,pred_poses,
            gt_labels,pred_labels=None,logger=None):

    all_rot_error,class_rot_error,recalls=eval_rotation_error(gt_rots,pred_rots,gt_labels,logger=logger)
    #calc_all_rot_error(all_rot_error)

    eval_translation_error(gt_poses,pred_poses,gt_labels,logger=logger)
    return all_rot_error,class_rot_error

def calc_all_rot_error(all_rot_error ,thrs=[20]): #计算一下rot和trans的小于thrs的误差
    img_num = all_rot_error.shape[0]
    print(all_rot_error.shape)
    prop_nums,error_ths=set_recall_param()
    for i in range(img_num):
        print(all_rot_error[i])
        print(all_rot_error[i]<=thrs)


def _eval_rotation_recall(all_rot_error, proposal_nums, thrs): #仿照recal._recall() 完成 后续要加入translation 一起计算
    img_num = all_rot_error.shape[0]
    total_gt_num = sum([rot_error.shape[0] for rot_error in all_rot_error])

    _rot_error = np.full((proposal_nums.size, total_gt_num),dtype=np.float32,fill_value=np.inf) #
    #_rot_error = np.zeros((proposal_nums.size,total_gt_num),dtype=np.float32)
    for k,proposal_num in enumerate(proposal_nums): #先从遍历proposal_num开始
        tmp_rot_error = np.zeros(0)
        for i in range (img_num): #遍历图片数量 以3个sequence为例则为117
            rot_error = all_rot_error[i][:, :proposal_num].copy() #获取第i个图像的前proposal_num个预测结果和gt的误差矩阵
            gt_rot_error = np.full((rot_error.shape[0]),np.inf) #长度为第i个图中的gt框的数量
            #gt_rot_error =np.zeros((rot_error.shape[0]))
            if rot_error.size == 0:
                tmp_rot_error = np.hstack((tmp_rot_error,gt_rot_error)) #没有误差矩阵则加入一行全为np.inf值的结果
                continue
            for j in range (rot_error.shape[0]):
                gt_min_rot_error = rot_error.argmin(axis=1)
                #print("gt_min",gt_min_rot_error)
                #print("rot_error",rot_error)
                min_rot_error = rot_error[np.arange(0,rot_error.shape[0]),gt_min_rot_error]
                #print("min_rot_error",min_rot_error)
                gt_idx = min_rot_error.argmin()
                gt_rot_error[j]= min_rot_error[gt_idx]
                box_idx = gt_min_rot_error[gt_idx]
                rot_error[gt_idx,:] = np.inf
                rot_error[:,box_idx] = np.inf
                #print("gt_rot",gt_rot_error)
            tmp_rot_error = np.hstack((tmp_rot_error,gt_rot_error))
        _rot_error[k,:] = tmp_rot_error

    _rot_error = np.fliplr(np.sort(_rot_error,axis=1))
    # print(_rot_error)
    # print(_rot_error.shape)
    recalls = np.zeros((proposal_nums.size,thrs.size))
    print(recalls.shape)
    for i,thr in enumerate(thrs):
        #print(_rot_error<=thr)
        recalls[:,i] = (_rot_error<=thr).sum(axis=1) /float(total_gt_num) #origin
        #recalls[j,:] = (_rot_error<=thr).sum(axis=0) /float(proposal_nums)
        # print((_rot_error<=thr).sum(axis=0))
        # sum_list=np.array((_rot_error<=thr).sum(axis=1))
        # proposal_nums=np.array(proposal_nums)
        # recalls[j,:]=sum_list/proposal_nums #有问题
        
        # for j,pnum in enumerate(proposal_nums):
        #     recalls[j,i] = (_rot_error[j]<=thr).sum() /float(pnum)
    
    return recalls


def set_recall_param(proposal_nums=None, error_thrs=None):
    """Check proposal_nums and iou_thrs and set correct format."""
    if proposal_nums is None:
        _proposal_nums = np.array([1,5,10,20,50,300])
    elif isinstance(proposal_nums, Sequence):
        _proposal_nums = np.array(proposal_nums)
    elif isinstance(proposal_nums, int):
        _proposal_nums = np.array([proposal_nums])
    else:
        _proposal_nums = proposal_nums

    if error_thrs is None:
        _error_thrs = np.array([100,20,10,5,3,2,1,0.5])
    elif isinstance(error_thrs, Sequence):
        _error_thrs = np.array(error_thrs)
    elif isinstance(error_thrs, float):
        _error_thrs = np.array([error_thrs])
    else:
        _error_thrs = error_thrs

    return _proposal_nums, _error_thrs


def print_recall_summary(recalls,
                         proposal_nums,
                         iou_thrs,
                         row_idxs=None,
                         col_idxs=None,
                         logger=None):
    """Print recalls in a table.

    Args:
        recalls (ndarray): calculated from `bbox_recalls`
        proposal_nums (ndarray or list): top N proposals
        iou_thrs (ndarray or list): iou thresholds
        row_idxs (ndarray): which rows(proposal nums) to print
        col_idxs (ndarray): which cols(iou thresholds) to print
        logger (logging.Logger | str | None): The way to print the recall
            summary. See `mmengine.logging.print_log()` for details.
            Default: None.
    """
    proposal_nums = np.array(proposal_nums, dtype=np.int32)
    iou_thrs = np.array(iou_thrs)
    if row_idxs is None:
        row_idxs = np.arange(proposal_nums.size)
    if col_idxs is None:
        col_idxs = np.arange(iou_thrs.size)
    row_header = [''] + iou_thrs[col_idxs].tolist()
    table_data = [row_header]
    for i, num in enumerate(proposal_nums[row_idxs]):
        row = [f'{val:.3f}' for val in recalls[row_idxs[i], col_idxs].tolist()]
        row.insert(0, num)
        table_data.append(row)
    table = AsciiTable(table_data)
    print_log('\n' + table.table, logger=logger)


def eval_rotation_error(gts,
                        proposals,
                        gt_labels,
                        proposal_nums=[1,5,8,10,20],
                        error_thrs=[100,20,10,5,3,2,1,0.5],
                        logger=None):
    """ Calculate rotation error by class

    """
    img_num = len(gts)
    logger.info(f"TEST gts len {len(gts)} proposals len {len(proposals)} gt_labels len {len(gt_labels)}")
    assert img_num == len(proposals)
    all_rot_error = []
    proposal_nums, error_thrs = set_recall_param(proposal_nums, error_thrs)
    class_num = np.max(gt_labels)
    logger.info(f"TEST class_num {class_num}")
    class_rot_error = []
    # logger.info(f"TEST class_rot_error {class_rot_error}")
    for i in range(img_num):
        img_proposals=proposals[i]
        if gts[i] is None or gts[i].shape[0]==0:
            rot_error = np.zeros((0,img_proposals.shape[0]),dtype=np.float32)
        else:
            #logger.info(f"TEST gt len {len(gts[i])} label len {len(gt_labels[i])}")
            rot_error = calc_rotation_error(gts[i],img_proposals,gt_labels[i],class_rot_error,logger)
        #logger.info(f"TEST rot_error {rot_error}")
        all_rot_error.append(rot_error)
        #print(all_rot_error)
    all_rot_error = np.array(all_rot_error) 
    # 这里可以把后续两步和前面拆开
    # recalls = _eval_rotation_recall(all_rot_error,proposal_nums,error_thrs)
    recalls=0
    # print_recall_summary(recalls, proposal_nums, error_thrs, logger=logger)

    return all_rot_error,class_rot_error,recalls

def eval_translation_error(gts,
                        proposals,
                        gt_labels,
                        logger=None):
    """ Calculate translation error by class

    """


    pass

    

def calc_rotation_error(rot_gt, rot_pred,label_gt,class_rot_error,logger=None):
    """
    Calculaten the geodesic distance between two rotations
    """
    #logger.info(f"TEST rot_gt {rot_gt},rot_pred {rot_pred}")
    #logger.info(f"TEST rot_gt len{len(rot_gt)},rot_pred len{len(rot_pred)}")
    rot_pred=rot_pred.reshape(-1,3,3)
    rot_gt=rot_gt.reshape(-1,3,3)
    rows = rot_gt.shape[0]
    cols = rot_pred.shape[0]
    rotation_error = np.full((rows,cols),np.inf) #初始化一个最大值矩阵
    for i in range(rows):
        for j in range (cols):
            # s_pred=1
            # s_gt=1
            s_pred=np.cbrt(np.linalg.det(rot_pred[i,:3, :3]))
            s_gt=np.cbrt(np.linalg.det(rot_gt[i,:3, :3]))
            rot_pred[i]=rot_pred[i]/s_pred
            rot_gt[i]=rot_gt[i]/s_gt
            #print("s_pred,s_gt",s_pred,s_gt)
            rot = np.matmul(rot_pred[j], rot_gt[i].T)
            trace = np.trace(rot)
            # if trace < -1.0: #加了可能会把设为nan的结果变成0
            #     trace = -1
            # elif trace > 3.0:
            #     trace = 3.0 #trace = 3 degree=0 trace=-1 degree=180
            #print("trace",trace)
            cos_theta = (trace - 1) / 2
            angle_diff = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
            #angle_diff = np.degrees(np.arccos(0.5 * (trace - 1)))
            if angle_diff==np.nan: 
                angle_diff=np.inf
            rotation_error[i][j]=angle_diff
    #logger.info(f"TEST rotation error shape {rotation_error.shape}")
    return rotation_error


def calc_translation_error(t_gt, t_pred):
    """
    Calculate the L1 error between two translations.
    This function only processes a single quaternion pair // not suited for batches
    Differentiate between tensors and regular arrays
    """

    error = torch.sqrt(torch.square((t_gt - t_pred)).sum(dim=0))
    return error
