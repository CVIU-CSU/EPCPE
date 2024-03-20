from utils import vis_utils
from utils import data_utils
from utils import bop_utils
import json
import cv2
import numpy as np


dataset_path='/root/commonfile/fxf/phocal/PhoCAL_release/'

# get file
anno_filename='/root/userfolder/github/poet/workdirs/default_dir/bop_gt/bop_eval_on_ckpts25.csv'
bop_results=bop_utils.load_bop_results(anno_filename)
#print(bop_results)

with open(dataset_path + 'class_obj_taxonomy.json', 'r') as f:
    class_annotations = json.load(f)

# parse R and T
# for result in bop_results: #一个目标物体一张图
#     scene_id=result['scene_id']
#     img_id=result['im_id']
#     obj_id=result['obj_id']
#     pose_R=result['R']
#     pose_T=np.divide(result['t'],1000) #还原平移向量
#     print(pose_R,pose_T)

#     # visual result
#     rgb_file_path=dataset_path+'sequence_'+str(scene_id)+'/rgb/'+str('%06d'%img_id)+'.png' #获取到图像信息
#     print(rgb_file_path)
#     #rgb_img=cv2.imread(rgb_file_path)
#     seq_dir=dataset_path+'sequence_'+str(scene_id)+'/'
#     with open(seq_dir + 'scene_camera.json', 'r') as f:
#         camera_annotations = json.load(f)

#     pose_RT=np.concatenate([pose_R,pose_T],axis=-1)
#     print(pose_RT)
#     print(pose_RT.shape)
#     K,_=data_utils.get_K(camera_annotations)
#     bbox_size=class_annotations[str(obj_id)]["scales"]['0'] #获取bbox的大小从而获取角点位置
#     bbox_conner=data_utils.get_bbox_conner(bbox_size)

#     vis_utils.save_demo_image(
#         pose_RT,K,rgb_file_path,bbox_conner,save_path='/root/userfolder/github/poet/workdirs/default_dir/bop_gt/vis_img/'+str(scene_id)+str('%06d'%img_id)+'/'+str(obj_id)+'.png'
#     )

scene_id_last=1 #用于记录场景和img_id是否发生变化 先初始化为第一个场景第一幅图
img_id_last=16
pose_RTs=[]
bbox_conners=[]

for result in bop_results: #一个场景一张图
    scene_id=result['scene_id']
    img_id=result['im_id']
    obj_id=result['obj_id']
    pose_R=result['R']
    pose_T=np.divide(result['t'],1000) #还原平移向量

    print(scene_id,img_id)
    if scene_id!=scene_id_last or img_id!=img_id_last:
        vis_utils.save_demo_image(
            pose_RTs,K,rgb_file_path,bbox_conners,save_path='/root/userfolder/github/poet/workdirs/default_dir/bop_gt/vis_img/'+str(scene_id_last)+str('%06d'%img_id_last)+'.png'
        )
        pose_RTs=[]
        bbox_conners=[]

    print(pose_R,pose_T)

    # visual result
    rgb_file_path=dataset_path+'sequence_'+str(scene_id)+'/rgb/'+str('%06d'%img_id)+'.png' #获取到图像信息
    print(rgb_file_path)
    #rgb_img=cv2.imread(rgb_file_path)
    seq_dir=dataset_path+'sequence_'+str(scene_id)+'/'
    with open(seq_dir + 'scene_camera.json', 'r') as f:
        camera_annotations = json.load(f)

    pose_RT=np.concatenate([pose_R,pose_T],axis=-1)
    print(pose_RT)
    print(pose_RT.shape)
    K,_=data_utils.get_K(camera_annotations)
    bbox_size=class_annotations[str(obj_id)]["scales"]['0'] #获取bbox的大小从而获取角点位置
    bbox_conner=data_utils.get_bbox_conner(bbox_size)

    pose_RTs.append(pose_RT)
    bbox_conners.append(bbox_conner)

    scene_id_last=scene_id
    img_id_last=img_id

        

    
