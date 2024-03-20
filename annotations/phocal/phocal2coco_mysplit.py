# ------------------------------------------------------------------------
# modified by fan xiaofeng to generate coco annotations on phocal
# ------------------------------------------------------------------------
# PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation
# Copyright (c) 2022 Thomas Jantos (thomas.jantos@aau.at), University of Klagenfurt - Control of Networked Systems (CNS). All Rights Reserved.
# Licensed under the BSD-2-Clause-License with no commercial use [see LICENSE for details]
# ------------------------------------------------------------------------
# this script is used to generalize the phocal offical train and test.json
import json
import os
import cv2
import numpy as np
from utils import data_utils,vis_utils


annotation_set = "train" # Choices: train, test

base_path = '/root/commonfile/fxf/phocal/PhoCAL_release/'
output_base_path = './'
annotation_path = annotation_set+'.json'

train_idx=[0,1,2,3,4,5,6,7,8,9,10,11]
test_idx=[12,13,14,15,16,17,18,19,20,21,22,23]
# train_idx=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
# test_idx=[22,23]

categories = [ #只用超类作为监督信息
    {'supercategory': 'bottle', 'id': 0, 'name': 'bottle'},
    {'supercategory': 'box', 'id': 1, 'name': 'box'},
    {'supercategory': 'can', 'id': 2, 'name': 'can'},
    {'supercategory': 'cup', 'id': 3, 'name': 'cup'},
    {'supercategory': 'remote', 'id': 4, 'name': 'remote'},
    {'supercategory': 'teapot', 'id': 5, 'name': 'teapot'},
    {'supercategory': 'cutlery', 'id': 6, 'name': 'cutlery'},
    {'supercategory': 'glass', 'id': 7, 'name': 'glass'},
]

annotations = {'images': [],
               'annotations': [],
               'categories': categories}
image_id = 0
annotation_id = 0


sequence_dirs = [d.name for d in os.scandir(base_path) if d.is_dir() and d.name.startswith("sequence")]
sequence_dirs.sort(key=lambda arr: (arr[:9], int(arr[9:]))) #解决10排在2前面的问题
#print(sequence_dirs)

with open(base_path + 'class_obj_taxonomy.json', 'r') as f:
    class_annotations = json.load(f)

print("Annotating: {}".format(annotation_set)) 
for idx,seq_dir in  enumerate(sequence_dirs):
    print(idx)
    if annotation_set=='train':
        if idx not in train_idx:
            print("skip: ",seq_dir)
            continue
    elif annotation_set=='test':
        if idx not in test_idx:
            print("skip: ",seq_dir)
            continue
    # if idx>2:
    #     break
    print("Annotating: {}".format(seq_dir))
    seq_dir=base_path + '/' + seq_dir +'/'
    with open(seq_dir + 'rgb_scene_gt.json', 'r') as f: #get json annotations
        pose_annotations = json.load(f)
    with open(seq_dir + 'scene_camera.json', 'r') as f:
        camera_annotations = json.load(f)
    img_names = [img for img in os.listdir(seq_dir + 'rgb/') if img[img.rfind('.'):] in ['.png', '.jpg']]
    img_names.sort()
    #print(img_names,len(img_names))
    #print(pose_annotations)
    mask_names = [mask for mask in os.listdir(seq_dir + 'mask/') if mask[mask.rfind('.'):] in ['.png', '.jpg']]
    mask_names.sort()
    # print(mask_names,len(mask_names))
    # Check if annotation length is the same
    n_imgs = len(img_names)
    if len(pose_annotations) != n_imgs or len(mask_names) != n_imgs:
        raise ValueError
    
    # splits=np.load(seq_dir + 'train_test_split.npz')
    # if annotation_set=="train":
    #     sp=splits["train_idxs"]
    # elif annotation_set=="test":
    #     sp=splits["test_idxs"]
    # else :
    #     raise TypeError
    
    K,_=data_utils.get_K(camera_annotations) #内参矩阵


    for img_name , mask_name , pose_annos in zip (img_names , mask_names , pose_annotations):
        # if int(img_name[:img_name.rfind('.png')]) not in sp:
        #     continue
        img_annotation_counter = 0
        file_name = seq_dir + 'rgb/' +img_name
        # print(file_name)
        # print(camera_annotations['rgb'])
        
        mask_file_name=seq_dir+'mask/'+mask_name
        #print(mask_file_name)
        mask_img=cv2.imread(mask_file_name,cv2.IMREAD_GRAYSCALE)
        visib_bboxs,_=data_utils.gen_visib_bbox(mask_img)
        if len(visib_bboxs)!=len(pose_annotations[pose_annos]): #可见的bbox的数量应当等于pose的数量
            print(len(visib_bboxs)+'not equals to'+ len(pose_annotations[pose_annos]))
            raise ValueError


        for pose , visib_bbox in zip(pose_annotations[pose_annos],visib_bboxs):
            #print(pose['cam_t_m2c'])
            pose_T=pose['cam_t_m2c']
            pose_T=np.array(pose_T).reshape(-1,1)
            pose_R=pose['cam_R_m2c']
            pose_R=np.array(pose_R).reshape(3,3)
            pose_RT=np.concatenate([pose_R,pose_T],axis=-1)
            #print(pose_RT) #获取RT矩阵

            class_id=pose['class_id'] #记录pose对应的类别和实例id
            inst_id=pose['inst_id']

            bbox_size=class_annotations[str(class_id)]["scales"][str(inst_id)] #获取bbox的大小从而获取角点位置
            bbox_conner=data_utils.get_bbox_conner(bbox_size)

            pts_2d=data_utils.reproj_3d_2d( #获取3D bbox在二维平面的投影
                K,
                pose_RT,
                bbox_conner
            )
            max_bbox=data_utils.get_max_bbox(pts_2d) #获取3d bbox 外接的bbox

            #print(max_bbox,visib_bbox)
            iou=data_utils.cal_iou(max_bbox,visib_bbox)
            #print(iou)
            bbox={}
            if iou>=0.5: #通过计算iou来决定选择哪个bbox
                bbox['bbox_obj']=visib_bbox
                bbox['bbox_visib']=visib_bbox
                iscrowd=0
            else :
                #print("object is crowded")
                iscrowd=1
                bbox['bbox_obj']=max_bbox
                bbox['bbox_visib']=visib_bbox


            obj_annotation = {
                        'id': annotation_id,
                        'image_id': image_id,
                        'relative_pose': {
                            'position': pose['cam_t_m2c'],
                            'rotation': pose['cam_R_m2c']
                        },
                        'bbox': bbox['bbox_obj'],
                        'bbox_info': bbox,
                        'bbox_3d' : pts_2d.tolist(),
                        'bbox_3d_size' : bbox_size,
                        'bbox_3d_conner' : bbox_conner.tolist(),
                        'area': bbox['bbox_obj'][2] * bbox['bbox_obj'][3],
                        'iscrowd': iscrowd,
                        'category_id': pose['class_id']
                    }
            img_annotation_counter += 1
            annotations['annotations'].append(obj_annotation)
            annotation_id += 1

        if img_annotation_counter == 0:
                print("Image skipped! No annotations valid!")
                continue
        img_annotation = { #获得img_annotation
                'file_name': file_name,
                'id': image_id,
                'width': 640,
                'height': 480,
                'intrinsics': camera_annotations['rgb'],
                'type' : annotation_set
            }
        annotations['images'].append(img_annotation)
        image_id += 1

#print(annotations)
with open(output_base_path + annotation_path, 'w') as out_file:
    json.dump(annotations, out_file , indent=4)
        
        
        
