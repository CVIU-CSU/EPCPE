# ------------------------------------------------------------------------
# modified by fan xiaofeng to generate coco annotations on sunrgbd
# ------------------------------------------------------------------------
# PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation
# Copyright (c) 2022 Thomas Jantos (thomas.jantos@aau.at), University of Klagenfurt - Control of Networked Systems (CNS). All Rights Reserved.
# Licensed under the BSD-2-Clause-License with no commercial use [see LICENSE for details]
# ------------------------------------------------------------------------
# this script is used to generalize the phocal offical train and test.json
import json
import os
import os.path as osp
import cv2
import numpy as np
import _pickle as pkl
from tqdm import tqdm
import glob
from random import sample
from utils.sunrgbd_utils import *


# annotation_set = "test_all" # Choices: train, test
# data_set = "real" #Choices: camera, real

base_path = '/root/commonfile/fxf/sunrgbd/' #2080
#base_path = '/home1/fanxiaofeng/sunrgbd/' #3090

annotation_set = "training"
data_path=osp.join(base_path,'sunrgbd_trainval')
output_base_path = './'

if annotation_set == "testing":
    data_dirs=osp.join(data_path,'val_data_idx.txt')
    annotation_path='test_cppf.json'
elif annotation_set == "training":
    data_dirs=osp.join(data_path,'train_data_idx.txt')
    annotation_path='train_cppf.json'
else:
    raise KeyError



 # 'bottle', 'bowl', 'camera', 'can', 'laptop', 'mug'
categories = [ #只用超类作为监督信息
    {'supercategory': 'bed', 'id': 0, 'name': 'bed'},
    {'supercategory': 'table', 'id': 1, 'name': 'table'},
    {'supercategory': 'sofa', 'id': 2, 'name': 'sofa'},
    {'supercategory': 'chair', 'id': 3, 'name': 'chair'},
    {'supercategory': 'bookshelf', 'id': 4, 'name': 'bookshelf'},
    {'supercategory': 'bathtub', 'id': 5, 'name': 'bathtub'},
]

cat_id_list=['bed','table','sofa','chair','bookshelf','bathtub'] # 用于查找新的cat_id

annotations = {'images': [],
               'annotations': [],
               'categories': categories}
image_id = 0
annotation_id = 0


#print(len(data_dirs))

print("Annotating: {}".format(annotation_set)) 


dataset = sunrgbd_object(data_path,annotation_set,True)
data_idx_list = [int(line.rstrip()) for line in open(data_dirs)]

for data_idx in tqdm(data_idx_list):
    img_path = dataset.get_image_path(data_idx)
    objects = dataset.get_label_objects(data_idx)
    calib = dataset.get_calibration(data_idx)
    K=calib.K
    Rtilt= calib.Rtilt
    img_annotation_counter = 0

    for obj in objects:
        try:
            cat_id=cat_id_list.index(obj.classname) #对于不在list里的标注 跳过
        except:
            print(obj.classname)
            continue
        pos=obj.centroid
        bbox=[obj.xmin,obj.ymin,obj.xmax-obj.xmin,obj.ymax-obj.ymin]
        angle=-1*obj.heading_angle
        rot=rotz(angle)
        pose_init=np.concatenate([rot.reshape(3,3),pos.reshape(3,1)],axis=1)
        scale=np.array([obj.l*2,obj.w*2,obj.h*2],dtype=np.float32)

        permute_mat=np.array([[1,0,0],
                            [0,0,-1],
                            [0,1,0]]) #from depth to image
        pose_final=permute_mat@Rtilt.T@pose_init
        rot=pose_final[:3,:3]
        trans=pose_final[:3,3]
        


        obj_annotation = {
                        'id': annotation_id,
                        'image_id': image_id,
                        'relative_pose': {
                            'position': trans.reshape(3).tolist(),
                            'rotation': rot.reshape(9).tolist()
                        },
                        'norm_pose' : {
                            'rotation_norm': rot.reshape(9).tolist(),
                            'scale_norm' : float(1)
                        },
                        'bbox': bbox,
                        'bbox_3d_size' : scale.tolist(),
                        'area': int(bbox[2] * bbox[3]),
                        'iscrowd': 0,
                        'handle_visibility': 1,
                        'category_id': cat_id
                    }

        img_annotation_counter += 1
        annotations['annotations'].append(obj_annotation)
        annotation_id += 1

    if img_annotation_counter == 0:
            print("Image skipped! No annotations valid!")
            continue
    img_annotation = { #获得img_annotation
            'file_name': img_path,
            'id': image_id,
            'width': 730,
            'height': 530,
            'intrinsics': K.reshape(9).tolist(),
            'type' : annotation_set
        }
    annotations['images'].append(img_annotation)
    image_id += 1

#print(annotations)
with open(output_base_path + annotation_path, 'w') as out_file:
    json.dump(annotations, out_file , indent=4)