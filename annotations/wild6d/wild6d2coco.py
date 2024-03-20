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
import os.path as osp
import cv2
import numpy as np
import _pickle as pkl
from tqdm import tqdm
import glob
from random import sample


# annotation_set = "test_all" # Choices: train, test
# data_set = "real" #Choices: camera, real

base_path = '/root/commonfile/fxf/wild6d' #2080
#base_path = '/home1/fanxiaofeng/wild6d/' #3090

test_path = os.path.join(base_path,'test_set')
output_base_path = './'
annotation_path = 'test_wild6d.json'


def gen_visb_bbox(inst_map): #visb_box
    boxes = []
    for id in np.unique(inst_map)[1:]:
        inst = inst_map == int(id)
        lr = inst.max(axis=0)
        td = inst.max(axis=1)
        lr_nz = np.nonzero(lr)
        td_nz = np.nonzero(td)
        if lr_nz[0].sum() == 0 and td_nz[0].sum() == 0:
            return []
        l, r, t, d = float(lr_nz[0][0]), float(lr_nz[0][-1]), float(td_nz[0][0]), float(td_nz[0][-1])
        boxes=[l, t, r-l, d-t]
    return boxes


 # 'bottle', 'bowl', 'camera', 'can', 'laptop', 'mug'
categories = [ #只用超类作为监督信息
    {'supercategory': 'bottle', 'id': 0, 'name': 'bottle'},
    {'supercategory': 'bowl', 'id': 1, 'name': 'bowl'},
    {'supercategory': 'camera', 'id': 2, 'name': 'camera'},
    {'supercategory': 'can', 'id': 3, 'name': 'can'},
    {'supercategory': 'laptop', 'id': 4, 'name': 'laptop'},
    {'supercategory': 'mug', 'id': 5, 'name': 'mug'},
]

cat_id_list=['bottle','bowl','camera','can','laptop','mug'] # 用于查找新的cat_id

annotations = {'images': [],
               'annotations': [],
               'categories': categories}
image_id = 0
annotation_id = 0


#print(len(data_dirs))

print("Annotating: {}".format(test_path)) 

ann_paths=sorted(glob.glob(os.path.join(test_path,'pkl_annotations/*/*.pkl')))

for ann_path in tqdm(ann_paths):
    anns = pkl.load(open(ann_path, 'rb'))
    sampled_ann = sample(anns['annotations'],10)
    for ann in sampled_ann:
    #for ann in anns['annotations']:
        img_annotation_counter =0
        try:
            cls_n, seq_idx, obj_idx, frame_idx = ann['name'].split('/')
            if cls_n!="laptop":
                continue
            if cls_n=="cup":
                cls_n="mug"
            base_path = os.path.join(test_path,'{}/{}/{}'.format(cls_n, seq_idx, obj_idx))
            img_path = os.path.join(base_path, 'images/{}.jpg'.format(int(frame_idx)))
            cat_id=cat_id_list.index(cls_n)
            meta_path = os.path.join(base_path, 'metadata')
            
            #load image and meta file
            meta = json.load(open(meta_path, 'rb'))

            # load 6d pose annotations
            scale = ann['size']
            rot = ann['rotation']
            trans = ann['translation']
            RTs = np.eye(4)
            RTs[:3, :3] = rot
            RTs[:3, 3] = trans
            K = np.array(meta['K']).reshape(3, 3).T

            img_mask_path = os.path.join(base_path, 'images/{}-mask.png'.format(int(frame_idx)))
            img_mask = img=cv2.imread(img_mask_path, cv2.IMREAD_GRAYSCALE)
            bbox=gen_visb_bbox(img_mask)

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
        except:
            pass

        if img_annotation_counter == 0:
                print("Image skipped! No annotations valid!")
                continue
        img_annotation = { #获得img_annotation
                'file_name': img_path,
                'id': image_id,
                'width': 480,
                'height': 640,
                'intrinsics': K.reshape(9).tolist(),
                'type' : "test_wild6d"
            }
        annotations['images'].append(img_annotation)
        image_id += 1

#print(annotations)
with open(output_base_path + annotation_path, 'w') as out_file:
    json.dump(annotations, out_file , indent=4)