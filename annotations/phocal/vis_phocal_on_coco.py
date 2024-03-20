from utils import vis_utils
from utils import data_utils
import json
import cv2
import numpy as np

annotation_file='train.json'

draw_cat_id=[6]

with open (annotation_file,'r') as f:
    annos=json.load(f)

print(len(annos['annotations']))
print(len(annos['images']))

for img in annos['images']:
    img_id=img['id']
    filename=img['file_name']
    image=cv2.imread(filename)
    for anno,iter in zip(annos['annotations'],range(len(annos['annotations']))):
        bbox=anno['bbox']
        cat_id=anno['category_id']
        bbox_3d=np.array(anno['bbox_3d'])
        image_id=anno['image_id']
        if image_id!=img_id:
            break
        elif cat_id not in draw_cat_id:
            continue
        else:
            cv2.rectangle(image,(int(bbox[0]),int(bbox[1])),(int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])),60,thickness=2)
            vis_utils.draw_3d_box(image, bbox_3d, color='b' ,linewidth=1)
    cv2.imwrite("annos.png",image)
    cv2.waitKey(0)
    break
        

    
