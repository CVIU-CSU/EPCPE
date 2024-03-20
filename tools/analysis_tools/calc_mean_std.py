import numpy as np
import os
import cv2


data_root='/root/commonfile/DCIS/whole_data/image'

mean=np.zeros([3],dtype=np.float32)
var=np.zeros([3],dtype=np.float32)

rchannel=0
gchannel=0
bchannel=0
num=0

for file in os.listdir(data_root):
    path=os.path.join(data_root,file)
    if path.endswith('jpg'):
        png=cv2.imread(path)
        shape=png.shape[:2]
        num+=shape[0]*shape[1]
        rchannel+=np.sum(png[:,:,2])
        gchannel+=np.sum(png[:,:,1])
        bchannel+=np.sum(png[:,:,0])

mean[2]=rchannel/num
mean[1]=gchannel/num
mean[0]=bchannel/num

rchannel=0
gchannel=0
bchannel=0

for file in os.listdir(data_root):
    path=os.path.join(data_root,file)
    if path.endswith('jpg'):
        png=cv2.imread(path)
        rchannel+=np.sum((png[:,:,2]-mean[2])**2)
        gchannel+=np.sum((png[:,:,1]-mean[1])**2)
        bchannel+=np.sum((png[:,:,0]-mean[0])**2)

var[2] = rchannel/num
var[1] = gchannel/num
var[0] = bchannel/num

print('mean:',mean)
print('var:',var)
