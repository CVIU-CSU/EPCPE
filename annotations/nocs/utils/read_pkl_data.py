import _pickle as cPickle


dir='/root/commonfile/fxf/nocs/CAMERA/train/01521/0001_label.pkl'

with open(dir,'rb') as f:
    label=cPickle.load(f)
for i in range(len(label['class_ids'])):
    bbox_size=label['size'][i]
    print(label)
    print(bbox_size)
    if 0 in bbox_size:
        print("dirty size data")
