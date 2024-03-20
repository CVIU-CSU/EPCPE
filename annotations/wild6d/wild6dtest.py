import _pickle as cPickle

dir='/root/commonfile/fxf/wild6d/test_set/pkl_annotations/mug/mug-0001-1.pkl'

with open (dir,'rb') as f:
    data=cPickle.load(f)

print(data)
print(len(data))
print(data['intrinsics'])
#print(data['rotation'])
print(data.keys())
print(data['num_frames'])