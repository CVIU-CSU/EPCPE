#!/bin/bash

epoch_array=(24 23 22 21 20 12)
echo ${epoch_array[@]}

epoch_dir="/root/userfolder/github/mmdetection/work_dirs/dab-detr_nocs_ViTDET_Norm_CRtrain_4enc_4dec/"
config_file="/root/userfolder/github/mmdetection/work_dirs/dab-detr_nocs_ViTDET_Norm_CRtrain_4enc_4dec/dab-detr_nocs_ViTDET_Norm_CRtrain_4enc_4dec_eval_on_camera.py"
work_dir="/root/userfolder/github/mmdetection/work_dirs/dab-detr_nocs_ViTDET_Norm_CRtrain_4enc_4dec/eval_on_objectron/camera"

gpu_id=0

for epoch in ${epoch_array[@]}
do
echo CUDA_VISIBLE_DEVICES="$gpu_id" python tools/test.py $config_file $epoch_dir'epoch_'$epoch'.pth' --work-dir $work_dir 
CUDA_VISIBLE_DEVICES="$gpu_id" python tools/test.py $config_file $epoch_dir'epoch_'$epoch'.pth' --work-dir $work_dir 
done

