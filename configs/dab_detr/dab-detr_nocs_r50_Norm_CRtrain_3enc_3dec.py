__author__= 'fanxiaofeng'
_base_ = [
    '../_base_/datasets/coco_detection_on_nocs3d_ViT.py', '../_base_/default_runtime.py'
]
model = dict(
    type='DABDETRPhoCal',
    num_queries=100,
    with_random_refpoints=False,
    num_patterns=0,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=None,
        num_outs=1),
    encoder=dict(
        num_layers=3,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, num_heads=8, dropout=0., batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.,
                act_cfg=dict(type='PReLU')))),
    decoder=dict(
        num_layers=3,
        query_dim=4,
        query_scale_type='cond_elewise',
        with_modulated_hw_attn=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256,
                num_heads=8,
                attn_drop=0.,
                proj_drop=0.,
                cross_attn=False),
            cross_attn_cfg=dict(
                embed_dims=256,
                num_heads=8,
                attn_drop=0.,
                proj_drop=0.,
                cross_attn=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.,
                act_cfg=dict(type='PReLU'))),
        return_intermediate=True),
    positional_encoding=dict(num_feats=128, temperature=20, normalize=True),
    bbox_head=dict(
        type='DABDETRHeadNOCSNorm',
        num_classes=6,
        embed_dims=256,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=4.0),
        loss_R=dict(type='SmoothL1Loss', loss_weight=5.0),
        loss_RE=dict(type='SmoothL1Loss', loss_weight=3.0),
        loss_T=dict(type='MSELoss', loss_weight=6.0),
        loss_size=dict(type='MSELoss', loss_weight=6.0),
        loss_scale=dict(type='MSELoss', loss_weight=4.0),
        ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2., eps=1e-8),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])
            ),
        
    test_cfg=dict(max_per_img=100))


# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
file_client_args = dict(backend='disk')
dataset_type = 'CocoDatasetNOCS3D'
data_root = '/root/commonfile/fxf/nocs/' #2080

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotationsPhocal', with_bbox=True,with_pose=True), #这里加入导入rot pos label
    dict(type='Resize', scale=(640, 480), keep_ratio=True),
    dict(type='RandomFlip', prob=0.0), 
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(640, 480), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotationsPhocal', with_bbox=True,with_pose=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/nocs/crtrain.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/nocs/test_real.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetricNOCS', #要改
    ann_file=data_root + 'annotations/nocs/test_real.json',
    metric=['pose','bbox'],
    #metric='pose',
    #metric='proposal_fast',
    format_only=False)
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
                    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
                    'S_Adapter': dict(lr_mult=1.0 ),
                    # 'patch_embed': dict(lr_mult=0.0), #冻结adapter以外的backbone
                    # 'pose_embed': dict(lr_mult=0.0),
                    # 'neck': dict(lr_mult=0.0),
                    # 'norm1': dict(lr_mult=0.0),
                    # 'norm2': dict(lr_mult=0.0),
                    # 'attn': dict(lr_mult=0.0),
                    # 'mlp': dict(lr_mult=0.0),
                    # 'encoder': dict(lr_mult=1.0),
                    # 'decoder': dict(lr_mult=1.0),
                    }))

# learning policy
max_epochs = 48
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[40],
        gamma=0.1)
]

checkpoint_config = dict(interval=2)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16, enable=False)
#checkpoint_config = dict(interval=5)
