dataset_type = 'CocoDatasetSUNRGBD'
data_root = '/root/commonfile/fxf/sunrgbd/'
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotationsPhocal', with_bbox=True, with_pose=True),
    dict(type='Resize', scale=(730, 530), keep_ratio=True),
    dict(type='RandomFlip', prob=0.0),
    dict(
        type='RandAugment',
        aug_space=[[{
            'type': 'Sharpness'
        }], [{
            'type': 'Color'
        }], [{
            'type': 'Brightness'
        }]]),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='Resize', scale=(730, 530), keep_ratio=True),
    dict(type='LoadAnnotationsPhocal', with_bbox=True, with_pose=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDatasetSUNRGBD',
        data_root='/root/commonfile/fxf/sunrgbd/',
        ann_file='annotations/sunrgbd/train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotationsPhocal', with_bbox=True, with_pose=True),
            dict(type='Resize', scale=(730, 530), keep_ratio=True),
            dict(type='RandomFlip', prob=0.0),
            dict(
                type='RandAugment',
                aug_space=[[{
                    'type': 'Sharpness'
                }], [{
                    'type': 'Color'
                }], [{
                    'type': 'Brightness'
                }]]),
            dict(type='PackDetInputs')
        ]))
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDatasetSUNRGBD',
        data_root='/root/commonfile/fxf/sunrgbd/',
        ann_file='annotations/sunrgbd/test.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='Resize', scale=(730, 530), keep_ratio=True),
            dict(type='LoadAnnotationsPhocal', with_bbox=True, with_pose=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))
test_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDatasetSUNRGBD',
        data_root='/root/commonfile/fxf/sunrgbd/',
        ann_file='annotations/sunrgbd/test.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='Resize', scale=(730, 530), keep_ratio=True),
            dict(type='LoadAnnotationsPhocal', with_bbox=True, with_pose=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))
val_evaluator = dict(
    type='CocoMetricCPPF',
    ann_file='/root/commonfile/fxf/sunrgbd/annotations/sunrgbd/test.json',
    metric=['pose', 'bbox'],
    format_only=False,
    synset_names=[
        'BG', 'bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
        'night_stand', 'bookshelf', 'bathtub'
    ],
    dataset_name='sunrgbd',
    full_rot=True)
test_evaluator = dict(
    type='CocoMetricCPPF',
    ann_file='/root/commonfile/fxf/sunrgbd/annotations/sunrgbd/test.json',
    metric=['pose', 'bbox'],
    format_only=False,
    synset_names=[
        'BG', 'bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
        'night_stand', 'bookshelf', 'bathtub'
    ],
    dataset_name='sunrgbd',
    full_rot=True)
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
model = dict(
    type='DeformableDETR',
    num_queries=100,
    num_feature_levels=4,
    with_box_refine=False,
    as_two_stage=False,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='VisionTransformer',
        arch='base',
        img_size=518,
        patch_size=14,
        layer_scale_init_value=1e-05,
        out_type='featmap',
        out_indices=(5, 8, 11),
        frozen_stages=12,
        pretrained='dino-b'),
    neck=dict(
        type='ChannelMapper',
        in_channels=[768, 768, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=3,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, batch_first=True),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1))),
    decoder=dict(
        num_layers=3,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, num_heads=8, dropout=0.1, batch_first=True),
            cross_attn_cfg=dict(embed_dims=256, batch_first=True),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1)),
        post_norm_cfg=None),
    positional_encoding=dict(num_feats=128, normalize=True, offset=-0.5),
    bbox_head=dict(
        type='DeformableDETRHeadNOCSNormSS',
        num_classes=10,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=4.0),
        loss_R=dict(type='SmoothL1Loss', loss_weight=5.0),
        loss_RE=dict(type='SmoothL1Loss', loss_weight=3.0),
        loss_T=dict(type='MSELoss', loss_weight=6.0),
        loss_size=dict(type='MSELoss', loss_weight=6.0),
        loss_scale=dict(type='MSELoss', loss_weight=4.0)),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0, eps=1e-08),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=100))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.0),
            sampling_offsets=dict(lr_mult=0.1),
            reference_points=dict(lr_mult=0.1))))
max_epochs = 200
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=50)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=200,
        by_epoch=True,
        milestones=[40],
        gamma=0.1)
]
auto_scale_lr = dict(base_batch_size=32, enable=True)
launcher = 'pytorch'
work_dir = './work_dirs/deformable-detr_sunrgbd_ViTDINO_3enc_3dec'
