dataset_type = 'CocoDatasetNOCS3D'
data_root = '/root/commonfile/fxf/nocs/'
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotationsPhocal', with_bbox=True, with_pose=True),
    dict(type='Resize', scale=(640, 480), keep_ratio=True),
    dict(type='RandomFlip', prob=0.0),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='Resize', scale=(640, 480), keep_ratio=True),
    dict(type='LoadAnnotationsPhocal', with_bbox=True, with_pose=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=8,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDatasetNOCS3D',
        data_root='/root/commonfile/fxf/nocs/',
        ann_file='annotations/nocs/crtrain.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotationsPhocal', with_bbox=True, with_pose=True),
            dict(type='Resize', scale=(640, 480), keep_ratio=True),
            dict(type='RandomFlip', prob=0.0),
            dict(type='PackDetInputs')
        ]))
val_dataloader = dict(
    batch_size=8,
    num_workers=16,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDatasetNOCS3D',
        data_root='/root/commonfile/fxf/nocs/',
        ann_file='annotations/nocs/test_real.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='Resize', scale=(640, 480), keep_ratio=True),
            dict(type='LoadAnnotationsPhocal', with_bbox=True, with_pose=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))
test_dataloader = dict(
    batch_size=8,
    num_workers=16,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDatasetNOCS3D',
        data_root='/root/commonfile/fxf/nocs/',
        ann_file='annotations/nocs/test_real.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='Resize', scale=(640, 480), keep_ratio=True),
            dict(type='LoadAnnotationsPhocal', with_bbox=True, with_pose=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))
val_evaluator = dict(
    type='CocoMetricNOCS',
    ann_file='/root/commonfile/fxf/nocs/annotations/nocs/test_real.json',
    metric=['pose', 'bbox'],
    format_only=False)
test_evaluator = dict(
    type='CocoMetricNOCS',
    ann_file='/root/commonfile/fxf/nocs/annotations/nocs/test_real.json',
    metric=['pose', 'bbox'],
    format_only=False)
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=2),
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
        type='VisionTransformer',
        arch='base',
        img_size=518,
        patch_size=14,
        layer_scale_init_value=1e-05,
        out_type='featmap',
        out_indices=(11, ),
        frozen_stages=12,
        pretrained='dino-b'),
    neck=dict(
        type='ChannelMapper',
        in_channels=[768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=None,
        num_outs=1),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, num_heads=8, dropout=0.0, batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.0,
                act_cfg=dict(type='PReLU')))),
    decoder=dict(
        num_layers=6,
        query_dim=4,
        query_scale_type='cond_elewise',
        with_modulated_hw_attn=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256,
                num_heads=8,
                attn_drop=0.0,
                proj_drop=0.0,
                cross_attn=False),
            cross_attn_cfg=dict(
                embed_dims=256,
                num_heads=8,
                attn_drop=0.0,
                proj_drop=0.0,
                cross_attn=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.0,
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
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.0))))
max_epochs = 48
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=48, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=48,
        by_epoch=True,
        milestones=[40],
        gamma=0.1)
]
auto_scale_lr = dict(base_batch_size=8, enable=False)
launcher = 'pytorch'
work_dir = './work_dirs/dab-detr_nocs_ViTDINO_Norm_CRtrain_6enc_6dec'
