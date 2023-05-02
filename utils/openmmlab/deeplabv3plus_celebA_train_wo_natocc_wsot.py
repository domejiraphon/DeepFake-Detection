norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=30000)
checkpoint_config = dict(by_epoch=False, interval=400)
evaluation = dict(
    interval=400, metric=['mIoU', 'mDice', 'mFscore'], pre_eval=True)
dataset_type = 'FaceOccluded'
data_root = 'data'
crop_size = (512, 512)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', degree=(-30, 30), prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
dataset_train_A = dict(
    type='FaceOccluded',
    data_root='data',
    img_dir='CelebAMask-HQ-original/image',
    ann_dir='CelebAMask-HQ-original/mask_edited',
    split='CelebAMask-HQ-original/split/train_wo.txt',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(512, 512)),
        dict(type='RandomFlip', prob=0.5),
        dict(type='RandomRotate', degree=(-30, 30), prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ])
dataset_train_B = dict(
    type='FaceOccluded',
    data_root='data',
    img_dir='Face-Occluded-Dataset/celeba_train-wo-aug_sot/image',
    ann_dir='Face-Occluded-Dataset/celeba_train-wo-aug_sot/mask',
    split='Face-Occluded-Dataset/split/train.txt',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(512, 512)),
        dict(type='RandomFlip', prob=0.5),
        dict(type='RandomRotate', degree=(-30, 30), prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ])
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=[
        dict(
            type='FaceOccluded',
            data_root='data',
            img_dir='CelebAMask-HQ-original/image',
            ann_dir='CelebAMask-HQ-original/mask_edited',
            split='CelebAMask-HQ-original/split/train_wo.txt',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(type='Resize', img_scale=(512, 512)),
                dict(type='RandomFlip', prob=0.5),
                dict(type='RandomRotate', degree=(-30, 30), prob=0.5),
                dict(type='PhotoMetricDistortion'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ]),
        dict(
            type='FaceOccluded',
            data_root='data',
            img_dir='Face-Occluded-Dataset/celeba_train-wo-aug_sot/image',
            ann_dir='Face-Occluded-Dataset/celeba_train-wo-aug_sot/mask',
            split='Face-Occluded-Dataset/split/train.txt',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(type='Resize', img_scale=(512, 512)),
                dict(type='RandomFlip', prob=0.5),
                dict(type='RandomRotate', degree=(-30, 30), prob=0.5),
                dict(type='PhotoMetricDistortion'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ])
    ],
    val=dict(
        type='FaceOccluded',
        data_root='data',
        img_dir='Occluded-Face-Mask-Dataset/image',
        ann_dir='Occluded-Face-Mask-Dataset/mask',
        split='Occluded-Face-Mask-Dataset/split/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='ResizeToMultiple', size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='FaceOccluded',
        data_root='data',
        img_dir='kennyvoo_Failed_aligned/image',
        ann_dir='kennyvoo_Failed_aligned/mask',
        split='kennyvoo_Failed_aligned/split.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='ResizeToMultiple', size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
work_dir = './work_dirs/deeplabv3plus_celebA_train_wo+no_aug_sot'
gpu_ids = range(0, 4)
auto_resume = False
