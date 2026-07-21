_prompt_embedding_path = '/data5/2025/ldh/New/artifacts/generated/remoteclip_vit_b32_dior_r_s2_hierarchy_prompt_embeddings.pt'
angle_version = 'le90'
ckpt_interval = 4
class_name = [
    'airplane',
    'airport',
    'baseballfield',
    'basketballcourt',
    'bridge',
    'chimney',
    'dam',
    'Expressway-Service-area',
    'Expressway-toll-station',
    'golffield',
    'groundtrackfield',
    'harbor',
    'overpass',
    'ship',
    'stadium',
    'storagetank',
    'tenniscourt',
    'trainstation',
    'vehicle',
    'windmill',
]
custom_hooks = [
    dict(type='mmdet.NumClassCheckHook'),
    dict(
        ema_type='mmdet.ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='EMAHook',
        update_buffers=True),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmrotate.models',
        'mmrotate.datasets',
        'geonexus_mmrotate.prompt_bbox_head',
    ])
data_root = 'data/DIOR_R_dota_sanitized_invalidsize_20260612/'
dataset_type = 'DOTADataset'
default_hooks = dict(
    checkpoint=dict(interval=4, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmrotate'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(backend='disk')
img_scale = (
    800,
    800,
)
launcher = 'none'
load_from = 'work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_s2e12_rep0_20260614/epoch_8.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 12
metainfo = dict(
    classes=[
        'airplane',
        'airport',
        'baseballfield',
        'basketballcourt',
        'bridge',
        'chimney',
        'dam',
        'Expressway-Service-area',
        'Expressway-toll-station',
        'golffield',
        'groundtrackfield',
        'harbor',
        'overpass',
        'ship',
        'stadium',
        'storagetank',
        'tenniscourt',
        'trainstation',
        'vehicle',
        'windmill',
    ],
    palette=[
        (
            220,
            20,
            60,
        ),
    ])
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='mmdet.ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        boxtype2tensor=False,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='mmdet.DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        type='mmdet.FPN'),
    roi_head=dict(
        bbox_head=[
            dict(
                bbox_coder=dict(
                    angle_version='le90',
                    edge_swap=True,
                    norm_factor=2,
                    target_means=(
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ),
                    target_stds=(
                        0.1,
                        0.1,
                        0.2,
                        0.2,
                        0.1,
                    ),
                    type='DeltaXYWHTHBBoxCoder',
                    use_box_type=True),
                cls_predictor_cfg=dict(type='mmdet.Linear'),
                fc_out_channels=1024,
                in_channels=256,
                learnable_prompt_bias=True,
                learnable_prompt_offsets=True,
                loss_bbox=dict(
                    beta=1.0, loss_weight=1.0, type='mmdet.SmoothL1Loss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='mmdet.CrossEntropyLoss',
                    use_sigmoid=False),
                num_classes=20,
                predict_box_type='rbox',
                prompt_dim=512,
                prompt_embedding_key='embeddings',
                prompt_embedding_path=
                '/data5/2025/ldh/New/artifacts/generated/remoteclip_vit_b32_dior_r_s2_hierarchy_prompt_embeddings.pt',
                prompt_logit_scale=10.0,
                reg_class_agnostic=True,
                reg_predictor_cfg=dict(type='mmdet.Linear'),
                roi_feat_size=7,
                scene_adapter_dim=256,
                scene_adapter_identity_init=True,
                scene_adapter_residual_scale=0.1,
                type='PromptShared2FCBBoxHead',
                use_scene_adapter=True),
            dict(
                bbox_coder=dict(
                    angle_version='le90',
                    edge_swap=True,
                    norm_factor=None,
                    proj_xy=True,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.05,
                        0.05,
                        0.1,
                        0.1,
                        0.05,
                    ],
                    type='DeltaXYWHTRBBoxCoder'),
                cls_predictor_cfg=dict(type='mmdet.Linear'),
                fc_out_channels=1024,
                in_channels=256,
                learnable_prompt_bias=True,
                learnable_prompt_offsets=True,
                loss_bbox=dict(
                    beta=1.0, loss_weight=1.0, type='mmdet.SmoothL1Loss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='mmdet.CrossEntropyLoss',
                    use_sigmoid=False),
                num_classes=20,
                predict_box_type='rbox',
                prompt_dim=512,
                prompt_embedding_key='embeddings',
                prompt_embedding_path=
                '/data5/2025/ldh/New/artifacts/generated/remoteclip_vit_b32_dior_r_s2_hierarchy_prompt_embeddings.pt',
                prompt_logit_scale=10.0,
                reg_class_agnostic=False,
                reg_predictor_cfg=dict(type='mmdet.Linear'),
                roi_feat_size=7,
                scene_adapter_dim=256,
                scene_adapter_identity_init=True,
                scene_adapter_residual_scale=0.1,
                type='PromptShared2FCBBoxHead',
                use_scene_adapter=True),
        ],
        bbox_roi_extractor=[
            dict(
                featmap_strides=[
                    4,
                    8,
                    16,
                    32,
                ],
                out_channels=256,
                roi_layer=dict(
                    output_size=7, sampling_ratio=0, type='RoIAlign'),
                type='mmdet.SingleRoIExtractor'),
            dict(
                featmap_strides=[
                    4,
                    8,
                    16,
                    32,
                ],
                out_channels=256,
                roi_layer=dict(
                    clockwise=True,
                    out_size=7,
                    sample_num=2,
                    type='RoIAlignRotated'),
                type='RotatedSingleRoIExtractor'),
        ],
        num_stages=2,
        stage_loss_weights=[
            1,
            1,
        ],
        type='mmdet.CascadeRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='mmdet.AnchorGenerator',
            use_box_type=True),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHHBBoxCoder',
            use_box_type=True),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(
            beta=0.1111111111111111,
            loss_weight=1.0,
            type='mmdet.SmoothL1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='mmdet.CrossEntropyLoss', use_sigmoid=True),
        type='mmdet.RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=2000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.1, type='nms_rotated'),
            nms_pre=2000,
            score_thr=0.05),
        rpn=dict(
            max_per_img=2000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    train_cfg=dict(
        rcnn=[
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='RBbox2HBboxOverlaps2D'),
                    match_low_quality=False,
                    min_pos_iou=0.5,
                    neg_iou_thr=0.5,
                    pos_iou_thr=0.5,
                    type='mmdet.MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='mmdet.RandomSampler')),
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='RBboxOverlaps2D'),
                    match_low_quality=False,
                    min_pos_iou=0.5,
                    neg_iou_thr=0.5,
                    pos_iou_thr=0.5,
                    type='mmdet.MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='mmdet.RandomSampler')),
        ],
        rpn=dict(
            allowed_border=0,
            assigner=dict(
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBbox2HBboxOverlaps2D'),
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='mmdet.MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='mmdet.RandomSampler')),
        rpn_proposal=dict(
            max_per_img=2000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    type='mmdet.CascadeRCNN')
num_classes = 20
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.0025, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=500,
        start_factor=0.3333333333333333,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
]
randomness = dict(deterministic=False, seed=13407)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='test/labelTxt/',
        data_prefix=dict(img_path='test/images/'),
        data_root='data/DIOR_R_dota_sanitized_invalidsize_20260612/',
        filter_cfg=dict(filter_empty_gt=True),
        img_shape=(
            800,
            800,
        ),
        metainfo=dict(
            classes=[
                'airplane',
                'airport',
                'baseballfield',
                'basketballcourt',
                'bridge',
                'chimney',
                'dam',
                'Expressway-Service-area',
                'Expressway-toll-station',
                'golffield',
                'groundtrackfield',
                'harbor',
                'overpass',
                'ship',
                'stadium',
                'storagetank',
                'tenniscourt',
                'trainstation',
                'vehicle',
                'windmill',
            ],
            palette=[
                (
                    220,
                    20,
                    60,
                ),
            ]),
        pipeline=[
            dict(
                file_client_args=dict(backend='disk'),
                type='mmdet.LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                800,
                800,
            ), type='mmdet.Resize'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    800,
                    800,
                ),
                type='mmdet.Pad'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='DOTADataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(metric='mAP', type='DOTAMetric')
test_pipeline = [
    dict(
        file_client_args=dict(backend='disk'), type='mmdet.LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        800,
        800,
    ), type='mmdet.Resize'),
    dict(
        pad_val=dict(img=(
            114,
            114,
            114,
        )),
        size=(
            800,
            800,
        ),
        type='mmdet.Pad'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.PackDetInputs'),
]
train_ann_file = 'train_val/labelTxt/'
train_batch_size = 2
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=4)
train_dataloader = dict(
    batch_sampler=None,
    batch_size=2,
    dataset=dict(
        ann_file='train_val/labelTxt/',
        data_prefix=dict(img_path='train_val/images/'),
        data_root='data/DIOR_R_dota_sanitized_invalidsize_20260612/',
        filter_cfg=dict(filter_empty_gt=True),
        img_shape=(
            800,
            800,
        ),
        metainfo=dict(
            classes=[
                'airplane',
                'airport',
                'baseballfield',
                'basketballcourt',
                'bridge',
                'chimney',
                'dam',
                'Expressway-Service-area',
                'Expressway-toll-station',
                'golffield',
                'groundtrackfield',
                'harbor',
                'overpass',
                'ship',
                'stadium',
                'storagetank',
                'tenniscourt',
                'trainstation',
                'vehicle',
                'windmill',
            ],
            palette=[
                (
                    220,
                    20,
                    60,
                ),
            ]),
        pipeline=[
            dict(
                file_client_args=dict(backend='disk'),
                type='mmdet.LoadImageFromFile'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(keep_ratio=True, scale=(
                800,
                800,
            ), type='mmdet.Resize'),
            dict(
                direction=[
                    'horizontal',
                    'vertical',
                    'diagonal',
                ],
                prob=0.75,
                type='mmdet.RandomFlip'),
            dict(
                angle_range=180,
                prob=0.5,
                rect_obj_labels=[
                    9,
                    11,
                ],
                type='RandomRotate'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    800,
                    800,
                ),
                type='mmdet.Pad'),
            dict(type='mmdet.PackDetInputs'),
        ],
        type='DOTADataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_img_path = 'train_val/images/'
train_pipeline = [
    dict(
        file_client_args=dict(backend='disk'), type='mmdet.LoadImageFromFile'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(keep_ratio=True, scale=(
        800,
        800,
    ), type='mmdet.Resize'),
    dict(
        direction=[
            'horizontal',
            'vertical',
            'diagonal',
        ],
        prob=0.75,
        type='mmdet.RandomFlip'),
    dict(
        angle_range=180,
        prob=0.5,
        rect_obj_labels=[
            9,
            11,
        ],
        type='RandomRotate'),
    dict(
        pad_val=dict(img=(
            114,
            114,
            114,
        )),
        size=(
            800,
            800,
        ),
        type='mmdet.Pad'),
    dict(type='mmdet.PackDetInputs'),
]
val_ann_file = 'test/labelTxt/'
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='test/labelTxt/',
        data_prefix=dict(img_path='test/images/'),
        data_root='data/DIOR_R_dota_sanitized_invalidsize_20260612/',
        filter_cfg=dict(filter_empty_gt=True),
        img_shape=(
            800,
            800,
        ),
        metainfo=dict(
            classes=[
                'airplane',
                'airport',
                'baseballfield',
                'basketballcourt',
                'bridge',
                'chimney',
                'dam',
                'Expressway-Service-area',
                'Expressway-toll-station',
                'golffield',
                'groundtrackfield',
                'harbor',
                'overpass',
                'ship',
                'stadium',
                'storagetank',
                'tenniscourt',
                'trainstation',
                'vehicle',
                'windmill',
            ],
            palette=[
                (
                    220,
                    20,
                    60,
                ),
            ]),
        pipeline=[
            dict(
                file_client_args=dict(backend='disk'),
                type='mmdet.LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                800,
                800,
            ), type='mmdet.Resize'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    800,
                    800,
                ),
                type='mmdet.Pad'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='DOTADataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(metric='mAP', type='DOTAMetric')
val_img_path = 'test/images/'
val_interval = 4
val_pipeline = [
    dict(
        file_client_args=dict(backend='disk'), type='mmdet.LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        800,
        800,
    ), type='mmdet.Resize'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(
        pad_val=dict(img=(
            114,
            114,
            114,
        )),
        size=(
            800,
            800,
        ),
        type='mmdet.Pad'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.PackDetInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='RotLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dirs/paper_eval_20260617/dior_r_s3_rep0_epoch8'
