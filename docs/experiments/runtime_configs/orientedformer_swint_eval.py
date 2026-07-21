angle_version = 'le90'
backbone_embed_multi = dict(decay_mult=0.0, lr_mult=0.1)
backbone_norm_multi = dict(decay_mult=0.0, lr_mult=0.1)
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'projects.OrientedFormer.orientedformer',
    ])
custom_keys = dict({
    'absolute_pos_embed':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.patch_embed.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.0.blocks.0.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.0.blocks.1.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.0.downsample.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.1.blocks.0.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.1.blocks.1.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.1.downsample.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.0.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.1.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.2.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.3.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.4.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.5.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.downsample.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.3.blocks.0.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.3.blocks.1.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'level_embed':
    dict(decay_mult=0.0, lr_mult=1.0),
    'query_embed':
    dict(decay_mult=0.0, lr_mult=1.0),
    'query_feat':
    dict(decay_mult=0.0, lr_mult=1.0),
    'relative_position_bias_table':
    dict(decay_mult=0.0, lr_mult=0.1)
})
data_root = '/data5/2025/ldh/orientedformer_protocol_eval_20260702/DIOR_R_geonexus_xml_20260702/'
dataset_type = 'DIORDataset'
default_hooks = dict(
    checkpoint=dict(_scope_='mmrotate', interval=1, type='CheckpointHook'),
    logger=dict(_scope_='mmrotate', interval=50, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmrotate', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmrotate', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmrotate', type='IterTimerHook'),
    visualization=dict(_scope_='mmrotate', type='mmdet.DetVisualizationHook'))
default_scope = 'mmrotate'
depths = [
    2,
    2,
    6,
    2,
]
embed_multi = dict(decay_mult=0.0, lr_mult=1.0)
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(backend='disk')
launcher = 'none'
load_from = '/data5/2025/ldh/orientedformer_protocol_eval_20260702/checkpoints/orientedformer_hf/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dior/epoch_12.pth'
log_level = 'INFO'
log_processor = dict(
    _scope_='mmrotate', by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        convert_weights=True,
        depths=[
            2,
            2,
            6,
            2,
        ],
        drop_path_rate=0.2,
        drop_rate=0.0,
        embed_dims=96,
        init_cfg=dict(
            checkpoint=
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
            type='Pretrained'),
        mlp_ratio=4,
        num_heads=[
            3,
            6,
            12,
            24,
        ],
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_norm=True,
        qk_scale=None,
        qkv_bias=True,
        type='mmdet.SwinTransformer',
        window_size=7,
        with_cp=False),
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
            96,
            192,
            384,
            768,
        ],
        kernel_size=1,
        num_outs=5,
        out_channels=256,
        type='ChannelMapperWithGN'),
    roi_head=dict(
        bbox_head=[
            dict(
                angle_version='le90',
                bbox_coder=dict(type='DeltaXYWHTRBBoxCoder'),
                cls_predictor_cfg=dict(type='mmdet.Linear'),
                content_dim=256,
                ffn_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    embed_dims=256,
                    feedforward_channels=2048,
                    ffn_drop=0.0,
                    num_fcs=2),
                loss_bbox=dict(loss_weight=2.0, type='mmdet.L1Loss'),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type='mmdet.FocalLoss',
                    use_sigmoid=True),
                loss_iou=dict(
                    loss_weight=5.0, mode='linear', type='RotatedIoULoss'),
                num_classes=20,
                num_cls_fcs=1,
                num_reg_fcs=1,
                o3d_attn_cfg=dict(
                    embed_dims=256,
                    n_heads=64,
                    n_points=32,
                    reduction=4,
                    type='OrientedAttention'),
                reg_predictor_cfg=dict(type='mmdet.Linear'),
                self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8),
                target_means=(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ),
                target_stds=(
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ),
                type='OrientedFormerDecoderLayer'),
            dict(
                angle_version='le90',
                bbox_coder=dict(type='DeltaXYWHTRBBoxCoder'),
                cls_predictor_cfg=dict(type='mmdet.Linear'),
                content_dim=256,
                ffn_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    embed_dims=256,
                    feedforward_channels=2048,
                    ffn_drop=0.0,
                    num_fcs=2),
                loss_bbox=dict(loss_weight=2.0, type='mmdet.L1Loss'),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type='mmdet.FocalLoss',
                    use_sigmoid=True),
                loss_iou=dict(
                    loss_weight=5.0, mode='linear', type='RotatedIoULoss'),
                num_classes=20,
                num_cls_fcs=1,
                num_reg_fcs=1,
                o3d_attn_cfg=dict(
                    embed_dims=256,
                    n_heads=64,
                    n_points=32,
                    reduction=4,
                    type='OrientedAttention'),
                reg_predictor_cfg=dict(type='mmdet.Linear'),
                self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8),
                target_means=(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ),
                target_stds=(
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ),
                type='OrientedFormerDecoderLayer'),
        ],
        content_dim=256,
        featmap_strides=[
            4,
            8,
            16,
            32,
            64,
        ],
        num_stages=2,
        stage_loss_weights=[
            1,
            1,
        ],
        type='OrientedAdaMixerDecoder'),
    rpn_head=dict(
        angle_version='le90',
        aux_loss=dict(
            loss_bbox=dict(
                loss_weight=5.0, mode='linear', type='RotatedIoULoss'),
            loss_cls=dict(
                activated=True,
                beta=2.0,
                loss_weight=1.0,
                type='mmdet.QualityFocalLoss',
                use_sigmoid=True),
            train_cfg=dict(
                alpha=1,
                assigner=dict(
                    cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                    iou_calculator=dict(type='RBboxOverlaps2D'),
                    iou_cost=dict(
                        iou_mode='iou', type='RotatedIoUCost', weight=5.0),
                    reg_cost=dict(
                        angle_version='le90',
                        box_format='xywht',
                        type='RBBoxL1Cost',
                        weight=2.0),
                    topk=8,
                    type='TopkHungarianAssigner'),
                beta=6)),
        ddq_num_classes=20,
        dqs_cfg=dict(iou_threshold=0.7, nms_pre=1000, type='nms_rotated'),
        feat_channels=256,
        in_channels=256,
        main_loss=dict(
            loss_bbox=dict(
                loss_weight=5.0, mode='linear', type='RotatedIoULoss'),
            loss_cls=dict(
                activated=True,
                beta=2.0,
                loss_weight=1.0,
                type='mmdet.QualityFocalLoss',
                use_sigmoid=True),
            train_cfg=dict(
                alpha=1,
                assigner=dict(
                    cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                    iou_calculator=dict(type='RBboxOverlaps2D'),
                    iou_cost=dict(
                        iou_mode='iou', type='RotatedIoUCost', weight=5.0),
                    reg_cost=dict(
                        angle_version='le90',
                        box_format='xywht',
                        type='RBBoxL1Cost',
                        weight=2.0),
                    topk=8,
                    type='TopkHungarianAssigner'),
                beta=6)),
        norm_cfg=dict(num_groups=32, requires_grad=True, type='GN'),
        num_proposals=300,
        offset=0.5,
        strides=[
            4,
            8,
            16,
            32,
            64,
        ],
        type='OrientedAdaMixerDDQ'),
    test_cfg=dict(rcnn=dict(max_per_img=300), rpn=None),
    train_cfg=dict(
        rcnn=[
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type='mmdet.FocalLossCost', weight=2.0),
                        dict(
                            angle_version='le90',
                            box_format='xywht',
                            type='RBBoxL1Cost',
                            weight=2.0),
                        dict(
                            iou_mode='iou', type='RotatedIoUCost', weight=5.0),
                    ],
                    type='mmdet.HungarianAssigner'),
                pos_weight=1,
                sampler=dict(type='mmdet.PseudoSampler')),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type='mmdet.FocalLossCost', weight=2.0),
                        dict(
                            angle_version='le90',
                            box_format='xywht',
                            type='RBBoxL1Cost',
                            weight=2.0),
                        dict(
                            iou_mode='iou', type='RotatedIoUCost', weight=5.0),
                    ],
                    type='mmdet.HungarianAssigner'),
                pos_weight=1,
                sampler=dict(type='mmdet.PseudoSampler')),
        ],
        rpn=None),
    type='OrientedDDQRCNN')
num_classes = 20
num_proposals = 300
num_stages = 2
optim_wrapper = dict(
    _scope_='mmrotate',
    clip_grad=dict(max_norm=1, norm_type=2),
    optimizer=dict(lr=5e-05, type='AdamW', weight_decay=1e-06),
    type='OptimWrapper')
param_scheduler = [
    dict(
        _scope_='mmrotate',
        begin=0,
        by_epoch=False,
        end=500,
        start_factor=0.3333333333333333,
        type='LinearLR'),
    dict(
        _scope_='mmrotate',
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
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
resume = False
test_cfg = dict(_scope_='mmrotate', type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmrotate',
        ann_file='ImageSets/Main/test.txt',
        data_prefix=dict(img_path='JPEGImages-test'),
        data_root=
        '/data5/2025/ldh/orientedformer_protocol_eval_20260702/DIOR_R_geonexus_xml_20260702/',
        file_client_args=dict(backend='disk'),
        pipeline=[
            dict(
                _scope_='mmrotate',
                file_client_args=dict(backend='disk'),
                type='mmdet.LoadImageFromFile'),
            dict(
                _scope_='mmrotate',
                keep_ratio=True,
                scale=(
                    800,
                    800,
                ),
                type='mmdet.Resize'),
            dict(
                _scope_='mmrotate',
                box_type='qbox',
                type='mmdet.LoadAnnotations',
                with_bbox=True),
            dict(
                _scope_='mmrotate',
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(
                _scope_='mmrotate',
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
        type='DIORDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(_scope_='mmrotate', shuffle=False, type='DefaultSampler'))
test_evaluator = dict(_scope_='mmrotate', metric='mAP', type='DOTAMetric')
test_pipeline = [
    dict(
        _scope_='mmrotate',
        file_client_args=dict(backend='disk'),
        type='mmdet.LoadImageFromFile'),
    dict(
        _scope_='mmrotate',
        keep_ratio=True,
        scale=(
            800,
            800,
        ),
        type='mmdet.Resize'),
    dict(
        _scope_='mmrotate',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.PackDetInputs'),
]
train_cfg = dict(
    _scope_='mmrotate',
    max_epochs=12,
    type='EpochBasedTrainLoop',
    val_interval=4)
train_dataloader = dict(
    batch_sampler=None,
    batch_size=4,
    dataset=dict(
        _scope_='mmrotate',
        datasets=[
            dict(
                ann_file='ImageSets/Main/train.txt',
                data_prefix=dict(img_path='JPEGImages-trainval'),
                data_root='data/DIOR/',
                filter_cfg=dict(filter_empty_gt=True),
                pipeline=[
                    dict(
                        file_client_args=dict(backend='disk'),
                        type='mmdet.LoadImageFromFile'),
                    dict(
                        box_type='qbox',
                        type='mmdet.LoadAnnotations',
                        with_bbox=True),
                    dict(
                        box_type_mapping=dict(gt_bboxes='rbox'),
                        type='ConvertBoxType'),
                    dict(
                        keep_ratio=True,
                        scale=(
                            800,
                            800,
                        ),
                        type='mmdet.Resize'),
                    dict(
                        direction=[
                            'horizontal',
                            'vertical',
                            'diagonal',
                        ],
                        prob=0.75,
                        type='mmdet.RandomFlip'),
                    dict(type='mmdet.PackDetInputs'),
                ],
                type='DIORDataset'),
            dict(
                ann_file='ImageSets/Main/val.txt',
                data_prefix=dict(img_path='JPEGImages-trainval'),
                data_root='data/DIOR/',
                filter_cfg=dict(filter_empty_gt=True),
                pipeline=[
                    dict(
                        file_client_args=dict(backend='disk'),
                        type='mmdet.LoadImageFromFile'),
                    dict(
                        box_type='qbox',
                        type='mmdet.LoadAnnotations',
                        with_bbox=True),
                    dict(
                        box_type_mapping=dict(gt_bboxes='rbox'),
                        type='ConvertBoxType'),
                    dict(
                        keep_ratio=True,
                        scale=(
                            800,
                            800,
                        ),
                        type='mmdet.Resize'),
                    dict(
                        direction=[
                            'horizontal',
                            'vertical',
                            'diagonal',
                        ],
                        prob=0.75,
                        type='mmdet.RandomFlip'),
                    dict(type='mmdet.PackDetInputs'),
                ],
                type='DIORDataset'),
        ],
        ignore_keys=[
            'DATASET_TYPE',
        ],
        type='ConcatDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(_scope_='mmrotate', shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        _scope_='mmrotate',
        file_client_args=dict(backend='disk'),
        type='mmdet.LoadImageFromFile'),
    dict(
        _scope_='mmrotate',
        box_type='qbox',
        type='mmdet.LoadAnnotations',
        with_bbox=True),
    dict(
        _scope_='mmrotate',
        box_type_mapping=dict(gt_bboxes='rbox'),
        type='ConvertBoxType'),
    dict(
        _scope_='mmrotate',
        keep_ratio=True,
        scale=(
            800,
            800,
        ),
        type='mmdet.Resize'),
    dict(
        _scope_='mmrotate',
        direction=[
            'horizontal',
            'vertical',
            'diagonal',
        ],
        prob=0.75,
        type='mmdet.RandomFlip'),
    dict(_scope_='mmrotate', type='mmdet.PackDetInputs'),
]
val_cfg = dict(_scope_='mmrotate', type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmrotate',
        ann_file='ImageSets/Main/test.txt',
        data_prefix=dict(img_path='JPEGImages-test'),
        data_root=
        '/data5/2025/ldh/orientedformer_protocol_eval_20260702/DIOR_R_geonexus_xml_20260702/',
        file_client_args=dict(backend='disk'),
        pipeline=[
            dict(
                _scope_='mmrotate',
                file_client_args=dict(backend='disk'),
                type='mmdet.LoadImageFromFile'),
            dict(
                _scope_='mmrotate',
                keep_ratio=True,
                scale=(
                    800,
                    800,
                ),
                type='mmdet.Resize'),
            dict(
                _scope_='mmrotate',
                box_type='qbox',
                type='mmdet.LoadAnnotations',
                with_bbox=True),
            dict(
                _scope_='mmrotate',
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(
                _scope_='mmrotate',
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
        type='DIORDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(_scope_='mmrotate', shuffle=False, type='DefaultSampler'))
val_evaluator = dict(_scope_='mmrotate', metric='mAP', type='DOTAMetric')
val_pipeline = [
    dict(
        _scope_='mmrotate',
        file_client_args=dict(backend='disk'),
        type='mmdet.LoadImageFromFile'),
    dict(
        _scope_='mmrotate',
        keep_ratio=True,
        scale=(
            800,
            800,
        ),
        type='mmdet.Resize'),
    dict(
        _scope_='mmrotate',
        box_type='qbox',
        type='mmdet.LoadAnnotations',
        with_bbox=True),
    dict(
        _scope_='mmrotate',
        box_type_mapping=dict(gt_bboxes='rbox'),
        type='ConvertBoxType'),
    dict(
        _scope_='mmrotate',
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
    dict(_scope_='mmrotate', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmrotate',
    name='visualizer',
    type='RotLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/data5/2025/ldh/orientedformer_protocol_eval_20260702/orientedformer_swin_t_dior_r_geonexus_eval_20260704_rerun'
