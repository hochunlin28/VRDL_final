_base_ = '../swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py'

model = dict(
    roi_head=dict(
        mask_head=dict(num_classes=1),
        bbox_head=dict(num_classes=1),
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                gpu_assign_thr=50,
            ),
        ),
        rcnn=dict(
            assigner=dict(
                gpu_assign_thr=50,
            ),
        )
    )
)

dataset_type = 'COCODataset'
classes = ('pneumonia',)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        img_prefix='data/coco/train',
        classes=classes,
        ann_file='data/coco/annotations/train_coco.json'),
    val=dict(
        img_prefix='data/coco/val',
        classes=classes,
        ann_file='data/coco/annotations/val_coco.json'),
    test=dict(
        img_prefix='data/coco/test',
        classes=classes,
        ann_file='data/coco/annotations/test_coco.json'))

optimizer = dict(
    type='AdamW',
    # *0.1 1214
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
# optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001)
# fp16 = dict(loss_scale=512.)

# 1 epoch for training and 1 epoch for validation will be run iteratively.
# workflow = [('train', 1), ('val', 1)]

# runner = dict(type='EpochBasedRunner', max_epochs=36)


# pretrained_model
# load_from = './checkpoints/mask_rcnn_swin_tiny_patch4_window7.pth'

# ep36 (best)
load_from = None

# If stop accidentally
# load_from = './work_dirs/1214_ep72_mask_rcnn_swin-t/latest.pth'
# resume_from = './work_dirs/1214_ep72_mask_rcnn_swin-t/latest.pth'

# 1213_ep
work_dir = './work_dirs/swin-t_0.3:1'