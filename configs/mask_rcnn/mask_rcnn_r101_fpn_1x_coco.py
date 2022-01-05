_base_ = './mask_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='checkpoint/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth')))
