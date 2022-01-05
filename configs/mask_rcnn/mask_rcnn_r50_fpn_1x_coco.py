_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

load_from = 'checkpoint/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth'