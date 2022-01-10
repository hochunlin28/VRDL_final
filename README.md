# VRDL_Final

## Install Packages

* install pytorch from https://pytorch.org/get-started/previous-versions/

* install mmdetection
  * Please refer to mmdetection for installation.
```
# quick install:
# install pytorch correspond to your cuda version
conda install pytorch==1.10.0 torchvision==0.11.1 cudatoolkit=10.2 -c pytorch
pip install openmim
mim install mmdet
```
## Data Preparation
* First, transform dicom image to png image. Therefore, split train image and validation image to 9:1. 
* transfer_normal_image.py to split the proportion of normal image.
* generate_coco.py to generate coco format of train/val directory.

## Data Downloading
* Please put data in data/coco directory

## Select Config file
* swin transform: swinT.py

## Training
* train model with pretrained model
```
python tools/train.py configs/swinT/swinT.py
```
## download pretrainted model
* After we train, it will get epoch5.pth in work_dir/swin-t_0:1, please put epoch5.pth in work_dir/swin-t_0:1 to run generate_csv.py
https://drive.google.com/file/d/1ARuXW_dw24XpTkbl5H-bxZmMBTBBgTh1/view?usp=sharing

## Generate csv file
* It will generate the csv file. modify the `checkpoint` variable to select different model weight.
```
python generate_csv.py
```

## Team Notes
https://hackmd.io/@Bmj6Z_QbTMy769jUvLGShA/VRDL_Final/edit

## Ensemble

