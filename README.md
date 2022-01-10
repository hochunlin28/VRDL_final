# VRDL_Final

## Install Packages

* install pytorch from https://pytorch.org/get-started/previous-versions/

* install mmdetection
  * Please refer to mmdetection for installation.
```
# quick install:
# install pytorch correspond to your cuda version
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
pip install openmim
mim install mmdet
```
## Data Preparation
* First, transform dicom image to png image. Therefore, split train image and validation image to 9:1. 
* transfer_normal_image.py to split the proportion of normal image.
* generate_coco.py to generate coco format of train/val directory.

## Select Config file
* swin transform: swinT.py

## Download Pretrained Model

## Training
* train model with pretrained model
```
configs/swinT/swinT.py python tools/train.py
```
## Generate csv file
* It will generate the csv file. modify the `checkpoint` variable to select different model weight.
```
python generate_csv.py
```

## Team Notes
https://hackmd.io/@Bmj6Z_QbTMy769jUvLGShA/VRDL_Final/edit

## Ensemble code
* After 
