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
## Data Downloading
* Please put data in data/coco directory
https://drive.google.com/file/d/16U4OkGQ_403cQapEIWxC-W3FWuJ9fgId/view?usp=sharing

## Data Preparation
* First, transform dicom image to png image. Therefore, split train image and validation image to 9:1. 
* transfer_normal_image.py to split the proportion of normal image, we can modeify `scale` variable to change the proportion.
* generate_coco.py to generate coco format of train/val directory, it will get train_coco.json and val_coco.json.

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

## Inference
* It will generate result.box.json and result.segm.json file and we can analyze with these files
```
python tools/test.py configs/swinT/swinT.py ./work_dir/swin-t_0:1/epoch_5.pth --format-only --options "jsonfile_prefix=./results"

```

## Generate csv file
* It will generate the csv file which will uploaded to kaggle. Please Modify the `checkpoint` variable to select different model weight.
```
python generate_csv.py
```

## Team Notes
https://hackmd.io/@Bmj6Z_QbTMy769jUvLGShA/VRDL_Final/edit

## Ensemble

