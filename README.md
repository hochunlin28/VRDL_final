# VRDL_Final

## Install Packages

* install pytorch from https://pytorch.org/get-started/previous-versions/

* install mmdetection
  * Please refer to [mmdetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation.
```
# quick install for cuda 10.2:
# install pytorch correspond to your cuda version
conda install pytorch==1.10.0 torchvision==0.11.1 cudatoolkit=10.2 -c pytorch
pip install openmim
mim install mmdet

```
* install packages
```
pip install tqdm
pip install opencv-python  
pip install pandas
pip install pydicom
pip install future tensorboard

```

### Apex (optional):
We use apex for mixed precision training by default. To install apex, run:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Data Preparation
* First, transform dicom image to png image. Afterwards, split train image and validation image to 9:1. 
* We have done this for you. Simply download data from [Google drive](https://drive.google.com/file/d/1VhoRWMAb8l1p7EKqA8a1KmVWevdCDcI5/view?usp=sharing), and put data in data/coco directory.


* Go to directory: `cd data/coco`
* `python transfer_normal_image.py` to split the proportion of normal image. Modify `scale` variable to change the proportion.
* `python generate_coco.py` to generate coco format of train/val directory. It will produce train_coco.json and val_coco.json.
* After generate train_coco.json and val_coco.json, please put in annotation/ directory.

```
# in data/coco directory
mkdir annotations
cp train_coco.json ./annotations
cp val_coco.json ./annotations
cp test_coco.json ./annotations
```
## Select Config file
* swin transformer: "configs/swinT/swinT.py"

## Training
* train model with [pretrained model](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)
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

## Ensemble
These are the csv answers we ensemble. [Google drive](https://drive.google.com/drive/folders/1GSD8JdPbntLMF76tnEeN7kv83_SEVVDl?usp=sharing)


```
python ensemble.py

# Ensembling methods for object detection.
    # USAGE: In ensemble.py, edit variable "a", "b" as the csv file you want to ensemble
    # If you want to change size of width, height, modify convert_ratio. 
    # There should be './stage_2_sample_submission.csv' in your working directory.
    # Output will be './ensemble_answer.csv'
```


## Team Report
https://drive.google.com/file/d/1uDGuslwrFidzXOXYAF-0-RNTavEEcEAH/view?usp=sharing
