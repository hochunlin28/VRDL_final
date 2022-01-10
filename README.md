# VRDL_Final

## Competetion
https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

## Introduction
The goal of this competition is to reduce the clinicians’ stress of reading high volumes of
images. And our objective is to detect lung opacities on chest radiographs and conduct furt
her experiments to detect Pneumonia by our model.
* Dataset:
* There are three classes: Normal, Lung Opacity, or No Lung Opacity but Not Normal.
* 26,684 training images, 3000 testing images
![](https://i.imgur.com/Nf9Ablr.png)

* Process chart of our approach
![](https://i.imgur.com/gPt2611.png)

* Result
Our private score is 0.15814, and our public score is 0.05729.
![](https://i.imgur.com/qlN1Flj.png)


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
* We have done this for you. Simply download data from [Google drive](https://drive.google.com/file/d/1uZDPDqx_8SV7O-pWyP_ClygY2N9CBrNj/view?usp=sharing), and put data in data/coco directory.


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
* After we train, it will produce epoch5.pth in work_dirs/swin-t_0:1, please put epoch5.pth in work_dir/swin-t_0:1. [download pretrainted model1](https://drive.google.com/file/d/1ARuXW_dw24XpTkbl5H-bxZmMBTBBgTh1/view?usp=sharing)
* (Optional) If you want to ensemble models, [download pretrainted model2](https://drive.google.com/file/d/1QmftZPFuDphWB5y1Mk7SKJjKdsuuXjEq/view?usp=sharing), and modify the model path in the following command.


## Inference
* It will generate result.box.json and result.segm.json file and we can analyze with these files
```
python tools/test.py configs/swinT/swinT.py ./work_dirs/swin-t_0:1/epoch_5.pth --format-only --options "jsonfile_prefix=./results"

```

## Generate csv file
* It will generate the csv file which will uploaded to kaggle. Please Modify the `checkpoint` variable to select different model weight. After execution, we will get answer_1.csv file.
```
python generate_csv.py
```

## Ensemble (optional)
* These are the csv answers we ensemble (produced by the above weight). [Google drive](https://drive.google.com/drive/folders/1GSD8JdPbntLMF76tnEeN7kv83_SEVVDl?usp=sharing)

* Run following command to ensemble.
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

## Reference
https://github.com/open-mmlab/mmdetection
