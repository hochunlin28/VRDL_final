import glob, pandas as pd
import numpy as np
import cv2
import os
from os import listdir
import json
import shutil
from tqdm import tqdm

# normal:unormal = scale:1
scale = 0.3

df = pd.read_csv('stage_2_train_labels.csv')

    
def split_train_image():
    total_unormal_count = 0
    normal_count = 0
    # classify normal/unormal image to train/train_normal directory
    for n, row in df.iterrows():
        # unormal count
        if os.path.exists('./train/' + row['patientId'] + '.png') and row['Target'] == 0:
            shutil.move('./train/' + row['patientId'] + '.png', './train_normal')
    
    train_dir = listdir('./train')
    total_unormal_count = len(train_dir)
    
    #move normal image to train directory
    normal_count = total_unormal_count * scale
    print(normal_count)
    train_normal_dir = listdir('./train_normal')
    for i in range(int(normal_count)):
        shutil.move('./train_normal/' + train_normal_dir[i] , './train/')

def split_val_image():
    total_unormal_count = 0
    normal_count = 0
    # classify normal/unormal image to train/train_normal directory
    for n, row in df.iterrows():
        # unormal count
        if os.path.exists('./val/' + row['patientId'] + '.png') and row['Target'] == 0:
            shutil.move('./val/' + row['patientId'] + '.png', './val_normal')
    
    val_dir = listdir('./val')
    total_unormal_count = len(val_dir)
    
    #move normal image to train directory
    normal_count = total_unormal_count * scale
    print(normal_count)
    train_normal_dir = listdir('./val_normal')
    for i in range(int(normal_count)):
        shutil.move('./val_normal/' + train_normal_dir[i] , './val/')
        
def transfer_all_normal_train():
    train_dir = listdir('./train_normal')
    for img in train_dir:
        shutil.move('./train_normal/' + img , './train/')

def transfer_all_normal_val():
    val_dir = listdir('./val_normal')
    for img in val_dir:
        shutil.move('./val_normal/' + img , './val/')


transfer_all_normal_train() 
transfer_all_normal_val()        
split_train_image()
split_val_image()
