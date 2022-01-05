import asyncio
from argparse import ArgumentParser
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import os
import json
import csv
from tqdm import tqdm
import torch
from PIL import Image
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as utils
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn

inputdir = 'data/coco/test/'
test_list = sorted(os.listdir(inputdir))
#print((test_list))

config = 'configs/swinT/swinT.py'
checkpoint = 'work_dirs/swin-t_0.3:1/epoch_5.pth'
device = 'cuda:0'

f = open('answer_1.csv', 'w')

# create the csv writer
writer = csv.writer(f)
# write a row to the csv file
header = ['patientID', 'PredictionString']
writer.writerow(header)

# load model
model = init_detector(config, checkpoint, device)
#model_cls = torch.load("epoch2.pt")

data = []
path = 'data/coco/test/'
for x in tqdm(range(len(test_list))):
    result = inference_detector(model, path+test_list[x])
    
    #classification
    '''
    img = Image.open(path+test_list[x]).convert('RGB')
    data_transf = transforms.Compose([transforms.Resize([512, 512]), transforms.ToTensor()])
    img = data_transf(img)
    img = img.unsqueeze(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img = img.to(device)
    predicted_class = model_cls(img)  # the predicted category
    n = nn.Softmax(dim = 1)
    predicted_class = n(predicted_class)
    print(predicted_class.max(1)[1].item())
    '''
    #print(result[0])
    #print(len(result[0][0]))
    ans=[]
    id=test_list[x].split('.')[0]
    ans.append(id)
    for i in range (1): 
        temp=[]
        if(len(result[i][0])!=0): #detected
            if len(result[i][0]) >=  2:
                cnt = 2
            else:
                cnt = len(result[i][0])     
            for j in range(cnt):
                if result[i][0][j][4] > 0.55:
                    temp.append(result[i][0][j][4])
                    temp.append(result[i][0][j][0])
                    temp.append(result[i][0][j][1])
                    temp.append(result[i][0][j][2]-result[i][0][j][0])
                    temp.append(result[i][0][j][3]-result[i][0][j][1])
            temp = ' '.join(str(i) for i in temp)
            ans.append(str(temp))

    writer.writerow(ans)

f.close()
