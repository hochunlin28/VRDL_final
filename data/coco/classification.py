import numpy as np
import pandas as pd
import os
import matplotlib.image as mpimg

import torch
import torch.nn as nn
import torch.optim as optim 

import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as utils
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
from efficientnet_pytorch import EfficientNet
from PIL import Image
from os import listdir
from torch.utils.data import Dataset, DataLoader

data_dir = './'
train_dir = data_dir + 'train/'
train_normal_dir = data_dir + 'train_normal/'
test_dir = data_dir + 'test/'
validate_dir = data_dir +'validate/'

# get each image id and label (0 for normal, 1 for unormal)
train_id = listdir(train_dir)
train_normal_id = listdir(train_normal_dir)
labels = []
for img in train_id:
    labels.append([img,1])
for img in train_normal_id:
    labels.append([img,0])
    
class ImageData(Dataset):
    def __init__(self, df, data_dir, transform):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):       
        img_name = self.df[index][0]
        number = self.df[index][1]
        #label = np.zeros(200)
        #label[number-1] = 1.0
        label = number
        if(label == 0):
            img_path = os.path.join(train_normal_dir, img_name)
        else:
            img_path = os.path.join(train_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        image = self.transform(image)

        return image, label


data_transf = transforms.Compose([transforms.Resize([512, 512]), transforms.ToTensor()])
train_data = ImageData(df = labels, data_dir = train_dir, transform = data_transf)
train_loader = DataLoader(dataset = train_data, batch_size = 16, shuffle = True)

#load model from pretrained
model = EfficientNet.from_pretrained('efficientnet-b1')
#model = torch.load("./save1.pt")
print("load_success")

# Unfreeze model weights
for param in model.parameters():
    param.requires_grad = True

#change the last fully connected layer
feature = model._fc.in_features
model._fc = nn.Linear(in_features=feature,out_features=2,bias=True)

#use GPU to train
model = model.to('cuda')

#set optimize function and loss function
optimizer = optim.Adam(model.parameters(),lr = 0.02)
#optimizer = optim.SGD(model.parameters(),lr = 0.005, momentum = 0.9)
loss_func = nn.CrossEntropyLoss()

# Train model
loss_log = []


for epoch in range(50):    
    model.train()
    train_correct = 0    
    for ii, (data, target) in enumerate(train_loader):
        #target = np.array(target).astype(float)
        #target = torch.from_numpy(target)
        #target = target.float()                

        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        #m = nn.Sigmoid()
        #n = nn.Softmax(dim = 1)
        #output = n(output)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        train_correct += (output.max(1)[1] == target).sum()
        optimizer.step()  
        print("batch: %s" %ii)
        if ii % 1000 == 0:
            loss_log.append(loss.item())
            
    torch.save(model,"epoch" + str(epoch) + ".pt")   
    print('Epoch: {} - Loss: {:.6f} - Correct: {}'.format(epoch + 1, loss.item(), train_correct))



