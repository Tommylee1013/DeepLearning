#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models, transforms

import matplotlib.pyplot as plt
import random
import time
import os

import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

!unzip /content/gdrive/MyDrive/Project_dataset/Font_npy_100_val.zip
valid_data = MyDataset("/content/Font_npy_100_val")

valid_loader = torch.utils.data.DataLoader(dataset=valid_data,
                                           batch_size=batch_size,
                                           shuffle=True)

def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

fix_seed(42)
np.random.seed(42)

##CNN Model##

input_size = 100
num_classes = 52
learning_rate = 0.001

class CNN(nn.Module) :
    def __init__(self, input_size, num_classes) :
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 1,# 16 * 50*50
                      out_channels = 16,
                      kernel_size = 5,
                      stride = 2,
                      padding = 2),
            nn.BatchNorm2d(num_features = 16),
            nn.SiLU()
        )
        
        self.layer2 = nn.Sequential( # 32 * 48*48
            nn.Conv2d(in_channels = 16,
                      out_channels = 32,
                      kernel_size = 5,
                      stride = 1, 
                      padding = 1),
            nn.BatchNorm2d(num_features = 32),
            nn.SiLU()
        )
        
        self.layer3 = nn.Sequential( # 64*24*24
            nn.Conv2d(in_channels = 32,
                      out_channels = 64,
                      kernel_size = 5,
                      stride = 2, 
                      padding = 2),
            nn.BatchNorm2d(num_features = 64),
            nn.SiLU()
        )

        self.layer4 = nn.Sequential( # 128*24*24
            nn.Conv2d(in_channels = 64,
                      out_channels = 128,
                      kernel_size = 5,
                      stride = 1, 
                      padding = 2),
            nn.BatchNorm2d(num_features = 128),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size = 2) # 128*12*12
        )

        self.layer5 = nn.Sequential( # 256*12*12
            nn.Conv2d(in_channels = 128,
                      out_channels = 256,
                      kernel_size = 3,
                      stride = 1, 
                      padding = 1),
            nn.BatchNorm2d(num_features = 256),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size = 2) # 256*6*6
        )

        self.layer6 = nn.Sequential( # 256*6*6
            nn.Conv2d(in_channels = 256,
                      out_channels = 256,
                      kernel_size = 3,
                      stride = 1, 
                      padding = 1),
            nn.BatchNorm2d(num_features = 256),
            nn.SiLU()
        )
        self.layer7 = nn.Sequential( # 256*6*6
            nn.Conv2d(in_channels = 256,
                      out_channels = 256,
                      kernel_size = 3,
                      stride = 1, 
                      padding = 1),
            nn.BatchNorm2d(num_features = 256),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size = 2) # 256*3*3
        )
        
        self.fc1 = nn.Linear(in_features = 256*3*3, out_features = 256)
        self.fc2 = nn.Linear(in_features = 256, out_features = 96)
        self.fc3 = nn.Linear(in_features = 96, out_features = num_classes) 
    
    def forward(self, x) :
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = F.silu(x)
        x = x.reshape(x.size(0), -1)
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = self.fc3(x)
        #x = F.softmax(x)
        return x

model = CNN(input_size, num_classes)

##Test Data##

test_model = CNN(input_size, num_classes).to(device)

test_model.load_state_dict(torch.load('20180594.pth'))
acc_list = []
test_model.eval()

# metrics
test_acc = 0
with torch.no_grad():
    for image, label in valid_loader:
        image, label = image.to(device), label.to(device)

        # forward pass
        out = test_model(image)

        # acc
        _, pred = torch.max(out, 1)
        test_acc += (pred==label).sum()
        
    print(f'Accuracy: {test_acc.cpu().numpy()/len(valid_data) * 100}%')

