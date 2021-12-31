from torch.utils import data
import torchvision.transforms as T
import pandas as pd
import numpy as np
import shutil
import os
from PIL import Image
import torch
import pickle
import pdb

class PicAlertFeature(data.Dataset):
    '''
    root 特征所在目录
    annotation 标注数据文件路径
    '''
    def __init__(self, roots=[], annotation=None):
        # 读取标注数据，格式 文件名, 类别
        self.dataset = pd.read_csv(annotation)

        self.imgs, self.labels = zip(*(self.dataset.values.tolist()))
        self.paths = []
        for root in roots:
            self.paths.append(['{}/{}.npy'.format(root, img) for img in self.imgs])

    # 读取特征
    def __getitem__(self, index):
        label = self.labels[index]
        all_data = []
        for i in range(len(self.paths)):
            feature_path = self.paths[i][index]
            data = np.load(feature_path)
            data.astype(np.float32)
            all_data.append(data)
        return all_data, label

    def __len__(self):
        return len(self.imgs)

    
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
import pdb
import numpy as np
from branch_fc_models import *



class MldrNet(nn.Module):
    def __init__(self):
        super(MldrNet, self).__init__()
        self.branches = nn.ModuleList([load_branches(i) for i in range(5)])
        self.fc = nn.Linear(in_features=2560, out_features=2)

    def forward(self, x):
        brhs = []
        for i in range(5):
            brh = self.branches[i](x[i])
            brh = brh.unsqueeze(1)
            brhs.append(brh)
        output = torch.cat(brhs, dim=1)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


import torch.utils.data as Data
from train import train_msml

roots = [
    "./feature/branch0map",
    "./feature/branch1map",
    "./feature/branch2map",
    "./feature/branch3map",
    "./feature/branch4map"
]

# 加载数据
trainset = PicAlertFeature(roots=roots, annotation='./dataset/train.csv')
validset = PicAlertFeature(roots=roots, annotation='./dataset/valid.csv')
testset = PicAlertFeature(roots=roots, annotation='./dataset/test.csv')
trainloader = Data.DataLoader(trainset, batch_size=32, shuffle=True)
validloader = Data.DataLoader(validset, batch_size=32, shuffle=True)
testloader = Data.DataLoader(testset, batch_size=32, shuffle=True)


# MldrNet
model = MldrNet()

# 训练模型
f1 = train_msml(model, trainloader, validloader, testloader=testloader, lr=0.00003, epoch=10, describe='MldrNet')