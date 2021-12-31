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

# 提取后的特征数据
class PicAlertFeature(data.Dataset):

    '''
    root 特征所在目录
    annotation 标注数据文件路径
    '''
    def __init__(self, root=None, annotation=None):
        # 读取标注数据，格式 文件名, 类别
        self.dataset = pd.read_csv(annotation)

        self.imgs, self.labels = zip(*(self.dataset.values.tolist()))
        self.imgs = ['{}/{}.npy'.format(root, img) for img in self.imgs]

    # 读取特征
    def __getitem__(self, index):
        feature_path = self.imgs[index]
        label = self.labels[index]
        # 加载保存的特征
        data = np.load(feature_path)
        data.astype(np.float32)
        return data, label


    def __len__(self):
        return len(self.imgs)


import torch.utils.data as Data
from object_train import train
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
import pdb
import numpy as np
from branch_fc_models import *

class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=2)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


root = "./feature/branch5map"
# 加载数据
trainset = PicAlertFeature(root=root, annotation='./dataset/train.csv')
validset = PicAlertFeature(root=root, annotation='./dataset/valid.csv')
testset = PicAlertFeature(root=root, annotation='./dataset/test.csv')
trainloader = Data.DataLoader(trainset, batch_size=32, shuffle=True)
validloader = Data.DataLoader(validset, batch_size=32, shuffle=True)
testloader = Data.DataLoader(testset, batch_size=32, shuffle=True)


# 加载模型
model = resnet()
# 训练模型
f1 = train(model, trainloader, validloader, testloader=testloader, lr=0.0003, epoch=10, describe='resnet')