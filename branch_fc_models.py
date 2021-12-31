import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
import pdb
import numpy as np


class resnetBranch0(nn.Module):
    def __init__(self):
        super(resnetBranch0, self).__init__()
        self.branch = nn.Sequential(
                nn.Linear(64 * 56 * 56, 512),
                nn.ReLU(inplace=True)
               )

    def forward(self, x):
        brh0 = x.view(x.size(0), -1)
        brh0 = self.branch(brh0)
        return brh0


class resnetBranch0pool1(nn.Module):
    def __init__(self):
        super(resnetBranch0pool1, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.branch = nn.Sequential(
                nn.Linear(64 * 28 * 28, 512),
                nn.ReLU(inplace=True)
               )


    def forward(self, x):
        x = self.pool(x)
        brh0 = x.view(x.size(0), -1)
        brh0 = self.branch(brh0)
        return brh0


class resnetBranch0pool2(nn.Module):
    def __init__(self):
        super(resnetBranch0pool2, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.branch = nn.Sequential(
                nn.Linear(64 * 14 * 14, 512),
                nn.ReLU(inplace=True)
               )


    def forward(self, x):
        x = self.pool(x)
        x = self.pool(x)
        brh0 = x.view(x.size(0), -1)
        brh0 = self.branch(brh0)

        return brh0

class resnetBranch1(nn.Module):
    def __init__(self):
        super(resnetBranch1, self).__init__()
        self.branch = nn.Sequential(
                nn.Linear(64 * 56 * 56, 512),
                nn.ReLU(inplace=True)
               )

    def forward(self, x):
        brh1 = x.view(x.size(0), -1)
        brh1 = self.branch(brh1)
        return brh1


class resnetBranch1pool1(nn.Module):
    def __init__(self):
        super(resnetBranch1pool1, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.branch = nn.Sequential(
                nn.Linear(64 * 28 * 28, 512),
                nn.ReLU(inplace=True)
               )

    def forward(self, x):
        x = self.pool(x)
        brh1 = x.view(x.size(0), -1)
        brh1 = self.branch(brh1)
        return brh1


class resnetBranch1pool2(nn.Module):
    def __init__(self):
        super(resnetBranch1pool2, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.branch = nn.Sequential(
                nn.Linear(64 * 14 * 14, 512),
                nn.ReLU(inplace=True)
               )

    def forward(self, x):
        x = self.pool(x)
        x = self.pool(x)
        brh1 = x.view(x.size(0), -1)
        brh1 = self.branch(brh1)
        return brh1

class resnetBranch2(nn.Module):
    def __init__(self):
        super(resnetBranch2, self).__init__()
        self.branch = nn.Sequential(
                nn.Linear(128 * 28 * 28, 512),
                nn.ReLU(inplace=True)
               )

    def forward(self, x):
        brh2 = x.view(x.size(0), -1)
        brh2 = self.branch(brh2)
        return brh2


class resnetBranch2pool1(nn.Module):
    def __init__(self):
        super(resnetBranch2pool1, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.branch = nn.Sequential(
                nn.Linear(128 * 14 * 14, 512),
                nn.ReLU(inplace=True)
               )

    def forward(self, x):
        x = self.pool(x)
        brh2 = x.view(x.size(0), -1)
        brh2 = self.branch(brh2)
        return brh2


class resnetBranch2pool2(nn.Module):
    def __init__(self):
        super(resnetBranch2pool2, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.branch = nn.Sequential(
                nn.Linear(128 * 7 * 7, 512),
                
               )

    def forward(self, x):
        x = self.pool(x)
        x = self.pool(x)
        brh2 = x.view(x.size(0), -1)
        brh2 = self.branch(brh2)
        return brh2


class resnetBranch3(nn.Module):
    def __init__(self):
        super(resnetBranch3, self).__init__()
        self.branch = nn.Sequential(
                nn.Linear(256 * 14 * 14, 512),
                
               )

    def forward(self, x):
        brh3 = x.view(x.size(0), -1)
        brh3 = self.branch(brh3)
        return brh3


class resnetBranch3pool1(nn.Module):
    def __init__(self):
        super(resnetBranch3pool1, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.branch = nn.Sequential(
                nn.Linear(256 * 7 * 7, 512),
                nn.ReLU(inplace=True)
               )

    def forward(self, x):
        x = self.pool(x)
        brh3 = x.view(x.size(0), -1)
        brh3 = self.branch(brh3)
        return brh3


class resnetBranch3pool2(nn.Module):
    def __init__(self):
        super(resnetBranch3pool2, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.branch = nn.Sequential(
                nn.Linear(256 * 3 * 3, 512),
                nn.ReLU(inplace=True)
               )

    def forward(self, x):
        x = self.pool(x)
        x = self.pool(x)
        brh3 = x.view(x.size(0), -1)
        brh3 = self.branch(brh3)
        return brh3

class resnetBranch4(nn.Module):
    def __init__(self):
        super(resnetBranch4, self).__init__()
        self.branch = nn.Sequential(
                nn.Linear(512 * 7 * 7, 512),
                nn.ReLU(inplace=True)
               )

    def forward(self, x):
        brh4 = x.view(x.size(0), -1)
        brh4 = self.branch(brh4)
        return brh4

class resnetBranch4pool1(nn.Module):
    def __init__(self):
        super(resnetBranch4pool1, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.branch = nn.Sequential(
                nn.Linear(512 * 3 * 3, 512),
                nn.ReLU(inplace=True)
               )

    def forward(self, x):
        x = self.pool(x)
        brh4 = x.view(x.size(0), -1)
        brh4 = self.branch(brh4)
        return brh4


class resnetBranch4pool2(nn.Module):
    def __init__(self):
        super(resnetBranch4pool2, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.branch = nn.Sequential(
                nn.Linear(512 * 1 * 1, 512),
                nn.ReLU(inplace=True)
               )

    def forward(self, x):
        x = self.pool(x)
        x = self.pool(x)
        brh4 = x.view(x.size(0), -1)
        brh4 = self.branch(brh4)
        return brh4

def load_branches(index=None):
    model = None
    if index==0:
        model = resnetBranch0()
    if index==1:
        model = resnetBranch1()
    if index==2:
        model = resnetBranch2()
    if index==3:
        model = resnetBranch3()
    if index==4:
        model = resnetBranch4()
    if index==5:
        model = resnetBranch0pool1()
    if index==6:
        model = resnetBranch1pool1()
    if index==7:
        model = resnetBranch2pool1()
    if index==8:
        model = resnetBranch3pool1()
    if index==9:
        model = resnetBranch4pool1()
    if index==10:
        model = resnetBranch0pool2()
    if index==11:
        model = resnetBranch1pool2()
    if index==12:
        model = resnetBranch2pool2()
    if index==13:
        model = resnetBranch3pool2()
    if index==14:
        model = resnetBranch4pool2()
    return model
