import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
import pdb
import numpy as np


class resnetBranch0(nn.Module):
    def __init__(self):
        super(resnetBranch0, self).__init__()
        # branch1
        self.branch0 = nn.Sequential(
                nn.Linear(64 * 56 * 56, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )

        self.classifier0 = nn.Linear(512, 2)

    def forward(self, x):
        # branch1
        brh0 = x.view(x.size(0), -1)
        brh0 = self.branch0(brh0)
        bcls0 = self.classifier0(brh0)

        return bcls0


class resnetBranch0pool1(nn.Module):
    def __init__(self):
        super(resnetBranch0pool1, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        # branch1
        self.branch0 = nn.Sequential(
                nn.Linear(64 * 28 * 28, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )

        self.classifier0 = nn.Linear(512, 2)

    def forward(self, x):
        # branch1
        x = self.pool(x)
        brh0 = x.view(x.size(0), -1)
        brh0 = self.branch0(brh0)
        bcls0 = self.classifier0(brh0)

        return bcls0


class resnetBranch0pool2(nn.Module):
    def __init__(self):
        super(resnetBranch0pool2, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        # branch1
        self.branch0 = nn.Sequential(
                nn.Linear(64 * 14 * 14, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )

        self.classifier0 = nn.Linear(512, 2)

    def forward(self, x):
        # branch1
        x = self.pool(x)
        x = self.pool(x)
        brh0 = x.view(x.size(0), -1)
        brh0 = self.branch0(brh0)
        bcls0 = self.classifier0(brh0)

        return bcls0

class resnetBranch1(nn.Module):
    def __init__(self):
        super(resnetBranch1, self).__init__()
        # branch1
        self.branch1 = nn.Sequential(
                nn.Linear(64 * 56 * 56, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )

        self.classifier1 = nn.Linear(512, 2)

    def forward(self, x):
        brh1 = x.view(x.size(0), -1)
        brh1 = self.branch1(brh1)
        bcls1 = self.classifier1(brh1)

        return bcls1


class resnetBranch1pool1(nn.Module):
    def __init__(self):
        super(resnetBranch1pool1, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        # branch1
        self.branch1 = nn.Sequential(
                nn.Linear(64 * 28 * 28, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )

        self.classifier1 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(x)
        brh1 = x.view(x.size(0), -1)
        brh1 = self.branch1(brh1)
        bcls1 = self.classifier1(brh1)

        return bcls1


class resnetBranch1pool2(nn.Module):
    def __init__(self):
        super(resnetBranch1pool2, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        # branch1
        self.branch1 = nn.Sequential(
                nn.Linear(64 * 14 * 14, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )

        self.classifier1 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(x)
        x = self.pool(x)
        brh1 = x.view(x.size(0), -1)
        brh1 = self.branch1(brh1)
        bcls1 = self.classifier1(brh1)

        return bcls1


class resnetBranch2(nn.Module):
    def __init__(self):
        super(resnetBranch2, self).__init__()
        # branch2
        self.branch2 = nn.Sequential(
                nn.Linear(128 * 28 * 28, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )

        self.classifier2 = nn.Linear(512, 2)

    def forward(self, x):
        brh2 = x.view(x.size(0), -1)
        brh2 = self.branch2(brh2)
        bcls2 = self.classifier2(brh2)

        return bcls2


class resnetBranch2pool1(nn.Module):
    def __init__(self):
        super(resnetBranch2pool1, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        # branch2
        self.branch2 = nn.Sequential(
                nn.Linear(128 * 14 * 14, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )

        self.classifier2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(x)
        brh2 = x.view(x.size(0), -1)
        brh2 = self.branch2(brh2)
        bcls2 = self.classifier2(brh2)

        return bcls2


class resnetBranch2pool2(nn.Module):
    def __init__(self):
        super(resnetBranch2pool2, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        # branch2
        self.branch2 = nn.Sequential(
                nn.Linear(128 * 7 * 7, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )

        self.classifier2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(x)
        x = self.pool(x)
        brh2 = x.view(x.size(0), -1)
        brh2 = self.branch2(brh2)
        bcls2 = self.classifier2(brh2)

        return bcls2


class resnetBranch3(nn.Module):
    def __init__(self):
        super(resnetBranch3, self).__init__()
        # branch3
        self.branch3 = nn.Sequential(
                nn.Linear(256 * 14 * 14, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
               )

        self.classifier3 = nn.Linear(512, 2)

    def forward(self, x):
        brh3 = x.view(x.size(0), -1)
        brh3 = self.branch3(brh3)
        bcls3 = self.classifier3(brh3)

        return bcls3


class resnetBranch3pool1(nn.Module):
    def __init__(self):
        super(resnetBranch3pool1, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        # branch3
        self.branch3 = nn.Sequential(
                nn.Linear(256 * 7 * 7, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
               )

        self.classifier3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(x)
        brh3 = x.view(x.size(0), -1)
        brh3 = self.branch3(brh3)
        bcls3 = self.classifier3(brh3)

        return bcls3


class resnetBranch3pool2(nn.Module):
    def __init__(self):
        super(resnetBranch3pool2, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        # branch3
        self.branch3 = nn.Sequential(
                nn.Linear(256 * 3 * 3, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
               )

        self.classifier3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(x)
        x = self.pool(x)
        brh3 = x.view(x.size(0), -1)
        brh3 = self.branch3(brh3)
        bcls3 = self.classifier3(brh3)

        return bcls3

class resnetBranch4(nn.Module):
    def __init__(self):
        super(resnetBranch4, self).__init__()
        # branch4
        self.branch4 = nn.Sequential(
                nn.Linear(512 * 7 * 7, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )

        self.classifier4 = nn.Linear(512, 2)

    def forward(self, x):
        brh4 = x.view(x.size(0), -1)
        brh4 = self.branch4(brh4)
        bcls4 = self.classifier4(brh4)

        return bcls4

class resnetBranch4pool1(nn.Module):
    def __init__(self):
        super(resnetBranch4pool1, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        # branch4
        self.branch4 = nn.Sequential(
                nn.Linear(512 * 3 * 3, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )

        self.classifier4 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(x)
        brh4 = x.view(x.size(0), -1)
        brh4 = self.branch4(brh4)
        bcls4 = self.classifier4(brh4)

        return bcls4


class resnetBranch4pool2(nn.Module):
    def __init__(self):
        super(resnetBranch4pool2, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        # branch4
        self.branch4 = nn.Sequential(
                nn.Linear(512 * 1 * 1, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )

        self.classifier4 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(x)
        x = self.pool(x)
        brh4 = x.view(x.size(0), -1)
        brh4 = self.branch4(brh4)
        bcls4 = self.classifier4(brh4)

        return bcls4

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


class resnetBranch0fc(nn.Module):
    def __init__(self):
        super(resnetBranch0fc, self).__init__()
        # branch1
        self.branch0 = nn.Sequential(
                nn.Linear(64 * 56 * 56, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )


    def forward(self, x):
        # branch1
        brh0 = x.view(x.size(0), -1)
        brh0 = self.branch0(brh0)
        return brh0

class resnetBranch0pool1fc(nn.Module):
    def __init__(self):
        super(resnetBranch0pool1fc, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        # branch1
        self.branch0 = nn.Sequential(
                nn.Linear(64 * 28 * 28, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )


    def forward(self, x):
        # branch1
        x = self.pool(x)
        brh0 = x.view(x.size(0), -1)
        brh0 = self.branch0(brh0)
        return brh0


class resnetBranch0pool2fc(nn.Module):
    def __init__(self):
        super(resnetBranch0pool2fc, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        # branch1
        self.branch0 = nn.Sequential(
                nn.Linear(64 * 14 * 14, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )


    def forward(self, x):
        # branch1
        x = self.pool(x)
        x = self.pool(x)
        brh0 = x.view(x.size(0), -1)
        brh0 = self.branch0(brh0)
        return brh0

class resnetBranch1fc(nn.Module):
    def __init__(self):
        super(resnetBranch1fc, self).__init__()
        # branch1
        self.branch1 = nn.Sequential(
                nn.Linear(64 * 56 * 56, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )


    def forward(self, x):
        brh1 = x.view(x.size(0), -1)
        brh1 = self.branch1(brh1)
        return brh1

class resnetBranch1pool1fc(nn.Module):
    def __init__(self):
        super(resnetBranch1pool1fc, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        # branch1
        self.branch1 = nn.Sequential(
                nn.Linear(64 * 28 * 28, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )


    def forward(self, x):
        x = self.pool(x)
        brh1 = x.view(x.size(0), -1)
        brh1 = self.branch1(brh1)
        return brh1

class resnetBranch1pool2fc(nn.Module):
    def __init__(self):
        super(resnetBranch1pool2fc, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        # branch1
        self.branch1 = nn.Sequential(
                nn.Linear(64 * 14 * 14, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )


    def forward(self, x):
        x = self.pool(x)
        x = self.pool(x)
        brh1 = x.view(x.size(0), -1)
        brh1 = self.branch1(brh1)
        return brh1

class resnetBranch2fc(nn.Module):
    def __init__(self):
        super(resnetBranch2fc, self).__init__()
        # branch2
        self.branch2 = nn.Sequential(
                nn.Linear(128 * 28 * 28, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )


    def forward(self, x):
        brh2 = x.view(x.size(0), -1)
        brh2 = self.branch2(brh2)
        return brh2

class resnetBranch2pool1fc(nn.Module):
    def __init__(self):
        super(resnetBranch2pool1fc, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        # branch2
        self.branch2 = nn.Sequential(
                nn.Linear(128 * 14 * 14, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )


    def forward(self, x):
        x = self.pool(x)
        brh2 = x.view(x.size(0), -1)
        brh2 = self.branch2(brh2)
        return brh2

class resnetBranch2pool2fc(nn.Module):
    def __init__(self):
        super(resnetBranch2pool2fc, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        # branch2
        self.branch2 = nn.Sequential(
                nn.Linear(128 * 7 * 7, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )


    def forward(self, x):
        x = self.pool(x)
        x = self.pool(x)
        brh2 = x.view(x.size(0), -1)
        brh2 = self.branch2(brh2)
        return brh2

class resnetBranch3fc(nn.Module):
    def __init__(self):
        super(resnetBranch3fc, self).__init__()
        # branch3
        self.branch3 = nn.Sequential(
                nn.Linear(256 * 14 * 14, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
               )

    def forward(self, x):
        brh3 = x.view(x.size(0), -1)
        brh3 = self.branch3(brh3)
        return brh3

class resnetBranch3pool1fc(nn.Module):
    def __init__(self):
        super(resnetBranch3pool1fc, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        # branch3
        self.branch3 = nn.Sequential(
                nn.Linear(256 * 7 * 7, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
               )

    def forward(self, x):
        x = self.pool(x)
        brh3 = x.view(x.size(0), -1)
        brh3 = self.branch3(brh3)
        return brh3

class resnetBranch3pool2fc(nn.Module):
    def __init__(self):
        super(resnetBranch3pool2fc, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        # branch3
        self.branch3 = nn.Sequential(
                nn.Linear(256 * 3 * 3, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
               )

    def forward(self, x):
        x = self.pool(x)
        x = self.pool(x)
        brh3 = x.view(x.size(0), -1)
        brh3 = self.branch3(brh3)
        return brh3

class resnetBranch4fc(nn.Module):
    def __init__(self):
        super(resnetBranch4fc, self).__init__()
        # branch4
        self.branch4 = nn.Sequential(
                nn.Linear(512 * 7 * 7, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )


    def forward(self, x):
        brh4 = x.view(x.size(0), -1)
        brh4 = self.branch4(brh4)
        return brh4


class resnetBranch4pool1fc(nn.Module):
    def __init__(self):
        super(resnetBranch4pool1fc, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        # branch4
        self.branch4 = nn.Sequential(
                nn.Linear(512 * 3 * 3, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )


    def forward(self, x):
        x = self.pool(x)
        brh4 = x.view(x.size(0), -1)
        brh4 = self.branch4(brh4)
        return brh4

class resnetBranch4pool2fc(nn.Module):
    def __init__(self):
        super(resnetBranch4pool2fc, self).__init__()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        # branch4
        self.branch4 = nn.Sequential(
                nn.Linear(512 * 1 * 1, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )


    def forward(self, x):
        x = self.pool(x)
        x = self.pool(x)
        brh4 = x.view(x.size(0), -1)
        brh4 = self.branch4(brh4)
        return brh4

def load_branch_fcs(index=None):
    model = None
    if index==0:
        model = resnetBranch0fc()
    if index==1:
        model = resnetBranch1fc()
    if index==2:
        model = resnetBranch2fc()
    if index==3:
        model = resnetBranch3fc()
    if index==4:
        model = resnetBranch4fc()
    if index==5:
        model = resnetBranch0pool1fc()
    if index==6:
        model = resnetBranch1pool1fc()
    if index==7:
        model = resnetBranch2pool1fc()
    if index==8:
        model = resnetBranch3pool1fc()
    if index==9:
        model = resnetBranch4pool1fc()
    if index==10:
        model = resnetBranch0pool2fc()
    if index==11:
        model = resnetBranch1pool2fc()
    if index==12:
        model = resnetBranch2pool2fc()
    if index==13:
        model = resnetBranch3pool2fc()
    if index==14:
        model = resnetBranch4pool2fc()

    return model
