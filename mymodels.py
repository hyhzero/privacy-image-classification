import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
import pdb
import numpy as np


class resnetBranch0Map(nn.Module):
    def __init__(self):
        super(resnetBranch0Map, self).__init__()
        model = models.resnet18(pretrained=True)
        for layer in model.parameters():
            layer.requires_grad = False

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class resnetBranch1Map(nn.Module):
    def __init__(self):
        super(resnetBranch1Map, self).__init__()
        model = models.resnet18(pretrained=True)
        for layer in model.parameters():
            layer.requires_grad = False

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        return x


class resnetBranch2Map(nn.Module):
    def __init__(self):
        super(resnetBranch2Map, self).__init__()
        model = models.resnet18(pretrained=True)
        for layer in model.parameters():
            layer.requires_grad = False

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class resnetBranch3Map(nn.Module):
    def __init__(self):
        super(resnetBranch3Map, self).__init__()
        model = models.resnet18(pretrained=True)
        for layer in model.parameters():
            layer.requires_grad = False

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool


        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x



class resnetBranch4Map(nn.Module):
    def __init__(self):
        super(resnetBranch4Map, self).__init__()
        model = models.resnet18(pretrained=True)
        for layer in model.parameters():
            layer.requires_grad = False

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class resnetBranch5Map(nn.Module):
    def __init__(self):
        super(resnetBranch5Map, self).__init__()
        model = models.resnet18(pretrained=True)
        for layer in model.parameters():
            layer.requires_grad = False

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x

