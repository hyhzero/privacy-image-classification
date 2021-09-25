from mymodels import *
from torchvision import models
import torch.nn as nn
import collections
import torch
import os



def cnn_lstm_attention(index=None):
    model = None

    if isinstance(index, int):
        model = cnn_branch(index)
    if index=='cnn_concat':
        model = cnn_concat()
    if index=='cnn_concat_scale':
        model = cnn_concat_scale()
    if index=='bi_lstm':
        model = bi_lstm()

    return model

def load_branches(index=None):
    resnet = models.resnet18(pretrained=True)
    model = None
    if index==0:
        model = resnetBranch0(resnet)
    if index==1:
        model = resnetBranch1(resnet)
    if index==2:
        model = resnetBranch2(resnet)
    if index==3:
        model = resnetBranch3(resnet)
    if index==4:
        model = resnetBranch4(resnet)
    if index==5:
        model = resnetBranch5(resnet)

    return model
