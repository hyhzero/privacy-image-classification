from modelloader import *
from dataloader import *
from train import *
import torch
from dataset import *
import pandas as pd

path = ["train0", "valid0", "test0"]
trainloader, validloader, testloader = load_general_feature(path = path)

torch.manual_seed(100)
model = cnn_lstm_attention(index='cnn_concat')
f1 = train(model, trainloader, validloader, testloader, describe='cnn_concat')
print(f1)
