from modelloader import *
from dataloader import *
from train import *
import torch
from dataset import *
import pandas as pd

path = ["train_branch5", "valid_branch5", "test_branch5"]
root = "./train_test_valid/"
trainloader, validloader, testloader = load_general_feature(root=root, path = path)


torch.manual_seed(100)
model = cnn_lstm_attention(index='cnn_concat_scale')
f1 = train(model, trainloader, validloader, testloader, describe='cnn_concat')
print(f1)

