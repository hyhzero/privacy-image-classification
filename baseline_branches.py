from modelloader import *
from dataloader import *
from train import *
import torch
from dataset import *
import pandas as pd

path = ["train0", "valid0", "test0"]
trainloader, validloader, testloader = load_general_feature(path = path)

torch.manual_seed(100)
f1_scores = []
for index in [0, 1, 2, 3, 4]:
    model = cnn_lstm_attention(index=index)
    f1 = train(model, trainloader, validloader, testloader, describe='branch{}'.format(index))
    f1_scores.append(f1)
print(f1_scores)
