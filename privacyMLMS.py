from dataloader import *
from mymodels import *
from train import *
import torch
from dataset import *
import pandas as pd

path = ["train15", "valid15", "test15"]
trainloader, validloader, testloader = load_general_feature(path = path)

torch.manual_seed(100)
model = privacy_MLMS()
f1 = train(model, trainloader, validloader, testloader, describe='branch{}'.format(15))
print(f1)
