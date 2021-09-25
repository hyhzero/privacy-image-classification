from branch_models import *
from dataloader import *
from train import *
from sklearn.metrics import confusion_matrix, classification_report
import time
import torch
import json
import os
import pickle
import collections
import pdb


root = "./train_test_valid"
for i in range(0, 15):
    path = ["train_branch"+str(i%5), "valid_branch"+str(i%5), "test_branch"+str(i%5)]
    trainloader, validloader, testloader = load_general_feature(root=root, path = path, save=True)
    model = load_branch_fcs(index=i)
    extract_feature(model, trainloader, dst='train_test_valid/feature_train_branch{}'.format(i))
    extract_feature(model, validloader, dst='train_test_valid/feature_valid_branch{}'.format(i))
    extract_feature(model, testloader, dst='train_test_valid/feature_test_branch{}'.format(i))
