from modelloader import *
from dataloader import load_train_test_valid
from train import *

# load data
root = "./picAlert"
trainloader, validloader, testloader = load_train_test_valid(root=root)

# extract feature maps
for i in range(6):
    model = load_branches(index=i)
    extract_feature(model, trainloader, dst='train_test_valid/train_branch{}'.format(i))
    extract_feature(model, validloader, dst='train_test_valid/valid_branch{}'.format(i))
    extract_feature(model, testloader, dst='train_test_valid/test_branch{}'.format(i))
