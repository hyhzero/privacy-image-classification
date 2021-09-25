import dataset
import torch.utils.data as Data
from torch.utils.data.sampler import WeightedRandomSampler
import pandas as pd


def load_train_test_valid(root='./picAlert', batch_size = 64, weight=None):

	# load dataset
	train_dataset = dataset.PicAlert(root=root, fold='train', train=False) 
	valid_dataset = dataset.PicAlert(root=root, fold='valid', train=False)
	test_dataset = dataset.PicAlert(root=root, fold='test', train=False)

	# dataloader
	trainloader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	validloader = Data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
	testloader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	return trainloader, validloader, testloader


def load_feature(root='train_test_valid', batch_size=64):
	# load dataset
	train_dataset = dataset.PicAlertF(root=root, fold='train')
	valid_dataset = dataset.PicAlertF(root=root, fold='valid')
	test_dataset = dataset.PicAlertF(root=root, fold='test')

	trainloader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	validloader = Data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
	testloader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
	return trainloader, validloader, testloader


def load_general_feature(root='train_test_valid', path=['train_cat', 'valid_cat', 'test_cat'], batch_size=64, save=False):
	# load dataset
	train_dataset = dataset.PicAlertF(root=root, fold=path[0], save=save)
	valid_dataset = dataset.PicAlertF(root=root, fold=path[1], save=save)
	test_dataset = dataset.PicAlertF(root=root, fold=path[2], save=save)

	trainloader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	validloader = Data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
	testloader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
	return trainloader, validloader, testloader
