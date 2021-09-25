from torch.utils import data
import torchvision.transforms as T
import pandas as pd
import numpy as np
import shutil
import os
from PIL import Image
import pickle
import pdb



class PicAlert(data.Dataset):

    def __init__(self, root='picAlert', train=True, fold='train', transforms=None):
        self.dataset = pd.read_csv('train_test_valid/{}.csv'.format(fold))
        self.imgs, self.labels = zip(*(self.dataset.values.tolist()))
        self.imgs = [os.path.join(root, img) for img in self.imgs]
        self.transforms = transforms

        if self.transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            #normalize = T.Normalize(mean=[0.485], std=[0.229])

            if not train:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                    ])
            else:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                    ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 1 if self.labels[index]=='private' else 0
        try:
            data = Image.open(img_path)
        except Exception as e:
            print(e)
        else:
            if len(data.split())>3:
                data = data.convert('RGB')
            #print(len(data.split()))

            if self.transforms:
                data = self.transforms(data)

            return data, label

        return None, None


    def __len__(self):
        return len(self.imgs)


class PicAlertF(data.Dataset):

    def __init__(self, root='dataset', fold='train', save=False):
        self.imgs = os.listdir('{}/{}'.format(root, fold))

        if save==True:
            self.labels = self.imgs
        else:
            self.labels = [int(fn.split('_')[-1]) for fn in self.imgs]

        self.imgs = ['{}/{}/{}'.format(root, fold, img) for img in self.imgs]


    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = self.labels[index]

        with open(img_path, 'rb') as f:
            data = pickle.load(f)

        return data, label

    def __len__(self):
        return len(self.imgs)

