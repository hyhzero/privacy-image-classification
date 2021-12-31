from torch.utils import data
import torchvision.transforms as T
import pandas as pd
import numpy as np
import shutil
import os
from PIL import Image
import torch
import pickle
import pdb


# 图片数据
class PicAlert(data.Dataset):

    ''' 读取路径,设置transforms
    root 图片所在目录
    annotation 标注数据文件路径
    train 训练or测试，训练模式会进行数据增强
    '''
    def __init__(self, root=None, annotation=None, train=True, transforms=None, extract_feature=False):
        # 读取标注数据，格式 文件名, 类别
        self.dataset = pd.read_csv(annotation)
        self.extract_feature = extract_feature

        self.imgs, self.labels = zip(*(self.dataset.values.tolist()))
        self.imgs = ['{}/{}'.format(root, img) for img in self.imgs]

        self.transforms = transforms

        if self.transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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

    # 读取图片,执行transforms
    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = self.labels[index]
        
        # 提取特征
        if self.extract_feature:
            label = self.imgs[index].rsplit('/', 1)[-1]
        try:
            data = Image.open(img_path)
        except Exception as e:
            print(e)
        else:
            if len(data.split())!=3:
                data = data.convert('RGB')
            #print(len(data.split()))

            if self.transforms:
                data = self.transforms(data)

            return data, label

        return torch.zeros(3, 224, 224), 0


    def __len__(self):
        return len(self.imgs)


# 提取后的特征数据
class PicAlertFeature(data.Dataset):

    '''
    root 特征所在目录
    annotation 标注数据文件路径
    '''
    def __init__(self, root=None, annotation=None):
        # 读取标注数据，格式 文件名, 类别
        self.dataset = pd.read_csv(annotation)

        self.imgs, self.labels = zip(*(self.dataset.values.tolist()))
        self.imgs = ['{}/{}.npy'.format(root, img) for img in self.imgs]

    # 读取特征
    def __getitem__(self, index):
        feature_path = self.imgs[index]
        label = self.labels[index]

        # 加载保存的特征
        data = np.load(feature_path)
#         data = np.where(data > 0, 1, 0)
        data = np.where(data > data[data.argsort()[::-1][40]], 1.0, 0.0)
        data.astype(np.float32)

        return data, label


    def __len__(self):
        return len(self.imgs)

