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

# image feature map and tags
class MapTag(data.Dataset):
    ''' 读取路径,设置transforms
    root 图片所在目录
    annotation 标注数据文件路径
    '''
    def __init__(self, root=None, annotation=None):
        # 读取标注数据，格式 文件名, 类别
        self.dataset = pd.read_csv(annotation)
        self.imgs, self.labels = zip(*(self.dataset.values.tolist()))
        self.imgs = ['{}/{}'.format(root, img) for img in self.imgs]

    # 读取图片,执行transforms
    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = self.labels[index]

    def __len__(self):
        return len(self.imgs)

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


# 修复bug
class PicAlert1(data.Dataset):

    ''' 读取路径,设置transforms
    root 图片所在目录
    annotation 标注数据文件路径
    train 训练or测试，训练模式会进行数据增强
    '''
    def __init__(self, root=None, annotation=None, train=True, transforms=None):
        # 读取标注数据，格式 文件名, 类别
        self.dataset = pd.read_csv(annotation)

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


# 目标检测提取的特征数据
class PicAlertOD(data.Dataset):

    '''
    root 特征所在目录
    annotation 标注数据文件路径
    '''
    def __init__(self, root=None, annotation=None):
        # 读取标注数据，格式 文件名, 类别
        self.dataset = pd.read_csv(annotation)

        self.imgs, self.labels = zip(*(self.dataset.values.tolist()))

        self.root = root

    # 读取特征
    def __getitem__(self, index):
        feature_path = '{}/{}_fc'.format(self.root, self.imgs[index].split('.')[0])
        label = self.labels[index]

        # 加载保存的特征
        with open(feature_path, 'rb') as f:
            data = pickle.load(f)

        features = data['features']
# #         probs = np.array([v1*v2 for v1, v2 in zip(data1['scores'], data1['prob_values'])])
        probs = data['prob_values']
        idx = probs.argsort()[-3:][0]

#         data0 = np.load('{}/{}.npy'.format('../feature/feature_vgg', self.imgs[index]))

#         data = np.concatenate([data1['features'][idx].astype(np.float32)[0], data0], axis=0)


        return data['features'][idx].astype(np.float32), label


    def __len__(self):
        return len(self.imgs)


'''
class PicAlertF(data.Dataset):

    # 读取路径,设置transforms
    def __init__(self, cv_index=0, root='dataset', train=True):
        if train:
            self.dataset = pd.read_csv('crossvalid/train_{}.csv'.format(cv_index))
        else:
            self.dataset = pd.read_csv('crossvalid/test_{}.csv'.format(cv_index))
        self.imgs, self.labels = zip(*(self.dataset.values.tolist()))
        self.imgs = [os.path.join(root, img) for img in self.imgs]


    # 读取图片,执行transforms
    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = self.labels[index]

        with open(img_path, 'rb') as f:
            data = pickle.load(f)

        return data, label

    def __len__(self):
        return len(self.imgs)


def train_test_split(root = 'dataset', n_fold=4, dst='crossvalid'):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.mkdir(dst)

    fns = os.listdir(root)
    fns.sort()
    annots = pd.DataFrame(fns, columns=['fns'])
    annots['label'] = annots.fns.str.split('_', expand=True)[1]
    annots.to_csv('{}/dataset.csv'.format(dst), index=False)

    pos = annots[annots.label == '1'].reset_index(drop=True)
    pos_index = np.random.permutation(len(pos))
    pos_fold = np.arange(0, len(pos_index), step=len(pos_index) // n_fold)

    neg = annots[annots.label == '0'].reset_index(drop=True)
    neg_index = np.random.permutation(len(neg))
    neg_fold = np.arange(0, len(neg_index), step=len(neg_index) // n_fold)

    for i in range(n_fold):
        if (i + 1) == n_fold:
            pos_train_index = pos_index[pos_fold[i]:]
            neg_train_index = neg_index[neg_fold[i]:]
        else:
            pos_train_index = pos_index[pos_fold[i]:pos_fold[i + 1]]
            neg_train_index = neg_index[neg_fold[i]:neg_fold[i + 1]]

        pos_test_index = list(set(pos_index) - set(pos_train_index))
        neg_test_index = list(set(neg_index) - set(neg_train_index))

        # print(len(neg_train_index), len(neg_test_index), (set(neg_train_index) | set(neg_test_index)) == set(neg_index))
        pos_train = pos.iloc[pos_train_index, :]
        pos_test = pos.iloc[pos_test_index, :]

        neg_train = neg.iloc[neg_train_index, :]
        neg_test = neg.iloc[neg_test_index, :]

        train = pd.concat([pos_train, neg_train])
        test = pd.concat([pos_test, neg_test])


        train.to_csv('{}/train_{}.csv'.format(dst, i), index=False)
        test.to_csv('{}/test_{}.csv'.format(dst, i), index=False)


def valid_train_set_split(n_fold=4, cv='crossvalid'):
    annots = pd.read_csv('{}/dataset.csv'.format(cv))

    for i in range(n_fold):
        train = pd.read_csv('crossvalid/train_{}.csv'.format(i))
        test = pd.read_csv('crossvalid/test_{}.csv'.format(i))
        print(i, (set(train.fns) | set(test.fns)) == set(annots.fns))

    fns = []
    for i in range(n_fold):
        train = pd.read_csv('crossvalid/train_{}.csv'.format(i))
        fns.extend(train.fns.values.tolist())
    print(set(fns)==set(annots.fns))

def valid_dataset():
    dst = 'dataset'
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.mkdir(dst)


    model = load_multiBranch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    index = 0
    with torch.no_grad():
        for _, (data, target) in enumerate(trainloader):
            data = data.to(device)
            target = target.to(device)
            features = model(data)
            features = [f.unsqueeze(1) for f in features]
            features = torch.cat(features, dim=1)  # N*d

            for i in range(len(target)):
                with open('dataset/{}_{}'.format(str(index).zfill(6), target[i]), 'wb') as f:
                    pickle.dump(features[i], f)
                index = index + 1
            print(index)


    fns = os.listdir('dataset')
    fns.sort()
    annots = pd.DataFrame(fns, columns=['fns'])
    annots['label'] = annots.fns.str.split('_', expand=True)[1]
    annots.to_csv('trainf.csv'.format(dataset), index=False)


    data = load_feature()
    for _, (d, target) in enumerate(data):
        print(d.shape, target.shape)'''
