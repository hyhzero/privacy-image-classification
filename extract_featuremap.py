from dataset import PicAlert
import torch.utils.data as Data
from mymodels import *
from PIL import ImageFile
import os
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True

root = '/home/yhhan/Desktop/picAlert'
trainset = PicAlert(root=root, annotation='./dataset/train.csv', train=False, extract_feature=True)
validset = PicAlert(root=root, annotation='./dataset/valid.csv', train=False, extract_feature=True)
testset = PicAlert(root=root, annotation='./dataset/test.csv', train=False, extract_feature=True)

trainloader = Data.DataLoader(trainset, batch_size=32, shuffle=True)
validloader = Data.DataLoader(validset, batch_size=32, shuffle=True)
testloader = Data.DataLoader(testset, batch_size=32, shuffle=True)


models = [
    resnetBranch0Map(),
    resnetBranch1Map(),
    resnetBranch2Map(),
    resnetBranch3Map(),
    resnetBranch4Map(),
    resnetBranch5Map()
]

feature_paths = [
    "./feature/branch0map",
    "./feature/branch1map",
    "./feature/branch2map",
    "./feature/branch3map",
    "./feature/branch4map",
    "./feature/branch5map",
]

for i in range(len(models)):
    model = models[i]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    FEATURE_PATH = feature_paths[i]

    if not os.path.exists(FEATURE_PATH):
        os.makedirs(FEATURE_PATH)
        
    for dataloader in [trainloader, validloader, testloader]:
        model.eval()
        with torch.no_grad():
            for data, target in tqdm(dataloader):
                data = data.to(device)
                features = model(data)

                features = features.squeeze()
                # 每张图片提取的特征保存到单个文件中
                for feature, fn in zip(features, target):
                    np.save('{}/{}'.format(FEATURE_PATH, fn), feature.cpu().numpy())