import numpy as np
import os
from tqdm import tqdm
import pickle
import pdb

# trainset
fns = os.listdir("./train_test_valid/feature_train_branch0")

dst = "./train_test_valid/train0"
if not os.path.exists(dst):
    os.mkdir(dst)

for fn in tqdm(fns):

    features = []
    for branch in range(0, 5):
       path = "./train_test_valid/feature_train_branch{}/{}".format(branch, fn)
       with open(path, "rb") as f:
            data = pickle.load(f)
            features.append(data.tolist())

    features = np.array(features, dtype=np.float32)
    with open("{}/{}".format(dst, fn), "wb") as f:
        pickle.dump(features, f)


# validset
fns = os.listdir("./train_test_valid/feature_valid_branch0")

dst = "./train_test_valid/valid0"
if not os.path.exists(dst):
    os.mkdir(dst)

for fn in tqdm(fns):

    features = []
    for branch in range(0, 5):
       path = "./train_test_valid/feature_valid_branch{}/{}".format(branch, fn)
       with open(path, "rb") as f:
            data = pickle.load(f)
            features.append(data.tolist())

    features = np.array(features, dtype=np.float32)
    with open("{}/{}".format(dst, fn), "wb") as f:
        pickle.dump(features, f)


# testset
fns = os.listdir("./train_test_valid/feature_test_branch0")

dst = "./train_test_valid/test0"
if not os.path.exists(dst):
    os.mkdir(dst)

for fn in tqdm(fns):
    features = []
    for branch in range(0, 5):
       path = "./train_test_valid/feature_test_branch{}/{}".format(branch, fn)
       with open(path, "rb") as f:
            data = pickle.load(f)
            features.append(data.tolist())

    features = np.array(features, dtype=np.float32)
    with open("{}/{}".format(dst, fn), "wb") as f:
        pickle.dump(features, f)


