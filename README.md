# Privacy Image Classification
PyTorch implementation of **Learning Multi-Level and Multi-Scale Deep Representations for Privacy Image Classifification**.

## Requirements
Please install the following packages
- python-3.6.5
- numpy-1.19.5
- torch-1.3.1
- scikit-learn-0.22
- torch-1.3.1
- torchvision-0.4.2
- tqdm-4.43.0

## Download Dataset

[PicAlert](http://l3s.de/picalert/) is a large dataset for privacy image classification with over 30000 public and privacy images. The annotated data set contains the photo id of the images on flickr and the images are downloadable with flickr api. The annotated data set is available [here](http://l3s.de/picalert/cleaned.zip).

## Feature Extraction

To simplify the training process, the features are first extracted with pretrained resnet18. 

```sh
python extract_featuremap.py
```

## Train Models

With the extracted features, the models can be trained respectively.

```sh
python resnet.py
python mldrnet.py
python bilstm.py
python mlms.py
python msml.py
```

