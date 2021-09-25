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
python3 extract_featuremaps.py
python3 extract_branch_feature.py
python3 databuilder.py
python3 multi_level_scale_data.py
```

## Train Models

With the extracted features, the models can be trained respectively.

```sh
python3 baseline_resnet18.py
python3 baseline_mldrnet.py
python3 baseline_cnnrnn.py
python3 privacyMSML.py
python3 privacyMLMS.py
```

