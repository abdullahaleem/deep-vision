import os
from shutil import ignore_patterns
import warnings
import torch
import torchvision
import torch.nn as nn

# Location where pretrained models will be downloaded
os.environ['TORCH_HOME'] = "../output/models/pretrained/"

# MobileNets
def mobilenet_v2(in_channels, out_features, pretrained):
    model = torchvision.models.mobilenet_v2(pretrained=pretrained)
    model.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=out_features, bias=True)
    return _cuda_enabled(model)

# ResNets
def resnet50(in_channels, out_features, pretrained):
    model = torchvision.models.resnet50(pretrained=pretrained)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, out_features)
    return _cuda_enabled(model)

def resnet101(in_channels, out_features, pretrained):
    model = torchvision.models.resnet101(pretrained=pretrained)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, out_features)
    return _cuda_enabled(model)

def wide_resnet50_2(in_channels, out_features, pretrained):
    model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, out_features)
    return _cuda_enabled(model)

def wide_resnet101_2(in_channels, out_features, pretrained):
    model = torchvision.models.wide_resnet101_2(pretrained=pretrained)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, out_features)
    return _cuda_enabled(model)

# DenseNets
def densenet121(in_channels, out_features, pretrained, dropout):
    model = torchvision.models.densenet121(pretrained=pretrained, drop_rate=dropout)
    model.features.conv0 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.classifier = nn.Linear(in_features=1024, out_features=out_features, bias=True)
    return _cuda_enabled(model)


def _cuda_enabled(model):
    if torch.cuda.is_available():
        model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model