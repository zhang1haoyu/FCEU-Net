# -*- coding:utf-8  -*-
from .model import *
from .model.FCEU_Net import FECU_Net
from .model.UNet import UNet
def SSegmentationSet(model: str, num_classes=5, pretrained='', img_size=256):
    if model == 'baseline':
        return Baseline(img_size=img_size, pretrained=pretrained, num_classes=num_classes)
    elif model == 'fecunet':
        return FECU_Net(img_size=img_size, pretrained=pretrained, num_classes=num_classes)
    elif model == 'unet':
        return UNet(num_classes=num_classes)


