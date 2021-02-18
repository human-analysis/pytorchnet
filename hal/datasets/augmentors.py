#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: augmentors.py

import io
from PIL import Image
import torchvision.transforms as transforms

class Augmentor:
    def __init__(self,img_size, isTrain):

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        
        if isTrain:
            self.transform = transforms.Compose([
                transforms.RandomSizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    
    def imagenet_augmentor(self, dp):
        image, label = dp
        image = Image.open(io.BytesIO(image))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)
        return image, label
