# celeba.py

import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader

import os
import csv
import math
import torch
from PIL import Image
from random import shuffle

__all__ = ['CelebA']


def loader_image(path):
    return Image.open(path).convert('RGB')


class PrepareCelebA:
    def __init__(self, ifile, root=None, split=1.0, transform=None,  prefetch=False):
        self.root = root
        self.ifile = ifile
        self.split = split
        self.prefetch = prefetch
        self.transform = transform

        self.nattributes = 0
        datalist = []
        classname = []
        if ifile is not None:
            with open(ifile, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    if self.nattributes <= len(row):
                        self.nattributes = len(row)
                    datalist.append(row)
                    classname.append(row[1])
            csvfile.close()
        else:
            datalist = []

        if (self.split < 1.0) & (self.split > 0.0):
            if len(datalist) > 0:
                datalist = shuffle(datalist, classname)
                num = math.floor(self.split * len(datalist))
                self.data = datalist[0:num]
            else:
                self.data = []

        elif self.split == 1.0:
            if len(datalist) > 0:
                self.data = datalist
            else:
                self.data = []

        self.classname = list(set(classname))
        self.classname.sort()

        if prefetch is True:
            print('Prefetching data, feel free to stretch your legs !!')
            self.objects = []
            for index in range(len(self.data)):
                if self.root is not None:
                    path = os.path.join(self.root, self.data[index][0])
                else:
                    path = self.data[index][0]
                self.objects.append(loader_image(path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if len(self.data) > 0:
            if self.prefetch is False:
                if self.root is not None:
                    path = os.path.join(self.root, self.data[index][0])
                else:
                    path = self.data[index][0]
                image = loader_image(path)
            elif self.prefetch is True:
                image = self.objects[index]

            attributes = self.data[index]
            attributes = attributes[1:]
            attributes[0] = self.classname.index(attributes[0])
            if len(attributes) > 1:
                try:
                    attributes[1:] = [int(x) for x in attributes[1:]]
                except:
                    attributes[1:] = [0 for x in attributes[1:]]
            else:
                attributes[1:] = [0 for x in range(self.nattributes - 2)]

        if self.transform is not None:
            image = self.transform(image)

        attributes = torch.Tensor(attributes)
        y = attributes[32].long()
        s = attributes[20]

        return image, y, s

# 0 is person PID train_bal, test_bal,
# 1  5_o_Clock_Shadow      88.83, 90.01
# 2  Arched_Eyebrows       73.41, 71.55
# 3  Attractive            51.36, 50.41  ***
# 4  Bags_Under_Eyes       79.55, 79.74
# 5  Bald                  97.72, 97.88
# 6  Bangs                 84.83, 84.42
# 7  Big_Lips              75.91, 67.30
# 8  Big_Nose              76.44, 78.80
# 9  Black_Hair            76.10, 72.84
# 10  Blond_Hair           85.10, 86.67
# 11  Blurry               94.86, 94.94
# 12  Brown_Hair           79.61, 82.04
# 13  Bushy_Eyebrows       85.63, 87.04
# 14  Chubby               94.23, 94.70
# 15  Double_Chin          95.35, 95.43
# 16  Eyeglasses           93.54, 93.55
# 17  Goatee               93.65, 95.42
# 18  Gray_Hair            95.76, 96.81
# 19  Heavy_Makeup         61.57, 59.50  **
# 20  High_Cheekbones      54.76, 51.82  ***
# 21  Male                 58.06, 61.35  ***
# 22  Mouth_Slightly_Open  51.78, 50.49  ***
# 23  Mustache             95.92, 96.13
# 24  Narrow_Eyes          88.41, 85.13
# 25  No_Beard             83.42, 85.37
# 26  Oval_Face            71.68, 70.44
# 27  Pale_Skin            95.70, 95.80
# 28  Pointy_Nose          72.45, 71.42
# 29  Receding_Hairline    91.99, 91.51
# 30  Rosy_Cheeks          99.53, 92.83
# 31  Sideburns            94.37, 95.36
# 32  Smiling              52.03, 50.03  ***
# 33  Straight_Hair        79.14, 79.01
# 34  Wavy_Hair            68.06, 63.59  *
# 35  Wearing_Earrings     81.35, 79.33
# 36  Wearing_Hat          95.06, 95.80
# 37  Wearing_Lipstick     53.04, 52.19 ***
# 38  Wearing_Necklace     87.86, 86.21
# 39  Wearing_Necktie      92.70, 92.99
# 40  Young                77.89, 75.71

# (19, 20):  0.0729, 0.0951 *
# (19, 21):  0.4439, 0.4227 ****
# (19, 22):  0.0106, 0.0130
# (19, 32):  0.0308, 0.0382
# (19, 34):  0.1041, 0.1072 **
# (19, 37):  0.6434, 0.5791 ****

# (20, 21):  0.0615, 0.0781
# (20, 22):  0.1747, 0.1741 ***
# (20, 32):  0.4662, 0.4582 ****
# (20, 34):  0.0131, 0.0130
# (20, 37):  0.0793, 0.0936


class CelebA(pl.LightningDataModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        if opts.ngpu == 0:
            self.pin_memory = False
        else:
            self.pin_memory = True

    def train_dataloader(self):
        dataset = PrepareCelebA(
            root=self.opts.dataroot,
            ifile=self.opts.input_filename_train,
            transform=transforms.Compose([
                transforms.Resize((self.opts.resolution_high,
                                   self.opts.resolution_wide)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_train,
            shuffle=True,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self):
        dataset = PrepareCelebA(
            root=self.opts.dataroot,
            ifile=self.opts.input_filename_test,
            transform=transforms.Compose([
                transforms.Resize((self.opts.resolution_high,
                                   self.opts.resolution_wide)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_test,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_dataloader(self):
        dataset = PrepareCelebA(
            root=self.opts.dataroot,
            ifile=self.opts.input_filename_test,
            transform=transforms.Compose([
                transforms.Resize((self.opts.resolution_high,
                                   self.opts.resolution_wide)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_test,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader