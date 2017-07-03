# resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, shortcut=None):
        super(BasicBlock, self).__init__()
        self.layers = nn.Sequential(
            conv3x3(in_planes, planes, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, shortcut=None):
        super(Bottleneck, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y


class ResNet(nn.Module):
    def __init__(self, block, nblocks, nchannels, nfilters, nclasses):
        super(ResNet, self).__init__()
        self.in_planes = nfilters
        self.pre_layers = nn.Sequential(
            conv3x3(nchannels,nfilters),
            nn.BatchNorm2d(nfilters),
            nn.ReLU(True),
        )
        self.layer1 = self._make_layer(block, 1*nfilters, nblocks[0])
        self.layer2 = self._make_layer(block, 2*nfilters, nblocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4*nfilters, nblocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8*nfilters, nblocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.linear = nn.Linear(8*nfilters*block.expansion, nclasses)

    def _make_layer(self, block, planes, nblocks, stride=1):
        shortcut = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def resnet18(nchannels, nfilters, nclasses):
    return ResNet(BasicBlock, [2,2,2,2], nchannels=nchannels, nfilters=nfilters, nclasses=nclasses)

def resnet34(nchannels, nfilters, nclasses):
    return ResNet(BasicBlock, [3,4,6,3], nchannels=nchannels, nfilters=nfilters, nclasses=nclasses)

def resnet50(nchannels, nfilters, nclasses):
    return ResNet(Bottleneck, [3,4,6,3], nchannels=nchannels, nfilters=nfilters, nclasses=nclasses)

def resnet101(nchannels, nfilters, nclasses):
    return ResNet(Bottleneck, [3,4,23,3], nchannels=nchannels, nfilters=nfilters, nclasses=nclasses)

def resnet152(nchannels, nfilters, nclasses):
    return ResNet(Bottleneck, [3,8,36,3], nchannels=nchannels, nfilters=nfilters, nclasses=nclasses)