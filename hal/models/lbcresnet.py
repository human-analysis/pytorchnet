# pnnresnet.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LBCResNet', 'lbcresnet18', 'lbcresnet34',
           'lbcresnet50', 'lbcresnet101', 'lbcresnet152']


def conv3x3(in_planes, out_planes, stride=1):
    conv2d = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                       stride=stride, padding=1, bias=False
                       )
    return conv2d


def lbcconv3x3(in_planes, out_planes, stride=1, sparsity=0.1):
    conv2d = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                       stride=stride, padding=1, bias=False
                       )
    conv2d.weight.requires_grad = False
    conv2d.weight.fill_(0.0)
    num = conv2d.weight.numel()
    index = torch.Tensor(math.floor(sparsity * num)).random_(num).int()
    conv2d.weight.resize_(in_planes * out_planes * 3 * 3)
    for i in range(index.numel()):
        conv2d.weight[index[i]] = torch.bernoulli(torch.Tensor([0.5])) * 2 - 1

    conv2d.weight.resize_(out_planes, in_planes, 3, 3)
    return conv2d


class LBCLayer(nn.Module):
    def __init__(self, in_planes, out_planes, stride, sparsity):
        super(LBCLayer, self).__init__()
        self.layers = nn.Sequential(
            lbcconv3x3(in_planes, out_planes, stride, sparsity),
            nn.ReLU(),
            nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1),
        )

    def forward(self, x):
        z = self.layers(x)
        return z


class LBCBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut=None,
                 sparsity=0.1):
        super(LBCBasicBlock, self).__init__()
        self.layers = nn.Sequential(
            LBCLayer(in_planes, planes, stride, sparsity),
            nn.BatchNorm2d(planes),
            LBCLayer(planes, planes, 1, sparsity),
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


class LBCBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, shortcut=None,
                 sparsity=0.1):
        super(LBCBottleneck, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            LBCLayer(planes, planes, stride, sparsity),
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


class LBCResNet(nn.Module):
    def __init__(self, block, nblocks, nchannels, nfilters,
                 nclasses, sparsity):
        super(LBCResNet, self).__init__()
        self.in_planes = nfilters
        self.pre_layers = nn.Sequential(
            nn.Conv2d(nchannels, nfilters, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(nfilters),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(block, 1 * nfilters, nblocks[0],
                                       sparsity=sparsity)
        self.layer2 = self._make_layer(block, 2 * nfilters, nblocks[1],
                                       stride=2, sparsity=sparsity)
        self.layer3 = self._make_layer(block, 4 * nfilters, nblocks[2],
                                       stride=2, sparsity=sparsity)
        self.layer4 = self._make_layer(block, 8 * nfilters, nblocks[3],
                                       stride=2, sparsity=sparsity)
        self.linear = nn.Linear(8 * nfilters * block.expansion, nclasses)

    def _make_layer(self, block, planes, nblocks, stride=1, sparsity=0.1):
        shortcut = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut,
                            sparsity=sparsity))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes, sparsity=sparsity))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.pre_layers(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = F.avg_pool2d(x5, kernel_size=x5.size()[2:])
        x7 = x6.view(x6.size(0), -1)
        x8 = self.linear(x7)
        return x8


def lbcresnet18(**kwargs):
    """Constructs a LBCResNet-18 model.
    """
    return LBCResNet(LBCBasicBlock, [2, 2, 2, 2], **kwargs)


def lbcresnet34(**kwargs):
    """Constructs a LBCResNet-34 model.
    """
    return LBCResNet(LBCBasicBlock, [3, 4, 6, 3], **kwargs)


def lbcresnet50(**kwargs):
    """Constructs a LBCResNet-50 model.
    """
    return LBCResNet(LBCBottleneck, [3, 4, 6, 3], **kwargs)


def lbcresnet101(**kwargs):
    """Constructs a LBCResNet-101 model.
    """
    return LBCResNet(LBCBottleneck, [3, 4, 23, 3], **kwargs)


def lbcresnet152(**kwargs):
    """Constructs a LBCResNet-152 model.
    """
    return LBCResNet(LBCBottleneck, [3, 8, 36, 3], **kwargs)
