import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_TRACK_RUNNING_STATS = False

__all__ = ['NonLearnableLayer']


# modified from pytorchnet's implementation
# https://github.com/human-analysis/pytorchnet/blob/master/models/lbcresnet.py
def LBConv(in_planes, out_planes, kernel_size=3, stride=1,
           padding=1, dilation=1, groups=1, bias=False, sparsity=0.1, binary=True):
    conv2d = nn.Conv2d(
        in_planes, out_planes, kernel_size=kernel_size,
        stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
    )
    conv2d.weight.requires_grad = False
    if conv2d.bias:
        conv2d.bias.requires_grad = False

    if binary:
        conv2d.weight.fill_(0.0)
        num = conv2d.weight.numel()
        shape = conv2d.weight.size()
        index = torch.Tensor(math.floor(sparsity * num)).random_(num).int()
        conv2d.weight.resize_(num)
        for i in range(index.numel()):
            conv2d.weight[index[i]] = torch.bernoulli(torch.Tensor([0.5])) * 2 - 1
        conv2d.weight.resize_(shape)

    return conv2d


# non-learnable layer
class NonLearnableLayer(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, groups, sparsity=0.5, binary=True, drop=0.0):
        super(NonLearnableLayer, self).__init__()
        self.conv = LBConv(in_planes=C_in, out_planes=C_in, kernel_size=kernel_size, stride=stride,
                           padding=padding, groups=groups, bias=False, sparsity=sparsity, binary=binary)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.linear_combination = nn.Conv2d(3 * C_in, C_out, kernel_size=1, padding=0, bias=False)
        self.drop = drop

    def forward(self, x):
        x1 = F.relu(self.conv(x), inplace=True)
        x2 = self.maxpool(x)
        x3 = self.avgpool(x)
        x_cat = torch.cat([x1, x2, x3], dim=1)
        if self.drop > 0:
            x_cat = F.dropout(x_cat, p=self.drop, training=self.training)
        return self.linear_combination(x_cat)


# internal non-learnable layer
class _NonLearnableLayer(nn.Module):
    def __init__(self, C_in, kernel_size, stride, padding, groups, sparsity=0.5, binary=True):
        super(_NonLearnableLayer, self).__init__()
        self.conv = LBConv(in_planes=C_in, out_planes=C_in, kernel_size=kernel_size, stride=stride,
                           padding=padding, groups=groups, bias=False, sparsity=sparsity, binary=binary)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x1 = F.relu(self.conv(x), inplace=True)
        x2 = self.maxpool(x)
        x3 = self.avgpool(x)
        return torch.cat([x1, x2, x3], dim=1)
