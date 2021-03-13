# renyi.py

import torch
import torch.nn as nn
import hal.models as models

__all__ = ['MetricRenyi']

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetricRenyi(nn.Module):
    def __init__(self, opts):
        super(MetricRenyi, self).__init__()
        self.kernel = getattr(models, opts.kernel_type)(**opts.kernel_options)
        self.type = opts.control_options['type']
        self.alpha = opts.control_options['alpha']
        self.normalize = opts.control_options['normalize']

    def renyi_entropy(self, x):        
        k = self.kernel(x)
        k = k / torch.trace(k)
        eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
        eig_pow = eigv ** alpha
        entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
        return entropy

    def joint_entropy(self, x, y):
        x = self.kernel(x)
        y = self.kernel(y)
        k = torch.mul(x, y)
        k = k / torch.trace(k)
        eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
        eig_pow =  eigv ** self.alpha
        entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
        return entropy

    def forward(self, inputs, target, sensitive):

        if self.type == 'conditional':
            inputs = inputs[target == 1]
            sensitive = sensitive[target == 1]

        Hx = self.renyi_entropy(inputs)
        Hy = self.renyi_entropy(sensitive)
        Hxy = self.joint_entropy(inputs, sensitive)
        
        if self.normalize:
            Ixy = Hx + Hy - Hxy
            Ixy = Ixy / (torch.max(Hx, Hy))
        else:
            Ixy = Hx + Hy - Hxy
        
        return Ixy