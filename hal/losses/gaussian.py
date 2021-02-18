# gaussian.py

import torch
from torch import nn

__all__ = ['GaussianEM', 'GaussianKL', 'GaussianOT', 'GaussianMSE']


class GaussianKL(nn.Module):

    def __init__(self):
        super(GaussianKL).__init__()
    
    def __call__(self, mean1, covariance1, mean2, covariance2):
        distance = None # TBD
        return distance


class GaussianOT(nn.Module):

    def __init__(self):
        super(GaussianOT).__init__()

    def __call__(self, mean1, covariance1, mean2, covariance2):
        distance = None  # TBD
        return distance


class GaussianMSE(nn.Module):

    def __init__(self):
        super(GaussianMSE).__init__()

    def __call__(self, mean1, covariance1, mean2, covariance2):
        distance = torch.norm(mean1-mean2) + torch.norm(covariance1-covariance2)
        return distance

class GaussianEM(nn.Module):

    def __init__(self, dim, nclasses, distance):
        super(GaussianEM, self).__init__()
        self.nclasses = nclasses
        self.mean = nn.Parameter(torch.zeros(nclasses, dim))
        self.covariance = nn.Parameter(torch.zeros(nclasses, dim, dim))
        self.distance = distance

    def __call__(self, inputs, target):
        loss = 0
        for i in range(self.nclasses):
            x = inputs[target==i]
            mean = x.mean(dim=0)
            temp = x - mean
            covariance = torch.mm(temp, torch.t(temp)).mean(dim=0)
            loss = self.distance(mean, covariance, self.mean, self.covariance)
        return loss
