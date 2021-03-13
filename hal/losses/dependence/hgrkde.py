# hgrkde.py

import torch
import torch.nn as nn
from math import pi, sqrt
import torch.nn.functional as F

__all__ = ['MetricHGRKDE', 'KDE']


class KDE:
    """
    A Gaussian KDE implemented in pytorch for the gradients to flow in pytorch optimization.
    Keep in mind that KDE does not scale well with the number of dimensions and this implementation is not really
    optimized...
    """

    def __init__(self, x_train):
        n, d = x_train.shape

        self.n = n
        self.d = d

        self.bandwidth = (n * (d + 2) / 4.) ** (-1. / (d + 4))
        self.std = self.bandwidth

        self.x_train = x_train

    def pdf(self, x):
        s = x.shape
        d = s[-1]
        s = s[:-1]

        assert d == self.d

        data = x.unsqueeze(-2)
        x_train = _unsqueeze_multiple_times(self.x_train, 0, len(s))
        cuda_check = x_train.is_cuda
        if cuda_check:
            data = data.cuda()
        pdf_values = (
            torch.exp(-((data - x_train).norm(dim=-1)
                        ** 2 / (self.bandwidth ** 2) / 2))
        ).mean(dim=-1) / sqrt(2 * pi) / self.bandwidth

        return pdf_values


def _unsqueeze_multiple_times(input, axis, times):
    """
    Utils function to unsqueeze tensor to avoid cumbersome code
    :param input: A pytorch Tensor of dimensions (D_1,..., D_k)
    :param axis: the axis to unsqueeze repeatedly
    :param times: the number of repetitions of the unsqueeze
    :return: the unsqueezed tensor. ex: dimensions (D_1,... D_i, 0,0,0, D_{i+1}, ... D_k) for unsqueezing 3x axis i.
    """
    output = input
    for i in range(times):
        output = output.unsqueeze(axis)
    return output


class MetricHGRKDE(nn.Module):
    def __init__(self, type, density):
        super(MetricHGRKDE, self).__init__()
        self.type = type
        self.damping = 1e-10
        self.density = density

    def _joint_2(self, X, Y):
        X = (X - X.mean()) / X.std()
        Y = (Y - Y.mean()) / Y.std()
        data = torch.cat([X.unsqueeze(-1), Y.unsqueeze(-1)], -1)
        joint_density = self.density(data)

        nbins = int(min(50, 5. / joint_density.std))
        x_centers = torch.linspace(-2.5, 2.5, nbins)
        y_centers = torch.linspace(-2.5, 2.5, nbins)

        xx, yy = torch.meshgrid([x_centers, y_centers])
        grid = torch.cat([xx.unsqueeze(-1), yy.unsqueeze(-1)], -1)
        h2d = joint_density.pdf(grid) + self.damping
        h2d /= h2d.sum()
        return h2d

    def _joint_3(self, X, Y, Z):
        X = (X - X.mean()) / X.std()
        Y = (Y - Y.mean()) / Y.std()
        Z = (Z - Z.mean()) / Z.std()
        data = torch.cat(
            [X.unsqueeze(-1), Y.unsqueeze(-1), Z.unsqueeze(-1)], -1)
        joint_density = self.density(data)  # + damping

        nbins = int(min(50, 5. / joint_density.std))
        x_centers = torch.linspace(-2.5, 2.5, nbins)
        y_centers = torch.linspace(-2.5, 2.5, nbins)
        z_centers = torch.linspace(-2.5, 2.5, nbins)
        xx, yy, zz = torch.meshgrid([x_centers, y_centers, z_centers])
        grid = torch.cat(
            [xx.unsqueeze(-1), yy.unsqueeze(-1), zz.unsqueeze(-1)], -1)

        h3d = joint_density.pdf(grid) + self.damping
        h3d /= h3d.sum()
        return h3d

    def chi2(self, input):
        """
        The \chi^2 divergence between the joint distribution on (x,y) and the product of marginals. This is known to be the
        square of an upper-bound on the Hirschfeld-Gebelein-Renyi maximum correlation coefficient. We compute it here on an
        empirical and discretized density estimated from the input data.
        :param input: A list of torch 1-D Tensors
        :return: numerical value between 0 and \infty (0: independent)
        """
        if self.type == 'conditional':
            X, Y, Z = input
            X = X[Z == 1]
            Y = Y[Z == 1]
            hd = self._joint_2(X, Y)
        else:
            X, Y, Z = input
            hd = self._joint_2(X, Y)

        marginal_x = hd.sum(dim=1).unsqueeze(1)
        marginal_y = hd.sum(dim=0).unsqueeze(0)
        Q = hd / (torch.sqrt(marginal_x) * torch.sqrt(marginal_y))
        return ((Q ** 2).sum(dim=[0, 1]) - 1.)

    def hgr(self, input):
        """
        An estimator of the function z -> HGR(x|z, y|z) where HGR is the Hirschfeld-Gebelein-Renyi maximum correlation
        coefficient computed using Witsenhausen’s Characterization: HGR(x,y) is the second highest eigenvalue of the joint
        density on (x,y). We compute here the second eigenvalue on an empirical and discretized density estimated from the
        input data.
        :param input: A list of torch 1-D Tensors
        :param density: so far only kde is supported
        :return: A torch 1-D Tensor of same size as Z. (0: independent, 1:linked by a deterministic equation)
        """
        if self.type == 'conditional':
            X, Y, Z = input
            X = X[Z == 1]
            Y = Y[Z == 1]
            hd = self._joint_2(X, Y)
        else:
            X, Y = input
            hd = self._joint_2(X, Y)
        marginal_x = hd.sum(dim=1).unsqueeze(1)
        marginal_y = hd.sum(dim=0).unsqueeze(0)
        Q = hd / (torch.sqrt(marginal_x) * torch.sqrt(marginal_y))

        if self.type == 'conditional':
            return torch.Tensor(([torch.svd(Q[:, :, i])[1][1] for i in range(Q.shape[2])]))
        else:
            return ((Q ** 2).sum(dim=[0, 1]) - 1.)

    def forward(self, z, y, s):
        loss = 0
        for dim in range(z.size(1)):
            loss += self.chi2([z[:, dim], s.squeeze(), y])
        return loss
