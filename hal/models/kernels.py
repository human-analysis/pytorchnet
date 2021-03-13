# kernels.py

import torch

__all__ = ['GaussianKernel']

class GaussianKernel:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x, s=None):

        if s is None:
            n = x.shape[0]
            x = x.view(x.shape[0], -1)
            x_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
            dist= -2 * torch.mm(x, x.t()) + x_norm + x_norm.t()
            kernel = torch.exp(-dist / self.sigma)
        else:
            n_x = x.shape[0]
            n_s = s.shape[0]

            x_norm = torch.pow(torch.norm(x, dim=1).reshape([1, n_x]), 2)
            s_norm = torch.pow(torch.norm(s, dim=1).reshape([1, n_s]), 2)

            ones_x = torch.ones([1, n_x]).to(device=x.device)
            ones_s = torch.ones([1, n_s]).to(device=x.device)

            kernel = torch.exp(
                (-torch.mm(torch.t(x_norm), ones_s) -
                 torch.mm(torch.t(ones_x), s_norm) + 2 * torch.mm(x, torch.t(s)))
                / self.sigma)

            x = x.view(x.shape[0], -1)
            s = s.view(s.shape[0], -1)
            x_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
            s_norm = torch.sum(s ** 2, -1).reshape((-1, 1))
            dist= -2 * torch.mm(x, s.t()) + x_norm + s_norm.t()
            kernel1 = torch.exp(-dist / self.sigma)

        return kernel