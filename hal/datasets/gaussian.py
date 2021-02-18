# gaussian.py

import math
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

__all__ = ['Gaussian']


class GaussianBashir:
    def __init__(self, nsamples):
        sigma = 0.01
        p0 = torch.distributions.MultivariateNormal(
            torch.Tensor([0., 1.]), torch.Tensor([[sigma, 0.], [0., sigma]]))
        p1 = torch.distributions.MultivariateNormal(
            torch.Tensor([1., 1.]), torch.Tensor([[sigma, 0.], [0., sigma]]))

        x0 = p0.sample((nsamples // 2,))
        x1 = p1.sample((nsamples // 2,))

        self.x = torch.zeros(nsamples, 2)
        self.x[:nsamples // 2] = x0
        self.x[nsamples // 2:] = x1
        self.s = torch.cos(math.pi * self.x[:, 0] + math.pi / 2 * torch.ones(self.x[:, 0].shape)).unsqueeze(1)
        self.x[:, 0] = torch.sin(math.pi * self.x[:, 0] + math.pi / 2 * torch.ones(self.x[:, 0].shape))
        
        self.y = torch.zeros(nsamples).long()
        self.y[nsamples // 2:] = 1

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, index):
        x, y, s = self.x[index], self.y[index], self.s[index]
        # return x, y, s
        return x, x, s


class GaussianHGR:
    def __init__(self, nsamples):
            p0 = torch.distributions.MultivariateNormal(
                torch.Tensor([0., 0.]), torch.Tensor([[1., -0.5], [-0.5, 1.]]))
            p1 = torch.distributions.MultivariateNormal(
                torch.Tensor([1., 1.]), torch.eye(2))
            ps = torch.distributions.Normal(
                torch.tensor([0.0]), torch.tensor([1.0]))

            x0 = p0.sample((nsamples // 2,))
            x1 = p1.sample((nsamples // 2,))
            self.s = ps.sample((nsamples,))
            x1[:, 1] += torch.sin(self.s[nsamples // 2:]).squeeze().mul(3)
            self.x = torch.vstack([x0, x1])
            self.y = torch.zeros(nsamples).long()
            self.y[nsamples // 2:] = 1

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, index):
        x, y, s = self.x[index], self.y[index], self.s[index]
        return x, y, s


class Gaussian(pl.LightningDataModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        if opts.ngpu == 0:
            self.pin_memory = False
        else:
            self.pin_memory = True

    def train_dataloader(self):
        batch_size = self.opts.batch_size_train
        if self.opts.dataset_type == 'GaussianHGR':
            dataset = GaussianHGR(**self.opts.dataset_options)
        elif self.opts.dataset_type == 'GaussianBashir':
            dataset = GaussianBashir(**self.opts.dataset_options)
        else:
            print('Gaussian dataset type does not exist')
            dataset = None
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self):
        batch_size = self.opts.batch_size_test
        if self.opts.dataset_type == 'GaussianHGR':
            dataset = GaussianHGR(**self.opts.dataset_options)
        elif self.opts.dataset_type == 'GaussianBashir':
            dataset = GaussianBashir(**self.opts.dataset_options)
        else:
            print('Gaussian dataset type does not exist')
            dataset = None
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader
