# adult.py

import torch
import scipy.io as sio
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class PrepareAdult(pl.LightningDataModule):
    def __init__(self, root, split):
        self.data = sio.loadmat(root + 'adult_binary.mat')

        if split == 'train':
            x = torch.from_numpy(self.data['D']).float()
            y = torch.from_numpy(self.data['Y']).long().squeeze()
            s = x[71, :]
            self.x = torch.t(x)
            self.y = torch.t(y)
            self.s = torch.t(s).unsqueeze(1)
        if split == 'test' or split == 'val':
            x = torch.from_numpy(self.data['D_test']).float()
            y = torch.from_numpy(self.data['Y_test']).long().squeeze()
            s = x[71, :]
            self.x = torch.t(x)
            self.y = torch.t(y)
            self.s = torch.t(s).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x, y, s = self.x[index], self.y[index], self.s[index]
        return x, y, s


class Adult(pl.LightningDataModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        if opts.ngpu == 0:
            self.pin_memory = False
        else:
            self.pin_memory = True

    def train_dataloader(self):
        dataset = PrepareAdult(self.opts.dataroot, 'train')
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_train,
            shuffle=True,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self):
        dataset = PrepareAdult(self.opts.dataroot, 'val')
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_test,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_dataloader(self):
        dataset = PrepareAdult(self.opts.dataroot, 'test')
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_test,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader
