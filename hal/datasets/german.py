# german.py

import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader

__all__ = ['German']

class PrepareGerman:
    def __init__(self, data_dir, split):
        self.data = np.loadtxt(data_dir + 'data.txt')
        self.data = torch.from_numpy(self.data)

        if split == 'train':
            self.x = self.data[0:700, 0:24].float()
            self.y = abs(self.data[0:700, 24]  - 2).long().squeeze()
            self.s = abs(self.data[0:700, 14] - 2).float().unsqueeze(1)
        elif split == 'val':
            self.x = self.data[700:850, 0:24].float()
            self.y = abs(self.data[700:850, 24] - 2).long().squeeze()
            self.s = abs(self.data[700:850, 14] - 2).float().unsqueeze(1)
        else:
            self.x = self.data[850:, 0:24].float()
            self.y = abs(self.data[850:, 24] - 2).long().squeeze()
            self.s = abs(self.data[850:, 14] - 2).float().unsqueeze(1)


    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x, y, s = self.x[index], self.y[index], self.s[index]
        return x, y, s

class German(pl.LightningDataModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        if opts.ngpu == 0:
            self.pin_memory = False
        else:
            self.pin_memory = True

    def train_dataloader(self):
        dataset = PrepareGerman(self.opts.dataroot, 'train')
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_train,
            shuffle=True,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self):
        dataset = PrepareGerman(self.opts.dataroot, 'val')
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_val,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_dataloader(self):
        dataset = PrepareGerman(self.opts.dataroot, 'test')
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_test,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader
