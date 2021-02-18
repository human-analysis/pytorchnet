# cifar.py

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler

__all__ = ['CASIAWebFace']

# this is incomplete
class CASIAWebFace(pl.LightningDataModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts

        if opts.transform_trn:
            self.transform_trn = opts.transform_trn
        else:
            self.transform_trn = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        if opts.transform_tst:
            self.transform_tst = opts.transform_tst
        else:
            self.transform_tst = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    def train_dataloader(self):
        batch_size = self.opts.batch_size_train

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            sampler=trn_sampler,
            num_workers=self.opts.nthreads,
            pin_memory=True,
        )

        return loader

    def val_dataloader(self):
        batch_size = self.opts.batch_size_test

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=self.opts.nthreads,
            pin_memory=True,            
        )

        return loader

    def test_dataloader(self):
        batch_size = self.opts.batch_size_test
        

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=tst_sampler,
            num_workers=self.opts.nthreads,
            pin_memory=True,
        )

        return loader
