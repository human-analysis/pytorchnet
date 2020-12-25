# cifar.py

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torch.utils.data.distributed import DistributedSampler

__all__ = ['CIFAR10', 'CIFAR100']

class CIFAR10(pl.LightningDataModule):
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

    def prepare_data(self):
        dataset = datasets.CIFAR10(root=self.opts.dataroot, train=True, download=True)
        dataset = datasets.CIFAR10(root=self.opts.dataroot, train=False, download=True)

    def train_dataloader(self):
        batch_size = self.opts.batch_size_train
        trn_sampler = None
        dataset = datasets.CIFAR10(root=self.opts.dataroot, train=True, transform=self.transform_trn)
        # if self.trainer.use_ddp:
        #     trn_sampler = DistributedSampler(dataset, rank=self.trainer.proc_rank)
        #     batch_size = self.opts.batch_size // self.trainer.world_size  # scale batch size

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
        val_sampler = None
        dataset = datasets.CIFAR10(root=self.opts.dataroot, train=False, transform=self.transform_tst)
        # if self.trainer.use_ddp:
        #     val_sampler = DistributedSampler(dataset, rank=self.trainer.proc_rank)
        #     batch_size = self.opts.batch_size // self.trainer.world_size  # scale batch size

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
        tst_sampler = None
        dataset = datasets.CIFAR10(root=self.opts.dataroot, train=False, transform=self.transform_tst)
        # if self.trainer.use_ddp:
        #     tst_sampler = DistributedSampler(dataset, rank=self.trainer.proc_rank)
        #     batch_size = self.opts.batch_size // self.trainer.world_size  # scale batch size

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=tst_sampler,
            num_workers=self.opts.nthreads,
            pin_memory=True,
        )

        return loader

class CIFAR100(pl.LightningDataModule):
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

    def prepare_data(self):
        dataset = datasets.CIFAR100(root=self.opts.dataroot, train=True, download=True)
        dataset = datasets.CIFAR100(root=self.opts.dataroot, train=False, download=True)

    def train_dataloader(self):
        batch_size = self.opts.batch_size_train
        trn_sampler = None
        dataset = datasets.CIFAR100(root=self.opts.dataroot, train=True, transform=self.transform_trn)
        # if self.trainer.use_ddp:
        #     trn_sampler = DistributedSampler(dataset, rank=self.trainer.proc_rank)
        #     batch_size = batch_size // self.trainer.world_size  # scale batch size

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            sampler=trn_sampler
        )

        return loader

    def val_dataloader(self):
        batch_size = self.opts.batch_size_test
        val_sampler = None
        dataset = datasets.CIFAR100(root=self.opts.dataroot, train=False, transform=self.transform_tst)
        # if self.trainer.use_ddp:
        #     val_sampler = DistributedSampler(dataset, rank=self.trainer.proc_rank)
        #     batch_size = batch_size // self.trainer.world_size  # scale batch size

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler
        )

        return loader

    def test_dataloader(self):
        batch_size = self.opts.batch_size_test
        tst_sampler = None
        dataset = datasets.CIFAR100(root=self.opts.dataroot, train=False, transform=self.transform_tst)
        # if self.trainer.use_ddp:
        #     tst_sampler = DistributedSampler(dataset, rank=self.trainer.proc_rank)
        #     batch_size = batch_size // self.trainer.world_size  # scale batch size

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=tst_sampler
        )

        return loader
