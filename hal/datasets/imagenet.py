# imagenet.py

import torch
import pytorch_lightning as pl
from hal.dataflow.serialize import LMDBSerializer
from hal.dataflow.parallel import MultiProcessRunnerZMQ
from hal.dataflow.common import LocallyShuffleData, MapData, BatchData

from .augmentors import Augmentor

__all__ = ['ImageNet']

class ImageNet(pl.LightningDataModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts

    def setup(self):

        num_replicas = self.opts.ngpus
        if self.opts.ngpus > 1:
            rank = torch.distributed.get_rank()
        else:
            rank = 1

        train_dataset = Augmentor(
            (self.opts.resolution_wide, self.opts.resolution_high), isTrain=True)
        self.ds_train = LMDBSerializer.load(self.opts.data_filename_train, rank, num_replicas, shuffle=False)
        self.ds_train = LocallyShuffleData(self.ds_train, self.opts.cache_size)
        self.ds_train = MapData(self.ds_train, train_dataset.imagenet_augmentor)
        self.ds_train = MultiProcessRunnerZMQ(self.ds_train, self.opts.nthreads)
        self.ds_train = BatchData(self.ds_train, self.opts.batch_size_train, remainder=False)
        self.ds_train.reset_state()

        val_dataset = Augmentor(
            (self.opts.resolution_wide, self.opts.resolution_high), isTrain=False)
        self.ds_val = LMDBSerializer.load(self.opts.data_filename_val, rank, num_replicas, shuffle=False)
        self.ds_val = MapData(self.ds_val, val_dataset.imagenet_augmentor)
        self.ds_val = MultiProcessRunnerZMQ(self.ds_val, self.opts.nthreads)
        self.ds_val = BatchData(self.ds_val, self.opts.batch_size_test, remainder=True)
        self.ds_val.reset_state()

    def train_dataloader(self):
        return enumerate(self.ds_train)

    def val_dataloader(self):
        return enumerate(self.ds_val)
