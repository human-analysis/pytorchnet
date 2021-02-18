# mscoco.py

import hal.datasets as datasets
from hal.dataflow.serialize import LMDBSerializer
from hal.dataflow.parallel import MultiProcessRunnerZMQ
from hal.dataflow.common import LocallyShuffleData, MapData, BatchData

__all__ = ['MSCOCOPose']

class MSCOCOPose:
    def __init__(self, opts):
        super().__init__()
        self.opts = opts

    def setup(self, stage=None):
        train_dataset = datasets.Augmentor((self.opts.resolution_wide, self.opts.resolution_high),**self.opts.augmentor_train)
        self.ds_train = LMDBSerializer.load(self.opts.data_filename_train, shuffle=False)
        self.ds_train = LocallyShuffleData(self.ds_train, self.opts.cache_size)
        self.ds_train = MapData(self.ds_train,train_dataset.openpose_augmentor)
        self.ds_train = MultiProcessRunnerZMQ(self.ds_train, self.opts.nthreads)
        self.ds_train = BatchData(self.ds_train, self.opts.batch_size_train)
        self.ds_train.reset_state()

        val_dataset = datasets.Augmentor((self.opts.resolution_wide, self.opts.resolution_high),**self.opts.augmentor_test)
        self.ds_val = LMDBSerializer.load(self.opts.data_filename_test, shuffle=False)
        self.ds_val = MapData(self.ds_val, val_dataset.openpose_augmentor)
        self.ds_val = MultiProcessRunnerZMQ(self.ds_val, self.opts.nthreads)
        self.ds_val = BatchData(self.ds_val, self.opts.batch_size_test)
        self.ds_val.reset_state()

    def train_dataloader(self):
        return enumerate(self.ds_train)

    def val_dataloader(self):
        return enumerate(self.ds_val)

class MSCOCOSegmentation:
    def __init__(self, opts):
        super().__init__()
        self.opts = opts

    def setup(self, stage=None):
        train_dataset = datasets.Augmentor((self.opts.resolution_wide, self.opts.resolution_high),**self.opts.augmentor_train)
        self.ds_train = LMDBSerializer.load(self.opts.data_filename_train, shuffle=False)
        self.ds_train = LocallyShuffleData(self.ds_train, self.opts.cache_size)
        self.ds_train = MapData(self.ds_train,train_dataset.openpose_augmentor)
        self.ds_train = MultiProcessRunnerZMQ(self.ds_train, self.opts.nthreads)
        self.ds_train = BatchData(self.ds_train, self.opts.batch_size_train)
        self.ds_train.reset_state()

        val_dataset = datasets.Augmentor((self.opts.resolution_wide, self.opts.resolution_high),**self.opts.augmentor_test)
        self.ds_val = LMDBSerializer.load(self.opts.data_filename_test, shuffle=False)
        self.ds_val = MapData(self.ds_val, val_dataset.openpose_augmentor)
        self.ds_val = MultiProcessRunnerZMQ(self.ds_val, self.opts.nthreads)
        self.ds_val = BatchData(self.ds_val, self.opts.batch_size_test)
        self.ds_val.reset_state()

    def train_dataloader(self):
        return enumerate(self.ds_train)

    def val_dataloader(self):
        return enumerate(self.ds_val)