# model.py

import torch
import pytorch_lightning as pl
from collections import OrderedDict

import hal.models as models
import hal.losses as losses
import hal.metrics as metrics

class Model(pl.LightningModule):
    def __init__(self, opts, dataloader):
        super().__init__()
        self.save_hyperparameters()
        self.opts = opts

        self.val_dataloader = dataloader.val_dataloader
        self.train_dataloader = dataloader.train_dataloader


        self.model = getattr(models, opts.model_type)(**opts.model_options)
        self.val_loss = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.train_loss = getattr(losses, opts.loss_type)(**opts.loss_options)
        
        self.acc_trn = getattr(metrics, opts.evaluation_type)(**opts.evaluation_options)
        self.acc_val = getattr(metrics, opts.evaluation_type)(**opts.evaluation_options)
        self.acc_tst = getattr(metrics, opts.evaluation_type)(**opts.evaluation_options)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        out = self.model(images)
        loss = self.train_loss(out, labels)
        acc = self.acc_trn(out, labels)        
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        output = OrderedDict({
            'loss': loss,
            'acc': acc
        })
        return output

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        out = self.model(images)
        loss = self.val_loss(out, labels)
        acc = self.acc_val(out, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        output = OrderedDict({
            'loss': loss,
            'acc': acc
        })
        return output
    
    def testing_step(self, batch, batch_idx):
        images, labels = batch
        out = self.model(images)
        acc = self.acc_tst(out, labels)        
        self.log('test_acc', acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.opts.optim_method)(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.opts.learning_rate, **self.opts.optim_options)
        if self.opts.scheduler_method is not None:
            scheduler = getattr(torch.optim.lr_scheduler, self.opts.scheduler_method)(
                optimizer, **self.opts.scheduler_options
            )
        return [optimizer], [scheduler]
