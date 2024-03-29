# model.py

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from collections import OrderedDict

import hal.models as models
import hal.losses as losses
import hal.metrics as metrics

class Model(pl.LightningModule):
    def __init__(self, opts, dataloader=None):
        super().__init__()
        self.save_hyperparameters(opts)

        if dataloader is not None:
            self.val_dataloader = dataloader.val_dataloader
            self.train_dataloader = dataloader.train_dataloader

        self.model = getattr(models, self.hparams.model_type)(**self.hparams.model_options)
        if self.hparams.loss_type is not None:
            self.val_loss = getattr(losses, self.hparams.loss_type)(**self.hparams.loss_options)
        if self.hparams.loss_type is not None:
            self.train_loss = getattr(losses, self.hparams.loss_type)(**self.hparams.loss_options)
        
        self.acc_trn = getattr(metrics, self.hparams.evaluation_type)(**self.hparams.evaluation_options)
        self.acc_val = getattr(metrics, self.hparams.evaluation_type)(**self.hparams.evaluation_options)
        self.acc_tst = getattr(metrics, self.hparams.evaluation_type)(**self.hparams.evaluation_options)

    def on_train_start(self):
        if self.logger is not None:
            if self.global_rank == 0:
                temp = torch.zeros((1, 3, self.hparams.resolution_high, self.hparams.resolution_wide)).to(self.device)
                self.logger.experiment.add_graph(self.model, temp)
                temp = []

    def training_step(self, batch, batch_idx):
        images, labels = batch
        out = self.model(images)
        loss = self.train_loss(out, labels)
        acc = self.acc_trn(F.softmax(out, dim=1), labels)        
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, logger=True,  prog_bar=True)
        output = OrderedDict({
            'loss': loss,
            'acc': acc
        })
        return output

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        out = self.model(images)
        loss = self.val_loss(out, labels)
        acc = self.acc_val(F.softmax(out, dim=1), labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, logger=True,  prog_bar=True)
        output = OrderedDict({
            'loss': loss,
            'acc': acc
        })
        return output
    
    def testing_step(self, batch, batch_idx):
        images, labels = batch
        out = self.model(images)
        acc = self.acc_tst(F.softmax(out, dim=1), labels)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optim_method)(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hparams.learning_rate, **self.hparams.optim_options)
        if self.hparams.scheduler_method is not None:
            scheduler = getattr(torch.optim.lr_scheduler, self.hparams.scheduler_method)(
                optimizer, **self.hparams.scheduler_options
            )
            return [optimizer], [scheduler]
        else:
            return [optimizer]