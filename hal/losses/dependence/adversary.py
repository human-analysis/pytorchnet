# adversary.py

import torch
import torch.nn as nn
import hal.models as models
import hal.losses as losses

__all__ = ['MetricAdversary']


class MetricAdversary(nn.Module):
    def __init__(self, opts):
        super(MetricAdversary, self).__init__()
        self.model = getattr(models, opts.control_model)(**opts.control_model_options)
        self.criterion = getattr(losses, opts.control_criterion)(**opts.control_criterion_options)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        self.mode = opts.control_mode
        self.niters = opts.control_niters
        self.train_flag = False
        self.model.eval()

    def forward(self, inputs, target):
        # if niters is more than 1, make sure the batch size is large
        inputs_temp = inputs.detach().clone()
        target_temp = target.detach().clone()
        loss = None
        if self.train_flag is True:
            self.model.train()
            with torch.enable_grad():
                for i in range(self.niters):
                    pred = self.model(inputs_temp)
                    loss = self.criterion(pred, target_temp)
                    self.model.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            self.model.eval()

        if self.mode == 'ML-ARL':
            pred = self.model(inputs)
            loss = -self.criterion(inputs, target)
            return loss
        elif self.mode == 'MaxEnt-ARL':
            pred = self.model(inputs)
            loss = -Entropy()(pred)
            return loss
        else:
            print('Adversary evaluation metric not implemented')
            return None