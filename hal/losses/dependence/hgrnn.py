# hgrnn.py

import torch
import torch.nn as nn
import hal.models as models

__all__ = ['MetricHGRNN']


class MetricHGRNN(nn.Module):
    def __init__(self, opts):
        super(MetricHGRNN, self).__init__()

        self.epsilon = 1e-8
        self.model_x = getattr(models, opts.control_model_1)(**opts.control_model_options_1)
        self.model_y = getattr(models, opts.control_model_2)(**opts.control_model_options_2)
        self.optimizer_x = torch.optim.Adam(self.model_x.parameters(), lr=0.0005)
        self.optimizer_y = torch.optim.Adam(self.model_y.parameters(), lr=0.0005)
        self.niters = opts.control_niters
        self.train_flag = False
        self.model_x.eval()
        self.model_y.eval()

    def forward(self, inputs, target):
        # if niters is more than 1, make sure the batch size is large
        inputs_temp = inputs.detach().clone()
        target_temp = target.detach().clone()
        if self.train_flag is True:
            with torch.enable_grad():
                self.model_x.train()
                self.model_y.train()
                for i in range(self.niters):
                    pred_x = self.model_x(inputs_temp)
                    pred_y = self.model_y(target_temp)
                    pred_x_norm = (pred_x-pred_x.mean())/torch.sqrt((torch.std(pred_x).pow(2)+self.epsilon))
                    pred_y_norm = (pred_y-pred_y.mean())/torch.sqrt((torch.std(pred_y).pow(2)+self.epsilon))
                    loss_temp = -(pred_x_norm*pred_y_norm).mean()
                    self.model_x.zero_grad()
                    self.model_y.zero_grad()
                    loss_temp.backward()
                    self.optimizer_x.step()
                    self.optimizer_y.step()
            self.model_x.eval()
            self.model_y.eval()

        pred_x = self.model_x(inputs)
        pred_y = self.model_y(target)
        pred_x_norm = (pred_x-pred_x.mean())/torch.sqrt((torch.std(pred_x).pow(2)+self.epsilon))
        pred_y_norm = (pred_y-pred_y.mean())/torch.sqrt((torch.std(pred_y).pow(2)+self.epsilon))
        out = (pred_x_norm*pred_y_norm).mean()
        return out