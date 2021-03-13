# hsic.py

import torch
import torch.nn as nn
import hal.models as models

__all__ = ['MetricHSIC']


class MetricHSIC(nn.Module):
    def __init__(self, opts):
        super(MetricHSIC, self).__init__()
        self.kernel1 = getattr(models, opts.kernel_type)(**opts.kernel_options)
        self.kernel2 = getattr(models, opts.kernel_type)(**opts.kernel_options)
        self.type = opts.control_options['type']
        self.biased = opts.control_options['biased']

    def forward(self, inputs, target, sensitive):

        if self.type == 'conditional':
            inputs = inputs[target == 1]
            sensitive = sensitive[target == 1]

        batch_size = inputs.size(0)

        if self.biased == False:
            b1 = int(batch_size / 2)
            b2 = batch_size - b1
                
            x1 = inputs[:b1, :]
            x2 = inputs[b2:, :] # skip a sample if batch size is odd
            s1 = sensitive[:b1, :]
            s2 = sensitive[b2:, :]  # skip a sample if batch size is odd

            k_s = self.kernel1(s1, s2)
            k_x = self.kernel2(x1, x2)

            ones = torch.ones((b1, 1)).to(device=x1.device)

            A = torch.trace(torch.mm(k_x, torch.t(k_s)))
            B = torch.mm(torch.mm(torch.t(ones), k_x),
                         torch.mm(torch.t(k_s), ones))
            B = B[0][0]
            C = torch.mm(torch.mm(torch.t(ones), k_x), ones) * \
                torch.mm(torch.mm(torch.t(ones), torch.t(k_s)), ones)
            C = C[0][0]

            loss = (1 / b1 ** 2 + 2 / ((b1 - 1) * b1 ** 2) +
                    1 / ((b1 - 1) ** 2 * b1 ** 2)) * A
            loss += -2 * (1 / ((b1 - 1) * b1 ** 2) + 1 /
                          ((b1 - 1) ** 2 * b1 ** 2)) * B
            loss += 1 / ((b1 - 1) ** 2 * b1 ** 2) * C
        else:
            K = kernel1(inputs)
            L = kernel2(sensitive)
            H = torch.eye(batch_size) - 1.0/m * torch.ones((batch_size, batch_size))
            H = H.double().cuda()
            loss = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((batch_size - 1) ** 2)

        return loss