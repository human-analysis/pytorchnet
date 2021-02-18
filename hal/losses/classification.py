# classification.py

from torch import nn

__all__ = ['Classification']


# REVIEW: does this have to inherit nn.Module?
class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, inputs, targets):
        loss = self.loss(inputs, targets)
        return loss

class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()
        self.softmax = nn.Softmax

    def __call__(self, inputs):
        inputs = self.softmax(inputs)
        loss = -(inputs * inputs.log()).sum(dim=1)
        return loss