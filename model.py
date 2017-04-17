# models.py

import torch
from torch import nn
import models
import losses

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Model:

	def __init__(self, args):
		self.cuda = args.cuda
		self.ndim = args.ndim
		self.nunits = args.nunits
		self.dropout = args.dropout
		self.nclasses = args.nclasses
		self.net_type = args.net_type

	def setup(self, checkpoints):
		model = models.resnet18()
		criterion = losses.Classification()

		if checkpoints.latest('resume') == None:
			model.apply(weights_init)
		else:
			tmp = checkpoints.load(checkpoints['resume'])
			model.load_state_dict(tmp)

		if self.cuda:
			model = model.cuda()
			criterion = criterion.cuda()

		return model, criterion