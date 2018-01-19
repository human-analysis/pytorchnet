# main.py

import torch
import random

import utils
from model import Model
from test import Tester
from train import Trainer
from config import parser
from dataloader import Dataloader
from checkpoints import Checkpoints


def main():
    # parse the arguments
    args = parser.parse_args()
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    utils.saveargs(args)

    # initialize the checkpoint class
    checkpoints = Checkpoints(args)

    # Create Model
    models = Model(args)
    model, criterion = models.setup(checkpoints)

    # Data Loading
    dataloader = Dataloader(args)
    loaders = dataloader.create()

    # The trainer handles the training loop
    trainer = Trainer(args, model, criterion)
    # The trainer handles the evaluation on validation set
    tester = Tester(args, model, criterion)

    # start training !!!
    loss_best = 1e10
    for epoch in range(args.nepochs):

        # train for a single epoch
        loss_train = trainer.train(epoch, loaders)
        loss_test = tester.test(epoch, loaders)

        if loss_best > loss_test:
            model_best = True
            loss_best = loss_test
            checkpoints.save(epoch, model, model_best)


if __name__ == "__main__":
    main()
