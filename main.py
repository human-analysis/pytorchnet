# main.py

import sys
import traceback
import torch
import random
import config
import utils
from model import Model
from test import Tester
from train import Trainer
from dataloader import Dataloader
from checkpoints import Checkpoints


def main():
    # parse the arguments
    args = config.parse_args()
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if args.save_results:
        utils.saveargs(args)

    # initialize the checkpoint class
    checkpoints = Checkpoints(args)

    # Create Model
    models = Model(args)
    model, criterion, evaluation = models.setup(checkpoints)

    print('Model:\n\t{model}\nTotal params:\n\t{npar:.2f}M'.format(
          model=args.model_type,
          npar=sum(p.numel() for p in model.parameters()) / 1000000.0))

    # Data Loading
    dataloader = Dataloader(args)
    loaders = dataloader.create()

    # The trainer handles the training loop
    trainer = Trainer(args, model, criterion, evaluation)
    # The trainer handles the evaluation on validation set
    tester = Tester(args, model, criterion, evaluation)

    # start training !!!
    loss_best = 1e10
    for epoch in range(args.nepochs):
        print('\nEpoch %d/%d\n' % (epoch + 1, args.nepochs))

        # train for a single epoch
        loss_train = trainer.train(epoch, loaders)
        loss_test = tester.test(epoch, loaders)

        if loss_best > loss_test:
            model_best = True
            loss_best = loss_test
            if args.save_results:
                checkpoints.save(epoch, model, model_best)


if __name__ == "__main__":
    utils.setup_graceful_exit()
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        # do not print stack trace when ctrl-c is pressed
        pass
    except Exception:
        traceback.print_exc(file=sys.stdout)
    finally:
        utils.cleanup()
