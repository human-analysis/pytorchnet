# main.py

import os
import sys
import config
import traceback
from hal.utils import misc

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model import Model
import hal.datasets as datasets


def main():
    # parse the arguments
    args = config.parse_args()

    if args.ngpu == 0:
        args.device = 'cpu'

    pl.seed_everything(args.manual_seed)

    logger = TensorBoardLogger(
        save_dir=args.logs_dir,
        log_graph=True,
        name=args.project_name
    )

    dataloader = getattr(datasets, args.dataset)(args)
    model = Model(args, dataloader)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_dir, args.project_name),
        filename=args.project_name + '-{epoch:03d}-{val_loss:.3f}',
        monitor='val_loss',
        save_top_k=3)

    if args.ngpu == 0:
        accelerator = None
        sync_batchnorm = False
    elif args.ngpu > 1:
        accelerator = 'ddp'
        sync_batchnorm = True
    else:
        accelerator = 'dp'
        sync_batchnorm = False

    trainer = pl.Trainer(
        gpus=args.ngpu,
        accelerator=accelerator,
        sync_batchnorm=sync_batchnorm,
        benchmark=True,
        checkpoint_callback=checkpoint_callback,
        logger=logger,
        min_epochs=1,
        max_epochs=args.nepochs,
        precision=args.precision,
        reload_dataloaders_every_epoch=True,
        check_val_every_n_epoch=args.check_val_every_n_epochs
        )

    trainer.fit(model)
    trainer.predict(model, test_dataloaders=dataloader.test_dataloader())


if __name__ == "__main__":
    misc.setup_graceful_exit()
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        # do not print stack trace when ctrl-c is pressed
        pass
    except Exception:
        traceback.print_exc(file=sys.stdout)
    finally:
        traceback.print_exc(file=sys.stdout)
        misc.cleanup()