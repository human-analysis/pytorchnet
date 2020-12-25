# main.py

import os
import sys
import utils
import config
import traceback

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import datasets
from model import Model

def main():
    # parse the arguments
    args = config.parse_args()

    pl.seed_everything(args.manual_seed)    
    utils.saveargs(args)

    logger = TensorBoardLogger(
        save_dir=args.logs_dir,
        log_graph=True,
        name=args.project_name
    )

    data = getattr(datasets, args.dataset)(args)
    model = Model(args)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.save_dir, args.project_name + '-{epoch:03d}-{val_loss:.3f}'),
        monitor='val_loss',
        save_top_k=3)

    trainer = pl.Trainer(
        gpus=args.ngpu,
        accelerator='ddp',
        sync_batchnorm=True,
        benchmark=True,
        checkpoint_callback=checkpoint_callback,
        logger=logger,
        min_epochs=1,
        max_epochs=args.nepochs,        
        precision=args.precision,
        reload_dataloaders_every_epoch=True,
    )

    trainer.fit(model, data)

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
        traceback.print_exc(file=sys.stdout)
        utils.cleanup()