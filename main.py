# main.py

import os
import sys
import config
import traceback
from hal.utils import misc

import pytorch_lightning as pl
import pytorch_lightning.callbacks as cbs
from pytorch_lightning.loggers import TensorBoardLogger

from model import Model
import hal.datasets as datasets


def main():
    # parse the arguments
    args = config.parse_args()

    if args.ngpu == 0:
        args.device = 'cpu'

    pl.seed_everything(args.manual_seed)

    callbacks = [cbs.RichProgressBar()]
    if args.save_results:
        logger = TensorBoardLogger(
            save_dir=args.logs_dir,
            log_graph=True,
            name=args.project_name
        )
        checkpoint = cbs.ModelCheckpoint(
            dirpath=os.path.join(args.save_dir, args.project_name),
            filename=args.project_name + '-{epoch:03d}-{val_loss:.3f}',
            monitor='val_loss',
            save_top_k=args.checkpoint_max_history,
            save_weights_only=True
            )
        enable_checkpointing = True
        callbacks.append(checkpoint)
    else:
        logger=False
        checkpoint=None
        enable_checkpointing=False
    
    if args.swa:
        callbacks.append(cbs.StochasticWeightAveraging())

    dataloader = getattr(datasets, args.dataset)(args)
    model = Model(args, dataloader)

    if args.ngpu == 0:
        strategy = None
        sync_batchnorm = False
    elif args.ngpu > 1:
        strategy = 'ddp'
        sync_batchnorm = True
    else:
        strategy = 'dp'
        sync_batchnorm = False

    trainer = pl.Trainer(
        gpus=args.ngpu,
        strategy=strategy,
        sync_batchnorm=sync_batchnorm,
        benchmark=True,
        callbacks=callbacks,
        enable_checkpointing=enable_checkpointing,
        logger=logger,
        min_epochs=1,
        max_epochs=args.nepochs,
        precision=args.precision
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