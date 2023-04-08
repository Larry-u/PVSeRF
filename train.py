import datetime
import time
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

try:
    from pytorch_lightning.plugins.ddp_plugin import DDPPlugin
except ModuleNotFoundError:
    from pytorch_lightning.plugins.training_type import DDPPlugin

import pl_models
from utils import create_logdir, save_config, compile_remarks, merge_from_sys_argv, store_cmd, load_config, init_weights


def main():
    parser = ArgumentParser()
    parser.add_argument("--cfg", '-c', default=None, required=True, help="path to config file")
    parser.add_argument("--resume", '-r', default=None, help="resume path to ckpt")
    parser.add_argument('--debug', '-d', default=False, action='store_true', help='set debugging mode, shorten epochs')
    parser.add_argument('--verbose', '-v', default=False, action='store_true', help='if print config content')
    parser.add_argument('--acc', default=None, help='specify gpu accelerator')
    args, _ = parser.parse_known_args()

    if args.debug:
        print('WARNING: debug mode on')

    # create log dir and dump config into json
    if not args.resume:
        print(f'INFO: config path: {args.cfg}')
        c = load_config(args.cfg)
        merge_from_sys_argv(c)
        compile_remarks(c)
        create_logdir(c, resume_path=None, args=args)
        store_cmd(c)
        save_config(c)
    else:
        print(f'INFO: resuming from {args.resume}')
        c = create_logdir(resume_path=args.resume)

    if args.verbose or c.trainer.fast_dev_run:
        print('----------------- Configs ---------------')
        print(c)
        print('----------------- End -------------------')

    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED. Ensure reproducibility
    pl.seed_everything(c.trainer.rnd_seed)

    # init model
    model = getattr(pl_models, c.trainer.model)(c)
    try:
        if c.trainer.init_type is not None and not args.resume:
            init_weights(model, init_type=c.trainer.init_type, init_gain=c.trainer.init_gain)
    except AttributeError:
        print('WARNING: config does not contains init_type, skipping init_weight...')

    # checkpoint configuration
    if c.trainer.only_save_topK:
        checkpoint_kwargs = dict(
            save_top_k=3,
            filename='{epoch}-{step}-{%s:.%df}' % (c.trainer.moniter, c.trainer.metric_decimal),
            monitor=c.trainer.moniter,
            mode=c.trainer.moniter_mode
        )
    else:
        checkpoint_kwargs = dict(
            every_n_epochs=c.trainer.every_n_epochs
        )
    checkpoint_callback = ModelCheckpoint(
        dirpath='%s/checkpoints/' % c.work_dir,
        verbose=True,
        save_last=True,
        **checkpoint_kwargs
    )

    # modify default logger in pl
    logger = TensorBoardLogger(
        save_dir=c.work_dir,
        version='tb_logs',
        name='lightning_logs'
    )

    # most basic trainer, uses good defaults
    trainer = pl.Trainer(
        gpus=c.trainer.gpus,
        logger=logger,
        accelerator=args.acc,
        plugins=[DDPPlugin(find_unused_parameters=True)] if args.acc == 'ddp' else None,
        default_root_dir=c.work_dir,
        resume_from_checkpoint=args.resume,
        callbacks=[checkpoint_callback],
        max_epochs=c.trainer.n_epochs,
        fast_dev_run=c.trainer.fast_dev_run,
        log_every_n_steps=c.trainer.log_interval,
        flush_logs_every_n_steps=c.trainer.log_interval,  # save logs to file/web every n rows
        limit_val_batches=c.trainer.limit_val_batches,
        val_check_interval=c.trainer.val_check_interval if not args.debug else 1000,
        check_val_every_n_epoch=c.trainer.check_val_every_n_epoch if hasattr(c.trainer, 'check_val_every_n_epoch') else 1
    )
    trainer.fit(model)


if __name__ == '__main__':
    main_st = time.time()
    main()
    print('\nTraining time: {:0>8}'.format(str(datetime.timedelta(seconds=round(time.time() - main_st)))))
