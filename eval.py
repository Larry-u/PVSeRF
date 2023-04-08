import datetime
import time
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import pytorch_lightning as pl
import colored_traceback
from pytorch_lightning.loggers import TensorBoardLogger

import pl_models
from utils import create_logdir, merge_from_sys_argv


def main():
    parser = ArgumentParser()
    parser.add_argument("--resume", '-r', required=True, default=None, help="resume path")
    parser.add_argument('--debug', '-d', default=False, action='store_true', help='set debugging mode, shorten epochs')
    parser.add_argument('--verbose', '-v', default=False, action='store_true', help='if print config content')
    parser.add_argument("--remarks", '-m', type=str, default='', help="remarks added to the end of test_dir name")
    parser.add_argument("--split", '-s', type=str, default='test', help="which split to eval")
    parser.add_argument("--outroot", '-o', type=str, default='./test_results', help="root dir of test results")
    parser.add_argument('--vis_geometry', default=False, action='store_true',
                        help='if visualize geometry (pcloud/voxel)')
    parser.add_argument('--reload', default=False, action='store_true',
                        help='load original pretrained voxel-net weights')

    parser.add_argument(
        "--source", "-P", type=str, default="2",
        help="Source view(s) for each object. Alternatively, specify -L to viewlist file and leave this blank.",
    )
    parser.add_argument(
        "--viewlist", "-L", type=str, default="viewlist/src_dvr.txt",
        help="Path to source view list e.g. src_dvr.txt; if specified, overrides source/P",
    )
    parser.add_argument(
        "--include_src", action="store_true", help="Include source views in calculation"
    )
    parser.add_argument("--write_depth", action="store_true", help="Write depth image")
    parser.add_argument(
        "--write_compare", action="store_true", help="Write GT comparison image"
    )
    parser.add_argument(
        "--free_pose", action="store_true",
        help="Set to indicate poses may change between objects. In most of our datasets, the test set has fixed poses.",
    )
    parser.add_argument(
        "--no_compare_gt", action="store_true",
        help="Skip GT comparison (metric won't be computed) and only render images",
    )
    parser.add_argument(
        "--multicat", action="store_true",
        help="Prepend category id to object id. Specify if model fits multiple categories.",
    )
    parser.add_argument(
        "--eval_ray_batch_size", "-R", type=int, default=200_000, help="Ray batch size",
    )
    parser.add_argument(
        "--eval_pts_batch_size", type=int, default=300_000, help="Ray batch size",
    )
    parser.add_argument(
        "--limit_test_batches", type=float, default=1.0,
        help="How much of test dataset to check. 0.25 means 25%, 1000 means 1000 batches"
    )

    args, _ = parser.parse_known_args()

    if args.debug:
        print('WARNING: debug mode on')

    # create log dir and dump config into json
    assert args.resume, 'The resume path is required!'

    c = create_logdir(resume_path=args.resume, args=args)
    merge_from_sys_argv(c)
    c.renderer.eval_batch_size = args.eval_pts_batch_size

    # don't load pretrained weights because these weights are already stored in ckpt
    if not args.vis_geometry:
        for ckpt_type in ['vox_encoder_ckpt', 'graphx_ckpt']:
            if hasattr(c.model, ckpt_type):
                setattr(c.model, ckpt_type, None)

    print(c)

    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED. Ensure reproducibility
    pl.seed_everything(c.trainer.rnd_seed)

    # init model
    model = getattr(pl_models, c.trainer.model).load_from_checkpoint(
        checkpoint_path=args.resume,
        cfg=c
    )
    model.test_init(args)

    # temp code (use original voxel weights to overwrite latest weights)
    if args.vis_geometry and args.reload:
        print('INFO: Loading original encoder weights...')
        state_dict = torch.load(c.model.vox_encoder_ckpt)
        rename_dict = lambda dict: OrderedDict([(key.split("module.")[-1], dict[key]) for key in dict])

        model.nerf_net.voxel_encoder.encoder.load_state_dict(rename_dict(state_dict['encoder_state_dict']))
        model.nerf_net.voxel_encoder.decoder.load_state_dict(rename_dict(state_dict['decoder_state_dict']))

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
        accelerator='ddp',
        default_root_dir=c.work_dir,
        resume_from_checkpoint=args.resume,
        log_every_n_steps=c.trainer.log_interval,
        flush_logs_every_n_steps=c.trainer.log_interval,  # save logs to file/web every n rows
        limit_test_batches=args.limit_test_batches
    )
    trainer.test(model)


if __name__ == '__main__':
    colored_traceback.add_hook(always=True)

    main_st = time.time()
    main()
    print('\nTesting time: {:0>8}'.format(str(datetime.timedelta(seconds=round(time.time() - main_st)))))
