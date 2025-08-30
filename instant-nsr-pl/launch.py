import sys
import argparse
import os
import time
import logging
from datetime import datetime
import numpy as np
import utils.camera as camera
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='path to config file')
    parser.add_argument('--gpu', default='0', help='GPU(s) to be used')
    parser.add_argument('--resume', default=None, help='path to the weights to be resumed')
    parser.add_argument(
        '--resume_weights_only',
        action='store_true',
        help='specify this argument to restore only the weights (w/o training states), e.g. --resume path/to/resume --resume_weights_only'
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--validate', action='store_true')
    group.add_argument('--test', action='store_true')
    group.add_argument('--save_pose', action='store_true')
    group.add_argument('--predict', action='store_true')
    

    parser.add_argument('--exp_dir', default='./exp')
    parser.add_argument('--runs_dir', default='./runs')
    parser.add_argument('--verbose', action='store_true', help='if true, set logging level to DEBUG')
    args, extras = parser.parse_known_args()

    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(args.gpu.split(','))

    import datasets
    import systems
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
    from utils.callbacks import CodeSnapshotCallback, ConfigSnapshotCallback, CustomProgressBar
    from utils.misc import load_config    

    # parse YAML config to OmegaConf
    config = load_config(args.config, cli_args=extras)
    config.cmd_args = vars(args)

    config.trial_name = config.get('trial_name') or (config.tag + datetime.now().strftime('@%Y%m%d-%H%M%S'))
    config.exp_dir = config.get('exp_dir') or os.path.join(args.exp_dir, config.name)
    config.save_dir = config.get('save_dir') or os.path.join(config.exp_dir, config.trial_name, 'save')
    config.ckpt_dir = config.get('ckpt_dir') or os.path.join(config.exp_dir, config.trial_name, 'ckpt')
    config.code_dir = config.get('code_dir') or os.path.join(config.exp_dir, config.trial_name, 'code')
    config.config_dir = config.get('config_dir') or os.path.join(config.exp_dir, config.trial_name, 'config')

    logger = logging.getLogger('pytorch_lightning')
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if 'seed' not in config:
        config.seed = int(time.time() * 1000) % 1000
    pl.seed_everything(config.seed)

    if hasattr(config, 'use_barf') and config.use_barf:
        config.model.n_frames = (config.dataset.end_idx - config.dataset.start_idx) // config.dataset.step + int((config.dataset.end_idx - config.dataset.start_idx) % config.dataset.step != 0)
        if args.save_pose:
            args.resume_weights_only = True

    dm = datasets.make(config.dataset.name, config.dataset)
    system = systems.make(config.system.name, config, load_from_checkpoint=None if not args.resume_weights_only else args.resume)
    if hasattr(config, 'load_coarse_geo') and config.load_coarse_geo:
        coarse_geo_weights = torch.load(os.path.dirname(config.dataset.obj_path).replace('save', 'ckpt') + '/epoch=0-step=5000.ckpt')
        system.load_state_dict(coarse_geo_weights['state_dict'], strict=False)
    callbacks = []
    if args.train:
        callbacks += [
            ModelCheckpoint(
                dirpath=config.ckpt_dir,
                **config.checkpoint
            ),
            LearningRateMonitor(logging_interval='step'),
            CodeSnapshotCallback(
                config.code_dir, use_version=False
            ),
            ConfigSnapshotCallback(
                config, config.config_dir, use_version=False
            ),
            CustomProgressBar(refresh_rate=1),
        ]

    loggers = []
    if args.train:
        loggers += [
            TensorBoardLogger(os.path.join(args.runs_dir, args.exp_dir.split('/')[-1]), name=config.name, version=config.trial_name),
            CSVLogger(config.exp_dir, name=config.trial_name, version='csv_logs')
        ]
    
    if sys.platform == 'win32':
        # does not support multi-gpu on windows
        strategy = 'dp'
        assert n_gpus == 1
    else:
        strategy = 'ddp_find_unused_parameters_false'
    
    trainer = Trainer(
        devices=n_gpus,
        accelerator='gpu',
        callbacks=callbacks,
        logger=loggers,
        strategy=None,
        **config.trainer)

    if args.train:
        if args.resume and not args.resume_weights_only:
            # FIXME: different behavior in pytorch-lighting>1.9 ?
            trainer.fit(system, datamodule=dm, ckpt_path=args.resume)
        else:
            trainer.fit(system, datamodule=dm)
        trainer.test(system, datamodule=dm)
        # save the final poses
        c2w = dm.train_dataset.all_c2w.cpu().clone()
        c2w_refine = camera.lie.se3_to_SE3(system.se3_refine.weight)
        c2w = camera.pose.compose([c2w_refine, c2w])
        w2c = camera.pose.invert(c2w)
        w2c[:, 1:] *= -1
        c2w = camera.pose.invert(w2c)
        np.savetxt(os.path.join(config.save_dir, 'poses.txt'), c2w.reshape(-1, 12).detach().cpu().numpy())
    elif args.validate:
        trainer.fit(system, datamodule=dm, ckpt_path=args.resume)
        trainer.validate(system, datamodule=dm, ckpt_path=args.resume)
    elif args.test:
        trainer.fit(system, datamodule=dm, ckpt_path=args.resume)
        trainer.test(system, datamodule=dm, ckpt_path=args.resume)
    elif args.predict:
        trainer.predict(system, datamodule=dm, ckpt_path=args.resume)
    elif args.save_pose:
        dm.setup('predict')
        c2w = dm.predict_dataset.all_c2w.cpu()
        c2w_refine = camera.lie.se3_to_SE3(system.se3_refine.weight)
        c2w = camera.pose.compose([c2w_refine, c2w])
        w2c = camera.pose.invert(c2w)
        w2c[:, 1:] *= -1
        c2w = camera.pose.invert(w2c)
        np.savetxt(os.path.join(config.save_dir, 'poses.txt'), c2w.reshape(-1, 12).detach().cpu().numpy())

if __name__ == '__main__':
    main()
