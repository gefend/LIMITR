import argparse
import torch
import LIMITR
import datetime
import os


from dateutil import tz
from omegaconf import OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        metavar="base_config.yaml",
        help="paths to base config",
        required=True,
    )
    parser.add_argument(
        "--train", action="store_true", default=False, help="specify to train model"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="specify to test model"
        "By default run.py trains a model based on config file",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="Checkpoint path of the desired test model"
    )
    parser.add_argument("--random_seed", type=int, default=23, help="Random seed")
    parser = Trainer.add_argparse_args(parser)

    return parser


def main(cfg, args):
    # get datamodule
    dm = LIMITR.builder.build_data_module(cfg)

    # define lightning module
    model = LIMITR.builder.build_lightning_model(cfg, dm)

    # callbacks
    callbacks = [LearningRateMonitor(logging_interval="step")]
    if "checkpoint_callback" in cfg.lightning:
        checkpoint_callback = ModelCheckpoint(**cfg.lightning.checkpoint_callback)
        callbacks.append(checkpoint_callback)
    if "early_stopping_callback" in cfg.lightning:
        early_stopping_callback = EarlyStopping(**cfg.lightning.early_stopping_callback)
        callbacks.append(early_stopping_callback)
    if cfg.train.scheduler is not None:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    # logging
    if "logger" in cfg.lightning:
        logger_type = cfg.lightning.logger.pop("logger_type")
        logger_class = getattr(pl_loggers, logger_type)
        cfg.lightning.logger.name = f"{cfg.experiment_name}_{cfg.extension}"
        logger = logger_class(**cfg.lightning.logger)
        cfg.lightning.logger.logger_type = logger_type
    else:
        logger = None

    # setup pytorch-lightning trainer
    cfg.lightning.trainer.val_check_interval = args.val_check_interval
    cfg.lightning.trainer.auto_lr_find = args.auto_lr_find
    trainer_args = argparse.Namespace(**cfg.lightning.trainer)
    trainer = Trainer.from_argparse_args(
        args=trainer_args, deterministic=True, callbacks=callbacks, logger=logger
    )

    # learning rate finder
    if trainer_args.auto_lr_find is not False:
        lr_finder = trainer.tuner.lr_find(model, datamodule=dm)
        new_lr = lr_finder.suggestion()
        model.lr = new_lr
        print("=" * 80 + f"\nLearning rate updated to {new_lr}\n" + "=" * 80)

    if args.train:
        trainer.fit(model, dm)
    if args.test:
        ckpt = args.ckpt_path
        model = model.load_from_checkpoint(ckpt, strict=False)
        trainer.test(model=model, datamodule=dm)

    # save top weights paths to yaml
    if "checkpoint_callback" in cfg.lightning:
        ckpt_paths = os.path.join(
            cfg.lightning.checkpoint_callback.dirpath, "best_ckpts.yaml"
        )
        checkpoint_callback.to_yaml(filepath=ckpt_paths)


if __name__ == "__main__":

    # parse arguments
    parser = get_parser()
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    # loop over the number of independent training splits, defaults to 1 split
    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")

    # set directory names
    cfg.extension = timestamp
    cfg.output_dir = f"./data/output/{cfg.experiment_name}/{cfg.extension}"
    cfg.lightning.checkpoint_callback.dirpath = os.path.join(
        cfg.lightning.checkpoint_callback.dirpath,
        f"{cfg.experiment_name}/{cfg.extension}",
    )

    # create directories
    if not os.path.exists(cfg.lightning.logger.save_dir):
        os.makedirs(cfg.lightning.logger.save_dir)
    if not os.path.exists(cfg.lightning.checkpoint_callback.dirpath):
        os.makedirs(cfg.lightning.checkpoint_callback.dirpath)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    # save config
    config_path = os.path.join(cfg.output_dir, "config.yaml")
    with open(config_path, "w") as fp:
        OmegaConf.save(config=cfg, f=fp.name)

    main(cfg, args)
