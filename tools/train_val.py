import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch
import wandb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import argparse
import datetime


import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import utils
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from dataclasses import dataclass


from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed


# parser = argparse.ArgumentParser(description='Depth-aware Transformer for Monocular 3D Object Detection')
# parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
# parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')
# args = parser.parse_args()

# Define a dataclass for each group of parameters



@hydra.main(config_path="../configs", config_name="monodetr")
def main(cfg : DictConfig) -> None:
    set_random_seed(cfg.get('random_seed', 444))

    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    cfg_wandb = cfg
    if cfg['evaluate_only']:
        model_name = cfg['model_name']
        cfg_wandb['model_name'] = "Val_" + model_name
    else:
        model_name = cfg['model_name'] + "_" + date_time


    output_path = os.path.join(cfg["trainer"]['save_path'], model_name)
    os.makedirs(output_path, exist_ok=True)

    log_file = os.path.join(output_path, 'train.log.%s' % date_time)
    logger = create_logger(log_file)

    logger.info(OmegaConf.to_yaml(cfg_wandb))

    # ✨ W&B: setup
    # wandb_cfg = {
    #     "epochs": cfg['trainer']['max_epoch'],
    #     "learning_rate": cfg['optimizer']['lr'],
    #     "batch_size": cfg['dataset']['batch_size'],
    #     "seed": cfg.get('random_seed', 444),
    #     "model": model_name,
    #     "dataset": cfg['dataset']['root_dir'],
    #     "optimizer": cfg['optimizer']['type'],
    #     "scheduler": cfg['lr_scheduler']['type'],
    #     "depth_guidance": cfg['model']['depth_guidance'],
    # }

    wandb_cfg = OmegaConf.to_container(cfg_wandb, resolve=True)
    wandb.init(
        project="MonoDETR",
        entity="adlcv",
        config=wandb_cfg,
        job_type="train",
        name=cfg_wandb['model_name'],
        dir=output_path,
    )
    
    # build dataloader
    train_loader, test_loader = build_dataloader(cfg['dataset'])

    # build model
    model, loss = build_model(cfg['model'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_ids = list(map(int, cfg['trainer']['gpu_ids'].split(',')))

    if len(gpu_ids) == 1:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).to(device)

    if cfg['evaluate_only']:
        logger.info('###################  Evaluation Only  ##################')
        tester = Tester(cfg=cfg['tester'],
                        model=model,
                        dataloader=test_loader,
                        logger=logger,
                        train_cfg=cfg['trainer'],
                        model_name=model_name)
        tester.test()
        return
    #ipdb.set_trace()
    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], model)
    # build lr scheduler
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    trainer = Trainer(cfg=cfg['trainer'],
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      loss=loss,
                      model_name=model_name)

    tester = Tester(cfg=cfg['tester'],
                    model=trainer.model,
                    dataloader=test_loader,
                    logger=logger,
                    train_cfg=cfg['trainer'],
                    model_name=model_name)
    if cfg['dataset']['test_split'] != 'test':
        trainer.tester = tester

    logger.info('###################  Training  ##################')
    logger.info('Batch Size: %d' % (cfg['dataset']['batch_size']))
    logger.info('Learning Rate: %f' % (cfg['optimizer']['lr']))

    trainer.train()

    if cfg['dataset']['test_split'] == 'test':
        return

    logger.info('###################  Testing  ##################')
    logger.info('Batch Size: %d' % (cfg['dataset']['batch_size']))
    logger.info('Split: %s' % (cfg['dataset']['test_split']))

    tester.test()


if __name__ == '__main__':
    main()
