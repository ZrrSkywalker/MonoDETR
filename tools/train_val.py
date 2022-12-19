import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
import warnings

warnings.filterwarnings("ignore")

import os
from pathlib import Path
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import argparse
import datetime

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.regularization_helper import build_regularization
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed
from utils import misc


@record
def main(args):
    assert (os.path.exists(args.config))
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    set_random_seed(cfg.get('random_seed', 444))

    model_name = cfg['model_name']
    output_path = os.path.join('./' + cfg["trainer"]['save_path'], model_name)
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, args.config.name), 'w') as f:
        yaml.dump(cfg, f)

    log_file = os.path.join(output_path, 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    misc.init_distributed_mode(args)
    logger = create_logger(log_file, rank=misc.get_rank())
    logger.info(f'rank: {misc.get_rank()}')

    # build dataloader
    train_loader, test_loader = build_dataloader(cfg['dataset'])

    # build model
    model, loss = build_model(cfg['model'], cfg['loss'])
    device = torch.device("cuda", index=misc.get_rank())
    model, loss = model.to(device), loss.to(device)

    model = DDP(model, device_ids=[device])

    if args.evaluate_only:
        logger.info('###################  Evaluation Only  ##################')
        tester = Tester(cfg=cfg['tester'],
                        model=model,
                        dataloader=test_loader,
                        logger=logger,
                        device=device,
                        train_cfg=cfg['trainer'],
                        model_name=model_name)
        tester.test()
        return

    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], model)

    # build lr scheduler
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    # build regularization. regularization can be null.
    regularization = build_regularization(cfg['regularization']) if 'regularization' in cfg else None

    trainer = Trainer(cfg=cfg['trainer'],
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      regularization=regularization,
                      logger=logger,
                      device=device,
                      loss=loss,
                      model_name=model_name)

    tester = Tester(cfg=cfg['tester'],
                    model=trainer.model,
                    dataloader=test_loader,
                    logger=logger,
                    device=device,
                    train_cfg=cfg['trainer'],
                    model_name=model_name)
    if cfg['dataset']['test_split'] != 'test':
        trainer.tester = tester

    logger.info('###################  Training  ##################')
    logger.info('Batch Size per GPU: %d' % (cfg['dataset']['batch_size']))
    logger.info('Learning Rate: %f' % (cfg['optimizer']['lr']))

    trainer.train()

    if cfg['dataset']['test_split'] == 'test':
        return

    logger.info('###################  Testing  ##################')
    logger.info('Batch Size per GPU: %d' % (cfg['dataset']['batch_size']))
    logger.info('Split: %s' % (cfg['dataset']['test_split']))

    tester.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth-aware Transformer for Monocular 3D Object Detection')
    parser.add_argument('-c', '--config', type=Path, help='settings of detection in yaml format')
    parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')
    parser.add_argument('--local_rank', type=int)
    args = parser.parse_args()

    main(args)
