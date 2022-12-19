import os
import time
from typing import Dict, List, Optional, Union
import tqdm
import logging

import torch
from torch.types import Number
from torch import nn
from torch.utils.data import DataLoader
import torch.distributed as dist
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from lib.helpers.dataloader_helper import prepare_targets
from lib.helpers.regularization_helper import Regularization
from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint
from lib.models.monodetr.monodetr import SetCriterion
from utils import misc


class Trainer(object):
    def __init__(self,
                 cfg,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger: logging.Logger,
                 device: torch.device,
                 loss: SetCriterion,
                 regularization: Optional[Regularization],
                 model_name: str,
                 with_tensorboard: bool = True):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.best_result = 0
        self.best_epoch = 0
        self.device = device
        self.detr_loss = loss
        self.regularization = regularization
        self.model_name = model_name
        self.output_dir = os.path.join('./' + cfg['save_path'], model_name)
        self.tester = None
        self.with_tensorboard = with_tensorboard and misc.is_main_process()
        if self.with_tensorboard:
            self.writer = SummaryWriter(self.output_dir)

        # loading pretrain/resume model
        if cfg.get('pretrain_model'):
            assert os.path.exists(cfg['pretrain_model'])
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=cfg['pretrain_model'],
                            map_location=self.device,
                            logger=self.logger)

        if cfg.get('resume_model', None):
            resume_model_path = os.path.join(self.output_dir, "checkpoint.pth")
            assert os.path.exists(resume_model_path)
            self.epoch, self.best_result, self.best_epoch = load_checkpoint(
                model=self.model.to(self.device),
                optimizer=self.optimizer,
                filename=resume_model_path,
                map_location=self.device,
                logger=self.logger)
            self.lr_scheduler.last_epoch = self.epoch - 1
            self.logger.info("Loading Checkpoint... Best Result:{}, Best Epoch:{}".format(self.best_result, self.best_epoch))

    def train(self):
        start_epoch = self.epoch

        best_result = self.best_result
        best_epoch = self.best_epoch
        if misc.is_main_process():
            progress = tqdm.trange(start_epoch, self.cfg['max_epoch'], total=self.cfg['max_epoch'], initial=start_epoch, dynamic_ncols=True, leave=True, desc='epochs')
        else:
            progress = range(start_epoch, self.cfg['max_epoch'])
        for epoch in progress:
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + self.epoch)

            # wait all processes to start training
            dist.barrier()
            # train one epoch
            self.train_one_epoch(epoch)
            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()
            if self.with_tensorboard:
                self.log_tensorboard({'learning_rate': self.optimizer.param_groups[0]['lr']}, global_step=self.epoch)

            # save trained model
            if self.epoch % self.cfg['save_frequency'] == 0:
                with torch.no_grad():
                    if misc.is_main_process():
                        os.makedirs(self.output_dir, exist_ok=True)
                        if self.cfg['save_all']:
                            ckpt_name = os.path.join(self.output_dir, 'checkpoint_epoch_%d' % self.epoch)
                        else:
                            ckpt_name = os.path.join(self.output_dir, 'checkpoint')
                        save_checkpoint(
                            get_checkpoint_state(self.model, self.optimizer, self.epoch, best_result, best_epoch),
                            ckpt_name)
                    if self.tester:
                        self.logger.info(f'Test Epoch {self.epoch}')
                        val_losses_log_dict = self.tester.inference(loss=self.detr_loss, return_loss=True)
                        dist.barrier()
                        if misc.is_main_process():
                            result_dict, cur_result = self.tester.evaluate()
                            if cur_result > best_result:
                                best_result = cur_result
                                best_epoch = self.epoch
                                ckpt_name = os.path.join(self.output_dir, 'checkpoint_best')
                                save_checkpoint(
                                    get_checkpoint_state(self.model, self.optimizer, self.epoch, best_result, best_epoch),
                                    ckpt_name)
                            self.logger.info(f'Best Result: {best_result}, epoch: {best_epoch}')
                            if self.with_tensorboard:
                                self.log_tensorboard(result_dict, global_step=self.epoch, tag='val')
                                self.log_tensorboard(val_losses_log_dict, global_step=self.epoch, tag='val')

        if misc.is_main_process():
            self.logger.info(f'Best Result: {best_result}, epoch: {best_epoch}')

        return None

    def train_one_epoch(self, epoch):
        torch.set_grad_enabled(True)
        self.train_loader.sampler.set_epoch(epoch)
        self.model.train()
        self.logger.info(f'Train Epoch {epoch + 1:03}/{self.cfg["max_epoch"]}')

        dataloader_len = len(self.train_loader)
        # log to tensorboard `self.cfg['log_frequency']` times per epoch
        steps_to_log = set(np.linspace(dataloader_len // self.cfg['log_frequency'], dataloader_len, min(dataloader_len, self.cfg['log_frequency']), dtype=int))
        for batch_idx, (inputs, calibs, targets, info) in enumerate(tqdm.tqdm(self.train_loader, leave=True, desc='iters')):
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            for key in targets:
                targets[key] = targets[key].to(self.device)
            img_sizes = targets['img_size']
            targets = prepare_targets(targets, inputs.shape[0])

            # train one batch
            self.optimizer.zero_grad()
            outputs = self.model(inputs, calibs, img_sizes)

            detr_losses_dict, unweighted_losses_log_dict = self.detr_loss(outputs, targets)
            detr_losses = torch.stack(list(detr_losses_dict.values())).sum()
            if self.regularization is not None:
                regularization_dict, unweighted_regularization_log_dict = self.regularization(self.model)
                detr_losses += torch.stack(list(regularization_dict.values())).sum()
            else:
                unweighted_regularization_log_dict = {}

            unweighted_losses_log_dict['loss_detr'] = detr_losses.item()

            if batch_idx in steps_to_log:
                if self.with_tensorboard:
                    global_step = epoch * 100 + int(batch_idx / len(self.train_loader) * 100)
                    self.log_tensorboard(
                        unweighted_losses_log_dict,
                        global_step=global_step,
                        tag='train')
                    self.log_tensorboard(
                        unweighted_regularization_log_dict,
                        global_step=global_step,
                        tag='train')

            detr_losses.backward()
            self.optimizer.step()

    def log_tensorboard(self, log_dict: Dict[str, Number], global_step: int, tag: str = ''):
        for key, value in log_dict.items():
            name = f'{tag}/{key}' if tag else key
            self.writer.add_scalar(name, value, global_step=global_step)
