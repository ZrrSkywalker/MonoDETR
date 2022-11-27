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
        self.model_name = model_name
        self.output_dir = os.path.join('./' + cfg['save_path'], model_name)
        self.tester = None
        self.with_tensorboard = with_tensorboard
        if with_tensorboard:
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
                        self.logger.info("Test Epoch {}".format(self.epoch))
                        self.tester.inference()
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
                            self.logger.info("Best Result: {}, epoch: {}".format(best_result, best_epoch))
                            if self.with_tensorboard:
                                self.log_tensorboard(result_dict, global_step=self.epoch, tag='val')

        self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))

        return None

    def train_one_epoch(self, epoch):
        torch.set_grad_enabled(True)
        self.train_loader.sampler.set_epoch(epoch)
        self.model.train()
        # print(">>>>>>> Epoch:", str(epoch) + ":")
        self.logger.info(f'Train Epoch {epoch + 1:03}/{self.cfg["max_epoch"]}')

        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=(self.epoch + 1 == self.cfg['max_epoch']), desc='iters')
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            for key in targets:
                targets[key] = targets[key].to(self.device)
            img_sizes = targets['img_size']
            targets = self.prepare_targets(targets, inputs.shape[0])

            # train one batch
            self.optimizer.zero_grad()
            outputs = self.model(inputs, calibs, img_sizes)

            detr_losses_dict: Dict[str, torch.Tensor] = self.detr_loss(outputs, targets)

            detr_losses = torch.stack(list(detr_losses_dict.values())).sum()
            # weight_dict = self.detr_loss.weight_dict
            # detr_losses_dict_weighted = [detr_losses_dict[k] * weight_dict[k] for k in detr_losses_dict.keys() if k in weight_dict]
            # detr_losses = torch.stack(detr_losses_dict_weighted).sum()

            detr_losses_dict = misc.reduce_dict(detr_losses_dict)
            detr_losses_dict_log = {loss_name: loss.item() for loss_name, loss in detr_losses_dict.items()}
            detr_losses_dict_log['loss_detr'] = detr_losses.item()

            # detr_losses_dict = misc.reduce_dict(detr_losses_dict)
            # detr_losses_log = 0
            # for k in detr_losses_dict.keys():
            #     if k in weight_dict:
            #         detr_losses_dict_log[k] = (detr_losses_dict[k] * weight_dict[k]).item()
            #         detr_losses_log += detr_losses_dict_log[k]
            # detr_losses_dict_log["loss_detr"] = detr_losses_log

            # flags = [True] * 5
            if batch_idx % 30 == 0:
                # print("----", batch_idx, "----")
                # print("%s: %.2f, " % ("loss_detr", detr_losses_dict_log["loss_detr"]))
                # for key, val in detr_losses_dict_log.items():
                #     if key == "loss_detr":
                #         continue
                #     if "0" in key or "1" in key or "2" in key or "3" in key or "4" in key or "5" in key:
                #         if flags[int(key[-1])]:
                #             print("")
                #             flags[int(key[-1])] = False
                #     print("%s: %.2f, " % (key, val), end="")
                # print("")
                # print("")

                if self.with_tensorboard:
                    self.log_tensorboard(
                        detr_losses_dict_log,
                        global_step=epoch * len(self.train_loader) + batch_idx,
                        tag='train')

            detr_losses.backward()
            self.optimizer.step()

            progress_bar.update()
        progress_bar.close()

    # def validate(self, )
    def prepare_targets(self, targets: Dict[str, torch.Tensor], batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Organizes targets to list of dicts.

        Args:
            targets: A dict with following keys:
                * 'calibs'
                * 'indices'
                * 'img_size'
                * 'labels'
                * 'boxes'
                * 'boxes_3d'
                * 'depth'
                * 'size_2d'
                * 'size_3d'
                * 'src_size_3d'
                * 'heading_bin'
                * 'heading_res'
                * 'mask_2d'
                Each value of the key is a tensor.

        Returns:
            A list of dicts of tensors.
            [{batch_0_dict}, {batch_1_dict}, ...]
        """
        targets_list = []
        mask = targets['mask_2d']

        key_list = ['labels', 'boxes', 'calibs', 'depth', 'size_3d', 'heading_bin', 'heading_res', 'boxes_3d']
        for bz in range(batch_size):
            target_dict = {}
            for key, val in targets.items():
                if key in key_list:
                    target_dict[key] = val[bz][mask[bz]]
            targets_list.append(target_dict)
        return targets_list

    def log_tensorboard(self, log_dict: Dict[str, Number], global_step: int, tag: str = ''):
        if not misc.is_main_process():
            return
        for key, value in log_dict.items():
            name = f'{tag}/{key}' if tag else key
            self.writer.add_scalar(name, value, global_step=global_step)
