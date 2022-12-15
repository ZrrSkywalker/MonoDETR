import os
from typing import Dict, Optional, Tuple
import tqdm
import logging

import torch
from torch import nn
from torch.types import Number
from torch.utils.data import DataLoader
from lib.helpers.dataloader_helper import prepare_targets
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
import time


class Tester(object):
    def __init__(self,
                 cfg,
                 model: nn.Module,
                 dataloader: DataLoader,
                 logger: logging.Logger,
                 device: torch.device,
                 train_cfg,
                 model_name: str):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.class_name = dataloader.dataset.class_name
        self.output_dir = os.path.join('./' + train_cfg['save_path'], model_name)
        self.dataset_type = cfg.get('type', 'KITTI')
        self.device = device
        self.logger = logger
        self.train_cfg = train_cfg
        self.model_name = model_name
        self.epoch = 0

    def test(self):
        assert self.cfg['mode'] in ['single', 'all']

        # test a single checkpoint
        if self.cfg['mode'] == 'single' or not self.train_cfg["save_all"]:
            if self.train_cfg["save_all"]:
                checkpoint_path = os.path.join(self.output_dir, "checkpoint_epoch_{}.pth".format(self.cfg['checkpoint']))
            else:
                checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")
            assert os.path.exists(checkpoint_path)
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=checkpoint_path,
                            map_location=self.device,
                            logger=self.logger)
            self.model.to(self.device)
            self.inference()
            self.evaluate()

        # test all checkpoints in the given dir
        elif self.cfg['mode'] == 'all' and self.train_cfg["save_all"]:
            start_epoch = int(self.cfg['checkpoint'])
            checkpoints_list = []
            for _, _, files in os.walk(self.output_dir):
                for f in files:
                    if f.endswith(".pth") and int(f[17:-4]) >= start_epoch:
                        checkpoints_list.append(os.path.join(self.output_dir, f))
            checkpoints_list.sort(key=os.path.getmtime)

            for checkpoint in checkpoints_list:
                load_checkpoint(model=self.model,
                                optimizer=None,
                                filename=checkpoint,
                                map_location=self.device,
                                logger=self.logger)
                self.model.to(self.device)
                self.inference()
                self.evaluate()

    @torch.no_grad()
    def inference(self, loss: Optional[nn.Module] = None, return_loss: bool = False) -> Dict[str, Number]:
        self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        self.model.eval()

        results = {}
        model_infer_time = 0
        log_dict = {}
        for batch_idx, (inputs, calibs, targets, info) in enumerate(tqdm.tqdm(self.dataloader, dynamic_ncols=True, desc='Evaluation Progress')):
            # load evaluation data and move data to GPU.
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            img_sizes = info['img_size'].to(self.device)

            start_time = time.time()
            outputs = self.model(inputs, calibs, img_sizes)
            if loss and return_loss:
                for key in targets:
                    targets[key] = targets[key].to(self.device)
                targets = prepare_targets(targets, inputs.shape[0])

                loss_dict, unweighted_loss_log_dict = loss(outputs, targets)
                loss_detr = torch.stack(list(loss_dict.values())).sum()
                unweighted_loss_log_dict['loss_detr'] = loss_detr.item()

                for loss_name, value in unweighted_loss_log_dict.items():
                    log_dict[loss_name] = log_dict.get(loss_name, 0) + value * inputs.shape[0]

            end_time = time.time()
            model_infer_time += end_time - start_time

            dets = extract_dets_from_outputs(outputs=outputs, topk=self.cfg['topk'])

            dets = dets.detach().cpu().numpy()

            # get corresponding calibs & transform tensor to numpy
            calibs = [self.dataloader.dataset.get_calib(index) for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            cls_mean_size = self.dataloader.dataset.cls_mean_size
            dets = decode_detections(
                dets=dets,
                info=info,
                calibs=calibs,
                cls_mean_size=cls_mean_size,
                threshold=self.cfg.get('threshold', 0.2))

            results.update(dets)

        for loss_name, value in log_dict.items():
            log_dict[loss_name] /= len(self.dataloader.dataset)
        self.logger.info(f'Inference on {len(self.dataloader)} images by {model_infer_time / len(self.dataloader)}/per image.')

        # save the result for evaluation.
        self.logger.info(f"==> Saving to {os.path.join(self.output_dir, 'outputs', 'data')}...")
        self.save_results(results)
        return log_dict

    def save_results(self, results):
        output_dir = os.path.join(self.output_dir, 'outputs', 'data')
        os.makedirs(output_dir, exist_ok=True)

        for img_id in results.keys():
            if self.dataset_type == 'KITTI':
                output_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            else:
                os.makedirs(os.path.join(output_dir, self.dataloader.dataset.get_sensor_modality(img_id)), exist_ok=True)
                output_path = os.path.join(output_dir,
                                           self.dataloader.dataset.get_sensor_modality(img_id),
                                           self.dataloader.dataset.get_sample_token(img_id) + '.txt')

            f = open(output_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()

    def evaluate(self) -> Tuple[Dict[str, float], float]:
        results_dir = os.path.join(self.output_dir, 'outputs', 'data')
        assert os.path.exists(results_dir)
        result_dict, car_moderate = self.dataloader.dataset.eval(results_dir=results_dir, logger=self.logger)
        return result_dict, car_moderate
