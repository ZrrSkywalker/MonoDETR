from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset


# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def prepare_targets(targets: Dict[str, torch.Tensor], batch_size: int) -> List[Dict[str, torch.Tensor]]:
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

    key_list = ['labels', 'boxes', 'calibs', 'depth', 'size_3d', 'src_size_3d', 'heading_bin', 'heading_res', 'boxes_3d']
    for bz in range(batch_size):
        target_dict = {}
        for key, val in targets.items():
            if key in key_list:
                target_dict[key] = val[bz][mask[bz]]
        targets_list.append(target_dict)
    return targets_list


def build_dataloader(cfg):
    # perpare dataset
    if cfg['type'] == 'KITTI':
        train_set = KITTI_Dataset(split=cfg['train_split'], cfg=cfg)
        test_set = KITTI_Dataset(split=cfg['test_split'], cfg=cfg)
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])

    # prepare dataloader
    train_loader = DataLoader(dataset=train_set,
                              batch_size=cfg['batch_size'],
                              num_workers=cfg['num_workers'],
                              worker_init_fn=my_worker_init_fn,
                              shuffle=False,
                              pin_memory=False,
                              drop_last=False,
                              sampler=DistributedSampler(train_set))
    test_loader = DataLoader(dataset=test_set,
                             batch_size=cfg['batch_size'],
                             num_workers=cfg['num_workers'],
                             worker_init_fn=my_worker_init_fn,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False,
                             sampler=DistributedSampler(test_set))

    return train_loader, test_loader
