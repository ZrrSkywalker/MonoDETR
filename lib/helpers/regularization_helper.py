from typing import Any, Dict, List, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.types import Number

from utils import misc


class Regularization(nn.Module):
    def __init__(self, loss_names: List[str], weight_dict: Dict[str, float], loss_args: Dict[str, Any] = {}):
        super().__init__()
        self.loss_names = loss_names
        self.weight_dict = weight_dict
        self.loss_args = loss_args
        self.loss_functions = {
            'depth_embed_regularization': self.depth_embed_regularization
        }
        assert len(self.weight_dict) == len(self.loss_names), f'The length of `weight_dict`({len(self.weight_dict)}) and `loss_names`({len(self.loss_names)}) should be consistent.'

    def depth_embed_regularization(self, model, num_group_members: int = 5, margin: float = 1., p: int = 1) -> torch.Tensor:
        weight: torch.Tensor = model.depth_predictor.depth_pos_embed.weight
        num_group_members = num_group_members
        loss = weight.new_zeros(1)
        num_samples = 0
        for i in range(0, len(weight), num_group_members):
            # [N, 1, 1, 256]
            anchors = weight[i:i + num_group_members][:, None, None]
            # [1, N, 1, 256]
            positives = weight[i:i + num_group_members][None, :, None]
            mask = weight.new_ones(weight.shape[0], dtype=torch.bool)
            mask[i:i + num_group_members] = False
            # [1, 1, len(weight) - N, 256]
            negatives = weight[mask][None, None, :]
            anchors, positives, negatives = torch.broadcast_tensors(anchors, positives, negatives)
            anchors = anchors.flatten(0, -2)
            positives = positives.flatten(0, -2)
            negatives = negatives.flatten(0, -2)
            loss += F.triplet_margin_loss(anchors, positives, negatives, margin=margin, p=p, reduction='sum')
            num_samples += anchors.shape[0]

        loss /= num_samples
        return loss

    def get_loss(self, loss_name, model) -> torch.Tensor:
        assert loss_name in self.loss_functions, f'"{loss_name} loss is not found."'
        return self.loss_functions[loss_name](model, **self.loss_args.get(loss_name, {}))

    def forward(self, model) -> Tuple[Dict[str, torch.Tensor], Dict[str, Number]]:
        if isinstance(model, DDP):
            model = model.module
        loss_dict, unweighted_loss_log_dict = {}, {}
        for loss_name in self.loss_names:
            loss = self.get_loss(loss_name, model)
            unweighted_loss_log_dict[loss_name] = loss
            loss_dict[loss_name] = loss * self.weight_dict[loss_name]

        unweighted_loss_log_dict = misc.reduce_dict(unweighted_loss_log_dict)
        unweighted_loss_log_dict = {loss_name: loss.item() for loss_name, loss in unweighted_loss_log_dict.items()}
        return loss_dict, unweighted_loss_log_dict


def build_regularization(cfg):
    return Regularization(loss_names=cfg['losses'], weight_dict=cfg['weights'], loss_args=cfg.get('args', {}))
