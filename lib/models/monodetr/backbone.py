# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict
from typing import Dict, List

import timm
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from utils.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, images):
        xs = self.body(images)
        out = {}
        for name, x in xs.items():
            m = torch.zeros(x.shape[0], x.shape[2], x.shape[3]).to(torch.bool).to(x.device)
            out[name] = NestedTensor(x, m)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class DINOv3Backbone(nn.Module):
    """ConvNeXt backbone pretrained with DINOv3, loaded via timm.

    Outputs three feature maps at strides [8, 16, 32] — identical contract
    to the ResNet-50 backbone — so the rest of MonoDETR requires no changes.

    Args:
        model_name: timm model identifier, e.g. 'convnext_tiny.dinov3_lvd1689m'.
        train_backbone: if True, train stages.1-3; stem and stages.0 are always frozen.
    """

    strides: List[int]
    num_channels: List[int]

    def __init__(self, model_name: str, train_backbone: bool):
        super().__init__()
        # out_indices=(1,2,3) → stages 1/2/3 → strides 8/16/32
        self.body = timm.create_model(
            model_name,
            pretrained=is_main_process(),
            features_only=True,
            out_indices=(1, 2, 3),
        )
        self.strides = list(self.body.feature_info.reduction())    # [8, 16, 32]
        self.num_channels = list(self.body.feature_info.channels())  # [192, 384, 768]

        # Stem and stages.0 are always frozen (not in outputs, just carries gradients).
        # stages.1-3 follow train_backbone flag.
        for name, param in self.body.named_parameters():
            trainable = train_backbone and not (
                name.startswith("stem") or name.startswith("stages.0")
            )
            param.requires_grad_(trainable)

    def forward(self, images: torch.Tensor) -> Dict[str, NestedTensor]:
        feats = self.body(images)
        out: Dict[str, NestedTensor] = {}
        for i, x in enumerate(feats):
            mask = torch.zeros(
                x.shape[0], x.shape[2], x.shape[3],
                dtype=torch.bool, device=x.device,
            )
            out[str(i)] = NestedTensor(x, mask)
        return out


class ViTDINOv3Backbone(nn.Module):
    """ViT backbone pretrained with DINOv3, loaded via timm.

    Uses an **Intermediate-block FPN** to provide genuine semantic AND spatial
    diversity across the three output levels — unlike Simple FPN (which derives
    all levels from the same last-block tensor).

    Block selection (auto-computed as thirds of the total depth):
        early  (≈ 1/3 depth) → Conv + 2× upsample   → stride  8  (local features)
        middle (≈ 2/3 depth) → Conv pass-through     → stride 16  (mid-level)
        last   (full depth)  → Conv + 2× downsample  → stride 32  (semantic)

    This follows the EVA-02 / ViTDet-FPN pattern: extract features at
    semantically increasing depths, then spatially adapt each to hit the
    required stride.

    Args:
        model_name: timm model identifier, e.g.
            'vit_small_plus_patch16_dinov3.lvd1689m'.
        train_backbone: if True, transformer blocks are trainable; patch_embed
            is always frozen.
    """

    strides: List[int]
    num_channels: List[int]

    def __init__(self, model_name: str, train_backbone: bool):
        super().__init__()
        # Determine total number of blocks to auto-select third/two-thirds/last.
        _probe = timm.create_model(model_name, pretrained=False, features_only=True)
        n_blocks = len(_probe.feature_info)
        del _probe

        # Block indices: early (~1/3), middle (~2/3), last.
        i_early  = n_blocks // 3 - 1
        i_mid    = 2 * n_blocks // 3 - 1
        i_last   = n_blocks - 1

        self.body = timm.create_model(
            model_name,
            pretrained=is_main_process(),
            features_only=True,
            out_indices=(i_early, i_mid, i_last),
        )
        embed_dim: int = self.body.feature_info.channels()[0]  # same for all ViT blocks

        # Spatial adaptation heads — each maps stride-16 to the target stride.
        # Conv2d(C, C, 1) applied first in every branch ensures independent
        # learned projections so gradients don't collapse the three levels.
        self.adapt_early = nn.Sequential(          # stride 16 → stride 8
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.adapt_mid  = nn.Conv2d(embed_dim, embed_dim, 1)  # stride 16 → stride 16
        self.adapt_last = nn.Sequential(           # stride 16 → stride 32
            nn.Conv2d(embed_dim, embed_dim, 3, stride=2, padding=1),
        )

        self.strides = [8, 16, 32]
        self.num_channels = [embed_dim, embed_dim, embed_dim]

        # patch_embed always frozen; transformer blocks follow train_backbone.
        for name, param in self.body.named_parameters():
            param.requires_grad_(train_backbone and not name.startswith('patch_embed'))

    def forward(self, images: torch.Tensor) -> Dict[str, NestedTensor]:
        f_early, f_mid, f_last = self.body(images)  # all [B, C, H/16, W/16]
        levels = [
            self.adapt_early(f_early),  # [B, C, H/8,  W/8 ]
            self.adapt_mid(f_mid),      # [B, C, H/16, W/16]
            self.adapt_last(f_last),    # [B, C, H/32, W/32]
        ]
        out: Dict[str, NestedTensor] = {}
        for i, x in enumerate(levels):
            mask = torch.zeros(
                x.shape[0], x.shape[2], x.shape[3],
                dtype=torch.bool, device=x.device,
            )
            out[str(i)] = NestedTensor(x, mask)
        return out


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, images):
        xs = self[0](images)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(cfg):
    position_embedding = build_position_encoding(cfg)
    backbone_name = cfg['backbone']

    if backbone_name.startswith('vit_') and ('dinov3' in backbone_name or 'dinov2' in backbone_name):
        backbone = ViTDINOv3Backbone(backbone_name, cfg['train_backbone'])
    elif backbone_name.startswith('convnext'):
        backbone = DINOv3Backbone(backbone_name, cfg['train_backbone'])
    else:
        return_interm_layers = cfg['masks'] or cfg['num_feature_levels'] > 1
        backbone = Backbone(backbone_name, cfg['train_backbone'], return_interm_layers, cfg['dilation'])

    return Joiner(backbone, position_embedding)

    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, images):
        xs = self.body(images)
        out = {}
        for name, x in xs.items():
            m = torch.zeros(x.shape[0], x.shape[2], x.shape[3]).to(torch.bool).to(x.device)
            out[name] = NestedTensor(x, m)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2
