"""
MonoDETR: Depth-aware Transformer for Monocular 3D Object Detection
"""
from typing import Dict, List, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from torch.types import Number
import torch.distributed as dist
import math
import copy
from lib.datasets.utils import class2angle

from utils import box_ops
from utils import misc
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                        accuracy, get_world_size, interpolate,
                        is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .depthaware_transformer import build_depthaware_transformer
from .depth_predictor import DepthPredictor
from .depth_predictor.ddn_loss import DDNLoss
from lib.losses.focal_loss import sigmoid_focal_loss
from lib.losses.RDIoU import rdiou, box_dict_to_xyzlwht


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MonoDETR(nn.Module):
    """ This is the MonoDETR module that performs monocualr 3D object detection """

    def __init__(self, backbone, depthaware_transformer, depth_predictor, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, init_box=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            depthaware_transformer: depth-aware transformer architecture. See depth_aware_transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For KITTI, we recommend 50 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage MonoDETR
        """
        super().__init__()

        self.num_queries = num_queries
        self.depthaware_transformer = depthaware_transformer
        self.depth_predictor = depth_predictor
        hidden_dim = depthaware_transformer.d_model
        self.num_feature_levels = num_feature_levels

        # prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)  # [3d_cx, 3d_cy, l, r, t, b]
        self.dim_embed_3d = MLP(hidden_dim, hidden_dim, 3, 2)  # [h, w, l] - mean_size
        self.angle_embed = MLP(hidden_dim, hidden_dim, 24, 2)  # 12 classes + 12 offset for each classes
        self.depth_embed = MLP(hidden_dim, hidden_dim, 2, 2)  # depth and deviation
        self.depth_ave_layer = nn.Linear(3, 1)
        nn.init.constant_(self.depth_ave_layer.weight, 1 / 3)
        nn.init.zeros_(self.depth_ave_layer.bias)

        if init_box:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (depthaware_transformer.decoder.num_layers + 1) if two_stage else depthaware_transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.depthaware_transformer.decoder.bbox_embed = self.bbox_embed
            self.dim_embed_3d = _get_clones(self.dim_embed_3d, num_pred)
            self.depthaware_transformer.decoder.dim_embed = self.dim_embed_3d
            self.angle_embed = _get_clones(self.angle_embed, num_pred)
            self.depth_embed = _get_clones(self.depth_embed, num_pred)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.dim_embed_3d = nn.ModuleList([self.dim_embed_3d for _ in range(num_pred)])
            self.angle_embed = nn.ModuleList([self.angle_embed for _ in range(num_pred)])
            self.depth_embed = nn.ModuleList([self.depth_embed for _ in range(num_pred)])
            self.depthaware_transformer.decoder.bbox_embed = None

        if two_stage:
            # hack implementation for two-stage
            self.depthaware_transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, images, calibs, img_sizes) -> Dict[str, Union[torch.Tensor, List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]]:
        """The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        Returns:
            A dict of Tensors with key:
            * pred_logits: predicted class logits with shape [batch, num_boxes, num_classes]
            * pred_boxes: predicted normalized 3D bbox (3d_cx, 3d_cy, l, r, t, b) with shape
                [batch, num_boxes, 6]. Each element is in [0, 1].
            * pred_3d_dim: predicted 3D bbox dimension (x, y, z) - mean_size with shape
                [batch, num_boxes, 3].
            * pred_depth: predicted depth for each 3D bbox with shape [batch, num_boxes]
            * pred_angle: predicted angle class logits and offsets with shape
                [batch, num_boxes, 24]. 12 for classes and 12 for class offsets.
            * pred_depth_map_logits: predicted depth map logits with shape
                [batch, num_depth_bins, H, W].
        """

        features, pos = self.backbone(images)

        srcs = []
        masks = []
        for lvl, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[lvl](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for lvl in range(_len_srcs, self.num_feature_levels):
                if lvl == _len_srcs:
                    src = self.input_proj[lvl](features[-1].tensors)
                else:
                    src = self.input_proj[lvl](srcs[-1])
                m = torch.zeros(src.shape[0], src.shape[2], src.shape[3]).to(torch.bool).to(src.device)
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None

        if not self.two_stage:
            query_embeds = self.query_embed.weight

        pred_depth_map_logits, depth_pos_embed, weighted_depth = self.depth_predictor(srcs, masks[1], pos[1])

        hs, init_reference, inter_references, inter_references_dim, enc_outputs_class, enc_outputs_coord_unact = self.depthaware_transformer(
            srcs, masks, pos, query_embeds, depth_pos_embed)

        outputs_coords = []
        outputs_classes = []
        outputs_3d_dims = []
        outputs_depths = []
        outputs_angles = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 6:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            # 3d center + 2d box
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)

            # classes
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_classes.append(outputs_class)

            # 3D sizes
            size3d = inter_references_dim[lvl]
            outputs_3d_dims.append(size3d)

            # depth_geo
            box2d_height_norm = outputs_coord[:, :, 4] + outputs_coord[:, :, 5]
            box2d_height = torch.clamp(box2d_height_norm * img_sizes[:, 1: 2], min=1.0)
            depth_geo = size3d[:, :, 0] / box2d_height * calibs[:, 0, 0].unsqueeze(1)

            # depth_reg
            depth_reg = self.depth_embed[lvl](hs[lvl])

            # depth_map
            outputs_center3d = ((outputs_coord[..., :2] - 0.5) * 2).unsqueeze(2).detach()
            depth_map = F.grid_sample(
                weighted_depth.unsqueeze(1),
                outputs_center3d,
                mode='bilinear',
                align_corners=True).squeeze(1)

            # depth average + sigma
            # depth_ave = ((1. / (depth_reg[:, :, 0: 1].sigmoid() + 1e-6) - 1.) + depth_geo.unsqueeze(-1) + depth_map) / 3
            depth_ave = self.depth_ave_layer(torch.cat([(1. / (depth_reg[:, :, 0: 1].sigmoid() + 1e-6) - 1.), depth_geo.unsqueeze(-1), depth_map], dim=-1))
            depth_ave = torch.cat([depth_ave, depth_reg[:, :, 1: 2]], -1)
            outputs_depths.append(depth_ave)

            # angles
            outputs_angle = self.angle_embed[lvl](hs[lvl])
            outputs_angles.append(outputs_angle)

        outputs_coord = torch.stack(outputs_coords)
        outputs_class = torch.stack(outputs_classes)
        outputs_3d_dim = torch.stack(outputs_3d_dims)
        outputs_depth = torch.stack(outputs_depths)
        outputs_angle = torch.stack(outputs_angles)

        out: Dict[str, Union[torch.Tensor, List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = {}
        out['pred_logits'] = outputs_class[-1]
        out['pred_boxes'] = outputs_coord[-1]
        out['pred_3d_dim'] = outputs_3d_dim[-1]
        out['pred_depth'] = outputs_depth[-1]
        out['pred_angle'] = outputs_angle[-1]
        out['pred_depth_map_logits'] = pred_depth_map_logits

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth) -> List[Dict[str, torch.Tensor]]:
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b,
                 'pred_3d_dim': c, 'pred_angle': d, 'pred_depth': e}
                for a, b, c, d, e in zip(outputs_class[:-1], outputs_coord[:-1],
                                         outputs_3d_dim[:-1], outputs_angle[:-1], outputs_depth[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for MonoDETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes: int, matcher: nn.Module, weight_dict: Dict[str, float], focal_alpha: float, losses: List[str]):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.loss_names = losses
        assert len(self.weight_dict) == len(self.loss_names), f'The length of `weight_dict`({len(self.weight_dict)}) and `loss_names`({len(self.loss_names)}) should be consistent.'
        self.focal_alpha = focal_alpha
        self.ddn_loss = DDNLoss()  # for depth map

    def loss_labels(self, outputs, targets, indices, num_boxes, **kwargs) -> torch.Tensor:
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o.squeeze().long()

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_cls = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        return loss_cls

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, **kwargs) -> torch.Tensor:
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return card_err

    def loss_3dcenter(self, outputs, targets, indices, num_boxes, **kwargs) -> torch.Tensor:
        matched_outputs, matched_targets = kwargs['matched_outputs'], kwargs['matched_targets']
        src_3dcenter = matched_outputs['pred_boxes'][..., :2]
        target_3dcenter = matched_targets['boxes_3d'][..., :2]

        loss_3dcenter = F.l1_loss(src_3dcenter, target_3dcenter, reduction='sum') / num_boxes
        return loss_3dcenter

    def loss_boxes(self, outputs, targets, indices, num_boxes, **kwargs) -> torch.Tensor:
        matched_outputs, matched_targets = kwargs['matched_outputs'], kwargs['matched_targets']
        src_2dboxes = matched_outputs['pred_boxes'][..., 2:6]
        target_2dboxes = matched_targets['boxes_3d'][..., 2:6]

        # l1
        loss_bbox = F.l1_loss(src_2dboxes, target_2dboxes, reduction='sum') / num_boxes
        return loss_bbox

    def loss_giou(self, outputs, targets, indices, num_boxes, **kwargs) -> torch.Tensor:
        matched_outputs, matched_targets = kwargs['matched_outputs'], kwargs['matched_targets']
        src_boxes = matched_outputs['pred_boxes']
        target_boxes = matched_targets['boxes_3d']
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcylrtb_to_xyxy(src_boxes),
            box_ops.box_cxcylrtb_to_xyxy(target_boxes)))
        loss_giou = loss_giou.sum() / num_boxes
        return loss_giou

    def loss_rdiou(self,
                   outputs: Dict[str, torch.Tensor],
                   targets: List[Dict[str, torch.Tensor]],
                   indices: List[Tuple[torch.Tensor, torch.Tensor]],
                   num_boxes: int,
                   **kwargs) -> torch.Tensor:
        matched_outputs, matched_targets = kwargs['matched_outputs'], kwargs['matched_targets']
        mean_size = matched_targets['src_size_3d'] - matched_targets['size_3d']
        bboxes = box_dict_to_xyzlwht(matched_outputs, is_target=False, mean_size=mean_size)
        target_bboxes = box_dict_to_xyzlwht(matched_targets, is_target=True)

        center_distance_penalty, iou = rdiou(bboxes, target_bboxes)
        loss_rdiou = 1 - torch.clamp(iou - center_distance_penalty, min=-1.0, max=1.0)
        loss_rdiou = loss_rdiou.sum() / num_boxes
        return loss_rdiou

    def loss_depths(self, outputs, targets, indices, num_boxes, **kwargs) -> torch.Tensor:
        matched_outputs, matched_targets = kwargs['matched_outputs'], kwargs['matched_targets']
        src_depths = matched_outputs['pred_depth']
        target_depths = matched_targets['depth'].squeeze()

        depth_input, depth_log_variance = src_depths[:, 0], src_depths[:, 1]
        depth_loss = 1.4142 * torch.exp(-depth_log_variance) * torch.abs(depth_input - target_depths) + depth_log_variance
        depth_loss = depth_loss.sum() / num_boxes
        return depth_loss

    def loss_dims(self, outputs, targets, indices, num_boxes, **kwargs) -> torch.Tensor:
        matched_outputs, matched_targets = kwargs['matched_outputs'], kwargs['matched_targets']
        src_dims = matched_outputs['pred_3d_dim']
        target_dims = matched_targets['size_3d']

        dimension = target_dims.clone().detach()
        dim_loss = torch.abs(src_dims - target_dims)
        dim_loss /= dimension
        with torch.no_grad():
            compensation_weight = F.l1_loss(src_dims, target_dims) / dim_loss.mean()
        dim_loss *= compensation_weight
        dim_loss = dim_loss.sum() / num_boxes
        return dim_loss

    def loss_angles(self,
                    outputs: Dict[str, torch.Tensor],
                    targets: List[Dict[str, torch.Tensor]],
                    indices: List[Tuple[torch.Tensor, torch.Tensor]],
                    num_boxes: int,
                    **kwargs) -> torch.Tensor:
        matched_outputs, matched_targets = kwargs['matched_outputs'], kwargs['matched_targets']
        heading_input = matched_outputs['pred_angle']
        target_heading_cls = matched_targets['heading_bin']
        target_heading_res = matched_targets['heading_res']

        heading_input = heading_input.view(-1, 24)
        heading_target_cls = target_heading_cls.view(-1).long()
        heading_target_res = target_heading_res.view(-1)

        # classification loss
        heading_input_cls = heading_input[:, 0:12]
        cls_loss = F.cross_entropy(heading_input_cls, heading_target_cls, reduction='sum')

        # regression loss
        heading_input_res = heading_input[:, 12:24]
        heading_input_res = heading_input_res.gather(dim=1, index=heading_target_cls.view(-1, 1)).squeeze()
        reg_loss = F.l1_loss(heading_input_res, heading_target_res, reduction='sum')

        angle_loss = cls_loss + reg_loss
        angle_loss = angle_loss / num_boxes
        return angle_loss

    def loss_depth_map(self,
                       outputs: Dict[str, torch.Tensor],
                       targets: List[Dict[str, torch.Tensor]],
                       indices: List[Tuple[torch.Tensor, torch.Tensor]],
                       num_boxes: int,
                       **kwargs) -> torch.Tensor:
        depth_map_logits = outputs['pred_depth_map_logits']

        num_gt_per_img = [len(t['boxes']) for t in targets]
        gt_boxes2d = torch.cat([t['boxes'] for t in targets], dim=0) * depth_map_logits.new_tensor([80, 24, 80, 24])
        gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)
        gt_center_depth = torch.cat([t['depth'] for t in targets], dim=0).squeeze(dim=1)

        depth_map_loss = self.ddn_loss(
            depth_map_logits, gt_boxes2d, num_gt_per_img, gt_center_depth)
        return depth_map_loss

    def _get_src_permutation_idx(self, indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs) -> torch.Tensor:

        loss_map = {
            'loss_cls': self.loss_labels,
            'cardinality_error': self.loss_cardinality,
            'loss_bbox': self.loss_boxes,
            'loss_giou': self.loss_giou,
            'loss_rdiou': self.loss_rdiou,
            'loss_depth': self.loss_depths,
            'loss_dim': self.loss_dims,
            'loss_angle': self.loss_angles,
            'loss_center': self.loss_3dcenter,
            'loss_depth_map': self.loss_depth_map,
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def _extract_matched_pairs(self,
                               outputs: Dict[str, torch.Tensor],
                               targets: List[Dict[str, torch.Tensor]],
                               indices: List[Tuple[torch.Tensor, torch.Tensor]]
                               ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Extracts matched pairs from outputs and targets by indices.

        Args:
            outputs: A dict of tensors outputed by the model.
            targets: A list of dicts, such that len(targets) == batch_size.
            indices: A list of size batch_size, containing tuples of (index_i, index_j) where:
                * index_i is the indices of the selected predictions (in order)
                * index_j is the indices of the corresponding selected targets (in order)
                For each batch element, it holds:
                    len(index_i) = len(index_j) = min(num_queries, num_target_boxes)

        Returns:
            (matched_outputs, matched_targets), both are dicts of tensors.
            matched_outputs[key][i] is matched with matched_targets[key][i] for any key, i.
            len(matched_outputs[key_i]) = len(matched_targets[key_j]) = num_boxes_among_whole_batch for any key_i, key_j.
        """
        idx = self._get_src_permutation_idx(indices)
        matched_outputs = {k: v[idx] for k, v in outputs.items() if k != 'aux_outputs'}
        matched_targets = {k: torch.cat([target[k][i] for target, (_, i) in zip(targets, indices)], dim=0)
                           for k in targets[0]}

        return matched_outputs, matched_targets

    def forward(self, outputs, targets) -> Tuple[Dict[str, torch.Tensor], Dict[str, Number]]:
        """This performs the loss computation.

        Args:
            outputs: dict of tensors, see the output specification of the model for the format
            targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

        Returns:
            losses: A dict of weighted loss tensors.
            unweighted_losses_log_dict: A dict of unweighted loss numbers for logging purposes only.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        matched_outputs, matched_targets = self._extract_matched_pairs(outputs, targets, indices)

        device = next(iter(outputs.values())).device
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = torch.tensor([len(t["labels"]) for t in targets], dtype=torch.float, device=device).sum()
        if is_dist_avail_and_initialized():
            dist.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        unweighted_losses_log_dict = {}
        losses = {}
        # Compute all the requested losses
        for loss_name in self.loss_names:
            optional_params = dict(matched_outputs=matched_outputs, matched_targets=matched_targets)
            loss = self.get_loss(loss_name, outputs, targets, indices, num_boxes, **optional_params)
            losses[loss_name] = loss * self.weight_dict[loss_name]
            unweighted_losses_log_dict[loss_name] = loss

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                matched_outputs, matched_targets = self._extract_matched_pairs(aux_outputs, targets, indices)
                optional_params = dict(matched_outputs=matched_outputs, matched_targets=matched_targets)
                for loss_name in self.loss_names:
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    if loss_name == 'loss_depth_map':
                        continue
                    loss = self.get_loss(loss_name, aux_outputs, targets, indices, num_boxes, **optional_params)
                    losses[f'{loss_name}_{i}'] = loss * self.weight_dict[loss_name]
                    unweighted_losses_log_dict[f'{loss_name}_{i}'] = loss

        unweighted_losses_log_dict = misc.reduce_dict(unweighted_losses_log_dict)
        unweighted_losses_log_dict = {loss_name: loss.item() for loss_name, loss in unweighted_losses_log_dict.items()}

        return losses, unweighted_losses_log_dict


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(model_cfg, loss_cfg):
    # backbone
    backbone = build_backbone(model_cfg)

    # detr
    depthaware_transformer = build_depthaware_transformer(model_cfg)

    # depth prediction module
    depth_predictor = DepthPredictor(model_cfg)

    model = MonoDETR(
        backbone,
        depthaware_transformer,
        depth_predictor,
        num_classes=model_cfg['num_classes'],
        num_queries=model_cfg['num_queries'],
        aux_loss=loss_cfg['aux_loss'],
        num_feature_levels=model_cfg['num_feature_levels'],
        with_box_refine=model_cfg['with_box_refine'],
        two_stage=model_cfg['two_stage'],
        init_box=model_cfg['init_box'])

    # matcher
    matcher = build_matcher(model_cfg['matcher'])

    # loss
    weight_dict = loss_cfg['weights']

    # TODO this is a hack
    # if loss_cfg['aux_loss']:
    #     aux_weight_dict = {}
    #     for i in range(model_cfg['dec_layers'] - 1):
    #         aux_weight_dict.update({f'{loss_name}_{i}': v for loss_name, v in weight_dict.items()})
    #     aux_weight_dict.update({f'{loss_name}_enc': v for loss_name, v in weight_dict.items()})
    #     weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(
        model_cfg['num_classes'],
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=loss_cfg['focal_alpha'],
        losses=loss_cfg['losses'])

    return model, criterion
