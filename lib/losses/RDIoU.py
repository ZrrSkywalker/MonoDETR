from typing import Dict, List, Optional, Tuple, Union
import torch

from lib.datasets.utils import class2angle


def box_dict_to_xyzlwht(box_dict: Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
                        is_target: Optional[bool] = True,
                        mean_size: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Converts a dict of box tensors to a tensor whose last dimension is [x, y, z, l, h, w, theta].

    Args:
        box_dict: A dict of tensors or a list of dict of tensors. If `is_target` is True,
            the dict should contain the following keys:
            * boxes_3d
            * src_size_3d
            * depth
            * heading_bin
            * heading_res

            Otherwise, the dict should contain
            * pred_boxes
            * pred_depth
            * pred_3d_dim
            * pred_angle
        is_target: Whether `box_dict` is the target dict.
        mean_size: The mean size of the box. Only useful when `is_target` is False.

    Returns:
        A tensor of boxes with the last dimension [x, y, z, l, h, w, theta].
    """
    if isinstance(box_dict, list):
        box_dict = {k: torch.cat([target[k] for target in box_dict], dim=0) for k in box_dict[0]}
    if is_target:
        target_3dcenter = box_dict['boxes_3d'][..., :2]
        target_src_dims = box_dict['src_size_3d']
        # [num_boxes, 1]
        target_depths = box_dict['depth']

        # [num_boxes, 1]
        target_heading_cls = box_dict['heading_bin']
        # [num_boxes, 1]
        target_heading_res = box_dict['heading_res']
        # [num_boxes, 1]
        target_heading_angle = class2angle(target_heading_cls, target_heading_res, to_label_format=True)

        # [num_boxes, 7]
        target_bboxes = torch.cat([target_3dcenter, target_depths, target_src_dims, target_heading_angle], dim=-1)
        return target_bboxes

    src_3dcenter = box_dict['pred_boxes'][..., :2]
    src_depths = box_dict['pred_depth']
    depth_input = src_depths[..., 0:1]

    src_dims = box_dict['pred_3d_dim']
    if mean_size is not None:
        src_dims += mean_size

    heading_input = box_dict['pred_angle']
    heading_cls = heading_input[..., :heading_input.shape[-1] // 2].argmax(-1, keepdim=True)
    # [num_boxes, 12]
    heading_res = heading_input[..., heading_input.shape[-1] // 2:]
    # [num_boxes, 1]
    heading_res = heading_res.gather(dim=-1, index=heading_cls)
    # [num_boxes, 1]
    heading_angle = class2angle(heading_cls, heading_res, to_label_format=True)
    bboxes = torch.cat([src_3dcenter, depth_input, src_dims, heading_angle], dim=-1)
    return bboxes


def rdiou(bboxes1: torch.Tensor, bboxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gets the RDIoU from two tensors of bboxes.

    Args:
        bboxes1: A tensor of bboxes with shape [*, num_boxes, C], where C >= 6 and each element represents (x, y, z, l, w, h, theta), respectively.
        bboxes2: A tensor of bboxes with shape [*, num_boxes, C], where C >= 6 and each element represents (x, y, z, l, w, h, theta), respectively.

        x, y, z should be in [0, 1]. l, w, h are the 3D bbox size. theta is in [-pi, pi].

    Returns:
        center_distance_penalty: A tensor with shape [*, num_boxes].
        rdiou: A tensor with shape [*, num_boxes]. Each element represents the RDIoU of each pair of bboxes.
    """
    x1u, y1u, z1u = bboxes1[..., 0], bboxes1[..., 1], bboxes1[..., 2]
    l1, w1, h1 = torch.exp(bboxes1[..., 3]), torch.exp(bboxes1[..., 4]), torch.exp(bboxes1[..., 5])  # test
    t1 = torch.sin(bboxes1[..., 6]) * torch.cos(bboxes2[..., 6])
    x2u, y2u, z2u = bboxes2[..., 0], bboxes2[..., 1], bboxes2[..., 2]
    l2, w2, h2 = torch.exp(bboxes2[..., 3]), torch.exp(bboxes2[..., 4]), torch.exp(bboxes2[..., 5])  # test
    t2 = torch.cos(bboxes1[..., 6]) * torch.sin(bboxes2[..., 6])

    # we emperically scale the y/z to make their predictions more sensitive.
    x1 = x1u
    y1 = y1u * 2
    z1 = z1u * 2
    x2 = x2u
    y2 = y2u * 2
    z2 = z2u * 2

    # clamp is necessray to aviod inf.
    l1, w1, h1 = torch.clamp(l1, max=60), torch.clamp(w1, max=60), torch.clamp(h1, max=60)
    # emperically set to one to achieve the best performance
    j1, j2 = torch.ones_like(h2), torch.ones_like(h2)

    volume_1 = l1 * w1 * h1 * j1
    volume_2 = l2 * w2 * h2 * j2

    inter_l = torch.max(x1 - l1 / 2, x2 - l2 / 2)
    inter_r = torch.min(x1 + l1 / 2, x2 + l2 / 2)
    inter_t = torch.max(y1 - w1 / 2, y2 - w2 / 2)
    inter_b = torch.min(y1 + w1 / 2, y2 + w2 / 2)
    inter_u = torch.max(z1 - h1 / 2, z2 - h2 / 2)
    inter_d = torch.min(z1 + h1 / 2, z2 + h2 / 2)
    inter_m = torch.max(t1 - j1 / 2, t2 - j2 / 2)
    inter_n = torch.min(t1 + j1 / 2, t2 + j2 / 2)

    inter_volume = torch.clamp((inter_r - inter_l), min=0) * torch.clamp((inter_b - inter_t), min=0) \
        * torch.clamp((inter_d - inter_u), min=0) * torch.clamp((inter_n - inter_m), min=0)

    c_l = torch.min(x1 - l1 / 2, x2 - l2 / 2)
    c_r = torch.max(x1 + l1 / 2, x2 + l2 / 2)
    c_t = torch.min(y1 - w1 / 2, y2 - w2 / 2)
    c_b = torch.max(y1 + w1 / 2, y2 + w2 / 2)
    c_u = torch.min(z1 - h1 / 2, z2 - h2 / 2)
    c_d = torch.max(z1 + h1 / 2, z2 + h2 / 2)
    c_m = torch.min(t1 - j1 / 2, t2 - j2 / 2)
    c_n = torch.max(t1 + j1 / 2, t2 + j2 / 2)

    inter_diag = (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2 + (t2 - t1)**2
    c_diag = torch.clamp((c_r - c_l), min=0)**2 + torch.clamp((c_b - c_t), min=0)**2 + torch.clamp((c_d - c_u), min=0)**2 + torch.clamp((c_n - c_m), min=0)**2

    union = volume_1 + volume_2 - inter_volume
    center_distance_penalty = (inter_diag) / c_diag
    rdiou = inter_volume / union
    return center_distance_penalty, rdiou
