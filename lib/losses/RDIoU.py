from typing import Tuple
import torch


def rdiou(bboxes1: torch.Tensor, bboxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gets the RDIoU from two tensors of bboxes.

    Args:
        bboxes1: A tensor of bboxes with shape [*, num_boxes, C], where C >= 6 and each element represents (x, y, z, l, w, h, theta), respectively.
        bboxes2: A tensor of bboxes with shape [*, num_boxes, C], where C >= 6 and each element represents (x, y, z, l, w, h, theta), respectively.

        x, y, z should be in [0, 1]. l, w, h are the 3D bbox size - mean_size. theta is in [-pi, pi].

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
    l1, w1, h1 = torch.clamp(l1, max=10), torch.clamp(w1, max=10), torch.clamp(h1, max=10)
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
