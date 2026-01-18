# -*- coding: utf-8 -*-
"""
Khoảng cách bề mặt & SDF/DTM bổ sung.
- compute_dtm: DTM foreground hoặc fg+bg cho mask nhị phân.
- hd_loss: Hausdorff-like loss (weighted MSE theo DTM) – có hỗ trợ multi-class.
- compute_sdf: import từ util để tránh trùng logic.
"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
from .util import compute_sdf_numpy


def compute_dtm(img_gt, out_shape, normalize=False, fg=False):
    """
    Distance Transform Map cho mask nhị phân.
    img_gt: ndarray [B,D,H,W] hoặc [D,H,W]
    fg=False: tổng fg+bg; fg=True: chỉ fg
    """
    arr = img_gt.astype(np.uint8)
    if arr.ndim == 3:
        arr = arr[None, ...]
    B = arr.shape[0]
    fg_dtm = np.zeros((B, *arr.shape[1:]), dtype=np.float32)

    for b in range(B):
        posmask = arr[b].astype(bool)
        if not posmask.any():
            continue

        if not fg:
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            if normalize:
                posdis = (posdis - posdis.min()) / max(1e-6, (posdis.max() - posdis.min()))
                negdis = (negdis - negdis.min()) / max(1e-6, (negdis.max() - negdis.min()))
            fg_dtm[b] = posdis + negdis
            fg_dtm[b][boundary == 1] = 0
        else:
            posdis = distance(posmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            if normalize:
                posdis = (posdis - posdis.min()) / max(1e-6, (posdis.max() - posdis.min()))
            fg_dtm[b] = posdis
            fg_dtm[b][boundary == 1] = 0

    return fg_dtm.reshape(out_shape).astype(np.float32)


def hd_loss(seg_soft: torch.Tensor,
            gt: torch.Tensor,
            gt_dtm: torch.Tensor | None = None,
            one_side: bool = True,
            seg_dtm: torch.Tensor | None = None,
            num_classes: int | None = None):
    """
    Hausdorff-like boundary loss (Karimi et al. style).

    Hỗ trợ:
    - Binary: seg_soft [B,D,H,W], gt [B,D,H,W] (0/1)
    - Multi-class: seg_soft [B,C,D,H,W], gt [B,D,H,W] (0..C-1)
      -> tính trung bình HD-loss trên các lớp foreground (1..C-1).
    """
    if seg_soft.dim() == 4:
        # Binary trường hợp cũ
        if gt_dtm is None:
            gt_np = gt.detach().cpu().numpy().astype(np.uint8)
            dtm_np = compute_dtm(gt_np, gt_np.shape, normalize=False, fg=True)
            gt_dtm = torch.from_numpy(dtm_np).to(seg_soft.device, seg_soft.dtype)

        delta_s = (seg_soft - gt.float()) ** 2
        if one_side:
            dtm = gt_dtm ** 2
        else:
            if seg_dtm is None:
                with torch.no_grad():
                    seg_bin = (seg_soft > 0.5).float()
                    seg_np = seg_bin.detach().cpu().numpy().astype(np.uint8)
                    seg_dtm_np = compute_dtm(seg_np, seg_np.shape, normalize=False, fg=True)
                    seg_dtm = torch.from_numpy(seg_dtm_np).to(seg_soft.device, seg_soft.dtype)
            dtm = (gt_dtm ** 2) + (seg_dtm ** 2)

        multipled = delta_s * dtm
        return multipled.mean()

    # ---- Multi-class ----
    assert seg_soft.dim() == 5, "seg_soft phải là [B,C,D,H,W] hoặc [B,D,H,W]"
    B, C = seg_soft.shape[:2]
    if num_classes is None:
        num_classes = C

    gt = gt.long()
    hd_list = []

    for c in range(1, num_classes):  # foreground classes
        prob_c = seg_soft[:, c]          # [B,D,H,W]
        gt_c = (gt == c).float()         # [B,D,H,W]

        if gt_dtm is None:
            gt_np = gt_c.detach().cpu().numpy().astype(np.uint8)
            dtm_np = compute_dtm(gt_np, gt_np.shape, normalize=False, fg=True)
            gt_dtm_c = torch.from_numpy(dtm_np).to(seg_soft.device, seg_soft.dtype)
        else:
            gt_dtm_c = gt_dtm[:, c]

        delta_s = (prob_c - gt_c) ** 2
        if one_side:
            dtm = gt_dtm_c ** 2
        else:
            if seg_dtm is None:
                with torch.no_grad():
                    seg_bin = (prob_c > 0.5).float()
                    seg_np = seg_bin.detach().cpu().numpy().astype(np.uint8)
                    seg_dtm_np = compute_dtm(seg_np, seg_np.shape, normalize=False, fg=True)
                    seg_dtm_c = torch.from_numpy(seg_dtm_np).to(seg_soft.device, seg_soft.dtype)
            else:
                seg_dtm_c = seg_dtm[:, c]
            dtm = (gt_dtm_c ** 2) + (seg_dtm_c ** 2)

        multipled = delta_s * dtm
        hd_list.append(multipled.mean())

    if not hd_list:
        return torch.tensor(0.0, device=seg_soft.device, dtype=seg_soft.dtype)
    return sum(hd_list) / len(hd_list)


def compute_sdf(img_gt, out_shape=None):
    sdf = compute_sdf_numpy(img_gt)
    if out_shape is not None:
        sdf = sdf.reshape(out_shape)
    return sdf.astype(np.float32)
