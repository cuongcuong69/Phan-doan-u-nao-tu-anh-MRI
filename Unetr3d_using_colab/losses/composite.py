# -*- coding: utf-8 -*-
"""
Bộ loss hợp thành cho DTC (seg + sdf + consistency) – phiên bản multi-class (BraTS).
- DiceCELoss: CrossEntropy + multi-class Dice (ổn định hơn chỉ dùng CE).
- SDFRegLoss: SmoothL1 giữa sdf_pred và sdf_gt (WT vs background).
- DualTaskConsistency: ép nhất quán WT (tumor vs background) giữa seg ↔ SDF.
- LossDTC: gói tất cả cho pipeline supervised / semi-supervised.

Giả định:
- Segmentation logits: [N, C, D, H, W], C>=2 (BraTS: C=4, nhãn 0..3).
- SDF head: [N, 1, D, H, W], SDF cho Whole Tumor (label > 0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import (
    dice_loss1,
    softmax_mse_loss,
    softmax_kl_loss,
)
from .ramps import sigmoid_rampup


class DiceCELoss(nn.Module):
    """
    CE + multi-class Dice trên các lớp foreground (1..C-1).

    logits: [N, C, ...]
    target: [N, D, H, W] (int 0..C-1) hoặc [N,1,D,H,W]
    """

    def __init__(self, weight_ce: float = 1.0, weight_dice: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.wce = float(weight_ce)
        self.wdice = float(weight_dice)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == logits.dim():
            # [N,1,D,H,W] -> [N,D,H,W]
            target = target[:, 0]

        # CrossEntropy multi-class
        ce = self.ce(logits, target.long())

        # Multi-class Dice (foreground classes only)
        probs = torch.softmax(logits, dim=1)       # [N,C,...]
        N, C = probs.shape[:2]
        target_oh = F.one_hot(target.long(), num_classes=C)  # [N,D,H,W,C]
        target_oh = target_oh.permute(0, 4, 1, 2, 3).float()  # [N,C,D,H,W]

        dice_list = []
        for c in range(1, C):  # bỏ background
            pc = probs[:, c]
            tc = target_oh[:, c]
            dice_c = dice_loss1(pc, tc)           # scalar
            dice_list.append(dice_c)

        dice_mean = sum(dice_list) / max(1, len(dice_list))
        return self.wce * ce + self.wdice * dice_mean


class SDFRegLoss(nn.Module):
    """
    Regularization cho SDF head (toàn WT vs background).
    pred_sdf: [N,1,D,H,W] hoặc [N,K,...] (sẽ lấy kênh đầu).
    gt_sdf:   [N,1,D,H,W]
    """

    def __init__(self, mode: str = "smoothl1", weight: float = 0.5):
        super().__init__()
        self.weight = float(weight)
        self.fn = nn.SmoothL1Loss() if mode.lower() == "smoothl1" else nn.L1Loss()

    def forward(self, pred_sdf: torch.Tensor, gt_sdf: torch.Tensor) -> torch.Tensor:
        if pred_sdf.dim() == 5 and pred_sdf.size(1) != 1:
            pred_sdf = pred_sdf[:, :1]
        return self.weight * self.fn(pred_sdf, gt_sdf)


class DualTaskConsistency(nn.Module):
    """
    Ép nhất quán giữa segmentation (WT vs BG) và SDF.

    Ý tưởng:
    - Từ SDF (1 kênh), sinh ra logits 2 kênh (BG/WT).
    - Từ seg multi-class logits (C kênh), nén về 2 kênh:
        logit_BG = logit[:,0]
        logit_WT = logsumexp(logit[:,1:])
    - So sánh 2 phân phối 2-kênh bằng MSE / KL, có ramp-up theo iter_frac.
    """

    def __init__(self, max_w: float = 1.0, ramp_len: float = 0.3, use_kl: bool = False):
        super().__init__()
        self.max_w = float(max_w)
        self.ramp_len = float(ramp_len)
        self.use_kl = bool(use_kl)

    @staticmethod
    def sdf_to_mask_logits(sdf: torch.Tensor, mult: float = 6.0) -> torch.Tensor:
        """
        sdf: [N,1,D,H,W] (giá trị ~[-1,1])
        Trả về logits 2 kênh: [N,2,D,H,W]
        - Kênh 1 (WT) có logit cao khi sdf < 0 (inside).
        """
        if sdf.dim() == 5 and sdf.size(1) != 1:
            sdf = sdf[:, :1]

        logit_wt = (-mult * sdf).clamp(-12, 12)  # foreground
        logit_bg = (+mult * sdf).clamp(-12, 12)  # background
        return torch.cat([logit_bg, logit_wt], dim=1)

    @staticmethod
    def logits_multi_to_binary(seg_logits: torch.Tensor) -> torch.Tensor:
        """
        Từ logits [N,C,...] (C>=2) -> logits 2 kênh BG/WT:
            logit_BG = logit[:,0]
            logit_WT = logsumexp(logit[:,1:])
        Nếu đã là 2 kênh thì giữ nguyên.
        """
        if seg_logits.shape[1] == 2:
            return seg_logits
        bg = seg_logits[:, 0:1]
        fg = torch.logsumexp(seg_logits[:, 1:], dim=1, keepdim=True)
        return torch.cat([bg, fg], dim=1)  # [N,2,...]

    def forward(self, seg_logits: torch.Tensor, sdf_pred: torch.Tensor, iter_frac: float) -> torch.Tensor:
        seg_bin_logits = self.logits_multi_to_binary(seg_logits)
        target_logits = self.sdf_to_mask_logits(sdf_pred)

        if self.use_kl:
            loss = softmax_kl_loss(seg_bin_logits, target_logits, sigmoid=False)
        else:
            loss_map = softmax_mse_loss(seg_bin_logits, target_logits, sigmoid=False)
            loss = loss_map.mean()

        w = self.max_w * sigmoid_rampup(iter_frac, self.ramp_len)
        return w * loss


class LossDTC(nn.Module):
    """
    Loss tổng hợp cho mô hình DTC multi-class (BraTS):
    - L_seg: CE + multi-class Dice (C lớp, Dice tính trên foreground).
    - L_sdf: SmoothL1 giữa sdf_pred và sdf_gt (WT vs BG).
    - L_cons: consistency giữa seg (WT vs BG) và SDF (WT vs BG).

    Trong huấn luyện semi-supervised:
    - batch_lab: dùng đủ 3 thành phần.
    - batch_unlab: chỉ dùng consistency (L_cons_unl).
    """

    def __init__(self, w_seg: float = 1.0, w_sdf: float = 0.1,
                 w_cons: float = 0.1, cons_type: str = "mse"):
        super().__init__()
        self.w_seg = float(w_seg)
        self.w_sdf = float(w_sdf)
        self.w_cons = float(w_cons)

        if cons_type == "mse":
            self.cons_crit = softmax_mse_loss
        elif cons_type == "kl":
            self.cons_crit = softmax_kl_loss
        else:
            raise ValueError(f"cons_type={cons_type}")

        self.ce = nn.CrossEntropyLoss()
        self.sdf_l1 = nn.SmoothL1Loss(beta=1.0, reduction="mean")

    def _seg_loss(self, seg_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Multi-class segmentation loss:
        - CrossEntropy trên nhãn 0..C-1
        - Dice trung bình trên foreground classes (1..C-1)
        """
        if mask.dim() == seg_logits.dim():
            # [N,1,D,H,W] -> [N,D,H,W]
            mask = mask[:, 0]

        # CrossEntropy
        ce = self.ce(seg_logits, mask.long())

        # Multi-class Dice
        probs = torch.softmax(seg_logits, dim=1)
        N, C = probs.shape[:2]
        mask_oh = F.one_hot(mask.long(), num_classes=C)  # [N,D,H,W,C]
        mask_oh = mask_oh.permute(0, 4, 1, 2, 3).float()

        dice_list = []
        for c in range(1, C):  # foreground only
            pc = probs[:, c]
            tc = mask_oh[:, c]
            dice_c = dice_loss1(pc, tc)
            dice_list.append(dice_c)

        dice_mean = sum(dice_list) / max(1, len(dice_list))
        return ce + dice_mean

    def _sdf_loss(self, sdf_pred: torch.Tensor, sdf_gt: torch.Tensor) -> torch.Tensor:
        if sdf_pred.dim() == 5 and sdf_pred.size(1) != 1:
            sdf_pred = sdf_pred[:, :1]
        return self.sdf_l1(sdf_pred, sdf_gt)

    def _consistency_from_sdf(self, seg_logits: torch.Tensor, sdf_pred: torch.Tensor) -> torch.Tensor:
        """
        Consistency: seg (WT vs BG) ↔ SDF (WT vs BG) theo DTC-style.
        """
        # Từ seg logits (C kênh) -> 2 kênh BG/WT
        if seg_logits.shape[1] == 2:
            seg_bin = seg_logits
        else:
            bg = seg_logits[:, 0:1]
            fg = torch.logsumexp(seg_logits[:, 1:], dim=1, keepdim=True)
            seg_bin = torch.cat([bg, fg], dim=1)

        # Từ SDF -> logits BG/WT
        mult = 6.0
        if sdf_pred.dim() == 5 and sdf_pred.size(1) != 1:
            sdf_pred = sdf_pred[:, :1]
        logit_wt = (-mult * sdf_pred).clamp(-12, 12)
        logit_bg = (+mult * sdf_pred).clamp(-12, 12)
        target_bin = torch.cat([logit_bg, logit_wt], dim=1)

        # Consistency giữa 2 phân phối 2-kênh
        if self.cons_crit is softmax_mse_loss:
            loss_map = self.cons_crit(seg_bin, target_bin, sigmoid=False)
            return loss_map.mean()
        else:
            return self.cons_crit(seg_bin, target_bin, sigmoid=False)

    def forward(self,
                batch_lab: dict | None = None,
                batch_unlab: dict | None = None,
                iter_frac: float = 1.0,
                compute_consistency: bool = True):
        """
        batch_lab: dict với các khóa:
            - "seg_logits": [N,C,D,H,W]
            - "sdf_pred":   [N,1,D,H,W]
            - "mask":       [N,D,H,W] hoặc [N,1,D,H,W]
            - "sdf_gt":     [N,1,D,H,W] (tuỳ chọn)
        batch_unlab:
            - "seg_logits": [N,C,D,H,W]
            - "sdf_pred":   [N,1,D,H,W]
        """

        L_seg = torch.tensor(0.0)
        L_sdf = torch.tensor(0.0)
        Lc_lab = torch.tensor(0.0)
        Lc_unl = torch.tensor(0.0)

        if batch_lab is not None:
            seg_logits = batch_lab["seg_logits"]
            sdf_pred = batch_lab["sdf_pred"]
            mask = batch_lab["mask"]
            sdf_gt = batch_lab.get("sdf_gt", None)

            if sdf_gt is None:
                sdf_gt = torch.zeros_like(sdf_pred)

            L_seg = self._seg_loss(seg_logits, mask)
            L_sdf = self._sdf_loss(sdf_pred, sdf_gt)

            if compute_consistency:
                Lc_lab = self._consistency_from_sdf(seg_logits, sdf_pred)

        if compute_consistency and (batch_unlab is not None):
            seg_logits_u = batch_unlab["seg_logits"]
            sdf_pred_u = batch_unlab["sdf_pred"]
            Lc_unl = self._consistency_from_sdf(seg_logits_u, sdf_pred_u)

        loss = self.w_seg * L_seg + self.w_sdf * L_sdf + self.w_cons * (Lc_lab + Lc_unl)

        return {
            "loss": loss,
            "L_seg": L_seg.detach(),
            "L_sdf": L_sdf.detach(),
            "L_cons": (Lc_lab + Lc_unl).detach(),
        }
