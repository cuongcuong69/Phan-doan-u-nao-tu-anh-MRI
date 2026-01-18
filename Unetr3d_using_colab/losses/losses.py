# -*- coding: utf-8 -*-
"""
Các loss cơ bản (Dice/Entropy/Consistency) dùng cho segmentation và DTC.
ĐÃ SỬA: bỏ .cuda() cứng, dùng device theo input.
Hỗ trợ multi-class (C>=2).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ----------------- Dice -----------------
def dice_loss(score, target):
    """
    Dice loss cho một kênh (binary) – dùng được cho bất kỳ shape nào.

    score:  [N,1,D,H,W] hoặc [N,...]
    target: [N,1,D,H,W] hoặc [N,...] (0/1)
    """
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2.0 * intersect + smooth) / (z_sum + y_sum + smooth)
    return 1.0 - loss


def dice_loss1(score, target):
    """
    Biến thể: mẫu số dùng tổng giá trị (không bình phương).
    Thích hợp khi score/target là xác suất / mask 0-1.
    """
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2.0 * intersect + smooth) / (z_sum + y_sum + smooth)
    return 1.0 - loss


# ----------------- Entropy -----------------
def entropy_loss(p, C=None):
    """
    Entropy trung bình của phân phối p.
    p: [N,C,...] xác suất (softmax/sigmoid output)
    C: số lớp (nếu None -> p.size(1))
    """
    device = p.device
    if C is None:
        C = p.size(1)
    denom = torch.tensor(np.log(C), device=device, dtype=p.dtype)
    y1 = -torch.sum(p * torch.log(p + 1e-6), dim=1) / denom
    return torch.mean(y1)


def entropy_loss_map(p, C=None):
    device = p.device
    if C is None:
        C = p.size(1)
    denom = torch.tensor(np.log(C), device=device, dtype=p.dtype)
    ent = -torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True) / denom
    return ent


def entropy_minmization(p):
    y1 = -torch.sum(p * torch.log(p + 1e-6), dim=1)
    return torch.mean(y1)


def entropy_map(p):
    return -torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)


# ----------------- Consistency -----------------
def softmax_dice_loss(input_logits, target_logits):
    """
    Dice giữa 2 logits (sau softmax). Tính trung bình theo kênh.
    Hỗ trợ C>=2.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0.0
    for i in range(n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    return dice / n


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """
    MSE giữa 2 phân phối (softmax/sigmoid) — dùng cho consistency.
    Trả về map loss (không reduce) để linh hoạt.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_p = torch.sigmoid(input_logits)
        target_p = torch.sigmoid(target_logits)
    else:
        input_p = F.softmax(input_logits, dim=1)
        target_p = F.softmax(target_logits, dim=1)
    return (input_p - target_p) ** 2


def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    """
    KL(input || target) trên phân phối (softmax hoặc sigmoid).
    Trả về scalar mean.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_logp = torch.log(torch.sigmoid(input_logits) + 1e-6)
        target_p = torch.sigmoid(target_logits)
    else:
        input_logp = F.log_softmax(input_logits, dim=1)
        target_p = F.softmax(target_logits, dim=1)
    return F.kl_div(input_logp, target_p, reduction='batchmean')


def symmetric_mse_loss(input1, input2):
    """MSE gửi gradient về cả 2 phía."""
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2) ** 2)


# ----------------- Focal -----------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, size_average=True):
        super().__init__()
        self.gamma = gamma
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha], dtype=torch.float32)
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
        self.size_average = size_average

    def forward(self, input, target):
        """
        input: [N,C,...], target: [N,...] (int 0..C-1)
        """
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2).contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target).view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            alpha = self.alpha.to(input.device, input.dtype)
            at = alpha.gather(0, target.view(-1))
            logpt = logpt * at

        loss = -(1 - pt) ** self.gamma * logpt
        return loss.mean() if self.size_average else loss.sum()
