# -*- coding: utf-8 -*-
"""
Huấn luyện VNet 3D đa lớp cho BraTS2020 (4 modality, 4 lớp seg) với full-volume dataloader.

Khác với bản patch-based:
- Dataloader: data/dataloader_brats3d_full.py (build_brats3d_full_*_loader)
- Toàn bộ volume 3D được resize về kích thước cố định (vd 128x128x128) rồi đưa vào mạng,
  không crop patch nữa.

Giống bản train_vnet_brats3d.py cũ:
- Model: models.vnet.VNet (n_channels=4, n_classes=4).
- Loss: CE / Dice / Dice+CE (w_ce, w_dice).
- Tích hợp:
    + tqdm.auto cho progress bar
    + wandb để log (nếu có)
    + Eval mỗi EVAL_EVERY epoch:
        * multi-class Dice theo lớp (1,2,3) và mean_fg
        * Dice trên WT, TC, ET (theo định nghĩa BraTS)
    + Lưu best/last checkpoints, snapshot mỗi SAVE_EVERY epoch
    + Resume từ checkpoint (RESUME_CKPT)
    + Giảm LR: nếu val_dice_struct = avg(WT,TC,ET) không cải thiện sau `patience` lần eval
      thì nhân LR với `factor`.
"""

from __future__ import annotations
import os
import sys
import time
import random
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# ---------------------- optional wandb ----------------------
try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    wandb = None
    _HAS_WANDB = False


# =============================================================================
# CONFIG (CFG)
# =============================================================================
CFG: Dict[str, Any] = {

    # --------------------- Experiment ---------------------
    "EXP_NAME": "brats3d_vnet_sup_fullvolume",  # đổi tên cho dễ phân biệt
    "SEED": 2025,

    # --------------------- Data (FULL VOLUME) ---------------------
    # Kích thước volume 3D sau khi resize (D,H,W)
    "VOLUME_SIZE": (128, 128, 128),

    "TRAIN_BATCH": 2,      # thường để 1 cho full volume cho đỡ tốn VRAM
    "VAL_BATCH": 2,
    "NUM_WORKERS_TRAIN": 4,
    "NUM_WORKERS_VAL": 4,

    "NUM_CHANNELS": 4,   # FLAIR, T1, T1CE, T2
    "NUM_CLASSES": 4,    # 0,1,2,3 (sau khi map 4 -> 3)

    # --------------------- Model ---------------------
    "VNET": {
        "n_channels": 4,
        "n_classes": 4,
        "n_filters": 16,
        "normalization": "batchnorm",  # "batchnorm" | "groupnorm" | "instancenorm" | "none"
        "has_dropout": True,
    },

    # --------------------- Optimizer ---------------------
    "OPTIM": {
        "LR": 1e-3,
        "WEIGHT_DECAY": 1e-4,
        "BETAS": (0.9, 0.999),
        "MAX_EPOCH": 200,
    },

    # --------------------- LR Scheduler (reduce on plateau) ---------------------
    "LR_SCHED": {
        "use": True,
        "factor": 0.5,     # LR = LR * factor
        "patience": 6,     # số lần eval liên tiếp không cải thiện => giảm LR
        "min_epoch": 15,   # chỉ bắt đầu xem xét giảm LR sau epoch này
    },

    # --------------------- Loss ---------------------
    "LOSS": {
        # "ce" | "dice" | "dicece"
        "loss_type": "dicece",

        "w_dice": 1.0,
        "w_ce": 1.0,
    },

    # --------------------- Validation ---------------------
    "EVAL_EVERY": 1,   # validate mỗi epoch

    # --------------------- Checkpoint ---------------------
    "SAVE_EVERY": 1000,
    "RESUME_CKPT": "",

    # --------------------- WandB ---------------------
    "WANDB": {
        "use_wandb": True,
        "project": "brats2020-vnet3d-sup-fullvolume",
        "entity": None,   # hoặc "username"
    },

    # --------------------- Device ---------------------
    "DEVICE": "cuda",  # "cuda" hoặc "cpu"
}


# =============================================================================
# PATH & IMPORTS RELATIVE TO PROJECT ROOT
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# ----------------- Dataloader FULL-VOLUME (đã resize) -----------------
from data.dataloader_brats3d_full import (
    build_brats3d_full_train_loader,
    build_brats3d_full_val_loader,
)

# model VNet
from models.vnet import VNet


# =============================================================================
# Local utilities: AverageMeter, Logger, cal_dice, region dice
# =============================================================================

class AverageMeter:
    """Theo dõi giá trị hiện tại và trung bình."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(1, self.count)


class Logger:
    """
    Ghi log (list các dict) vào file pickle.
    Dùng: logger = Logger('.../train_log.pkl'); logger.log({'epoch':1,'loss':...})
    """
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.data = []

    def log(self, train_point: Dict[str, Any]):
        self.data.append(train_point)
        with open(self.path, "wb") as fp:
            pickle.dump(self.data, fp, -1)


def cal_dice(prediction: np.ndarray, label: np.ndarray, num: int = 2) -> np.ndarray:
    """
    Multi-class Dice cho từng lớp 1..num-1.
    prediction, label: ndarray int (0..C-1)
    Trả về: array [num-1], phần tử i-1 là dice của lớp i.
    """
    total_dice = np.zeros(num - 1, dtype=np.float32)
    for i in range(1, num):
        p = (prediction == i).astype(float)
        g = (label == i).astype(float)
        denom = (p.sum() + g.sum())
        total_dice[i - 1] = (2.0 * (p * g).sum() / denom) if denom > 0 else 1.0
    return total_dice


def dice_binary_torch(pred_bin: torch.Tensor, gt_bin: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Dice nhị phân cho mask torch.bool / 0-1.
    pred_bin, gt_bin: cùng shape (B,D,H,W) hoặc (N,...).
    """
    pred_f = pred_bin.float()
    gt_f = gt_bin.float()
    intersection = (pred_f * gt_f).sum()
    den = pred_f.sum() + gt_f.sum()
    if den.item() == 0:
        # cả pred & gt rỗng => xem như Dice = 1.0
        return torch.tensor(1.0, device=pred_bin.device)
    return (2.0 * intersection + eps) / (den + eps)


def brats_region_dice_torch(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    """
    Tính Dice WT/TC/ET trên tensor Torch.

    pred, gt: (B,D,H,W) với nhãn 0..3
      - WT: label > 0
      - TC: label == 1 or 3
      - ET: label == 3
    """
    # WT
    mp_wt = (pred > 0)
    mg_wt = (gt > 0)
    dice_wt = dice_binary_torch(mp_wt, mg_wt).item()

    # TC
    mp_tc = (pred == 1) | (pred == 3)
    mg_tc = (gt == 1) | (gt == 3)
    dice_tc = dice_binary_torch(mp_tc, mg_tc).item()

    # ET
    mp_et = (pred == 3)
    mg_et = (gt == 3)
    dice_et = dice_binary_torch(mp_et, mg_et).item()

    return {"wt": dice_wt, "tc": dice_tc, "et": dice_et}


# =============================================================================
# Utility: seed, device, dice loss
# =============================================================================

def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    dev = CFG["DEVICE"]
    if dev == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA không khả dụng, chuyển sang CPU.")
        dev = "cpu"
    return torch.device(dev)


def multiclass_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    eps: float = 1e-5,
    ignore_bg: bool = True
) -> torch.Tensor:
    """
    Multi-class Dice loss (trung bình theo lớp).
    logits: (N,C,D,H,W)
    targets: (N,D,H,W) [0..C-1]
    """
    probs = torch.softmax(logits, dim=1)  # (N,C,D,H,W)
    one_hot = F.one_hot(targets.long(), num_classes=num_classes)  # (N,D,H,W,C)
    one_hot = one_hot.permute(0, 4, 1, 2, 3).float()  # (N,C,D,H,W)

    dims = (0, 2, 3, 4)
    intersection = torch.sum(probs * one_hot, dims)
    cardinality = torch.sum(probs + one_hot, dims)

    dice_per_class = (2.0 * intersection + eps) / (cardinality + eps)

    if ignore_bg and dice_per_class.numel() > 1:
        return 1.0 - dice_per_class[1:].mean()
    else:
        return 1.0 - dice_per_class.mean()


# =============================================================================
# Build model, optimizer, loaders
# =============================================================================

def build_model_and_opt(device: torch.device):
    vcfg = CFG["VNET"]
    model = VNet(
        n_channels=vcfg["n_channels"],
        n_classes=vcfg["n_classes"],
        n_filters=vcfg["n_filters"],
        normalization=vcfg["normalization"],
        has_dropout=vcfg["has_dropout"],
    )
    model = model.to(device)

    # nếu muốn multi-GPU:
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        print(f"[INFO] Dùng DataParallel trên {torch.cuda.device_count()} GPU")
        model = torch.nn.DataParallel(model)

    ocfg = CFG["OPTIM"]
    optimizer = optim.AdamW(
        model.parameters(),
        lr=ocfg["LR"],
        weight_decay=ocfg["WEIGHT_DECAY"],
        betas=tuple(ocfg["BETAS"]),
    )

    return model, optimizer


def build_loaders():
    """
    FULL VOLUME LOADER (KHÔNG PATCH):
    - dùng build_brats3d_full_*_loader
    - volume_size là kích thước resize 3D (D,H,W)
    """
    volume_size = CFG["VOLUME_SIZE"]

    # TRAIN
    train_loader = build_brats3d_full_train_loader(
        volume_size=volume_size,
        batch_size=CFG["TRAIN_BATCH"],
        num_workers=CFG["NUM_WORKERS_TRAIN"],
        seed=CFG["SEED"],
        # norm_mode: dùng default của dataloader (NORM_MODE) hoặc truyền thêm nếu cần
    )

    # VAL
    val_loader = build_brats3d_full_val_loader(
        volume_size=volume_size,
        batch_size=CFG["VAL_BATCH"],
        num_workers=CFG["NUM_WORKERS_VAL"],
        seed=CFG["SEED"],
    )
    return train_loader, val_loader


# =============================================================================
# LR scheduler: reduce on plateau (val_dice_struct)
# =============================================================================

def maybe_reduce_lr_on_plateau(
    optimizer: optim.Optimizer,
    cur_metric: float | None,
    lr_state: Dict[str, Any],
    epoch: int,
    wandb_run=None,
):
    """
    Giảm LR nếu cur_metric (val_dice_struct) không cải thiện sau 'patience' lần eval.
    lr_state gồm:
      - 'best_metric': best val_dice_struct đã thấy
      - 'epochs_no_improve': số lần eval liên tiếp không cải thiện
    """
    scfg = CFG["LR_SCHED"]
    if not scfg.get("use", False):
        return

    factor = float(scfg.get("factor", 0.5))
    patience = int(scfg.get("patience", 10))
    min_epoch = int(scfg.get("min_epoch", 0))

    if cur_metric is None:
        return
    if epoch < min_epoch:
        return

    best_metric = lr_state.get("best_metric", 0.0)
    epochs_no_improve = lr_state.get("epochs_no_improve", 0)

    # Nếu cải thiện thì reset counter + update best_metric
    if cur_metric > best_metric + 1e-6:
        lr_state["best_metric"] = cur_metric
        lr_state["epochs_no_improve"] = 0
        return

    # Không cải thiện
    epochs_no_improve += 1
    lr_state["epochs_no_improve"] = epochs_no_improve

    if epochs_no_improve >= patience:
        # Giảm LR
        for pg in optimizer.param_groups:
            pg["lr"] *= factor
        new_lr = optimizer.param_groups[0]["lr"]
        print(f"[LR] Epoch {epoch}: val_dice_struct không cải thiện {patience} lần eval, "
              f"giảm LR nhân {factor}, LR mới = {new_lr:.6g}")
        lr_state["epochs_no_improve"] = 0  # reset sau khi giảm

        if wandb_run is not None:
            wandb_run.log({"lr": new_lr, "lr/epoch": epoch})


# =============================================================================
# Train / Val one epoch
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    max_epoch: int,
    w_dice: float,
    w_ce: float,
    loss_type: str,
    wandb_run=None,
) -> Dict[str, float]:

    model.train()
    loss_meter = AverageMeter()
    ce_meter = AverageMeter()
    dice_meter = AverageMeter()

    dice_wt_meter = AverageMeter()
    dice_tc_meter = AverageMeter()
    dice_et_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}/{max_epoch}")

    num_classes = CFG["NUM_CLASSES"]

    for step, batch in enumerate(pbar):
        images = batch["image"].to(device)        # (B,4,D,H,W) - full volume đã resize
        labels = batch["label"].to(device)        # (B,1,D,H,W)
        labels = labels.squeeze(1).long()         # (B,D,H,W)

        # forward
        out = model(images)
        if isinstance(out, dict):
            logits = out["seg"]
        else:
            logits = out

        # CE
        ce_loss = F.cross_entropy(logits, labels)

        # Dice (multi-class)
        dice_loss_val = multiclass_dice_loss(
            logits, labels, num_classes=num_classes, ignore_bg=True
        )

        # combine
        if loss_type == "ce":
            loss = ce_loss
        elif loss_type == "dice":
            loss = dice_loss_val
        elif loss_type == "dicece":
            loss = w_ce * ce_loss + w_dice * dice_loss_val
        else:
            raise ValueError(f"LOSS.loss_type không hợp lệ: {loss_type}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- region Dice (WT / TC / ET) trên batch hiện tại ----
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)  # (B,D,H,W)
            region_d = brats_region_dice_torch(preds, labels)
            dice_wt = region_d["wt"]
            dice_tc = region_d["tc"]
            dice_et = region_d["et"]

        # update meters
        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        ce_meter.update(ce_loss.item(), bs)
        dice_meter.update(dice_loss_val.item(), bs)

        dice_wt_meter.update(dice_wt, bs)
        dice_tc_meter.update(dice_tc, bs)
        dice_et_meter.update(dice_et, bs)

        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "ce_loss": f"{ce_meter.avg:.4f}",
            "dice_loss": f"{dice_meter.avg:.4f}",
            "dice(WT/TC/ET)": f"{dice_wt_meter.avg:.3f}/{dice_tc_meter.avg:.3f}/{dice_et_meter.avg:.3f}",
        })

        if wandb_run is not None:
            wandb_run.log({
                "train/loss": loss.item(),
                "train/ce_loss": ce_loss.item(),
                "train/dice_loss": dice_loss_val.item(),
                "train/dice_wt": dice_wt,
                "train/dice_tc": dice_tc,
                "train/dice_et": dice_et,
                "train/epoch": epoch,
            })

    return {
        "loss": loss_meter.avg,
        "ce_loss": ce_meter.avg,
        "dice_loss": dice_meter.avg,
        "dice_wt": dice_wt_meter.avg,
        "dice_tc": dice_tc_meter.avg,
        "dice_et": dice_et_meter.avg,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    max_epoch: int,
    wandb_run=None,
) -> Dict[str, float]:
    model.eval()

    loss_meter = AverageMeter()
    ce_meter = AverageMeter()
    dice_loss_meter = AverageMeter()

    num_classes = CFG["NUM_CLASSES"]

    # theo dõi dice per-class (numpy)
    dice_class_sum = np.zeros(num_classes - 1, dtype=np.float64)
    dice_class_cnt = 0

    # theo dõi region dice WT/TC/ET
    dice_wt_meter = AverageMeter()
    dice_tc_meter = AverageMeter()
    dice_et_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"[Val]   Epoch {epoch}/{max_epoch}")

    for batch in pbar:
        images = batch["image"].to(device)          # (B,4,D,H,W)
        labels = batch["label"].to(device)          # (B,1,D,H,W)
        labels = labels.squeeze(1).long()           # (B,D,H,W)

        out = model(images)  # dùng model.eval() để tắt dropout
        if isinstance(out, dict):
            logits = out["seg"]
        else:
            logits = out

        ce = F.cross_entropy(logits, labels)
        dice_l = multiclass_dice_loss(
            logits, labels, num_classes=num_classes, ignore_bg=True
        )
        loss = ce + dice_l  # val loss = CE + Dice cho đơn giản

        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        ce_meter.update(ce.item(), bs)
        dice_loss_meter.update(dice_l.item(), bs)

        # ---- region Dice (WT / TC / ET) ----
        preds_t = torch.argmax(logits, dim=1)  # (B,D,H,W)
        region_d = brats_region_dice_torch(preds_t, labels)
        dice_wt = region_d["wt"]
        dice_tc = region_d["tc"]
        dice_et = region_d["et"]

        dice_wt_meter.update(dice_wt, bs)
        dice_tc_meter.update(dice_tc, bs)
        dice_et_meter.update(dice_et, bs)

        # ---- multi-class dice per-class via numpy (như cũ) ----
        preds = preds_t.detach().cpu().numpy()
        gts = labels.detach().cpu().numpy()

        for b in range(preds.shape[0]):
            dpc = cal_dice(preds[b], gts[b], num=num_classes)  # (num_classes-1,)
            mask_valid = ~np.isnan(dpc)
            dice_class_sum[mask_valid] += dpc[mask_valid]
            dice_class_cnt += 1

        mean_dice_fg = (dice_class_sum / max(1, dice_class_cnt)).mean()
        mean_dice_struct = (dice_wt_meter.avg + dice_tc_meter.avg + dice_et_meter.avg) / 3.0

        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "mean_dice_fg": f"{mean_dice_fg:.4f}",
            "dice(WT/TC/ET)": f"{dice_wt_meter.avg:.3f}/{dice_tc_meter.avg:.3f}/{dice_et_meter.avg:.3f}",
            "dice_struct": f"{mean_dice_struct:.3f}",
        })

    mean_dice_fg_vec = dice_class_sum / max(1, dice_class_cnt)
    mean_dice_struct = (dice_wt_meter.avg + dice_tc_meter.avg + dice_et_meter.avg) / 3.0

    if wandb_run is not None:
        log_dict = {
            "val/loss": loss_meter.avg,
            "val/ce_loss": ce_meter.avg,
            "val/dice_loss": dice_loss_meter.avg,
            "val/mean_dice_fg": float(mean_dice_fg_vec.mean()),
            "val/dice_wt": dice_wt_meter.avg,
            "val/dice_tc": dice_tc_meter.avg,
            "val/dice_et": dice_et_meter.avg,
            "val/dice_struct": mean_dice_struct,
            "val/epoch": epoch,
        }
        for c in range(1, num_classes):
            log_dict[f"val/dice_class_{c}"] = float(mean_dice_fg_vec[c - 1])
        wandb_run.log(log_dict)

    return {
        "loss": loss_meter.avg,
        "ce_loss": ce_meter.avg,
        "dice_loss": dice_loss_meter.avg,
        "mean_dice_fg": float(mean_dice_fg_vec.mean()),
        "dice_wt": dice_wt_meter.avg,
        "dice_tc": dice_tc_meter.avg,
        "dice_et": dice_et_meter.avg,
        "dice_struct": mean_dice_struct,
    }


# =============================================================================
# Checkpoint helpers
# =============================================================================

def save_checkpoint(
    state: Dict[str, Any],
    ckpt_dir: Path,
    filename: str,
):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / filename
    torch.save(state, path)
    print(f"[CKPT] Saved: {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    ckpt_path: str,
    device: torch.device,
):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val_dice = ckpt.get("best_val_dice", 0.0)
    print(f"[CKPT] Loaded checkpoint from {ckpt_path} (epoch={start_epoch-1}, best_val_dice={best_val_dice:.4f})")
    return start_epoch, best_val_dice


# =============================================================================
# MAIN
# =============================================================================

def main():
    set_seed(CFG["SEED"])
    device = get_device()

    exp_name = CFG["EXP_NAME"]
    exp_dir = ROOT / "experiments" / exp_name
    ckpt_dir = exp_dir / "checkpoints"
    log_dir = exp_dir / "logs"
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=== TRAIN VNET BRATS3D SUPERVISED (FULL VOLUME) ===")
    print(f"Root:      {ROOT}")
    print(f"Exp dir:   {exp_dir}")
    print(f"Device:    {device}")
    print(f"Volume sz: {CFG['VOLUME_SIZE']}")

    # Logger (pickle)
    train_logger = Logger(str(log_dir / "train_log.pkl"))
    val_logger = Logger(str(log_dir / "val_log.pkl"))

    # wandb
    use_wandb = CFG["WANDB"]["use_wandb"] and _HAS_WANDB
    if CFG["WANDB"]["use_wandb"] and not _HAS_WANDB:
        print("[WARN] wandb chưa cài, tắt logging wandb.")
        use_wandb = False

    wandb_run = None
    if use_wandb:
        wandb_run = wandb.init(
            project=CFG["WANDB"]["project"],
            entity=CFG["WANDB"]["entity"],
            name=exp_name,
            config=CFG,
        )

    # model & optim
    model, optimizer = build_model_and_opt(device)

    # loaders (FULL VOLUME)
    train_loader, val_loader = build_loaders()

    max_epoch = CFG["OPTIM"]["MAX_EPOCH"]
    loss_cfg = CFG["LOSS"]
    loss_type = loss_cfg["loss_type"].lower()
    w_dice = float(loss_cfg["w_dice"])
    w_ce = float(loss_cfg["w_ce"])

    # resume
    start_epoch = 1
    best_val_dice = 0.0  # avg(WT,TC,ET)
    resume_ckpt = CFG.get("RESUME_CKPT", "")
    if resume_ckpt and os.path.isfile(resume_ckpt):
        start_epoch, best_val_dice = load_checkpoint(
            model, optimizer, resume_ckpt, device
        )

    # state cho LR scheduler (reduce on plateau)
    lr_state: Dict[str, Any] = {
        "best_metric": best_val_dice,
        "epochs_no_improve": 0,
    }

    print(f"[INFO] Start training from epoch {start_epoch} / {max_epoch}, "
          f"best_val_dice={best_val_dice:.4f}")

    # main loop
    for epoch in range(start_epoch, max_epoch + 1):
        t0 = time.time()

        # ---- Train ----
        train_stats = train_one_epoch(
            model, optimizer, train_loader, device, epoch, max_epoch,
            w_dice=w_dice,
            w_ce=w_ce,
            loss_type=loss_type,
            wandb_run=wandb_run,
        )
        train_stats["epoch"] = epoch
        train_logger.log(train_stats)

        # ---- Eval (nếu đến kỳ) ----
        do_eval = (epoch % CFG["EVAL_EVERY"] == 0)
        val_stats = None
        cur_dice_struct = None
        if do_eval:
            val_stats = validate(
                model, val_loader, device, epoch, max_epoch,
                wandb_run=wandb_run,
            )
            val_stats["epoch"] = epoch
            val_logger.log(val_stats)

            cur_dice_struct = val_stats["dice_struct"]  # avg(WT,TC,ET)

            # chọn best checkpoint theo val_dice_struct
            if cur_dice_struct > best_val_dice:
                best_val_dice = cur_dice_struct
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_val_dice": best_val_dice,
                        "cfg": CFG,
                    },
                    ckpt_dir,
                    "best_checkpoint_VNet_sup.pth",
                )

            # giảm LR nếu cần (reduce on plateau)
            maybe_reduce_lr_on_plateau(
                optimizer,
                cur_metric=cur_dice_struct,
                lr_state=lr_state,
                epoch=epoch,
                wandb_run=wandb_run,
            )

        # ---- Save last + snapshot ----
        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
                "cfg": CFG,
            },
            ckpt_dir,
            "last_checkpoint_VNet_sup.pth",
        )

        if epoch % CFG["SAVE_EVERY"] == 0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_dice": best_val_dice,
                    "cfg": CFG,
                },
                ckpt_dir,
                f"epoch_{epoch:03d}_VNet_sup.pth",
            )

        dt = time.time() - t0
        msg = (
            f"[Epoch {epoch}/{max_epoch}] "
            f"train_loss={train_stats['loss']:.4f} "
            f"| train_dice(WT/TC/ET)={train_stats['dice_wt']:.3f}/"
            f"{train_stats['dice_tc']:.3f}/{train_stats['dice_et']:.3f}"
        )
        if val_stats is not None:
            msg += (
                f" | val_loss={val_stats['loss']:.4f} "
                f"| val_meanDiceFG={val_stats['mean_dice_fg']:.4f} "
                f"| val_dice(WT/TC/ET)={val_stats['dice_wt']:.3f}/"
                f"{val_stats['dice_tc']:.3f}/{val_stats['dice_et']:.3f} "
                f"| val_dice_struct={val_stats['dice_struct']:.3f}"
            )
        msg += f" | time={dt:.1f}s"
        print(msg)

    if wandb_run is not None:
        wandb_run.finish()

    print("[OK] Training finished.")


if __name__ == "__main__":
    main()
