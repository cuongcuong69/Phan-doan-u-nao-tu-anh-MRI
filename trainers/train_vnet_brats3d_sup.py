# -*- coding: utf-8 -*-
"""
Huấn luyện VNet 3D đa lớp cho BraTS2020 (4 modality, 4 lớp seg).

- Dùng data/processed/3d/labeled + configs/splits_2d/{train.txt,val.txt,test.txt}
  (theo Brain_ID).
- Dataloader: data/dataloader_brats3d_sup.py (build_brats3d_sup_*_loader).
- Model: models.vnet.VNet (n_channels=4, n_classes=4).
- Loss: CE / Dice / Dice+CE (w_ce, w_dice).
- Tùy chọn SDF loss: nếu model trả ra {"seg": logits, "sdf": sdf_pred} và dataset có "sdf".

Tích hợp:
- tqdm.auto cho progress bar
- wandb để log (nếu có)
- Eval mỗi EVAL_EVERY epoch, tính multi-class Dice
- Lưu best/last checkpoints, snapshot mỗi SAVE_EVERY epoch
- Resume từ checkpoint (RESUME_CKPT)
- Giảm LR: sau LR_DECAY_START và mỗi LR_DECAY_EVERY epoch nhân LR với LR_DECAY_FACTOR
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
    "EXP_NAME": "brats3d_vnet_sup2",
    "SEED": 2025,

    # --------------------- Data ---------------------
    "PATCH_SIZE": (64, 64, 64), # (D,H,W)

    "TRAIN_BATCH": 8,
    "VAL_BATCH": 4,
    "NUM_WORKERS_TRAIN": 0,
    "NUM_WORKERS_VAL": 0,

    "NUM_CHANNELS": 4,   # FLAIR, T1, T1CE, T2
    "NUM_CLASSES": 4,    # 0,1,2,3 (sau khi map 4 -> 3)

    # --------------------- Model ---------------------
    "VNET": {
        "n_channels": 4,
        "n_classes": 4,
        "n_filters": 32,
        "normalization": "instancenorm",  # "batchnorm" | "groupnorm" | "instancenorm" | "none"
        "has_dropout": True,
    },

    # --------------------- Optimizer ---------------------
    "OPTIM": {
        "LR": 1e-3,
        "WEIGHT_DECAY": 1e-4,
        "BETAS": (0.9, 0.999),
        "MAX_EPOCH": 200,
    },

    # --------------------- LR Scheduler (multi-step) ---------------------
    # Giảm LR bằng cách nhân LR_DECAY_FACTOR sau LR_DECAY_START
    # và sau đó MỖI LR_DECAY_EVERY epoch.
    "LR_SCHED": {
        "use": True,
        "start_epoch": 50,        # bắt đầu giảm từ epoch này
        "every": 50,              # sau mỗi N epoch lại giảm (50, 100, 150, ...)
        "factor": 0.5,            # LR = LR * factor (0.1 => giảm 10 lần)
    },

    # --------------------- Loss ---------------------
    "LOSS": {
        # "ce" | "dice" | "dicece"
        "loss_type": "dicece",

        "w_dice": 1.0,
        "w_ce": 1.0,

        # chỉ dùng nếu model có nhánh SDF riêng
        "use_sdf_loss": False,
        "w_sdf": 0.1,
    },

    # --------------------- Sampling (patch) ---------------------
    # sampling_mode: "random"|"rejection"|"center_fg"|"mixed"
    "TRAIN_SAMPLING_MODE": "mixed",
    "VAL_SAMPLING_MODE": "mixed",   # bạn yêu cầu val cũng mixed
    "REJECTION_THRESH": 0.01,
    "REJECTION_MAX": 8,
    "TRAIN_MIXED_WEIGHTS": {"center_fg": 0.3, "random": 0.7},
    "VAL_MIXED_WEIGHTS":   {"center_fg": 0.5, "random": 0.5},

    # --------------------- Validation ---------------------
    "EVAL_EVERY": 5,   # validate mỗi epoch

    # --------------------- Checkpoint ---------------------
    "SAVE_EVERY": 25,
    "RESUME_CKPT": r"D:\Project Advanced CV\experiments\brats3d_vnet_sup2\checkpoints\last_checkpoint_VNet_sup.pth",     # đường dẫn ckpt để resume (nếu có)

    # --------------------- WandB ---------------------
    "WANDB": {
        "use_wandb": True,
        "project": "brats2020-vnet3d-sup",
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

# data loader 3D BraTS đã viết
from data.dataloader_brats3d_sup import (
    build_brats3d_sup_train_loader,
    build_brats3d_sup_val_loader,
)

# model VNet
from models.vnet import VNet


# =============================================================================
# Local utilities: AverageMeter, Logger, cal_dice
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
    probs = torch.softmax(logits, dim=1)               # (N,C,D,H,W)
    one_hot = F.one_hot(targets.long(), num_classes=num_classes)  # (N,D,H,W,C)
    one_hot = one_hot.permute(0, 4, 1, 2, 3).float()             # (N,C,D,H,W)

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
    patch_size = CFG["PATCH_SIZE"]
    # TRAIN
    train_loader = build_brats3d_sup_train_loader(
        patch_size=patch_size,
        batch_size=CFG["TRAIN_BATCH"],
        num_workers=CFG["NUM_WORKERS_TRAIN"],
        with_sdf=False,  # nếu dataset có SDF, bật True & sửa dataloader
        sampling_mode=CFG["TRAIN_SAMPLING_MODE"],
        rejection_thresh=CFG["REJECTION_THRESH"],
        rejection_max=CFG["REJECTION_MAX"],
        mixed_weights=CFG["TRAIN_MIXED_WEIGHTS"],
        seed=CFG["SEED"],
    )
    # VAL
    val_loader = build_brats3d_sup_val_loader(
        patch_size=patch_size,
        batch_size=CFG["VAL_BATCH"],
        num_workers=CFG["NUM_WORKERS_VAL"],
        with_sdf=False,    # tương tự
        sampling_mode=CFG["VAL_SAMPLING_MODE"],
        rejection_thresh=CFG["REJECTION_THRESH"],
        rejection_max=CFG["REJECTION_MAX"],
        mixed_weights=CFG["VAL_MIXED_WEIGHTS"],
        seed=CFG["SEED"],
    )
    return train_loader, val_loader


# =============================================================================
# LR scheduler helper
# =============================================================================

def maybe_adjust_lr(optimizer: optim.Optimizer, epoch: int, wandb_run=None):
    """
    Giảm LR bằng cách nhân factor sau start_epoch
    và sau đó mỗi 'every' epoch.
    Ví dụ: start_epoch=50, every=50, factor=0.1
    => lr giảm ở epoch 50, 100, 150, ...
    """
    scfg = CFG["LR_SCHED"]
    if not scfg["use"]:
        return

    start = int(scfg["start_epoch"])
    every = int(scfg["every"])
    factor = float(scfg["factor"])

    if epoch >= start and (epoch - start) % every == 0:
        for pg in optimizer.param_groups:
            pg["lr"] *= factor
        new_lr = optimizer.param_groups[0]["lr"]
        print(f"[LR] Epoch {epoch}: giảm LR, nhân {factor}, LR mới = {new_lr:.6g}")
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
    use_sdf_loss: bool,
    w_dice: float,
    w_ce: float,
    w_sdf: float,
    loss_type: str,
    wandb_run=None,
) -> Dict[str, float]:

    model.train()
    loss_meter = AverageMeter()
    ce_meter = AverageMeter()
    dice_meter = AverageMeter()
    sdf_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}/{max_epoch}")

    num_classes = CFG["NUM_CLASSES"]

    for step, batch in enumerate(pbar):
        images = batch["image"].to(device)        # (B,4,D,H,W)
        labels = batch["label"].to(device)        # (B,1,D,H,W)
        labels = labels.squeeze(1).long()         # (B,D,H,W)

        # forward
        out = model(images)
        if isinstance(out, dict):
            logits = out["seg"]
            sdf_pred = out.get("sdf", None)
        else:
            logits = out
            sdf_pred = None

        # CE
        ce_loss = F.cross_entropy(logits, labels)

        # Dice
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

        # optional SDF auxiliary loss
        sdf_loss_val = torch.tensor(0.0, device=device)
        if use_sdf_loss and sdf_pred is not None and ("sdf" in batch):
            sdf_gt = batch["sdf"].to(device).float()  # (B,1,D,H,W)
            sdf_loss_val = F.smooth_l1_loss(sdf_pred, sdf_gt)
            loss = loss + w_sdf * sdf_loss_val

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update meters
        loss_meter.update(loss.item(), images.size(0))
        ce_meter.update(ce_loss.item(), images.size(0))
        dice_meter.update(dice_loss_val.item(), images.size(0))
        sdf_meter.update(sdf_loss_val.item(), images.size(0))

        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "ce_loss": f"{ce_meter.avg:.4f}",
            "dice_loss": f"{dice_meter.avg:.4f}",
        })

        if wandb_run is not None:
            wandb_run.log({
                "train/loss": loss.item(),
                "train/ce_loss": ce_loss.item(),
                "train/dice_loss": dice_loss_val.item(),
                "train/sdf_loss": sdf_loss_val.item(),
                "train/epoch": epoch,
            })

    return {
        "loss": loss_meter.avg,
        "ce_loss": ce_meter.avg,
        "dice_loss": dice_meter.avg,
        "sdf_loss": sdf_meter.avg,
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

        loss_meter.update(loss.item(), images.size(0))
        ce_meter.update(ce.item(), images.size(0))
        dice_loss_meter.update(dice_l.item(), images.size(0))

        # chuyển về numpy để tính dice per-class (cal_dice)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()  # (B,D,H,W)
        gts = labels.detach().cpu().numpy()

        for b in range(preds.shape[0]):
            dpc = cal_dice(preds[b], gts[b], num=num_classes)  # (num_classes-1,)
            mask_valid = ~np.isnan(dpc)
            dice_class_sum[mask_valid] += dpc[mask_valid]
            dice_class_cnt += 1

        mean_dice_fg = (dice_class_sum / max(1, dice_class_cnt)).mean()

        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "mean_dice": f"{mean_dice_fg:.4f}",
        })

    mean_dice_fg_vec = dice_class_sum / max(1, dice_class_cnt)

    if wandb_run is not None:
        log_dict = {
            "val/loss": loss_meter.avg,
            "val/ce_loss": ce_meter.avg,
            "val/dice_loss": dice_loss_meter.avg,
            "val/mean_dice_fg": float(mean_dice_fg_vec.mean()),
            "val/epoch": epoch,
        }
        for c in range(1, num_classes):
            log_dict[f"val/dice_class_{c}"] = float(mean_dice_fg_vec[c-1])
        wandb_run.log(log_dict)

    return {
        "loss": loss_meter.avg,
        "ce_loss": ce_meter.avg,
        "dice_loss": dice_loss_meter.avg,
        "mean_dice_fg": float(mean_dice_fg_vec.mean()),
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

    print("=== TRAIN VNET BRATS3D SUPERVISED ===")
    print(f"Root:      {ROOT}")
    print(f"Exp dir:   {exp_dir}")
    print(f"Device:    {device}")

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

    # loaders
    train_loader, val_loader = build_loaders()

    max_epoch = CFG["OPTIM"]["MAX_EPOCH"]
    loss_cfg = CFG["LOSS"]
    loss_type = loss_cfg["loss_type"].lower()
    w_dice = float(loss_cfg["w_dice"])
    w_ce = float(loss_cfg["w_ce"])
    use_sdf_loss = bool(loss_cfg["use_sdf_loss"])
    w_sdf = float(loss_cfg["w_sdf"])

    # resume
    start_epoch = 1
    best_val_dice = 0.0
    resume_ckpt = CFG.get("RESUME_CKPT", "")
    if resume_ckpt and os.path.isfile(resume_ckpt):
        start_epoch, best_val_dice = load_checkpoint(
            model, optimizer, resume_ckpt, device
        )

    print(f"[INFO] Start training from epoch {start_epoch} / {max_epoch}, best_val_dice={best_val_dice:.4f}")

    # main loop
    for epoch in range(start_epoch, max_epoch + 1):
        t0 = time.time()

        # ---- Train ----
        train_stats = train_one_epoch(
            model, optimizer, train_loader, device, epoch, max_epoch,
            use_sdf_loss=use_sdf_loss,
            w_dice=w_dice,
            w_ce=w_ce,
            w_sdf=w_sdf,
            loss_type=loss_type,
            wandb_run=wandb_run,
        )
        train_stats["epoch"] = epoch
        train_logger.log(train_stats)

        # ---- Eval (nếu đến kỳ) ----
        do_eval = (epoch % CFG["EVAL_EVERY"] == 0)
        val_stats = None
        if do_eval:
            val_stats = validate(
                model, val_loader, device, epoch, max_epoch,
                wandb_run=wandb_run,
            )
            val_stats["epoch"] = epoch
            val_logger.log(val_stats)

            cur_dice = val_stats["mean_dice_fg"]
            if cur_dice > best_val_dice:
                best_val_dice = cur_dice
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

        # ---- Giảm LR nếu tới mốc (50, 100, 150, ...) ----
        maybe_adjust_lr(optimizer, epoch, wandb_run=wandb_run)

        dt = time.time() - t0
        msg = f"[Epoch {epoch}/{max_epoch}] train_loss={train_stats['loss']:.4f}"
        if val_stats is not None:
            msg += f" | val_loss={val_stats['loss']:.4f} | val_meanDice={val_stats['mean_dice_fg']:.4f}"
        msg += f" | time={dt:.1f}s"
        print(msg)

    if wandb_run is not None:
        wandb_run.finish()

    print("[OK] Training finished.")


if __name__ == "__main__":
    main()
