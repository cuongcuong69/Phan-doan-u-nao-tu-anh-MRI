# -*- coding: utf-8 -*-
"""
Huấn luyện UNETR 3D cho BraTS2020 (4 modality, 4 lớp seg).

- Dùng data/processed/3d/labeled + configs/splits_2d/{train.txt,val.txt}
- Dataloader: data/dataloader_brats3d_full.py (full volume, zscore normalization)
- Model: models.unetr.UNETR (Vision Transformer + CNN decoder)
- Loss: Soft Dice + Cross-Entropy
- Metrics: Dice, IoU, HD95, ASD cho WT/TC/ET

Tích hợp:
- tqdm.auto cho progress bar
- wandb để log (nếu có)
- Eval mỗi EVAL_EVERY epoch với đầy đủ metrics
- Lưu best/last checkpoints, snapshot mỗi SAVE_EVERY epoch
- Resume từ checkpoint (RESUME_CKPT)
- Cosine annealing LR scheduler với warmup
"""

from __future__ import annotations
import os
import sys
import time
import random
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# ---------------------- WandB API Key Setup ----------------------
# Set your WandB API key here to avoid interactive login prompt
# Get your API key from: https://wandb.ai/authorize
WANDB_API_KEY = "aec3b62823dd56e12bd6c275d4ed6dc05573c49b"  # Paste your API key here, or set via environment variable

if WANDB_API_KEY:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY

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
    "EXP_NAME": "brats3d_unetr_full",
    "SEED": 2025,

    # --------------------- Data ---------------------
    "VOLUME_SIZE": (128, 128, 128),  # (D,H,W)
    "NORM_MODE": "zscore",  # Sử dụng z-score normalization

    "TRAIN_BATCH": 4,  # UNETR cần nhiều memory
    "VAL_BATCH": 4,
    "NUM_WORKERS_TRAIN": 4,
    "NUM_WORKERS_VAL": 4,

    "NUM_CHANNELS": 4,   # FLAIR, T1, T1CE, T2
    "NUM_CLASSES": 4,    # 0,1,2,3 (sau khi map 4 -> 3)

    # --------------------- Model UNETR ---------------------
    "UNETR": {
        "n_channels": 4,
        "n_classes": 4,
        "img_size": (128, 128, 128),
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "dropout": 0.1,
    },

    # --------------------- Optimizer ---------------------
    "OPTIM": {
        "LR": 1e-4,  # Thấp hơn VNet vì Transformer
        "WEIGHT_DECAY": 1e-5,
        "BETAS": (0.9, 0.999),
        "MAX_EPOCH": 100,
    },

    # --------------------- LR Scheduler (Cosine Annealing) ---------------------
    "LR_SCHED": {
        "use": True,
        "type": "cosine",  # cosine annealing with warmup
        "warmup_epochs": 10,
        "min_lr": 1e-6,
    },

    # --------------------- Loss ---------------------
    "LOSS": {
        "w_dice": 1.0,
        "w_ce": 1.0,
        "smooth": 1e-5,
    },

    # --------------------- Validation ---------------------
    "EVAL_EVERY": 1,   # validate mỗi epoch
    "COMPUTE_HD": False,  # Tính HD95/ASD (chậm, bật khi cần)

    # --------------------- Checkpoint ---------------------
    "CKPT_DIR": "checkpoints/Unetr_3d_checkpoint",
    "LOG_DIR": "logs/Unetr3d_log",
    "SAVE_EVERY": 10,
    "RESUME_CKPT": "",

    # --------------------- WandB ---------------------
    "WANDB": {
        "use_wandb": True,
        "project": "brats2020-unetr3d",
        "entity": None,   # hoặc "username"
        "resume_id": None,  # WandB run ID để resume (None = tạo run mới)
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

# data loader 3D BraTS full volume
from data.dataloader_brats3d_full import (
    build_brats3d_full_train_loader,
    build_brats3d_full_val_loader,
)

# model UNETR
from models.unetr import UNETR

# loss & metrics
from losses.combined_loss import CombinedLoss
from losses.metrics_unetr import compute_metrics_batch


# =============================================================================
# Local utilities: AverageMeter, Logger
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
    """
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Load existing data if file exists (for resume training)
        if os.path.exists(path):
            try:
                with open(path, "rb") as fp:
                    self.data = pickle.load(fp)
                print(f"[Logger] Loaded {len(self.data)} existing entries from {os.path.basename(path)}")
            except Exception as e:
                print(f"[Logger] Could not load existing log, starting fresh: {e}")
                self.data = []
        else:
            self.data = []

    def log(self, train_point: Dict[str, Any]):
        self.data.append(train_point)
        with open(self.path, "wb") as fp:
            pickle.dump(self.data, fp, -1)


# =============================================================================
# Utility: seed, device
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


# =============================================================================
# Build model, optimizer, loaders
# =============================================================================

def build_model_and_opt(device: torch.device):
    ucfg = CFG["UNETR"]
    model = UNETR(
        n_channels=ucfg["n_channels"],
        n_classes=ucfg["n_classes"],
        img_size=tuple(ucfg["img_size"]),
        patch_size=ucfg["patch_size"],
        embed_dim=ucfg["embed_dim"],
        depth=ucfg["depth"],
        num_heads=ucfg["num_heads"],
        mlp_ratio=ucfg["mlp_ratio"],
        dropout=ucfg["dropout"],
    )
    model = model.to(device)

    # Multi-GPU nếu có
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
    volume_size = CFG["VOLUME_SIZE"]
    norm_mode = CFG["NORM_MODE"]
    
    # TRAIN
    train_loader = build_brats3d_full_train_loader(
        volume_size=volume_size,
        batch_size=CFG["TRAIN_BATCH"],
        num_workers=CFG["NUM_WORKERS_TRAIN"],
        seed=CFG["SEED"],
        norm_mode=norm_mode,
    )
    # VAL
    val_loader = build_brats3d_full_val_loader(
        volume_size=volume_size,
        batch_size=CFG["VAL_BATCH"],
        num_workers=CFG["NUM_WORKERS_VAL"],
        seed=CFG["SEED"],
        norm_mode=norm_mode,
    )
    return train_loader, val_loader


# =============================================================================
# LR scheduler: Cosine Annealing with Warmup
# =============================================================================

class CosineAnnealingWarmupScheduler:
    """
    Cosine annealing learning rate scheduler with warmup.
    """
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        base_lr: float,
        min_lr: float,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        
    def step(self, epoch: int):
        """Update learning rate based on current epoch."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


# =============================================================================
# Train / Val one epoch
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loader: DataLoader,
    loss_fn: CombinedLoss,
    device: torch.device,
    epoch: int,
    max_epoch: int,
    wandb_run=None,
) -> Dict[str, float]:

    model.train()
    loss_meter = AverageMeter()
    dice_loss_meter = AverageMeter()
    ce_loss_meter = AverageMeter()

    dice_wt_meter = AverageMeter()
    dice_tc_meter = AverageMeter()
    dice_et_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}/{max_epoch}")

    for step, batch in enumerate(pbar):
        images = batch["image"].to(device)        # (B,4,D,H,W)
        labels = batch["label"].to(device)        # (B,1,D,H,W)
        labels = labels.squeeze(1).long()         # (B,D,H,W)

        # forward
        logits = model(images)  # (B, 4, D, H, W)

        # loss
        loss_dict = loss_fn(logits, labels)
        total_loss = loss_dict['total']
        dice_loss = loss_dict['dice']
        ce_loss = loss_dict['ce']

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # ---- metrics (WT / TC / ET) ----
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)  # (B,D,H,W)
            metrics = compute_metrics_batch(preds, labels, num_classes=4, compute_hd=False)
            
            dice_wt = metrics.get('dice_wt', 0.0)
            dice_tc = metrics.get('dice_tc', 0.0)
            dice_et = metrics.get('dice_et', 0.0)

        # update meters
        bs = images.size(0)
        loss_meter.update(total_loss.item(), bs)
        dice_loss_meter.update(dice_loss.item(), bs)
        ce_loss_meter.update(ce_loss.item(), bs)

        dice_wt_meter.update(dice_wt, bs)
        dice_tc_meter.update(dice_tc, bs)
        dice_et_meter.update(dice_et, bs)

        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "dice_loss": f"{dice_loss_meter.avg:.4f}",
            "ce_loss": f"{ce_loss_meter.avg:.4f}",
            "dice(WT/TC/ET)": f"{dice_wt_meter.avg:.3f}/{dice_tc_meter.avg:.3f}/{dice_et_meter.avg:.3f}",
        })

        if wandb_run is not None:
            wandb_run.log({
                "train/loss": total_loss.item(),
                "train/dice_loss": dice_loss.item(),
                "train/ce_loss": ce_loss.item(),
                "train/dice_wt": dice_wt,
                "train/dice_tc": dice_tc,
                "train/dice_et": dice_et,
                "train/epoch": epoch,
            })

    return {
        "loss": loss_meter.avg,
        "dice_loss": dice_loss_meter.avg,
        "ce_loss": ce_loss_meter.avg,
        "dice_wt": dice_wt_meter.avg,
        "dice_tc": dice_tc_meter.avg,
        "dice_et": dice_et_meter.avg,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: CombinedLoss,
    device: torch.device,
    epoch: int,
    max_epoch: int,
    compute_hd: bool = False,
    wandb_run=None,
) -> Dict[str, float]:
    model.eval()

    loss_meter = AverageMeter()
    dice_loss_meter = AverageMeter()
    ce_loss_meter = AverageMeter()

    # Metrics accumulators
    dice_wt_meter = AverageMeter()
    dice_tc_meter = AverageMeter()
    dice_et_meter = AverageMeter()
    
    iou_wt_meter = AverageMeter()
    iou_tc_meter = AverageMeter()
    iou_et_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"[Val]   Epoch {epoch}/{max_epoch}")

    for batch in pbar:
        images = batch["image"].to(device)          # (B,4,D,H,W)
        labels = batch["label"].to(device)          # (B,1,D,H,W)
        labels = labels.squeeze(1).long()           # (B,D,H,W)

        logits = model(images)

        # loss
        loss_dict = loss_fn(logits, labels)
        total_loss = loss_dict['total']
        dice_loss = loss_dict['dice']
        ce_loss = loss_dict['ce']

        bs = images.size(0)
        loss_meter.update(total_loss.item(), bs)
        dice_loss_meter.update(dice_loss.item(), bs)
        ce_loss_meter.update(ce_loss.item(), bs)

        # ---- metrics ----
        preds = torch.argmax(logits, dim=1)  # (B,D,H,W)
        metrics = compute_metrics_batch(preds, labels, num_classes=4, compute_hd=compute_hd)

        dice_wt = metrics.get('dice_wt', 0.0)
        dice_tc = metrics.get('dice_tc', 0.0)
        dice_et = metrics.get('dice_et', 0.0)
        
        iou_wt = metrics.get('iou_wt', 0.0)
        iou_tc = metrics.get('iou_tc', 0.0)
        iou_et = metrics.get('iou_et', 0.0)

        dice_wt_meter.update(dice_wt, bs)
        dice_tc_meter.update(dice_tc, bs)
        dice_et_meter.update(dice_et, bs)
        
        iou_wt_meter.update(iou_wt, bs)
        iou_tc_meter.update(iou_tc, bs)
        iou_et_meter.update(iou_et, bs)

        mean_dice = (dice_wt_meter.avg + dice_tc_meter.avg + dice_et_meter.avg) / 3.0

        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "dice(WT/TC/ET)": f"{dice_wt_meter.avg:.3f}/{dice_tc_meter.avg:.3f}/{dice_et_meter.avg:.3f}",
            "mean_dice": f"{mean_dice:.3f}",
        })

    mean_dice = (dice_wt_meter.avg + dice_tc_meter.avg + dice_et_meter.avg) / 3.0
    mean_iou = (iou_wt_meter.avg + iou_tc_meter.avg + iou_et_meter.avg) / 3.0

    if wandb_run is not None:
        log_dict = {
            "val/loss": loss_meter.avg,
            "val/dice_loss": dice_loss_meter.avg,
            "val/ce_loss": ce_loss_meter.avg,
            "val/dice_wt": dice_wt_meter.avg,
            "val/dice_tc": dice_tc_meter.avg,
            "val/dice_et": dice_et_meter.avg,
            "val/mean_dice": mean_dice,
            "val/iou_wt": iou_wt_meter.avg,
            "val/iou_tc": iou_tc_meter.avg,
            "val/iou_et": iou_et_meter.avg,
            "val/mean_iou": mean_iou,
            "val/epoch": epoch,
        }
        wandb_run.log(log_dict)

    return {
        "loss": loss_meter.avg,
        "dice_loss": dice_loss_meter.avg,
        "ce_loss": ce_loss_meter.avg,
        "dice_wt": dice_wt_meter.avg,
        "dice_tc": dice_tc_meter.avg,
        "dice_et": dice_et_meter.avg,
        "mean_dice": mean_dice,
        "iou_wt": iou_wt_meter.avg,
        "iou_tc": iou_tc_meter.avg,
        "iou_et": iou_et_meter.avg,
        "mean_iou": mean_iou,
    }


# =============================================================================
# Checkpoint helpers
# =============================================================================

import shutil
import argparse

# ... (rest of imports)

def save_checkpoint(
    state: Dict[str, Any],
    ckpt_dir: Path,
    filename: str,
    drive_dir: Optional[Path] = None,
):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / filename
    torch.save(state, path)
    print(f"[CKPT] Saved: {path}")

    if drive_dir:
        drive_path = drive_dir / filename
        try:
            # Delete old file if exists to avoid Drive versioning
            if drive_path.exists():
                os.remove(str(drive_path))
            
            shutil.copy(path, drive_path)
            print(f"[CKPT] Copied to Drive: {drive_path}")
        except Exception as e:
            print(f"[WARN] Failed to copy checkpoint to Drive: {e}")


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    ckpt_path: str,
    device: torch.device,
):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive_path", type=str, default=None, help="Path to Google Drive checkpoint folder")
    args = parser.parse_args()
    
    drive_dir = Path(args.drive_path) if args.drive_path else None
    if drive_dir:
        drive_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Auto-save checkpoints to Drive: {drive_dir}")

    set_seed(CFG["SEED"])
    device = get_device()

    exp_name = CFG["EXP_NAME"]
    ckpt_dir = ROOT / CFG["CKPT_DIR"]
    log_dir = ROOT / CFG["LOG_DIR"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=== TRAIN UNETR BRATS3D ===")
    print(f"Root:      {ROOT}")
    print(f"Ckpt dir:  {ckpt_dir}")
    print(f"Log dir:   {log_dir}")
    print(f"Drive dir: {drive_dir}")
    print(f"Device:    {device}")
    print(f"Norm mode: {CFG['NORM_MODE']}")

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
        resume_id = CFG["WANDB"].get("resume_id", None)
        if resume_id:
            print(f"[INFO] Resuming WandB run with ID: {resume_id}")
            wandb_run = wandb.init(
                project=CFG["WANDB"]["project"],
                entity=CFG["WANDB"]["entity"],
                id=resume_id,
                resume="must",
                config=CFG,
            )
        else:
            wandb_run = wandb.init(
                project=CFG["WANDB"]["project"],
                entity=CFG["WANDB"]["entity"],
                name=exp_name,
                config=CFG,
            )

    # model & optim
    model, optimizer = build_model_and_opt(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Total parameters: {total_params:,}")

    # loaders
    train_loader, val_loader = build_loaders()

    # loss function
    loss_cfg = CFG["LOSS"]
    loss_fn = CombinedLoss(
        num_classes=CFG["NUM_CLASSES"],
        w_dice=loss_cfg["w_dice"],
        w_ce=loss_cfg["w_ce"],
        smooth=loss_cfg["smooth"],
        ignore_bg=True,
    ).to(device)

    max_epoch = CFG["OPTIM"]["MAX_EPOCH"]

    # LR scheduler
    scheduler = None
    if CFG["LR_SCHED"]["use"]:
        scheduler = CosineAnnealingWarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=CFG["LR_SCHED"]["warmup_epochs"],
            max_epochs=max_epoch,
            base_lr=CFG["OPTIM"]["LR"],
            min_lr=CFG["LR_SCHED"]["min_lr"],
        )

    # resume
    start_epoch = 1
    best_val_dice = 0.0
    resume_ckpt = CFG.get("RESUME_CKPT", "")
    if resume_ckpt and os.path.isfile(resume_ckpt):
        start_epoch, best_val_dice = load_checkpoint(
            model, optimizer, resume_ckpt, device
        )

    print(f"[INFO] Start training from epoch {start_epoch} / {max_epoch}, "
          f"best_val_dice={best_val_dice:.4f}")

    # main loop
    for epoch in range(start_epoch, max_epoch + 1):
        t0 = time.time()

        # Update LR
        if scheduler is not None:
            current_lr = scheduler.step(epoch - 1)  # epoch is 1-indexed
            if wandb_run is not None:
                wandb_run.log({"lr": current_lr, "lr/epoch": epoch})

        # ---- Train ----
        train_stats = train_one_epoch(
            model, optimizer, train_loader, loss_fn, device, epoch, max_epoch,
            wandb_run=wandb_run,
        )
        train_stats["epoch"] = epoch
        train_logger.log(train_stats)

        # ---- Eval (nếu đến kỳ) ----
        do_eval = (epoch % CFG["EVAL_EVERY"] == 0)
        val_stats = None
        cur_mean_dice = None
        if do_eval:
            val_stats = validate(
                model, val_loader, loss_fn, device, epoch, max_epoch,
                compute_hd=CFG["COMPUTE_HD"],
                wandb_run=wandb_run,
            )
            val_stats["epoch"] = epoch
            val_logger.log(val_stats)

            cur_mean_dice = val_stats["mean_dice"]

            # chọn best checkpoint theo mean_dice
            if cur_mean_dice > best_val_dice:
                best_val_dice = cur_mean_dice
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_val_dice": best_val_dice,
                        "cfg": CFG,
                    },
                    ckpt_dir,
                    "best_checkpoint_unetr.pth",
                    drive_dir=drive_dir
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
            "last_checkpoint_unetr.pth",
            drive_dir=drive_dir
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
                f"epoch_{epoch:03d}_unetr.pth",
                drive_dir=drive_dir
            )

        # ---- Copy logs to Drive ----
        if drive_dir:
            # Logs go to separate logs folder, not checkpoints folder
            drive_log_dir = drive_dir.parent / "logs" / "Unetr3d_log"
            drive_log_dir.mkdir(parents=True, exist_ok=True)
            
            for log_file in ["train_log.pkl", "val_log.pkl"]:
                local_log_path = log_dir / log_file
                drive_log_path = drive_log_dir / log_file
                
                try:
                    # Delete old file if exists to avoid Drive versioning
                    if drive_log_path.exists():
                        os.remove(str(drive_log_path))
                    
                    shutil.copy2(local_log_path, drive_log_path)
                except Exception as e:
                    print(f"[WARN] Failed to copy {log_file} to Drive: {e}")


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
                f"| val_dice(WT/TC/ET)={val_stats['dice_wt']:.3f}/"
                f"{val_stats['dice_tc']:.3f}/{val_stats['dice_et']:.3f} "
                f"| val_mean_dice={val_stats['mean_dice']:.3f}"
            )
        if scheduler is not None:
            msg += f" | LR={optimizer.param_groups[0]['lr']:.6f}"
        msg += f" | time={dt:.1f}s"
        print(msg)

    if wandb_run is not None:
        wandb_run.finish()

    print("[OK] Training finished.")


if __name__ == "__main__":
    main()
