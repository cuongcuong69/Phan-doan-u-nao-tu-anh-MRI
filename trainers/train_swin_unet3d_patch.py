from __future__ import annotations
import os
import sys
import time
import random
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

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
    "EXP_NAME": "swin_unet3d_patch128",
    "SEED": 2025,

    # --------------------- Data (PATCH) ---------------------
    "PATCH_SIZE": (128, 128, 128),
    "TRAIN_BATCH": 1,
    "VAL_BATCH": 1,
    "TEST_BATCH": 1,
    "NUM_WORKERS_TRAIN": 4,
    "NUM_WORKERS_VAL": 4,
    "NUM_WORKERS_TEST": 4,

    "NUM_CHANNELS": 4,
    "NUM_CLASSES": 4,

    # --------------------- Sampling (patch) ---------------------
    "TRAIN_SAMPLING_MODE": "mixed",
    "VAL_SAMPLING_MODE": "mixed",
    "TEST_SAMPLING_MODE": "mixed",
    "REJECTION_THRESH": 0.01,
    "REJECTION_MAX": 8,
    "TRAIN_MIXED_WEIGHTS": {"center_fg": 0.7, "random": 0.3},
    "VAL_MIXED_WEIGHTS":   {"center_fg": 1, "random": 0},
    "TEST_MIXED_WEIGHTS":  {"center_fg": 1, "random": 0},
    
    
    

    # --------------------- Model ---------------------
    "SWIN_UNET3D": {
        "in_channels": 4,
        "num_classes": 4,
        "embed_dim": 96,
        "depths": (2, 2, 2, 2),
        "num_heads": (3, 6, 12, 24),
        "window_size": (4, 4, 4),
        "mlp_ratio": 4.0,
    },

    # --------------------- Optimizer ---------------------
    "OPTIM": {
        "LR": 5e-4,
        "WEIGHT_DECAY": 1e-2,
        "BETAS": (0.9, 0.999),
        "MAX_EPOCH": 120,
        "USE_AMP": True,
        "GRAD_CLIP_NORM": 1.0,
    },

    # --------------------- LR Scheduler ---------------------
    "SCHED": {
        "use": True,
        "warmup_ratio": 0.1,
        "min_lr": 2e-6,
    },

    # --------------------- Loss ---------------------
    "LOSS": {
        "loss_type": "dicece",   # "ce" | "dice" | "dicece"
        "w_dice": 1.0,
        "w_ce": 1.0,
    },

    # --------------------- Validation ---------------------
    "EVAL_EVERY": 1,

    # --------------------- Checkpoint ---------------------
    "SAVE_EVERY": 50,
    # Nên để path tuyệt đối theo ROOT cho chắc, mình sẽ resolve tự động.
    "RESUME_CKPT":  "",  # ví dụ: "experiments/swin_unet3d_patch128/checkpoints/last_checkpoint_SwinUNet3D_patch128.pth"

    # --------------------- WandB ---------------------
    "WANDB": {
        "use_wandb": True,
        "project": "brats2020-swinunet3d-sup-patch",
        "entity": None,
    },

    # --------------------- Device ---------------------
    "DEVICE": "cuda",
}


# =============================================================================
# PATH & IMPORTS RELATIVE TO PROJECT ROOT
# =============================================================================
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data.dataloader_brats3d_task01_sup import (
    build_brats3d_task01_sup_train_loader,
    build_brats3d_sup_val_loader,
    build_brats3d_sup_test_loader,
)

from models.swin_unet_3d import SwinUnet3D, SwinUnet3DConfig


# =============================================================================
# Local utilities
# =============================================================================
class AverageMeter:
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
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.data = []

    def log(self, point: Dict[str, Any]):
        self.data.append(point)
        with open(self.path, "wb") as fp:
            pickle.dump(self.data, fp, -1)


def cal_dice(prediction: np.ndarray, label: np.ndarray, num: int = 2) -> np.ndarray:
    total_dice = np.zeros(num - 1, dtype=np.float32)
    for i in range(1, num):
        p = (prediction == i).astype(float)
        g = (label == i).astype(float)
        denom = (p.sum() + g.sum())
        total_dice[i - 1] = (2.0 * (p * g).sum() / denom) if denom > 0 else 1.0
    return total_dice


def dice_binary_torch(pred_bin: torch.Tensor, gt_bin: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    pred_f = pred_bin.float()
    gt_f = gt_bin.float()
    inter = (pred_f * gt_f).sum()
    den = pred_f.sum() + gt_f.sum()
    if den.item() == 0:
        return torch.tensor(1.0, device=pred_bin.device)
    return (2.0 * inter + eps) / (den + eps)


def brats_region_dice_torch(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    mp_wt = pred > 0
    mg_wt = gt > 0
    dice_wt = dice_binary_torch(mp_wt, mg_wt).item()

    mp_tc = (pred == 1) | (pred == 3)
    mg_tc = (gt == 1) | (gt == 3)
    dice_tc = dice_binary_torch(mp_tc, mg_tc).item()

    mp_et = (pred == 3)
    mg_et = (gt == 3)
    dice_et = dice_binary_torch(mp_et, mg_et).item()
    return {"wt": dice_wt, "tc": dice_tc, "et": dice_et}


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
    probs = torch.softmax(logits, dim=1)  # (N,C,D,H,W)
    one_hot = F.one_hot(targets.long(), num_classes=num_classes)  # (N,D,H,W,C)
    one_hot = one_hot.permute(0, 4, 1, 2, 3).float()  # (N,C,D,H,W)

    dims = (0, 2, 3, 4)
    inter = torch.sum(probs * one_hot, dims)
    card = torch.sum(probs + one_hot, dims)
    dice_pc = (2.0 * inter + eps) / (card + eps)

    if ignore_bg and dice_pc.numel() > 1:
        return 1.0 - dice_pc[1:].mean()
    return 1.0 - dice_pc.mean()


# =============================================================================
# Warmup + Cosine scheduler (step-based)
# =============================================================================
class WarmupCosineLR:
    def __init__(self, optimizer: optim.Optimizer, base_lr: float, total_steps: int,
                 warmup_steps: int, min_lr: float = 1e-6, last_step: int = 0):
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)

        self.step_idx = int(last_step)
        # set initial lr according to last_step
        self._set_lr(self._compute_lr(self.step_idx))

        print(f"[SCHED] WarmupCosineLR: total_steps={self.total_steps}, "
              f"warmup_steps={self.warmup_steps}, min_lr={self.min_lr}, resume_step={self.step_idx}")

    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def get_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _compute_lr(self, step_idx: int) -> float:
        t = max(0, int(step_idx))
        if self.total_steps <= 0:
            return self.base_lr
        if self.warmup_steps > 0 and t <= self.warmup_steps:
            return self.base_lr * t / max(1, self.warmup_steps)

        if self.total_steps == self.warmup_steps:
            return self.min_lr

        t2 = min(t, self.total_steps)
        progress = (t2 - self.warmup_steps) / max(1, (self.total_steps - self.warmup_steps))
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def step(self):
        self.step_idx += 1
        lr = self._compute_lr(self.step_idx)
        self._set_lr(float(lr))


# =============================================================================
# Build model/optimizer/loaders
# =============================================================================
def build_model_and_opt(device: torch.device):
    mcfg = CFG["SWIN_UNET3D"]
    cfg = SwinUnet3DConfig(
        in_channels=int(mcfg["in_channels"]),
        num_classes=int(mcfg["num_classes"]),
        embed_dim=int(mcfg["embed_dim"]),
        depths=tuple(mcfg["depths"]),
        num_heads=tuple(mcfg["num_heads"]),
        window_size=tuple(mcfg["window_size"]),
        mlp_ratio=float(mcfg["mlp_ratio"]),
    )
    model = SwinUnet3D(cfg).to(device)

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        print(f"[INFO] Dùng DataParallel trên {torch.cuda.device_count()} GPU")
        model = torch.nn.DataParallel(model)

    ocfg = CFG["OPTIM"]
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(ocfg["LR"]),
        weight_decay=float(ocfg["WEIGHT_DECAY"]),
        betas=tuple(ocfg["BETAS"]),
    )
    return model, optimizer


def build_loaders():
    patch_size = CFG["PATCH_SIZE"]

    train_loader = build_brats3d_task01_sup_train_loader(
        patch_size=patch_size,
        batch_size=CFG["TRAIN_BATCH"],
        num_workers=CFG["NUM_WORKERS_TRAIN"],
        sampling_mode=CFG["TRAIN_SAMPLING_MODE"],
        rejection_thresh=CFG["REJECTION_THRESH"],
        rejection_max=CFG["REJECTION_MAX"],
        mixed_weights=CFG["TRAIN_MIXED_WEIGHTS"],
        seed=CFG["SEED"],
    )

    val_loader = build_brats3d_sup_val_loader(
        patch_size=patch_size,
        batch_size=CFG["VAL_BATCH"],
        num_workers=CFG["NUM_WORKERS_VAL"],
        sampling_mode=CFG["VAL_SAMPLING_MODE"],
        rejection_thresh=CFG["REJECTION_THRESH"],
        rejection_max=CFG["REJECTION_MAX"],
        mixed_weights=CFG["VAL_MIXED_WEIGHTS"],
        seed=CFG["SEED"],
    )

    test_loader = build_brats3d_sup_test_loader(
        patch_size=patch_size,
        batch_size=CFG["TEST_BATCH"],
        num_workers=CFG["NUM_WORKERS_TEST"],
        sampling_mode=CFG["TEST_SAMPLING_MODE"],
        rejection_thresh=CFG["REJECTION_THRESH"],
        rejection_max=CFG["REJECTION_MAX"],
        mixed_weights=CFG["TEST_MIXED_WEIGHTS"],
        seed=CFG["SEED"],
    )
    return train_loader, val_loader, test_loader


# =============================================================================
# Train / Val/Test
# =============================================================================
def train_one_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[WarmupCosineLR],
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    max_epoch: int,
    w_dice: float,
    w_ce: float,
    loss_type: str,
    use_amp: bool,
    grad_clip_norm: float,
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

    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    for _, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).squeeze(1).long()

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=("cuda" if device.type == "cuda" else "cpu"),
            enabled=(use_amp and device.type == "cuda"),
            dtype=torch.float16
        ):
            logits = model(images)
            if isinstance(logits, dict):
                logits = logits.get("seg", logits)

            ce_loss = F.cross_entropy(logits, labels)
            dice_loss_val = multiclass_dice_loss(logits, labels, num_classes=num_classes, ignore_bg=True)

            if loss_type == "ce":
                loss = ce_loss
            elif loss_type == "dice":
                loss = dice_loss_val
            elif loss_type == "dicece":
                loss = w_ce * ce_loss + w_dice * dice_loss_val
            else:
                raise ValueError(f"LOSS.loss_type không hợp lệ: {loss_type}")

        scaler.scale(loss).backward()

        if grad_clip_norm and grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()
            cur_lr = scheduler.get_lr()
        else:
            cur_lr = optimizer.param_groups[0]["lr"]

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            region_d = brats_region_dice_torch(preds, labels)

        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        ce_meter.update(ce_loss.item(), bs)
        dice_meter.update(dice_loss_val.item(), bs)
        dice_wt_meter.update(region_d["wt"], bs)
        dice_tc_meter.update(region_d["tc"], bs)
        dice_et_meter.update(region_d["et"], bs)

        pbar.set_postfix({
            "lr": f"{cur_lr:.2e}",
            "loss": f"{loss_meter.avg:.4f}",
            "ce": f"{ce_meter.avg:.4f}",
            "diceL": f"{dice_meter.avg:.4f}",
            "dice(WT/TC/ET)": f"{dice_wt_meter.avg:.3f}/{dice_tc_meter.avg:.3f}/{dice_et_meter.avg:.3f}",
        })

        if wandb_run is not None:
            wandb_run.log({
                "lr": float(cur_lr),
                "train/loss": loss.item(),
                "train/ce_loss": ce_loss.item(),
                "train/dice_loss": dice_loss_val.item(),
                "train/dice_wt": region_d["wt"],
                "train/dice_tc": region_d["tc"],
                "train/dice_et": region_d["et"],
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
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    max_epoch: int,
    prefix: str,           # "val" hoặc "test"
    wandb_run=None,
) -> Dict[str, float]:

    model.eval()
    loss_meter = AverageMeter()
    ce_meter = AverageMeter()
    dice_loss_meter = AverageMeter()

    num_classes = CFG["NUM_CLASSES"]
    dice_class_sum = np.zeros(num_classes - 1, dtype=np.float64)
    dice_class_cnt = 0

    dice_wt_meter = AverageMeter()
    dice_tc_meter = AverageMeter()
    dice_et_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"[{prefix.capitalize()}] Epoch {epoch}/{max_epoch}")

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).squeeze(1).long()

        logits = model(images)
        if isinstance(logits, dict):
            logits = logits.get("seg", logits)

        ce = F.cross_entropy(logits, labels)
        dice_l = multiclass_dice_loss(logits, labels, num_classes=num_classes, ignore_bg=True)
        loss = ce + dice_l

        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        ce_meter.update(ce.item(), bs)
        dice_loss_meter.update(dice_l.item(), bs)

        preds_t = torch.argmax(logits, dim=1)
        region_d = brats_region_dice_torch(preds_t, labels)
        dice_wt_meter.update(region_d["wt"], bs)
        dice_tc_meter.update(region_d["tc"], bs)
        dice_et_meter.update(region_d["et"], bs)

        preds = preds_t.detach().cpu().numpy()
        gts = labels.detach().cpu().numpy()
        for b in range(preds.shape[0]):
            dpc = cal_dice(preds[b], gts[b], num=num_classes)
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

    out = {
        "loss": loss_meter.avg,
        "ce_loss": ce_meter.avg,
        "dice_loss": dice_loss_meter.avg,
        "mean_dice_fg": float(mean_dice_fg_vec.mean()),
        "dice_wt": dice_wt_meter.avg,
        "dice_tc": dice_tc_meter.avg,
        "dice_et": dice_et_meter.avg,
        "dice_struct": mean_dice_struct,
    }
    for c in range(1, num_classes):
        out[f"dice_class_{c}"] = float(mean_dice_fg_vec[c - 1])

    if wandb_run is not None:
        log_dict = {
            f"{prefix}/loss": out["loss"],
            f"{prefix}/ce_loss": out["ce_loss"],
            f"{prefix}/dice_loss": out["dice_loss"],
            f"{prefix}/mean_dice_fg": out["mean_dice_fg"],
            f"{prefix}/dice_wt": out["dice_wt"],
            f"{prefix}/dice_tc": out["dice_tc"],
            f"{prefix}/dice_et": out["dice_et"],
            f"{prefix}/dice_struct": out["dice_struct"],
            f"{prefix}/epoch": epoch,
        }
        for c in range(1, num_classes):
            log_dict[f"{prefix}/dice_class_{c}"] = out[f"dice_class_{c}"]
        wandb_run.log(log_dict)

    return out


# =============================================================================
# Checkpoint helpers (robust)
# =============================================================================
def save_checkpoint(state: Dict[str, Any], ckpt_dir: Path, filename: str):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / filename
    torch.save(state, path)
    print(f"[CKPT] Saved: {path}")


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def _add_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {f"module.{k}": v for k, v in state_dict.items()}


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    scheduler: Optional[WarmupCosineLR],
    ckpt_path: str,
    device: torch.device,
) -> Tuple[int, float, int]:
    """
    Return:
      start_epoch, best_val_dice, sched_step_idx
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    if "state_dict" not in ckpt:
        raise KeyError(f"Checkpoint thiếu key 'state_dict': {ckpt_path}")

    loaded_sd = ckpt["state_dict"]
    model_sd = model.state_dict()

    # Handle DataParallel mismatch
    loaded_has_module = any(k.startswith("module.") for k in loaded_sd.keys())
    model_has_module = any(k.startswith("module.") for k in model_sd.keys())

    if loaded_has_module and not model_has_module:
        loaded_sd = _strip_module_prefix(loaded_sd)
    elif (not loaded_has_module) and model_has_module:
        loaded_sd = _add_module_prefix(loaded_sd)

    # Load (strict=False để tránh crash nếu thay đổi nhỏ)
    missing, unexpected = model.load_state_dict(loaded_sd, strict=False)
    if missing:
        print(f"[CKPT][WARN] Missing keys ({len(missing)}): {missing[:10]}{'...' if len(missing)>10 else ''}")
    if unexpected:
        print(f"[CKPT][WARN] Unexpected keys ({len(unexpected)}): {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")

    # Optimizer (tolerant)
    if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            print(f"[CKPT][WARN] Không load được optimizer state, bỏ qua. Reason: {e}")

    # Scheduler step
    sched_step = int(ckpt.get("sched_step", 0))

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val_dice = float(ckpt.get("best_val_dice", 0.0))
    print(f"[CKPT] Loaded: {ckpt_path} | epoch={start_epoch-1} | best_val_dice={best_val_dice:.4f} | sched_step={sched_step}")

    # If scheduler exists, set its step_idx to sched_step and update lr accordingly
    if scheduler is not None:
        scheduler.step_idx = sched_step
        # set lr to correct value at current step
        lr = scheduler._compute_lr(scheduler.step_idx)
        scheduler._set_lr(lr)

    return start_epoch, best_val_dice, sched_step


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

    print("=== TRAIN SWIN-UNET3D BRATS3D SUPERVISED (PATCH 128) ===")
    print(f"Root:      {ROOT}")
    print(f"Exp dir:   {exp_dir}")
    print(f"Device:    {device}")
    print(f"Patch sz:  {CFG['PATCH_SIZE']}")

    train_logger = Logger(str(log_dir / "train_log.pkl"))
    val_logger = Logger(str(log_dir / "val_log.pkl"))
    test_logger = Logger(str(log_dir / "test_log.pkl"))

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
    train_loader, val_loader, test_loader = build_loaders()

    max_epoch = int(CFG["OPTIM"]["MAX_EPOCH"])
    loss_cfg = CFG["LOSS"]
    loss_type = str(loss_cfg["loss_type"]).lower()
    w_dice = float(loss_cfg["w_dice"])
    w_ce = float(loss_cfg["w_ce"])

    # scheduler init (step-based)
    scheduler = None
    if CFG["SCHED"]["use"]:
        total_steps = len(train_loader) * max_epoch
        warmup_steps = int(float(CFG["SCHED"]["warmup_ratio"]) * total_steps)
        min_lr = float(CFG["SCHED"]["min_lr"])
        scheduler = WarmupCosineLR(
            optimizer=optimizer,
            base_lr=float(CFG["OPTIM"]["LR"]),
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr=min_lr,
            last_step=0,
        )

    # resume
    start_epoch = 1
    best_val_dice = 0.0  # avg(WT,TC,ET) trên VAL
    resume_ckpt = str(CFG.get("RESUME_CKPT", "")).strip()
    if resume_ckpt:
        # resolve relative to ROOT if not absolute
        resume_path = Path(resume_ckpt)
        if not resume_path.is_absolute():
            resume_path = (ROOT / resume_path).resolve()
        if resume_path.is_file():
            start_epoch, best_val_dice, _ = load_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ckpt_path=str(resume_path),
                device=device,
            )
        else:
            print(f"[CKPT][WARN] RESUME_CKPT not found: {resume_path}")

    print(f"[INFO] Start training from epoch {start_epoch}/{max_epoch}, best_val_dice={best_val_dice:.4f}")

    use_amp = bool(CFG["OPTIM"]["USE_AMP"])
    grad_clip = float(CFG["OPTIM"]["GRAD_CLIP_NORM"])

    for epoch in range(start_epoch, max_epoch + 1):
        t0 = time.time()

        train_stats = train_one_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loader=train_loader,
            device=device,
            epoch=epoch,
            max_epoch=max_epoch,
            w_dice=w_dice,
            w_ce=w_ce,
            loss_type=loss_type,
            use_amp=use_amp,
            grad_clip_norm=grad_clip,
            wandb_run=wandb_run,
        )
        train_stats["epoch"] = epoch
        train_logger.log(train_stats)

        do_eval = (epoch % int(CFG["EVAL_EVERY"]) == 0)
        val_stats = None
        test_stats = None

        if do_eval:
            # ---- VAL ----
            val_stats = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                epoch=epoch,
                max_epoch=max_epoch,
                prefix="val",
                wandb_run=wandb_run,
            )
            val_stats["epoch"] = epoch
            val_logger.log(val_stats)

            # ---- TEST (sau VAL) ----
            test_stats = evaluate(
                model=model,
                loader=test_loader,
                device=device,
                epoch=epoch,
                max_epoch=max_epoch,
                prefix="test",
                wandb_run=wandb_run,
            )
            test_stats["epoch"] = epoch
            test_logger.log(test_stats)

            # best theo VAL
            cur_dice_struct = float(val_stats["dice_struct"])
            if cur_dice_struct > best_val_dice:
                best_val_dice = cur_dice_struct
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_val_dice": best_val_dice,
                        "sched_step": int(getattr(scheduler, "step_idx", 0)) if scheduler is not None else 0,
                        "cfg": CFG,
                    },
                    ckpt_dir,
                    "best_checkpoint_SwinUNet3D_patch128.pth",
                )

        # save last
        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
                "sched_step": int(getattr(scheduler, "step_idx", 0)) if scheduler is not None else 0,
                "cfg": CFG,
            },
            ckpt_dir,
            "last_checkpoint_SwinUNet3D_patch128.pth",
        )

        if epoch % int(CFG["SAVE_EVERY"]) == 0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_dice": best_val_dice,
                    "sched_step": int(getattr(scheduler, "step_idx", 0)) if scheduler is not None else 0,
                    "cfg": CFG,
                },
                ckpt_dir,
                f"epoch_{epoch:03d}_SwinUNet3D_patch128.pth",
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
        if test_stats is not None:
            msg += (
                f" || test_loss={test_stats['loss']:.4f} "
                f"| test_meanDiceFG={test_stats['mean_dice_fg']:.4f} "
                f"| test_dice(WT/TC/ET)={test_stats['dice_wt']:.3f}/"
                f"{test_stats['dice_tc']:.3f}/{test_stats['dice_et']:.3f} "
                f"| test_dice_struct={test_stats['dice_struct']:.3f}"
            )
        msg += f" | time={dt:.1f}s"
        print(msg)

    if wandb_run is not None:
        wandb_run.finish()

    print("[OK] Training finished.")


if __name__ == "__main__":
    main()
