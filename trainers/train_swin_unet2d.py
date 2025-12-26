# -*- coding: utf-8 -*-
"""
trainers/train_swin_unet2d.py

Train SwinUNet 2D (BraTS-like 4 modalities, 4 classes 0..3) with W&B logging.

CHANGES (the ones you requested):
- tqdm progress bar for BOTH training and validation
- Compute + log: dice_WT, dice_TC, dice_ET (BraTS regions)
- save_best_by = val/dice_struct = mean(dice_WT, dice_TC, dice_ET)
- Keep "NO CLI args": edit CFG below.
- Keep import-fix style (add project root to sys.path).
- Model from your models/swin_unet.py:
    from models.swin_unet import build_swin_unet_tiny_224
- Dataloader: robust import (tries common builder names).

Run:
    python -m trainers.train_swin_unet2d
or:
    python trainers/train_swin_unet2d.py
"""

from __future__ import annotations

import sys
import time
import math
import random
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# =============================================================================
# Project root + import fix
# =============================================================================

def _project_root() -> Path:
    # trainers/train_swin_unet2d.py -> trainers -> project root
    return Path(__file__).resolve().parents[1]


ROOT = _project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =============================================================================
# Config (NO CLI ARGS)
# =============================================================================

CFG: Dict[str, Any] = {
    # data
    "image_size": 224,
    "in_chans": 4,
    "num_classes": 4,  # 0..3 (0=bg,1,2,3)

    # train
    "seed": 2025,
    "epochs": 60,
    "batch_size": 8,
    "val_batch_size": 8,
    "num_workers": 4,
    "pin_memory": False,

    # optimizer
    "lr": 3e-4,
    "weight_decay": 1e-2,
    "betas": (0.9, 0.999),

    # scheduler
    "scheduler": "cosine",  # "cosine" | "none"
    "warmup_epochs": 3,

    # amp + stability
    "amp": True,
    "grad_clip_norm": 1.0,

    # pretrained init (encoder-only in your model)
    "use_imagenet_pretrained_encoder": True,

    # checkpointing
    "out_dir": str(ROOT / "experiments" / "swin_unet2d"),
    "save_best_by": "val/dice_struct",   # mean(WT,TC,ET)
    "save_every_epochs": 5,

    # wandb
    "wandb": True,
    "wandb_project": "AdvancedCV-SwinUNet2D",
    "wandb_run_name": "swinunet2d_224",
    "wandb_entity": None,   # set string if needed, else None
    "wandb_tags": ["swin-unet", "2d", "brats", "224"],
    "log_images_every": 1,  # epochs
    "max_log_images": 6,
}


# =============================================================================
# Utils
# =============================================================================

def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # reproducibility/perf tradeoff
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


# =============================================================================
# Dice BraTS Regions
# label mapping assumed:
#   0 = background
#   1 = NCR/NET
#   2 = ED
#   3 = ET
# Regions:
#   WT = {1,2,3}
#   TC = {1,3}
#   ET = {3}
# =============================================================================

@torch.no_grad()
def dice_binary(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    pred, gt: bool tensors (N,H,W) or (N,*,H,W)
    returns mean dice over batch (scalar tensor)
    """
    pred = pred.float()
    gt = gt.float()
    dims = tuple(range(1, pred.ndim))
    inter = (pred * gt).sum(dim=dims)
    den = pred.sum(dim=dims) + gt.sum(dim=dims)
    dice = (2.0 * inter + eps) / (den + eps)
    return dice.mean()


@torch.no_grad()
def brats_region_dice_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    logits: (N,C,H,W)
    target: (N,H,W) or (N,1,H,W)
    returns floats: dice_WT, dice_TC, dice_ET, dice_struct(mean)
    """
    if target.ndim == 4:
        target = target.squeeze(1)
    pred = torch.argmax(logits, dim=1)  # (N,H,W)

    wt_pred = pred > 0
    wt_gt = target > 0

    tc_pred = (pred == 1) | (pred == 3)
    tc_gt = (target == 1) | (target == 3)

    et_pred = pred == 3
    et_gt = target == 3

    d_wt = float(dice_binary(wt_pred, wt_gt, eps=eps).item())
    d_tc = float(dice_binary(tc_pred, tc_gt, eps=eps).item())
    d_et = float(dice_binary(et_pred, et_gt, eps=eps).item())
    d_struct = (d_wt + d_tc + d_et) / 3.0
    return {"dice_WT": d_wt, "dice_TC": d_tc, "dice_ET": d_et, "dice_struct": d_struct}


# =============================================================================
# Visualization helpers (optional W&B overlays)
# =============================================================================

def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    mask: (H,W) values 0..3
    return RGB uint8 (H,W,3) - fixed palette
    """
    palette = np.array([
        [0, 0, 0],       # 0 bg
        [0, 255, 255],   # 1 cyan
        [255, 128, 0],   # 2 orange
        [255, 255, 0],   # 3 yellow
    ], dtype=np.uint8)
    mask = np.clip(mask.astype(np.int32), 0, 3)
    return palette[mask]


def make_overlay(img: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """
    img: (H,W) float
    mask_rgb: (H,W,3) uint8
    return: (H,W,3) uint8
    """
    img = img.astype(np.float32)
    vmin, vmax = np.percentile(img, 1), np.percentile(img, 99)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    img_n = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    img_rgb = (img_n[..., None] * 255.0).astype(np.uint8)
    out = (img_rgb * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)
    return out


# =============================================================================
# Robust dataloader import
# =============================================================================

def build_loaders() -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Tries common dataloader module/function names.
    Expected batch keys:
      batch["image"]: (N,4,224,224)
      batch["label"]: (N,1,224,224) or (N,224,224)
    """
    candidates = [
        ("data.dataloader_brats2d", "build_brats2d_sup_train_loader", "build_brats2d_sup_val_loader"),
    ]

    last_err = None
    for mod_name, fn_tr, fn_val in candidates:
        try:
            mod = __import__(mod_name, fromlist=[fn_tr, fn_val])
            if hasattr(mod, fn_tr) and hasattr(mod, fn_val):
                build_train = getattr(mod, fn_tr)
                build_val = getattr(mod, fn_val)

                train_loader = build_train(
                    image_size=CFG["image_size"],
                    batch_size=CFG["batch_size"],
                    num_workers=CFG["num_workers"],
                    seed=CFG["seed"],
                    pin_memory=CFG["pin_memory"],
                )
                val_loader = build_val(
                    image_size=CFG["image_size"],
                    batch_size=CFG["val_batch_size"],
                    num_workers=CFG["num_workers"],
                    seed=CFG["seed"],
                    pin_memory=CFG["pin_memory"],
                )
                return train_loader, val_loader
        except Exception as e:
            last_err = e

    raise ImportError(
        "Không build được dataloader 2D. "
        "Hãy kiểm tra file data/dataloader_*.py của bạn có builder train/val không.\n"
        f"Last error: {repr(last_err)}"
    ) from last_err


# =============================================================================
# Model
# =============================================================================

def build_model() -> nn.Module:
    from models.swin_unet import build_swin_unet_tiny_224

    model = build_swin_unet_tiny_224(
        in_chans=CFG["in_chans"],
        num_classes=CFG["num_classes"],
        use_imagenet_pretrained_encoder=CFG["use_imagenet_pretrained_encoder"],
    )
    return model


# =============================================================================
# Scheduler (cosine + warmup)
# =============================================================================

def build_scheduler(optimizer: optim.Optimizer, steps_per_epoch: int):
    if CFG["scheduler"] == "none":
        return None

    if CFG["scheduler"] != "cosine":
        raise ValueError(f"Unsupported scheduler: {CFG['scheduler']}")

    total_steps = int(CFG["epochs"] * steps_per_epoch)
    warmup_steps = int(CFG["warmup_epochs"] * steps_per_epoch)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        t = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * t))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# =============================================================================
# Train / Eval loops (WITH tqdm)
# =============================================================================

def compute_loss(logits: torch.Tensor, target: torch.Tensor, ce: nn.Module) -> torch.Tensor:
    if target.ndim == 4:
        target = target.squeeze(1)
    return ce(logits, target.long())


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    wandb_run=None,
) -> Dict[str, float]:
    model.train()
    ce = nn.CrossEntropyLoss()

    t0 = time.time()
    loss_sum = 0.0
    wt_sum = 0.0
    tc_sum = 0.0
    et_sum = 0.0
    struct_sum = 0.0
    step_count = 0

    pbar = tqdm(loader, desc=f"Train {epoch}", dynamic_ncols=True, leave=False)

    for step, batch in enumerate(pbar, start=1):
        x = batch["image"].to(device, non_blocking=True).float()  # (N,4,224,224)
        y = batch["label"].to(device, non_blocking=True)          # (N,1,224,224) or (N,224,224)

        optimizer.zero_grad(set_to_none=True)

        if CFG["amp"] and scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = compute_loss(logits, y, ce)
            scaler.scale(loss).backward()

            if CFG["grad_clip_norm"] is not None and CFG["grad_clip_norm"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip_norm"])

            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = compute_loss(logits, y, ce)
            loss.backward()
            if CFG["grad_clip_norm"] is not None and CFG["grad_clip_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip_norm"])
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            dice = brats_region_dice_from_logits(logits, y)

        loss_val = float(loss.item())
        loss_sum += loss_val
        wt_sum += float(dice["dice_WT"])
        tc_sum += float(dice["dice_TC"])
        et_sum += float(dice["dice_ET"])
        struct_sum += float(dice["dice_struct"])
        step_count += 1

        loss_avg = loss_sum / step_count
        wt_avg = wt_sum / step_count
        tc_avg = tc_sum / step_count
        et_avg = et_sum / step_count
        struct_avg = struct_sum / step_count
        lr = float(optimizer.param_groups[0]["lr"])

        pbar.set_postfix({
            "loss": f"{loss_avg:.4f}",
            "WT": f"{wt_avg:.4f}",
            "TC": f"{tc_avg:.4f}",
            "ET": f"{et_avg:.4f}",
            "S": f"{struct_avg:.4f}",
            "lr": f"{lr:.2e}",
        })

        if wandb_run is not None:
            # step log (lightweight)
            wandb_run.log({
                "train/loss_step": loss_val,
                "train/dice_WT_step": dice["dice_WT"],
                "train/dice_TC_step": dice["dice_TC"],
                "train/dice_ET_step": dice["dice_ET"],
                "train/dice_struct_step": dice["dice_struct"],
                "train/lr": lr,
                "epoch": epoch,
            })

    return {
        "train/loss": loss_sum / max(1, step_count),
        "train/dice_WT": wt_sum / max(1, step_count),
        "train/dice_TC": tc_sum / max(1, step_count),
        "train/dice_ET": et_sum / max(1, step_count),
        "train/dice_struct": struct_sum / max(1, step_count),
        "time/train_epoch_sec": time.time() - t0,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    wandb_run=None,
) -> Dict[str, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()

    t0 = time.time()
    loss_sum = 0.0
    wt_sum = 0.0
    tc_sum = 0.0
    et_sum = 0.0
    struct_sum = 0.0
    step_count = 0

    pbar = tqdm(loader, desc=f"Val   {epoch}", dynamic_ncols=True, leave=False)

    for _, batch in enumerate(pbar, start=1):
        x = batch["image"].to(device, non_blocking=True).float()
        y = batch["label"].to(device, non_blocking=True)

        logits = model(x)
        loss = compute_loss(logits, y, ce)
        dice = brats_region_dice_from_logits(logits, y)

        loss_val = float(loss.item())
        loss_sum += loss_val
        wt_sum += float(dice["dice_WT"])
        tc_sum += float(dice["dice_TC"])
        et_sum += float(dice["dice_ET"])
        struct_sum += float(dice["dice_struct"])
        step_count += 1

        loss_avg = loss_sum / step_count
        wt_avg = wt_sum / step_count
        tc_avg = tc_sum / step_count
        et_avg = et_sum / step_count
        struct_avg = struct_sum / step_count

        pbar.set_postfix({
            "loss": f"{loss_avg:.4f}",
            "WT": f"{wt_avg:.4f}",
            "TC": f"{tc_avg:.4f}",
            "ET": f"{et_avg:.4f}",
            "S": f"{struct_avg:.4f}",
        })

    out = {
        "val/loss": loss_sum / max(1, step_count),
        "val/dice_WT": wt_sum / max(1, step_count),
        "val/dice_TC": tc_sum / max(1, step_count),
        "val/dice_ET": et_sum / max(1, step_count),
        "val/dice_struct": struct_sum / max(1, step_count),
        "time/val_epoch_sec": time.time() - t0,
    }

    # log a few images
    if wandb_run is not None and (epoch % int(CFG["log_images_every"]) == 0):
        try:
            import wandb
            max_imgs = int(CFG["max_log_images"])

            b0 = next(iter(loader))
            xb = b0["image"][:max_imgs].to(device).float()
            yb = b0["label"][:max_imgs].to(device)

            logits_b = model(xb)
            pred_b = torch.argmax(logits_b, dim=1).detach().cpu().numpy()
            if yb.ndim == 4:
                gt_b = yb.squeeze(1).detach().cpu().numpy()
            else:
                gt_b = yb.detach().cpu().numpy()

            xb_np = xb.detach().cpu().numpy()  # (N,4,H,W)
            images = []
            for i in range(min(max_imgs, xb_np.shape[0])):
                flair = xb_np[i, 0]
                gt_overlay = make_overlay(flair, colorize_mask(gt_b[i]), alpha=0.55)
                pr_overlay = make_overlay(flair, colorize_mask(pred_b[i]), alpha=0.55)
                images.append(wandb.Image(gt_overlay, caption=f"GT overlay #{i}"))
                images.append(wandb.Image(pr_overlay, caption=f"PRED overlay #{i}"))

            wandb_run.log({"val/overlays": images, "epoch": epoch})
        except Exception:
            pass

    if wandb_run is not None:
        wandb_run.log({**out, "epoch": epoch})

    return out


# =============================================================================
# Checkpointing
# =============================================================================

def save_ckpt(path: str | Path, model: nn.Module, optimizer: optim.Optimizer, epoch: int, best_score: float):
    path = Path(path)
    ensure_dir(path.parent)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_score": best_score,
        "cfg": CFG,
    }, str(path))


# =============================================================================
# Main
# =============================================================================

def main():
    set_seed(int(CFG["seed"]))
    ensure_dir(CFG["out_dir"])

    device = get_device()
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Project root: {ROOT}")

    # wandb init
    wandb_run = None
    if CFG["wandb"]:
        try:
            import wandb
            wandb_run = wandb.init(
                project=CFG["wandb_project"],
                name=CFG["wandb_run_name"],
                entity=CFG["wandb_entity"],
                tags=CFG["wandb_tags"],
                config=CFG,
            )
        except Exception as e:
            print(f"[WARN] wandb init failed: {repr(e)}")
            wandb_run = None

    # data
    train_loader, val_loader = build_loaders()

    # model
    model = build_model().to(device)
    print("[INFO] Model built:", model.__class__.__name__)

    # optim
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(CFG["lr"]),
        betas=tuple(CFG["betas"]),
        weight_decay=float(CFG["weight_decay"]),
    )

    steps_per_epoch = max(1, len(train_loader))
    scheduler = build_scheduler(optimizer, steps_per_epoch=steps_per_epoch)

    scaler = torch.cuda.amp.GradScaler(enabled=bool(CFG["amp"]) and device.type == "cuda")

    # train
    best_score = -1.0
    best_path = Path(CFG["out_dir"]) / "best.pt"
    last_path = Path(CFG["out_dir"]) / "last.pt"

    for epoch in range(1, int(CFG["epochs"]) + 1):
        print(f"\n===== Epoch {epoch}/{CFG['epochs']} =====")

        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            epoch=epoch,
            wandb_run=wandb_run,
        )
        va = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            epoch=epoch,
            wandb_run=wandb_run,
        )

        logs = {**tr, **va, "epoch": epoch}

        print(
            f"  train: loss={logs['train/loss']:.4f} WT={logs['train/dice_WT']:.4f} "
            f"TC={logs['train/dice_TC']:.4f} ET={logs['train/dice_ET']:.4f} S={logs['train/dice_struct']:.4f} | "
            f"val: loss={logs['val/loss']:.4f} WT={logs['val/dice_WT']:.4f} "
            f"TC={logs['val/dice_TC']:.4f} ET={logs['val/dice_ET']:.4f} S={logs['val/dice_struct']:.4f}"
        )

        if wandb_run is not None:
            wandb_run.log(logs)

        # checkpoint
        metric_key = str(CFG["save_best_by"])
        score = float(logs.get(metric_key, logs.get("val/dice_struct", -1.0)))

        save_ckpt(last_path, model, optimizer, epoch, best_score)

        if score > best_score:
            best_score = score
            save_ckpt(best_path, model, optimizer, epoch, best_score)
            print(f"[OK] New best: {metric_key}={best_score:.4f} -> saved {best_path}")

        if (epoch % int(CFG["save_every_epochs"])) == 0:
            p = Path(CFG["out_dir"]) / f"epoch_{epoch:03d}.pt"
            save_ckpt(p, model, optimizer, epoch, best_score)
            print(f"[OK] Saved checkpoint: {p}")

    print(f"\n[DONE] Best {CFG['save_best_by']} = {best_score:.4f}")
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
