# -*- coding: utf-8 -*-
"""
Huấn luyện VNetMultiHead 3D cho BraTS2020 (4 modality, 3 region WT/TC/ET).

- Input: 4 modality (FLAIR, T1, T1CE, T2), mask 0..3 (sau khi remap 4 -> 3)
- Region heads:
    WT (Whole Tumor) : label > 0  => {1,2,3}
    TC (Tumor Core)  : label in {1,3}
    ET (Enhancing)   : label == 3

- Model: models.vnet_multihead.VNetMultiHead (1 encoder + 1 decoder + 3 head nhị phân).
  Mỗi head output (B,2,D,H,W).

- Loss: CE / Dice / Dice+CE (w_ce, w_dice) cho *từng head* rồi trung bình.
- Dataloader: data.dataloader_brats3d_sup (build_brats3d_sup_*_loader)

Tích hợp:
- tqdm.auto cho progress bar
- wandb để log (nếu có)
- Eval mỗi EVAL_EVERY epoch, tính Dice cho WT/TC/ET
- Lưu best/last checkpoints, snapshot mỗi SAVE_EVERY epoch
- Resume từ checkpoint (RESUME_CKPT)
- Giảm LR: nếu val_dice_struct (mean_dice_all) không cải thiện sau `patience` lần eval
  thì nhân LR với `factor` (ở đây = 0.5).
"""

from __future__ import annotations
import os
import sys
import time
import random
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple

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
    "EXP_NAME": "brats3d_vnetmh_sup",
    "SEED": 2025,

    # --------------------- Data ---------------------
    "PATCH_SIZE": (128, 128, 128),  # (D,H,W)

    "TRAIN_BATCH": 2,
    "VAL_BATCH": 2,
    "NUM_WORKERS_TRAIN": 4,
    "NUM_WORKERS_VAL": 4,

    "NUM_CHANNELS": 4,   # FLAIR, T1, T1CE, T2

    # --------------------- Model (MultiHead) ---------------------
    "VNET_MH": {
        "n_channels": 4,
        "n_filters": 16,
        "normalization": "groupnorm",  # "batchnorm" | "groupnorm" | "instancenorm" | "none"
        "has_dropout": True,
    },

    # --------------------- Optimizer ---------------------
    "OPTIM": {
        "LR": 1e-3,
        "WEIGHT_DECAY": 1e-4,
        "BETAS": (0.9, 0.999),
        "MAX_EPOCH": 200,
    },

    # --------------------- LR Scheduler (reduce on plateau) -------------
    # Giảm LR nếu val_dice_struct (mean_dice_all) không cải thiện sau 'patience' lần eval.
    "LR_SCHED": {
        "use": True,
        "factor": 0.5,     # LR = LR * factor
        "patience": 8,     # số lần eval không cải thiện => giảm LR
        "min_epoch": 20,   # chỉ bắt đầu xem xét giảm LR sau epoch này
    },

    # --------------------- Loss ---------------------
    "LOSS": {
        # "ce" | "dice" | "dicece"
        "loss_type": "dicece",
        "w_dice": 1.0,
        "w_ce": 1.0,
    },

    # --------------------- Sampling (patch) ---------------------
    # sampling_mode: "random"|"rejection"|"center_fg"|"mixed"
    "TRAIN_SAMPLING_MODE": "mixed",
    "VAL_SAMPLING_MODE": "mixed",
    "REJECTION_THRESH": 0.01,
    "REJECTION_MAX": 8,
    "TRAIN_MIXED_WEIGHTS": {"center_fg": 0.5, "random": 0.5},
    "VAL_MIXED_WEIGHTS":   {"center_fg": 0.5, "random": 0.5},

    # --------------------- Validation ---------------------
    "EVAL_EVERY": 1,   # validate mỗi epoch

    # --------------------- Checkpoint ---------------------
    "SAVE_EVERY": 1000,
    "RESUME_CKPT": "",  # đường dẫn ckpt để resume (nếu có)

    # --------------------- WandB ---------------------
    "WANDB": {
        "use_wandb": True,
        "project": "brats2020-vnet3d-multihead",
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

# model VNetMultiHead
from models.vnet_multihead import VNetMultiHead


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


# =============================================================================
# Utility: seed, device, dice/binary labels
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


def make_region_labels(labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    labels: (B,D,H,W) với giá trị {0,1,2,3}
    Trả về:
      - y_wt: Whole Tumor (WT)  : 1 nếu label>0
      - y_tc: Tumor Core (TC)   : 1 nếu label in {1,3}
      - y_et: Enhancing (ET)    : 1 nếu label==3
    """
    y_wt = (labels > 0).long()
    y_tc = ((labels == 1) | (labels == 3)).long()
    y_et = (labels == 3).long()
    return y_wt, y_tc, y_et


def binary_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Dice loss nhị phân cho output (B,2,D,H,W), targets (B,D,H,W) ∈ {0,1}.
    Lấy kênh foreground (channel=1) để tính Dice.
    """
    probs = torch.softmax(logits, dim=1)[:, 1, ...]  # (B,D,H,W)
    targets_f = targets.float()

    dims = (0, 1, 2, 3)
    intersection = torch.sum(probs * targets_f, dims)
    cardinality = torch.sum(probs + targets_f, dims)

    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()  # Dice loss


def binary_dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Dice score (1 - loss) cho nhị phân, dùng để log metric.
    """
    return 1.0 - binary_dice_loss(logits, targets, eps=eps)


# =============================================================================
# Build model, optimizer, loaders
# =============================================================================

def build_model_and_opt(device: torch.device):
    vcfg = CFG["VNET_MH"]
    model = VNetMultiHead(
        n_channels=vcfg["n_channels"],
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
        sampling_mode=CFG["VAL_SAMPLING_MODE"],
        rejection_thresh=CFG["REJECTION_THRESH"],
        rejection_max=CFG["REJECTION_MAX"],
        mixed_weights=CFG["VAL_MIXED_WEIGHTS"],
        seed=CFG["SEED"],
    )
    return train_loader, val_loader


# =============================================================================
# Train / Val one epoch (Multi-head)
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
    dice_loss_meter = AverageMeter()

    dice_wt_meter = AverageMeter()
    dice_tc_meter = AverageMeter()
    dice_et_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"[Train-MH] Epoch {epoch}/{max_epoch}")

    for step, batch in enumerate(pbar):
        images = batch["image"].to(device)        # (B,4,D,H,W)
        labels = batch["label"].to(device)        # (B,1,D,H,W)
        labels = labels.squeeze(1).long()         # (B,D,H,W)

        # Tạo label nhị phân cho từng head
        y_wt, y_tc, y_et = make_region_labels(labels)  # (B,D,H,W)

        # forward
        out = model(images)  # dict {"wt","tc","et"}
        logits_wt = out["wt"]  # (B,2,D,H,W)
        logits_tc = out["tc"]
        logits_et = out["et"]

        # CE per head
        ce_wt = F.cross_entropy(logits_wt, y_wt)
        ce_tc = F.cross_entropy(logits_tc, y_tc)
        ce_et = F.cross_entropy(logits_et, y_et)
        ce_total = (ce_wt + ce_tc + ce_et) / 3.0

        # Dice loss per head
        dice_loss_wt = binary_dice_loss(logits_wt, y_wt)
        dice_loss_tc = binary_dice_loss(logits_tc, y_tc)
        dice_loss_et = binary_dice_loss(logits_et, y_et)
        dice_loss_total = (dice_loss_wt + dice_loss_tc + dice_loss_et) / 3.0

        # combine
        if loss_type == "ce":
            loss = ce_total
        elif loss_type == "dice":
            loss = dice_loss_total
        elif loss_type == "dicece":
            loss = w_ce * ce_total + w_dice * dice_loss_total
        else:
            raise ValueError(f"LOSS.loss_type không hợp lệ: {loss_type}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Dice score (not loss) cho từng head
        dice_wt_score = binary_dice_score(logits_wt, y_wt).item()
        dice_tc_score = binary_dice_score(logits_tc, y_tc).item()
        dice_et_score = binary_dice_score(logits_et, y_et).item()

        # update meters
        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        ce_meter.update(ce_total.item(), bs)
        dice_loss_meter.update(dice_loss_total.item(), bs)

        dice_wt_meter.update(dice_wt_score, bs)
        dice_tc_meter.update(dice_tc_score, bs)
        dice_et_meter.update(dice_et_score, bs)

        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "ce": f"{ce_meter.avg:.4f}",
            "dice_loss": f"{dice_loss_meter.avg:.4f}",
            "dice(WT/TC/ET)": f"{dice_wt_meter.avg:.3f}/{dice_tc_meter.avg:.3f}/{dice_et_meter.avg:.3f}",
        })

        if wandb_run is not None:
            wandb_run.log({
                "train/loss": loss.item(),
                "train/ce": ce_total.item(),
                "train/dice_loss": dice_loss_total.item(),
                "train/dice_wt": dice_wt_score,
                "train/dice_tc": dice_tc_score,
                "train/dice_et": dice_et_score,
                "train/epoch": epoch,
            })

    return {
        "loss": loss_meter.avg,
        "ce": ce_meter.avg,
        "dice_loss": dice_loss_meter.avg,
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

    dice_wt_meter = AverageMeter()
    dice_tc_meter = AverageMeter()
    dice_et_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"[Val-MH]   Epoch {epoch}/{max_epoch}")

    for batch in pbar:
        images = batch["image"].to(device)          # (B,4,D,H,W)
        labels = batch["label"].to(device)          # (B,1,D,H,W)
        labels = labels.squeeze(1).long()           # (B,D,H,W)

        y_wt, y_tc, y_et = make_region_labels(labels)

        out = model(images)
        logits_wt = out["wt"]
        logits_tc = out["tc"]
        logits_et = out["et"]

        # CE
        ce_wt = F.cross_entropy(logits_wt, y_wt)
        ce_tc = F.cross_entropy(logits_tc, y_tc)
        ce_et = F.cross_entropy(logits_et, y_et)
        ce_total = (ce_wt + ce_tc + ce_et) / 3.0

        # Dice loss
        dice_loss_wt = binary_dice_loss(logits_wt, y_wt)
        dice_loss_tc = binary_dice_loss(logits_tc, y_tc)
        dice_loss_et = binary_dice_loss(logits_et, y_et)
        dice_loss_total = (dice_loss_wt + dice_loss_tc + dice_loss_et) / 3.0

        # tổng val loss = CE + DiceLoss (giống bản VNet cũ, chỉ để theo dõi)
        loss = ce_total + dice_loss_total

        # dice score
        dice_wt_score = binary_dice_score(logits_wt, y_wt).item()
        dice_tc_score = binary_dice_score(logits_tc, y_tc).item()
        dice_et_score = binary_dice_score(logits_et, y_et).item()

        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        ce_meter.update(ce_total.item(), bs)
        dice_loss_meter.update(dice_loss_total.item(), bs)

        dice_wt_meter.update(dice_wt_score, bs)
        dice_tc_meter.update(dice_tc_score, bs)
        dice_et_meter.update(dice_et_score, bs)

        mean_dice_all = (dice_wt_meter.avg + dice_tc_meter.avg + dice_et_meter.avg) / 3.0

        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "dice(WT/TC/ET)": f"{dice_wt_meter.avg:.3f}/{dice_tc_meter.avg:.3f}/{dice_et_meter.avg:.3f}",
            "mean_dice_all": f"{mean_dice_all:.3f}",
        })

    mean_dice_all = (dice_wt_meter.avg + dice_tc_meter.avg + dice_et_meter.avg) / 3.0

    if wandb_run is not None:
        wandb_run.log({
            "val/loss": loss_meter.avg,
            "val/ce": ce_meter.avg,
            "val/dice_loss": dice_loss_meter.avg,
            "val/dice_wt": dice_wt_meter.avg,
            "val/dice_tc": dice_tc_meter.avg,
            "val/dice_et": dice_et_meter.avg,
            "val/mean_dice_all": mean_dice_all,
            "val/epoch": epoch,
        })

    return {
        "loss": loss_meter.avg,
        "ce": ce_meter.avg,
        "dice_loss": dice_loss_meter.avg,
        "dice_wt": dice_wt_meter.avg,
        "dice_tc": dice_tc_meter.avg,
        "dice_et": dice_et_meter.avg,
        "mean_dice_all": mean_dice_all,
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
# MAIN
# =============================================================================

def main():
    set_seed(CFG["SEED"])
    device = get_device()

    exp_name = CFG["EXP_NAME"]
    exp_dir = ROOT / "experiments" / exp_name
    ckpt_dir = exp_dir / "checkpoints_dicece"
    log_dir = exp_dir / "logs_dicece"
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=== TRAIN VNET-MULTIHEAD BRATS3D SUPERVISED ===")
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

    # resume
    start_epoch = 1
    best_val_dice = 0.0  # dùng avg(WT,TC,ET) làm tiêu chí chọn best
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

            cur_dice_struct = val_stats["mean_dice_all"]  # WT/TC/ET mean

            # chọn best checkpoint theo cur_dice_struct
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
                    "best_checkpoint_VNetMH_sup.pth",
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
            "last_checkpoint_VNetMH_sup.pth",
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
                f"epoch_{epoch:03d}_VNetMH_sup.pth",
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
                f"| val_dice(WT/TC/ET)={val_stats['dice_wt']:.3f}/"
                f"{val_stats['dice_tc']:.3f}/{val_stats['dice_et']:.3f} "
                f"| val_meanDiceAll={val_stats['mean_dice_all']:.3f}"
            )
        msg += f" | time={dt:.1f}s"
        print(msg)

    if wandb_run is not None:
        wandb_run.finish()

    print("[OK] Training finished.")


if __name__ == "__main__":
    main()
