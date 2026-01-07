# -*- coding: utf-8 -*-
"""
inference/infer_swin_unet3d.py

Inference cho SwinUnet3D đa lớp trên BraTS2020 (preprocessed 3D):

- Đọc 4 modality: flair.nii.gz, t1.nii.gz, t1ce.nii.gz, t2.nii.gz
- Ground truth: mask.nii.gz (0..3, đã remap 4 -> 3 nếu còn)
- Model: models.swin_unet_3d.SwinUnet3D (4 lớp: 0,1,2,3)
- Sliding window 3D:
    + patch_size = (128,128,128)
    + stride có thể cấu hình
- TTA (Test-Time Augmentation) tùy chọn:
    + Hiện hỗ trợ flip theo (D/H/W)
    + Trung bình xác suất softmax qua nhiều biến thể
- Tính metrics per-case cho WT, TC, ET: Dice, IoU, ASD, HD95
- Lưu segmentation dự đoán và CSV metrics

Chạy:
    python -u inference/infer_swin_unet3d.py
"""

from __future__ import annotations
import os
import sys
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

# ---------------------- optional nibabel + medpy ----------------------
try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None

try:
    from medpy import metric
except ImportError:  # pragma: no cover
    metric = None


# =============================================================================
# CONFIG INFERENCE
# =============================================================================
CFG_INFER: Dict[str, Any] = {
    # Tên thí nghiệm (để lấy ckpt & thư mục output)
    "EXP_NAME": "swin_unet3d_patch128",

    # Patch & stride (D,H,W)
    "PATCH_SIZE": (128, 128, 128),
    "STRIDE": (128, 128, 128),

    # Đường dẫn tương đối (theo ROOT)
    "DATA_ROOT_3D": "data/processed/3d/labeled",
    "SPLIT_ROOT": "configs/splits_2d",
    "TEST_LIST": "test.txt",

    # Checkpoint (relative theo ROOT hoặc absolute)
    # Ví dụ:
    # "CKPT_PATH": "experiments/swin_unet3d_patch128/checkpoints/best_checkpoint_SwinUNet3D_patch128.pth",
    "CKPT_PATH": "experiments/swin_unet3d_patch128/checkpoints/best_checkpoint_SwinUNet3D_patch128.pth",

    # Nơi lưu kết quả inference
    "OUT_DIR": "experiments/swin_unet3d_patch128/inference",

    # Device
    "DEVICE": "cuda",  # "cuda" hoặc "cpu"

    # ---------------- TTA ----------------
    "USE_TTA": True,     # True để bật TTA
    "TTA_NUM": 8,         # số biến thể TTA dùng (1-> no-tta). max=8 cho flip-combos
    "TTA_MODE": "flip",   # hiện hỗ trợ: "flip"

    # ---------------- Model config (PHẢI KHỚP LÚC TRAIN) ----------------
    "MODEL": {
        "in_channels": 4,
        "num_classes": 4,
        "embed_dim": 96,
        "depths": (2, 2, 2, 2),
        "num_heads": (3, 6, 12, 24),
        "window_size": (4, 4, 4),
        "mlp_ratio": 4.0,
    },
}


# =============================================================================
# PATH & IMPORTS RELATIVE TO PROJECT ROOT
# =============================================================================
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Model (bạn đã có trong train)
from models.swin_unet_3d import SwinUnet3D, SwinUnet3DConfig  # noqa: E402


# =============================================================================
# Utility
# =============================================================================
def get_device() -> torch.device:
    dev = CFG_INFER["DEVICE"]
    if dev == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA không khả dụng, chuyển sang CPU.")
        dev = "cpu"
    return torch.device(dev)


def read_case_list(list_path: Path) -> List[str]:
    ids: List[str] = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.append(line)
    return ids


def _resolve_path(p: str) -> Path:
    p = str(p).strip()
    if not p:
        raise ValueError("Empty path.")
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (ROOT / pp).resolve()


# -----------------------------------------------------------------------------
# Loading NIfTI volumes
# -----------------------------------------------------------------------------
def load_volume_4ch_from_modalities(case_dir: Path) -> Tuple[np.ndarray, "nib.Nifti1Image"]:
    """
    Load 4 modality trong case_dir:
        flair.nii.gz, t1.nii.gz, t1ce.nii.gz, t2.nii.gz

    Return:
        vol_4ch: (4, D, H, W) float32  (NOTE: D,H,W ở NIfTI gốc)
        flair_nii: Nifti1Image của flair để lấy affine/header khi save
    """
    if nib is None:
        raise RuntimeError("Nibabel chưa cài, không đọc được NIfTI.")

    flair_path = case_dir / "flair.nii.gz"
    t1_path = case_dir / "t1.nii.gz"
    t1ce_path = case_dir / "t1ce.nii.gz"
    t2_path = case_dir / "t2.nii.gz"

    for p in [flair_path, t1_path, t1ce_path, t2_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing modality file: {p}")

    flair_nii = nib.load(str(flair_path))
    t1_nii = nib.load(str(t1_path))
    t1ce_nii = nib.load(str(t1ce_path))
    t2_nii = nib.load(str(t2_path))

    flair = flair_nii.get_fdata().astype(np.float32)
    t1 = t1_nii.get_fdata().astype(np.float32)
    t1ce = t1ce_nii.get_fdata().astype(np.float32)
    t2 = t2_nii.get_fdata().astype(np.float32)

    if not (flair.shape == t1.shape == t1ce.shape == t2.shape):
        raise ValueError(
            f"Shape mismatch: flair {flair.shape}, t1 {t1.shape}, t1ce {t1ce.shape}, t2 {t2.shape}"
        )

    vol_4ch = np.stack([flair, t1, t1ce, t2], axis=0)  # (4, D, H, W)
    return vol_4ch.astype(np.float32), flair_nii


def load_label_3d(lbl_path: Path) -> Tuple[np.ndarray, "nib.Nifti1Image"]:
    """
    Load mask.nii.gz (0..3); return seg (D,H,W) int16.
    """
    if nib is None:
        raise RuntimeError("Nibabel chưa cài, không đọc được NIfTI.")

    lbl_nii = nib.load(str(lbl_path))
    seg = lbl_nii.get_fdata().astype(np.int16)

    # đảm bảo không còn label 4
    seg[seg == 4] = 3

    return seg, lbl_nii


# -----------------------------------------------------------------------------
# Sliding window inference + TTA
# -----------------------------------------------------------------------------
def _compute_steps(dim: int, patch: int, stride: int) -> List[int]:
    """Tính các vị trí bắt đầu cho sliding window trên 1 trục."""
    if dim <= patch:
        return [0]
    steps = list(range(0, dim - patch + 1, stride))
    if steps[-1] != dim - patch:
        steps.append(dim - patch)
    return steps


def _apply_flip_3d(x: torch.Tensor, axes: Tuple[int, ...]) -> torch.Tensor:
    """
    x: (B,C,D,H,W)
    axes: subset of {2,3,4}
    """
    if len(axes) == 0:
        return x
    return torch.flip(x, dims=list(axes))


def _sample_tta_flip_axes(num: int) -> List[Tuple[int, ...]]:
    """
    Danh sách flip axes cho TTA (tối đa 8 tổ hợp).
    dims input: (B,C,D,H,W) => không gian là (2,3,4)
    """
    base = [
        (),         # no flip
        (4,),       # flip W
        (3,),       # flip H
        (2,),       # flip D
        (3, 4),     # flip H,W
        (2, 4),     # flip D,W
        (2, 3),     # flip D,H
        (2, 3, 4),  # flip D,H,W
    ]
    if num <= 0:
        return [()]
    if num >= len(base):
        return base
    return base[:num]


@torch.no_grad()
def _predict_patch_with_tta(
    model: torch.nn.Module,
    patch_t: torch.Tensor,
    num_classes: int,
    use_tta: bool,
    tta_num: int,
    tta_mode: str = "flip",
) -> np.ndarray:
    """
    patch_t: (1,4,pd,ph,pw) on device
    return: probs_np (C,pd,ph,pw)
    """
    model.eval()

    if (not use_tta) or int(tta_num) <= 1:
        out = model(patch_t)
        logits = out.get("seg", out) if isinstance(out, dict) else out
        probs = torch.softmax(logits, dim=1)[0]
        return probs.detach().cpu().numpy().astype(np.float32)

    tta_mode = str(tta_mode).lower().strip()
    if tta_mode != "flip":
        raise ValueError(f"TTA_MODE='{tta_mode}' chưa hỗ trợ. Hiện chỉ hỗ trợ 'flip'.")

    axes_list = _sample_tta_flip_axes(int(tta_num))

    prob_sum: Optional[torch.Tensor] = None
    for axes in axes_list:
        x_aug = _apply_flip_3d(patch_t, axes)
        out = model(x_aug)
        logits = out.get("seg", out) if isinstance(out, dict) else out
        probs = torch.softmax(logits, dim=1)  # (1,C,pd,ph,pw)

        # unflip output
        probs = _apply_flip_3d(probs, axes)

        prob_sum = probs if prob_sum is None else (prob_sum + probs)

    prob_mean = prob_sum / float(len(axes_list))
    return prob_mean[0].detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def sliding_window_multiclass(
    model: torch.nn.Module,
    vol_4ch: np.ndarray,
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    device: torch.device,
    num_classes: int = 4,
    use_tta: bool = False,
    tta_num: int = 1,
    tta_mode: str = "flip",
) -> np.ndarray:
    """
    vol_4ch: (4, D, H, W)
    Return:
        prob_vol: (C, D, H, W) softmax trung bình chồng chéo (+TTA nếu bật)
    """
    model.eval()

    _, D, H, W = vol_4ch.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    steps_d = _compute_steps(D, pd, sd)
    steps_h = _compute_steps(H, ph, sh)
    steps_w = _compute_steps(W, pw, sw)

    prob_sum = np.zeros((num_classes, D, H, W), dtype=np.float32)
    count = np.zeros((D, H, W), dtype=np.float32)

    for z in steps_d:
        for y in steps_h:
            for x in steps_w:
                patch = vol_4ch[:, z:z + pd, y:y + ph, x:x + pw]  # (4,pd,ph,pw)
                patch_t = torch.from_numpy(patch[None, ...]).to(device)  # (1,4,*,*,*)

                probs_np = _predict_patch_with_tta(
                    model=model,
                    patch_t=patch_t,
                    num_classes=num_classes,
                    use_tta=bool(use_tta),
                    tta_num=int(tta_num),
                    tta_mode=str(tta_mode),
                )  # (C,pd,ph,pw)

                prob_sum[:, z:z + pd, y:y + ph, x:x + pw] += probs_np
                count[z:z + pd, y:y + ph, x:x + pw] += 1.0

    count[count == 0] = 1.0
    prob_vol = prob_sum / count[None, ...]
    return prob_vol


def probs_to_seg(prob_vol: np.ndarray) -> np.ndarray:
    """
    prob_vol: (C,D,H,W) -> seg: (D,H,W) int16 = argmax
    """
    return np.argmax(prob_vol, axis=0).astype(np.int16)


# -----------------------------------------------------------------------------
# Metrics (Dice, IoU, ASD, HD95) cho nhị phân WT/TC/ET
# -----------------------------------------------------------------------------
def compute_binary_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    voxelspacing: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, float]:
    """
    pred, gt: nhị phân (0/1), cùng shape.
    Return: dict dice, iou, asd, hd95 (có thể NaN).
    """
    if metric is None:
        raise RuntimeError("medpy chưa cài, không tính được ASD/HD95.")

    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)

    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())

    # cả 2 rỗng -> NaN
    if gt_sum == 0 and pred_sum == 0:
        return {"dice": np.nan, "iou": np.nan, "asd": np.nan, "hd95": np.nan}

    # một trong hai rỗng -> dice/iou=0, distance=NaN
    if gt_sum == 0 or pred_sum == 0:
        return {"dice": 0.0, "iou": 0.0, "asd": np.nan, "hd95": np.nan}

    res: Dict[str, float] = {}
    try:
        res["dice"] = float(metric.binary.dc(pred, gt))
    except Exception:
        res["dice"] = np.nan

    try:
        res["iou"] = float(metric.binary.jc(pred, gt))
    except Exception:
        res["iou"] = np.nan

    try:
        res["asd"] = float(metric.binary.asd(pred, gt, voxelspacing=voxelspacing))
    except Exception:
        res["asd"] = np.nan

    try:
        if hasattr(metric.binary, "hd95"):
            res["hd95"] = float(metric.binary.hd95(pred, gt, voxelspacing=voxelspacing))
        else:
            res["hd95"] = float(metric.binary.hd(pred, gt, voxelspacing=voxelspacing))
    except Exception:
        res["hd95"] = np.nan

    return res


def compute_region_metrics(
    pred_seg: np.ndarray,
    gt_seg: np.ndarray,
    voxelspacing: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    WT: label > 0
    TC: label == 1 or 3
    ET: label == 3
    """
    gt_wt = (gt_seg > 0).astype(np.uint8)
    gt_tc = ((gt_seg == 1) | (gt_seg == 3)).astype(np.uint8)
    gt_et = (gt_seg == 3).astype(np.uint8)

    pred_wt = (pred_seg > 0).astype(np.uint8)
    pred_tc = ((pred_seg == 1) | (pred_seg == 3)).astype(np.uint8)
    pred_et = (pred_seg == 3).astype(np.uint8)

    m_wt = compute_binary_metrics(pred_wt, gt_wt, voxelspacing=voxelspacing)
    m_tc = compute_binary_metrics(pred_tc, gt_tc, voxelspacing=voxelspacing)
    m_et = compute_binary_metrics(pred_et, gt_et, voxelspacing=voxelspacing)

    return {"WT": m_wt, "TC": m_tc, "ET": m_et}


# -----------------------------------------------------------------------------
# Robust checkpoint loading (DataParallel tolerant)
# -----------------------------------------------------------------------------
def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in sd.keys()):
        return sd
    return {k[len("module."):]: v for k, v in sd.items()}


def _add_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in sd.keys()):
        return sd
    return {f"module.{k}": v for k, v in sd.items()}


def load_model_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if "state_dict" not in ckpt:
        raise KeyError(f"Checkpoint thiếu key 'state_dict': {ckpt_path}")

    loaded_sd = ckpt["state_dict"]
    model_sd = model.state_dict()

    loaded_has_module = any(k.startswith("module.") for k in loaded_sd.keys())
    model_has_module = any(k.startswith("module.") for k in model_sd.keys())

    if loaded_has_module and not model_has_module:
        loaded_sd = _strip_module_prefix(loaded_sd)
    elif (not loaded_has_module) and model_has_module:
        loaded_sd = _add_module_prefix(loaded_sd)

    missing, unexpected = model.load_state_dict(loaded_sd, strict=False)
    if missing:
        print(f"[CKPT][WARN] Missing keys ({len(missing)}): {missing[:10]}{'...' if len(missing)>10 else ''}")
    if unexpected:
        print(f"[CKPT][WARN] Unexpected keys ({len(unexpected)}): {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")

    print(f"[CKPT] Loaded model from {ckpt_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    if nib is None:
        raise RuntimeError("Vui lòng cài nibabel: pip install nibabel")
    if metric is None:
        raise RuntimeError("Vui lòng cài medpy: pip install medpy")

    device = get_device()

    # ---- Paths ----
    data_root_3d = _resolve_path(CFG_INFER["DATA_ROOT_3D"])
    split_root = _resolve_path(CFG_INFER["SPLIT_ROOT"])
    test_list_path = split_root / CFG_INFER["TEST_LIST"]

    ckpt_path = _resolve_path(CFG_INFER["CKPT_PATH"])

    out_dir = _resolve_path(CFG_INFER["OUT_DIR"])
    out_pred_dir = out_dir / "preds"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pred_dir.mkdir(parents=True, exist_ok=True)

    patch_size = tuple(CFG_INFER["PATCH_SIZE"])
    stride = tuple(CFG_INFER["STRIDE"])

    use_tta = bool(CFG_INFER["USE_TTA"])
    tta_num = int(CFG_INFER["TTA_NUM"])
    tta_mode = str(CFG_INFER["TTA_MODE"])

    print("=== INFERENCE SwinUnet3D BraTS3D ===")
    print(f"ROOT:           {ROOT}")
    print(f"Data root 3D:   {data_root_3d}")
    print(f"Test list:      {test_list_path}")
    print(f"Checkpoint:     {ckpt_path}")
    print(f"Out dir:        {out_dir}")
    print(f"Device:         {device}")
    print(f"Patch/Stride:   patch={patch_size}, stride={stride}")
    print(f"TTA:            USE_TTA={use_tta} | TTA_NUM={tta_num} | MODE={tta_mode}")

    # ---- Read test IDs ----
    test_ids = read_case_list(test_list_path)
    print(f"[INFO] #cases in test list: {len(test_ids)}")

    # ---- Build model ----
    mcfg = CFG_INFER["MODEL"]
    num_classes = int(mcfg["num_classes"])

    model_cfg = SwinUnet3DConfig(
        in_channels=int(mcfg["in_channels"]),
        num_classes=int(mcfg["num_classes"]),
        embed_dim=int(mcfg["embed_dim"]),
        depths=tuple(mcfg["depths"]),
        num_heads=tuple(mcfg["num_heads"]),
        window_size=tuple(mcfg["window_size"]),
        mlp_ratio=float(mcfg["mlp_ratio"]),
    )
    model = SwinUnet3D(model_cfg).to(device)

    # ---- Load checkpoint ----
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    load_model_checkpoint(model, ckpt_path, device)

    # ---- CSV ----
    csv_path = out_dir / "metrics_swinunet3d_test.csv"
    csv_headers = [
        "case_id",
        "dice_wt", "iou_wt", "asd_wt", "hd95_wt",
        "dice_tc", "iou_tc", "asd_tc", "hd95_tc",
        "dice_et", "iou_et", "asd_et", "hd95_et",
    ]
    csv_rows: List[List[Any]] = []

    metrics_all: Dict[str, List[float]] = {
        "dice_wt": [], "iou_wt": [], "asd_wt": [], "hd95_wt": [],
        "dice_tc": [], "iou_tc": [], "asd_tc": [], "hd95_tc": [],
        "dice_et": [], "iou_et": [], "asd_et": [], "hd95_et": [],
    }

    pbar = tqdm(test_ids, desc="[Test cases]")

    for case_id in pbar:
        case_dir = data_root_3d / case_id

        lbl_path = case_dir / "mask.nii.gz"
        if not lbl_path.exists():
            print(f"[WARN] Missing label: {lbl_path}, skip.")
            continue

        try:
            vol_4ch, img_nii = load_volume_4ch_from_modalities(case_dir)
        except FileNotFoundError as e:
            print(f"[WARN] {e} -> skip {case_id}")
            continue

        gt_seg, lbl_nii = load_label_3d(lbl_path)
        spacing = lbl_nii.header.get_zooms()[:3]

        # ---- Sliding window inference (+ optional TTA) ----
        prob_vol = sliding_window_multiclass(
            model=model,
            vol_4ch=vol_4ch,
            patch_size=patch_size,
            stride=stride,
            device=device,
            num_classes=num_classes,
            use_tta=use_tta,
            tta_num=tta_num,
            tta_mode=tta_mode,
        )
        pred_seg = probs_to_seg(prob_vol)

        # ---- Save predicted NIfTI ----
        pred_nii = nib.Nifti1Image(
            pred_seg.astype(np.int16),
            affine=img_nii.affine,
            header=img_nii.header,
        )
        out_pred_path = out_pred_dir / f"{case_id}_pred.nii.gz"
        nib.save(pred_nii, str(out_pred_path))

        # ---- Metrics WT/TC/ET ----
        region_metrics = compute_region_metrics(pred_seg, gt_seg, voxelspacing=spacing)
        m_wt = region_metrics["WT"]
        m_tc = region_metrics["TC"]
        m_et = region_metrics["ET"]

        row = [
            case_id,
            m_wt["dice"], m_wt["iou"], m_wt["asd"], m_wt["hd95"],
            m_tc["dice"], m_tc["iou"], m_tc["asd"], m_tc["hd95"],
            m_et["dice"], m_et["iou"], m_et["asd"], m_et["hd95"],
        ]
        csv_rows.append(row)

        for key, val in zip(csv_headers[1:], row[1:]):
            metrics_all[key].append(val)

    # ---- Save CSV ----
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        writer.writerows(csv_rows)
    print(f"[SAVE] CSV metrics: {csv_path}")

    # ---- Mean metrics (ignore NaN) ----
    def _nanmean(lst: List[float]) -> float:
        arr = np.array(lst, dtype=float)
        if arr.size == 0:
            return float("nan")
        return float(np.nanmean(arr))

    mean_wt = {
        "dice": _nanmean(metrics_all["dice_wt"]),
        "iou": _nanmean(metrics_all["iou_wt"]),
        "asd": _nanmean(metrics_all["asd_wt"]),
        "hd95": _nanmean(metrics_all["hd95_wt"]),
    }
    mean_tc = {
        "dice": _nanmean(metrics_all["dice_tc"]),
        "iou": _nanmean(metrics_all["iou_tc"]),
        "asd": _nanmean(metrics_all["asd_tc"]),
        "hd95": _nanmean(metrics_all["hd95_tc"]),
    }
    mean_et = {
        "dice": _nanmean(metrics_all["dice_et"]),
        "iou": _nanmean(metrics_all["iou_et"]),
        "asd": _nanmean(metrics_all["asd_et"]),
        "hd95": _nanmean(metrics_all["hd95_et"]),
    }

    print("\n=== Mean metrics trên test set (bỏ qua NaN) ===")
    print(f"WT: Dice={mean_wt['dice']:.4f}, IoU={mean_wt['iou']:.4f}, ASD={mean_wt['asd']:.4f}, HD95={mean_wt['hd95']:.4f}")
    print(f"TC: Dice={mean_tc['dice']:.4f}, IoU={mean_tc['iou']:.4f}, ASD={mean_tc['asd']:.4f}, HD95={mean_tc['hd95']:.4f}")
    print(f"ET: Dice={mean_et['dice']:.4f}, IoU={mean_et['iou']:.4f}, ASD={mean_et['asd']:.4f}, HD95={mean_et['hd95']:.4f}")


if __name__ == "__main__":
    main()
