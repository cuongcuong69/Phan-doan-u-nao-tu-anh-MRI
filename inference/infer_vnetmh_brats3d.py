# -*- coding: utf-8 -*-
"""
Inference cho VNetMultiHead 3D trên BraTS2020:

- Đọc 4 modality riêng: flair.nii.gz, t1.nii.gz, t1ce.nii.gz, t2.nii.gz
- Ground truth: mask.nii.gz (giá trị 0..3 sau khi remap 4 -> 3)
- Model: models.vnet_multihead.VNetMultiHead (3 head nhị phân WT/TC/ET)
- Sliding window 3D với patch_size = (64,64,64), stride = 50
- Tính metrics per-case cho WT, TC, ET: Dice, IoU, ASD, HD95
- Lưu segmentation dự đoán và CSV metrics
"""

from __future__ import annotations
import os
import sys
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple

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
    "EXP_NAME": "brats3d_vnetmh_sup",

    # Patch & stride
    "PATCH_SIZE": (64, 64, 64),   # (D, H, W)
    "STRIDE": (20, 20, 20),       # (Dz, Dy, Dx)

    # Đường dẫn tương đối (theo ROOT)
    "DATA_ROOT_3D": "data/processed/3d/labeled",
    "SPLIT_ROOT": "configs/splits_2d",
    "TEST_LIST": "test.txt",      # danh sách Brain_ID
    "CKPT_NAME": "best_checkpoint_VNetMH_sup.pth",

    # Nơi lưu kết quả inference
    "OUT_DIR": "experiments/brats3d_vnetmh_sup/inference",

    # Device
    "DEVICE": "cuda",             # "cuda" hoặc "cpu"

    # Ngưỡng xác định foreground từ prob
    "THRESH": 0.5,
}


# =============================================================================
# PATH & IMPORTS RELATIVE TO PROJECT ROOT
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Model
from models.vnet_multihead import VNetMultiHead  # noqa: E402


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
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.append(line)
    return ids


# -----------------------------------------------------------------------------
# Loading NIfTI volumes
# -----------------------------------------------------------------------------

def load_volume_4ch_from_modalities(case_dir: Path) -> Tuple[np.ndarray, "nib.Nifti1Image"]:
    """
    Load 4 modality riêng biệt trong 1 thư mục case_dir:

        flair.nii.gz, t1.nii.gz, t1ce.nii.gz, t2.nii.gz

    Trả về:
        vol: np.ndarray (4, D, H, W) float32
        nib_img: Nifti1Image của flair (dùng affine/header để save seg)
    """
    if nib is None:
        raise RuntimeError("Nibabel chưa cài, không đọc được NIfTI.")

    flair_path = case_dir / "flair.nii.gz"
    t1_path    = case_dir / "t1.nii.gz"
    t1ce_path  = case_dir / "t1ce.nii.gz"
    t2_path    = case_dir / "t2.nii.gz"

    for p in [flair_path, t1_path, t1ce_path, t2_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing modality file: {p}")

    flair_nii = nib.load(str(flair_path))
    t1_nii    = nib.load(str(t1_path))
    t1ce_nii  = nib.load(str(t1ce_path))
    t2_nii    = nib.load(str(t2_path))

    flair = flair_nii.get_fdata().astype(np.float32)
    t1    = t1_nii.get_fdata().astype(np.float32)
    t1ce  = t1ce_nii.get_fdata().astype(np.float32)
    t2    = t2_nii.get_fdata().astype(np.float32)

    assert flair.shape == t1.shape == t1ce.shape == t2.shape, \
        f"Shape mismatch: flair {flair.shape}, t1 {t1.shape}, t1ce {t1ce.shape}, t2 {t2.shape}"

    vol_4ch = np.stack([flair, t1, t1ce, t2], axis=0)  # (4, D, H, W)
    return vol_4ch.astype(np.float32), flair_nii


def load_label_3d(lbl_path: Path) -> Tuple[np.ndarray, "nib.Nifti1Image"]:
    """
    Load mask.nii.gz (0..3); trả về seg (D,H,W) int16.
    """
    if nib is None:
        raise RuntimeError("Nibabel chưa cài, không đọc được NIfTI.")

    lbl_nii = nib.load(str(lbl_path))
    seg = lbl_nii.get_fdata().astype(np.int16)

    # đảm bảo không còn label 4 nếu trước đó đã remap 4 -> 3
    seg[seg == 4] = 3

    return seg, lbl_nii


# -----------------------------------------------------------------------------
# Sliding window inference
# -----------------------------------------------------------------------------

def _compute_steps(dim: int, patch: int, stride: int) -> List[int]:
    """Tính các vị trí bắt đầu cho sliding window trên 1 trục."""
    if dim <= patch:
        return [0]
    steps = list(range(0, dim - patch + 1, stride))
    if steps[-1] != dim - patch:
        steps.append(dim - patch)
    return steps


@torch.no_grad()
def sliding_window_multihead(
    model: torch.nn.Module,
    vol_4ch: np.ndarray,
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    vol_4ch: (4, D, H, W)
    Trả về:
        prob_wt, prob_tc, prob_et: (D, H, W) - xác suất foreground mỗi region.
    """
    model.eval()

    C, D, H, W = vol_4ch.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    steps_d = _compute_steps(D, pd, sd)
    steps_h = _compute_steps(H, ph, sh)
    steps_w = _compute_steps(W, pw, sw)

    # Tích lũy xác suất foreground
    prob_wt_sum = np.zeros((D, H, W), dtype=np.float32)
    prob_tc_sum = np.zeros((D, H, W), dtype=np.float32)
    prob_et_sum = np.zeros((D, H, W), dtype=np.float32)
    count = np.zeros((D, H, W), dtype=np.float32)

    for z in steps_d:
        for y in steps_h:
            for x in steps_w:
                patch = vol_4ch[:, z:z+pd, y:y+ph, x:x+pw]  # (4, pd, ph, pw)
                patch_t = torch.from_numpy(patch[None, ...]).to(device)  # (1,4,*,*,*)

                out = model(patch_t)  # dict: wt/tc/et, each (1,2,*,*,*)
                logits_wt = out["wt"]
                logits_tc = out["tc"]
                logits_et = out["et"]

                # foreground prob = softmax(...)[...,1]
                prob_wt = torch.softmax(logits_wt, dim=1)[:, 1].cpu().numpy()[0]
                prob_tc = torch.softmax(logits_tc, dim=1)[:, 1].cpu().numpy()[0]
                prob_et = torch.softmax(logits_et, dim=1)[:, 1].cpu().numpy()[0]

                prob_wt_sum[z:z+pd, y:y+ph, x:x+pw] += prob_wt
                prob_tc_sum[z:z+pd, y:y+ph, x:x+pw] += prob_tc
                prob_et_sum[z:z+pd, y:y+ph, x:x+pw] += prob_et
                count[z:z+pd, y:y+ph, x:x+pw] += 1.0

    count[count == 0] = 1.0  # tránh chia 0 (trường hợp cực đoan)

    prob_wt = prob_wt_sum / count
    prob_tc = prob_tc_sum / count
    prob_et = prob_et_sum / count

    return prob_wt, prob_tc, prob_et


def region_probs_to_seg(
    prob_wt: np.ndarray,
    prob_tc: np.ndarray,
    prob_et: np.ndarray,
    thresh: float = 0.5,
) -> np.ndarray:
    """
    Convert xác suất WT/TC/ET thành nhãn BraTS (0..3).

    Quy ước:
      - ET: label 3
      - TC (không ET): label 1
      - WT (chỉ edema): label 2
      - Nền: 0
    """
    bw = prob_wt >= thresh
    bt = prob_tc >= thresh
    be = prob_et >= thresh

    seg = np.zeros(prob_wt.shape, dtype=np.int16)

    # ET
    seg[be] = 3
    # Tumor core không enhancing
    seg[bt & (~be)] = 1
    # Edema (WT mà không thuộc TC/ET)
    seg[bw & (~bt) & (~be)] = 2

    return seg


# -----------------------------------------------------------------------------
# Metrics (Dice, IoU, ASD, HD95) cho nhị phân
# -----------------------------------------------------------------------------

def compute_binary_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    voxelspacing: Tuple[float, float, float] | None = None,
) -> Dict[str, float]:
    """
    pred, gt: nhị phân (0/1), cùng shape.
    Trả về dict: dice, iou, asd, hd95 (có thể là np.nan).
    """
    if metric is None:
        raise RuntimeError("medpy chưa cài, không tính được ASD/HD95.")

    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)

    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())

    # Cả GT và pred đều rỗng -> không đánh giá (NaN)
    if gt_sum == 0 and pred_sum == 0:
        return {"dice": np.nan, "iou": np.nan, "asd": np.nan, "hd95": np.nan}

    # Một trong hai rỗng -> Dice/IoU = 0, ASD/HD95 không xác định
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

    # Một số phiên bản có hd95, một số chỉ có hd
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
    voxelspacing: Tuple[float, float, float] | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    Tính metrics WT/TC/ET từ seg dự đoán & GT (0..3).

    WT: label > 0
    TC: label == 1 or 3
    ET: label == 3
    """
    # Ground truth
    gt_wt = (gt_seg > 0).astype(np.uint8)
    gt_tc = ((gt_seg == 1) | (gt_seg == 3)).astype(np.uint8)
    gt_et = (gt_seg == 3).astype(np.uint8)

    # Prediction
    pred_wt = (pred_seg > 0).astype(np.uint8)
    pred_tc = ((pred_seg == 1) | (pred_seg == 3)).astype(np.uint8)
    pred_et = (pred_seg == 3).astype(np.uint8)

    m_wt = compute_binary_metrics(pred_wt, gt_wt, voxelspacing)
    m_tc = compute_binary_metrics(pred_tc, gt_tc, voxelspacing)
    m_et = compute_binary_metrics(pred_et, gt_et, voxelspacing)

    return {"WT": m_wt, "TC": m_tc, "ET": m_et}


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
    data_root_3d = ROOT / CFG_INFER["DATA_ROOT_3D"]
    split_root = ROOT / CFG_INFER["SPLIT_ROOT"]
    test_list_path = split_root / CFG_INFER["TEST_LIST"]

    exp_name = CFG_INFER["EXP_NAME"]
    ckpt_path = ROOT / "experiments" / exp_name / "checkpoints" / CFG_INFER["CKPT_NAME"]
    out_dir = ROOT / CFG_INFER["OUT_DIR"]
    out_pred_dir = out_dir / "preds"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pred_dir.mkdir(parents=True, exist_ok=True)

    print("=== INFERENCE VNetMultiHead BraTS3D ===")
    print(f"ROOT:           {ROOT}")
    print(f"Data root 3D:   {data_root_3d}")
    print(f"Test list:      {test_list_path}")
    print(f"Checkpoint:     {ckpt_path}")
    print(f"Out dir:        {out_dir}")
    print(f"Device:         {device}")

    # ---- Read test IDs ----
    test_ids = read_case_list(test_list_path)
    print(f"[INFO] #cases in test list: {len(test_ids)}")

    # ---- Build model & load checkpoint ----
    # Thông số model phải khớp lúc train
    model = VNetMultiHead(
        n_channels=4,
        n_filters=32,
        normalization="batchnorm",
        has_dropout=True,
    ).to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"[CKPT] Loaded model from {ckpt_path}")

    patch_size = tuple(CFG_INFER["PATCH_SIZE"])
    stride = tuple(CFG_INFER["STRIDE"])
    thresh = float(CFG_INFER["THRESH"])

    # ---- CSV ----
    csv_path = out_dir / "metrics_vnetmh_test.csv"
    csv_headers = [
        "case_id",
        # WT
        "dice_wt", "iou_wt", "asd_wt", "hd95_wt",
        # TC
        "dice_tc", "iou_tc", "asd_tc", "hd95_tc",
        # ET
        "dice_et", "iou_et", "asd_et", "hd95_et",
    ]
    csv_rows: List[List[Any]] = []

    # Để tính mean metrics cuối cùng
    metrics_all: Dict[str, List[float]] = {
        "dice_wt": [], "iou_wt": [], "asd_wt": [], "hd95_wt": [],
        "dice_tc": [], "iou_tc": [], "asd_tc": [], "hd95_tc": [],
        "dice_et": [], "iou_et": [], "asd_et": [], "hd95_et": [],
    }

    pbar = tqdm(test_ids, desc="[Test cases]")

    for case_id in pbar:
        case_dir = data_root_3d / case_id

        # Ground-truth mask: mask.nii.gz
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

        # spacing (z,y,x) dùng cho ASD/HD95
        spacing = lbl_nii.header.get_zooms()[:3]

        # ---- Sliding window inference ----
        prob_wt, prob_tc, prob_et = sliding_window_multihead(
            model, vol_4ch, patch_size, stride, device
        )
        pred_seg = region_probs_to_seg(prob_wt, prob_tc, prob_et, thresh=thresh)

        # ---- Lưu NIfTI dự đoán ----
        pred_nii = nib.Nifti1Image(pred_seg.astype(np.int16), affine=img_nii.affine, header=img_nii.header)
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

        # gom metrics để tính mean (giữ cả NaN, sẽ dùng nanmean)
        for key, val in zip(csv_headers[1:], row[1:]):
            metrics_all[key].append(val)

    # ---- Save CSV ----
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        writer.writerows(csv_rows)
    print(f"[SAVE] CSV metrics: {csv_path}")

    # ---- In mean metrics (bỏ qua NaN) ----
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
    print(
        f"WT: Dice={mean_wt['dice']:.4f}, IoU={mean_wt['iou']:.4f}, "
        f"ASD={mean_wt['asd']:.4f}, HD95={mean_wt['hd95']:.4f}"
    )
    print(
        f"TC: Dice={mean_tc['dice']:.4f}, IoU={mean_tc['iou']:.4f}, "
        f"ASD={mean_tc['asd']:.4f}, HD95={mean_tc['hd95']:.4f}"
    )
    print(
        f"ET: Dice={mean_et['dice']:.4f}, IoU={mean_et['iou']:.4f}, "
        f"ASD={mean_et['asd']:.4f}, HD95={mean_et['hd95']:.4f}"
    )


if __name__ == "__main__":
    main()
