# -*- coding: utf-8 -*-
"""
Visualize kết quả dự đoán VNetMultiHead trên BraTS:

Mỗi lát cắt (slice) vẽ figure 4 hàng × 2 cột:

Hàng 1 (WT):
    - Cột 1: WT Ground Truth overlay
    - Cột 2: WT Prediction overlay

Hàng 2 (TC):
    - Cột 1: TC Ground Truth overlay
    - Cột 2: TC Prediction overlay

Hàng 3 (ET):
    - Cột 1: ET Ground Truth overlay
    - Cột 2: ET Prediction overlay

Hàng 4 (Subregions):
    - Cột 1: ED + NCR/NET + ET Ground Truth overlay (0..3)
    - Cột 2: ED + NCR/NET + ET Prediction overlay

Các độ đo Dice, IoU, ASD, HD95 được IN RA MÀN HÌNH (stdout),
không ghi đè lên ảnh.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nibabel as nib
from medpy import metric


# =============================================================================
# CONFIG
# =============================================================================
CFG_VIS: Dict[str, Any] = {
    "CASE_ID": "Brain_011",
    "BASE_MODALITY": "flair",

    # Số lát sẽ vẽ / danh sách lát cố định
    "NUM_SLICES": 6,
    "FIXED_SLICES": [i for i in range(60, 100, 5)],

    # Đường dẫn tương đối (theo ROOT)
    "DATA_ROOT_3D": "data/processed/3d/labeled",
    "PRED_ROOT": "experiments/brats3d_vnet_sup/inference/preds",
    "OUT_DIR": "experiments/brats3d_vnet_sup/vis",

    # Kích thước figure cho 4x2 subplot (tăng lên để ảnh to hơn)
    "FIGSIZE": (18, 33),

    # Màu multi-class cho subregions (NCR/NET, ED, ET)
    "LABEL_COLORS": {
        1: (1.0, 0.3, 0.3, 0.55),   # NCR/NET – đỏ
        2: (0.4, 1.0, 0.4, 0.55),   # ED – xanh lá
        3: (0.3, 0.3, 1.0, 0.55),   # ET – xanh dương
    },

    # Màu overlay cho ROIs nhị phân (WT, TC, ET)
    "ROI_COLORS": {
        "WT": (0.4, 1.0, 0.4, 0.55),
        "TC": (1.0, 0.3, 0.3, 0.55),
        "ET": (0.95, 0.95, 0.1, 0.55),
    },

    "LEGEND_NAMES": {
        1: "NCR/NET",
        2: "ED",
        3: "ET",
    },
}


# =============================================================================
# ROOT
# =============================================================================
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


# =============================================================================
# IO helpers
# =============================================================================
def load_modality(case_dir: Path, modality: str) -> np.ndarray:
    nii = nib.load(str(case_dir / f"{modality}.nii.gz"))
    return nii.get_fdata().astype(np.float32)


def load_label(case_dir: Path) -> tuple[np.ndarray, Tuple[float, float]]:
    """
    Load mask GT (0..3), remap 4 -> 3.
    Trả về:
        seg_vol: (H, W, D)
        spacing_2d: (sx, sy) dùng cho ASD/HD95 theo lát (x, y).
    """
    nii = nib.load(str(case_dir / "mask.nii.gz"))
    seg = nii.get_fdata().astype(np.int16)
    seg[seg == 4] = 3

    zooms = nii.header.get_zooms()[:2]  # (sx, sy)
    spacing_2d = (float(zooms[0]), float(zooms[1]))
    return seg, spacing_2d


def load_pred(pred_root: Path, case_id: str) -> np.ndarray:
    nii = nib.load(str(pred_root / f"{case_id}_pred.nii.gz"))
    seg = nii.get_fdata().astype(np.int16)
    seg[seg == 4] = 3
    return seg


def choose_slices(D: int, num_slices: int, fixed_slices):
    if fixed_slices:
        return [z for z in fixed_slices if 0 <= z < D]
    if D <= num_slices:
        return list(range(D))
    return list(np.linspace(0, D - 1, num_slices, dtype=int))


# =============================================================================
# Masks & overlay
# =============================================================================
def extract_roi_masks(seg2d: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Từ seg 2D (0..3) suy ra:
      - ROIs: WT, TC, ET
      - Subregions: ED, NCR/NET, ET_sub (để phân biệt tên)
    """
    return {
        # ROIs
        "WT": (seg2d > 0).astype(np.uint8),
        "TC": ((seg2d == 1) | (seg2d == 3)).astype(np.uint8),
        "ET": (seg2d == 3).astype(np.uint8),
        # Subregions
        "ED": (seg2d == 2).astype(np.uint8),
        "NCR_NET": (seg2d == 1).astype(np.uint8),
        "ET_sub": (seg2d == 3).astype(np.uint8),
    }


def overlay_binary(mask: np.ndarray, color: tuple) -> np.ndarray:
    H, W = mask.shape
    out = np.zeros((H, W, 4), dtype=np.float32)
    out[mask.astype(bool)] = color
    return out


def overlay_multiclass(seg: np.ndarray, colors: Dict[int, tuple]) -> np.ndarray:
    H, W = seg.shape
    out = np.zeros((H, W, 4), dtype=np.float32)
    for lb, col in colors.items():
        out[seg == lb] = col
    return out


# =============================================================================
# Metrics
# =============================================================================
def compute_binary_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: Tuple[float, float] | None,
) -> tuple[float, float, float, float]:
    """
    Metrics nhị phân cho 1 lát cắt: Dice, IoU, ASD, HD95.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if gt.sum() == 0 and pred.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan
    if gt.sum() == 0 or pred.sum() == 0:
        return 0.0, 0.0, np.nan, np.nan

    try:
        dice = metric.binary.dc(pred, gt)
    except Exception:
        dice = np.nan

    try:
        iou = metric.binary.jc(pred, gt)
    except Exception:
        iou = np.nan

    try:
        asd = metric.binary.asd(pred, gt, voxelspacing=spacing)
    except Exception:
        asd = np.nan

    try:
        if hasattr(metric.binary, "hd95"):
            hd = metric.binary.hd95(pred, gt, voxelspacing=spacing)
        else:
            hd = metric.binary.hd(pred, gt, voxelspacing=spacing)
    except Exception:
        hd = np.nan

    return float(dice), float(iou), float(asd), float(hd)


def fmt_metric(prefix: str, vals: tuple[float, float, float, float]) -> str:
    d, i, a, h = vals
    return f"{prefix}: D={d:.3f}, I={i:.3f}, A={a:.2f}, H={h:.2f}"


# =============================================================================
# MAIN VISUALIZATION
# =============================================================================
def visualize_case(
    case_id: str,
    base_modality: str,
    data_root: Path,
    pred_root: Path,
    out_dir: Path,
    num_slices: int,
    fixed_slices,
    figsize=(16, 22),
):
    case_dir = data_root / case_id
    if not case_dir.exists():
        raise FileNotFoundError(case_dir)

    # Load data
    img_vol = load_modality(case_dir, base_modality)
    gt_vol, spacing_2d = load_label(case_dir)
    pred_vol = load_pred(pred_root, case_id)

    H, W, D = img_vol.shape
    slice_ids = choose_slices(D, num_slices, fixed_slices)

    out_dir.mkdir(parents=True, exist_ok=True)

    roi_colors = CFG_VIS["ROI_COLORS"]
    mc_colors = CFG_VIS["LABEL_COLORS"]
    legend_names = CFG_VIS["LEGEND_NAMES"]

    legend_handles = [
        Patch(facecolor=mc_colors[lb][:3], edgecolor="k", label=name)
        for lb, name in legend_names.items()
    ]

    for z in slice_ids:
        img = img_vol[:, :, z]
        gt = gt_vol[:, :, z]
        pred = pred_vol[:, :, z]

        roi_gt = extract_roi_masks(gt)
        roi_pr = extract_roi_masks(pred)

        # ----- Tính metrics -----
        metrics: Dict[str, tuple[float, float, float, float]] = {}
        for key in roi_gt.keys():
            metrics[key] = compute_binary_metrics(
                roi_pr[key], roi_gt[key], spacing_2d
            )

        # ----- In metrics ra màn hình -----
        print(f"\n===== {case_id} | slice z={z} | modality={base_modality} =====")
        print("ROIs:")
        for k in ["WT", "TC", "ET"]:
            print("  " + fmt_metric(k, metrics[k]))

        print("Subregions:")
        print("  " + fmt_metric("ED",      metrics["ED"]))
        print("  " + fmt_metric("NCR/NET", metrics["NCR_NET"]))
        print("  " + fmt_metric("ET",      metrics["ET_sub"]))

        # ----- Vẽ figure 4x2 -----
        fig, axes = plt.subplots(4, 2, figsize=figsize)
        fig.suptitle(f"{case_id} – slice z={z} – modality={base_modality}")

        # Điều chỉnh lề để ảnh to hơn
        plt.subplots_adjust(
            left=0.05,
            right=0.95,
            top=0.93,
            bottom=0.05,
            wspace=0,
            hspace=0.3,
        )

        row_order = ["WT", "TC", "ET"]

        # Hàng 1–3: WT, TC, ET
        for i, key in enumerate(row_order):
            # GT
            ax = axes[i, 0]
            ax.imshow(img, cmap="gray", interpolation="nearest")
            ax.imshow(overlay_binary(roi_gt[key], roi_colors[key]),
                      interpolation="nearest")
            ax.set_title(f"{key} GT")
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")

            # Prediction
            ax = axes[i, 1]
            ax.imshow(img, cmap="gray", interpolation="nearest")
            ax.imshow(overlay_binary(roi_pr[key], roi_colors[key]),
                      interpolation="nearest")
            ax.set_title(f"{key} Prediction")
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")

        # Hàng 4: subregions multi-class
        # GT
        ax = axes[3, 0]
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.imshow(overlay_multiclass(gt, mc_colors), interpolation="nearest")
        ax.set_title("Subregions GT")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(handles=legend_handles, fontsize=8, loc="upper right")
        ax.axis("off")

        # Prediction
        ax = axes[3, 1]
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.imshow(overlay_multiclass(pred, mc_colors), interpolation="nearest")
        ax.set_title("Subregions Prediction")
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        # Lưu ảnh
        out_path = out_dir / f"{case_id}_z{z:03d}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[SAVE] {out_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    cfg = CFG_VIS
    visualize_case(
        case_id=cfg["CASE_ID"],
        base_modality=cfg["BASE_MODALITY"],
        data_root=ROOT / cfg["DATA_ROOT_3D"],
        pred_root=ROOT / cfg["PRED_ROOT"],
        out_dir=ROOT / cfg["OUT_DIR"] / cfg["CASE_ID"],
        num_slices=cfg["NUM_SLICES"],
        fixed_slices=cfg["FIXED_SLICES"],
        figsize=tuple(cfg["FIGSIZE"]),
    )
    print("\n[OK] Done.")


if __name__ == "__main__":
    main()
