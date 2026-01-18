# -*- coding: utf-8 -*-
"""
Visualize kết quả dự đoán UNETR trên BraTS2020:

Mỗi lát cắt (slice) vẽ figure 2 hàng × 4 cột (layout ngang):

Hàng 1:
    - Cột 1: WT Ground Truth overlay
    - Cột 2: WT Prediction overlay
    - Cột 3: TC Ground Truth overlay
    - Cột 4: TC Prediction overlay

Hàng 2:
    - Cột 1: ET Ground Truth overlay
    - Cột 2: ET Prediction overlay
    - Cột 3: Subregions Ground Truth overlay (0..3)
    - Cột 4: Subregions Prediction overlay

Các độ đo Dice, IoU, ASD, HD95 được IN RA MÀN HÌNH (stdout),
không ghi đè lên ảnh.
"""

from __future__ import annotations
import sys
import shutil
import argparse
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
    "CASE_ID": "Brain_091",
    "BASE_MODALITY": "flair",

    # Số lát sẽ vẽ / danh sách lát cố định
    "NUM_SLICES": 6,
    "FIXED_SLICES": [88],

    # Đường dẫn tuyệt đối
    "DATA_ROOT_3D": r"D:\Project Advanced CV\data\processed\3d\labeled",
    "PRED_ROOT": r"D:\Project Advanced CV\colab_project\experiments\brats3d_unetr_full\inference\preds",
    "OUT_DIR": r"D:\Project Advanced CV\colab_project\experiments\brats3d_unetr_full\vis",

    # Kích thước figure cho 2x4 subplot (layout ngang)
    "FIGSIZE": (24, 12),

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
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

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

        # ----- Vẽ figure 2x4 (layout ngang) -----
        fig, axes = plt.subplots(2, 4, figsize=figsize)

        # Điều chỉnh lề để ảnh to hơn
        plt.subplots_adjust(
            left=0.02,
            right=0.98,
            top=0.92,
            bottom=0.02,
            wspace=0.05,
            hspace=0.25,
        )

        # Hàng 1: WT GT, WT Pred, TC GT, TC Pred
        # WT GT
        ax = axes[0, 0]
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.imshow(overlay_binary(roi_gt["WT"], roi_colors["WT"]),
                  interpolation="nearest")
        ax.set_title("WT Ground Truth", fontsize=28)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        # WT Prediction
        ax = axes[0, 1]
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.imshow(overlay_binary(roi_pr["WT"], roi_colors["WT"]),
                  interpolation="nearest")
        ax.set_title("WT Prediction", fontsize=28)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        # TC GT
        ax = axes[0, 2]
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.imshow(overlay_binary(roi_gt["TC"], roi_colors["TC"]),
                  interpolation="nearest")
        ax.set_title("TC Ground Truth", fontsize=28)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        # TC Prediction
        ax = axes[0, 3]
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.imshow(overlay_binary(roi_pr["TC"], roi_colors["TC"]),
                  interpolation="nearest")
        ax.set_title("TC Prediction", fontsize=28)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        # Hàng 2: ET GT, ET Pred, Subregions GT, Subregions Pred
        # ET GT
        ax = axes[1, 0]
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.imshow(overlay_binary(roi_gt["ET"], roi_colors["ET"]),
                  interpolation="nearest")
        ax.set_title("ET Ground Truth", fontsize=28)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        # ET Prediction
        ax = axes[1, 1]
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.imshow(overlay_binary(roi_pr["ET"], roi_colors["ET"]),
                  interpolation="nearest")
        ax.set_title("ET Prediction", fontsize=28)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        # Subregions GT
        ax = axes[1, 2]
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.imshow(overlay_multiclass(gt, mc_colors), interpolation="nearest")
        ax.set_title("Subregions GT", fontsize=28)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(handles=legend_handles, fontsize=14, loc="upper right")
        ax.axis("off")

        # Subregions Prediction
        ax = axes[1, 3]
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.imshow(overlay_multiclass(pred, mc_colors), interpolation="nearest")
        ax.set_title("Subregions Pred", fontsize=28)
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
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive_path", type=str, default=None,
                       help="Path to Google Drive folder for saving visualization results")
    args = parser.parse_args()
    
    drive_dir = Path(args.drive_path) if args.drive_path else None
    if drive_dir:
        drive_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Auto-save visualization results to Drive: {drive_dir}")
    
    cfg = CFG_VIS
    
    # Convert string paths to Path objects
    data_root = Path(cfg["DATA_ROOT_3D"])
    pred_root = Path(cfg["PRED_ROOT"])
    out_dir = Path(cfg["OUT_DIR"]) / cfg["CASE_ID"]
    
    visualize_case(
        case_id=cfg["CASE_ID"],
        base_modality=cfg["BASE_MODALITY"],
        data_root=data_root,
        pred_root=pred_root,
        out_dir=out_dir,
        num_slices=cfg["NUM_SLICES"],
        fixed_slices=cfg["FIXED_SLICES"],
        figsize=tuple(cfg["FIGSIZE"]),
    )
    
    # ---- Copy to Google Drive ----
    if drive_dir is not None and out_dir.exists():
        print(f"\n[DRIVE] Copying visualization results to Drive...")
        
        # Copy visualization folder
        drive_vis_dir = drive_dir / "visualization" / cfg["CASE_ID"]
        if drive_vis_dir.exists():
            shutil.rmtree(drive_vis_dir)
        shutil.copytree(out_dir, drive_vis_dir)
        
        img_count = len(list(drive_vis_dir.glob("*.png")))
        print(f"[DRIVE] Copied {img_count} visualization images to {drive_vis_dir}")
        print(f"[DRIVE] ✅ All visualization results saved to Drive!")
    
    print("\n[OK] Done.")


if __name__ == "__main__":
    main()
