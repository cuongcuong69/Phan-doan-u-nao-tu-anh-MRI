# -*- coding: utf-8 -*-
"""
Visualize kết quả dự đoán VNetMultiHead trên BraTS:
- Overlay Ground Truth và Prediction lên ảnh nền (1 modality).
- Thêm legend WT – TC – ET cho multi-head.

Ảnh đã được normalize Z-score từ preprocessing --> không normalize lại.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import nibabel as nib

# =============================================================================
# CONFIG
# =============================================================================
CFG_VIS: Dict[str, Any] = {
    "CASE_ID": "Brain_016",
    "BASE_MODALITY": "t1ce",
    "NUM_SLICES": 6,
    "FIXED_SLICES": [i for i in range(60, 100, 5)],
    "DATA_ROOT_3D": "data/processed/3d/labeled",
    "PRED_ROOT": "experiments/brats3d_vnetmh_sup/inference/preds",
    "OUT_DIR": "experiments/brats3d_vnetmh_sup/vis",
    "FIGSIZE": (10, 5),

    # 🎨 OVERLAY MÀU CHUẨN BRA TS 2020
    "LABEL_COLORS": {
        1: (1.0, 0.30, 0.30, 0.55),   # NCR/NET – đỏ nhạt
        2: (0.40, 1.0, 0.40, 0.55),   # Edema – xanh lá nhạt
        3: (0.30, 0.30, 1.0, 0.55),   # ET – xanh dương
    },

    # Legend chuẩn multi-head
    "LEGEND_NAMES": {
        1: "TC (Tumor Core)",
        2: "WT (Whole Tumor)",
        3: "ET (Enhancing Tumor)",
    },
}


# =============================================================================
# ROOT
# =============================================================================
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


# =============================================================================
# Utility
# =============================================================================
def load_modality(case_dir: Path, modality: str) -> np.ndarray:
    nii = nib.load(str(case_dir / f"{modality}.nii.gz"))
    return nii.get_fdata().astype(np.float32)


def load_label(case_dir: Path) -> np.ndarray:
    nii = nib.load(str(case_dir / "mask.nii.gz"))
    seg = nii.get_fdata().astype(np.int16)
    seg[seg == 4] = 3
    return seg


def load_pred(pred_dir: Path, case_id: str) -> np.ndarray:
    nii = nib.load(str(pred_dir / f"{case_id}_pred.nii.gz"))
    seg = nii.get_fdata().astype(np.int16)
    seg[seg == 4] = 3
    return seg


def choose_slices(D: int, num_slices: int, fixed_slices):
    if fixed_slices is not None:
        return [int(z) for z in fixed_slices if 0 <= z < D]
    if D <= num_slices:
        return list(range(D))
    return list(np.linspace(0, D - 1, num_slices, dtype=int))


def build_overlay(seg: np.ndarray, colors: Dict[int, tuple]) -> np.ndarray:
    H, W = seg.shape
    out = np.zeros((H, W, 4), dtype=np.float32)
    for lb, col in colors.items():
        out[seg == lb] = col
    return out


# =============================================================================
# MAIN VISUALIZE
# =============================================================================
def visualize_case(
    case_id: str,
    base_modality: str,
    num_slices: int,
    fixed_slices,
    data_root: Path,
    pred_root: Path,
    out_dir: Path,
    figsize=(10, 5),
    label_colors=None,
    legend_names=None,
):
    if label_colors is None:
        label_colors = CFG_VIS["LABEL_COLORS"]
    if legend_names is None:
        legend_names = CFG_VIS["LEGEND_NAMES"]

    case_dir = data_root / case_id
    if not case_dir.exists():
        raise FileNotFoundError(case_dir)

    img_vol = load_modality(case_dir, base_modality)
    gt_vol = load_label(case_dir)
    pred_vol = load_pred(pred_root, case_id)

    H, W, D = img_vol.shape
    print(f"[{case_id}] Loaded shape = ({H}, {W}, {D})")

    slice_ids = choose_slices(D, num_slices, fixed_slices)
    print("Slices:", slice_ids)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 🎯 Legend handle
    legend_handles = []
    for lb, name in legend_names.items():
        rgba = label_colors[lb]
        legend_handles.append(
            Patch(facecolor=rgba[:3], edgecolor="k", label=name)
        )

    for z in slice_ids:
        img = img_vol[:, :, z]
        gt = gt_vol[:, :, z]
        pred = pred_vol[:, :, z]

        gt_overlay = build_overlay(gt, label_colors)
        pred_overlay = build_overlay(pred, label_colors)

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(
            f"{case_id} – slice z={z} – modality={base_modality}",
            fontsize=14
        )

        # ---- GT ----
        ax = axes[0]
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.imshow(gt_overlay, interpolation="nearest")
        ax.set_title("Ground Truth")
        ax.axis("off")
        ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

        # ---- Prediction ----
        ax = axes[1]
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.imshow(pred_overlay, interpolation="nearest")
        ax.set_title("Prediction")
        ax.axis("off")

        plt.tight_layout()
        out_path = out_dir / f"{case_id}_z{z:03d}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

        print("[SAVE]", out_path)


# =============================================================================
# ENTRY
# =============================================================================
def main():
    cfg = CFG_VIS
    case_id = cfg["CASE_ID"]

    visualize_case(
        case_id=case_id,
        base_modality=cfg["BASE_MODALITY"],
        num_slices=cfg["NUM_SLICES"],
        fixed_slices=cfg["FIXED_SLICES"],
        data_root=ROOT / cfg["DATA_ROOT_3D"],
        pred_root=ROOT / cfg["PRED_ROOT"],
        out_dir=ROOT / cfg["OUT_DIR"] / case_id,
        figsize=tuple(cfg["FIGSIZE"]),
        label_colors=cfg["LABEL_COLORS"],
        legend_names=cfg["LEGEND_NAMES"],
    )

    print("[OK] Done.")


if __name__ == "__main__":
    main()
