# -*- coding: utf-8 -*-
"""
Visualize kết quả dự đoán VNetMultiHead trên BraTS:
- Overlay Ground Truth và Prediction lên ảnh nền (1 modality).
- Mỗi slice vẽ 2 ảnh: GT overlay (trái) và Pred overlay (phải).

Giả định cấu trúc folder:
data/
  processed/
    3d/
      labeled/
        Brain_001/
          flair.nii.gz
          t1.nii.gz
          t1ce.nii.gz
          t2.nii.gz
          mask.nii.gz
experiments/
  brats3d_vnetmh_sup/
    inference/
      preds/
        Brain_001_pred.nii.gz
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt

try:
    import nibabel as nib
except ImportError:
    raise RuntimeError("Vui lòng cài nibabel: pip install nibabel")


# =============================================================================
# CONFIG
# =============================================================================
CFG_VIS: Dict[str, Any] = {
    # Case cần visualize
    "CASE_ID": "Brain_016",

    # Modality dùng làm ảnh nền: "flair" | "t1" | "t1ce" | "t2"
    "BASE_MODALITY": "t1ce",

    # Số lát axial cần vẽ. Nếu None => tự chọn 6 lát đều trên toàn thể tích
    "NUM_SLICES": 6,

    # Nếu muốn chỉ định chính xác các slice z (list[int]), set ở đây, ví dụ [60, 80, 100]
    # Nếu không, để None để script tự chọn đều.
    "FIXED_SLICES": [i for i in range(60, 100, 5)],

    # Thư mục dữ liệu & kết quả (tính từ ROOT)
    # "DATA_ROOT_3D": "data/processed/3d/labeled",
    # "PRED_ROOT": "experiments/brats3d_vnetmh_sup/inference/preds",

    # # Thư mục lưu hình output
    # "OUT_DIR": "experiments/brats3d_vnetmh_sup/vis",
    
    # Thư mục dữ liệu & kết quả (tính từ ROOT)
    "DATA_ROOT_3D": "data/processed/3d/labeled",
    "PRED_ROOT": "experiments/brats3d_vnetmh_sup/inference/preds",

    # Thư mục lưu hình output
    "OUT_DIR": "experiments/brats3d_vnetmh_sup/vis",

    # Kích thước figure (width, height) cho mỗi slice
    "FIGSIZE": (10, 5),

    # Màu overlay cho các label (RGBA)
    # 0: background (transparent) – không dùng
    "LABEL_COLORS": {
        1: (1.0, 0.0, 0.0, 0.5),  # label 1  -> đỏ
        2: (0.0, 1.0, 0.0, 0.5),  # label 2  -> xanh lá
        3: (0.0, 0.0, 1.0, 0.5),  # label 3  -> xanh dương
    },
}


# =============================================================================
# PATH & ROOT
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


# =============================================================================
# Utility
# =============================================================================

def load_modality(case_dir: Path, modality: str) -> np.ndarray:
    """
    Load 1 modality (flair / t1 / t1ce / t2).
    Trả về: vol (H,W,D) float32.
    """
    fname = f"{modality}.nii.gz"
    path = case_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file modality: {path}")

    nii = nib.load(str(path))
    vol = nii.get_fdata().astype(np.float32)  # (H,W,D) cho BraTS
    return vol


def load_label(case_dir: Path, name: str = "mask.nii.gz") -> np.ndarray:
    path = case_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy mask GT: {path}")
    nii = nib.load(str(path))
    seg = nii.get_fdata().astype(np.int16)  # (H,W,D)
    seg[seg == 4] = 3  # phòng trường hợp còn label 4
    return seg


def load_pred(pred_dir: Path, case_id: str) -> np.ndarray:
    path = pred_dir / f"{case_id}_pred.nii.gz"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy mask dự đoán: {path}")
    nii = nib.load(str(path))
    seg = nii.get_fdata().astype(np.int16)  # (H,W,D)
    seg[seg == 4] = 3
    return seg


def normalize_to_0_1(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    vmin = np.percentile(img, 1)
    vmax = np.percentile(img, 99)
    img = np.clip(img, vmin, vmax)
    if vmax - vmin < 1e-6:
        return np.zeros_like(img)
    img = (img - vmin) / (vmax - vmin)
    return img


def choose_slices(D: int, num_slices: int, fixed_slices: List[int] | None = None) -> List[int]:
    if fixed_slices is not None:
        return [int(z) for z in fixed_slices if 0 <= int(z) < D]
    if num_slices is None or num_slices <= 0:
        num_slices = 6
    if D <= num_slices:
        return list(range(D))
    indices = np.linspace(0, D - 1, num_slices, dtype=int)
    # ép về int thường để tránh np.int64 in log
    return [int(i) for i in indices]


def build_overlay_rgba(seg_slice: np.ndarray, label_colors: Dict[int, tuple]) -> np.ndarray:
    """
    seg_slice: (H,W) int (0..3)
    label_colors: mapping label -> (r,g,b,a)
    Trả về: rgba (H,W,4)
    """
    H, W = seg_slice.shape
    overlay = np.zeros((H, W, 4), dtype=np.float32)  # mặc định alpha=0

    for lb, color in label_colors.items():
        mask = seg_slice == lb
        if not np.any(mask):
            continue
        overlay[mask] = color

    return overlay


def visualize_case(
    case_id: str,
    base_modality: str,
    num_slices: int,
    fixed_slices,
    data_root: Path,
    pred_root: Path,
    out_dir: Path,
    figsize=(10, 5),
    label_colors: Dict[int, tuple] | None = None,
):
    if label_colors is None:
        label_colors = CFG_VIS["LABEL_COLORS"]

    case_dir = data_root / case_id
    if not case_dir.exists():
        raise FileNotFoundError(f"Thư mục case không tồn tại: {case_dir}")

    # Load volumes: (H,W,D)
    img_vol = load_modality(case_dir, base_modality)
    gt_vol = load_label(case_dir, "mask.nii.gz")
    pred_vol = load_pred(pred_root, case_id)

    H, W, D = img_vol.shape
    print(f"[{case_id}] shape = (H={H}, W={W}, D={D})")

    img_norm = normalize_to_0_1(img_vol)

    # Chọn slice (axial: theo trục D = axis 2)
    slice_ids = choose_slices(D, num_slices, fixed_slices)
    print(f"[{case_id}] visualize axial slices (z over D): {slice_ids}")

    out_dir.mkdir(parents=True, exist_ok=True)

    for z in slice_ids:
        # axial slice: [:,:,z]
        img_z = img_norm[:, :, z]   # (H,W)
        gt_z = gt_vol[:, :, z]
        pred_z = pred_vol[:, :, z]

        gt_overlay = build_overlay_rgba(gt_z, label_colors)
        pred_overlay = build_overlay_rgba(pred_z, label_colors)

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f"{case_id} - axial slice z={z} - modality={base_modality}", fontsize=12)

        # --- GT ---
        ax = axes[0]
        ax.imshow(img_z, cmap="gray", interpolation="nearest")
        ax.imshow(gt_overlay, interpolation="nearest")
        ax.set_title("Ground Truth")
        ax.axis("off")

        # --- Pred ---
        ax = axes[1]
        ax.imshow(img_z, cmap="gray", interpolation="nearest")
        ax.imshow(pred_overlay, interpolation="nearest")
        ax.set_title("Prediction")
        ax.axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        out_path = out_dir / f"{case_id}_slice_{z:03d}.png"
        plt.savefig(str(out_path), dpi=150)
        plt.close(fig)

        print(f"[SAVE] {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    case_id = CFG_VIS["CASE_ID"]
    base_modality = CFG_VIS["BASE_MODALITY"]
    num_slices = CFG_VIS["NUM_SLICES"]
    fixed_slices = CFG_VIS["FIXED_SLICES"]

    data_root = ROOT / CFG_VIS["DATA_ROOT_3D"]
    pred_root = ROOT / CFG_VIS["PRED_ROOT"]
    out_dir = ROOT / CFG_VIS["OUT_DIR"] / case_id

    print("=== VISUALIZE VNetMultiHead BraTS ===")
    print(f"ROOT:         {ROOT}")
    print(f"Case ID:      {case_id}")
    print(f"Data root:    {data_root}")
    print(f"Pred root:    {pred_root}")
    print(f"Out dir:      {out_dir}")
    print(f"Modality:     {base_modality}")

    visualize_case(
        case_id=case_id,
        base_modality=base_modality,
        num_slices=num_slices,
        fixed_slices=fixed_slices,
        data_root=data_root,
        pred_root=pred_root,
        out_dir=out_dir,
        figsize=tuple(CFG_VIS["FIGSIZE"]),
        label_colors=CFG_VIS["LABEL_COLORS"],
    )

    print("[OK] Visualization done.")


if __name__ == "__main__":
    main()
