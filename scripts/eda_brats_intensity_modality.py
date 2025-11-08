# -*- coding: utf-8 -*-
"""
EDA BraTS2020 (raw 3D) — Intensity Analysis theo modality.

Mục tiêu:
1) Phân bố histogram của FLAIR, T1, T1CE, T2 trong VÙNG U (tumor).
   - Dùng chuẩn hoá "robust" per-volume về [0,1] theo percentile (1–99) trên vùng nonzero.
   - Nền (voxel gốc == 0) luôn = 0.
   - Cộng gộp histogram across-cases (pooled) để có 01 đường phân bố / modality.

2) Tính tương quan Pearson giữa TRUNG BÌNH VOXEL TỪNG MODALITY trong VÙNG U (per-case).
   - Kết quả: ma trận tương quan 4×4 (FLAIR, T1, T1CE, T2).
   - Trực quan: heatmap.

3) So sánh mean signal trong TUMOR vs NÃO CÒN LẠI (brain − tumor) cho từng modality (per-case).
   - Xuất bảng CSV chứa các cột mean_tumor_* và mean_brainrest_* để bạn vẽ boxplot sau nếu muốn.

Biểu đồ xuất:
- Overlay histogram (4 modality) trong vùng U — trục x là cường độ đã normalize [0,1].
- Heatmap tương quan giữa các modality (Pearson) trên vector mean_tumor_* (per-case).

Yêu cầu:
    pip install nibabel numpy pandas matplotlib seaborn tqdm
"""

import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# ---------------- CONFIG ----------------
RAW_TRAIN = r"D:\Project Advanced CV\data\BraST2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
OUT_DIR   = r"D:\Project Advanced CV\experiments\eda\intensity_modality"

# Output file names
CSV_OUT                 = "intensity_modality_means_per_case.csv"
HIST_OVERLAY_FIG_OUT    = "overlay_hist_tumor_modalities_norm01.png"
CORR_HEATMAP_FIG_OUT    = "corr_heatmap_tumor_means.png"

# Robust normalization percentiles
PMIN = 1.0
PMAX = 99.0

# Histogram bins
NBINS = 256
# ---------------------------------------


def as_ras_data(fp: Path) -> tuple[np.ndarray, tuple]:
    img = nib.load(str(fp))
    img = nib.as_closest_canonical(img)  # RAS
    return img.get_fdata(dtype=np.float32), img.header.get_zooms()


def find_case_id(name: str) -> str:
    m = re.search(r"_(\d+)$", name)
    return m.group(1) if m else name


def normalize_to_01(vol: np.ndarray, pmin=1.0, pmax=99.0, eps=1e-8) -> np.ndarray:
    """Percentile normalization trên vùng >0 rồi scale về [0,1]; giữ nền=0."""
    vol = vol.astype(np.float32, copy=False)
    nz = vol[vol > 0]
    if nz.size >= 10:
        lo, hi = np.percentile(nz, [pmin, pmax]).astype(np.float32)
    else:
        lo, hi = np.percentile(vol, [pmin, pmax]).astype(np.float32)
    if hi <= lo:
        hi = lo + 1.0
    x = np.clip(vol, lo, hi)
    x = (x - lo) / (hi - lo + eps)
    x[vol == 0] = 0.0
    return x


def scan_training_cases(root_dir: str) -> list[Path]:
    root = Path(root_dir)
    cases = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("BraTS20_Training_")])
    return cases


def pooled_histogram(values01: np.ndarray, bins: int = 256, range_=(0.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    """Trả về (hist_counts, bin_edges) từ mảng đã normalize [0,1]."""
    h, edges = np.histogram(values01, bins=bins, range=range_)
    return h.astype(np.float64), edges  # float để cộng dồn an toàn


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = scan_training_cases(RAW_TRAIN)

    # Bộ đếm histogram gộp cho vùng TUMOR
    hist_pool = {
        "flair": np.zeros(NBINS, dtype=np.float64),
        "t1":    np.zeros(NBINS, dtype=np.float64),
        "t1ce":  np.zeros(NBINS, dtype=np.float64),
        "t2":    np.zeros(NBINS, dtype=np.float64),
    }
    bin_edges_ref = None

    # Bảng mean tín hiệu trong tumor & brain-rest (per-case)
    rows = []

    for cdir in tqdm(cases, desc="Intensity analysis (training with labels)"):
        name = cdir.name
        # Đường dẫn file
        flair_fp = cdir / f"{name}_flair.nii"
        t1_fp    = cdir / f"{name}_t1.nii"
        t1ce_fp  = cdir / f"{name}_t1ce.nii"
        t2_fp    = cdir / f"{name}_t2.nii"
        seg_fp   = cdir / f"{name}_seg.nii"

        if not all(p.exists() for p in [flair_fp, t1_fp, t1ce_fp, t2_fp, seg_fp]):
            # thiếu file — bỏ qua case
            continue

        flair, _ = as_ras_data(flair_fp)
        t1, _    = as_ras_data(t1_fp)
        t1ce, _  = as_ras_data(t1ce_fp)
        t2, _    = as_ras_data(t2_fp)
        seg, _   = as_ras_data(seg_fp)

        # Robust normalization về [0,1] (per-volume, nền=0)
        flair01 = normalize_to_01(flair, PMIN, PMAX)
        t101    = normalize_to_01(t1, PMIN, PMAX)
        t1ce01  = normalize_to_01(t1ce, PMIN, PMAX)
        t201    = normalize_to_01(t2, PMIN, PMAX)

        # Mask tumor & brain
        tumor_mask = seg > 0
        brain_mask = t1 > 0
        brain_rest_mask = brain_mask & (~tumor_mask)

        # Nếu không có u (cực hiếm) thì bỏ qua để tránh NaN
        if not np.any(tumor_mask):
            continue

        # 1) POOL HISTOGRAM trong tumor cho từng modality
        for mod, arr01 in zip(["flair","t1","t1ce","t2"], [flair01, t101, t1ce01, t201]):
            vals = arr01[tumor_mask]
            h, edges = pooled_histogram(vals, bins=NBINS, range_=(0.0, 1.0))
            hist_pool[mod] += h
            if bin_edges_ref is None:
                bin_edges_ref = edges

        # 2) MEAN trong tumor vs brain-rest (per-case)
        means = {
            "mean_tumor_flair": float(np.mean(flair01[tumor_mask])),
            "mean_tumor_t1":    float(np.mean(t101[tumor_mask])),
            "mean_tumor_t1ce":  float(np.mean(t1ce01[tumor_mask])),
            "mean_tumor_t2":    float(np.mean(t201[tumor_mask])),
        }

        # brain-rest có thể rỗng nếu T1 gần như 0 (không xảy ra ở BraTS, nhưng cứ kiểm tra cho chắc)
        if np.any(brain_rest_mask):
            means.update({
                "mean_brainrest_flair": float(np.mean(flair01[brain_rest_mask])),
                "mean_brainrest_t1":    float(np.mean(t101[brain_rest_mask])),
                "mean_brainrest_t1ce":  float(np.mean(t1ce01[brain_rest_mask])),
                "mean_brainrest_t2":    float(np.mean(t201[brain_rest_mask])),
            })
        else:
            means.update({
                "mean_brainrest_flair": np.nan,
                "mean_brainrest_t1":    np.nan,
                "mean_brainrest_t1ce":  np.nan,
                "mean_brainrest_t2":    np.nan,
            })

        rows.append({
            "case": name,
            "id": find_case_id(name),
            **means
        })

    # --- Xuất bảng per-case ---
    df = pd.DataFrame(rows).sort_values("id")
    df.to_csv(Path(OUT_DIR) / CSV_OUT, index=False)

    # --- Overlay histogram (tumor, normalized [0,1]) ---
    # Chuyển counts -> density để so sánh trực quan
    plt.figure()
    centers = 0.5 * (bin_edges_ref[:-1] + bin_edges_ref[1:])
    for mod, label in [("flair","FLAIR"), ("t1","T1"), ("t1ce","T1CE"), ("t2","T2")]:
        counts = hist_pool[mod]
        density = counts / (counts.sum() + 1e-12)
        plt.plot(centers, density, label=label)
    plt.xlabel("Normalized intensity in tumor [0,1]")
    plt.ylabel("Density")
    plt.title("Overlay histogram in tumor region (pooled, normalized per-volume)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(OUT_DIR) / HIST_OVERLAY_FIG_OUT, dpi=220)
    plt.close()

    # --- Pearson correlation heatmap giữa các modality ---
    # Dùng các cột mean_tumor_* (per-case) để tính tương quan giữa 4 modality
    tumor_means = df[["mean_tumor_flair","mean_tumor_t1","mean_tumor_t1ce","mean_tumor_t2"]].dropna()
    tumor_means.columns = ["FLAIR","T1","T1CE","T2"]
    corr = tumor_means.corr(method="pearson")

    plt.figure(figsize=(5.2, 4.6))
    sns.heatmap(corr, annot=True, vmin=-1, vmax=1, cmap="coolwarm", square=True, fmt=".2f")
    plt.title("Pearson correlation (mean intensity in tumor)")
    plt.tight_layout()
    plt.savefig(Path(OUT_DIR) / CORR_HEATMAP_FIG_OUT, dpi=220)
    plt.close()

    print(f"[OK] Saved to: {OUT_DIR}")
    print(f" - CSV: {CSV_OUT}")
    print(f" - Overlay histogram: {HIST_OVERLAY_FIG_OUT}")
    print(f" - Corr heatmap: {CORR_HEATMAP_FIG_OUT}")


if __name__ == "__main__":
    main()
