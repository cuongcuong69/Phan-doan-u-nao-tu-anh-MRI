# -*- coding: utf-8 -*-
"""
BraTS2020 — Shape Analysis từ mask 3D (raw data, training có nhãn).

Trích cho mỗi ca:
- Volume (mm^3, mL)
- Surface area (mm^2) bằng marching cubes (tôn trọng voxel spacing)
- Compactness = (36π V^2) / A^3
- Sphericity = π^(1/3) * (6V)^(2/3) / A
- Elongation = sqrt(λ2 / λ1), Flatness = sqrt(λ3 / λ1) từ PCA (eigenvalues của tọa độ voxel)
  (λ1 ≥ λ2 ≥ λ3 là các trị riêng của ma trận hiệp phương sai điểm voxel (đơn vị mm^2))

Vùng phân tích:
- WT = (1|2|4)
- TC = (1|4)
- ET = (4)
- (thêm riêng: NCR=1, ED=2, ET=4)  # để đầy đủ

Yêu cầu:
    pip install nibabel numpy pandas scikit-image tqdm

Output:
    D:\Project Advanced CV\experiments\eda\shape\brats_shape_stats.csv
"""

import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from skimage import measure


# ---------------- CONFIG ----------------
RAW_TRAIN = r"D:\Project Advanced CV\data\BraST2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
OUT_DIR   = r"D:\Project Advanced CV\experiments\eda\shape"
CSV_OUT   = "brats_shape_stats.csv"
MIN_VOXELS = 20  # bỏ qua ROI quá nhỏ để tránh marching_cubes lỗi/mesh nhiễu
# ---------------------------------------


def as_ras_data(fp: Path):
    img = nib.load(str(fp))
    img = nib.as_closest_canonical(img)  # RAS
    data = img.get_fdata(dtype=np.float32)
    zooms = img.header.get_zooms()[:3]   # spacing (mm)
    return data, zooms


def find_case_id(name: str) -> str:
    m = re.search(r"_(\d+)$", name)
    return m.group(1) if m else name


def surface_area_from_mesh(verts: np.ndarray, faces: np.ndarray) -> float:
    """
    verts: (N, 3) in mm (đã scale bằng spacing)
    faces: (M, 3) indices
    Trả về diện tích bề mặt (mm^2)
    """
    tri = verts[faces]  # (M,3,3)
    a = tri[:, 1] - tri[:, 0]
    b = tri[:, 2] - tri[:, 0]
    cross = np.cross(a, b)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    return float(np.sum(area))


def compute_pca_axes_mm(mask: np.ndarray, zooms: tuple) -> tuple:
    """
    PCA trên tọa độ voxel (mm). Trả về eigenvalues λ1 ≥ λ2 ≥ λ3 (mm^2).
    """
    coords = np.argwhere(mask)  # indices (z,y,x) or (x,y,z)?
    # np.argwhere trên mảng (H,W,D) trả về (i,j,k) = (x,y,z) nếu mảng là (x,y,z).
    # Ở đây dữ liệu nib.load(as_closest_canonical) -> shape là (X,Y,Z) = (row, col, slice)
    # Ta convert sang mm bằng spacing (zooms) theo thứ tự (x_mm, y_mm, z_mm):
    X = coords.astype(np.float64)
    X[:, 0] *= zooms[0]
    X[:, 1] *= zooms[1]
    X[:, 2] *= zooms[2]
    # PCA qua covariance:
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)  # mm^2
    w, _ = np.linalg.eigh(cov)             # trả về tăng dần
    w = np.sort(w)[::-1]                   # giảm dần: λ1 ≥ λ2 ≥ λ3
    return w  # (3,)


def compute_shape_for_region(seg: np.ndarray, zooms: tuple, region_mask: np.ndarray) -> dict:
    """
    Tính shape metrics cho 1 region (mask nhị phân) trong cùng không gian với seg.
    """
    vox_count = int(np.count_nonzero(region_mask))
    if vox_count < MIN_VOXELS:
        return {
            "voxels": vox_count, "vol_mm3": np.nan, "vol_ml": np.nan,
            "surf_mm2": np.nan, "compactness": np.nan, "sphericity": np.nan,
            "elongation": np.nan, "flatness": np.nan
        }

    voxel_vol = float(np.prod(zooms))  # mm^3
    vol_mm3 = vox_count * voxel_vol

    # marching cubes: để có bề mặt tương đối mượt, dùng level=0.5 trên mask nhị phân
    try:
        verts, faces, _, _ = measure.marching_cubes(region_mask.astype(np.uint8), level=0.5, spacing=zooms)
        surf_mm2 = surface_area_from_mesh(verts, faces)
    except Exception:
        # fallback: không tính được mesh
        surf_mm2 = np.nan

    # Compactness, Sphericity
    if surf_mm2 and np.isfinite(surf_mm2) and surf_mm2 > 0:
        V = vol_mm3
        A = surf_mm2
        compactness = (36.0 * np.pi * (V ** 2)) / (A ** 3)
        sphericity = (np.pi ** (1.0/3.0)) * ((6.0 * V) ** (2.0/3.0)) / A
    else:
        compactness = np.nan
        sphericity = np.nan

    # PCA axes: eigenvalues λ1 ≥ λ2 ≥ λ3 (mm^2)
    try:
        eigvals = compute_pca_axes_mm(region_mask, zooms)
        if eigvals[0] <= 0:
            elong = np.nan
            flat = np.nan
        else:
            elong = float(np.sqrt(max(eigvals[1]/eigvals[0], 0.0)))
            flat  = float(np.sqrt(max(eigvals[2]/eigvals[0], 0.0)))
    except Exception:
        elong = np.nan
        flat = np.nan

    return {
        "voxels": vox_count,
        "vol_mm3": float(vol_mm3),
        "vol_ml": float(vol_mm3 / 1000.0),
        "surf_mm2": float(surf_mm2) if np.isfinite(surf_mm2) else np.nan,
        "compactness": float(compactness) if np.isfinite(compactness) else np.nan,
        "sphericity": float(sphericity) if np.isfinite(sphericity) else np.nan,
        "elongation": elong,
        "flatness": flat
    }


def process_case(case_dir: Path) -> dict:
    """
    Tính shape metrics cho các ROI: WT, TC, ET và (raw) NCR(1), ED(2), ET(4).
    """
    name = case_dir.name
    seg_fp = case_dir / f"{name}_seg.nii"
    if not seg_fp.exists():
        return {}

    seg, zooms = as_ras_data(seg_fp)

    # ROI masks
    m1  = (seg == 1)           # NCR/NET
    m2  = (seg == 2)           # ED
    m4  = (seg == 4)           # ET
    mWT = (seg > 0)            # WT
    mTC = ((seg == 1) | (seg == 4))  # TC

    out = {"case": name, "id": find_case_id(name)}
    # Tính cho mỗi ROI
    roi_defs = [
        ("WT", mWT),
        ("TC", mTC),
        ("ET", m4),
        ("NCR", m1),
        ("ED", m2),
        ("ET_raw", m4),  # ET đã có ở trên; để minh bạch, ET_raw = ET
    ]
    seen = set()
    for roi_name, mask in roi_defs:
        if roi_name in seen:
            continue
        seen.add(roi_name)
        stats = compute_shape_for_region(seg, zooms, mask)
        for k, v in stats.items():
            out[f"{roi_name}_{k}"] = v

    return out


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(RAW_TRAIN)
    cases = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("BraTS20_Training_")])

    rows = []
    for cdir in tqdm(cases, desc="Shape analysis (training)"):
        s = process_case(cdir)
        if s:
            rows.append(s)

    df = pd.DataFrame(rows)
    if not df.empty and "id" in df.columns:
        df["id"] = df["id"].astype(str)
        df.sort_values("id", inplace=True)

    csv_path = out_dir / CSV_OUT
    df.to_csv(csv_path, index=False)
    print(f"[OK] Saved shape stats to: {csv_path}")


if __name__ == "__main__":
    main()
