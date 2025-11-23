# -*- coding: utf-8 -*-
"""
Tiền xử lý 3D BraTS2020 → data/processed/3d/{labeled, unlabeled}

✔ Orientation RAS
✔ Crop không gian: x=22:216, y=16:210, z giữ nguyên
✔ Chuẩn hóa cường độ theo non-zero voxels, với 2 chế độ:
    - "minmax":  về [0, 1] (min–max per-volume, non-zero)
    - "zscore":  z-score trên non-zero (tùy chọn clip)
✔ Mapping mask: 4 → 3
✔ Lưu output theo format:
   data/processed/3d/labeled/Brain_XXX/{flair,t1,t1ce,t2,mask}.nii.gz
   data/processed/3d/unlabeled/Brain_XXX/{flair,t1,t1ce,t2}.nii.gz
"""

from pathlib import Path
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm


# ======================= CONFIG =======================
ROOT = Path(r"D:\Project Advanced CV")

RAW_TRAIN = ROOT / r"data\BraST2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
RAW_VALID = ROOT / r"data\BraST2020\BraTS2020_ValidationData\MICCAI_BraTS2020_ValidationData"

OUT_LABELED   = ROOT / r"data\processed\3d\labeled"
OUT_UNLABELED = ROOT / r"data\processed\3d\unlabeled"

CROP = {
    "x_min": 22,
    "x_max": 216,
    "y_min": 16,
    "y_max": 210
}

MOD_SUFFIX = {
    "flair": "flair",
    "t1":    "t1",
    "t1ce":  "t1ce",
    "t2":    "t2",
    "seg":   "seg",
}

# ---- CHỌN CHẾ ĐỘ CHUẨN HÓA ----
# "minmax" : chuẩn hóa về [0,1] theo min–max trên non-zero
# "zscore" : chuẩn hóa theo z-score trên non-zero (mean=0, std=1, có thể clip)
NORM_MODE = "zscore"   # hoặc "zscore"

# Nếu dùng z-score, có thể clip giá trị để tránh outlier quá lớn
# Đặt thành None nếu không muốn clip
ZSCORE_CLIP = (-5.0, 5.0)
# =======================================================


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def find_nifti(case_dir: Path, case_name: str, suffix: str):
    """Tìm file NIfTI dạng {case_name}_{suffix}.nii*"""
    pattern = str(case_dir / f"{case_name}_{suffix}.nii*")
    hits = glob.glob(pattern)
    return Path(hits[0]) if hits else None


def load_ras(path: Path):
    img = nib.load(str(path))
    img_ras = nib.as_closest_canonical(img)
    return img_ras.get_fdata(dtype=np.float32), img_ras


def crop_xy(vol: np.ndarray):
    x0, x1 = CROP["x_min"], CROP["x_max"]
    y0, y1 = CROP["y_min"], CROP["y_max"]
    return vol[x0:x1, y0:y1, :]


# ---------------- NORMALIZATION ----------------
def normalize_minmax_nonzero(vol: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa về [0,1] theo min–max trên các voxel non-zero.
    Các voxel =0 (ngoài não) giữ nguyên =0.
    """
    nz = vol[vol > 0]
    if nz.size == 0:
        return np.zeros_like(vol, dtype=np.float32)
    vmin, vmax = nz.min(), nz.max()
    if vmax <= vmin:
        return np.zeros_like(vol, dtype=np.float32)
    out = (vol - vmin) / (vmax - vmin)
    out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)


def normalize_zscore_nonzero(vol: np.ndarray,
                             clip: tuple | None = None,
                             eps: float = 1e-8) -> np.ndarray:
    """
    Chuẩn hóa theo z-score trên các voxel non-zero:
        x' = (x - mean) / std
    Các voxel =0 giữ nguyên =0.
    Có thể clip về [clip[0], clip[1]] nếu clip không phải None.
    """
    out = np.zeros_like(vol, dtype=np.float32)

    mask_nz = vol > 0
    if not np.any(mask_nz):
        return out

    vals = vol[mask_nz].astype(np.float32)
    mean = vals.mean()
    std = vals.std()
    if std < eps:
        # Nếu std quá nhỏ, tránh chia cho ~0 → trả về 0
        return out

    norm_vals = (vals - mean) / (std + eps)
    if clip is not None:
        lo, hi = clip
        norm_vals = np.clip(norm_vals, lo, hi)

    out[mask_nz] = norm_vals
    return out.astype(np.float32)


def normalize_intensity(vol: np.ndarray) -> np.ndarray:
    """
    Wrapper chọn hàm normalize theo NORM_MODE.
    """
    mode = NORM_MODE.lower()
    if mode == "minmax":
        return normalize_minmax_nonzero(vol)
    elif mode == "zscore":
        return normalize_zscore_nonzero(vol, clip=ZSCORE_CLIP)
    else:
        raise ValueError(f"Unknown NORM_MODE='{NORM_MODE}', expected 'minmax' or 'zscore'.")


# ---------------- MASK / SAVE ----------------
def map_seg(seg: np.ndarray):
    seg = seg.astype(np.uint8)
    return np.where(seg == 4, 3, seg)


def save_nifti(data: np.ndarray, ref_img, out_path: Path):
    img = nib.Nifti1Image(data, ref_img.affine, ref_img.header)
    nib.save(img, str(out_path))


# ---------------- PROCESS LABELED ----------------
def process_labeled(case_dir: Path):
    case_name = case_dir.name                # BraTS20_Training_001
    out_name  = f"Brain_{case_name.split('_')[-1]}"   # Brain_001
    out_dir   = OUT_LABELED / out_name
    ensure_dir(out_dir)

    # ---- 4 modality ----
    for mod in ["flair", "t1", "t1ce", "t2"]:
        fp = find_nifti(case_dir, case_name, MOD_SUFFIX[mod])
        if fp is None:
            raise FileNotFoundError(f"Missing {mod} in {case_name}")

        vol, img_ras = load_ras(fp)
        vol = crop_xy(vol)
        vol = normalize_intensity(vol)

        save_nifti(vol, img_ras, out_dir / f"{mod}.nii.gz")

    # ---- seg ----
    seg_fp = find_nifti(case_dir, case_name, "seg")
    if seg_fp is None:
        raise FileNotFoundError(f"No seg for {case_name}")

    seg_vol, seg_img = load_ras(seg_fp)
    seg_vol = crop_xy(seg_vol)
    seg_vol = map_seg(seg_vol)

    save_nifti(seg_vol.astype(np.uint8), seg_img, out_dir / "mask.nii.gz")


# ---------------- PROCESS UNLABELED ----------------
def process_unlabeled(case_dir: Path):
    case_name = case_dir.name
    out_name  = f"Brain_{case_name.split('_')[-1]}"
    out_dir   = OUT_UNLABELED / out_name
    ensure_dir(out_dir)

    for mod in ["flair", "t1", "t1ce", "t2"]:
        fp = find_nifti(case_dir, case_name, mod)
        if fp is None:
            raise FileNotFoundError(f"Missing {mod} in {case_name}")

        vol, img_ras = load_ras(fp)
        vol = crop_xy(vol)
        vol = normalize_intensity(vol)

        save_nifti(vol, img_ras, out_dir / f"{mod}.nii.gz")


# ---------------- MAIN ----------------
def main():
    print(f"[INFO] NORM_MODE     = {NORM_MODE}")
    if NORM_MODE.lower() == "zscore":
        print(f"[INFO] ZSCORE_CLIP   = {ZSCORE_CLIP}")

    ensure_dir(OUT_LABELED)
    ensure_dir(OUT_UNLABELED)

    train_cases = sorted([d for d in RAW_TRAIN.iterdir() if d.is_dir()])
    valid_cases = sorted([d for d in RAW_VALID.iterdir() if d.is_dir()])

    print(f"[INFO] Labeled cases:   {len(train_cases)}")
    print(f"[INFO] Unlabeled cases: {len(valid_cases)}")

    # -------- Labeled --------
    for c in tqdm(train_cases, desc="Processing labeled"):
        try:
            process_labeled(c)
        except Exception as e:
            print(f"[WARN] Skip {c.name}: {e}")

    # -------- Unlabeled --------
    for c in tqdm(valid_cases, desc="Processing unlabeled"):
        try:
            process_unlabeled(c)
        except Exception as e:
            print(f"[WARN] Skip {c.name}: {e}")

    print("[OK] Done preprocessing BraTS3D.")


if __name__ == "__main__":
    main()
