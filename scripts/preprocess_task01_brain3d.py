# -*- coding: utf-8 -*-
"""
Tiền xử lý 3D MSD Task01_BrainTumour → data/processed_task01/3d/{labeled, unlabeled}

Mục tiêu: đưa Task01 về format giống BraTS2020 đã tiền xử lý, để dùng chung pipeline & độ đo.

✔ Đọc ảnh 4D (stack 4 modality) và tách thành 4 ảnh 3D:
    - 0: FLAIR  → flair.nii.gz
    - 1: T1w    → t1.nii.gz
    - 2: t1gd   → t1ce.nii.gz
    - 3: T2w    → t2.nii.gz

✔ Orientation RAS cho cả ảnh & nhãn

✔ Crop không gian (X,Y) GIỐNG HỆT BraTS:
    x = 22:216
    y = 16:210
    z giữ nguyên

✔ Chuẩn hóa cường độ theo non-zero voxels, với 2 chế độ:
    - "minmax":  về [0, 1] (min–max per-volume, non-zero)
    - "zscore":  z-score trên non-zero (có thể clip)

✔ Remap nhãn Task01 → chuẩn BraTS2020:
    Task01:   0: background
              1: edema
              2: non-enhancing tumor
              3: enhancing tumour

    BraTS:    0: background
              1: NCR/NET (non-enhancing tumor core)
              2: ED (edema)
              3: ET (enhancing tumor)

    Mapping:
        0 → 0
        1 → 2
        2 → 1
        3 → 3

✔ Lưu output theo format:
   data/processed_task01/3d/labeled/Brain_XXX/{flair,t1,t1ce,t2,mask}.nii.gz
   data/processed_task01/3d/unlabeled/Brain_XXX/{flair,t1,t1ce,t2}.nii.gz
"""

from pathlib import Path
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm


# ======================= CONFIG =======================
ROOT = Path(r"D:\Project Advanced CV")

TASK01_ROOT = ROOT / r"data\Task01_BrainTumour\Task01_BrainTumour"
IMAGES_TR   = TASK01_ROOT / "imagesTr"
IMAGES_TS   = TASK01_ROOT / "imagesTs"
LABELS_TR   = TASK01_ROOT / "labelsTr"

OUT_ROOT      = ROOT / r"data\processed_task01\3d"
OUT_LABELED   = OUT_ROOT / "labeled"
OUT_UNLABELED = OUT_ROOT / "unlabeled"

# ---- VÙNG CROP CỐ ĐỊNH (GIỐNG BraTS) ----
CROP = {
    "x_min": 22,
    "x_max": 216,
    "y_min": 16,
    "y_max": 210
}

# ---- CHỌN CHẾ ĐỘ CHUẨN HÓA ----
# "minmax" : chuẩn hóa về [0,1] theo min–max trên non-zero
# "zscore" : chuẩn hóa theo z-score trên non-zero (mean=0, std=1, có thể clip)
NORM_MODE = "zscore"   # hoặc "minmax"

# Nếu dùng z-score, có thể clip giá trị để tránh outlier quá lớn
# Đặt thành None nếu không muốn clip
ZSCORE_CLIP = (-5.0, 5.0)
# =======================================================


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def list_nifti_files(folder: Path):
    """
    Liệt kê các file *.nii* trong folder, bỏ qua các file ẩn kiểu '._xxx.nii.gz'.
    """
    all_files = sorted(glob.glob(str(folder / "*.nii*")))
    return [Path(f) for f in all_files if not Path(f).name.startswith("._")]


def load_ras(path: Path):
    """
    Load NIfTI và convert về orientation RAS.
    Trả về: data (numpy array), ref_img (Nifti1Image sau khi canonical).
    """
    img = nib.load(str(path))
    img_ras = nib.as_closest_canonical(img)
    data = img_ras.get_fdata(dtype=np.float32)
    return data, img_ras


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


# ---------------- 4D → 3D MODALITIES ----------------
def split_modalities_4d(data_4d: np.ndarray):
    """
    data_4d: 4D volume chứa 4 modality của Task01_BrainTumour.
    Có thể có shape:
       - (4, X, Y, Z)  → channel_first
       - (X, Y, Z, 4)  → channel_last

    Trả về dict:
        {
            "flair": vol3d,
            "t1":    vol3d,
            "t1ce":  vol3d,
            "t2":    vol3d,
        }
    với thứ tự mapping:
        0: FLAIR, 1: T1w, 2: t1gd, 3: T2w
    """
    if data_4d.ndim != 4:
        raise ValueError(f"Expected 4D data, got shape={data_4d.shape}")

    # Xác định trục channel
    if data_4d.shape[0] == 4:
        # (4, X, Y, Z)
        flair = data_4d[0, ...]
        t1    = data_4d[1, ...]
        t1ce  = data_4d[2, ...]
        t2    = data_4d[3, ...]
    elif data_4d.shape[-1] == 4:
        # (X, Y, Z, 4)
        flair = data_4d[..., 0]
        t1    = data_4d[..., 1]
        t1ce  = data_4d[..., 2]
        t2    = data_4d[..., 3]
    else:
        raise ValueError(f"Cannot infer channel axis for shape={data_4d.shape}")

    return {
        "flair": flair.astype(np.float32),
        "t1":    t1.astype(np.float32),
        "t1ce":  t1ce.astype(np.float32),
        "t2":    t2.astype(np.float32),
    }


# ---------------- CROP ----------------
def crop_xy(vol: np.ndarray) -> np.ndarray:
    """
    Crop volume 3D theo CROP cố định, giữ nguyên trục z.

    vol shape: (X, Y, Z)
    CROP: {"x_min": 22, "x_max": 216, "y_min": 16, "y_max": 210}
    """
    X, Y, Z = vol.shape
    x0, x1 = CROP["x_min"], CROP["x_max"]
    y0, y1 = CROP["y_min"], CROP["y_max"]

    # Safety check đơn giản (nếu dataset cùng kích thước BraTS thì sẽ pass)
    if x1 > X or y1 > Y:
        raise ValueError(
            f"CROP range ({x0}:{x1}, {y0}:{y1}) vượt quá kích thước volume ({X}, {Y}, {Z})."
        )

    return vol[x0:x1, y0:y1, :]


def save_nifti(data: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path):
    """
    Lưu NIfTI mới dùng affine & header từ ref_img.
    """
    img = nib.Nifti1Image(data, ref_img.affine, ref_img.header)
    nib.save(img, str(out_path))


# ---------------- LABEL REMAP ----------------
def map_seg_task01_to_brats(seg: np.ndarray) -> np.ndarray:
    """
    Remap nhãn Task01_BrainTumour → chuẩn BraTS2020:

        Task01:   0: background
                  1: edema
                  2: non-enhancing tumor
                  3: enhancing tumour

        BraTS:    0: background
                  1: NCR/NET (non-enhancing tumor core)
                  2: ED (edema)
                  3: ET (enhancing tumor)

    Mapping:
        0 → 0
        1 → 2
        2 → 1
        3 → 3
    """
    seg = seg.astype(np.uint8)
    out = np.zeros_like(seg, dtype=np.uint8)

    out[seg == 0] = 0
    out[seg == 1] = 2   # edema → ED
    out[seg == 2] = 1   # non-enhancing tumor → NCR/NET
    out[seg == 3] = 3   # enhancing tumour → ET

    return out


# ---------------- PROCESS ONE CASE ----------------
def process_labeled_case(img_path: Path, seg_path: Path):
    """
    Xử lý một case có nhãn:
        - Tách 4 modality
        - Crop cố định (22:216, 16:210, :)
        - Chuẩn hóa cường độ cho từng modality riêng
        - Crop seg theo cùng vùng, remap nhãn Task01 → BraTS
        - Lưu theo format Brain_XXX trong OUT_LABELED
    """
    # Lấy id "XXX" từ "BRATS_XXX.nii.gz"
    base = img_path.name.replace(".nii.gz", "").replace(".nii", "")
    case_id = base.split("_")[-1]  # "001"
    out_name = f"Brain_{case_id.zfill(3)}"
    out_dir = OUT_LABELED / out_name
    ensure_dir(out_dir)

    # ---- Load image ----
    data_4d, img_ras = load_ras(img_path)
    mods = split_modalities_4d(data_4d)

    # ---- Crop & normalize mỗi modality ----
    for mod_key, out_fname in [
        ("flair", "flair.nii.gz"),
        ("t1",    "t1.nii.gz"),
        ("t1ce",  "t1ce.nii.gz"),
        ("t2",    "t2.nii.gz"),
    ]:
        vol = mods[mod_key]          # (X, Y, Z)
        vol = crop_xy(vol)
        vol = normalize_intensity(vol)
        save_nifti(vol, img_ras, out_dir / out_fname)

    # ---- Load & crop seg ----
    seg_data, seg_img = load_ras(seg_path)  # seg là 3D (X, Y, Z)
    if seg_data.ndim != 3:
        raise ValueError(f"Unexpected seg dim for {seg_path}: {seg_data.shape}")

    seg_crop = crop_xy(seg_data)

    # Remap Task01 labels → BraTS2020 labels
    seg_remap = map_seg_task01_to_brats(seg_crop)

    save_nifti(seg_remap.astype(np.uint8), seg_img, out_dir / "mask.nii.gz")


def process_unlabeled_case(img_path: Path):
    """
    Xử lý một case không nhãn (imagesTs):
        - Tách 4 modality
        - Crop cố định
        - Chuẩn hóa
        - Lưu vào OUT_UNLABELED/Brain_XXX
    """
    base = img_path.name.replace(".nii.gz", "").replace(".nii", "")
    case_id = base.split("_")[-1]
    out_name = f"Brain_{case_id.zfill(3)}"
    out_dir = OUT_UNLABELED / out_name
    ensure_dir(out_dir)

    data_4d, img_ras = load_ras(img_path)
    mods = split_modalities_4d(data_4d)

    for mod_key, out_fname in [
        ("flair", "flair.nii.gz"),
        ("t1",    "t1.nii.gz"),
        ("t1ce",  "t1ce.nii.gz"),
        ("t2",    "t2.nii.gz"),
    ]:
        vol = mods[mod_key]
        vol = crop_xy(vol)
        vol = normalize_intensity(vol)
        save_nifti(vol, img_ras, out_dir / out_fname)


# ---------------- MAIN ----------------
def main():
    print(f"[INFO] NORM_MODE   = {NORM_MODE}")
    if NORM_MODE.lower() == "zscore":
        print(f"[INFO] ZSCORE_CLIP = {ZSCORE_CLIP}")
    print(f"[INFO] CROP        = {CROP}")

    ensure_dir(OUT_LABELED)
    ensure_dir(OUT_UNLABELED)

    # ---- Liệt kê training images & labels ----
    train_imgs = list_nifti_files(IMAGES_TR)
    train_labels = list_nifti_files(LABELS_TR)

    # Map từ tên case → path label để dễ lookup
    label_dict = {}
    for lp in train_labels:
        base = lp.name.replace(".nii.gz", "").replace(".nii", "")
        label_dict[base] = lp

    print(f"[INFO] Found {len(train_imgs)} training images, {len(train_labels)} training labels.")

    # -------- Process LABELED (imagesTr + labelsTr) --------
    for img_path in tqdm(train_imgs, desc="Processing labeled (imagesTr)"):
        base = img_path.name.replace(".nii.gz", "").replace(".nii", "")
        if base not in label_dict:
            print(f"[WARN] No label found for {base}, skip.")
            continue

        seg_path = label_dict[base]
        try:
            process_labeled_case(img_path, seg_path)
        except Exception as e:
            print(f"[WARN] Skip labeled case {base}: {e}")

    # -------- Process UNLABELED (imagesTs) --------
    if IMAGES_TS.exists():
        test_imgs = list_nifti_files(IMAGES_TS)
        print(f"[INFO] Found {len(test_imgs)} test/unlabeled images.")

        for img_path in tqdm(test_imgs, desc="Processing unlabeled (imagesTs)"):
            base = img_path.name.replace(".nii.gz", "").replace(".nii", "")
            try:
                process_unlabeled_case(img_path)
            except Exception as e:
                print(f"[WARN] Skip unlabeled case {base}: {e}")
    else:
        print("[INFO] No imagesTs folder found; skip unlabeled preprocessing.")

    print("[OK] Done preprocessing Task01_BrainTumour 3D.")


if __name__ == "__main__":
    main()
