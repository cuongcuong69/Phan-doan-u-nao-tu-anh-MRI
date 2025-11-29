# -*- coding: utf-8 -*-
"""
Preprocess Task01_BrainTumour -> 2D slices (PNG) giống hệt pipeline BraTS2020 2D.

✔ Input: 4D NIfTI (4 modality stacked)
    - 0: FLAIR  → flair
    - 1: T1w    → t1
    - 2: t1gd   → t1ce
    - 3: T2w    → t2

✔ Mask 3D (labelsTr), map nhãn Task01 → BraTS:
    Task01:   0: background
              1: edema
              2: non-enhancing tumor
              3: enhancing tumour

    BraTS:    0: background
              1: NCR/NET
              2: ED
              3: ET

    Mapping:
        0 → 0
        1 → 2
        2 → 1
        3 → 3

✔ Bước xử lý:
    - Load NIfTI → RAS
    - Unstack 4D → 4 volume 3D
    - Percentile normalize [1,99] trên non-zero → [0,1], nền =0
    - Crop cố định:
        x = 22:216
        y = 16:210
      (z giữ nguyên)
    - Slice axial (axis=2)
    - Xoay/flip (ROTATE_K, FLIP_LR, FLIP_UD)
    - Resize 256×256 (INTER_LINEAR cho ảnh, INTER_NEAREST cho mask)
    - Ghi PNG uint8

✔ Output:
    processed_task01/2d/labeled/Brain_XXX/{flair,t1,t1ce,t2,mask}/*.png
    processed_task01/2d/unlabeled/Brain_XXX/{flair,t1,t1ce,t2}/*.png
"""

from pathlib import Path
import argparse
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm
import imageio.v3 as iio


# ---------------- orientation control ----------------
ROTATE_K = 1      # xoay 90° CCW k lần sau crop
FLIP_LR  = False  # lật trái-phải
FLIP_UD  = False  # lật trên-dưới


def orient_upright_2d(arr: np.ndarray) -> np.ndarray:
    """Xoay/flip slice cho thẳng đứng (áp dụng SAU crop)."""
    if ROTATE_K % 4 != 0:
        arr = np.rot90(arr, k=ROTATE_K)
    if FLIP_LR:
        arr = np.fliplr(arr)
    if FLIP_UD:
        arr = np.flipud(arr)
    return arr


# ---------------- utils ----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_ras(fp: Path) -> np.ndarray:
    """Load NIfTI và convert về orientation RAS."""
    nii = nib.load(str(fp))
    nii_ras = nib.as_closest_canonical(nii)
    return nii_ras.get_fdata(dtype=np.float32)


def percentile_norm(vol: np.ndarray, pmin=1, pmax=99, eps=1e-8) -> np.ndarray:
    """Normalize theo percentile [pmin,pmax] trên vùng non-zero, output [0,1], nền=0."""
    vol = vol.astype(np.float32)
    nz = vol[vol > 0]

    if nz.size >= 10:
        lo, hi = np.percentile(nz, [pmin, pmax])
    else:
        lo, hi = np.percentile(vol, [pmin, pmax])

    if hi <= lo:
        hi = lo + 1.0

    x = np.clip(vol, lo, hi)
    x = (x - lo) / (hi - lo + eps)
    x[vol == 0] = 0.0
    return x


def to_uint8(arr01: np.ndarray) -> np.ndarray:
    return np.clip(arr01 * 255.0, 0, 255).astype(np.uint8)


def write_png(arr_u8: np.ndarray, path: Path):
    iio.imwrite(path, arr_u8, plugin="pillow")


# ---------------- Label remap Task01 → BraTS ----------------
def remap_task01_to_brats(mask: np.ndarray) -> np.ndarray:
    """
    Task01 → BraTS2020 mapping:
        0 → 0
        1 (edema) → 2 (ED)
        2 (non-enhancing tumour) → 1 (NCR/NET)
        3 (enhancing tumour) → 3 (ET)
    """
    m = mask.astype(np.uint8)
    out = np.zeros_like(m, dtype=np.uint8)
    out[m == 0] = 0
    out[m == 1] = 2
    out[m == 2] = 1
    out[m == 3] = 3
    return out


# ---------------- CROP FIXED (same as BraTS) ----------------
CROP = {
    "x_min": 22,
    "x_max": 216,
    "y_min": 16,
    "y_max": 210
}


def crop_xy(vol: np.ndarray) -> np.ndarray:
    """
    Crop volume 3D theo vùng cố định (như BraTS).
    vol shape: (X, Y, Z)
    """
    x0, x1 = CROP["x_min"], CROP["x_max"]
    y0, y1 = CROP["y_min"], CROP["y_max"]
    X, Y, Z = vol.shape

    if x1 > X or y1 > Y:
        raise ValueError(
            f"CROP range ({x0}:{x1}, {y0}:{y1}) vượt quá kích thước volume ({X}, {Y}, {Z})."
        )

    return vol[x0:x1, y0:y1, :]


# ---------------- Unstack 4-channel 4D image ----------------
def split_modalities_4d(vol4d: np.ndarray):
    """
    vol4d shape có thể là:
        (4, X, Y, Z) hoặc (X, Y, Z, 4)
    Trả về dict 4 modality 3D.
    """
    if vol4d.ndim != 4:
        raise ValueError(f"Expected 4D, got {vol4d.shape}")

    if vol4d.shape[0] == 4:
        return {
            "flair": vol4d[0, ...],
            "t1":    vol4d[1, ...],
            "t1ce":  vol4d[2, ...],
            "t2":    vol4d[3, ...],
        }
    elif vol4d.shape[-1] == 4:
        return {
            "flair": vol4d[..., 0],
            "t1":    vol4d[..., 1],
            "t1ce":  vol4d[..., 2],
            "t2":    vol4d[..., 3],
        }
    else:
        raise ValueError(f"Cannot determine channel axis for shape {vol4d.shape}")


# ---------------- LIST NIFTI FILES (SKIP ._*) ----------------
def list_nifti_files(folder: Path):
    """
    Liệt kê các file NIfTI trong folder, bỏ qua file rác '._xxx.nii.gz'.
    Chấp nhận đuôi .nii, .nii.gz.
    """
    out = []
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        if p.name.startswith("._"):
            continue
        if p.suffix in [".nii", ".gz"]:  # .nii.gz -> suffix là ".gz"
            out.append(p)
    return out


# ---------------- PROCESS ONE CASE ----------------
def process_case(img_path: Path, mask_path: Path | None, out_root: Path,
                 pmin: float, pmax: float, target_size: int = 256):
    """
    - img_path: file 4D NIfTI (4 modality)
    - mask_path: nhãn 3D (cùng shape), hoặc None nếu unlabeled
    """
    # Lấy ID từ tên: BRATS_001.nii.gz -> "001"
    name = img_path.name
    base = name.replace(".nii.gz", "").replace(".nii", "")
    cid = base.split("_")[-1]  # "001", "002", ...

    brain_dir = out_root / f"Brain_{cid.zfill(3)}"
    # Với unlabeled, mask_path=None => thư mục mask có thể rỗng, không sao
    for sub in ["flair", "t1", "t1ce", "t2", "mask"]:
        ensure_dir(brain_dir / sub)

    # --- Load image 4D + unstack ---
    vol4d = load_ras(img_path)
    mods = split_modalities_4d(vol4d)

    # --- Normalize từng modality ---
    for k in mods:
        mods[k] = percentile_norm(mods[k], pmin, pmax)

    # --- Load & remap mask nếu có ---
    seg = None
    if mask_path is not None and mask_path.exists():
        seg = load_ras(mask_path).astype(np.uint8)
        seg = remap_task01_to_brats(seg)

    # --- Crop cố định ---
    for k in mods:
        mods[k] = crop_xy(mods[k])
    if seg is not None:
        seg = crop_xy(seg)

    D = mods["flair"].shape[2]

    for z in range(D):
        # Lấy lát
        f_slc  = mods["flair"][:, :, z]
        t1_slc = mods["t1"][:, :, z]
        c_slc  = mods["t1ce"][:, :, z]
        t2_slc = mods["t2"][:, :, z]
        m_slc  = seg[:, :, z] if seg is not None else None

        # Xoay/flip dựng dọc
        f_slc  = orient_upright_2d(f_slc)
        t1_slc = orient_upright_2d(t1_slc)
        c_slc  = orient_upright_2d(c_slc)
        t2_slc = orient_upright_2d(t2_slc)
        if m_slc is not None:
            m_slc = orient_upright_2d(m_slc)

        # Resize
        f_res  = cv2.resize(f_slc, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        t1_res = cv2.resize(t1_slc, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        c_res  = cv2.resize(c_slc, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        t2_res = cv2.resize(t2_slc, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        # Save image PNG
        write_png(to_uint8(f_res),  brain_dir / "flair" / f"flair_{z:03d}.png")
        write_png(to_uint8(t1_res), brain_dir / "t1"    / f"t1_{z:03d}.png")
        write_png(to_uint8(c_res),  brain_dir / "t1ce"  / f"t1ce_{z:03d}.png")
        write_png(to_uint8(t2_res), brain_dir / "t2"    / f"t2_{z:03d}.png")

        # Save mask PNG nếu có
        if m_slc is not None:
            m_res = cv2.resize(m_slc, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
            write_png(m_res.astype(np.uint8), brain_dir / "mask" / f"mask_{z:03d}.png")


# ---------------- MAIN ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task_root", type=str,
        default=r"D:\Project Advanced CV\data\Task01_BrainTumour\Task01_BrainTumour"
    )
    ap.add_argument(
        "--out_root", type=str,
        default=r"D:\Project Advanced CV\data\processed_task01\2d"
    )
    ap.add_argument("--pmin", type=float, default=1.0)
    ap.add_argument("--pmax", type=float, default=99.0)
    ap.add_argument("--target_size", type=int, default=256)
    ap.add_argument("--rotate_k", type=int, default=1)
    ap.add_argument("--flip_lr", action="store_true")
    ap.add_argument("--flip_ud", action="store_true")
    args = ap.parse_args()

    # Set orientation params
    global ROTATE_K, FLIP_LR, FLIP_UD
    ROTATE_K, FLIP_LR, FLIP_UD = args.rotate_k, args.flip_lr, args.flip_ud

    # Paths
    root = Path(args.task_root)
    images_tr = root / "imagesTr"
    labels_tr = root / "labelsTr"
    images_ts = root / "imagesTs"

    out_root = Path(args.out_root)
    labeled_root   = out_root / "labeled"
    unlabeled_root = out_root / "unlabeled"
    ensure_dir(labeled_root)
    ensure_dir(unlabeled_root)

    # -------- Labeled (imagesTr + labelsTr) --------
    print("Processing labeled Task01 cases...")
    train_imgs = list_nifti_files(images_tr)

    for img_path in tqdm(train_imgs, desc="Labeled"):
        # Tên mask: BRATS_xxx.nii.gz
        name = img_path.name
        base = name.replace(".nii.gz", "").replace(".nii", "")
        cid  = base.split("_")[-1]  # "001"
        seg_path = labels_tr / f"BRATS_{cid}.nii.gz"

        if not seg_path.exists():
            # thử fallback .nii
            seg_path_alt = labels_tr / f"BRATS_{cid}.nii"
            if seg_path_alt.exists():
                seg_path = seg_path_alt
            else:
                print(f"[WARN] Missing seg for {img_path.name}, skip label.")
                seg_path = None

        process_case(img_path, seg_path, labeled_root,
                     args.pmin, args.pmax, args.target_size)

    # -------- Unlabeled (imagesTs) --------
    if images_ts.exists():
        print("Processing unlabeled Task01 cases...")
        test_imgs = list_nifti_files(images_ts)
        for img_path in tqdm(test_imgs, desc="Unlabeled"):
            process_case(img_path, None, unlabeled_root,
                         args.pmin, args.pmax, args.target_size)
    else:
        print("[INFO] No imagesTs folder; skip unlabeled.")

    print("Done.")


if __name__ == "__main__":
    main()
