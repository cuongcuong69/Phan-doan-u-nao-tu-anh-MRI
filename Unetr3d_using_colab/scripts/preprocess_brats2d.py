# -*- coding: utf-8 -*-
"""
BraTS2020 -> 2D slices (PNG) for 2D segmentation / semi-supervised.
Version 3: percentile normalization [0,1] + orientation upright (optional).

- RAS reorientation (nibabel.as_closest_canonical)
- Percentile normalization [0,1] (nonzero voxels)
- Background forced to 0
- Segmentation labels remapped 4→3
- Orientation control via ROTATE_K, FLIP_LR, FLIP_UD
"""

import argparse
from pathlib import Path
import re
import numpy as np
import nibabel as nib
from tqdm import tqdm
import imageio.v3 as iio


# ---------- orientation control ----------
ROTATE_K = 1      # 90° CCW
FLIP_LR  = False  # left–right
FLIP_UD  = False  # up–down

def orient_upright_2d(arr2d: np.ndarray) -> np.ndarray:
    """Xoay/flip slice để não dựng dọc"""
    if ROTATE_K % 4 != 0:
        arr2d = np.rot90(arr2d, k=ROTATE_K)
    if FLIP_LR:
        arr2d = np.fliplr(arr2d)
    if FLIP_UD:
        arr2d = np.flipud(arr2d)
    return arr2d


# ---------- utilities ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_nii_ras(fp: Path) -> np.ndarray:
    nii = nib.load(str(fp))
    nii = nib.as_closest_canonical(nii)
    return nii.get_fdata(dtype=np.float32)

def normalize_to_01(vol: np.ndarray, pmin=1, pmax=99, eps=1e-8) -> np.ndarray:
    nz = vol[vol > 0]
    if nz.size >= 10:
        lo, hi = np.percentile(nz, [pmin, pmax])
    else:
        lo, hi = np.percentile(vol, [pmin, pmax])
    if hi <= lo: hi = lo + 1
    x = np.clip(vol, lo, hi)
    x = (x - lo) / (hi - lo + eps)
    x[vol == 0] = 0
    return x

def remap_mask_labels(mask_vol: np.ndarray) -> np.ndarray:
    m = mask_vol.astype(np.uint8)
    m[m == 4] = 3
    return m

def write_slice(arr01: np.ndarray, out_path: Path):
    iio.imwrite(out_path, np.round(arr01 * 255).astype(np.uint8), plugin="pillow")

def find_case_id(name: str) -> str:
    m = re.search(r"_(\d+)$", name)
    return m.group(1) if m else name


# ---------- main processing ----------
def process_case(case_dir: Path, out_dir: Path, labeled: bool, pmin: float, pmax: float):
    name = case_dir.name
    cid = find_case_id(name)
    flair = load_nii_ras(case_dir / f"{name}_flair.nii")
    t1    = load_nii_ras(case_dir / f"{name}_t1.nii")
    t1ce  = load_nii_ras(case_dir / f"{name}_t1ce.nii")
    t2    = load_nii_ras(case_dir / f"{name}_t2.nii")

    flair01 = normalize_to_01(flair, pmin, pmax)
    t101    = normalize_to_01(t1, pmin, pmax)
    t1ce01  = normalize_to_01(t1ce, pmin, pmax)
    t201    = normalize_to_01(t2, pmin, pmax)

    seg_u8 = None
    if labeled:
        seg = load_nii_ras(case_dir / f"{name}_seg.nii")
        seg_u8 = remap_mask_labels(seg)

    brain_dir = out_dir / f"Brain_{cid}"
    for k in range(flair01.shape[2]):
        for mod, arr in zip(["flair", "t1", "t1ce", "t2"], [flair01, t101, t1ce01, t201]):
            out_p = brain_dir / mod / f"{mod}_{k:03d}.png"
            ensure_dir(out_p.parent)
            write_slice(orient_upright_2d(arr[:, :, k]), out_p)
        if labeled and seg_u8 is not None:
            out_p = brain_dir / "mask" / f"mask_{k:03d}.png"
            ensure_dir(out_p.parent)
            iio.imwrite(out_p, orient_upright_2d(seg_u8[:, :, k]), plugin="pillow")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root",  type=str, default=r"D:\Project Advanced CV\data\BraST2020")
    ap.add_argument("--train_sub", type=str, default=r"BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData")
    ap.add_argument("--val_sub",   type=str, default=r"BraTS2020_ValidationData\MICCAI_BraTS2020_ValidationData")
    ap.add_argument("--out_root",  type=str, default=r"D:\Project Advanced CV\data\processed\2d")
    ap.add_argument("--pmin", type=float, default=1)
    ap.add_argument("--pmax", type=float, default=99)
    ap.add_argument("--rotate_k", type=int, default=1, help="Số lần xoay 90° CCW (mặc định 1: dựng dọc)")
    ap.add_argument("--flip_lr", action="store_true", help="Lật trái-phải")
    ap.add_argument("--flip_ud", action="store_true", help="Lật trên-dưới")
    args = ap.parse_args()

    global ROTATE_K, FLIP_LR, FLIP_UD
    ROTATE_K, FLIP_LR, FLIP_UD = args.rotate_k, args.flip_lr, args.flip_ud

    train_root = Path(args.raw_root) / args.train_sub
    val_root   = Path(args.raw_root) / args.val_sub
    out_root   = Path(args.out_root)
    labeled_out, unlabeled_out = out_root / "labeled", out_root / "unlabeled"
    ensure_dir(labeled_out); ensure_dir(unlabeled_out)

    for case in tqdm(sorted([p for p in train_root.iterdir() if p.is_dir()]), desc="Labeled"):
        process_case(case, labeled_out, True, args.pmin, args.pmax)
    for case in tqdm(sorted([p for p in val_root.iterdir() if p.is_dir()]), desc="Unlabeled"):
        process_case(case, unlabeled_out, False, args.pmin, args.pmax)

    print("Done.")


if __name__ == "__main__":
    main()
