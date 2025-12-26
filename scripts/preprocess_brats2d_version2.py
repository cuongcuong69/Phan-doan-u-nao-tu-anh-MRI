# -*- coding: utf-8 -*-
"""
BraTS2020 -> 2D slices (PNG) for 2D segmentation / semi-supervised.

Pipeline:
1) QUÉT TOÀN BỘ T1 (train + val) sau khi reorient RAS, nhị phân vol>0 -> tìm BBOX 2D (x,y)
   lớn nhất bao phủ não của mọi ca; ép BBOX thành hình vuông & nằm trong ảnh.
2) Tiền xử lý từng ca:
   - Reorient RAS (không xoay/flip thêm trước khi crop).
   - Chuẩn hoá percentile về [0,1] trên vùng >0, ép nền=0.
   - Cắt lát axial (axis=2) -> crop theo BBOX vuông -> (tuỳ chọn) dựng não thẳng với xoay/flip.
   - Resize ảnh về 256x256 (INTER_LINEAR), mask về 256x256 (INTER_NEAREST).
   - Lưu PNG uint8 (ảnh: [0,1]*255, mask: 0..3).
   - Map nhãn 4->3.

Output:
  processed/2d/labeled/Brain_XXX/{flair,t1,t1ce,t2,mask}/xxx_000.png
  processed/2d/unlabeled/Brain_XXX/{flair,t1,t1ce,t2}/xxx_000.png
"""

import argparse
from pathlib import Path
import re
import json
import numpy as np
import nibabel as nib
from tqdm import tqdm
import imageio.v3 as iio
import cv2


# ---------- orientation control (áp dụng SAU khi crop) ----------
ROTATE_K = 1      # số lần xoay 90° CCW để “dựng dọc”; 1 nghĩa là 90° CCW
FLIP_LR  = False  # lật trái-phải
FLIP_UD  = False  # lật trên-dưới

def orient_upright_2d(arr2d: np.ndarray) -> np.ndarray:
    """Xoay/flip slice để não dựng dọc (áp dụng sau crop)."""
    if ROTATE_K % 4 != 0:
        arr2d = np.rot90(arr2d, k=ROTATE_K)
    if FLIP_LR:
        arr2d = np.fliplr(arr2d)
    if FLIP_UD:
        arr2d = np.flipud(arr2d)
    return arr2d


# ---------- utils ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_nii_ras(fp: Path) -> np.ndarray:
    nii = nib.load(str(fp))
    nii = nib.as_closest_canonical(nii)  # đưa về RAS
    return nii.get_fdata(dtype=np.float32)

def normalize_to_01(vol: np.ndarray, pmin=1, pmax=99, eps=1e-8) -> np.ndarray:
    """Percentile normalize trên vùng >0, về [0,1], nền = 0."""
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

def remap_mask_labels(mask_vol: np.ndarray) -> np.ndarray:
    m = mask_vol.astype(np.uint8, copy=True)
    m[m == 4] = 3
    return m

def to_uint8_from01(arr01: np.ndarray) -> np.ndarray:
    return np.round(np.clip(arr01, 0.0, 1.0) * 255.0).astype(np.uint8)

def write_png_uint8(arr_u8: np.ndarray, path: Path):
    iio.imwrite(path, arr_u8, plugin="pillow")

def find_case_id(name: str) -> str:
    m = re.search(r"_(\d+)$", name)
    return m.group(1) if m else name


# ---------- global bounding box over all T1 ----------
def compute_global_square_bbox_xy(train_root: Path, val_root: Path, cache_file: Path | None = None):
    """
    Quét tất cả T1 (train + val) -> RAS -> mask = vol>0 -> lấy min/max theo (x,y) ở bất kỳ z,
    gộp toàn bộ ca -> tạo bbox 2D (x_min,x_max,y_min,y_max), sau đó ép thành hình vuông
    và ràng buộc trong kích thước ảnh.
    """
    if cache_file and cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            j = json.load(f)
        return (j["x_min"], j["x_max"], j["y_min"], j["y_max"])

    x_min_global, y_min_global = +1e9, +1e9
    x_max_global, y_max_global = -1e9, -1e9
    H_ref, W_ref = None, None

    def iter_case_dirs(root: Path, prefix: str):
        return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)])

    # Train (labeled)
    for case_dir in tqdm(iter_case_dirs(train_root, "BraTS20_Training_"), desc="Scan T1 (labeled)"):
        name = case_dir.name
        t1 = load_nii_ras(case_dir / f"{name}_t1.nii")
        if H_ref is None:
            H_ref, W_ref = t1.shape[0], t1.shape[1]
        mask = t1 > 0
        if not mask.any():
            continue
        coords = np.argwhere(mask)  # (x,y,z) với x:row, y:col
        xs, ys = coords[:, 0], coords[:, 1]
        x_min_global = min(x_min_global, int(xs.min()))
        x_max_global = max(x_max_global, int(xs.max()))
        y_min_global = min(y_min_global, int(ys.min()))
        y_max_global = max(y_max_global, int(ys.max()))

    # Val (unlabeled)
    for case_dir in tqdm(iter_case_dirs(val_root, "BraTS20_Validation_"), desc="Scan T1 (unlabeled)"):
        name = case_dir.name
        t1 = load_nii_ras(case_dir / f"{name}_t1.nii")
        if H_ref is None:
            H_ref, W_ref = t1.shape[0], t1.shape[1]
        mask = t1 > 0
        if not mask.any():
            continue
        coords = np.argwhere(mask)
        xs, ys = coords[:, 0], coords[:, 1]
        x_min_global = min(x_min_global, int(xs.min()))
        x_max_global = max(x_max_global, int(xs.max()))
        y_min_global = min(y_min_global, int(ys.min()))
        y_max_global = max(y_max_global, int(ys.max()))

    # Ràng buộc vào [0, H-1] × [0, W-1]
    x_min_global = max(0, x_min_global)
    y_min_global = max(0, y_min_global)
    x_max_global = min(H_ref - 1, x_max_global)
    y_max_global = min(W_ref - 1, y_max_global)

    # Ép hình vuông (theo (x,y); không cắt theo z)
    h = x_max_global - x_min_global + 1
    w = y_max_global - y_min_global + 1
    side = max(h, w)

    cx = (x_min_global + x_max_global) // 2
    cy = (y_min_global + y_max_global) // 2
    half = side // 2

    x_min_sq = max(0, cx - half)
    x_max_sq = x_min_sq + side - 1
    if x_max_sq >= H_ref:
        x_max_sq = H_ref - 1
        x_min_sq = x_max_sq - side + 1

    y_min_sq = max(0, cy - half)
    y_max_sq = y_min_sq + side - 1
    if y_max_sq >= W_ref:
        y_max_sq = W_ref - 1
        y_min_sq = y_max_sq - side + 1

    bbox = (int(x_min_sq), int(x_max_sq), int(y_min_sq), int(y_max_sq))

    if cache_file:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"x_min": bbox[0], "x_max": bbox[1], "y_min": bbox[2], "y_max": bbox[3]}, f, indent=2)

    print(f"[Global BBox square] x:[{bbox[0]}, {bbox[1]}] y:[{bbox[2]}, {bbox[3]}] side={side} on {H_ref}x{W_ref}")
    return bbox


# ---------- per-case processing with crop+resize ----------
def process_case(case_dir: Path, out_dir: Path, labeled: bool,
                 pmin: float, pmax: float,
                 bbox_xy: tuple[int, int, int, int],
                 target_size: int = 256):
    """
    bbox_xy = (x_min, x_max, y_min, y_max) trong toạ độ RAS 2D (x=row, y=col).
    Crop từng lát rồi mới dựng dọc & resize.
    """
    name = case_dir.name
    cid = find_case_id(name)

    # Load volumes (RAS)
    flair = load_nii_ras(case_dir / f"{name}_flair.nii")
    t1    = load_nii_ras(case_dir / f"{name}_t1.nii")
    t1ce  = load_nii_ras(case_dir / f"{name}_t1ce.nii")
    t2    = load_nii_ras(case_dir / f"{name}_t2.nii")

    # Normalize to [0,1] + background 0
    flair01 = normalize_to_01(flair, pmin, pmax)
    t101    = normalize_to_01(t1, pmin, pmax)
    t1ce01  = normalize_to_01(t1ce, pmin, pmax)
    t201    = normalize_to_01(t2, pmin, pmax)

    seg_u8 = None
    if labeled:
        seg = load_nii_ras(case_dir / f"{name}_seg.nii")
        seg_u8 = remap_mask_labels(seg)

    x_min, x_max, y_min, y_max = bbox_xy
    brain_dir = out_dir / f"Brain_{cid}"

    D = flair01.shape[2]
    for k in range(D):
        # crop theo bbox vuông (trên ảnh đã RAS, trước khi dựng dọc)
        f_slc = flair01[x_min:x_max+1, y_min:y_max+1, k]
        t1_slc = t101[x_min:x_max+1, y_min:y_max+1, k]
        t1ce_slc = t1ce01[x_min:x_max+1, y_min:y_max+1, k]
        t2_slc = t201[x_min:x_max+1, y_min:y_max+1, k]

        # dựng dọc (xoay/flip) *sau* crop để toạ độ bbox không lệch
        f_slc = orient_upright_2d(f_slc)
        t1_slc = orient_upright_2d(t1_slc)
        t1ce_slc = orient_upright_2d(t1ce_slc)
        t2_slc = orient_upright_2d(t2_slc)

        # resize về 256x256
        f_res = cv2.resize(f_slc, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        t1_res = cv2.resize(t1_slc, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        t1ce_res = cv2.resize(t1ce_slc, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        t2_res = cv2.resize(t2_slc, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        # ghi PNG uint8
        write_png_uint8(to_uint8_from01(f_res), brain_dir / "flair" / f"flair_{k:03d}.png")
        write_png_uint8(to_uint8_from01(t1_res), brain_dir / "t1" / f"t1_{k:03d}.png")
        write_png_uint8(to_uint8_from01(t1ce_res), brain_dir / "t1ce" / f"t1ce_{k:03d}.png")
        write_png_uint8(to_uint8_from01(t2_res), brain_dir / "t2" / f"t2_{k:03d}.png")

        if labeled and seg_u8 is not None:
            m_slc = seg_u8[x_min:x_max+1, y_min:y_max+1, k]
            m_slc = orient_upright_2d(m_slc)
            m_res = cv2.resize(m_slc, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
            write_png_uint8(m_res.astype(np.uint8), brain_dir / "mask" / f"mask_{k:03d}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root",  type=str, default=r"D:\Project Advanced CV\data\BraST2020")
    ap.add_argument("--train_sub", type=str, default=r"BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData")
    ap.add_argument("--val_sub",   type=str, default=r"BraTS2020_ValidationData\MICCAI_BraTS2020_ValidationData")
    ap.add_argument("--out_root",  type=str, default=r"D:\Project Advanced CV\data\processed\2d")
    ap.add_argument("--pmin", type=float, default=1.0)
    ap.add_argument("--pmax", type=float, default=99.0)
    ap.add_argument("--rotate_k", type=int, default=1, help="xoay 90° CCW k lần sau crop (mặc định 1)")
    ap.add_argument("--flip_lr", action="store_true", help="lật trái-phải sau crop")
    ap.add_argument("--flip_ud", action="store_true", help="lật trên-dưới sau crop")
    ap.add_argument("--target_size", type=int, default=256, help="kích thước đầu ra vuông (mặc định 256)")
    ap.add_argument("--bbox_cache", type=str, default=r"D:\Project Advanced CV\data\processed\2d\global_bbox.json",
                    help="nếu tồn tại sẽ dùng, nếu chưa sẽ tính và lưu")
    args = ap.parse_args()

    global ROTATE_K, FLIP_LR, FLIP_UD
    ROTATE_K, FLIP_LR, FLIP_UD = args.rotate_k, args.flip_lr, args.flip_ud

    # Paths
    raw_root   = Path(args.raw_root)
    train_root = raw_root / args.train_sub
    val_root   = raw_root / args.val_sub
    out_root   = Path(args.out_root)
    labeled_out   = out_root / "labeled"
    unlabeled_out = out_root / "unlabeled"
    ensure_dir(labeled_out); ensure_dir(unlabeled_out)

    # 1) Tính (hoặc đọc cache) bbox vuông toàn cục theo (x,y) trong không gian RAS
    cache_path = Path(args.bbox_cache) if args.bbox_cache else None
    bbox_xy = compute_global_square_bbox_xy(train_root, val_root, cache_path)

    # 2) Xử lý từng ca với bbox đã xác định
    train_cases = sorted([p for p in train_root.iterdir() if p.is_dir() and p.name.startswith("BraTS20_Training_")])
    for case in tqdm(train_cases, desc="Labeled"):
        process_case(case, labeled_out, True, args.pmin, args.pmax, bbox_xy, args.target_size)

    val_cases = sorted([p for p in val_root.iterdir() if p.is_dir() and p.name.startswith("BraTS20_Validation_")])
    for case in tqdm(val_cases, desc="Unlabeled"):
        process_case(case, unlabeled_out, False, args.pmin, args.pmax, bbox_xy, args.target_size)

    print("Done.")


if __name__ == "__main__":
    main()