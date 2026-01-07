# -*- coding: utf-8 -*-
"""
data/dataloader_2d.py

BraTS2020 2D PNG dataloader (4 modalities, 4 classes 0..3) with TRAIN oversampling.

OPTIMIZATION:
- Scan filesystem + read masks ONLY ONCE in __init__ to build:
    - all_items
    - tumor_items (mask > 0)
- Oversampled train list (self.items) is (re)sampled from those cached lists
  without rescanning/re-reading masks each epoch.

RESIZE:
- Always resize X and Y to (image_size, image_size) at __getitem__.
- Image resize: bilinear
- Mask resize : nearest

Sampling:
- TRAIN: oversampling tumor slices by tumor_ratio (default 0.7 tumor / 0.3 random)
- VAL/TEST: use all slices (no oversampling)

Returns:
- x: torch.FloatTensor [4,H,W] in [0,1]
- y: torch.LongTensor  [H,W] values 0..3
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple
import random

import numpy as np
import imageio.v3 as iio
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Resize backend
import cv2


# ============================================================
# Default paths (edit if needed)
# ============================================================
DEFAULT_ROOT_DIR = Path(r"D:\Project Advanced CV\data\processed\2d")
DEFAULT_SPLITS_DIR = Path(r"configs/splits_2d")

DEFAULT_TRAIN_LIST = DEFAULT_SPLITS_DIR / "train.txt"
DEFAULT_VAL_LIST = DEFAULT_SPLITS_DIR / "val.txt"
DEFAULT_TEST_LIST = DEFAULT_SPLITS_DIR / "test.txt"


# ============================================================
# Utilities
# ============================================================

def read_png_u8(path: Path) -> np.ndarray:
    """
    Read PNG as uint8 (H,W). If multi-channel -> take channel 0.
    """
    arr = iio.imread(path)
    if arr.ndim != 2:
        arr = arr[..., 0]
    return arr.astype(np.uint8, copy=False)


def parse_slice_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def read_split_list(txt_path: Path) -> List[str]:
    if not txt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file split: {txt_path}")
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def ensure_cases_exist(labeled_root: Path, case_names: List[str], verbose: bool = True) -> List[Path]:
    case_dirs: List[Path] = []
    missing: List[str] = []

    for name in case_names:
        p = labeled_root / name
        if p.exists() and p.is_dir():
            case_dirs.append(p)
        else:
            missing.append(name)

    if verbose and missing:
        print(f"[Cảnh báo] Có {len(missing)} case trong split không tồn tại trong {labeled_root}:")
        for m in missing[:20]:
            print("  -", m)
        if len(missing) > 20:
            print("  ...")

    return case_dirs


def resize_image_u8(img_u8: np.ndarray, out_size: int) -> np.ndarray:
    """
    img_u8: uint8 (H,W)
    return uint8 (out_size,out_size) using bilinear
    """
    if img_u8.shape[0] == out_size and img_u8.shape[1] == out_size:
        return img_u8
    return cv2.resize(img_u8, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def resize_mask_i64(mask_i64: np.ndarray, out_size: int) -> np.ndarray:
    """
    mask_i64: int64 (H,W) with labels 0..3
    return int64 (out_size,out_size) using nearest
    """
    if mask_i64.shape[0] == out_size and mask_i64.shape[1] == out_size:
        return mask_i64
    # cv2 works best with smaller int types -> convert to uint8 safely then back
    m = mask_i64.astype(np.uint8, copy=False)
    m = cv2.resize(m, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return m.astype(np.int64, copy=False)


@dataclass(frozen=True)
class SliceItem:
    brain_dir: Path
    k: int


# ============================================================
# Dataset
# ============================================================

class Brats2DPNGDataset(Dataset):
    """
    BraTS 2D PNG dataset with cached index.

    - Build cached all_items + tumor_items once by scanning masks.
    - Train: resample() creates oversampled self.items from cached lists (fast).
    - Val/Test: self.items = all_items

    image_size:
    - Resize X/Y at __getitem__ to (image_size,image_size)
    """

    def __init__(
        self,
        root_dir: Path,
        split_txt: Path,
        split_name: str,
        image_size: int = 224,
        tumor_ratio: float = 0.7,
        seed: int = 1337,
        transform: Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
        verbose: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.split_txt = Path(split_txt)
        self.split_name = split_name.lower().strip()
        self.image_size = int(image_size)
        self.tumor_ratio = float(tumor_ratio)
        self.transform = transform
        self.verbose = bool(verbose)

        if self.split_name not in {"train", "val", "test"}:
            raise ValueError(f"split_name phải là train/val/test, nhưng nhận: {split_name}")

        self.labeled_root = self.root_dir / "labeled"
        if not self.labeled_root.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục labeled: {self.labeled_root}")

        self.base_seed = int(seed)
        self.rng = random.Random(self.base_seed)
        self._epoch = 0

        # cases
        case_names = read_split_list(self.split_txt)
        self.case_dirs = ensure_cases_exist(self.labeled_root, case_names, verbose=self.verbose)
        if len(self.case_dirs) == 0:
            raise RuntimeError(f"Split '{self.split_name}' không có case hợp lệ. Kiểm tra: {self.split_txt}")

        # Build cached index ONCE
        self.all_items: List[SliceItem] = []
        self.tumor_items: List[SliceItem] = []
        self._build_cached_index_once()

        if len(self.all_items) == 0:
            raise RuntimeError(f"Split '{self.split_name}' không tìm thấy lát cắt nào. Kiểm tra dữ liệu/mask.")

        # Initial items
        if self.split_name in {"val", "test"}:
            self.items = self.all_items
        else:
            self.items = []
            self.resample()

    def _build_cached_index_once(self):
        """
        Scan mask_*.png once to fill all_items and tumor_items.
        """
        case_iter = tqdm(self.case_dirs, desc=f"Build cached index ({self.split_name})", leave=False) \
            if self.verbose else self.case_dirs

        for brain_dir in case_iter:
            mask_dir = brain_dir / "mask"
            if not mask_dir.exists():
                continue

            mask_paths = sorted(mask_dir.glob("mask_*.png"))
            if len(mask_paths) == 0:
                continue

            for mp in mask_paths:
                k = parse_slice_index(mp)
                item = SliceItem(brain_dir=brain_dir, k=k)
                self.all_items.append(item)

                mask = read_png_u8(mp)
                if (mask > 0).any():
                    self.tumor_items.append(item)

        if self.verbose:
            print(f"[Index] {self.split_name}: all={len(self.all_items)}, tumor={len(self.tumor_items)}")

    def resample(self):
        """
        Fast resample train list from cached all_items + tumor_items. No disk scan.
        """
        if self.split_name != "train":
            return

        n_total = len(self.all_items)
        if n_total == 0:
            self.items = []
            return

        if len(self.tumor_items) == 0:
            self.items = list(self.all_items)
            self.rng.shuffle(self.items)
            self._epoch += 1
            return

        n_tumor = int(round(n_total * self.tumor_ratio))
        n_random = n_total - n_tumor

        tumor_sampled = [self.rng.choice(self.tumor_items) for _ in range(n_tumor)]
        random_sampled = [self.rng.choice(self.all_items) for _ in range(n_random)]

        items = tumor_sampled + random_sampled
        self.rng.shuffle(items)
        self.items = items
        self._epoch += 1

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.items[idx]
        bdir, k = item.brain_dir, item.k
        S = self.image_size

        # Read modalities (uint8), resize to SxS, then normalize
        flair = resize_image_u8(read_png_u8(bdir / "flair" / f"flair_{k:03d}.png"), S)
        t1    = resize_image_u8(read_png_u8(bdir / "t1"    / f"t1_{k:03d}.png"), S)
        t1ce  = resize_image_u8(read_png_u8(bdir / "t1ce"  / f"t1ce_{k:03d}.png"), S)
        t2    = resize_image_u8(read_png_u8(bdir / "t2"    / f"t2_{k:03d}.png"), S)

        x = np.stack([flair, t1, t1ce, t2], axis=0).astype(np.float32) / 255.0  # [4,S,S]

        # Read mask (int64), resize with nearest
        y0 = read_png_u8(bdir / "mask" / f"mask_{k:03d}.png").astype(np.int64, copy=False)
        y = resize_mask_i64(y0, S)  # [S,S]

        if self.transform is not None:
            x, y = self.transform(x, y)

        return torch.from_numpy(x), torch.from_numpy(y)


# ============================================================
# Transforms
# ============================================================

class RandomHFlip:
    def __init__(self, p: float = 0.5, seed: int = 0):
        self.p = float(p)
        self.rng = random.Random(int(seed))

    def __call__(self, x: np.ndarray, y: np.ndarray):
        if self.rng.random() < self.p:
            x = x[..., ::-1].copy()
            y = y[..., ::-1].copy()
        return x, y


class Compose:
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = list(transforms)

    def __call__(self, x: np.ndarray, y: np.ndarray):
        for t in self.transforms:
            x, y = t(x, y)
        return x, y


# ============================================================
# Factory
# ============================================================

def make_loaders(
    root_dir: Path = DEFAULT_ROOT_DIR,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
    image_size: int = 224,
    batch_size: int = 8,
    val_batch_size: Optional[int] = None,
    num_workers: int = 2,
    tumor_ratio: float = 0.7,
    seed: int = 1337,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_txt = Path(splits_dir) / "train.txt"
    val_txt = Path(splits_dir) / "val.txt"
    test_txt = Path(splits_dir) / "test.txt"

    train_tf = Compose([RandomHFlip(p=0.5, seed=seed)])

    train_ds = Brats2DPNGDataset(
        root_dir=Path(root_dir),
        split_txt=train_txt,
        split_name="train",
        image_size=int(image_size),
        tumor_ratio=float(tumor_ratio),
        seed=int(seed),
        transform=train_tf,
        verbose=True,
    )
    val_ds = Brats2DPNGDataset(
        root_dir=Path(root_dir),
        split_txt=val_txt,
        split_name="val",
        image_size=int(image_size),
        tumor_ratio=float(tumor_ratio),
        seed=int(seed),
        transform=None,
        verbose=True,
    )
    test_ds = Brats2DPNGDataset(
        root_dir=Path(root_dir),
        split_txt=test_txt,
        split_name="test",
        image_size=int(image_size),
        tumor_ratio=float(tumor_ratio),
        seed=int(seed),
        transform=None,
        verbose=True,
    )

    if val_batch_size is None:
        val_batch_size = batch_size

    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(val_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(val_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    tr, va, te = make_loaders(image_size=224, batch_size=8, num_workers=2)
    x, y = next(iter(tr))
    print("Batch x:", tuple(x.shape), x.dtype, "y:", tuple(y.shape), y.dtype, "labels:", torch.unique(y).tolist())
    ds = tr.dataset
    if hasattr(ds, "resample"):
        ds.resample()
        ds.resample()
        print("Resampled train items:", len(ds))
