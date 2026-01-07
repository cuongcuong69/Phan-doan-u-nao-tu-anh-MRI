# -*- coding: utf-8 -*-
"""
data/dataloader_brats2d.py

Dataloader 2D PNG cho BraTS2020 (supervised, slice-based 2D) dùng Albumentations.

Cấu trúc:
data/processed/2d/labeled/Brain_xxx/
    flair/flair_000.png ...
    t1/t1_000.png ...
    t1ce/t1ce_000.png ...
    t2/t2_000.png ...
    mask/mask_000.png ...  (label 0..3)

Split:
configs/splits_2d/train.txt
configs/splits_2d/val.txt
configs/splits_2d/test.txt

Resize:
256x256 -> 224x224 (KHÔNG CROP)
- image: bilinear
- mask : nearest

Augmentation (TRAIN only, Albumentations):
- HorizontalFlip / VerticalFlip
- RandomRotate90
- (optional) Rotate(limit=15)  # default OFF

Self-test:
python -m data.dataloader_brats2d
"""

from __future__ import annotations
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import albumentations as A


# =============================================================================
# SELF-TEST CONFIG
# =============================================================================
SELFTEST_SEED = 2025
SELFTEST_SIZE = 256
SELFTEST_BATCH = 8
SELFTEST_NUM_WORKERS = 0
# =============================================================================

MODALITIES = ["flair", "t1", "t1ce", "t2"]


# ================== Helpers ==================

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _abs_from_root(rel: str) -> str:
    return str((_project_root() / rel).resolve())


def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _read_lines(txt_path: str) -> List[str]:
    items: List[str] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                items.append(ln)
    return items


def _load_png_float01(path: Union[str, Path]) -> np.ndarray:
    """
    Load grayscale PNG -> float32 [0,1]
    Assumption: processed đã minmax và thường lưu uint8 0..255.
    """
    img = Image.open(str(path)).convert("L")
    arr = np.array(img)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _load_mask_int(path: Union[str, Path]) -> np.ndarray:
    """
    Load mask PNG -> int64 0..3
    """
    m = Image.open(str(path)).convert("L")
    return np.array(m).astype(np.int64)


# ================== Albumentations ==================

def build_train_transform(
    image_size: int = 224,
    p_hflip: float = 0.5,
    p_vflip: float = 0.5,
    p_rot90: float = 0.5,
    use_small_rotate: bool = False,   # bật nếu bạn muốn rotate góc nhỏ
    rotate_limit: int = 15,
    p_rotate: float = 0.3,
) -> A.Compose:
    tfms: List[Any] = []

    # Resize trước (mask nearest, image bilinear)
    tfms.append(A.Resize(image_size, image_size, interpolation=1, mask_interpolation=0))

    # Augment
    tfms.append(A.HorizontalFlip(p=p_hflip))
    tfms.append(A.VerticalFlip(p=p_vflip))
    tfms.append(A.RandomRotate90(p=p_rot90))

    # OPTIONAL: rotate góc nhỏ (mask vẫn nearest)
    if use_small_rotate:
        tfms.append(
            A.Rotate(
                limit=rotate_limit,
                interpolation=1,
                mask_interpolation=0,
                border_mode=0,      # constant
                value=0,
                mask_value=0,
                p=p_rotate,
            )
        )

    return A.Compose(tfms)


def build_valtest_transform(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=1, mask_interpolation=0)
    ])


# ================== Dataset 2D ==================

class Brats2DSupervisedAlbumentations(Dataset):
    """
    Dataset 2D slice-based, Albumentations.
    Trả về:
        image: torch.float32 [4,H,W]
        label: torch.int64   [H,W]
    """
    def __init__(
        self,
        split_txt: str,
        root_2d: str = "data/processed/2d/labeled",
        image_size: int = 224,
        mode: str = "train",
        use_augment: bool = True,
        use_small_rotate: bool = False,
    ):
        self.mode = mode
        self.image_size = int(image_size)
        self.root_2d = Path(_abs_from_root(root_2d))

        self.use_augment = bool(use_augment) and (mode == "train")
        if self.use_augment:
            self.transform = build_train_transform(
                image_size=self.image_size,
                use_small_rotate=use_small_rotate,
            )
        else:
            self.transform = build_valtest_transform(image_size=self.image_size)

        self.cases: List[str] = _read_lines(split_txt)

        # flatten list (case, slice_idx)
        self.items: List[Tuple[str, int]] = []
        for case_id in self.cases:
            case_dir = self.root_2d / case_id
            flair_dir = case_dir / "flair"
            if not flair_dir.exists():
                raise FileNotFoundError(f"Missing: {flair_dir}")

            flair_files = sorted(flair_dir.glob("flair_*.png"))
            if len(flair_files) == 0:
                raise FileNotFoundError(f"No slices in: {flair_dir}")
            n = len(flair_files)

            # check aligned slice count
            for mod in ["t1", "t1ce", "t2", "mask"]:
                mod_dir = case_dir / mod
                files = sorted(mod_dir.glob(f"{mod}_*.png"))
                if len(files) != n:
                    raise ValueError(f"[{case_id}] {mod} has {len(files)} slices, flair has {n}.")

            for sidx in range(n):
                self.items.append((case_id, sidx))

        print(
            f"[Brats2DSupervisedAlbumentations] mode={mode} | cases={len(self.cases)} | items={len(self.items)} | "
            f"root_2d={self.root_2d} | image_size={self.image_size} | augment={self.use_augment}"
        )

    def __len__(self) -> int:
        return len(self.items)

    def _slice_path(self, case_dir: Path, modality: str, slice_idx: int) -> Path:
        return case_dir / modality / f"{modality}_{slice_idx:03d}.png"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case_id, sidx = self.items[idx]
        case_dir = self.root_2d / case_id

        # Load 4 modalities as float01
        chans = []
        for mod in MODALITIES:
            fp = self._slice_path(case_dir, mod, sidx)
            chans.append(_load_png_float01(fp))
        img = np.stack(chans, axis=-1).astype(np.float32)  # (H,W,4) for Albumentations

        # Load mask int64 (H,W)
        mfp = self._slice_path(case_dir, "mask", sidx)
        mask = _load_mask_int(mfp)

        # Apply transform (resize + aug)
        out = self.transform(image=img, mask=mask)
        img_t = out["image"]   # (H,W,4)
        msk_t = out["mask"]    # (H,W)

        # To torch: (C,H,W)
        x = torch.from_numpy(np.transpose(img_t, (2, 0, 1)).copy()).float()
        y = torch.from_numpy(msk_t.copy()).long()

        return {"image": x, "label": y, "case": case_id, "slice": sidx}


# ================== Builders ==================

def build_brats2d_sup_train_loader(
    image_size: int = 224,
    batch_size: int = 8,
    num_workers: int = 2,
    seed: int = 2025,
    use_small_rotate: bool = False,
    pin_memory: bool = True,                 # <-- added
    drop_last: Optional[bool] = None,        # <-- added
    shuffle: Optional[bool] = None,          # <-- added
) -> DataLoader:
    set_seed(seed)
    root = _project_root()
    split_txt = str(root / "configs" / "splits_2d" / "train.txt")

    ds = Brats2DSupervisedAlbumentations(
        split_txt=split_txt,
        root_2d="data/processed/2d/labeled",
        image_size=image_size,
        mode="train",
        use_augment=True,
        use_small_rotate=use_small_rotate,
    )

    if drop_last is None:
        drop_last = True
    if shuffle is None:
        shuffle = True

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=bool(shuffle),
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
        drop_last=bool(drop_last),
    )


def build_brats2d_sup_val_loader(
    image_size: int = 224,
    batch_size: int = 8,
    num_workers: int = 2,
    seed: Optional[int] = None,
    pin_memory: bool = True,                 # <-- added
    drop_last: Optional[bool] = None,        # <-- added
    shuffle: Optional[bool] = None,          # <-- added
) -> DataLoader:
    if seed is not None:
        set_seed(seed)
    root = _project_root()
    split_txt = str(root / "configs" / "splits_2d" / "val.txt")

    ds = Brats2DSupervisedAlbumentations(
        split_txt=split_txt,
        root_2d="data/processed/2d/labeled",
        image_size=image_size,
        mode="val",
        use_augment=False,
    )

    if drop_last is None:
        drop_last = False
    if shuffle is None:
        shuffle = False

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=bool(shuffle),
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
        drop_last=bool(drop_last),
    )


def build_brats2d_sup_test_loader(
    image_size: int = 224,
    batch_size: int = 8,
    num_workers: int = 2,
    seed: Optional[int] = None,
    pin_memory: bool = True,                 # <-- added
    drop_last: Optional[bool] = None,        # <-- added
    shuffle: Optional[bool] = None,          # <-- added
) -> DataLoader:
    if seed is not None:
        set_seed(seed)
    root = _project_root()
    split_txt = str(root / "configs" / "splits_2d" / "test.txt")

    ds = Brats2DSupervisedAlbumentations(
        split_txt=split_txt,
        root_2d="data/processed/2d/labeled",
        image_size=image_size,
        mode="test",
        use_augment=False,
    )

    if drop_last is None:
        drop_last = False
    if shuffle is None:
        shuffle = False

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=bool(shuffle),
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
        drop_last=bool(drop_last),
    )


# ================== Self-test ==================
if __name__ == "__main__":
    set_seed(SELFTEST_SEED)

    tr = build_brats2d_sup_train_loader(
        image_size=SELFTEST_SIZE,
        batch_size=SELFTEST_BATCH,
        num_workers=SELFTEST_NUM_WORKERS,
        seed=SELFTEST_SEED,
        use_small_rotate=False,
        pin_memory=False,
    )
    b = next(iter(tr))
    x, y = b["image"], b["label"]
    print(f"[TRAIN] x={tuple(x.shape)} (N,C,H,W) | y={tuple(y.shape)} (N,H,W)")
    print(f"        x min={x.min().item():.4f}, max={x.max().item():.4f}")
    print(f"        y unique (first batch)={torch.unique(y).tolist()[:10]}")
    print(f"        cases={list(b['case'])[:4]} ...")