# -*- coding: utf-8 -*-
"""
data/dataloader_brats3d_task01_sup.py

Dataloader 3D NIfTI cho phân đoạn khối u não (supervised, 3D patch-based),
trong đó:

- TRAIN loader = gộp:
    1) BraTS2020 đã preprocess:        data/processed/3d/labeled/<case>/
    2) Task01 (đã preprocess 3D):     data/processed_task01/3d/labeled/<case>/
  (Task01 lấy TOÀN BỘ case có đủ file, không phụ thuộc train.txt)

- VAL/TEST loader: GIỮ NGUYÊN như cũ, chỉ dùng BraTS2020 theo split:
    configs/splits_2d/val.txt
    configs/splits_2d/test.txt

Mỗi case folder yêu cầu các file:
    flair.nii.gz, t1.nii.gz, t1ce.nii.gz, t2.nii.gz, mask.nii.gz
Mask: 0..3 (đã map 4->3 nếu có)

Output item:
    {
      "image": FloatTensor [4,D,H,W],
      "label": LongTensor  [1,D,H,W],
      "case":  str (vd: "brats:Brain_001" hoặc "task01:Task01_XXX"),
    }

Self-test:
    python -m data.dataloader_brats3d_task01_sup
In ra số lượng case train/val/test và breakdown train (brats vs task01).
"""

from __future__ import annotations
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any, Union

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom as nd_zoom


# =============================================================================
# NORMALIZATION CONFIG (phải khớp với preprocess)
# =============================================================================
NORM_MODE = "zscore"   # hoặc "minmax"
ZSCORE_CLIP = (-5.0, 5.0)

# =============================================================================
# SELF-TEST CONFIG
# =============================================================================
SELFTEST_SEED = 2025
SELFTEST_PATCH = (128, 128, 128)   # (D,H,W)
SELFTEST_BATCH = 2
SELFTEST_NUM_WORKERS = 0
# =============================================================================


# ================== Helpers chung ==================

def _project_root() -> Path:
    """Thư mục gốc project (chứa folders: data/, configs/, trainers/, ...)."""
    return Path(__file__).resolve().parents[1]


def _abs_from_root(rel: str) -> str:
    return str((_project_root() / rel).resolve())


def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_nii_to_DHW(path: str) -> np.ndarray:
    """
    Đọc file NIfTI và chuyển shape (X,Y,Z) -> (D=Z, H=Y, W=X).
    """
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)  # (X,Y,Z)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={data.shape} at {path}")
    data = np.transpose(data, (2, 1, 0))  # (Z,Y,X) = (D,H,W)
    return data


def ensure_min_size_3d(vol: np.ndarray, out_size: Tuple[int, int, int]) -> np.ndarray:
    """
    Đảm bảo vol có size >= out_size trên ba trục (D,H,W).
    vol: [C,D,H,W] hoặc [D,H,W]
    """
    if vol.ndim == 4:
        _, D, H, W = vol.shape
        has_channel = True
    elif vol.ndim == 3:
        D, H, W = vol.shape
        has_channel = False
    else:
        raise ValueError(f"Volume must be 3D or 4D, got shape={vol.shape}")

    d, h, w = out_size
    pad_D = max(0, d - D)
    pad_H = max(0, h - H)
    pad_W = max(0, w - W)

    if pad_D == pad_H == pad_W == 0:
        return vol

    pad_before_D = pad_D // 2
    pad_after_D = pad_D - pad_before_D
    pad_before_H = pad_H // 2
    pad_after_H = pad_H - pad_before_H
    pad_before_W = pad_W // 2
    pad_after_W = pad_W - pad_before_W

    if has_channel:
        pads = (
            (0, 0),
            (pad_before_D, pad_after_D),
            (pad_before_H, pad_after_H),
            (pad_before_W, pad_after_W),
        )
    else:
        pads = (
            (pad_before_D, pad_after_D),
            (pad_before_H, pad_after_H),
            (pad_before_W, pad_after_W),
        )
    vol = np.pad(vol, pads, mode="constant", constant_values=0)
    return vol


def _random_crop_coords(D, H, W, d, h, w):
    z = np.random.randint(0, max(1, D - d + 1))
    y = np.random.randint(0, max(1, H - h + 1))
    x = np.random.randint(0, max(1, W - w + 1))
    return z, y, x


def _centered_crop_coords(zc, yc, xc, D, H, W, d, h, w):
    zs = int(np.clip(zc - d // 2, 0, max(0, D - d)))
    ys = int(np.clip(yc - h // 2, 0, max(0, H - h)))
    xs = int(np.clip(xc - w // 2, 0, max(0, W - w)))
    return zs, ys, xs


def _case_has_all_files(case_dir: Path) -> bool:
    req = ["flair.nii.gz", "t1.nii.gz", "t1ce.nii.gz", "t2.nii.gz", "mask.nii.gz"]
    return all((case_dir / r).is_file() for r in req)


# ================== Augmentations 3D ==================

class Random3DAugment(object):
    """
    Augmentation 3D:
    - Random flip theo trục D/H/W
    - Intensity jitter (scale+shift) từng channel
    - Gaussian noise
    - (tuỳ chọn) Random gamma (chỉ dùng khi NORM_MODE='minmax')
    - Random zoom 3D (đã fix lỗi shape bằng _resize_to_shape)

    Behavior clipping phụ thuộc NORM_MODE:
        - "minmax": clip về [0,1]
        - "zscore": clip về ZSCORE_CLIP (ví dụ [-5,5])
    """
    def __init__(
        self,
        p_flip: float = 0.5,
        p_jitter: float = 0.5,
        p_noise: float = 0.5,
        p_gamma: float = 0.0,
        p_zoom: float = 0.3,
        jitter_scale_range: Tuple[float, float] = (0.9, 1.1),
        jitter_shift_range: Tuple[float, float] = (-0.1, 0.1),
        noise_std: float = 0.02,
        gamma_range: Tuple[float, float] = (0.7, 1.5),
        zoom_range: Tuple[float, float] = (0.9, 1.1),
        norm_mode: str = NORM_MODE,
        clip_range: Optional[Tuple[float, float]] = None,
    ):
        self.p_flip = p_flip
        self.p_jitter = p_jitter
        self.p_noise = p_noise
        self.p_gamma = p_gamma
        self.p_zoom = p_zoom

        self.jitter_scale_range = jitter_scale_range
        self.jitter_shift_range = jitter_shift_range
        self.noise_std = noise_std
        self.gamma_range = gamma_range
        self.zoom_range = zoom_range

        self.norm_mode = norm_mode.lower().strip()
        if clip_range is not None:
            self.clip_min, self.clip_max = clip_range
        else:
            if self.norm_mode == "minmax":
                self.clip_min, self.clip_max = 0.0, 1.0
            elif self.norm_mode == "zscore":
                self.clip_min, self.clip_max = ZSCORE_CLIP
            else:
                self.clip_min, self.clip_max = None, None

    def _maybe_clip(self, img: np.ndarray) -> np.ndarray:
        if self.clip_min is not None and self.clip_max is not None:
            img = np.clip(img, self.clip_min, self.clip_max)
        return img

    def _resize_to_shape(self, vol: np.ndarray, out_shape: Tuple[int, int, int]) -> np.ndarray:
        d, h, w = out_shape
        D_in, H_in, W_in = vol.shape
        scale = (d / float(D_in), h / float(H_in), w / float(W_in))
        vol_resized = nd_zoom(vol, zoom=scale, order=1)

        D2, H2, W2 = vol_resized.shape
        pad_D = max(0, d - D2)
        pad_H = max(0, h - H2)
        pad_W = max(0, w - W2)
        if pad_D > 0 or pad_H > 0 or pad_W > 0:
            vol_resized = np.pad(
                vol_resized,
                (
                    (pad_D // 2, pad_D - pad_D // 2),
                    (pad_H // 2, pad_H - pad_H // 2),
                    (pad_W // 2, pad_W - pad_W // 2),
                ),
                mode="constant",
                constant_values=0,
            )
            D2, H2, W2 = vol_resized.shape

        if D2 > d or H2 > h or W2 > w:
            zs = max(0, (D2 - d) // 2)
            ys = max(0, (H2 - h) // 2)
            xs = max(0, (W2 - w) // 2)
            vol_resized = vol_resized[zs:zs + d, ys:ys + h, xs:xs + w]

        assert vol_resized.shape == (d, h, w), f"resize ra {vol_resized.shape}, cần {(d, h, w)}"
        return vol_resized

    def _zoom(self, image: np.ndarray, mask: np.ndarray, zoom_factor: float):
        C, D, H, W = image.shape
        z_img = np.zeros_like(image)
        for c in range(C):
            z_vol = nd_zoom(image[c], zoom=zoom_factor, order=1)
            z_img[c] = self._resize_to_shape(z_vol, (D, H, W))

        z_msk = nd_zoom(mask, zoom=zoom_factor, order=0)
        z_msk = self._resize_to_shape(z_msk, (D, H, W)).astype(mask.dtype)
        return z_img, z_msk

    def _gamma_transform(self, img: np.ndarray, gamma: float) -> np.ndarray:
        if self.norm_mode != "minmax":
            return img
        img = np.clip(img, 0.0, 1.0)
        return img ** gamma

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        C, D, H, W = image.shape

        if np.random.rand() < self.p_flip:
            for axis in [1, 2, 3]:  # D,H,W
                if np.random.rand() < 0.5:
                    image = np.flip(image, axis=axis)
                    mask = np.flip(mask, axis=axis - 1)

        if np.random.rand() < self.p_jitter:
            for c in range(C):
                scale = np.random.uniform(*self.jitter_scale_range)
                shift = np.random.uniform(*self.jitter_shift_range)
                image[c] = self._maybe_clip(image[c] * scale + shift)

        if np.random.rand() < self.p_noise:
            noise = np.random.normal(0.0, self.noise_std, size=image.shape).astype(np.float32)
            image = self._maybe_clip(image + noise)

        if self.p_gamma > 0 and np.random.rand() < self.p_gamma and self.norm_mode == "minmax":
            gamma = np.random.uniform(*self.gamma_range)
            for c in range(C):
                image[c] = self._gamma_transform(image[c], gamma)

        if self.p_zoom > 0 and np.random.rand() < self.p_zoom:
            zf = np.random.uniform(*self.zoom_range)
            image, mask = self._zoom(image, mask, zf)

        return image, mask


# ================== Dataset Supervised 3D (multi-root for TRAIN) ==================

class Brats3DTask01Supervised(Dataset):
    """
    Dataset GIÁM SÁT 3D cho TRAIN: gộp BraTS + Task01 (multi-root).

    - brats_split_txt: danh sách case BraTS (vd Brain_XXX)
    - brats_root_3d: data/processed/3d/labeled
    - task01_root_3d: data/processed_task01/3d/labeled
    - include_task01: nếu True sẽ quét toàn bộ case trong task01_root_3d
                      (case hợp lệ = folder con có đủ 5 file nii.gz)
    """
    def __init__(
        self,
        brats_split_txt: str,
        brats_root_3d: str = "data/processed/3d/labeled",
        task01_root_3d: str = "data/processed_task01/3d/labeled",
        include_task01: bool = True,
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        mode: str = "train",
        sampling_mode: str = "mixed",
        rejection_thresh: float = 0.01,
        rejection_max: int = 8,
        mixed_weights: Optional[Dict[str, float]] = None,
        use_augment: bool = True,
        norm_mode: str = NORM_MODE,
    ):
        self.mode = mode
        self.patch_size = tuple(map(int, patch_size))
        self.norm_mode = norm_mode.lower().strip()

        self.brats_root_3d = Path(_abs_from_root(brats_root_3d))
        self.task01_root_3d = Path(_abs_from_root(task01_root_3d))
        self.include_task01 = bool(include_task01)

        # sampling
        self.sampling_mode = sampling_mode
        self.rejection_thresh = float(rejection_thresh)
        self.rejection_max = int(rejection_max)

        if mixed_weights is None:
            mixed_weights = {"center_fg": 0.6, "random": 0.4}
        valid_keys = {"random", "rejection", "center_fg"}
        mixed_weights = {k: v for k, v in mixed_weights.items() if k in valid_keys and v > 0}
        s = sum(mixed_weights.values()) or 1.0
        self.mixed_weights = {k: v / s for k, v in mixed_weights.items()}

        # augmentation
        self.use_augment = use_augment and (mode == "train")
        self.augment = Random3DAugment(norm_mode=self.norm_mode)

        # ---- load BraTS cases from split ----
        brats_cases: List[str] = []
        with open(brats_split_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    brats_cases.append(line)

        # ---- scan Task01 cases (ALL) ----
        task01_cases: List[str] = []
        if self.include_task01 and self.task01_root_3d.is_dir():
            for p in sorted(self.task01_root_3d.iterdir()):
                if p.is_dir() and _case_has_all_files(p):
                    task01_cases.append(p.name)

        # ---- merge, avoid duplicates by (source, name) ----
        self.samples: List[Tuple[str, str]] = []
        for c in brats_cases:
            self.samples.append(("brats", c))
        for c in task01_cases:
            self.samples.append(("task01", c))

        self._n_brats = len(brats_cases)
        self._n_task01 = len(task01_cases)

        print(
            f"[Brats3DTask01Supervised] mode={mode} | total={len(self.samples)} "
            f"(brats={self._n_brats}, task01={self._n_task01}) | "
            f"brats_root={self.brats_root_3d} | task01_root={self.task01_root_3d} | norm_mode={self.norm_mode}"
        )

    @property
    def num_brats(self) -> int:
        return self._n_brats

    @property
    def num_task01(self) -> int:
        return self._n_task01

    def __len__(self) -> int:
        return len(self.samples)

    # ---- sampling helpers ----
    def _choose_mixed_mode(self) -> str:
        r = random.random()
        acc = 0.0
        for k, p in self.mixed_weights.items():
            acc += p
            if r <= acc:
                return k
        return list(self.mixed_weights.keys())[-1]

    def _crop_random(self, img4d: np.ndarray, msk3d: np.ndarray):
        _, D, H, W = img4d.shape
        d, h, w = self.patch_size
        z, y, x = _random_crop_coords(D, H, W, d, h, w)
        return (
            img4d[:, z:z + d, y:y + h, x:x + w],
            msk3d[z:z + d, y:y + h, x:x + w],
        )

    def _crop_rejection(self, img4d: np.ndarray, msk3d: np.ndarray):
        _, D, H, W = img4d.shape
        d, h, w = self.patch_size
        for _ in range(self.rejection_max):
            z, y, x = _random_crop_coords(D, H, W, d, h, w)
            sub = msk3d[z:z + d, y:y + h, x:x + w]
            if (sub > 0).mean() >= self.rejection_thresh:
                return img4d[:, z:z + d, y:y + h, x:x + w], sub
        return self._crop_random(img4d, msk3d)

    def _crop_center_fg(self, img4d: np.ndarray, msk3d: np.ndarray):
        pts = np.argwhere(msk3d > 0)
        if len(pts) == 0:
            return self._crop_random(img4d, msk3d)
        _, D, H, W = img4d.shape
        d, h, w = self.patch_size
        zc, yc, xc = pts[np.random.randint(len(pts))]
        zs, ys, xs = _centered_crop_coords(int(zc), int(yc), int(xc), D, H, W, d, h, w)
        return (
            img4d[:, zs:zs + d, ys:ys + h, xs:xs + w],
            msk3d[zs:zs + d, ys:ys + h, xs:xs + w],
        )

    def _sample_patch(self, img4d: np.ndarray, msk3d: np.ndarray):
        mode = self.sampling_mode
        if mode == "mixed":
            mode = self._choose_mixed_mode()
        if mode == "rejection":
            return self._crop_rejection(img4d, msk3d)
        if mode == "center_fg":
            return self._crop_center_fg(img4d, msk3d)
        return self._crop_random(img4d, msk3d)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        source, case_id = self.samples[idx]
        case_dir = (self.brats_root_3d / case_id) if source == "brats" else (self.task01_root_3d / case_id)

        flair_fp = case_dir / "flair.nii.gz"
        t1_fp    = case_dir / "t1.nii.gz"
        t1ce_fp  = case_dir / "t1ce.nii.gz"
        t2_fp    = case_dir / "t2.nii.gz"
        mask_fp  = case_dir / "mask.nii.gz"

        if not _case_has_all_files(case_dir):
            raise FileNotFoundError(f"Missing files in {case_dir}")

        flair = load_nii_to_DHW(str(flair_fp))
        t1    = load_nii_to_DHW(str(t1_fp))
        t1ce  = load_nii_to_DHW(str(t1ce_fp))
        t2    = load_nii_to_DHW(str(t2_fp))
        msk   = load_nii_to_DHW(str(mask_fp)).astype(np.int16)

        img4d = np.stack([flair, t1, t1ce, t2], axis=0)

        img4d = ensure_min_size_3d(img4d, self.patch_size)
        msk   = ensure_min_size_3d(msk,   self.patch_size)

        img_patch, msk_patch = self._sample_patch(img4d, msk)

        if self.use_augment:
            img_patch, msk_patch = self.augment(img_patch, msk_patch)

        img_t = torch.from_numpy(img_patch.astype(np.float32))
        lbl_t = torch.from_numpy(msk_patch.astype(np.int64)).unsqueeze(0)

        return {
            "image": img_t,
            "label": lbl_t,
            "case": f"{source}:{case_id}",
        }


# ================== Dataset Supervised 3D (single-root for VAL/TEST) ==================

class Brats3DSupervisedSingleRoot(Dataset):
    """
    Dataset GIÁM SÁT 3D cho VAL/TEST: giữ nguyên như cũ (chỉ BraTS, theo split).
    """
    def __init__(
        self,
        split_txt: str,
        root_3d: str = "data/processed/3d/labeled",
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        mode: str = "val",
        sampling_mode: str = "mixed",
        rejection_thresh: float = 0.01,
        rejection_max: int = 8,
        mixed_weights: Optional[Dict[str, float]] = None,
        use_augment: bool = False,
        norm_mode: str = NORM_MODE,
    ):
        self.mode = mode
        self.patch_size = tuple(map(int, patch_size))
        self.root_3d = Path(_abs_from_root(root_3d))
        self.norm_mode = norm_mode.lower().strip()

        self.sampling_mode = sampling_mode
        self.rejection_thresh = float(rejection_thresh)
        self.rejection_max = int(rejection_max)

        if mixed_weights is None:
            mixed_weights = {"center_fg": 0.6, "random": 0.4}
        valid_keys = {"random", "rejection", "center_fg"}
        mixed_weights = {k: v for k, v in mixed_weights.items() if k in valid_keys and v > 0}
        s = sum(mixed_weights.values()) or 1.0
        self.mixed_weights = {k: v / s for k, v in mixed_weights.items()}

        self.use_augment = use_augment and (mode == "train")
        self.augment = Random3DAugment(norm_mode=self.norm_mode)

        cases: List[str] = []
        with open(split_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(line)
        self.cases = cases

        print(f"[Brats3DSupervisedSingleRoot] mode={mode} | total={len(self.cases)} | root_3d={self.root_3d}")

    def __len__(self) -> int:
        return len(self.cases)

    def _choose_mixed_mode(self) -> str:
        r = random.random()
        acc = 0.0
        for k, p in self.mixed_weights.items():
            acc += p
            if r <= acc:
                return k
        return list(self.mixed_weights.keys())[-1]

    def _crop_random(self, img4d: np.ndarray, msk3d: np.ndarray):
        _, D, H, W = img4d.shape
        d, h, w = self.patch_size
        z, y, x = _random_crop_coords(D, H, W, d, h, w)
        return (
            img4d[:, z:z + d, y:y + h, x:x + w],
            msk3d[z:z + d, y:y + h, x:x + w],
        )

    def _crop_rejection(self, img4d: np.ndarray, msk3d: np.ndarray):
        _, D, H, W = img4d.shape
        d, h, w = self.patch_size
        for _ in range(self.rejection_max):
            z, y, x = _random_crop_coords(D, H, W, d, h, w)
            sub = msk3d[z:z + d, y:y + h, x:x + w]
            if (sub > 0).mean() >= self.rejection_thresh:
                return img4d[:, z:z + d, y:y + h, x:x + w], sub
        return self._crop_random(img4d, msk3d)

    def _crop_center_fg(self, img4d: np.ndarray, msk3d: np.ndarray):
        pts = np.argwhere(msk3d > 0)
        if len(pts) == 0:
            return self._crop_random(img4d, msk3d)
        _, D, H, W = img4d.shape
        d, h, w = self.patch_size
        zc, yc, xc = pts[np.random.randint(len(pts))]
        zs, ys, xs = _centered_crop_coords(int(zc), int(yc), int(xc), D, H, W, d, h, w)
        return (
            img4d[:, zs:zs + d, ys:ys + h, xs:xs + w],
            msk3d[zs:zs + d, ys:ys + h, xs:xs + w],
        )

    def _sample_patch(self, img4d: np.ndarray, msk3d: np.ndarray):
        mode = self.sampling_mode
        if mode == "mixed":
            mode = self._choose_mixed_mode()
        if mode == "rejection":
            return self._crop_rejection(img4d, msk3d)
        if mode == "center_fg":
            return self._crop_center_fg(img4d, msk3d)
        return self._crop_random(img4d, msk3d)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case_id = self.cases[idx]
        case_dir = self.root_3d / case_id

        flair_fp = case_dir / "flair.nii.gz"
        t1_fp    = case_dir / "t1.nii.gz"
        t1ce_fp  = case_dir / "t1ce.nii.gz"
        t2_fp    = case_dir / "t2.nii.gz"
        mask_fp  = case_dir / "mask.nii.gz"

        if not _case_has_all_files(case_dir):
            raise FileNotFoundError(f"Missing files in {case_dir}")

        flair = load_nii_to_DHW(str(flair_fp))
        t1    = load_nii_to_DHW(str(t1_fp))
        t1ce  = load_nii_to_DHW(str(t1ce_fp))
        t2    = load_nii_to_DHW(str(t2_fp))
        msk   = load_nii_to_DHW(str(mask_fp)).astype(np.int16)

        img4d = np.stack([flair, t1, t1ce, t2], axis=0)

        img4d = ensure_min_size_3d(img4d, self.patch_size)
        msk   = ensure_min_size_3d(msk,   self.patch_size)

        img_patch, msk_patch = self._sample_patch(img4d, msk)

        if self.use_augment:
            img_patch, msk_patch = self.augment(img_patch, msk_patch)

        img_t = torch.from_numpy(img_patch.astype(np.float32))
        lbl_t = torch.from_numpy(msk_patch.astype(np.int64)).unsqueeze(0)

        return {
            "image": img_t,
            "label": lbl_t,
            "case": f"brats:{case_id}",
        }


# ================== Builders ==================

def build_brats3d_task01_sup_train_loader(
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    batch_size: int = 1,
    num_workers: int = 2,
    seed: int = 2025,
    sampling_mode: str = "mixed",
    rejection_thresh: float = 0.01,
    rejection_max: int = 8,
    mixed_weights: Optional[Dict[str, float]] = None,
    norm_mode: str = NORM_MODE,
    include_task01: bool = True,
) -> DataLoader:
    set_seed(seed)
    root = _project_root()
    split_txt = str(root / "configs" / "splits_2d" / "train.txt")

    ds = Brats3DTask01Supervised(
        brats_split_txt=split_txt,
        brats_root_3d="data/processed/3d/labeled",
        task01_root_3d="data/processed_task01/3d/labeled",
        include_task01=include_task01,
        patch_size=patch_size,
        mode="train",
        sampling_mode=sampling_mode,
        rejection_thresh=rejection_thresh,
        rejection_max=rejection_max,
        mixed_weights=mixed_weights,
        use_augment=True,
        norm_mode=norm_mode,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def build_brats3d_sup_val_loader(
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    batch_size: int = 1,
    num_workers: int = 2,
    sampling_mode: str = "mixed",
    rejection_thresh: float = 0.01,
    rejection_max: int = 8,
    mixed_weights: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
    norm_mode: str = NORM_MODE,
) -> DataLoader:
    if seed is not None:
        set_seed(seed)
    root = _project_root()
    split_txt = str(root / "configs" / "splits_2d" / "val.txt")
    ds = Brats3DSupervisedSingleRoot(
        split_txt=split_txt,
        root_3d="data/processed/3d/labeled",
        patch_size=patch_size,
        mode="val",
        sampling_mode=sampling_mode,
        rejection_thresh=rejection_thresh,
        rejection_max=rejection_max,
        mixed_weights=mixed_weights,
        use_augment=False,
        norm_mode=norm_mode,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader


def build_brats3d_sup_test_loader(
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    batch_size: int = 1,
    num_workers: int = 2,
    sampling_mode: str = "mixed",
    rejection_thresh: float = 0.01,
    rejection_max: int = 8,
    mixed_weights: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
    norm_mode: str = NORM_MODE,
) -> DataLoader:
    if seed is not None:
        set_seed(seed)
    root = _project_root()
    split_txt = str(root / "configs" / "splits_2d" / "test.txt")
    ds = Brats3DSupervisedSingleRoot(
        split_txt=split_txt,
        root_3d="data/processed/3d/labeled",
        patch_size=patch_size,
        mode="test",
        sampling_mode=sampling_mode,
        rejection_thresh=rejection_thresh,
        rejection_max=rejection_max,
        mixed_weights=mixed_weights,
        use_augment=False,
        norm_mode=norm_mode,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader


# ================== Self-test ==================
if __name__ == "__main__":
    set_seed(SELFTEST_SEED)
    root = _project_root()
    print("=== BRATS3D + TASK01 SUPERVISED DATALOADER SELF-TEST ===")
    print(f"[INFO] Project root: {root}")
    print(f"[INFO] NORM_MODE = {NORM_MODE}")

    # ----- Train loader (BraTS + Task01) -----
    train_loader = build_brats3d_task01_sup_train_loader(
        patch_size=SELFTEST_PATCH,
        batch_size=SELFTEST_BATCH,
        num_workers=SELFTEST_NUM_WORKERS,
        sampling_mode="mixed",
        mixed_weights={"center_fg": 1, "random": 0},
        norm_mode=NORM_MODE,
        include_task01=True,
    )
    train_ds = train_loader.dataset
    print(
        f"[COUNT] TRAIN total={len(train_ds)} | "
        f"brats={getattr(train_ds, 'num_brats', 'NA')} | task01={getattr(train_ds, 'num_task01', 'NA')}"
    )

    # Peek 1 batch
    b = next(iter(train_loader))
    x_tr, y_tr = b["image"], b["label"]
    names_tr = list(b["case"])
    print(f"[TRAIN] x={tuple(x_tr.shape)} (N,C,D,H,W) | y={tuple(y_tr.shape)} (N,1,D,H,W) | cases(sample)={names_tr}")

    # ----- Val loader (BraTS only) -----
    val_loader = build_brats3d_sup_val_loader(
        patch_size=SELFTEST_PATCH,
        batch_size=2,
        num_workers=SELFTEST_NUM_WORKERS,
        sampling_mode="mixed",
        mixed_weights={"center_fg": 1, "random": 0},
        norm_mode=NORM_MODE,
        seed=SELFTEST_SEED,
    )
    print(f"[COUNT] VAL  total={len(val_loader.dataset)}")

    # ----- Test loader (BraTS only) -----
    test_loader = build_brats3d_sup_test_loader(
        patch_size=SELFTEST_PATCH,
        batch_size=2,
        num_workers=SELFTEST_NUM_WORKERS,
        sampling_mode="mixed",
        mixed_weights={"center_fg": 1, "random": 0},
        norm_mode=NORM_MODE,
        seed=SELFTEST_SEED,
    )
    print(f"[COUNT] TEST total={len(test_loader.dataset)}")

    print("[OK] Self-test done.")
