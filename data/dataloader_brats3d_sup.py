# -*- coding: utf-8 -*-
"""
Dataloader 3D NIfTI cho phân đoạn khối u não BraTS2020 (supervised, 3D patch-based).

- Đầu vào: data/processed/3d/labeled/Brain_xxx/:
    flair.nii.gz
    t1.nii.gz
    t1ce.nii.gz
    t2.nii.gz
    mask.nii.gz    (nhãn 0=bg, 1,2,3 là các vùng tumor đã map 4->3)

- 4 modality được stack thành: image shape = [4, D, H, W] (C,D,H,W)
- mask: [D, H, W] (giá trị 0..3), khi đưa vào tensor: [1, D, H, W]

- Patch-based crop 3D:
    sampling_mode:
        "random"    : crop ngẫu nhiên
        "rejection" : tránh patch toàn background
        "center_fg" : crop quanh voxel foreground
        "mixed"     : trộn các mode trên theo trọng số

- Split dựa vào:
    configs/splits_2d/train.txt
    configs/splits_2d/val.txt
    configs/splits_2d/test.txt

- Self-test:
    python -m data.dataloader_brats3d_sup
"""

from __future__ import annotations
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any, Union

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from scipy.ndimage import zoom as nd_zoom

# =============================================================================
# NORMALIZATION CONFIG (phải khớp với preprocess_brats3d.py)
# =============================================================================
# "minmax": dữ liệu đã được chuẩn hóa về [0,1] theo non-zero
# "zscore": dữ liệu đã được chuẩn hóa z-score trên non-zero, thường clip [-5,5]
NORM_MODE = "zscore"   # hoặc "minmax"
ZSCORE_CLIP = (-5.0, 5.0)
# =============================================================================
# SELF-TEST CONFIG
# =============================================================================
SELFTEST_SEED = 2025
SELFTEST_PATCH = (128, 128, 128)   # (D,H,W)
SELFTEST_BATCH = 4
SELFTEST_NUM_WORKERS = 0
SELFTEST_VIS_DIR = "experiments/vis_brats3d_sup"
SELFTEST_SLICE = "middle"  # hoặc int
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
    Giữ nguyên intensity (đã được chuẩn hóa ở bước preprocessing
    theo NORM_MODE: 'minmax' hoặc 'zscore').
    """
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)  # (X,Y,Z)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={data.shape} at {path}")
    # transpose (X,Y,Z) -> (Z,Y,X) = (D,H,W)
    data = np.transpose(data, (2, 1, 0))
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
        p_gamma: float = 0.0,         # đặt >0 nếu muốn dùng gamma
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
        """
        Resize vol (D_in,H_in,W_in) về đúng out_shape (d,h,w),
        dùng zoom theo tỉ lệ + pad/crop để đảm bảo shape khớp 100%.
        """
        d, h, w = out_shape
        D_in, H_in, W_in = vol.shape

        # zoom theo tỉ lệ
        scale = (
            d / float(D_in),
            h / float(H_in),
            w / float(W_in),
        )
        vol_resized = nd_zoom(vol, zoom=scale, order=1)

        # nếu vì rounding mà hơi lệch size thì pad/crop về đúng (d,h,w)
        D2, H2, W2 = vol_resized.shape

        # ---- pad nếu thiếu ----
        pad_D = max(0, d - D2)
        pad_H = max(0, h - H2)
        pad_W = max(0, w - W2)
        if pad_D > 0 or pad_H > 0 or pad_W > 0:
            pad_before_D = pad_D // 2
            pad_after_D = pad_D - pad_before_D
            pad_before_H = pad_H // 2
            pad_after_H = pad_H - pad_before_H
            pad_before_W = pad_W // 2
            pad_after_W = pad_W - pad_before_W
            vol_resized = np.pad(
                vol_resized,
                (
                    (pad_before_D, pad_after_D),
                    (pad_before_H, pad_after_H),
                    (pad_before_W, pad_after_W),
                ),
                mode="constant",
                constant_values=0,
            )
            D2, H2, W2 = vol_resized.shape

        # ---- crop nếu thừa ----
        if D2 > d or H2 > h or W2 > w:
            zs = max(0, (D2 - d) // 2)
            ys = max(0, (H2 - h) // 2)
            xs = max(0, (W2 - w) // 2)
            vol_resized = vol_resized[zs:zs + d, ys:ys + h, xs:xs + w]

        assert vol_resized.shape == (d, h, w), f"resize ra {vol_resized.shape}, cần {(d, h, w)}"
        return vol_resized

    def _zoom(self, image: np.ndarray, mask: np.ndarray, zoom_factor: float):
        """
        Zoom 3D cho cả image [C,D,H,W] và mask [D,H,W],
        sau đó resize lại đúng patch_size gốc.
        """
        C, D, H, W = image.shape
        z_img = np.zeros_like(image)

        # zoom từng channel ảnh
        for c in range(C):
            z_vol = nd_zoom(image[c], zoom=zoom_factor, order=1)
            z_img[c] = self._resize_to_shape(z_vol, (D, H, W))

        # zoom mask dùng nearest (order=0) để không phá nhãn
        z_msk = nd_zoom(mask, zoom=zoom_factor, order=0)
        z_msk = self._resize_to_shape(z_msk, (D, H, W)).astype(mask.dtype)

        return z_img, z_msk

    def _gamma_transform(self, img: np.ndarray, gamma: float) -> np.ndarray:
        """
        Gamma transform cho dữ liệu đã minmax [0,1].
        Với zscore thì mình không dùng gamma.
        """
        if self.norm_mode != "minmax":
            return img
        img = np.clip(img, 0.0, 1.0)
        img = img ** gamma
        return img

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        """
        image: [C,D,H,W], mask: [D,H,W]
        """
        C, D, H, W = image.shape

        # Random flip
        if np.random.rand() < self.p_flip:
            for axis in [1, 2, 3]:  # D,H,W
                if np.random.rand() < 0.5:
                    image = np.flip(image, axis=axis)
                    mask = np.flip(mask, axis=axis - 1)  # mask là [D,H,W]

        # Intensity jitter (riêng từng channel)
        if np.random.rand() < self.p_jitter:
            for c in range(C):
                scale = np.random.uniform(*self.jitter_scale_range)
                shift = np.random.uniform(*self.jitter_shift_range)
                img_c = image[c] * scale + shift
                img_c = self._maybe_clip(img_c)
                image[c] = img_c

        # Gaussian noise
        if np.random.rand() < self.p_noise:
            noise = np.random.normal(loc=0.0, scale=self.noise_std, size=image.shape).astype(np.float32)
            image = image + noise
            image = self._maybe_clip(image)

        # Gamma (chỉ thực sự meaningful với minmax)
        if self.p_gamma > 0 and np.random.rand() < self.p_gamma and self.norm_mode == "minmax":
            gamma = np.random.uniform(*self.gamma_range)
            for c in range(C):
                image[c] = self._gamma_transform(image[c], gamma)

        # Random zoom
        if self.p_zoom > 0 and np.random.rand() < self.p_zoom:
            zf = np.random.uniform(*self.zoom_range)
            image, mask = self._zoom(image, mask, zf)

        return image, mask


# ================== Dataset Supervised 3D ==================

class Brats3DSupervised(Dataset):
    """
    Dataset GIÁM SÁT 3D cho BraTS2020 đã tiền xử lý.
    - split_txt: mỗi dòng là 'Brain_XXX'
    - root_3d: thư mục chứa Brain_xxx/*
    - patch_size: (D,H,W)
    - sampling_mode: "random"|"rejection"|"center_fg"|"mixed"
    - norm_mode: "minmax" hoặc "zscore" (phải khớp preprocessing)
    """
    def __init__(
        self,
        split_txt: str,
        root_3d: str = "data/processed/3d/labeled",
        patch_size: Tuple[int, int, int] = (96, 96, 96),
        mode: str = "train",     # 'train'|'val'|'test'
        sampling_mode: str = "mixed",
        rejection_thresh: float = 0.01,
        rejection_max: int = 8,
        mixed_weights: Optional[Dict[str, float]] = None,
        use_augment: bool = True,
        norm_mode: str = NORM_MODE,
    ):
        self.mode = mode
        self.patch_size = tuple(map(int, patch_size))
        self.root_3d = Path(_abs_from_root(root_3d))
        self.norm_mode = norm_mode.lower().strip()

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

        # đọc danh sách case
        cases: List[str] = []
        with open(split_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cases.append(line)
        self.cases = cases

        print(f"[Brats3DSupervised] mode={mode} | total={len(self.cases)} | "
              f"root_3d={self.root_3d} | norm_mode={self.norm_mode}")

    def __len__(self) -> int:
        return len(self.cases)

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
        """
        img4d: [C,D,H,W], msk3d: [D,H,W]
        """
        _, D, H, W = img4d.shape
        d, h, w = self.patch_size
        z, y, x = _random_crop_coords(D, H, W, d, h, w)
        return (
            img4d[:, z:z+d, y:y+h, x:x+w],
            msk3d[z:z+d, y:y+h, x:x+w],
        )

    def _crop_rejection(self, img4d: np.ndarray, msk3d: np.ndarray):
        _, D, H, W = img4d.shape
        d, h, w = self.patch_size
        for _ in range(self.rejection_max):
            z, y, x = _random_crop_coords(D, H, W, d, h, w)
            sub = msk3d[z:z+d, y:y+h, x:x+w]
            # foreground tỉ lệ
            if (sub > 0).mean() >= self.rejection_thresh:
                return img4d[:, z:z+d, y:y+h, x:x+w], sub
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
            img4d[:, zs:zs+d, ys:ys+h, xs:xs+w],
            msk3d[zs:zs+d, ys:ys+h, xs:xs+w],
        )

    def _sample_patch(self, img4d: np.ndarray, msk3d: np.ndarray):
        mode = self.sampling_mode
        if mode == "mixed":
            mode = self._choose_mixed_mode()
        if mode == "rejection":
            return self._crop_rejection(img4d, msk3d)
        if mode == "center_fg":
            return self._crop_center_fg(img4d, msk3d)
        # default: random
        return self._crop_random(img4d, msk3d)

    # ---- __getitem__ ----
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case_id = self.cases[idx]
        case_dir = self.root_3d / case_id

        flair_fp = case_dir / "flair.nii.gz"
        t1_fp    = case_dir / "t1.nii.gz"
        t1ce_fp  = case_dir / "t1ce.nii.gz"
        t2_fp    = case_dir / "t2.nii.gz"
        mask_fp  = case_dir / "mask.nii.gz"

        # Load 3D volumes: chuyển về (D,H,W)
        flair = load_nii_to_DHW(str(flair_fp))
        t1    = load_nii_to_DHW(str(t1_fp))
        t1ce  = load_nii_to_DHW(str(t1ce_fp))
        t2    = load_nii_to_DHW(str(t2_fp))
        msk   = load_nii_to_DHW(str(mask_fp)).astype(np.int16)  # 0..3

        # Stack modality: [C,D,H,W]
        img4d = np.stack([flair, t1, t1ce, t2], axis=0)

        # Ensure min size trước khi crop
        img4d = ensure_min_size_3d(img4d, self.patch_size)
        msk   = ensure_min_size_3d(msk,   self.patch_size)

        # Sample patch (TRAIN / VAL / TEST đều dùng patch-based)
        img_patch, msk_patch = self._sample_patch(img4d, msk)

        # Augment (chỉ TRAIN)
        if self.use_augment:
            img_patch, msk_patch = self.augment(img_patch, msk_patch)

        # To tensor
        img_t = torch.from_numpy(img_patch.astype(np.float32))  # [4,D,H,W]
        lbl_t = torch.from_numpy(msk_patch.astype(np.int64)).unsqueeze(0)  # [1,D,H,W] multi-class 0..3

        out: Dict[str, Any] = {
            "image": img_t,    # (C,D,H,W)
            "label": lbl_t,    # (1,D,H,W)
            "case": case_id,
        }
        return out


# ================== Builders ==================

def build_brats3d_sup_train_loader(
    patch_size: Tuple[int, int, int] = (96, 96, 96),
    batch_size: int = 2,
    num_workers: int = 2,
    seed: int = 2025,
    sampling_mode: str = "mixed",
    rejection_thresh: float = 0.01,
    rejection_max: int = 8,
    mixed_weights: Optional[Dict[str, float]] = None,
    norm_mode: str = NORM_MODE,
) -> DataLoader:
    set_seed(seed)
    root = _project_root()
    split_txt = str(root / "configs" / "splits_2d" / "train.txt")
    ds = Brats3DSupervised(
        split_txt=split_txt,
        root_3d="data/processed/3d/labeled",
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
    patch_size: Tuple[int, int, int] = (96, 96, 96),
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
    ds = Brats3DSupervised(
        split_txt=split_txt,
        root_3d="data/processed/3d/labeled",
        patch_size=patch_size,
        mode="val",
        sampling_mode=sampling_mode,
        rejection_thresh=rejection_thresh,
        rejection_max=rejection_max,
        mixed_weights=mixed_weights,
        use_augment=False,             # không augment cho val
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
    patch_size: Tuple[int, int, int] = (96, 96, 96),
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
    ds = Brats3DSupervised(
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


# ================== Visualization (debug) ==================

def visualize_brats3d_batch(
    x: torch.Tensor,
    y: Optional[torch.Tensor],
    names: List[str],
    out_dir: Union[str, Path] = "experiments/vis_brats3d_sup",
    slice_idx: Union[int, str] = "middle",
    plane: str = "axial",
    prefix: str = "brats3d",
    dpi: int = 150,
    class_labels: Optional[Dict[int, str]] = None,
) -> List[Path]:
    """
    Vẽ vài lát cắt để debug.
    x: (N,4,D,H,W), y: (N,1,D,H,W)
    - Cột 1: FLAIR
    - Cột 2: FLAIR + overlay mask với colormap rời rạc + legend.
    """
    out_dir = Path(_abs_from_root(str(out_dir)))
    out_dir.mkdir(parents=True, exist_ok=True)

    x_np = x.detach().cpu().numpy()  # (N,4,D,H,W)
    y_np = y.detach().cpu().numpy() if y is not None else None

    N, C, D, H, W = x_np.shape
    plane = plane.lower().strip()
    assert plane in ("axial", "coronal", "sagittal")

    def _resolve_idx():
        if slice_idx == "middle":
            return {"axial": D // 2, "coronal": H // 2, "sagittal": W // 2}[plane]
        i = int(slice_idx)
        if plane == "axial":    i = max(0, min(D - 1, i))
        if plane == "coronal":  i = max(0, min(H - 1, i))
        if plane == "sagittal": i = max(0, min(W - 1, i))
        return i

    idx = _resolve_idx()

    def _slice2d(vol3d: np.ndarray, plane: str, idx: int) -> np.ndarray:
        if plane == "axial":     return vol3d[idx, :, :]
        if plane == "coronal":   return vol3d[:, idx, :]
        return vol3d[:, :, idx]

    # colormap rời rạc cho mask: 0=background (transparent), 1..3 = màu khác nhau
    colors = np.array([
        [0.0, 0.0, 0.0, 0.0],   # 0 - bg (transparent)
        [0.0, 1.0, 1.0, 0.7],   # 1 - cyan
        [1.0, 0.4, 0.0, 0.7],   # 2 - orange
        [1.0, 1.0, 0.0, 0.7],   # 3 - yellow
    ])
    cmap = mcolors.ListedColormap(colors)
    boundaries = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=cmap.N)

    # label cho legend
    if class_labels is None:
        class_labels = {
            1: "Label 1",
            2: "Label 2",
            3: "Label 3",
        }

    legend_handles = [
        Patch(facecolor=colors[c, :3], edgecolor="k", label=class_labels.get(c, f"Label {c}"))
        for c in [1, 2, 3]
    ]

    saved = []
    for i in range(N):
        img3d = x_np[i, 0]  # kênh FLAIR
        img2d = _slice2d(img3d, plane, idx)

        ncols = 1 + int(y_np is not None)
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
        if ncols == 1:
            axes = [axes]

        # --- cột 1: FLAIR ---
        ax0 = axes[0]
        vmin = np.percentile(img2d, 1)
        vmax = np.percentile(img2d, 99)
        ax0.imshow(img2d, cmap="gray", vmin=vmin, vmax=vmax)
        ax0.set_title(f"{names[i]} - FLAIR ({NORM_MODE})")
        ax0.axis("off")

        # --- cột 2: overlay mask ---
        if y_np is not None:
            ax1 = axes[1]
            m3d = y_np[i, 0]
            m2d = _slice2d(m3d, plane, idx)

            ax1.imshow(img2d, cmap="gray", vmin=vmin, vmax=vmax)
            ax1.imshow(m2d, cmap=cmap, norm=norm)
            ax1.set_title("FLAIR + mask (0..3)")
            ax1.axis("off")

            ax1.legend(
                handles=legend_handles,
                loc="upper right",
                bbox_to_anchor=(1.05, 1.0),
                borderaxespad=0.0,
                fontsize=9,
            )

        out_path = out_dir / f"{prefix}_{i:02d}_{names[i]}_{plane}{idx}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

    return saved


# ================== Self-test ==================
if __name__ == "__main__":
    set_seed(SELFTEST_SEED)
    root = _project_root()
    print("=== BRATS3D SUPERVISED DATALOADER SELF-TEST ===")
    print(f"[INFO] NORM_MODE = {NORM_MODE}")

    # ----- Train loader -----
    train_loader = build_brats3d_sup_train_loader(
        patch_size=SELFTEST_PATCH,
        batch_size=SELFTEST_BATCH,
        num_workers=SELFTEST_NUM_WORKERS,
        sampling_mode="mixed",
        mixed_weights={"center_fg": 1, "random": 0},
        norm_mode=NORM_MODE,
    )
    b = next(iter(train_loader))
    x_tr, y_tr = b["image"], b["label"]
    print(f"  x min={x_tr.min().item():.4f}, max={x_tr.max().item():.4f}")
    names_tr = list(b["case"])
    print(f"[TRAIN] x={tuple(x_tr.shape)} (N,C,D,H,W) | y={tuple(y_tr.shape)} (N,1,D,H,W) | "
          f"cases={names_tr}")

    # ----- Val loader -----
    val_loader = build_brats3d_sup_val_loader(
        patch_size=SELFTEST_PATCH,
        batch_size=2,
        num_workers=SELFTEST_NUM_WORKERS,
        sampling_mode="mixed",
        mixed_weights={"center_fg": 1, "random": 0},
        norm_mode=NORM_MODE,
    )
    vb = next(iter(val_loader))
    x_val, y_val = vb["image"], vb["label"]
    names_val = list(vb["case"])
    print(f"[VAL]   x={tuple(x_val.shape)} | y={tuple(y_val.shape)} | cases={names_val}")

    # Thử visualize 1 batch train
    try:
        visualize_brats3d_batch(
            x_tr, y_tr, names_tr,
            out_dir=root / SELFTEST_VIS_DIR,
            slice_idx="middle",
            plane="axial",
            prefix="train",
        )
        print(f"[OK] Saved visualization to {root / SELFTEST_VIS_DIR}")
    except Exception as e:
        print(f"[WARN] visualize failed: {e}")
        
    # Thử visualize 1 batch val
    try:
        visualize_brats3d_batch(
            x_val, y_val, names_val,
            out_dir=root / SELFTEST_VIS_DIR,
            slice_idx="middle",
            plane="axial",
            prefix="val",
        )
        print(f"[OK] Saved visualization to {root / SELFTEST_VIS_DIR}")
    except Exception as e:
        print(f"[WARN] visualize failed: {e}")

    print("[OK] BRATS3D supervised dataloader self-test done.")
