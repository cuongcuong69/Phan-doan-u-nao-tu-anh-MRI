from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from losses.metrics import calculate_metric_percase
from models.unet2d import UNet2D
from models.unetpp_2d import UNetPlusPlus
from models.transunet_2d import TransUNet
from models.vnet import VNet
from models.unetr import UNETR
from models.swin_unet_3d import SwinUnet3D, SwinUnet3DConfig

try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None

try:
    from scipy.ndimage import zoom as nd_zoom
except ImportError:  # pragma: no cover
    nd_zoom = None


@dataclass
class ModelResult:
    name: str
    kind: str
    pred_seg: np.ndarray
    gt_seg: np.ndarray
    spacing: Optional[Tuple[float, float, float]]
    time_sec: float
    metrics: Dict[str, Dict[str, float]]
    brain_mask: Optional[np.ndarray] = None
    input_size: str = ""


BASE_PHYSICAL_SIZE = 240.0


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_case_id(text: str) -> str:
    raw = text.strip()
    if raw.lower().startswith("brain_"):
        raw = raw.split("_", 1)[1]
    raw = raw.zfill(3)
    return f"Brain_{raw}"


def _load_png_float01(path: Path) -> np.ndarray:
    img = Image.open(str(path)).convert("L")
    arr = np.array(img)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _load_png_mask(path: Path) -> np.ndarray:
    img = Image.open(str(path)).convert("L")
    arr = np.array(img).astype(np.int16)
    arr[arr == 4] = 3
    return arr


def load_2d_case(case_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    mask_dir = case_dir / "mask"
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing 2D mask folder: {mask_dir}")

    mask_files = sorted(mask_dir.glob("mask_*.png"), key=lambda p: int(p.stem.split("_")[-1]))
    if not mask_files:
        raise FileNotFoundError(f"No 2D mask slices in: {mask_dir}")

    slices: List[np.ndarray] = []
    masks: List[np.ndarray] = []

    for mf in mask_files:
        sid = mf.stem.split("_")[-1]
        flair = _load_png_float01(case_dir / "flair" / f"flair_{sid}.png")
        t1 = _load_png_float01(case_dir / "t1" / f"t1_{sid}.png")
        t1ce = _load_png_float01(case_dir / "t1ce" / f"t1ce_{sid}.png")
        t2 = _load_png_float01(case_dir / "t2" / f"t2_{sid}.png")
        img = np.stack([flair, t1, t1ce, t2], axis=0)
        slices.append(img.astype(np.float32))
        masks.append(_load_png_mask(mf))

    vol_4ch = np.stack(slices, axis=0)
    gt_seg = np.stack(masks, axis=0)
    return vol_4ch, gt_seg


def load_3d_case(case_dir: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    if nib is None:
        raise RuntimeError("nibabel is required for 3D data loading.")

    flair_path = case_dir / "flair.nii.gz"
    t1_path = case_dir / "t1.nii.gz"
    t1ce_path = case_dir / "t1ce.nii.gz"
    t2_path = case_dir / "t2.nii.gz"
    mask_path = case_dir / "mask.nii.gz"
    for path in [flair_path, t1_path, t1ce_path, t2_path, mask_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing 3D file: {path}")

    flair_nii = nib.load(str(flair_path))
    flair = flair_nii.get_fdata().astype(np.float32)
    t1 = nib.load(str(t1_path)).get_fdata().astype(np.float32)
    t1ce = nib.load(str(t1ce_path)).get_fdata().astype(np.float32)
    t2 = nib.load(str(t2_path)).get_fdata().astype(np.float32)

    vol_4ch = np.stack([flair, t1, t1ce, t2], axis=0)
    gt_seg = nib.load(str(mask_path)).get_fdata().astype(np.int16)
    gt_seg[gt_seg == 4] = 3
    spacing = tuple(float(z) for z in flair_nii.header.get_zooms()[:3])
    return vol_4ch, gt_seg, spacing


def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in sd.keys()):
        return sd
    return {k[len("module."):]: v for k, v in sd.items()}


def _add_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in sd.keys()):
        return sd
    return {f"module.{k}": v for k, v in sd.items()}


def extract_state_dict(ckpt: object) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "models", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        return ckpt
    return ckpt


def load_model_state_dict(model: torch.nn.Module, sd: Dict[str, torch.Tensor]) -> None:
    model_sd = model.state_dict()

    loaded_has_module = any(k.startswith("module.") for k in sd.keys())
    model_has_module = any(k.startswith("module.") for k in model_sd.keys())
    if loaded_has_module and not model_has_module:
        sd = _strip_module_prefix(sd)
    elif (not loaded_has_module) and model_has_module:
        sd = _add_module_prefix(sd)

    filtered = {}
    skipped = []
    for key, val in sd.items():
        if key in model_sd and model_sd[key].shape == val.shape:
            filtered[key] = val
        else:
            skipped.append(key)

    model.load_state_dict(filtered, strict=False)
    if skipped:
        print(f"[CKPT][WARN] Skipped {len(skipped)} keys due to shape mismatch.")


def load_model_weights(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    sd = extract_state_dict(ckpt)
    load_model_state_dict(model, sd)
    return sd


def compute_region_metrics(
    pred_seg: np.ndarray,
    gt_seg: np.ndarray,
    spacing: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Dict[str, float]]:
    gt_wt = (gt_seg > 0).astype(np.uint8)
    gt_tc = ((gt_seg == 1) | (gt_seg == 3)).astype(np.uint8)
    gt_et = (gt_seg == 3).astype(np.uint8)

    pred_wt = (pred_seg > 0).astype(np.uint8)
    pred_tc = ((pred_seg == 1) | (pred_seg == 3)).astype(np.uint8)
    pred_et = (pred_seg == 3).astype(np.uint8)

    def _metric(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
        dice, iou, hd95, asd = calculate_metric_percase(pred, gt, spacing=spacing)
        return {"dice": float(dice), "iou": float(iou), "asd": float(asd), "hd95": float(hd95)}

    return {
        "WT": _metric(pred_wt, gt_wt),
        "TC": _metric(pred_tc, gt_tc),
        "ET": _metric(pred_et, gt_et),
    }


def spacing_from_shape(shape: Tuple[int, int, int], base_size: float = BASE_PHYSICAL_SIZE) -> Tuple[float, float, float]:
    d, h, w = shape
    return (base_size / float(d), base_size / float(h), base_size / float(w))


@torch.no_grad()
def predict_2d_model(
    model: torch.nn.Module,
    vol_4ch: np.ndarray,
    device: torch.device,
    batch_size: int = 4,
    input_size: Optional[Tuple[int, int]] = None,
    output_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    model.eval()
    total = vol_4ch.shape[0]
    preds: List[np.ndarray] = []

    for i in range(0, total, batch_size):
        batch = torch.from_numpy(vol_4ch[i : i + batch_size]).to(device)
        if input_size and batch.shape[-2:] != input_size:
            batch = F.interpolate(batch, size=input_size, mode="bilinear", align_corners=False)

        out = model(batch)
        if isinstance(out, list):
            out = out[-1]
        probs = torch.sigmoid(out)
        pred = probs > 0.5
        if output_size and pred.shape[-2:] != output_size:
            pred = F.interpolate(pred.float(), size=output_size, mode="nearest") > 0.5
        pred_np = pred.detach().cpu().numpy()

        seg = np.zeros((pred_np.shape[0], pred_np.shape[2], pred_np.shape[3]), dtype=np.uint8)
        wt = pred_np[:, 0]
        tc = pred_np[:, 1]
        et = pred_np[:, 2]
        seg[wt] = 2
        seg[tc] = 1
        seg[et] = 3
        preds.append(seg)

    return np.concatenate(preds, axis=0)


@torch.no_grad()
def predict_2d_multiclass(
    model: torch.nn.Module,
    vol_4ch: np.ndarray,
    device: torch.device,
    batch_size: int = 4,
    input_size: Optional[Tuple[int, int]] = None,
    output_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    model.eval()
    total = vol_4ch.shape[0]
    preds: List[np.ndarray] = []

    for i in range(0, total, batch_size):
        batch = torch.from_numpy(vol_4ch[i : i + batch_size]).to(device)
        if input_size and batch.shape[-2:] != input_size:
            batch = F.interpolate(batch, size=input_size, mode="bilinear", align_corners=False)

        out = model(batch)
        if isinstance(out, list):
            out = out[-1]
        logits = out
        if output_size and logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        pred = torch.softmax(logits, dim=1).argmax(dim=1)
        preds.append(pred.detach().cpu().numpy().astype(np.int16))

    return np.concatenate(preds, axis=0)


def _compute_steps(dim: int, patch: int, stride: int) -> List[int]:
    if dim <= patch:
        return [0]
    steps = list(range(0, dim - patch + 1, stride))
    if steps[-1] != dim - patch:
        steps.append(dim - patch)
    return steps


@torch.no_grad()
def sliding_window_multiclass(
    model: torch.nn.Module,
    vol_4ch: np.ndarray,
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    device: torch.device,
    num_classes: int = 4,
) -> np.ndarray:
    model.eval()

    _, D, H, W = vol_4ch.shape
    orig_shape = (D, H, W)
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    pad_d = max(0, pd - D)
    pad_h = max(0, ph - H)
    pad_w = max(0, pw - W)
    if pad_d or pad_h or pad_w:
        vol_4ch = np.pad(
            vol_4ch,
            ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
            mode="constant",
        )
        _, D, H, W = vol_4ch.shape

    steps_d = _compute_steps(D, pd, sd)
    steps_h = _compute_steps(H, ph, sh)
    steps_w = _compute_steps(W, pw, sw)

    prob_sum = np.zeros((num_classes, D, H, W), dtype=np.float32)
    count = np.zeros((D, H, W), dtype=np.float32)

    for z in steps_d:
        for y in steps_h:
            for x in steps_w:
                patch = vol_4ch[:, z : z + pd, y : y + ph, x : x + pw]
                patch_t = torch.from_numpy(patch[None, ...]).to(device)
                out = model(patch_t)
                logits = out.get("seg", out) if isinstance(out, dict) else out
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                prob_sum[:, z : z + pd, y : y + ph, x : x + pw] += probs
                count[z : z + pd, y : y + ph, x : x + pw] += 1.0

    count[count == 0] = 1.0
    prob_vol = prob_sum / count[None, ...]

    if pad_d or pad_h or pad_w:
        od, oh, ow = orig_shape
        prob_vol = prob_vol[:, :od, :oh, :ow]

    return prob_vol


def resize_vol_4ch_to_target(vol_4ch: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    if nd_zoom is None:
        raise RuntimeError("scipy is required for UNETR resizing.")
    c, d0, h0, w0 = vol_4ch.shape
    dt, ht, wt = target_shape
    out = np.zeros((c, dt, ht, wt), dtype=np.float32)
    spatial_zoom = (dt / float(d0), ht / float(h0), wt / float(w0))
    for ch in range(c):
        out[ch] = nd_zoom(vol_4ch[ch], zoom=spatial_zoom, order=1)
    return out.astype(np.float32)


def resize_seg_to_target(seg: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    if nd_zoom is None:
        raise RuntimeError("scipy is required for UNETR resizing.")
    d1, h1, w1 = seg.shape
    d0, h0, w0 = target_shape
    zoom_factors = (d0 / float(d1), h0 / float(h1), w0 / float(w1))
    return nd_zoom(seg.astype(np.int16), zoom=zoom_factors, order=0).astype(np.int16)


@torch.no_grad()
def infer_unetr_full_volume(
    model: torch.nn.Module,
    vol_4ch: np.ndarray,
    target_size: Tuple[int, int, int],
    device: torch.device,
    num_classes: int = 4,
) -> np.ndarray:
    model.eval()
    _, d0, h0, w0 = vol_4ch.shape
    vol_resized = resize_vol_4ch_to_target(vol_4ch, target_size)
    x = torch.from_numpy(vol_resized[None, ...]).to(device)
    out = model(x)
    logits = out.get("seg", out) if isinstance(out, dict) else out
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    seg_resized = np.argmax(probs, axis=0).astype(np.int16)
    return resize_seg_to_target(seg_resized, (d0, h0, w0))


def run_unet2d(vol_4ch: np.ndarray, gt_seg: np.ndarray, weights: Path, device: torch.device) -> ModelResult:
    model = UNet2D(in_channels=4, num_classes=3, base_ch=32).to(device)
    load_model_weights(model, weights, device)
    start = time.perf_counter()
    pred_seg = predict_2d_model(model, vol_4ch, device, batch_size=4)
    elapsed = time.perf_counter() - start
    spacing = spacing_from_shape(pred_seg.shape)
    metrics = compute_region_metrics(pred_seg, gt_seg, spacing=spacing)
    input_size = f"{vol_4ch.shape[0]}x{vol_4ch.shape[2]}x{vol_4ch.shape[3]}"
    return ModelResult("UNet", "2d", pred_seg, gt_seg, spacing, elapsed, metrics, input_size=input_size)


def run_unetpp(vol_4ch: np.ndarray, gt_seg: np.ndarray, weights: Path, device: torch.device) -> ModelResult:
    ckpt = torch.load(str(weights), map_location=device)
    sd = extract_state_dict(ckpt)
    out_ch = None
    for key in ("final.weight", "module.final.weight", "final1.weight", "module.final1.weight"):
        if key in sd:
            out_ch = int(sd[key].shape[0])
            break
    if out_ch is None:
        out_ch = 3

    model = UNetPlusPlus(in_channels=4, num_classes=out_ch, deep_supervision=False).to(device)
    load_model_state_dict(model, sd)

    start = time.perf_counter()
    if out_ch == 4:
        pred_seg = predict_2d_multiclass(model, vol_4ch, device, batch_size=4)
    else:
        pred_seg = predict_2d_model(model, vol_4ch, device, batch_size=4)
    elapsed = time.perf_counter() - start
    spacing = spacing_from_shape(pred_seg.shape)
    metrics = compute_region_metrics(pred_seg, gt_seg, spacing=spacing)
    input_size = f"{vol_4ch.shape[0]}x{vol_4ch.shape[2]}x{vol_4ch.shape[3]}"
    return ModelResult("UNet++", "2d", pred_seg, gt_seg, spacing, elapsed, metrics, input_size=input_size)


def run_transunet(vol_4ch: np.ndarray, gt_seg: np.ndarray, weights: Path, device: torch.device) -> ModelResult:
    model = TransUNet(in_channels=4, num_classes=3, img_dim=256).to(device)
    load_model_weights(model, weights, device)
    target_size = (vol_4ch.shape[2], vol_4ch.shape[3])
    start = time.perf_counter()
    pred_seg = predict_2d_model(
        model,
        vol_4ch,
        device,
        batch_size=2,
        input_size=(256, 256),
        output_size=target_size,
    )
    elapsed = time.perf_counter() - start
    spacing = spacing_from_shape(pred_seg.shape)
    metrics = compute_region_metrics(pred_seg, gt_seg, spacing=spacing)
    input_size = f"{target_size[0]}x{target_size[1]}"
    return ModelResult("TransUNet", "2d", pred_seg, gt_seg, spacing, elapsed, metrics, input_size=input_size)


def run_vnet(
    vol_4ch: np.ndarray,
    gt_seg: np.ndarray,
    weights: Path,
    device: torch.device,
    spacing: Optional[Tuple[float, float, float]] = None,
) -> ModelResult:
    model = VNet(
        n_channels=4,
        n_classes=4,
        n_filters=16,
        normalization="groupnorm",
        has_dropout=True,
    ).to(device)
    load_model_weights(model, weights, device)
    start = time.perf_counter()
    prob_vol = sliding_window_multiclass(
        model, vol_4ch, patch_size=(128, 128, 128), stride=(128, 128, 128), device=device, num_classes=4
    )
    pred_seg = np.argmax(prob_vol, axis=0).astype(np.int16)
    elapsed = time.perf_counter() - start
    spacing_out = spacing_from_shape(pred_seg.shape)
    metrics = compute_region_metrics(pred_seg, gt_seg, spacing=spacing_out)
    input_size = "128x128x128 (patch)"
    return ModelResult("VNet", "3d", pred_seg, gt_seg, spacing_out, elapsed, metrics, input_size=input_size)


def run_unetr(
    vol_4ch: np.ndarray,
    gt_seg: np.ndarray,
    weights: Path,
    device: torch.device,
    spacing: Optional[Tuple[float, float, float]] = None,
) -> ModelResult:
    model = UNETR(
        n_channels=4,
        n_classes=4,
        img_size=(128, 128, 128),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
    ).to(device)
    load_model_weights(model, weights, device)
    start = time.perf_counter()
    pred_seg = infer_unetr_full_volume(model, vol_4ch, (128, 128, 128), device, num_classes=4)
    elapsed = time.perf_counter() - start
    spacing_out = spacing_from_shape(pred_seg.shape)
    metrics = compute_region_metrics(pred_seg, gt_seg, spacing=spacing_out)
    input_size = "128x128x128"
    return ModelResult("UNETR", "3d", pred_seg, gt_seg, spacing_out, elapsed, metrics, input_size=input_size)


def run_swin_unet_3d(
    vol_4ch: np.ndarray,
    gt_seg: np.ndarray,
    weights: Path,
    device: torch.device,
    spacing: Optional[Tuple[float, float, float]] = None,
) -> ModelResult:
    cfg = SwinUnet3DConfig(
        in_channels=4,
        num_classes=4,
        embed_dim=96,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(4, 4, 4),
        mlp_ratio=4.0,
    )
    model = SwinUnet3D(cfg).to(device)
    load_model_weights(model, weights, device)
    start = time.perf_counter()
    prob_vol = sliding_window_multiclass(
        model, vol_4ch, patch_size=(128, 128, 128), stride=(128, 128, 128), device=device, num_classes=4
    )
    pred_seg = np.argmax(prob_vol, axis=0).astype(np.int16)
    elapsed = time.perf_counter() - start
    spacing_out = spacing_from_shape(pred_seg.shape)
    metrics = compute_region_metrics(pred_seg, gt_seg, spacing=spacing_out)
    input_size = "128x128x128 (patch)"
    return ModelResult("SwinUNet3D", "3d", pred_seg, gt_seg, spacing_out, elapsed, metrics, input_size=input_size)
