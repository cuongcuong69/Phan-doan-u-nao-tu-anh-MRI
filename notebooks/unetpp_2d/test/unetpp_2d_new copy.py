# %%
# ============================================================================
# 1. IMPORTS & SETUP
# ============================================================================
import os
import random
import json
import warnings
from pathlib import Path
from typing import Tuple, Optional, List
import csv
from torch.cuda.amp import GradScaler, autocast

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Metrics (medpy)
try:
    from medpy.metric.binary import hd95 as medpy_hd95, assd as medpy_assd
except Exception as e:
    medpy_hd95 = None
    medpy_assd = None
    print("Warning: medpy not available. Install it with `pip install medpy` to compute HD95/ASSD.")

warnings.filterwarnings('ignore')

# ============================================================================
# ### ĐƯỜNG DẪN ###
# ============================================================================
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path('.').resolve()
    print(f"Running in interactive mode, SCRIPT_DIR set to: {SCRIPT_DIR}")


PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / 'data' / 'processed' / '2d' / 'labeled'
CONFIG_DIR = PROJECT_ROOT / 'configs' / 'splits_2d'
LOG_DIR = Path(r'D:/Project Advanced CV/logs/unetpp_2d_multilabel_ds_dice_ce_sum')
CKPT_DIR = Path(r'D:/Project Advanced CV/checkpoints/unetpp_2d_multilabel_ds_dice_ce_sum')

LOG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ Seed set to {seed}")

set_seed(42)

# %%
# ============================================================================
# 2. LOGGER
# ============================================================================
import logging

def setup_logger(log_dir: Path, name: str = 'unetpp_2d_multilabel') -> logging.Logger:
    log_file = log_dir / 'training_log.txt'
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f'Logger initialized. Log file: {log_file}')
    return logger

logger = setup_logger(LOG_DIR)


# %%
# ============================================================================
# 3. DATASET & DATALOADER
# ============================================================================
class Brats2dDataset(Dataset):
    def __init__(self, data_root, split_file, resize_to: Tuple[int,int]=(128,128)):
        self.data_root = Path(data_root)
        self.resize_to = resize_to
        with open(split_file, 'r') as f:
            case_names = [ln.strip() for ln in f if ln.strip()]
        target_dirs = {f'Brain_{name.split("_")[-1]}' for name in case_names if name}
        self.samples = []
        for case_dir in sorted(self.data_root.iterdir()):
            if not case_dir.is_dir() or case_dir.name not in target_dirs:
                continue
            mask_dir = case_dir / 'mask'
            if not mask_dir.exists():
                continue
            for mask_file in sorted(mask_dir.glob('*.png')):
                sid = mask_file.stem.split('_')[-1]
                paths = {
                    'flair': case_dir/'flair'/f'flair_{sid}.png',
                    't1':    case_dir/'t1'/f't1_{sid}.png',
                    't1ce':  case_dir/'t1ce'/f't1ce_{sid}.png',
                    't2':    case_dir/'t2'/f't2_{sid}.png',
                    'mask':  mask_file
                }
                if all(p.exists() for p in paths.values()):
                    self.samples.append(paths)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        chans = []
        for mod in ['flair','t1','t1ce','t2']:
            im = Image.open(sample[mod]).convert('L')
            if self.resize_to:
                im = im.resize(self.resize_to, Image.BILINEAR)
            arr = np.array(im, dtype=np.float32) / 255.0
            chans.append(arr)
        image = np.stack(chans, axis=0)
        image_tensor = torch.from_numpy(image).float()

        mask_pil = Image.open(sample['mask'])
        if self.resize_to:
            mask_pil = mask_pil.resize(self.resize_to, Image.NEAREST)
        mask_raw = np.array(mask_pil, dtype=np.uint8)

        wt_mask = np.isin(mask_raw, [1, 2, 4]).astype(np.float32)
        tc_mask = np.isin(mask_raw, [1, 4]).astype(np.float32)
        et_mask = (mask_raw == 4).astype(np.float32)

        mask = np.stack([wt_mask, tc_mask, et_mask], axis=0)
        mask_tensor = torch.from_numpy(mask).float()

        return image_tensor, mask_tensor


def get_dataloaders(data_root: Path, config_dir: Path, batch_size: int = 8, resize_to: Tuple[int,int]=(128,128), num_workers: int = 4):
    train_file = config_dir / 'train.txt'
    val_file = config_dir / 'val.txt'
    train_ds = Brats2dDataset(data_root, train_file, resize_to)
    val_ds   = Brats2dDataset(data_root, val_file, resize_to)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    logger.info(f'Train samples: {len(train_ds)}, Val samples: {len(val_ds)}')
    return train_loader, val_loader

# %%
# ============================================================================
# 4. UNet++ MODEL
# ============================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=4, num_classes=3, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = ConvBlock(in_channels, nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4])

        self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = ConvBlock(nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = ConvBlock(nb_filter[1]*2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = ConvBlock(nb_filter[2]*2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = ConvBlock(nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = ConvBlock(nb_filter[1]*3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = ConvBlock(nb_filter[0]*4 + nb_filter[1], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            return self.final(x0_4)


# %%
# ============================================================================
# 5. ### CẬP NHẬT ### LOSS FUNCTION
# ============================================================================
class DiceLossMultiLabel(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.shape[0], probs.shape[1], -1)
        targets_flat = targets.view(targets.shape[0], targets.shape[1], -1)
        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        dice_per_channel = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_per_channel.mean()

class DiceCELoss(nn.Module):
    """
    Hàm loss kết hợp Dice Loss và Cross Entropy Loss bằng cách LẤY TỔNG.
    Công thức: loss = dice_loss + ce_loss
    
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.dice_loss = DiceLossMultiLabel(smooth=smooth)
        self.ce_loss = nn.BCEWithLogitsLoss()
        logger.info("Initialized DiceCELoss (Sum of Dice and Cross Entropy for Multi-Label)")

    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        ce = self.ce_loss(logits, targets)
        # Trả về tổng của hai loss
        return dice + ce

# %%
# ============================================================================
# 6. METRICS
# ============================================================================
def _calculate_binary_dice(pred, target, smooth=1e-6):
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def _calculate_binary_iou(pred, target, smooth=1e-6):
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def _calculate_hd95(pred_tensor: torch.Tensor, target_tensor: torch.Tensor) -> float:
    if medpy_hd95 is None: return 0.0
    pred_np, target_np = pred_tensor.cpu().numpy().astype(bool), target_tensor.cpu().numpy().astype(bool)
    H, W = pred_np.shape
    max_dist = np.sqrt(H**2 + W**2)
    if not pred_np.any() and not target_np.any(): return 0.0
    if not pred_np.any() or not target_np.any(): return max_dist
    try: return medpy_hd95(pred_np, target_np)
    except Exception: return max_dist

def _calculate_assd(pred_tensor: torch.Tensor, target_tensor: torch.Tensor) -> float:
    if medpy_assd is None: return 0.0
    pred_np, target_np = pred_tensor.cpu().numpy().astype(bool), target_tensor.cpu().numpy().astype(bool)
    H, W = pred_np.shape
    max_dist = np.sqrt(H**2 + W**2)
    if not pred_np.any() and not target_np.any(): return 0.0
    if not pred_np.any() or not target_np.any(): return max_dist
    try: return medpy_assd(pred_np, target_np)
    except Exception: return max_dist

# %%
# ============================================================================
# 7. CHECKPOINT HELPERS
# ============================================================================
def save_checkpoint(state: dict, path: Path):
    torch.save(state, str(path))
    logger.info(f"Saved checkpoint: {path}")

def load_checkpoint(path: Path, device=None):
    ckpt = torch.load(str(path), map_location=device)
    logger.info(f"Loaded checkpoint: {path}")
    return ckpt


# %%
# ============================================================================
# 8. TRAIN / VALIDATE LOOPS
# ============================================================================
def train_epoch_multilabel(model, loader, optimizer, criterion, scaler: GradScaler, device):
    model.train()
    running_loss = 0.0
    totals = {
        'dice_wt': 0.0, 'iou_wt': 0.0, 'dice_tc': 0.0,
        'iou_tc': 0.0, 'dice_et': 0.0, 'iou_et': 0.0,
    }
    n_samples = 0
    region_map = {0: 'wt', 1: 'tc', 2: 'et'}

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss = 0
            for output in outputs:
                loss += criterion(output, masks)
            loss /= len(outputs)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        with torch.no_grad():
            final_output = outputs[-1]
            preds = (torch.sigmoid(final_output) > 0.5).float()
            
            for i in range(batch_size):
                pred_i, mask_i = preds[i], masks[i]
                for c in range(3):
                    region = region_map[c]
                    pred_c = pred_i[c]
                    mask_c = mask_i[c]
                    totals[f'dice_{region}'] += _calculate_binary_dice(pred_c, mask_c).item()
                    totals[f'iou_{region}']  += _calculate_binary_iou(pred_c, mask_c).item()

    if n_samples == 0: return {}

    metrics = {'train_loss': running_loss / n_samples}
    for key, value in totals.items():
        metrics[f'train_{key}'] = value / n_samples

    avg_dice = (metrics['train_dice_wt'] + metrics['train_dice_tc'] + metrics['train_dice_et']) / 3.0
    avg_iou = (metrics['train_iou_wt'] + metrics['train_iou_tc'] + metrics['train_iou_et']) / 3.0
    metrics['train_dice_avg'] = avg_dice
    metrics['train_iou_avg'] = avg_iou

    pbar.set_postfix({'loss': f"{metrics['train_loss']:.4f}", 'avg_dice': f"{avg_dice:.4f}"})
    return metrics

@torch.no_grad()
def validate_epoch_multilabel(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    totals = {
        'dice_wt': 0.0, 'iou_wt': 0.0, 'dice_tc': 0.0,
        'iou_tc': 0.0, 'dice_et': 0.0, 'iou_et': 0.0,
    }
    n_samples = 0
    region_map = {0: 'wt', 1: 'tc', 2: 'et'}

    pbar = tqdm(loader, desc="Val", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast(enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss = 0
            for output in outputs:
                loss += criterion(output, masks)
            loss /= len(outputs)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size
        
        final_output = outputs[-1]
        preds = (torch.sigmoid(final_output) > 0.5).float()

        for i in range(batch_size):
            pred_i, mask_i = preds[i], masks[i]
            for c in range(3):
                region = region_map[c]
                pred_c = pred_i[c]
                mask_c = mask_i[c]
                totals[f'dice_{region}'] += _calculate_binary_dice(pred_c, mask_c).item()
                totals[f'iou_{region}']  += _calculate_binary_iou(pred_c, mask_c).item()

    if n_samples == 0: return {}

    metrics = {'val_loss': running_loss / n_samples}
    for key, value in totals.items():
        metrics[f'val_{key}'] = value / n_samples

    avg_dice = (metrics['val_dice_wt'] + metrics['val_dice_tc'] + metrics['val_dice_et']) / 3.0
    avg_iou = (metrics['val_iou_wt'] + metrics['val_iou_tc'] + metrics['val_iou_et']) / 3.0
    metrics['val_dice_avg'] = avg_dice
    metrics['val_iou_avg'] = avg_iou

    return metrics


# %%
# ============================================================================
# 9. ### CẬP NHẬT ### TRAINING MAIN FUNCTION
# ============================================================================
def train_model_main(
    data_root: Path, config_dir: Path, ckpt_dir: Path, log_dir: Path,
    num_epochs: int = 10, batch_size: int = 8, lr: float = 1e-4,
    resize_to: Tuple[int,int] = (128,128), num_workers: int = 4,
    deep_supervision: bool = True,
    resume_from: Optional[Path] = None, device: Optional[torch.device] = None
):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    logger.info(f'Using device: {device}')

    train_loader, val_loader = get_dataloaders(data_root, config_dir, batch_size, resize_to, num_workers)
    
    model = UNetPlusPlus(
        in_channels=4, 
        num_classes=3, 
        deep_supervision=deep_supervision
    ).to(device)
    logger.info(f"UNet++ Deep Supervision: {'ON' if deep_supervision else 'OFF'}")
    
    # ### CẬP NHẬT ###: Sử dụng hàm loss kết hợp mới
    criterion = DiceCELoss()
    
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    logger.info(f"Automatic Mixed Precision (AMP) enabled: {scaler.is_enabled()}")

    start_epoch = 0
    best_dice_avg = -1.0

    csv_file = log_dir / "metrics_log.csv"
    csv_headers = [
        "epoch", "train_loss", "train_dice_avg", "train_iou_avg",
        "train_dice_wt", "train_dice_tc", "train_dice_et",
        "train_iou_wt", "train_iou_tc", "train_iou_et",
        "val_loss", "val_dice_avg", "val_iou_avg",
        "val_dice_wt", "val_dice_tc", "val_dice_et",
        "val_iou_wt", "val_iou_tc", "val_iou_et",
    ]
    if not csv_file.exists():
        with open(csv_file, mode='w', newline='') as f:
            csv.writer(f).writerow(csv_headers)

    json_file = log_dir / "metrics_log.json"
    if not json_file.exists():
        with open(json_file, 'w') as f: json.dump([], f, indent=2)

    if resume_from and Path(resume_from).exists():
        ck = load_checkpoint(Path(resume_from), device)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        if 'scheduler' in ck: scheduler.load_state_dict(ck['scheduler'])
        start_epoch = ck.get('epoch', 0) + 1
        best_dice_avg = ck.get('best_dice_avg', best_dice_avg)
        logger.info(f"Resumed from {resume_from} at epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        train_metrics = train_epoch_multilabel(model, train_loader, optimizer, criterion, scaler, device)
        logger.info(f"[TRAIN] Loss={train_metrics.get('train_loss', 0):.4f} | Avg Dice={train_metrics.get('train_dice_avg', 0):.4f} | Avg IoU={train_metrics.get('train_iou_avg', 0):.4f}")
        logger.info(f"[TRAIN-WT] Dice={train_metrics.get('train_dice_wt', 0):.4f} | [TRAIN-TC] Dice={train_metrics.get('train_dice_tc', 0):.4f} | [TRAIN-ET] Dice={train_metrics.get('train_dice_et', 0):.4f}")

        val_metrics = validate_epoch_multilabel(model, val_loader, criterion, device)
        if val_metrics:
            logger.info(f"[VAL]   Loss={val_metrics.get('val_loss', 0):.4f} | Avg Dice={val_metrics.get('val_dice_avg', 0):.4f} | Avg IoU={val_metrics.get('val_iou_avg', 0):.4f}")
            logger.info(f"[VAL-WT]   Dice={val_metrics.get('val_dice_wt', 0):.4f} | [VAL-TC]   Dice={val_metrics.get('val_dice_tc', 0):.4f} | [VAL-ET]   Dice={val_metrics.get('val_dice_et', 0):.4f}")

        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            all_metrics = {**train_metrics, **val_metrics}
            row = [all_metrics.get(h, 0) for h in csv_headers[1:]]
            writer.writerow([epoch + 1] + row)

        epoch_data = {"epoch": epoch + 1, "train_metrics": train_metrics, "val_metrics": val_metrics}
        with open(json_file, 'r+') as f:
            data = json.load(f); data.append(epoch_data); f.seek(0); json.dump(data, f, indent=2)

        current_val_dice_avg = val_metrics.get('val_dice_avg', 0)
        if current_val_dice_avg > best_dice_avg:
            best_dice_avg = current_val_dice_avg
            best_ckpt_path = Path(ckpt_dir) / "best_checkpoint.pth"
            state_best = {
                'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'best_dice_avg': best_dice_avg, 'val_metrics': val_metrics
            }
            save_checkpoint(state_best, best_ckpt_path)
            logger.info(f"New best Avg Val Dice: {best_dice_avg:.4f}, saved to {best_ckpt_path}")

        scheduler.step()

        ckpt_path = Path(ckpt_dir) / f"checkpoint_epoch_{epoch+1:03d}.pth"
        state = {
            'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'best_dice_avg': best_dice_avg, 'val_metrics': val_metrics
        }
        save_checkpoint(state, ckpt_path)

# %%
# ============================================================================
# 10. CONFIG & RUN
# ============================================================================
CONFIG = {
    'data_root': DATA_ROOT,
    'config_dir': CONFIG_DIR,
    'ckpt_dir': CKPT_DIR,
    'log_dir': LOG_DIR,
    'num_epochs': 28,
    'batch_size': 8,
    'learning_rate': 1e-4,
    'resize_to': (128,128),
    'num_workers': 4,
    'deep_supervision': True,
    'resume_from': r"D:\Project Advanced CV\checkpoints\unetpp_2d_multilabel_ds_dice_ce_sum\checkpoint_epoch_028.pth"
}

if __name__ == '__main__':
    logger.info("Training Configuration:")
    logger.info(json.dumps({k: str(v) if isinstance(v, Path) else v for k,v in CONFIG.items()}, indent=2))

    train_model_main(
        data_root=CONFIG['data_root'],
        config_dir=CONFIG['config_dir'],
        ckpt_dir=CONFIG['ckpt_dir'],
        log_dir=CONFIG['log_dir'],
        num_epochs=CONFIG['num_epochs'],
        batch_size=CONFIG['batch_size'],
        lr=CONFIG['learning_rate'],
        resize_to=CONFIG['resize_to'],
        num_workers=CONFIG['num_workers'],
        deep_supervision=CONFIG['deep_supervision'],
        resume_from=CONFIG['resume_from']
    )