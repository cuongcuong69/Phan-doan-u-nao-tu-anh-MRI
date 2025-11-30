# ============================================================================
# FILE: evaluate.py
# MÔ TẢ: File độc lập để chạy inference và đánh giá 3D trên tập test.
#        Tương thích với model UNet++ đa nhãn (3 kênh) đã được huấn luyện.
# HƯỚNG DẪN:
# 1. Lưu file này với tên evaluate.py.
# 2. Sửa lại các đường dẫn trong mục "ĐỊNH NGHĨA CÁC ĐƯỜNG DẪN CỐ ĐỊNH".
# 3. Chạy file từ terminal: python evaluate.py
# ============================================================================

# ============================================================================
# 1. IMPORTS
# ============================================================================
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Tuple
from tqdm import tqdm
import cv2
from PIL import Image
import logging

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Metrics từ medpy
try:
    from medpy.metric.binary import hd95 as medpy_hd95, assd as medpy_assd
except ImportError:
    medpy_hd95, medpy_assd = None, None
    print("Warning: medpy not available. Install it with `pip install medpy` to compute HD95/ASSD.")

# ============================================================================
# 2. ### ĐỊNH NGHĨA CÁC ĐƯỜNG DẪN CỐ ĐỊNH ###
# Sửa lại các đường dẫn này cho phù hợp với máy của bạn
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # Giả sử file này nằm trong thư mục con của project
DATA_ROOT = PROJECT_ROOT / 'data' / 'processed' / '2d' / 'labeled'
CONFIG_DIR = PROJECT_ROOT / 'configs' / 'splits_2d'
LOG_DIR = Path(r'D:/Project Advanced CV/logs/unetpp_2d_multilabel_ds_dice_ce_sum')
CKPT_DIR = Path(r'D:/Project Advanced CV/checkpoints/unetpp_2d_multilabel_ds_dice_ce_sum')

# Đảm bảo thư mục log tồn tại để lưu kết quả
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 3. LOGGER
# ============================================================================
def setup_logger(log_dir: Path, name: str) -> logging.Logger:
    log_file = log_dir / 'evaluation_log.txt'
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
    logger.info(f'Logger initialized for evaluation. Log file: {log_file}')
    return logger

logger = setup_logger(LOG_DIR, name='unetpp_evaluation')

# ============================================================================
# 4. ĐỊNH NGHĨA KIẾN TRÚC MODEL (BẮT BUỘC)
# Phải giống hệt với kiến trúc đã dùng để huấn luyện.
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

# ============================================================================
# 5. DATASET CHO TẬP TEST
# ============================================================================
class Brats2dTestDataset(Dataset):
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
                    'flair': case_dir/'flair'/f'flair_{sid}.png', 't1': case_dir/'t1'/f't1_{sid}.png',
                    't1ce': case_dir/'t1ce'/f't1ce_{sid}.png', 't2': case_dir/'t2'/f't2_{sid}.png',
                    'mask': mask_file
                }
                if all(p.exists() for p in paths.values()):
                    self.samples.append({'paths': paths, 'patient_id': case_dir.name})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        paths = sample['paths']
        patient_id = sample['patient_id']

        chans = []
        for mod in ['flair','t1','t1ce','t2']:
            im = Image.open(paths[mod]).convert('L')
            if self.resize_to:
                im = im.resize(self.resize_to, Image.BILINEAR)
            arr = np.array(im, dtype=np.float32) / 255.0
            chans.append(arr)
        image = np.stack(chans, axis=0)
        image = torch.from_numpy(image).float()

        mask_pil = Image.open(paths['mask'])
        mask_arr = np.array(mask_pil, dtype=np.uint8)
        
        return image, mask_arr, patient_id

# ============================================================================
# 6. HÀM TÍNH METRIC 3D
# ============================================================================
def _compute_single_region_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict:
    metrics = {}
    intersection = np.logical_and(pred_mask, true_mask).sum()
    pred_sum = pred_mask.sum()
    true_sum = true_mask.sum()

    if pred_sum + true_sum == 0:
        metrics['dice'] = 1.0
        metrics['iou'] = 1.0
    else:
        metrics['dice'] = (2. * intersection) / (pred_sum + true_sum) if (pred_sum + true_sum) > 0 else 0.0
        iou_den = pred_sum + true_sum - intersection
        metrics['iou'] = intersection / iou_den if iou_den > 0 else 0.0
    
    if medpy_assd is None or medpy_hd95 is None:
        metrics['asd'], metrics['hd95'] = float('nan'), float('nan')
    else:
        if not np.any(pred_mask) and not np.any(true_mask):
            metrics['asd'], metrics['hd95'] = 0.0, 0.0
        elif not np.any(pred_mask) or not np.any(true_mask):
            metrics['asd'], metrics['hd95'] = float('inf'), float('inf')
        else:
            try:
                metrics['asd'] = medpy_assd(pred_mask, true_mask)
                metrics['hd95'] = medpy_hd95(pred_mask, true_mask)
            except Exception:
                metrics['asd'], metrics['hd95'] = float('inf'), float('inf')
    return metrics

def calculate_3d_metrics_for_patient(pred_vol: np.ndarray, true_vol: np.ndarray) -> dict:
    # Nhãn gốc: 1=NCR/NET, 2=ED, 4=ET
    pred_wt = np.isin(pred_vol, [1, 2, 4])
    true_wt = np.isin(true_vol, [1, 2, 4])
    
    pred_tc = np.isin(pred_vol, [1, 4])
    true_tc = np.isin(true_vol, [1, 4])
    
    pred_et = (pred_vol == 4)
    true_et = (true_vol == 4)
    
    metrics_wt = _compute_single_region_metrics(pred_wt, true_wt)
    metrics_tc = _compute_single_region_metrics(pred_tc, true_tc)
    metrics_et = _compute_single_region_metrics(pred_et, true_et)
    
    final_metrics = {}
    for k in metrics_wt.keys():
        final_metrics[f'{k}_wt'] = metrics_wt[k]
        final_metrics[f'{k}_tc'] = metrics_tc[k]
        final_metrics[f'{k}_et'] = metrics_et[k]
    return final_metrics

# ============================================================================
# 7. HÀM ĐÁNH GIÁ CHÍNH
# ============================================================================
@torch.no_grad()
def evaluate_on_test_set(
    model, 
    data_root: Path, 
    config_dir: Path, 
    batch_size: int = 8, 
    resize_input: Tuple[int,int] = (128,128),
    resize_metric: Tuple[int,int] = (240,240),
    device: torch.device = torch.device('cpu')
):
    model.to(device)
    model.eval()

    test_file = config_dir / 'test.txt'
    test_ds = Brats2dTestDataset(data_root, test_file, resize_input)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    logger.info(f"Bắt đầu inference trên {len(test_ds.samples)} lát cắt từ tập test...")

    preds_by_patient = defaultdict(list)
    masks_by_patient = defaultdict(list)
    
    pbar = tqdm(test_loader, desc="Inference trên các lát cắt 2D")
    for images, masks_np, patient_ids in pbar:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        
        final_output = outputs[-1] if isinstance(outputs, list) else outputs
        preds_3_channels = (torch.sigmoid(final_output) > 0.5).cpu().numpy()

        for i in range(len(patient_ids)):
            patient_id = patient_ids[i]
            
            pred_wt, pred_tc, pred_et = preds_3_channels[i, 0], preds_3_channels[i, 1], preds_3_channels[i, 2]
            
            reconstructed_pred = np.zeros_like(pred_wt, dtype=np.uint8)
            reconstructed_pred[pred_wt == 1] = 2
            reconstructed_pred[pred_tc == 1] = 1
            reconstructed_pred[pred_et == 1] = 4
            
            pred_resized = cv2.resize(reconstructed_pred, resize_metric, interpolation=cv2.INTER_NEAREST)
            mask_resized = cv2.resize(masks_np[i], resize_metric, interpolation=cv2.INTER_NEAREST)
            
            preds_by_patient[patient_id].append(pred_resized)
            masks_by_patient[patient_id].append(mask_resized)

    logger.info("Hoàn thành inference. Đang xếp chồng các lát cắt và tính toán metric 3D...")
    
    results = []
    patient_ids = sorted(preds_by_patient.keys())
    
    for patient_id in tqdm(patient_ids, desc="Đánh giá trên từng khối 3D"):
        pred_slices = preds_by_patient[patient_id]
        mask_slices = masks_by_patient[patient_id]
        
        pred_volume = np.stack(pred_slices, axis=0)
        mask_volume = np.stack(mask_slices, axis=0)

        patient_metrics = calculate_3d_metrics_for_patient(pred_volume, mask_volume)
        patient_metrics['patient_id'] = patient_id
        results.append(patient_metrics)

    return pd.DataFrame(results)

# ============================================================================
# 8. MAIN EXECUTION BLOCK
# ============================================================================
if __name__ == '__main__':
    INFERENCE_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BEST_CKPT_PATH = CKPT_DIR / "best_checkpoint.pth"

    if not BEST_CKPT_PATH.exists():
        logger.error(f"LỖI: Không tìm thấy file checkpoint tại '{BEST_CKPT_PATH}'")
        exit()

    logger.info(f"Đang tải model từ: {BEST_CKPT_PATH}")
    
    model = UNetPlusPlus(
        in_channels=4, 
        num_classes=3,
        deep_supervision=True
    )
    
    checkpoint = torch.load(BEST_CKPT_PATH, map_location=INFERENCE_DEVICE)
    model.load_state_dict(checkpoint['model'])

    results_df = evaluate_on_test_set(
        model=model,
        data_root=DATA_ROOT,
        config_dir=CONFIG_DIR,
        batch_size=16,
        resize_input=(128,128),
        resize_metric=(240,240),
        device=INFERENCE_DEVICE
    )

    # ============================================================================
    # 9. HIỂN THỊ VÀ LƯU KẾT QUẢ
    # ============================================================================
    logger.info("\n" + "="*50)
    logger.info(" KẾT QUẢ ĐÁNH GIÁ CHI TIẾT TRÊN TẬP TEST (3D METRICS) ".center(50, "="))
    logger.info("="*50)
    
    cols_order = ['patient_id'] + sorted([c for c in results_df.columns if c != 'patient_id'])
    results_df = results_df[cols_order]
    print(results_df.to_string())

    logger.info("\n" + "="*50)
    logger.info(" CÁC CHỈ SỐ TRUNG BÌNH TRÊN TẬP TEST ".center(50, "="))
    logger.info("="*50)
    
    mean_metrics = results_df.replace([np.inf, -np.inf], np.nan).mean(numeric_only=True)
    print(mean_metrics)

    results_csv_path = LOG_DIR / "test_set_3d_metrics_final.csv"
    results_df.to_csv(results_csv_path, index=False, float_format='%.5f')
    logger.info(f"\nĐã lưu kết quả chi tiết vào: {results_csv_path}")