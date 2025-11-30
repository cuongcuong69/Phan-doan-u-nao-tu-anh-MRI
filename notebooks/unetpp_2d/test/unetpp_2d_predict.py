# ============================================================================
# FILE: visualize.py (PHIÊN BẢN CUỐI - TÙY CHỈNH THRESHOLD)
# ============================================================================

# ============================================================================
# 1. IMPORTS
# ============================================================================
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
from PIL import Image
import logging
import warnings

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, Normalize

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# 2. ĐỊNH NGHĨA CÁC ĐƯỜNG DẪN CỐ ĐỊNH
# Sửa lại các đường dẫn này cho phù hợp với máy của bạn
# ============================================================================
DATA_ROOT = Path(r'D:\Project Advanced CV\data\processed\2d\labeled')
CKPT_DIR = Path(r'D:\Project Advanced CV\checkpoints\unetpp_2d_multilabel_ds_dice_ce_sum')

# ============================================================================
# 3. LOGGER
# ============================================================================
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# 4. ĐỊNH NGHĨA KIẾN TRÚC MODEL (PHẢI GIỐNG HỆT LÚC HUẤN LUYỆN)
# ============================================================================
class ConvBlock(nn.Module): # ... (Giữ nguyên định nghĩa model UNet++ như cũ)
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

class UNetPlusPlus(nn.Module): # ... (Giữ nguyên định nghĩa model UNet++ như cũ)
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
            return [self.final1(x0_1), self.final2(x0_2), self.final3(x0_3), self.final4(x0_4)]
        else:
            return self.final(x0_4)

# ============================================================================
# 5. HÀM HIỂN THỊ KẾT QUẢ
# ============================================================================
def visualize_multiple_slices(
    model, 
    data_root: Path,
    patient_id: str,
    slice_indices: list,
    ### THAY ĐỔI 1: Thêm tham số thresholds ###
    thresholds: Dict[str, float],
    resize_to: Tuple[int,int] = (128,128),
    device: torch.device = torch.device('cpu')
):
    plt.style.use('default')
    model.to(device)
    model.eval()

    patient_dir = data_root / patient_id
    if not patient_dir.exists():
        logger.error(f"Thư mục '{patient_id}' không tồn tại: {patient_dir}")
        return

    # ... (code tải dữ liệu giữ nguyên)
    mask_files = sorted(list((patient_dir / 'mask').glob('*.png')), key=lambda p: int(p.stem.split('_')[-1]))
    flair_files = sorted(list((patient_dir / 'flair').glob('*.png')), key=lambda p: int(p.stem.split('_')[-1]))
    valid_slice_indices = [idx for idx in slice_indices if 0 <= idx < len(mask_files)]
    if not valid_slice_indices:
        logger.error(f"Chỉ số lát cắt không hợp lệ.")
        return
    num_slices = len(valid_slice_indices)
    fig, axes = plt.subplots(num_slices, 3, figsize=(15, 5.5 * num_slices), squeeze=False)
    fig.set_facecolor('white')


    for i, slice_idx in enumerate(valid_slice_indices):
        sid = mask_files[slice_idx].stem.split('_')[-1]
        
        chans = []
        for mod in ['flair', 't1', 't1ce', 't2']:
            im = Image.open(patient_dir/mod/f'{mod}_{sid}.png').convert('L').resize(resize_to, Image.BILINEAR)
            chans.append(np.array(im, dtype=np.float32) / 255.0)
        image_tensor = torch.from_numpy(np.stack(chans, axis=0)).float()
        
        flair_img = np.array(Image.open(flair_files[slice_idx]).convert('L').resize(resize_to, Image.BILINEAR))
        true_mask_raw = np.array(Image.open(mask_files[slice_idx]).resize(resize_to, Image.NEAREST))

        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0).to(device))
            final_output = output[-1] if isinstance(output, list) else output

            ### THAY ĐỔI 2: Áp dụng ngưỡng riêng cho từng kênh ###
            # 1. Lấy xác suất (probabilities) thay vì mask nhị phân ngay lập tức
            probs = torch.sigmoid(final_output).squeeze(0).cpu().numpy()
            
            # 2. Áp dụng ngưỡng riêng cho từng kênh
            pred_wt = (probs[0] > thresholds['wt']).astype(np.uint8)
            pred_tc = (probs[1] > thresholds['tc']).astype(np.uint8)
            pred_et = (probs[2] > thresholds['et']).astype(np.uint8) # <--- Ngưỡng riêng cho ET

            # 3. Tái tạo mask đa lớp (giữ nguyên)
            pred_mask_reconstructed = np.zeros_like(pred_wt, dtype=np.uint8)
            pred_mask_reconstructed[pred_wt == 1] = 2  # Lớp 2 (ED)
            pred_mask_reconstructed[pred_tc == 1] = 1  # Lớp 1 (NCR/NET)
            pred_mask_reconstructed[pred_et == 1] = 4  # Lớp 4 (ET)
        
        # ... (phần code hiển thị giữ nguyên)
        ax_flair, ax_gt, ax_pred = axes[i, 0], axes[i, 1], axes[i, 2]
        colors = {0: '#000000', 1: '#FFFF00', 2: '#00FF00', 4: '#FF0000'}
        cmap = ListedColormap([colors[key] for key in sorted(colors) if key in [0, 1, 2, 4]])
        norm = Normalize(vmin=0, vmax=max(colors.keys()))
        ax_flair.imshow(flair_img, cmap='gray')
        ax_flair.set_title(f'FLAIR (Slice {slice_idx})')
        ax_gt.imshow(flair_img, cmap='gray')
        ax_gt.imshow(np.ma.masked_where(true_mask_raw == 0, true_mask_raw), cmap=cmap, norm=norm, alpha=0.6)
        ax_gt.set_title(f'Ground Truth (Slice {slice_idx})')
        ax_pred.imshow(flair_img, cmap='gray')
        ax_pred.imshow(np.ma.masked_where(pred_mask_reconstructed == 0, pred_mask_reconstructed), cmap=cmap, norm=norm, alpha=0.6)
        ax_pred.set_title(f'Prediction (Slice {slice_idx})')
        for ax in [ax_flair, ax_gt, ax_pred]:
            ax.set_xticks([]); ax.set_yticks([])

    # ... (phần code legend giữ nguyên)
    legend_patches = [
        mpatches.Patch(color='#FF0000', label='Enhancing Tumor (ET) - Lớp 4'),
        mpatches.Patch(color='#FFFF00', label='Necrotic Core (NCR/NET) - Lớp 1'),
        mpatches.Patch(color='#00FF00', label='Peritumoral Edema (ED) - Lớp 2'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=12)
    fig.suptitle(f'Kết quả Inference cho Bệnh nhân: {patient_id}', fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.show()

# ============================================================================
# 6. MAIN EXECUTION BLOCK
# ============================================================================
if __name__ == '__main__':
    # --- CẤU HÌNH ---
    PATIENT_ID_TO_VISUALIZE = "Brain_026"
    SLICE_INDICES_TO_VISUALIZE = [65] 
    
    ### THAY ĐỔI 3: Định nghĩa các ngưỡng ở đây ###
    # Thử nghiệm với các giá trị khác nhau cho 'et'
    # Bắt đầu với 0.3, nếu ra nhiều nhiễu quá thì tăng lên 0.4
    # Nếu vẫn không thấy gì thì giảm xuống 0.2
    THRESHOLDS_FOR_VIS = {
        'wt': 0.5,
        'tc': 0.5,
        'et': 1 # <-- HÃY THỬ NGHIỆM VỚI GIÁ TRỊ NÀY
    }
    
    INFERENCE_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BEST_CKPT_PATH = CKPT_DIR / "best_checkpoint.pth"

    if not BEST_CKPT_PATH.exists():
        logger.error(f"LỖI: Không tìm thấy file checkpoint tại '{BEST_CKPT_PATH}'")
    else:
        logger.info(f"Đang tải model từ: {BEST_CKPT_PATH}")
        logger.info(f"Sử dụng ngưỡng: WT={THRESHOLDS_FOR_VIS['wt']}, TC={THRESHOLDS_FOR_VIS['tc']}, ET={THRESHOLDS_FOR_VIS['et']}")
        
        viz_model = UNetPlusPlus(in_channels=4, num_classes=3, deep_supervision=True)
        checkpoint = torch.load(BEST_CKPT_PATH, map_location=INFERENCE_DEVICE)
        viz_model.load_state_dict(checkpoint['model'])

        ### THAY ĐỔI 4: Truyền các ngưỡng vào hàm ###
        visualize_multiple_slices(
            model=viz_model,
            data_root=DATA_ROOT,
            patient_id=PATIENT_ID_TO_VISUALIZE,
            slice_indices=SLICE_INDICES_TO_VISUALIZE,
            thresholds=THRESHOLDS_FOR_VIS, # <--- Truyền vào đây
            resize_to=(128,128),
            device=INFERENCE_DEVICE
        )