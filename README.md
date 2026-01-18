# Advanced CV - Brain Tumor Segmentation

# 📌 Phân Chia Công Việc Nhóm

## 👥 Thành viên & Nhiệm vụ

| **Thành viên**      | **Nhiệm vụ phụ trách**                |
| ------------------- | ------------------------------------- |
| **Lê Mạnh Cương**   | - Tiền xử lý dữ liệu                  |
|                     | - Thống kê và phân tích dữ liệu (EDA) |
|                     | - Mô hình **VNet, Swin UNet3D**       |
| **Nguyễn Tuấn Anh** | - Mô hình **UNet, TransUNet**         |
| **Phạm Quý Đô**     | - Mô hình **UNet++, UNetr3D**         |

---

## 📂 Bộ dữ liệu sử dụng

- **BraTS 2020 Dataset**  
  🔗 Link tải: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation

---

## Cách tổ chức các thư mục

- `configs/`: cấu hình split dữ liệu; `configs/splits_2d/` có `train.txt`, `val.txt`, `test.txt`, `train_small.txt` và file `splits.csv/json`; `configs/splits_task01/test.txt` cho Task01.
- `data/`: dữ liệu thô và đã tiền xử lý (`BraST2020/`, `processed/`, `processed_task01/`); dataloader 2D/3D: `dataloader_brats3d_sup.py`, `dataloader_brats3d_full.py`, `dataloader_brats3d_task01_sup.py`, `dataloader_brats2d.py`, `dataloader_2d.py`.
- `experiments/`: kết quả theo từng `EXP_NAME` (checkpoint, log, metrics, ảnh trực quan, inference); có `experiments/eda/` và các thư mục `vis_*`.
- `inference/`: kịch bản suy luận VNet & Swin-UNet 3D; `inference/experiments/` dùng lưu thử nghiệm 2D.
- `losses/`: loss/metrics & tiện ích (`losses.py`, `losses_2.py`, `metrics.py`, `composite.py`, `functional.py`, `util.py`, `ramps.py`).
- `models/`: kiến trúc VNet 3D, multi-head, multi-encoder và Swin-UNet 3D (`vnet.py`, `vnet_multihead.py`, `vnet_multi_enc_fusion.py`, `swin_unet_3d.py`).
- `scripts/`: tiền xử lý, tạo split và EDA; `scripts/eda/` chứa các thống kê; `visualize_vnetmh_results.py` trực quan kết quả multi-head.
- `trainers/`: kịch bản huấn luyện VNet 3D và Swin-UNet 3D (`train_vnet_brats3d_sup.py`, `train_vnet_full.py`, `train_vnetmh_brats3d_sup.py`, `train_vnet_multienc_brats3d_sup.py`, `train_swin_unet3d_patch.py`).
- `notebooks/`: thực nghiệm/ghi chú (UNet 2D, UNet++ 2D, trực quan 3D).
- `logs/`: log/metrics xuất từ các thực nghiệm 2D.
- `wandb/` và `trainers/wandb/`: log offline/online W&B.
- `requirements.txt`: danh sách package.

## Dữ liệu & tiền xử lý

- Đầu vào BraTS 2020 nằm ở `data/BraST2020/BraTS2020_TrainingData` và `.../BraTS2020_ValidationData`. Bộ Task01_BrainTumour nằm ở `data/Task01_BrainTumour`.
- Xuất 2D:
  - `scripts/preprocess_brats2d_version2.py`: chuẩn hóa RAS, tìm bbox toàn cục trên T1, cắt lưới axial, crop theo bbox vuông, chuẩn hóa percentile về [0,1], tùy chọn xoay/flip, resize 256x256, remap nhãn 4→3, lưu PNG uint8. Đầu ra: `data/processed/2d/{labeled,unlabeled}/Brain_xxx/{flair,t1,t1ce,t2,mask}/...`.
  - `scripts/preprocess_brats2d.py`: phiên bản đơn giản hơn (crop thủ công, augment ít hơn).
- Xuất 3D BraTS:
  - `scripts/preprocess_brats3d.py`: reorient RAS, crop không gian cố định (x:22–216, y:16–210), chuẩn hóa cường độ (minmax hoặc z-score trên voxel >0), remap nhãn 4→3, lưu NIfTI vào `data/processed/3d/{labeled,unlabeled}/Brain_xxx/{modality}.nii.gz`.
  - `scripts/preprocess_brats3d_version2.py` (nếu cần bounding box động) có bổ sung cache bbox và augment nhẹ trên lát.
- Xuất cho Task01_BrainTumour:
  - `scripts/preprocess_task01_brain3d.py`: xử lý 3D tương tự (RAS, crop, chuẩn hóa, remap nhãn), đầu ra `data/processed_task01/3d/...`.
  - `scripts/preprocess_task01_2d.py`: tương tự pipeline 2D cho Task01, lưu PNG ở `data/processed_task01/2d/...`.
- Phân tích/EDA: các script trong `scripts/eda/` tạo biểu đồ và thống kê (cohort, intensity per modality, radiomics texture, shape, volume). Kết quả được lưu trong `experiments/eda/...`.

## Chia tập

- `scripts/make_split.py`: tạo split 70/15/15 cho dữ liệu 2D BraTS đã tiền xử lý. Stratify theo Grade (từ `name_mapping.csv`), sự hiện diện vùng ET và kích thước khối u. Đầu ra: `configs/splits_2d/{train,val,test}.txt`, `splits.csv`, `splits.json`, kèm log `log_make_split.txt`.
- `configs/splits_task01/test.txt`: danh sách test cho bộ Task01_BrainTumour (dùng chung cho 2D/3D).

## Kịch bản thực nghiệm

- Tất cả script dùng `CFG`/`CFG_INFER`; kết quả lưu theo `experiments/<EXP_NAME>/...` (checkpoints, inference, metrics, hình ảnh).
- VNet 3D patch-based: `trainers/train_vnet_brats3d_sup.py` + `inference/infer_vnet_brats3d.py`.
- VNet 3D full-volume: `trainers/train_vnet_full.py` + `inference/inference_vnet_brats3d_fullvolume.py`.
- VNet 3D multi-head: `trainers/train_vnetmh_brats3d_sup.py` + `inference/infer_vnetmh_brats3d.py`; trực quan bằng `scripts/visualize_vnetmh_results.py`.
- VNet 3D multi-encoder: `trainers/train_vnet_multienc_brats3d_sup.py` + `inference/infer_vnet_multienc_brats3d.py`.
- Swin-UNet 3D patch-based: `trainers/train_swin_unet3d_patch.py` + `inference/infer_swin_unet3d.py` (TTA tùy chọn).
- UNETR 3D full-volume: `Unetr3d_using_colab/trainers/train_unetr3d_brats2020.py` + `Unetr3d_using_colab/inference/infer_unetr3d.py`; được tối ưu để chạy trên Google Colab với Vision Transformer encoder và CNN decoder.
- UNet/UNet++ 2D chạy qua notebook: `notebooks/unet2d_brats20_final.ipynb`, `notebooks/unetpp_2d/unetpp_2d_diceloss_celoss.ipynb`; log trong `logs/`.

## Model, loss, metric

- Model:
  - `models/vnet.py`: VNet 3D cơ bản (tùy chọn batch/group/instance norm, dropout).
  - `models/vnet_multihead.py`: encoder/decoder chung, 3 head WT/TC/ET (mỗi head 2-class).
  - `models/vnet_multi_enc_fusion.py`: nhiều encoder cho từng modality, trộn feature trước decoder.
  - `models/swin_unet_3d.py`: Swin-UNet 3D (backbone transformer theo cửa sổ), dùng cho huấn luyện patch-based.
  - `Unetr3d_using_colab/models/unetr.py`: UNETR 3D với Vision Transformer encoder (patch embedding, multi-head attention) kết hợp CNN decoder; hỗ trợ skip connection từ các layer transformer.
- Loss & metric:
  - `losses/losses.py`, `losses/losses_2.py`: Dice, Cross-Entropy, DiceCE, các biến thể có trọng số.
  - `losses/composite.py`: helper kết hợp loss/regularizer.
  - `losses/metrics.py`: Dice theo lớp và vùng cấu trúc (WT/TC/ET), IoU, ASD, HD95.
  - `Unetr3d_using_colab/losses/combined_loss.py`: Soft Dice Loss + Cross-Entropy cho UNETR.
  - `Unetr3d_using_colab/losses/metrics_unetr.py`: Dice, IoU, HD95, ASD cho từng lớp và vùng BraTS (WT/TC/ET).
  - `losses/ramps.py`: hàm ramp-up dùng cho semi-supervised (nếu cần).

## Notebook & log

- `notebooks/visualize_inference_3d.ipynb`: trực quan kết quả suy luận 3D (slice hoặc 3D volume).
- `notebooks/log.ipynb`, `notebooks/test.ipynb`: ghi chép/thử nghiệm nhanh.
- W&B: thư mục `wandb/` lưu log/offline run (file `.wandb`, `files/`, `logs/`, `tmp/`).

## Thiết lập môi trường & chạy thử

- Cài gói: `pip install -r requirements.txt`.
- Tiền xử lý:
  - 3D BraTS: `python scripts/preprocess_brats3d.py`
  - 2D BraTS: `python scripts/preprocess_brats2d_version2.py`
  - Task01 3D: `python scripts/preprocess_task01_brain3d.py`
  - Task01 2D: `python scripts/preprocess_task01_2d.py`
- Chia tập 2D BraTS: `python scripts/make_split.py` (tạo `configs/splits_2d`).
- Huấn luyện ví dụ: `python trainers/train_vnet_brats3d_sup.py` (patch) hoặc `python trainers/train_vnet_full.py` (full-volume). Cần chỉnh `CFG["EXP_NAME"]`, batch size và đường dẫn nếu thay đổi cấu trúc dữ liệu.
- Suy luận ví dụ: `python inference/infer_vnet_brats3d.py` hoặc `python inference/inference_vnet_brats3d_fullvolume.py` sau khi cập nhật `CFG_INFER["CKPT_NAME"]` trỏ đúng checkpoint.
