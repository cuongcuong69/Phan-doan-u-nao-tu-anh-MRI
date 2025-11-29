# Advanced CV - Brain Tumor Segmentation 
# 📌 Phân Chia Công Việc Nhóm

## 👥 Thành viên & Nhiệm vụ

| **Thành viên**       | **Nhiệm vụ phụ trách**                                                                 |
|-----------------------|-----------------------------------------------------------------------------------------|
| **Lê Mạnh Cương**     | - Tiền xử lý dữ liệu   
|                       | - Thống kê và phân tích dữ liệu (EDA)  
|                       | - Mô hình **VNet**                                                             |
| **Nguyễn Tuấn Anh**   | - Mô hình **UNet**                                               |
| **Phạm Quý Đô**       | - Mô hình **UNet++**                                             |

---

## 📂 Bộ dữ liệu sử dụng

- **BraTS 2020 Dataset**  
  🔗 Link tải: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation

---


## Cấu trúc thư mục chính
- `configs/`: cấu hình split dataset. `configs/splits_2d` chứa `train.txt`, `val.txt`, `test.txt` cùng bản JSON/CSV thống kê; `configs/splits_task01/test.txt` cho bộ Task01_BrainTumour.
- `data/`: dữ liệu thô và tiền xử lý. Bao gồm `BraST2020/`, `Task01_BrainTumour/`, kết quả tiền xử lý `processed/` (3D & 2D) và `processed_task01/`. Hai dataloader chính: `data/dataloader_brats3d_sup.py` (patch-based) và `data/dataloader_brats3d_full.py` (full-volume).
- `experiments/`: nơi lưu checkpoints, log, ảnh trực quan và kết quả inference cho từng thí nghiệm (ví dụ `brats3d_vnet_sup`, `brats3d_vnet_sup_fullvolume`, `brats3d_vnetmh_sup`, `brats3d_vnet_multienc_sup`, `task01_vnet`). Thư mục `eda/` chứa kết quả phân tích thống kê.
- `inference/`: script suy luận cho từng biến thể VNet (`infer_vnet_brats3d.py`, `infer_vnetmh_brats3d.py`, `infer_vnet_multienc_brats3d.py`, `inference_vnet_brats3d_fullvolume.py`).
- `losses/`: triển khai hàm mất mát, metrics và ramp scheduler (`losses.py`, `losses_2.py`, `composite.py`, `metrics.py`, `ramps.py`).
- `models/`: định nghĩa model VNet 3D (`vnet.py`), VNet multi-head (`vnet_multihead.py`) và multi-encoder fusion (`vnet_multi_enc_fusion.py`).
- `notebooks/`: notebook phục vụ log và trực quan (`log.ipynb`, `test.ipynb`, `visualize_inference_3d.ipynb`).
- `scripts/`: tiện ích tiền xử lý, chia tập và EDA. Thư mục con `scripts/eda/` chứa các phân tích thống kê (cohort, intensity, radiomics, shape).
- `trainers/`: kịch bản huấn luyện chính cho các biến thể VNet (supervised patch/full-volume, multi-head, multi-encoder).
- `wandb/`: log offline/online của Weights & Biases cho các lần chạy.
- `requirements.txt`: danh sách package cần thiết.

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

## Huấn luyện (trainers/)
- `trainers/train_vnet_brats3d_sup.py`: huấn luyện VNet 3D patch-based với `data/dataloader_brats3d_sup.py`. Cấu hình trong `CFG` (patch size, batch, loss CE/Dice/DiceCE, giảm LR on plateau, eval Dice WT/TC/ET). Checkpoint, log và ảnh trực quan lưu tại `experiments/brats3d_vnet_sup/` (các biến thể loss: `checkpoints_diceloss`, `checkpoints_celoss`).
- `trainers/train_vnet_full.py`: huấn luyện VNet 3D trên full-volume resize (default 128x128x128) với `data/dataloader_brats3d_full.py`. Kết quả lưu `experiments/brats3d_vnet_sup_fullvolume/`.
- `trainers/train_vnetmh_brats3d_sup.py`: VNet multi-head dự đoán WT/TC/ET (3 head nhánh). Log/ckpt tại `experiments/brats3d_vnetmh_sup/`.
- `trainers/train_vnet_multienc_brats3d_sup.py`: VNet multi-encoder fusion (nhiều encoder ghép feature). Kết quả trong `experiments/brats3d_vnet_multienc_sup/`.
- Mỗi trainer dùng `CFG["EXP_NAME"]` để định tên thư mục lưu checkpoint/log. Chạy trực tiếp: `python trainers/train_vnet_brats3d_sup.py` (sửa `CFG` nếu cần đường dẫn, batch, loss, scheduler, resume).

## Suy luận & trực quan (inference/)
- `inference/infer_vnet_brats3d.py`: suy luận cho mô hình patch-based VNet chuẩn. Đọc `configs/splits_2d/test.txt`, load ckpt từ `experiments/brats3d_vnet_sup/...`, xuất NIfTI/PNG overlay và CSV metrics vào `experiments/brats3d_vnet_sup/inference`.
- `inference/inference_vnet_brats3d_fullvolume.py`: suy luận full-volume (resize → forward → resize ngược). Config `CFG_INFER` điều chỉnh `VOLUME_SIZE`, `TEST_LIST`, `CKPT_NAME`, `OUT_DIR`.
- `inference/infer_vnetmh_brats3d.py`: suy luận mô hình multi-head (WT/TC/ET), tính Dice/IoU/ASD/HD95 từng vùng, lưu vào `experiments/brats3d_vnetmh_sup/inference`.
- `inference/infer_vnet_multienc_brats3d.py`: suy luận mô hình multi-encoder fusion.
- Script `scripts/visualize_vnetmh_results.py` dựng grid ảnh overlay dự đoán/GT cho các checkpoint multi-head.
- Ảnh trực quan mẫu nằm trong `experiments/vis_brats3d_sup/` và `experiments/vis_brats3d_full/`.

## Model, loss, metric
- Model:
  - `models/vnet.py`: VNet 3D cơ bản (tùy chọn batch/group/instance norm, dropout).
  - `models/vnet_multihead.py`: encoder/decoder chung, 3 head WT/TC/ET (mỗi head 2-class).
  - `models/vnet_multi_enc_fusion.py`: nhiều encoder cho từng modality, trộn feature trước decoder.
- Loss & metric:
  - `losses/losses.py`, `losses/losses_2.py`: Dice, Cross-Entropy, DiceCE, các biến thể có trọng số.
  - `losses/composite.py`: helper kết hợp loss/regularizer.
  - `losses/metrics.py`: Dice theo lớp và vùng cấu trúc (WT/TC/ET), IoU, ASD, HD95.
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
