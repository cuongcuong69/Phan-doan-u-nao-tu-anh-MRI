# Hướng dẫn chạy dự án trên Google Colab

Thư mục này (`colab_project`) chứa phiên bản rút gọn của dự án để chạy trên Google Colab.

## 1. Chuẩn bị Dữ liệu (Thực hiện trên máy Local)

Bạn cần nén dữ liệu tiền xử lý và các file cấu hình split thành một file zip để upload lên Colab cho nhanh.

### Bước 1: Nén dữ liệu

Vào thư mục gốc dự án (`d:\Project Advanced CV`), tạo file nén `brats2020_processed.zip` chứa 2 thư mục:

- `data/processed/3d/labeled/` (Chứa các thư mục Brain_xxx đã xử lý)
- `configs/` (Chứa splits_2d/\*.txt)

Cấu trúc file zip mong muốn:

```
brats2020_processed.zip
├── data
│   └── processed
│       └── 3d
│           └── labeled
│               ├── Brain_001
│               └── ...
└── configs
    └── splits_2d
        ├── train.txt
        ├── val.txt
        └── test.txt
```

> **Lưu ý**: Bạn có thể dùng WinRAR hoặc 7-Zip để nén. Đảm bảo đường dẫn bên trong bắt đầu từ `data/...` và `configs/...`.

## 2. Chuẩn bị Code (Thực hiện trên máy Local)

Bạn cần upload thư mục `colab_project` này lên Google Drive.

- Có thể nén toàn bộ thư mục `colab_project` thành `colab_project.zip`.

## 3. Upload lên Google Drive

1. Tạo thư mục mới trên Google Drive, ví dụ: `MyDrive/BraTS_Project`.
2. Upload `brats2020_processed.zip` vào thư mục đó.
3. Upload `colab_project` (hoặc giải nén `colab_project.zip`) vào thư mục đó. Đảm bảo bạn thấy file `Start_UNETR_Colab.ipynb` nằm trong đường dẫn Drive.

## 4. Chạy trên Google Colab

1. Mở file `Start_UNETR_Colab.ipynb` bằng Google Colab (chuột phải > Open with > Google Colaboratory).
2. Làm theo hướng dẫn từng cell trong Notebook:
   - Mount Drive.
   - Cài đặt thư viện.
   - Giải nén dữ liệu từ Drive vào ổ cứng Colab (`/content/data/...`) để train nhanh hơn.
   - Chạy lệnh Training.
   - Chạy lệnh Inference.

## Lưu ý về đường dẫn

Script `Start_UNETR_Colab.ipynb` được thiết lập để giả lập môi trường chạy tại `/content/colab_project`. Dữ liệu sẽ được giải nén ra `/content/data` để code có thể đọc được từ đường dẫn `../data` hoặc chỉnh lại `ROOT`.

Trong `train_unetr3d_brats2020.py` và các file khác, ROOT được xác định tương đối. Notebook sẽ thiết lập `os.chdir` phù hợp.
