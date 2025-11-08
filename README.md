# Công việc
## 1. Thu thập và xử lý dữ liệu:
- [ ] Tiền xử lý: Crop, nomalize, chia thành các lát cắt 2D nếu dùng mô hình 2D. Thực hiện các biện pháp tăng cường dữ liệu (giữa kì)
- [ ] Thu thập, bổ sung thêm một số mẫu dữ liệu cùng loại, cùng bài toán $\to$ thực hiện thao tác tiền xử lý để chuẩn bị cho các phần sau (giữa kì)
- [ ] Thực hiện thống kê, phân tích trên bộ dữ liệu, trực quan hóa phân bố của dữ liệu (giữa kì
## 2. Xây dựng mô hình
- [ ] Mô hình: Sử dụng $\ge$ 3 mô hình để giải quyết bài toán (giữa kì báo cáo ít nhất 1 mô hình)
	- [ ] Mô hình 1:
	- [ ] Mô hình 2:
	- [ ] Mô hình 3:
- [ ] Huấn luyện với Dice Loss và Dice Loss + CE loss
## 3. Đánh giá mô hình
- [ ] Sử dụng Dice, IoU, ASD, HD95
- [ ] Inference trên tập test và overlay với GT
## 3. Trình bày kết quả
- [ ] Hiển thị ảnh đầu vào, phân vùng thực và phân vùng dự đoán
- [ ] Viết báo cáo mô tả toàn bộ pipeline


# Chi tiết: Tiền xử lý dữ liệu
`scripts\preprocess_brats2d_version2.py`
1. Dò bounding box toàn cục trên T1 (train + val)
	- Đọc T1 $\to$ đưa về RAS
	- Nhị phân hóa **mask brain** = (T1 > 0)
	- Lấy min/max theo **(x,y)** với mọi lát **z** cho từng ca; gộp qua tất cả ca → được bbox toàn cục.
	- **Ép hình vuông** (lấy tâm bbox, nới chiều ngắn lên bằng chiều dài max; ràng buộc trong kích thước ảnh).
	- (Tuỳ chọn) **lưu cache** để tái sử dụng.
2. Tiền xử lý từng ca với BBox đã cố định
	- Reorient **RAS** cho 4 chuỗi (FLAIR/T1/T1CE/T2) và **seg** (nếu có).
	- **Chuẩn hoá [0,1] theo percentile** 1–99% trên vùng >0; ép nền = 0.
	- Cắt lát theo **axial** (axis=2).
	- **Crop từng lát theo BBox vuông** (toạ độ trong RAS).
	- **Dựng dọc** (xoay/flip theo tham số, mặc định xoay 90° CCW).
	- **Resize về 256×256**: ảnh dùng **INTER_LINEAR**, mask dùng **INTER_NEAREST**.
	- **Map nhãn 4→3** rồi lưu **PNG 8-bit** (ảnh: 0–255 từ [0,1], mask: 0...3).
3. Cấu trúc đầu ra:
```bash
processed/
  2d/
    labeled/Brain_001/{flair,t1,t1ce,t2,mask}/...
    unlabeled/Brain_001/{flair,t1,t1ce,t2}/...

```
Output khi chạy script
```bash
Scan T1 (labeled): 100%|███████████████████████████████████████████████████████████████████| 369/369 [00:45<00:00,  8.10it/s]
Scan T1 (unlabeled): 100%|█████████████████████████████████████████████████████████████████| 125/125 [00:15<00:00,  8.16it/s]
[Global BBox square] x:[22, 216] y:[16, 210] side=195 on 240x240
Labeled: 100%|█████████████████████████████████████████████████████████████████████████████| 369/369 [18:22<00:00,  2.99s/it]
Unlabeled: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 125/125 [05:50<00:00,  2.80s/it]
Done.
```
