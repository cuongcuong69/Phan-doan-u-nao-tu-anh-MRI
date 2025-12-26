import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- THAY ĐỔI ĐƯỜNG DẪN CSV CHO ĐÚNG --- 
csv_file_path = r"D:\Project Advanced CV\logs\unetpp_2d_multilabel_ds_dice_ce_sum\metrics_log.csv"

# Đọc dữ liệu
df = pd.read_csv(csv_file_path)
print("Các cột trong CSV:", df.columns)

# Thiết lập style cho biểu đồ
sns.set_style("whitegrid")

# Tạo figure với 2 biểu đồ
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))

# --- Biểu đồ 1: Loss ---
ax1.plot(df['epoch'], df['train_loss'], label='Train Loss')
ax1.plot(df['epoch'], df['val_loss'], label='Validation Loss')
ax1.set_title('So sánh Loss giữa Train và Validation', fontsize=16)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.legend()
ax1.grid(True)

# --- Biểu đồ 2: Dice Score ---
ax2.plot(df['epoch'], df['train_dice_avg'], label='Train Dice')
ax2.plot(df['epoch'], df['val_dice_avg'], label='Validation Dice')
ax2.set_title('Dice Score qua các Epoch', fontsize=16)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Dice Score', fontsize=12)
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
