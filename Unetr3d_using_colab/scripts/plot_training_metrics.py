import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
csv_path = r"D:\Project Advanced CV\colab_project\combined_log.csv"
df = pd.read_csv(csv_path)

print(f"✅ Loaded {len(df)} epochs")

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('UNETR Training Metrics', fontsize=16)

# Set common xticks for all subplots
xticks = [0, 25, 50, 75, 100, 125, 150]

# 1. Training vs Validation Loss (Top Left)
ax1 = axes[0, 0]
ax1.plot(df['epoch'], df['train_loss'], label='Train loss', color='#1f77b4', linewidth=1.5)
ax1.plot(df['epoch'], df['val_loss'], label='Val loss', color='#ff7f0e', linewidth=1.5)
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('Training vs Validation Loss', fontsize=12)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 150)
ax1.set_xticks(xticks)

# 2. Dice WT over epochs (Top Right)
ax2 = axes[0, 1]
ax2.plot(df['epoch'], df['train_dice_wt'], label='Train dice_wt', color='#1f77b4', linewidth=1.5)
ax2.plot(df['epoch'], df['val_dice_wt'], label='Val dice_wt', color='#ff7f0e', linewidth=1.5)
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Dice', fontsize=11)
ax2.set_title('dice_wt over epochs', fontsize=12)
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 150)
ax2.set_xticks(xticks)

# 3. Dice TC over epochs (Bottom Left)
ax3 = axes[1, 0]
ax3.plot(df['epoch'], df['train_dice_tc'], label='Train dice_tc', color='#1f77b4', linewidth=1.5)
ax3.plot(df['epoch'], df['val_dice_tc'], label='Val dice_tc', color='#ff7f0e', linewidth=1.5)
ax3.set_xlabel('Epoch', fontsize=11)
ax3.set_ylabel('Dice', fontsize=11)
ax3.set_title('dice_tc over epochs', fontsize=12)
ax3.legend(loc='lower right', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 150)
ax3.set_xticks(xticks)

# 4. Dice ET over epochs (Bottom Right)
ax4 = axes[1, 1]
ax4.plot(df['epoch'], df['train_dice_et'], label='Train dice_et', color='#1f77b4', linewidth=1.5)
ax4.plot(df['epoch'], df['val_dice_et'], label='Val dice_et', color='#ff7f0e', linewidth=1.5)
ax4.set_xlabel('Epoch', fontsize=11)
ax4.set_ylabel('Dice', fontsize=11)
ax4.set_title('dice_et over epochs', fontsize=12)
ax4.legend(loc='lower right', fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 150)
ax4.set_xticks(xticks)

# Adjust layout
plt.tight_layout(pad=2.0)

# Save figure
output_path = r"D:\Project Advanced CV\colab_project\training_metrics_4plots.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved figure to: {output_path}")

# Show plot
plt.show()

print("\n📊 Summary Statistics:")
print(f"Final Train Loss: {df['train_loss'].iloc[-1]:.4f}")
print(f"Final Val Loss: {df['val_loss'].iloc[-1]:.4f}")
print(f"Final Val Dice WT: {df['val_dice_wt'].iloc[-1]:.4f}")
print(f"Final Val Dice TC: {df['val_dice_tc'].iloc[-1]:.4f}")
print(f"Final Val Dice ET: {df['val_dice_et'].iloc[-1]:.4f}")
print(f"Best Val Dice WT: {df['val_dice_wt'].max():.4f} (Epoch {df['val_dice_wt'].idxmax() + 1})")
print(f"Best Val Dice TC: {df['val_dice_tc'].max():.4f} (Epoch {df['val_dice_tc'].idxmax() + 1})")
print(f"Best Val Dice ET: {df['val_dice_et'].max():.4f} (Epoch {df['val_dice_et'].idxmax() + 1})")
