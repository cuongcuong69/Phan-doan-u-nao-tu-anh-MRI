import pickle
import matplotlib.pyplot as plt
import argparse
import os
import sys

def plot_loss_from_pkl(train_pkl, val_pkl, output_path=None, show=True):
    """
    Plot train loss vs val loss from PKL files.
    
    Args:
        train_pkl: Path to train_log.pkl
        val_pkl: Path to val_log.pkl
        output_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    # Load train data
    try:
        with open(train_pkl, 'rb') as f:
            train_data = pickle.load(f)
        print(f"✅ Loaded {len(train_data)} train records")
    except Exception as e:
        print(f"❌ Error loading train data: {e}")
        return
    
    # Load val data
    try:
        with open(val_pkl, 'rb') as f:
            val_data = pickle.load(f)
        print(f"✅ Loaded {len(val_data)} val records")
    except Exception as e:
        print(f"❌ Error loading val data: {e}")
        return
    
    # Extract loss values
    train_losses = []
    val_losses = []
    
    for record in train_data:
        if 'loss' in record:
            train_losses.append(record['loss'])
    
    for record in val_data:
        if 'loss' in record:
            val_losses.append(record['loss'])
    
    if not train_losses and not val_losses:
        print("❌ No 'loss' key found in the data")
        return
    
    # Create epochs
    max_epochs = max(len(train_losses), len(val_losses))
    epochs = list(range(1, max_epochs + 1))
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    if train_losses:
        train_epochs = list(range(1, len(train_losses) + 1))
        plt.plot(train_epochs, train_losses, color='#1f77b4', label='Train loss', linewidth=2)
    
    if val_losses:
        val_epochs = list(range(1, len(val_losses) + 1))
        plt.plot(val_epochs, val_losses, color='#ff7f0e', label='Val loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {output_path}")
    
    # Show the plot
    if show:
        plt.show()
    
    plt.close()

def plot_loss_from_csv(csv_path, output_path=None, show=True):
    """
    Plot train loss vs val loss from CSV file.
    
    Args:
        csv_path: Path to combined_log.csv
        output_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    import csv
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        print(f"✅ Loaded {len(data)} records from CSV")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    # Extract data
    epochs = []
    train_losses = []
    val_losses = []
    
    for row in data:
        if 'epoch' in row and row['epoch']:
            epochs.append(int(row['epoch']))
        
        if 'train_loss' in row and row['train_loss']:
            train_losses.append(float(row['train_loss']))
        else:
            train_losses.append(None)
        
        if 'val_loss' in row and row['val_loss']:
            val_losses.append(float(row['val_loss']))
        else:
            val_losses.append(None)
    
    # Filter out None values
    train_data = [(e, l) for e, l in zip(epochs, train_losses) if l is not None]
    val_data = [(e, l) for e, l in zip(epochs, val_losses) if l is not None]
    
    if not train_data and not val_data:
        print("❌ No loss data found in CSV")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    if train_data:
        train_epochs, train_loss_values = zip(*train_data)
        plt.plot(train_epochs, train_loss_values, color='#1f77b4', label='Train loss', linewidth=2)
    
    if val_data:
        val_epochs, val_loss_values = zip(*val_data)
        plt.plot(val_epochs, val_loss_values, color='#ff7f0e', label='Val loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {output_path}")
    
    # Show the plot
    if show:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    DEFAULT_BASE_PATH = r"D:\Project Advanced CV\colab_project"
    DEFAULT_TRAIN = os.path.join(DEFAULT_BASE_PATH, "train_log.pkl")
    DEFAULT_VAL = os.path.join(DEFAULT_BASE_PATH, "val_log.pkl")
    DEFAULT_CSV = os.path.join(DEFAULT_BASE_PATH, "combined_log.csv")
    DEFAULT_OUTPUT = os.path.join(DEFAULT_BASE_PATH, "loss_comparison.png")
    
    parser = argparse.ArgumentParser(
        description="Plot train loss vs val loss comparison"
    )
    parser.add_argument(
        "--mode",
        choices=['pkl', 'csv'],
        default='csv',
        help="Input mode: 'pkl' to read from PKL files, 'csv' to read from CSV file"
    )
    parser.add_argument(
        "--train",
        default=DEFAULT_TRAIN,
        help="Path to train_log.pkl (only for pkl mode)"
    )
    parser.add_argument(
        "--val",
        default=DEFAULT_VAL,
        help="Path to val_log.pkl (only for pkl mode)"
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help="Path to combined_log.csv (only for csv mode)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT,
        help="Path to save the plot image"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot, only save to file"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Loss Comparison Plotter")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    
    if args.mode == 'pkl':
        print(f"Train file: {args.train}")
        print(f"Val file:   {args.val}")
        print("=" * 60)
        print()
        plot_loss_from_pkl(args.train, args.val, args.output, not args.no_show)
    else:  # csv mode
        print(f"CSV file: {args.csv}")
        print("=" * 60)
        print()
        plot_loss_from_csv(args.csv, args.output, not args.no_show)
