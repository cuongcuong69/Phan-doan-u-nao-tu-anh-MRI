import pickle
import csv
import argparse
import os
import sys

def convert_multiple_pkl_to_csv_horizontal(pkl_paths, csv_path):
    """
    Convert multiple PKL files to a single CSV file with horizontal layout.
    Train data columns on the left, val data columns on the right.
    Adds an 'epoch' column at the beginning.
    
    Args:
        pkl_paths: List of tuples (file_path, source_name)
        csv_path: Output CSV file path
    """
    data_dict = {}
    keys_per_source = {}
    
    # Load all PKL files
    for pkl_path, source_name in pkl_paths:
        if not os.path.exists(pkl_path):
            print(f"⚠️ Warning: File not found: {pkl_path} - Skipping...")
            continue
        
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            if not isinstance(data, list):
                print(f"⚠️ Warning: Expected a list in {pkl_path} - Skipping...")
                continue
            
            if not data:
                print(f"⚠️ Warning: Empty list in {pkl_path} - Skipping...")
                continue
            
            data_dict[source_name] = data
            
            # Get unique keys for this source
            all_keys_in_source = set()
            for record in data:
                all_keys_in_source.update(record.keys())
            keys_per_source[source_name] = sorted(all_keys_in_source)
            
            print(f"✅ Loaded {len(data)} records from '{pkl_path}' (source: {source_name})")
            print(f"   Keys: {', '.join(keys_per_source[source_name])}")
            
        except Exception as e:
            print(f"❌ Error loading {pkl_path}: {e}")
            continue
    
    if not data_dict:
        print("❌ No data to write. All files were empty or had errors.")
        return
    
    # Determine the maximum number of rows
    max_rows = max(len(data) for data in data_dict.values())
    
    # Create column headers: epoch first, then train columns, then val columns
    fieldnames = ['epoch']
    for source_name in ['train', 'val']:  # Ensure train comes first
        if source_name in keys_per_source:
            for key in keys_per_source[source_name]:
                fieldnames.append(f"{source_name}_{key}")
    
    # Create rows by combining data horizontally
    rows = []
    for i in range(max_rows):
        row = {'epoch': i + 1}  # Epoch starts from 1
        
        for source_name in ['train', 'val']:
            if source_name in data_dict:
                data = data_dict[source_name]
                if i < len(data):
                    record = data[i]
                    # Only add keys that exist in this record
                    for key in keys_per_source[source_name]:
                        if key in record:
                            row[f"{source_name}_{key}"] = record[key]
        rows.append(row)
    
    # Write to CSV
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"\n✅ Successfully created '{csv_path}'")
        print(f"📊 Total rows: {len(rows)} (epochs)")
        print(f"📋 Total columns: {len(fieldnames)}")
        for source_name, data in data_dict.items():
            print(f"   - {source_name}: {len(data)} records, {len(keys_per_source[source_name])} columns")
        
    except Exception as e:
        print(f"❌ Error writing CSV: {e}")

if __name__ == "__main__":
    # Default paths
    DEFAULT_BASE_PATH = r"D:\Project Advanced CV\colab_project"
    DEFAULT_TRAIN = os.path.join(DEFAULT_BASE_PATH, "train_log.pkl")
    DEFAULT_VAL = os.path.join(DEFAULT_BASE_PATH, "val_log.pkl")
    DEFAULT_OUTPUT = os.path.join(DEFAULT_BASE_PATH, "combined_log.csv")
    
    parser = argparse.ArgumentParser(
        description="Convert train_log.pkl and val_log.pkl to a single CSV file (horizontal layout)."
    )
    parser.add_argument(
        "--train", 
        default=DEFAULT_TRAIN, 
        help="Path to train_log.pkl file"
    )
    parser.add_argument(
        "--val", 
        default=DEFAULT_VAL, 
        help="Path to val_log.pkl file"
    )
    parser.add_argument(
        "--output", 
        "-o",
        default=DEFAULT_OUTPUT, 
        help="Path to output CSV file"
    )

    args = parser.parse_args()
    
    # Prepare list of (file_path, source_name) tuples
    pkl_files = [
        (args.train, "train"),
        (args.val, "val")
    ]
    
    print("=" * 60)
    print("PKL to CSV Converter - Horizontal Layout")
    print("=" * 60)
    print(f"Train file: {args.train}")
    print(f"Val file:   {args.val}")
    print(f"Output:     {args.output}")
    print("=" * 60)
    print()
    
    convert_multiple_pkl_to_csv_horizontal(pkl_files, args.output)

