# -*- coding: utf-8 -*-
"""
EDA BraTS2020 (raw 3D) — Thống kê theo nhãn phân đoạn & trực quan hoá.

Tính cho từng ca (training – có nhãn):
- Vùng nhãn nguyên thuỷ: 1 (NCR/NET), 2 (ED), 4 (ET).
- Tổ hợp BraTS: WT = 1|2|4; TC = 1|4; ET = 4.
- Thể tích mm^3 = voxel_count * voxel_volume (mm^3); mL = mm^3 / 1000
- % mỗi vùng so với thể tích não (binarize từ T1: T1>0) theo từng ca.

Tuỳ chọn:
- --by_grade: nếu bật, phân tích theo 3 nhóm: ALL, HGG, LGG (dựa vào name_mapping.csv / cột 'Grade').

Xuất:
- CSV thống kê per-case (+ cột Grade nếu có)
- Biểu đồ:
  (1) Histogram WT/TC/ET (mL)
  (2) Violin plot WT/TC/ET (mL)
  (3) Pie chart trung bình % WT–TC–ET (chồng lấn; chỉ minh hoạ)
  (4) Scatter plot (ED vs ET) (mL)

Yêu cầu:
    pip install nibabel numpy pandas matplotlib tqdm
"""

import os
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm


# ---------------- CONFIG ----------------
RAW_TRAIN = r"D:\Project Advanced CV\data\BraST2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
OUT_DIR   = r"D:\Project Advanced CV\experiments\eda"

# Tên file xuất (dùng cho từng nhóm)
CSV_OUT         = "brats_volume_stats.csv"
HIST_FIG_OUT    = "hist_WT_TC_ET_ml.png"
VIOLIN_FIG_OUT  = "violin_WT_TC_ET_ml.png"
PIE_FIG_OUT     = "pie_mean_WT_TC_ET_pct.png"
SCATTER_FIG_OUT = "scatter_ED_vs_ET_ml.png"
# ---------------------------------------


def as_ras_data(fp: Path):
    img = nib.load(str(fp))
    img = nib.as_closest_canonical(img)
    return img.get_fdata(dtype=np.float32), img.header.get_zooms()


def find_case_id(name: str) -> str:
    # BraTS20_Training_123 -> '123'
    m = re.search(r"_(\d+)$", name)
    return m.group(1) if m else name


def load_grade_mapping(train_root: str) -> dict:
    """
    Đọc name_mapping.csv trong thư mục RAW_TRAIN.
    Kỳ vọng có cột 'Grade' (HGG/LGG) và cột chứa tên ca tương ứng với folder (thường là 'BraTS_2020_subject_ID').
    Ta sẽ cố gắng tìm cột tên ca: ưu tiên 'BraTS_2020_subject_ID', fallback 'Subject_ID' hoặc 'name'.
    """
    csv_path = Path(train_root) / "name_mapping.csv"
    if not csv_path.exists():
        print(f"[WARN] name_mapping.csv not found at {csv_path}. Grade analysis will be skipped.")
        return {}

    df = pd.read_csv(csv_path)
    col_candidates = [c for c in df.columns if str(c).lower() in ["brats_2020_subject_id", "subject_id", "name", "case", "id"]]
    if not col_candidates:
        # cố gắng đoán cột có dạng BraTS20_Training_xxx
        for c in df.columns:
            if df[c].astype(str).str.contains("BraTS20_Training_").any():
                col_candidates.append(c)
                break
    if not col_candidates:
        raise ValueError("Không tìm thấy cột chứa subject ID trong name_mapping.csv")

    id_col = col_candidates[0]
    if "Grade" not in df.columns:
        raise ValueError("Không tìm thấy cột 'Grade' trong name_mapping.csv")

    mapping = {}
    for _, r in df.iterrows():
        k = str(r[id_col]).strip()
        g = str(r["Grade"]).strip().upper()
        # chuẩn hoá key giống tên folder
        # ví dụ: 'BraTS20_Training_001'
        mapping[k] = g
    return mapping


def compute_case_stats(case_dir: Path) -> dict:
    """Tính thống kê cho 1 ca (dựa trên seg + T1)."""
    name = case_dir.name
    seg_fp = case_dir / f"{name}_seg.nii"
    t1_fp  = case_dir / f"{name}_t1.nii"

    if not seg_fp.exists() or not t1_fp.exists():
        return {}

    seg, seg_zooms = as_ras_data(seg_fp)   # seg là 0/1/2/4
    t1,  _         = as_ras_data(t1_fp)

    # Voxel volume (mm^3)
    vx_vol_mm3 = float(np.prod(seg_zooms))

    # Voxel count theo nhãn
    n1 = int(np.sum(seg == 1))
    n2 = int(np.sum(seg == 2))
    n4 = int(np.sum(seg == 4))
    nWT = int(np.sum(seg > 0))
    nTC = int(np.sum((seg == 1) | (seg == 4)))
    nET = n4

    # Thể tích mm^3 / mL
    vol1_mm3 = n1 * vx_vol_mm3
    vol2_mm3 = n2 * vx_vol_mm3
    vol4_mm3 = n4 * vx_vol_mm3
    volWT_mm3 = nWT * vx_vol_mm3
    volTC_mm3 = nTC * vx_vol_mm3
    volET_mm3 = nET * vx_vol_mm3

    # Thể tích não từ T1>0
    brain_mask = t1 > 0
    n_brain = int(np.sum(brain_mask))
    brain_vol_mm3 = n_brain * vx_vol_mm3
    brain_vol_ml  = brain_vol_mm3 / 1000.0

    # % theo thể tích não
    def pct(x_mm3):
        return (x_mm3 / brain_vol_mm3 * 100.0) if brain_vol_mm3 > 0 else np.nan

    return {
        "case": name,
        "id": find_case_id(name),
        "voxel_volume_mm3": vx_vol_mm3,

        "vox_1": n1, "vox_2": n2, "vox_4": n4,
        "vox_WT": nWT, "vox_TC": nTC, "vox_ET": nET,

        "vol1_mm3": vol1_mm3, "vol2_mm3": vol2_mm3, "vol4_mm3": vol4_mm3,
        "volWT_mm3": volWT_mm3, "volTC_mm3": volTC_mm3, "volET_mm3": volET_mm3,

        "vol1_ml": vol1_mm3/1000.0, "vol2_ml": vol2_mm3/1000.0, "vol4_ml": vol4_mm3/1000.0,
        "volWT_ml": volWT_mm3/1000.0, "volTC_ml": volTC_mm3/1000.0, "volET_ml": volET_mm3/1000.0,

        "brain_vol_mm3": brain_vol_mm3, "brain_vol_ml": brain_vol_ml,

        "pct1_brain": pct(vol1_mm3),
        "pct2_brain": pct(vol2_mm3),
        "pct4_brain": pct(vol4_mm3),
        "pctWT_brain": pct(volWT_mm3),
        "pctTC_brain": pct(volTC_mm3),
        "pctET_brain": pct(volET_mm3),
    }


def collect_all_cases(raw_train: str) -> pd.DataFrame:
    root = Path(raw_train)
    cases = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("BraTS20_Training_")])
    rows = []
    for c in tqdm(cases, desc="Scanning training cases"):
        s = compute_case_stats(c)
        if s:
            rows.append(s)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["id"] = df["id"].astype(str)
        df.sort_values("id", inplace=True)
    return df


def make_histograms(df: pd.DataFrame, out_dir: Path):
    # WT / TC / ET (mL)
    for col, fname in [("volWT_ml", "hist_WT_ml.png"),
                       ("volTC_ml", "hist_TC_ml.png"),
                       ("volET_ml", "hist_ET_ml.png")]:
        plt.figure()
        plt.hist(df[col].dropna().values, bins=30)
        plt.xlabel(f"{col} (mL)")
        plt.ylabel("Count")
        plt.title(f"Histogram of {col}")
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=200)
        plt.close()

    # Tổng hợp 3 histogram
    plt.figure()
    plt.hist(df["volWT_ml"].dropna(), bins=30, alpha=0.5, label="WT")
    plt.hist(df["volTC_ml"].dropna(), bins=30, alpha=0.5, label="TC")
    plt.hist(df["volET_ml"].dropna(), bins=30, alpha=0.5, label="ET")
    plt.xlabel("Volume (mL)")
    plt.ylabel("Count")
    plt.title("Histogram WT / TC / ET")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "hist_WT_TC_ET_ml.png", dpi=200)
    plt.close()


def make_violin(df: pd.DataFrame, out_path: Path):
    data = [df["volWT_ml"].dropna().values,
            df["volTC_ml"].dropna().values,
            df["volET_ml"].dropna().values]
    plt.figure()
    _ = plt.violinplot(data, showmeans=True, showextrema=True, showmedians=True)
    plt.xticks([1,2,3], ["WT (mL)", "TC (mL)", "ET (mL)"])
    plt.title("Violin plot of WT / TC / ET volumes")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_pie_mean_pct(df: pd.DataFrame, out_path: Path):
    """
    Pie chart trung bình % WT–TC–ET so với thể tích não (chồng lấn).
    """
    mean_wt = float(df["pctWT_brain"].mean())
    mean_tc = float(df["pctTC_brain"].mean())
    mean_et = float(df["pctET_brain"].mean())
    vals = [mean_wt, mean_tc, mean_et]
    labels = [f"WT ({mean_wt:.2f}%)", f"TC ({mean_tc:.2f}%)", f"ET ({mean_et:.2f}%)"]

    plt.figure()
    plt.pie(vals, labels=labels, autopct="%.1f%%")
    plt.title("Mean % of WT / TC / ET relative to brain volume (overlapping sets)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_scatter_ed_vs_et(df: pd.DataFrame, out_path: Path):
    sub = df[["vol2_ml", "vol4_ml"]].dropna()
    x = sub["vol2_ml"].values  # ED
    y = sub["vol4_ml"].values  # ET
    plt.figure()
    plt.scatter(x, y, s=16)
    plt.xlabel("ED volume (mL)")
    plt.ylabel("ET volume (mL)")
    plt.title("Scatter: ED vs ET volumes")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_one_group(df: pd.DataFrame, out_dir_group: Path):
    out_dir_group.mkdir(parents=True, exist_ok=True)
    # Lưu CSV
    df.to_csv(out_dir_group / CSV_OUT, index=False)
    # Vẽ biểu đồ
    make_histograms(df, out_dir_group)
    make_violin(df, out_dir_group / VIOLIN_FIG_OUT)
    make_pie_mean_pct(df, out_dir_group / PIE_FIG_OUT)
    make_scatter_ed_vs_et(df, out_dir_group / SCATTER_FIG_OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_train", type=str, default=RAW_TRAIN)
    ap.add_argument("--out_dir",   type=str, default=OUT_DIR)
    ap.add_argument("--by_grade",  action="store_true", help="Phân tích theo Grade (HGG/LGG) + ALL")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Thu thập thống kê chung
    df_all = collect_all_cases(args.raw_train)

    if df_all.empty:
        print("[ERROR] Không có dữ liệu nào được đọc. Kiểm tra đường dẫn RAW_TRAIN.")
        return

    if not args.by_grade:
        # Phân tích chung (như bản cũ)
        print("[INFO] Running aggregated analysis (ALL). Use --by_grade để tách HGG/LGG.")
        run_one_group(df_all, out_root)
        print(f"[OK] Saved (ALL) to: {out_root}")
        return

    # 2) Ghép Grade từ name_mapping.csv
    try:
        grade_map = load_grade_mapping(args.raw_train)
    except Exception as e:
        print(f"[WARN] Không thể đọc/ghép Grade ({e}). Chạy phân tích chung.")
        run_one_group(df_all, out_root)
        return

    # Chuẩn hoá key để khớp: dùng tên folder 'case'
    # name_mapping thường có key đúng bằng tên folder (BraTS20_Training_xxx).
    df_all["Grade"] = df_all["case"].map(lambda k: grade_map.get(k, np.nan))

    # Báo các case không có Grade
    missing = df_all["Grade"].isna().sum()
    if missing > 0:
        print(f"[WARN] {missing} cases thiếu Grade trong name_mapping.csv (sẽ bị loại khỏi phân tích theo Grade).")

    df_with_grade = df_all.dropna(subset=["Grade"])
    if df_with_grade.empty:
        print("[WARN] Không có case nào có Grade. Chạy phân tích chung.")
        run_one_group(df_all, out_root)
        return

    # 3) Chạy cho ALL + từng nhóm HGG / LGG
    run_one_group(df_all, out_root / "ALL")

    for g in ["HGG", "LGG"]:
        sub = df_with_grade[df_with_grade["Grade"].str.upper() == g]
        if sub.empty:
            print(f"[WARN] Không có case thuộc nhóm {g}. Bỏ qua.")
            continue
        run_one_group(sub, out_root / g)

    print(f"[OK] Saved grouped results to: {out_root} (ALL/HGG/LGG)")


if __name__ == "__main__":
    main()
