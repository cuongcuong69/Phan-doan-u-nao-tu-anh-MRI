# -*- coding: utf-8 -*-
"""
Chia train/val/test 70/15/15 cho dữ liệu 2D đã export:
D:\Project Advanced CV\data\processed\2d\labeled\Brain_001, Brain_002, ...

✅ Stratify theo đặc tính quan trọng (per-case) tính từ mask 2D + Grade (HGG/LGG):
  - Grade (HGG/LGG) từ name_mapping.csv
  - has_ET (label==3)  [nhớ: trước đó map 4->3 khi export 2D]
  - size_bin theo quantile của tổng pixel u (tumor_area_total)

⚠️ Fallback an toàn:
  - Nếu lớp stratify quá nhỏ: tự động rút gọn (grade×hasET×size → grade×hasET → hasET → all)
  - Nếu không map được Grade, gán 'Unknown' (và vẫn tách được nhờ fallback)

📦 Xuất:
  - CSV:  splits.csv  (1 dòng / ca, kèm đặc trưng & split)
  - JSON: splits.json (list id cho train/val/test)
  - TXT:  train.txt, val.txt, test.txt
"""

from pathlib import Path
import json
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ============ CẤU HÌNH (sửa tại đây nếu cần) ============
ROOT                 = Path(r"D:\Project Advanced CV")
LABELED_REL          = Path(r"data\processed\2d\labeled")
OUT_REL              = Path(r"configs\splits_2d")

# Đường dẫn raw 3D & mapping Grade
RAW_TRAIN_DIR        = ROOT / r"data\BraST2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
NAME_MAPPING_CSV     = RAW_TRAIN_DIR / "name_mapping.csv"   # chứa cột Grade

SEED           = 2025
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15
TEST_RATIO     = 0.15
MIN_PER_STRAT  = 8      # tối thiểu / lớp khi stratify
# ========================================================


def find_mask_folder(brain_dir: Path) -> Path:
    """Hỗ trợ cả cấu trúc 'mask' hoặc 'Label0'."""
    for cand in (brain_dir / "mask", brain_dir / "Label0"):
        if cand.exists() and cand.is_dir():
            return cand
    raise FileNotFoundError(f"Không thấy thư mục mask/Label0 trong: {brain_dir}")


def imread_any(path: Path, flags=cv2.IMREAD_UNCHANGED):
    """Đọc ảnh unicode-safe trên Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, flags)


def load_mask_stats_for_case(brain_dir: Path) -> Dict:
    """
    Tính đặc trưng stratify từ toàn bộ mask PNG của 1 Brain_xxx.
    Giả định mask số hóa: 0=BG, 1=NCR, 2=ED, 3=ET.
    """
    mdir = find_mask_folder(brain_dir)
    pngs = sorted(mdir.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"Không có mask PNG trong {mdir}")

    tumor_area_total = 0
    et_area_total = 0
    slices_with_tumor = 0

    for p in pngs:
        m = imread_any(p, cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        if m.ndim == 3:
            # Nếu lỡ trỏ sang mask màu (Label1), cố đọc GRAY
            m = imread_any(p, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue

        m = m.astype(np.uint8)
        if (m > 0).any():
            slices_with_tumor += 1
            tumor_area_total += int((m > 0).sum())
            et_area_total    += int((m == 3).sum())

    return {
        "tumor_area_total": tumor_area_total,
        "et_area_total": et_area_total,
        "slices_with_tumor": slices_with_tumor,
        "has_tumor": int(tumor_area_total > 0),
        "has_et": int(et_area_total > 0),
    }


# ---------- Grade mapping ----------

def _read_brain_meta_case(brain_dir: Path) -> str:
    """
    Thử đọc tên case gốc từ meta trong Brain_xxx:
    - meta.json (key 'case_name')
    - _source_case.txt (1 dòng tên case)
    Không có thì trả ''.
    """
    meta_json = brain_dir / "meta.json"
    if meta_json.exists():
        try:
            jd = json.loads(meta_json.read_text(encoding="utf-8"))
            case_name = jd.get("case_name", "") or jd.get("case", "")
            if isinstance(case_name, str) and case_name.strip():
                return case_name.strip()
        except Exception:
            pass

    source_txt = brain_dir / "_source_case.txt"
    if source_txt.exists():
        try:
            line = source_txt.read_text(encoding="utf-8").strip()
            if line:
                return line
        except Exception:
            pass

    return ""


def _sorted_raw_cases() -> List[str]:
    """Danh sách case raw (đã sort) trong RAW_TRAIN_DIR, ví dụ 'BraTS20_Training_001'."""
    if not RAW_TRAIN_DIR.exists():
        return []
    cases = [d.name for d in RAW_TRAIN_DIR.iterdir()
             if d.is_dir() and d.name.startswith("BraTS20_Training_")]
    return sorted(cases)


def _fallback_map_brain_to_case(brains: List[Path]) -> Dict[str, str]:
    """
    Fallback: ánh xạ theo thứ tự — Brain_001 ↔ case thứ 1 (sort theo tên) …,
    giả định bạn đã export 2D theo đúng thứ tự case raw (script trước đó làm như vậy).
    """
    raw_cases_sorted = _sorted_raw_cases()
    mapping = {}
    if len(raw_cases_sorted) < len(brains):
        # vẫn map theo min(len))
        n = min(len(raw_cases_sorted), len(brains))
    else:
        n = len(brains)
    for i in range(n):
        mapping[brains[i].name] = raw_cases_sorted[i]
    return mapping


def load_grade_mapping() -> Dict[str, str]:
    """
    Đọc name_mapping.csv và trả dict: {case_name: Grade}
    Cố gắng dò tên cột chứa case_name & Grade linh hoạt.
    """
    if not NAME_MAPPING_CSV.exists():
        return {}

    df = pd.read_csv(NAME_MAPPING_CSV)
    # Tìm cột grade
    grade_col = None
    for c in df.columns:
        if c.strip().lower() == "grade":
            grade_col = c
            break
    if grade_col is None:
        # không có grade
        return {}

    # Tìm cột case_name
    cand_cols = ["BraTS_2020_subject_ID", "BraTS20ID", "BraTS_ID", "Case", "case", "Name", "name", "Subject", "subject"]
    case_col = None
    for c in df.columns:
        if c in cand_cols:
            case_col = c
            break
    if case_col is None:
        # thử suy diễn: chọn cột có giá trị giống pattern 'BraTS20_Training_'
        for c in df.columns:
            vals = df[c].astype(str)
            if vals.str.startswith("BraTS20_Training_").any():
                case_col = c
                break
    if case_col is None:
        return {}

    mapping = {}
    for _, r in df.iterrows():
        name = str(r[case_col]).strip()
        grade = str(r[grade_col]).strip()
        if name:
            mapping[name] = grade
    return mapping


def attach_grade(df_cases: pd.DataFrame, brain_to_case: Dict[str, str], case_to_grade: Dict[str, str]) -> pd.DataFrame:
    df = df_cases.copy()
    grades = []
    raw_cases = []
    for brain in df["brain"]:
        case_name = _read_brain_meta_case(Path(ROOT / LABELED_REL / brain))
        if not case_name:
            case_name = brain_to_case.get(brain, "")
        raw_cases.append(case_name if case_name else "Unknown")

        g = case_to_grade.get(case_name, "Unknown") if case_name else "Unknown"
        grades.append(g if g in ("HGG", "LGG") else ("Unknown" if g else "Unknown"))

    df["raw_case"] = raw_cases
    df["grade"] = grades
    return df


# ---------- Stratify helpers ----------

def build_size_bins(areas: np.ndarray, min_per_bin: int = 10) -> Tuple[np.ndarray, List[str]]:
    """
    Phân nhóm kích thước theo quantile: thử 3-bin → nếu thiếu mẫu, 2-bin → nếu vẫn thiếu, 1-bin.
    Trả về chỉ số bin (0..K-1) và tên bin.
    """
    a = areas.astype(np.float64)
    if np.all(a == 0):
        return np.zeros_like(a, dtype=int), ["all_zero"]

    # 3-bin
    q1, q2 = np.quantile(a, [1/3, 2/3])
    bins3 = np.digitize(a, [q1, q2], right=False)  # 0,1,2
    if min((bins3 == i).sum() for i in (0, 1, 2)) >= min_per_bin:
        return bins3, ["small", "medium", "large"]

    # 2-bin
    q = np.quantile(a, 0.5)
    bins2 = np.digitize(a, [q], right=False)  # 0,1
    if min((bins2 == i).sum() for i in (0, 1)) >= min_per_bin:
        return bins2, ["small", "large"]

    # 1-bin
    return np.zeros_like(a, dtype=int), ["all"]


def make_stratify_labels(df: pd.DataFrame, min_per_stratum: int = 8) -> Tuple[pd.Series, List[str]]:
    """
    Nhãn stratify ưu tiên đủ mạnh:
      1) grade × has_et × size_bin
      2) grade × has_et
      3) has_et
      4) all
    """
    bins_idx, bin_names = build_size_bins(df["tumor_area_total"].values, min_per_bin=min_per_stratum)
    tmp = df.copy()
    tmp["size_bin_idx"] = bins_idx

    # Cấp 1
    tmp["strat"] = (
        tmp["grade"].astype(str) + "_" +
        tmp["has_et"].astype(str) + "_" +
        tmp["size_bin_idx"].astype(str)
    )
    counts = tmp["strat"].value_counts()
    if counts.empty or counts.min() < min_per_stratum:
        # Cấp 2
        tmp["strat"] = tmp["grade"].astype(str) + "_" + tmp["has_et"].astype(str)
        counts = tmp["strat"].value_counts()
        if counts.empty or counts.min() < min_per_stratum:
            # Cấp 3
            tmp["strat"] = tmp["has_et"].astype(str)
            counts = tmp["strat"].value_counts()
            if counts.empty or counts.min() < min_per_stratum:
                # Cấp 4
                tmp["strat"] = "all"

    return tmp["strat"], bin_names


def main():
    # Kiểm tra tỉ lệ
    assert abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) < 1e-6, "Tổng tỉ lệ phải = 1.0"

    labeled_dir = ROOT / LABELED_REL
    out_dir = ROOT / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Liệt kê Brain_xxx
    brains = sorted(d for d in labeled_dir.iterdir() if d.is_dir() and d.name.lower().startswith("brain_"))
    if not brains:
        raise FileNotFoundError(f"Không tìm thấy Brain_xxx trong: {labeled_dir}")

    # 2) Tính đặc trưng stratify từ mask
    rows = []
    for b in brains:
        try:
            feats = load_mask_stats_for_case(b)
        except Exception as e:
            print(f"[WARN] Bỏ qua {b.name}: {e}")
            continue
        rows.append({"brain": b.name, **feats})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Không thu được đặc trưng nào để stratify.")

    # 3) Gán Grade
    case_to_grade = load_grade_mapping()
    brain_to_case = {}

    # Ưu tiên: đọc meta trong mỗi Brain_xxx
    for b in brains:
        case_name = _read_brain_meta_case(b)
        if case_name:
            brain_to_case[b.name] = case_name

    # Fallback theo thứ tự nếu chưa đủ mapping
    if len(brain_to_case) < len(brains):
        fallback_map = _fallback_map_brain_to_case(brains)
        for k, v in fallback_map.items():
            brain_to_case.setdefault(k, v)  # không ghi đè meta

    df = attach_grade(df, brain_to_case, case_to_grade)

    # 4) Tạo nhãn stratify tổng hợp
    strat_labels, bin_names = make_stratify_labels(df, min_per_stratum=MIN_PER_STRAT)
    df["strat_label"] = strat_labels

    ids = df["brain"].values
    strat = df["strat_label"].values

    # 5) Chia 70/15/15 (2 bước)
    ids_train, ids_temp, strat_train, strat_temp = train_test_split(
        ids, strat,
        test_size=(1.0 - TRAIN_RATIO),
        random_state=SEED,
        stratify=strat
    )
    temp_val_ratio = VAL_RATIO / (VAL_RATIO + TEST_RATIO + 1e-12)
    ids_val, ids_test, _, _ = train_test_split(
        ids_temp, strat_temp,
        test_size=(1.0 - temp_val_ratio),
        random_state=SEED,
        stratify=strat_temp
    )

    # 6) Gán nhãn split
    df["split"] = "none"
    df.loc[df["brain"].isin(ids_train), "split"] = "train"
    df.loc[df["brain"].isin(ids_val),   "split"] = "val"
    df.loc[df["brain"].isin(ids_test),  "split"] = "test"

    # 7) Lưu kết quả
    df_sorted = df.sort_values(["split", "brain"]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = out_dir / "splits.csv"
    json_path = out_dir / "splits.json"
    txt_train = out_dir / "train.txt"
    txt_val   = out_dir / "val.txt"
    txt_test  = out_dir / "test.txt"

    df_sorted.to_csv(csv_path, index=False)

    payload = {
        "seed": SEED,
        "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "n_total": int(len(ids)),
        "n_train": int(len(ids_train)),
        "n_val": int(len(ids_val)),
        "n_test": int(len(ids_test)),
        "stratify_bins": sorted(df_sorted["strat_label"].unique().tolist()),
        "notes": "Stratify ưu tiên: grade×hasET×size_bin → grade×hasET → hasET → all. size_bin bằng quantile trên tumor_area_total.",
        "train": sorted(map(str, ids_train)),
        "val":   sorted(map(str, ids_val)),
        "test":  sorted(map(str, ids_test)),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    for path, id_list in ((txt_train, ids_train), (txt_val, ids_val), (txt_test, ids_test)):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(map(str, id_list))) + "\n")

    # In tóm tắt
    print("[OK] Saved splits to:")
    print(" -", csv_path)
    print(" -", json_path)
    print(" -", txt_train)
    print(" -", txt_val)
    print(" -", txt_test)
    print(f"[INFO] Summary: train={len(ids_train)} | val={len(ids_val)} | test={len(ids_test)} / total={len(ids)}")
    # Phân bố grade giúp hậu kiểm
    print(df_sorted.groupby(["split", "grade"]).size())


if __name__ == "__main__":
    main()
