# -*- coding: utf-8 -*-
"""
BraTS2020 — Radiomics Texture Features (3D) theo ROI & modality.

ROI:
  - NCR (label=1), ED (label=2), ET (label=4)

Modality:
  - FLAIR, T1, T1CE, T2

Features:
  - First-order (histogram): mean, variance, skewness, kurtosis, entropy, ...
  - GLCM: contrast, homogeneity, correlation, ...
  - GLRLM, GLSZM, NGTDM: run-length, zone-size, neighboring gray tone, ...
  (Có thể bật toàn bộ features bằng --all_features)

Cách làm:
  - Dùng PyRadiomics (SimpleITK) đọc NIfTI 3D.
  - Thực thi extractor.execute(image, mask, label=<1|2|4>) cho từng (modality, ROI).
  - Lọc/đặt tên cột: {mod}_{roi}_{feature} (ví dụ: t1_ET_glcm_Contrast).

Yêu cầu:
    pip install pyradiomics SimpleITK nibabel numpy pandas tqdm

Kết quả:
    CSV: D:\Project Advanced CV\experiments\eda\radiomics\radiomics_texture_per_case.csv
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm import tqdm


# ---------------- CONFIG ----------------
RAW_TRAIN = r"D:\Project Advanced CV\data\BraST2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
OUT_DIR   = r"D:\Project Advanced CV\experiments\eda\radiomics"
CSV_OUT   = "radiomics_texture_per_case.csv"

MODS = ["flair", "t1", "t1ce", "t2"]
ROI_MAP = {1: "NCR", 2: "ED", 4: "ET"}  # BraTS labels
# ---------------------------------------


def find_case_id(name: str) -> str:
    m = re.search(r"_(\d+)$", name)
    return m.group(1) if m else name


def build_extractor(all_features: bool = False, include_shape2d: bool = False):
    import logging, radiomics
    # Giảm log ồn
    radiomics.setVerbosity(logging.ERROR)

    settings = {
        "binWidth": 25,
        "force2D": bool(include_shape2d),   # chỉ bật 2D nếu thật sự muốn shape2D
        "force2Ddimension": 0,
        "normalize": False,
        "symmetricalGLCM": True,
        "correctMask": True,
    }
    ext = featureextractor.RadiomicsFeatureExtractor(**settings)

    # Chỉ dùng ảnh gốc
    ext.disableAllImageTypes()
    ext.enableImageTypeByName("Original")

    # Chiến lược an toàn: luôn tắt hết rồi CHỌN BẬT những class cần
    ext.disableAllFeatures()

    if all_features:
        # “all features” theo nghĩa: bật toàn bộ NHÓM 3D chính + shape 3D; KHÔNG bật shape2D trừ khi --include-shape2d
        to_enable = ["firstorder", "glcm", "glrlm", "glszm", "ngtdm", "gldm", "shape"]
        if include_shape2d:
            to_enable.append("shape2D")
    else:
        # gọn nhẹ cho texture: không bật shape/shape2D
        to_enable = ["firstorder", "glcm", "glrlm", "glszm", "ngtdm"]
        # nếu muốn thêm gldm:
        # to_enable.append("gldm")

    for cls in to_enable:
        try:
            ext.enableFeatureClassByName(cls)
        except Exception:
            # Một số phiên bản cũ có thể thiếu class; bỏ qua nếu không có
            pass

    return ext



def get_paths_for_case(case_dir: Path) -> dict:
    """Trả về đường dẫn NIfTI cho 4 modality + seg."""
    name = case_dir.name
    fps = {
        "flair": case_dir / f"{name}_flair.nii",
        "t1":    case_dir / f"{name}_t1.nii",
        "t1ce":  case_dir / f"{name}_t1ce.nii",
        "t2":    case_dir / f"{name}_t2.nii",
        "seg":   case_dir / f"{name}_seg.nii",
    }
    for k, p in fps.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing file for {k}: {p}")
    return fps


def execute_one(extractor: featureextractor.RadiomicsFeatureExtractor,
                img_path: Path, mask_path: Path, roi_label: int) -> dict:
    """
    Chạy PyRadiomics cho 1 (image, mask, label).
    Trả về dict các đặc trưng (đã lọc các meta-keys).
    """
    # PyRadiomics chấp nhận SimpleITK Image hoặc đường dẫn
    img = sitk.ReadImage(str(img_path))
    msk = sitk.ReadImage(str(mask_path))

    res = extractor.execute(img, msk, label=roi_label)

    # Lọc bỏ các meta-keys (keys bắt đầu bằng 'diagnostics_')
    out = {k: v for k, v in res.items() if not k.startswith("diagnostics_")}
    return out


def flatten_keys(d: dict, mod: str, roi_name: str) -> dict:
    """
    Chuẩn hoá tên cột:
      'original_firstorder_Mean' -> 't1_ET_firstorder_Mean'
      'original_glcm_Contrast'   -> 't1_ET_glcm_Contrast'
    """
    out = {}
    for k, v in d.items():
        # Kỳ vọng key dạng 'original_<group>_<FeatureName>'
        if k.startswith("original_"):
            kk = k.replace("original_", "")
        else:
            kk = k
        col = f"{mod}_{roi_name}_{kk}"
        out[col] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_train", type=str, default=RAW_TRAIN,
                    help="Thư mục MICCAI_BraTS2020_TrainingData (có seg.nii)")
    ap.add_argument("--out_dir", type=str, default=OUT_DIR,
                    help="Thư mục xuất CSV")
    ap.add_argument("--all_features", action="store_true",
                    help="Bật nhiều nhóm feature (firstorder, glcm, glrlm, glszm, ngtdm, gldm, shape).")
    ap.add_argument("--include-shape2d", action="store_true",
                    help="(Hiếm khi cần) Bật shape2D theo lát; sẽ tự force2D=True và chạy chậm hơn.")
    ap.add_argument("--quiet", action="store_true", help="Giảm log PyRadiomics")
    args = ap.parse_args()
    if args.quiet:
        import logging, radiomics
        radiomics.setVerbosity(logging.ERROR)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / CSV_OUT

    extractor = build_extractor(all_features=args.all_features, include_shape2d=args.include_shape2d)

    root = Path(args.raw_train)
    cases = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("BraTS20_Training_")])

    rows = []
    for cdir in tqdm(cases, desc="Radiomics texture (training)"):
        try:
            fps = get_paths_for_case(cdir)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            continue

        row = {
            "case": cdir.name,
            "id": find_case_id(cdir.name),
        }

        # Với mỗi modality và mỗi ROI label
        for mod in MODS:
            for lbl, roi_name in ROI_MAP.items():
                try:
                    feats = execute_one(extractor, fps[mod], fps["seg"], lbl)
                except Exception as e:
                    # Nếu ROI không tồn tại (không có voxel), PyRadiomics có thể lỗi; set NaN cho nhóm này
                    # Hoặc xảy ra lỗi do dữ liệu — log cảnh báo và tiếp tục
                    print(f"[WARN] {cdir.name} | {mod} | ROI={roi_name} ({lbl}) -> {e}")
                    # tạo NaN placeholders cho các nhóm chính để schema không lệch
                    continue

                # Rút gọn tên cột & thêm prefix mod/roi
                feats_norm = flatten_keys(feats, mod, roi_name)
                row.update(feats_norm)

        rows.append(row)

    df = pd.DataFrame(rows)
    # Sắp theo id (chuỗi), giữ các cột case/id đầu tiên
    if not df.empty:
        df["id"] = df["id"].astype(str)
        df.sort_values("id", inplace=True)

    df.to_csv(csv_path, index=False)
    print(f"[OK] Saved radiomics texture features to: {csv_path}")
    print("Note: cột có dạng <mod>_<ROI>_<featureGroup>_<featureName>, ví dụ: t1_ET_glcm_Contrast.")


if __name__ == "__main__":
    main()
