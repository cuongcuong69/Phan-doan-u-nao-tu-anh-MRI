# -*- coding: utf-8 -*-
"""
BraTS2020 — Cohort-level Statistical Analysis with PCA / UMAP / LDA
(v3: thêm LDA 2D/3D, sửa lỗi fillna với Categorical)

Đầu vào (đã có từ các scripts trước):
  D:\Project Advanced CV\experiments\eda\
    ├─ shape\brats_shape_stats.csv
    ├─ intensity_modality\intensity_modality_means_per_case.csv
    └─ radiomics\radiomics_texture_per_case.csv
  D:\Project Advanced CV\data\BraST2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\
    ├─ survival_info.csv           (Brats20ID, Age, Survival_days, Extent_of_Resection)
    └─ name_mapping.csv            (cột Grade: HGG/LGG)

Chức năng:
  1) Gộp bảng đặc trưng (shape + intensity + radiomics) theo 'case'
  2) Gắn metadata: Grade, Age, OS_days, EOR
  3) Định nghĩa outcome nhóm theo OS: median_os (2 nhóm) / tertile_os (3 nhóm)
  4) So sánh nhóm: t-test / Welch t-test / Mann–Whitney U (+ FDR BH, effect size)
  5) Trực quan:
      - Boxplots top-N features theo q-value
      - Correlation heatmap (top-30 đa dạng)
      - PCA 2D/3D scatter
      - UMAP 2D scatter (nếu có umap-learn)
      - LDA 2D/3D scatter (nếu số lớp >= 2; LDA tối đa k-1 chiều)
  6) Xuất bảng completeness

Cách chạy:
    python "D:\Project Advanced CV\scripts\eda_brats_cohort_stats_v3.py"
    python "D:\Project Advanced CV\scripts\eda_brats_cohort_stats_v3.py" --outcome tertile_os
    python "D:\Project Advanced CV\scripts\eda_brats_cohort_stats_v3.py" --color-by grade
    python "D:\Project Advanced CV\scripts\eda_brats_cohort_stats_v3.py" --min-complete 40
"""

import argparse
from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ---------------- CONFIG ----------------
EXP_ROOT = r"D:\Project Advanced CV\experiments\eda"
RAW_TRAIN = r"D:\Project Advanced CV\data\BraST2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"

OUT_DIR = r"D:\Project Advanced CV\experiments\eda\cohort"

# Feature tables
SHAPE_CSV     = Path(EXP_ROOT) / "shape" / "brats_shape_stats.csv"
INTENSITY_CSV = Path(EXP_ROOT) / "intensity_modality" / "intensity_modality_means_per_case.csv"
RADS_CSV      = Path(EXP_ROOT) / "radiomics" / "radiomics_texture_per_case.csv"

# Metadata
SURVIVAL_CSV  = Path(RAW_TRAIN) / "survival_info.csv"
NAME_MAP_CSV  = Path(RAW_TRAIN) / "name_mapping.csv"

# Plot filenames
BOXPLOT_PDF          = "boxplots_top_features_by_outcome.pdf"
CORR_HEATMAP_PNG     = "corr_heatmap_top30.png"
PCA2_PNG             = "pca_2d_scatter.png"
PCA3_PNG             = "pca_3d_scatter.png"
UMAP2_PNG            = "umap_2d_scatter.png"
LDA2_PNG             = "lda_2d_scatter.png"
LDA3_PNG             = "lda_3d_scatter.png"

# Params
TOP_N_BOX = 12
TOP_N_CORR = 30
RANDOM_STATE = 42
# ---------------------------------------


def _find_case_id(name: str) -> str:
    m = re.search(r"_(\d+)$", str(name))
    return m.group(1) if m else str(name)


def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        warnings.warn(f"Missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_features() -> pd.DataFrame:
    """Merge shape + intensity + radiomics by 'case'."""
    df_shape = _read_csv_safe(SHAPE_CSV)
    df_int   = _read_csv_safe(INTENSITY_CSV)
    df_rad   = _read_csv_safe(RADS_CSV)

    for df in [df_shape, df_int, df_rad]:
        if df.empty: 
            continue
        if "case" not in df.columns:
            cands = [c for c in df.columns if str(c).lower() in ["subject","name","case_id","folder","path","case"]]
            if cands:
                df.rename(columns={cands[0]: "case"}, inplace=True)
        if "id" not in df.columns and "case" in df.columns:
            df["id"] = df["case"].map(_find_case_id)

    base = None
    for df in [df_shape, df_int, df_rad]:
        if df.empty: 
            continue
        cols = [c for c in df.columns if c != "id"]
        base = df.copy() if base is None else pd.merge(base, df[cols], on="case", how="outer")

    if base is None:
        return pd.DataFrame()
    if "id" not in base.columns:
        base["id"] = base["case"].map(_find_case_id)
    return base


def _load_metadata() -> pd.DataFrame:
    """Read survival_info.csv (Brats20ID, Age, Survival_days, Extent_of_Resection) + name_mapping.csv (Grade)."""
    surv = _read_csv_safe(SURVIVAL_CSV)
    nm   = _read_csv_safe(NAME_MAP_CSV)

    if not surv.empty:
        # normalize cols
        cols_lower = {c.lower(): c for c in surv.columns}
        id_col = "Brats20ID" if "Brats20ID" in surv.columns else cols_lower.get("brats20id", list(surv.columns)[0])
        surv = surv.rename(columns={id_col: "case"})
        if "Age" not in surv.columns:
            for c in surv.columns:
                if c.lower() == "age":
                    surv = surv.rename(columns={c: "Age"}); break
        if "Survival_days" not in surv.columns:
            for c in surv.columns:
                if c.lower() in ["survival_days","os_days","os","days"]:
                    surv = surv.rename(columns={c: "Survival_days"}); break
        if "Extent_of_Resection" not in surv.columns:
            for c in surv.columns:
                if c.lower() in ["extent_of_resection","eor","resection"]:
                    surv = surv.rename(columns={c: "Extent_of_Resection"}); break

        surv["case"] = surv["case"].astype(str)
        surv["case"] = surv["case"].str.replace("BRATS","BraTS", case=False, regex=False)
        surv["OS_days"] = pd.to_numeric(surv["Survival_days"], errors="coerce")
        surv["Age"] = pd.to_numeric(surv.get("Age", np.nan), errors="coerce")
        surv["EOR"] = surv.get("Extent_of_Resection", np.nan)
        if "EOR" in surv.columns:
            surv["EOR"] = surv["EOR"].astype(str).str.upper()
        meta_sv = surv[["case","OS_days","Age","EOR"]].copy()
    else:
        meta_sv = pd.DataFrame(columns=["case","OS_days","Age","EOR"])

    if not nm.empty:
        case_col_nm = None
        for c in nm.columns:
            if nm[c].astype(str).str.contains("BraTS20_Training_", case=False, na=False).any():
                case_col_nm = c; break
        case_col_nm = case_col_nm or nm.columns[0]
        nm = nm.rename(columns={case_col_nm: "case"})
        nm["Grade"] = nm["Grade"].astype(str).str.upper() if "Grade" in nm.columns else np.nan
        meta_nm = nm[["case","Grade"]].copy()
    else:
        meta_nm = pd.DataFrame(columns=["case","Grade"])

    meta = pd.merge(meta_sv, meta_nm, on="case", how="outer")
    return meta


def _define_groups(df: pd.DataFrame, outcome="median_os"):
    df = df.copy()
    if "OS_days" not in df.columns:
        df["outcome_group"] = np.nan
        return df
    if outcome == "median_os":
        q50 = df["OS_days"].median(skipna=True)
        df["outcome_group"] = np.where(df["OS_days"] >= q50, "OS_High", "OS_Low")
    elif outcome == "tertile_os":
        try:
            df["outcome_group"] = pd.qcut(df["OS_days"], 3, labels=["Low","Mid","High"])
            # tránh Categorical gây fillna lỗi
            df["outcome_group"] = df["outcome_group"].astype("object")
        except Exception:
            warnings.warn("Cannot compute tertiles; fallback to median_os")
            return _define_groups(df, "median_os")
    else:
        df["outcome_group"] = np.nan
    return df


def _select_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    ex = ["case","id","Grade","OS_days","Age","EOR","outcome_group"]
    num_cols = [c for c in df.columns if c not in ex and pd.api.types.is_numeric_dtype(df[c])]
    return df[["case","Grade","OS_days","Age","EOR","outcome_group"] + num_cols].copy()


def _effect_sizes_ttest(x, y):
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2)) if (nx+ny-2) > 0 else np.nan
    return (np.mean(x) - np.mean(y)) / (sp + 1e-12)


def _cliffs_delta(x, y):
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0: return np.nan
    x_sorted = np.sort(x); y_sorted = np.sort(y)
    i = j = gt = lt = 0
    while i < nx and j < ny:
        if x_sorted[i] > y_sorted[j]:
            gt += (nx - i); j += 1
        elif x_sorted[i] < y_sorted[j]:
            lt += (ny - j); i += 1
        else:
            i += 1; j += 1
    return (gt - lt) / (nx*ny)


def _bh_fdr(pvals: pd.Series):
    p = pvals.values
    n = len(p)
    order = np.argsort(p)
    ranked = np.empty(n, dtype=float)
    cummin = 1.0
    for k, idx in enumerate(order[::-1], start=1):
        q = p[idx] * n / (n - k + 1)
        cummin = min(cummin, q)
        ranked[idx] = cummin
    return pd.Series(ranked, index=pvals.index)


def run_group_stats(df: pd.DataFrame, group_col: str, alpha=0.05):
    df = df.copy()
    groups = df[group_col].dropna().unique().tolist()
    if len(groups) < 2:
        return pd.DataFrame(), []
    if len(groups) > 2:
        counts = df[group_col].value_counts().index.tolist()
        groups = counts[:2]
    g1, g2 = groups[0], groups[1]

    ex = ["case","id","Grade","OS_days","Age","EOR",group_col]
    feats = [c for c in df.columns if c not in ex and pd.api.types.is_numeric_dtype(df[c])]
    res = []
    for f in feats:
        x = df.loc[df[group_col]==g1, f].dropna().values
        y = df.loc[df[group_col]==g2, f].dropna().values
        if len(x) < 5 or len(y) < 5:
            continue
        try:
            p_norm_x = stats.shapiro(x).pvalue if len(x) <= 5000 else 0.5
            p_norm_y = stats.shapiro(y).pvalue if len(y) <= 5000 else 0.5
        except Exception:
            p_norm_x = p_norm_y = 0.0
        normal = (p_norm_x > 0.05) and (p_norm_y > 0.05)
        try:
            p_lev = stats.levene(x, y, center='median').pvalue
        except Exception:
            p_lev = 0.0
        equal_var = p_lev > 0.05

        if normal:
            t = stats.ttest_ind(x, y, equal_var=equal_var, alternative='two-sided')
            pval = t.pvalue
            eff = _effect_sizes_ttest(x, y)
            test = "t" if equal_var else "welch-t"
        else:
            u = stats.mannwhitneyu(x, y, alternative='two-sided')
            pval = u.pvalue
            eff = _cliffs_delta(x, y)
            test = "mannwhitney"

        res.append({
            "feature": f, "group1": g1, "group2": g2,
            "n1": len(x), "n2": len(y),
            "mean1": np.mean(x), "mean2": np.mean(y),
            "median1": np.median(x), "median2": np.median(y),
            "test": test, "pval": pval, "effect": eff,
            "normal": normal, "equal_var": equal_var
        })

    out = pd.DataFrame(res).sort_values("pval")
    if out.empty:
        return out, []
    out["qval"] = _bh_fdr(out["pval"])
    top_feats = out.nsmallest(TOP_N_BOX, "qval")["feature"].tolist()
    return out, top_feats


def _prepare_matrix(df: pd.DataFrame, feat_cols: list) -> np.ndarray:
    X = df[feat_cols].copy()
    imp = SimpleImputer(strategy="median")
    Ximp = imp.fit_transform(X)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(Ximp)
    return Xz


def safe_fillna_str(s: pd.Series, fill="NA"):
    # đảm bảo fillna hoạt động kể cả khi s là Categorical
    import pandas as pd
    if pd.api.types.is_categorical_dtype(s):
        if fill not in s.cat.categories:
            s = s.cat.add_categories([fill])
        return s.fillna(fill)
    return s.astype("object").fillna(fill)


def plot_boxplots(df: pd.DataFrame, features: list, group_col: str, out_pdf: Path):
    if not features: return
    per_page = 6
    pages = (len(features) + per_page - 1) // per_page
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out_pdf) as pdf:
        for p in range(pages):
            cols = features[p*per_page:(p+1)*per_page]
            if not cols: break
            n = len(cols)
            nrows = int(np.ceil(n/3))
            ncols = min(3, n)
            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.8*nrows))
            axes = np.atleast_1d(axes).ravel()
            for ax, f in zip(axes, cols):
                sns.boxplot(data=df, x=group_col, y=f, ax=ax)
                sns.stripplot(data=df, x=group_col, y=f, ax=ax, color="k", alpha=0.35, size=3)
                ax.set_title(f)
                ax.grid(axis="y", alpha=0.2, linestyle="--")
            for k in range(len(cols), len(axes)):
                fig.delaxes(axes[k])
            fig.tight_layout()
            pdf.savefig(fig, dpi=200)
            plt.close(fig)


def plot_corr_heatmap(df: pd.DataFrame, out_png: Path, top_n=30):
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) == 0: return
    var = df[num_cols].var().sort_values(ascending=False)
    cand = var.index.tolist()[:max(top_n*3, top_n)]
    sub = df[cand].copy()
    corr = sub.corr().abs()
    keep, used = [], set()
    for c in corr.columns:
        if c in used: continue
        keep.append(c)
        used.update(set(corr.index[corr[c] > 0.98].tolist()))
    keep = keep[:top_n]
    sub = sub[keep]
    corr = sub.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, annot=False, square=True)
    plt.title("Correlation heatmap (top features)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_pca(df: pd.DataFrame, feat_cols: list, color_series: pd.Series, out2: Path, out3: Path):
    if len(feat_cols) < 3: return
    Xz = _prepare_matrix(df, feat_cols)
    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    Z = pca.fit_transform(Xz)
    # 2D
    plt.figure(figsize=(6,5))
    plt.scatter(Z[:,0], Z[:,1], c=pd.Categorical(color_series).codes, s=16, alpha=0.9)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("PCA 2D")
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out2, dpi=200)
    plt.close()
    # 3D
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Z[:,0], Z[:,1], Z[:,2], c=pd.Categorical(color_series).codes, s=18, depthshade=True)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
    ax.set_title("PCA 3D")
    plt.tight_layout()
    plt.savefig(out3, dpi=220)
    plt.close()


def plot_umap(df: pd.DataFrame, feat_cols: list, color_series: pd.Series, out_png: Path):
    try:
        import umap
    except Exception:
        warnings.warn("umap-learn not installed; skip UMAP.")
        return
    if len(feat_cols) < 2: return
    Xz = _prepare_matrix(df, feat_cols)
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean",
                        random_state=RANDOM_STATE)
    U = reducer.fit_transform(Xz)
    plt.figure(figsize=(6,5))
    plt.scatter(U[:,0], U[:,1], c=pd.Categorical(color_series).codes, s=16, alpha=0.9)
    plt.title("UMAP 2D")
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_lda(df: pd.DataFrame, feat_cols: list, labels: pd.Series, out2: Path, out3: Path):
    """
    LDA 2D/3D. Số chiều tối đa = min(n_classes - 1, n_features).
    - Nếu chỉ có 2 lớp → chỉ 1 component → vẽ 2D bằng (LD1, 0), 3D bằng (LD1, 0, 0).
    """
    # Chuẩn hóa nhãn (string) và bỏ NA
    y = safe_fillna_str(labels, "NA")
    valid = y != "NA"
    if valid.sum() < 5:
        warnings.warn("Not enough labeled samples for LDA; skip.")
        return

    y = y[valid]
    X = df.loc[valid, feat_cols].copy()
    if X.shape[0] <= 5 or X.shape[1] < 1:
        return

    # Impute + scale
    imp = SimpleImputer(strategy="median")
    Ximp = imp.fit_transform(X)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(Ximp)

    classes = pd.Categorical(y).categories.tolist()
    n_classes = len(classes)
    if n_classes < 2:
        warnings.warn("LDA needs >=2 classes; skip.")
        return

    max_comp = min(n_classes - 1, 3)
    lda = LinearDiscriminantAnalysis(n_components=max_comp, solver="svd")
    Z = lda.fit_transform(Xz, y)

    # Nếu chỉ có 1 component, pad để vẽ 2D/3D
    if Z.shape[1] == 1:
        Z2 = np.hstack([Z, np.zeros((Z.shape[0], 1))])
        Z3 = np.hstack([Z, np.zeros((Z.shape[0], 2))])
    elif Z.shape[1] == 2:
        Z2 = Z
        Z3 = np.hstack([Z, np.zeros((Z.shape[0], 1))])
    else:
        Z2 = Z[:, :2]
        Z3 = Z[:, :3]

    # 2D
    plt.figure(figsize=(6,5))
    plt.scatter(Z2[:,0], Z2[:,1], c=pd.Categorical(y).codes, s=18, alpha=0.9)
    comp_names = ["LD1","LD2"][:Z2.shape[1]]
    while len(comp_names) < 2: comp_names.append("LD0")
    plt.xlabel(comp_names[0]); plt.ylabel(comp_names[1])
    plt.title(f"LDA 2D (classes={n_classes})")
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out2, dpi=200)
    plt.close()

    # 3D
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Z3[:,0], Z3[:,1], Z3[:,2], c=pd.Categorical(y).codes, s=20, depthshade=True)
    comp_names3 = ["LD1","LD2","LD3"][:Z3.shape[1]]
    while len(comp_names3) < 3: comp_names3.append("LD0")
    ax.set_xlabel(comp_names3[0]); ax.set_ylabel(comp_names3[1]); ax.set_zlabel(comp_names3[2])
    ax.set_title(f"LDA 3D (classes={n_classes})")
    plt.tight_layout()
    plt.savefig(out3, dpi=220)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", type=str, default="median_os",
                    choices=["median_os","tertile_os"],
                    help="Định nghĩa outcome nhóm theo OS.")
    ap.add_argument("--color-by", type=str, default="outcome",
                    choices=["outcome","grade","eor"],
                    help="Tô màu scatter theo outcome/grade/EOR (dùng cho PCA/UMAP/LDA).")
    ap.add_argument("--min-complete", type=int, default=30,
                    help="Giữ các features có >=min-complete giá trị không NaN.")
    args = ap.parse_args()

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load features
    feats = _load_features()
    if feats.empty:
        raise SystemExit("No feature tables found. Hãy chạy scripts shape/intensity/radiomics trước.")

    # 2) Metadata
    meta = _load_metadata()

    # 3) Join & define outcome groups
    df = pd.merge(feats, meta, on="case", how="left")
    df = _define_groups(df, outcome=args.outcome)

    # 4) Numeric features + completeness
    df_all = _select_numeric_features(df)
    keep_num = [c for c in df_all.columns if pd.api.types.is_numeric_dtype(df_all[c])]
    completeness = df_all[keep_num].notna().sum()
    keep_num = [c for c in keep_num if completeness[c] >= args.min_complete]
    df_num = pd.concat([df_all[["case","Grade","OS_days","Age","EOR","outcome_group"]], df_all[keep_num]], axis=1)

    # 5) Group comparison
    stats_tbl, top_box_feats = run_group_stats(df_num, "outcome_group")
    if not stats_tbl.empty:
        stats_tbl.to_csv(out_dir / "group_stats.csv", index=False)

    # 6) Plots
    # 6.1 Boxplots
    plot_boxplots(df_num, top_box_feats, "outcome_group", out_dir / BOXPLOT_PDF)

    # 6.2 Corr heatmap
    numeric_cols = [c for c in df_num.columns if c not in ["case","Grade","OS_days","Age","EOR","outcome_group"]
                    and pd.api.types.is_numeric_dtype(df_num[c])]
    plot_corr_heatmap(df_num[numeric_cols], out_dir / CORR_HEATMAP_PNG, top_n=TOP_N_CORR)

    # 6.3 Choose label series for coloring/lda target
    if args.color_by == "grade":
        color_series = safe_fillna_str(df_num["Grade"])
    elif args.color_by == "eor":
        color_series = safe_fillna_str(df_num["EOR"])
    else:
        color_series = safe_fillna_str(df_num["outcome_group"])

    # 6.4 Feature set for embeddings
    feat_cols = numeric_cols
    if feat_cols:
        # lọc phương sai ~0
        vt = VarianceThreshold(threshold=1e-12)
        try:
            _ = vt.fit_transform(df_num[feat_cols].fillna(df_num[feat_cols].median()))
            feat_cols = list(pd.Index(feat_cols)[vt.get_support()])
        except Exception:
            pass

        # PCA
        plot_pca(df_num, feat_cols, color_series, out_dir / PCA2_PNG, out_dir / PCA3_PNG)
        # UMAP
        plot_umap(df_num, feat_cols, color_series, out_dir / UMAP2_PNG)
        # LDA
        plot_lda(df_num, feat_cols, color_series, out_dir / LDA2_PNG, out_dir / LDA3_PNG)

    # 7) Completeness table
    comp_df = pd.DataFrame({
        "feature": numeric_cols,
        "non_null": [df_num[c].notna().sum() for c in numeric_cols]
    }).sort_values("non_null", ascending=False)
    comp_df.to_csv(out_dir / "feature_completeness.csv", index=False)

    # Summary
    n_cases = df_num["case"].nunique()
    print(f"[OK] Cohort stats v3 done. Cases: {n_cases}")
    print(f" - Output dir: {out_dir}")
    if not stats_tbl.empty:
        print(f" - group_stats.csv: {len(stats_tbl)} rows")
        print(f" - Boxplots (top {len(top_box_feats)} feats): {BOXPLOT_PDF}")
    print(" - Correlation heatmap:", CORR_HEATMAP_PNG)
    print(" - PCA 2D/3D:", PCA2_PNG, ",", PCA3_PNG)
    print(" - UMAP 2D:", UMAP2_PNG, "(nếu umap-learn có sẵn)")
    print(" - LDA 2D/3D:", LDA2_PNG, ",", LDA3_PNG)
    print(" - Feature completeness: feature_completeness.csv")


if __name__ == "__main__":
    main()
