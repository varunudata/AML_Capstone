"""
Stage 1 - Dataset Analysis (read-only)
=======================================
Loads the raw CSV, caches to parquet, and produces a comprehensive
dataset analysis report.

Usage:
    python -m src.stage1_analysis
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "Combined_dataset.csv"
PARQUET_CACHE = PROJECT_ROOT / "data" / "processed" / "raw_cached.parquet"
RESULTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Physical plausibility ranges for each chemistry variable
PLAUSIBLE_RANGES = {
    "Ammonia (mg/l)":                     (0, 50),
    "Biochemical Oxygen Demand (mg/l)":   (0, 100),
    "Dissolved Oxygen (mg/l)":            (0, 20),
    "Orthophosphate (mg/l)":              (0, 50),
    "pH (ph units)":                      (0, 14),
    "Temperature (cel)":                  (-5, 45),
    "Nitrogen (mg/l)":                    (0, 200),
    "Nitrate (mg/l)":                     (0, 200),
}

CHEMISTRY_COLS = list(PLAUSIBLE_RANGES.keys())
TARGET_CONT = "CCME_Values"
TARGET_CLASS = "CCME_WQI"
META_COLS = ["Country", "Area", "Waterbody Type", "Date"]


def load_data():
    """Load raw CSV with explicit dtypes, parse dates, cache to parquet."""
    if PARQUET_CACHE.exists():
        print(f"Loading cached parquet: {PARQUET_CACHE}")
        df = pd.read_parquet(PARQUET_CACHE)
    else:
        print(f"Loading raw CSV: {RAW_CSV} ...")
        dtype_map = {
            "Country": "category",
            "Area": "str",
            "Waterbody Type": "category",
        }
        # Chemistry + target columns as float
        for col in CHEMISTRY_COLS + [TARGET_CONT]:
            dtype_map[col] = "float64"
        dtype_map[TARGET_CLASS] = "category"

        df = pd.read_csv(
            RAW_CSV,
            dtype=dtype_map,
            parse_dates=["Date"],
            dayfirst=True,  # dd-mm-yyyy
        )
        # Cache
        PARQUET_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PARQUET_CACHE, index=False)
        print(f"Cached to {PARQUET_CACHE}")
    return df


def basic_info(df):
    """Shape, memory, dtypes, missingness, duplicates, unique counts."""
    info = OrderedDict()
    info["shape"] = df.shape
    info["memory_mb"] = round(df.memory_usage(deep=True).sum() / 1e6, 2)
    info["dtypes"] = df.dtypes.astype(str).to_dict()

    # Missingness
    miss = df.isnull().sum()
    info["missing_counts"] = miss.to_dict()
    info["total_missing"] = int(miss.sum())

    # Duplicates
    n_dup = df.duplicated().sum()
    info["exact_duplicates"] = int(n_dup)

    # Unique counts
    info["unique_counts"] = {col: int(df[col].nunique()) for col in df.columns}

    return info


def percentile_summary(df):
    """Per-column percentiles for numeric columns."""
    num_cols = CHEMISTRY_COLS + [TARGET_CONT]
    pcts = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    summary = df[num_cols].describe(percentiles=pcts).T
    return summary


def group_counts(df):
    """Rows per Country, Waterbody Type, Year, Area."""
    results = {}

    # Country
    c_counts = df["Country"].value_counts()
    results["by_country"] = c_counts.to_dict()
    results["country_df"] = pd.DataFrame([
        {"Country": str(c), "Rows": int(cnt), "Percentage": round(float(100 * cnt / len(df)), 2)}
        for c, cnt in c_counts.items()
    ])

    # Waterbody Type
    wb_counts = df["Waterbody Type"].value_counts()
    results["by_waterbody"] = wb_counts.to_dict()
    results["waterbody_df"] = pd.DataFrame([
        {"Waterbody Type": str(wb), "Rows": int(cnt), "Percentage": round(float(100 * cnt / len(df)), 2)}
        for wb, cnt in wb_counts.items()
    ])

    # Year
    df_year = df["Date"].dt.year
    year_counts = df_year.value_counts().sort_index()
    results["by_year"] = year_counts.to_dict()
    y_min, y_max = int(year_counts.index.min()), int(year_counts.index.max())
    results["year_range"] = (y_min, y_max)
    results["year_min"] = y_min
    results["year_max"] = y_max

    # Area (top 20 + summary stats)
    area_counts = df["Area"].value_counts()
    results["area_top20"] = area_counts.head(20).to_dict()
    results["unique_areas"] = int(area_counts.shape[0])
    results["areas_lt_10"] = int((area_counts < 10).sum())
    results["max_area_obs"] = int(area_counts.max())
    results["area_stats"] = {
        "n_distinct": int(area_counts.shape[0]),
        "max_rows": int(area_counts.max()),
        "max_area": area_counts.idxmax(),
        "median_rows": float(area_counts.median()),
        "mean_rows": round(float(area_counts.mean()), 2),
        "areas_lt_10_rows": int((area_counts < 10).sum()),
        "areas_lt_5_rows": int((area_counts < 5).sum()),
        "areas_eq_1_row": int((area_counts == 1).sum()),
    }

    return results


# Backwards compatibility alias
group_distributions = group_counts


def plausibility_check(df):
    """Check each chemistry variable against physical plausibility ranges."""
    results = {}
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        below = (df[col] < lo).sum()
        above = (df[col] > hi).sum()
        zero = (df[col] == 0).sum()
        negative = (df[col] < 0).sum()
        results[col] = {
            "below_min": int(below),
            "above_max": int(above),
            "exactly_zero": int(zero),
            "negative": int(negative),
            "plausible_range": f"[{lo}, {hi}]",
            "actual_min": float(df[col].min()),
            "actual_max": float(df[col].max()),
        }
    return results


def reverse_engineer_ccme_classes(df):
    """Derive exact thresholds between CCME_WQI classes from CCME_Values."""
    results = {}
    classes = df[TARGET_CLASS].cat.categories.tolist()
    results["classes_in_data"] = classes

    for cls in classes:
        subset = df[df[TARGET_CLASS] == cls][TARGET_CONT]
        results[cls] = {
            "count": int(len(subset)),
            "min": float(subset.min()),
            "max": float(subset.max()),
            "mean": round(float(subset.mean()), 4),
            "pct_of_total": round(100 * len(subset) / len(df), 2),
        }

    # Derive boundaries
    boundaries = {}
    for cls in classes:
        subset = df[df[TARGET_CLASS] == cls][TARGET_CONT]
        boundaries[cls] = (float(subset.min()), float(subset.max()))
    results["boundaries"] = boundaries

    # Count of CCME_Values == 100
    at_100 = (df[TARGET_CONT] == 100.0).sum()
    results["values_at_100"] = int(at_100)
    results["pct_at_100"] = round(100 * at_100 / len(df), 2)
    results["n_capped_100"] = int(at_100)
    results["pct_capped_100"] = round(100 * at_100 / len(df), 2)

    # Derived Class Thresholds summary DataFrame
    summary_rows = []
    for cls in classes:
        c_info = results[cls]
        summary_rows.append({
            "Class": cls,
            "Min CCME": c_info["min"],
            "Max CCME": c_info["max"],
            "Mean CCME": c_info["mean"],
            "Count": f"{c_info['count']:,}",
            "% of Dataset": f"{c_info['pct_of_total']}%"
        })
    results["summary_df"] = pd.DataFrame(summary_rows)
    results["country_class_pct"] = pd.crosstab(df["Country"], df[TARGET_CLASS], normalize="index") * 100

    return results


# Backwards compatibility alias
reverse_engineer_ccme = reverse_engineer_ccme_classes


def leakage_analysis(df):
    """
    Quantify how well simple features explain the target,
    given that CCME WQI is computed from the same chemistry variables.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_absolute_error

    results = {}

    # Use a sample for speed
    sample = df.dropna(subset=CHEMISTRY_COLS + [TARGET_CONT]).sample(
        n=min(500_000, len(df)), random_state=SEED
    )

    X = sample[CHEMISTRY_COLS].values
    y = sample[TARGET_CONT].values

    # Simple linear regression with raw chemistry features
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred = lr.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    results["linear_regression_raw_features"] = {
        "r2": round(r2, 4),
        "mae": round(mae, 4),
        "n_samples": len(sample),
        "features_used": CHEMISTRY_COLS,
    }

    # Correlation of each feature with target
    corrs_pearson = sample[CHEMISTRY_COLS + [TARGET_CONT]].corr(method="pearson")[TARGET_CONT].drop(TARGET_CONT)
    corrs_spearman = sample[CHEMISTRY_COLS + [TARGET_CONT]].corr(method="spearman")[TARGET_CONT].drop(TARGET_CONT)
    results["feature_target_correlations"] = {
        "pearson": corrs_pearson.round(4).to_dict(),
        "spearman": corrs_spearman.round(4).to_dict(),
    }

    # Simple exceedance-based features: count how many variables exceed
    # typical guideline thresholds (rough Canadian guidelines)
    guidelines = {
        "Ammonia (mg/l)": (None, 1.0),
        "Biochemical Oxygen Demand (mg/l)": (None, 4.0),
        "Dissolved Oxygen (mg/l)": (6.5, None),  # minimum
        "Orthophosphate (mg/l)": (None, 0.1),
        "pH (ph units)": (6.5, 8.5),
        "Temperature (cel)": (None, 25.0),
        "Nitrogen (mg/l)": (None, 2.9),
        "Nitrate (mg/l)": (None, 13.0),
    }

    exceedance_count = np.zeros(len(sample))
    for col, (lo, hi) in guidelines.items():
        vals = sample[col].values
        if lo is not None and hi is not None:
            exceedance_count += ((vals < lo) | (vals > hi)).astype(int)
        elif lo is not None:
            exceedance_count += (vals < lo).astype(int)
        elif hi is not None:
            exceedance_count += (vals > hi).astype(int)

    from sklearn.metrics import r2_score as r2s
    # Fit LR on exceedance count alone
    lr_exc = LinearRegression()
    lr_exc.fit(exceedance_count.reshape(-1, 1), y)
    y_pred_exc = lr_exc.predict(exceedance_count.reshape(-1, 1))
    r2_exc = r2s(y, y_pred_exc)
    mae_exc = mean_absolute_error(y, y_pred_exc)

    results["exceedance_features_only"] = {
        "r2": round(r2_exc, 4),
        "mae": round(mae_exc, 4),
        "guidelines_used": {k: str(v) for k, v in guidelines.items()},
        "note": "Simple count of exceedances using approximate Canadian guidelines",
    }
    results["r2_exceedance_alone"] = round(r2_exc, 4)

    # Exceedance count correlations
    exc_s = pd.Series(exceedance_count)
    y_s = pd.Series(y)
    p_corr = float(exc_s.corr(y_s, method="pearson"))
    s_corr = float(exc_s.corr(y_s, method="spearman"))
    results["correlation_exceedance_target"] = {
        "pearson": round(p_corr, 4),
        "spearman": round(s_corr, 4),
    }

    # Mean CCME_Values by exceedance count table
    exc_df = pd.DataFrame({"Exceedance Count": exceedance_count.astype(int), "CCME_Values": y})
    exc_table = (
        exc_df.groupby("Exceedance Count")["CCME_Values"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
        .rename(columns={"count": "N", "mean": "Mean", "std": "Std", "median": "Median"})
    )
    results["exceedance_stats"] = exc_table.round(2)

    results["leakage_statement"] = (
        "CCME WQI is computed from guideline exceedances (F1 scope, F2 frequency, "
        "F3 amplitude) of the SAME chemistry variables used as predictors. "
        "This means the ML task is partly 're-learning a deterministic formula'. "
        "A simple linear regression on raw chemistry features achieves R²={r2:.3f}. "
        "Even a single exceedance-count feature achieves R²={r2_exc:.3f}. "
        "This inherent information overlap must be disclosed. The task still has "
        "value: (1) the exact guidelines/thresholds may vary by jurisdiction, "
        "(2) the CCME formula is non-linear, (3) we can study generalization "
        "across Countries and time."
    ).format(r2=r2, r2_exc=r2_exc)

    return results


def distribution_shift_analysis(df):
    """Check target distribution shift by Country, Waterbody Type, and era."""
    results = {}

    # By Country
    country_stats = {}
    for country in df["Country"].cat.categories:
        subset = df[df["Country"] == country][TARGET_CONT]
        if len(subset) == 0:
            continue
        country_stats[country] = {
            "n": int(len(subset)),
            "mean": round(float(subset.mean()), 2),
            "median": round(float(subset.median()), 2),
            "std": round(float(subset.std()), 2),
            "pct_at_100": round(100 * (subset == 100.0).sum() / len(subset), 2),
        }
        # Class distribution
        cls_dist = df[df["Country"] == country][TARGET_CLASS].value_counts(normalize=True)
        country_stats[country]["class_distribution"] = {k: round(v * 100, 2) for k, v in cls_dist.items()}
    results["by_country"] = country_stats
    results["country_target"] = (
        pd.DataFrame.from_dict(country_stats, orient="index")[["n", "mean", "median", "std", "pct_at_100"]]
        .reset_index()
        .rename(columns={"index": "Country", "n": "N", "mean": "Mean", "median": "Median", "std": "Std", "pct_at_100": "% at 100"})
    )

    # By Waterbody Type
    wb_stats = {}
    for wb in df["Waterbody Type"].cat.categories:
        subset = df[df["Waterbody Type"] == wb][TARGET_CONT]
        if len(subset) == 0:
            continue
        wb_stats[wb] = {
            "n": int(len(subset)),
            "mean": round(float(subset.mean()), 2),
            "median": round(float(subset.median()), 2),
            "std": round(float(subset.std()), 2),
        }
    results["by_waterbody_type"] = wb_stats
    results["waterbody_target"] = (
        pd.DataFrame.from_dict(wb_stats, orient="index")[["n", "mean", "median", "std"]]
        .reset_index()
        .rename(columns={"index": "Waterbody Type", "n": "N", "mean": "Mean", "median": "Median", "std": "Std"})
    )

    # By era
    df_year = df["Date"].dt.year
    era_bins = [(1940, 1970), (1970, 1990), (1990, 2000), (2000, 2010),
                (2010, 2015), (2015, 2020), (2020, 2024)]
    era_stats = {}
    for start, end in era_bins:
        mask = (df_year >= start) & (df_year < end)
        subset = df.loc[mask, TARGET_CONT]
        if len(subset) == 0:
            continue
        era_stats[f"{start}-{end}"] = {
            "n": int(len(subset)),
            "mean": round(float(subset.mean()), 2),
            "median": round(float(subset.median()), 2),
            "std": round(float(subset.std()), 2),
            "pct_at_100": round(100 * (subset == 100.0).sum() / len(subset), 2),
        }
    results["by_era"] = era_stats

    return results


def generate_report(all_results):
    """Generate the markdown report from collected results."""
    r = all_results
    lines = []
    lines.append("# Stage 1 — Dataset Analysis Report\n")
    lines.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("**Dataset:** `data/raw/Combined_dataset.csv`\n")

    # ------ Basic Info ------
    lines.append("## 1. Basic Information\n")
    bi = r["basic_info"]
    lines.append(f"- **Shape:** {bi['shape'][0]:,} rows × {bi['shape'][1]} columns")
    lines.append(f"- **Memory:** {bi['memory_mb']} MB")
    lines.append(f"- **Total missing values:** {bi['total_missing']}")
    lines.append(f"- **Exact duplicate rows:** {bi['exact_duplicates']:,}")
    lines.append("")

    lines.append("### Dtypes\n")
    lines.append("| Column | Dtype |")
    lines.append("|--------|-------|")
    for col, dt in bi["dtypes"].items():
        lines.append(f"| {col} | {dt} |")
    lines.append("")

    lines.append("### Unique Counts\n")
    lines.append("| Column | Unique Values |")
    lines.append("|--------|---------------|")
    for col, cnt in bi["unique_counts"].items():
        lines.append(f"| {col} | {cnt:,} |")
    lines.append("")

    # ------ Percentiles ------
    lines.append("## 2. Numeric Percentile Summary\n")
    pct_df = r["percentiles"]
    lines.append(pct_df.to_markdown())
    lines.append("")

    # ------ Group Counts ------
    lines.append("## 3. Row Counts by Groups\n")
    gc = r["group_counts"]

    lines.append("### By Country\n")
    lines.append("| Country | Rows | % |")
    lines.append("|---------|------|---|")
    total = bi["shape"][0]
    for country, cnt in gc["by_country"].items():
        lines.append(f"| {country} | {cnt:,} | {100*cnt/total:.1f}% |")
    lines.append("")

    lines.append("### By Waterbody Type\n")
    lines.append("| Waterbody Type | Rows | % |")
    lines.append("|----------------|------|---|")
    for wb, cnt in gc["by_waterbody"].items():
        lines.append(f"| {wb} | {cnt:,} | {100*cnt/total:.1f}% |")
    lines.append("")

    lines.append(f"### Year Range: {gc['year_range'][0]}–{gc['year_range'][1]}\n")
    lines.append("Top 10 years by row count:\n")
    lines.append("| Year | Rows |")
    lines.append("|------|------|")
    sorted_years = sorted(gc["by_year"].items(), key=lambda x: x[1], reverse=True)[:10]
    for yr, cnt in sorted_years:
        lines.append(f"| {yr} | {cnt:,} |")
    lines.append("")

    lines.append("### Area (Site) Statistics\n")
    a = gc["area_stats"]
    lines.append(f"- **Distinct areas:** {a['n_distinct']:,}")
    lines.append(f"- **Max rows in one area:** {a['max_rows']:,} (`{a['max_area']}`)")
    lines.append(f"- **Median rows per area:** {a['median_rows']}")
    lines.append(f"- **Mean rows per area:** {a['mean_rows']}")
    lines.append(f"- **Areas with <10 rows:** {a['areas_lt_10_rows']:,}")
    lines.append(f"- **Areas with <5 rows:** {a['areas_lt_5_rows']:,}")
    lines.append(f"- **Areas with exactly 1 row:** {a['areas_eq_1_row']:,}")
    lines.append("")

    # ------ Plausibility ------
    lines.append("## 4. Plausibility Check\n")
    lines.append("| Variable | Range | Actual Min | Actual Max | Below Min | Above Max | Exactly 0 | Negative |")
    lines.append("|----------|-------|-----------|-----------|-----------|-----------|-----------|----------|")
    for col, vals in r["plausibility"].items():
        lines.append(
            f"| {col} | {vals['plausible_range']} | {vals['actual_min']:.4f} | "
            f"{vals['actual_max']:.4f} | {vals['below_min']:,} | {vals['above_max']:,} | "
            f"{vals['exactly_zero']:,} | {vals['negative']:,} |"
        )
    lines.append("")

    # ------ CCME Classes ------
    lines.append("## 5. CCME_WQI Class Analysis (Reverse-Engineered)\n")
    cc = r["ccme_classes"]
    lines.append(f"- **Classes in data:** {cc['classes_in_data']}")
    lines.append(f"- **Rows with CCME_Values = 100.0:** {cc['values_at_100']:,} ({cc['pct_at_100']}%)")
    lines.append("")

    lines.append("| Class | Count | % of Total | Min Value | Max Value | Mean |")
    lines.append("|-------|-------|-----------|-----------|-----------|------|")
    for cls in cc["classes_in_data"]:
        d = cc[cls]
        lines.append(f"| {cls} | {d['count']:,} | {d['pct_of_total']}% | {d['min']:.2f} | {d['max']:.2f} | {d['mean']:.2f} |")
    lines.append("")

    lines.append("### Derived Boundaries\n")
    lines.append("| Class | Min | Max |")
    lines.append("|-------|-----|-----|")
    for cls, (lo, hi) in cc["boundaries"].items():
        lines.append(f"| {cls} | {lo:.2f} | {hi:.2f} |")
    lines.append("")

    # ------ Leakage ------
    lines.append("## 6. Leakage Analysis ⚠️\n")
    la = r["leakage"]
    lr_raw = la["linear_regression_raw_features"]
    exc = la["exceedance_features_only"]

    lines.append(f"### Linear Regression on Raw Chemistry Features\n")
    lines.append(f"- **R²:** {lr_raw['r2']}")
    lines.append(f"- **MAE:** {lr_raw['mae']}")
    lines.append(f"- **N samples:** {lr_raw['n_samples']:,}")
    lines.append("")

    lines.append(f"### Exceedance-Count Feature Only\n")
    lines.append(f"- **R²:** {exc['r2']}")
    lines.append(f"- **MAE:** {exc['mae']}")
    lines.append("")

    lines.append("### Feature–Target Correlations\n")
    lines.append("| Feature | Pearson | Spearman |")
    lines.append("|---------|---------|----------|")
    for feat in CHEMISTRY_COLS:
        p = la["feature_target_correlations"]["pearson"].get(feat, "N/A")
        s = la["feature_target_correlations"]["spearman"].get(feat, "N/A")
        lines.append(f"| {feat} | {p} | {s} |")
    lines.append("")

    lines.append("> **Leakage Statement:**")
    lines.append(f"> {la['leakage_statement']}")
    lines.append("")

    # ------ Distribution Shift ------
    lines.append("## 7. Distribution Shift Analysis\n")

    lines.append("### By Country\n")
    lines.append("| Country | N | Mean | Median | Std | %=100 |")
    lines.append("|---------|---|------|--------|-----|-------|")
    for country, stats in r["distribution_shift"]["by_country"].items():
        lines.append(
            f"| {country} | {stats['n']:,} | {stats['mean']} | {stats['median']} | "
            f"{stats['std']} | {stats['pct_at_100']}% |"
        )
    lines.append("")

    lines.append("### Class Distribution by Country\n")
    for country, stats in r["distribution_shift"]["by_country"].items():
        cd = stats.get("class_distribution", {})
        lines.append(f"**{country}:** " + ", ".join(f"{k}: {v}%" for k, v in cd.items()))
    lines.append("")

    lines.append("### By Waterbody Type\n")
    lines.append("| Waterbody Type | N | Mean | Median | Std |")
    lines.append("|----------------|---|------|--------|-----|")
    for wb, stats in r["distribution_shift"]["by_waterbody_type"].items():
        lines.append(f"| {wb} | {stats['n']:,} | {stats['mean']} | {stats['median']} | {stats['std']} |")
    lines.append("")

    lines.append("### By Era\n")
    lines.append("| Era | N | Mean | Median | Std | %=100 |")
    lines.append("|-----|---|------|--------|-----|-------|")
    for era, stats in r["distribution_shift"]["by_era"].items():
        lines.append(
            f"| {era} | {stats['n']:,} | {stats['mean']} | {stats['median']} | "
            f"{stats['std']} | {stats['pct_at_100']}% |"
        )
    lines.append("")

    # ------ Data Dictionary ------
    lines.append("## 8. Data Dictionary\n")
    lines.append("| # | Column | Type | Role | Description |")
    lines.append("|---|--------|------|------|-------------|")
    data_dict = [
        ("1", "Country", "Categorical", "Metadata (split)", "Country of origin. NOT a model feature (leakage risk)."),
        ("2", "Area", "String", "Metadata (split)", "Site/station identifier. Used for group-aware splitting."),
        ("3", "Waterbody Type", "Categorical", "Metadata/Feature", "Type of water body (River, Lake, etc.). Ablation candidate."),
        ("4", "Date", "Datetime", "Metadata", "Measurement date (dd-mm-yyyy). Month extracted for seasonality."),
        ("5", "Ammonia (mg/l)", "Float", "Feature", "Ammonia concentration. Positively skewed."),
        ("6", "Biochemical Oxygen Demand (mg/l)", "Float", "Feature", "BOD. Organic pollution indicator."),
        ("7", "Dissolved Oxygen (mg/l)", "Float", "Feature", "DO. Essential for aquatic life. Inversely related to pollution."),
        ("8", "Orthophosphate (mg/l)", "Float", "Feature", "Phosphate concentration. Eutrophication indicator."),
        ("9", "pH (ph units)", "Float", "Feature", "Acidity/alkalinity. Expected 6–9 for ambient water."),
        ("10", "Temperature (cel)", "Float", "Feature", "Water temperature. Affects DO saturation."),
        ("11", "Nitrogen (mg/l)", "Float", "Feature", "Total nitrogen. Likely collinear with Nitrate."),
        ("12", "Nitrate (mg/l)", "Float", "Feature", "Nitrate concentration. Component of total nitrogen."),
        ("13", "CCME_Values", "Float", "**TARGET**", "Continuous WQI score (31.3–100). Censored at 100."),
        ("14", "CCME_WQI", "Categorical", "Derived", "5-class label derived from CCME_Values. NOT a separate target."),
    ]
    for row in data_dict:
        lines.append(f"| {' | '.join(row)} |")
    lines.append("")

    lines.append("> **Note on missing features:** The dataset does NOT contain turbidity, ")
    lines.append("> conductivity, or dissolved solids, which may be listed in the project brief. ")
    lines.append("> We use the 8 available chemistry variables and document this deviation.\n")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("STAGE 1 — DATASET ANALYSIS")
    print("=" * 60)

    # Load
    df = load_data()
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    # 1. Basic info
    print("Computing basic info...")
    bi = basic_info(df)
    print(f"  Duplicates: {bi['exact_duplicates']:,}")
    print(f"  Missing: {bi['total_missing']}")

    # 2. Percentile summary
    print("Computing percentile summary...")
    pct = percentile_summary(df)

    # 3. Group counts
    print("Computing group counts...")
    gc = group_counts(df)

    # 4. Plausibility check
    print("Running plausibility check...")
    plaus = plausibility_check(df)

    # 5. CCME class analysis
    print("Reverse-engineering CCME classes...")
    ccme = reverse_engineer_ccme_classes(df)

    # 6. Leakage analysis
    print("Running leakage analysis...")
    leak = leakage_analysis(df)

    # 7. Distribution shift
    print("Analyzing distribution shift...")
    dist = distribution_shift_analysis(df)

    # Collect all results
    all_results = {
        "basic_info": bi,
        "percentiles": pct,
        "group_counts": gc,
        "plausibility": plaus,
        "ccme_classes": ccme,
        "leakage": leak,
        "distribution_shift": dist,
    }

    # Save raw results as JSON (for reproducibility)
    json_path = RESULTS_DIR / "01_dataset_analysis_results.json"
    # Convert non-serializable items
    def default_serializer(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        if isinstance(obj, tuple):
            return list(obj)
        return str(obj)

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=default_serializer)
    print(f"\nResults saved: {json_path}")

    # Generate report
    report = generate_report(all_results)
    report_path = RESULTS_DIR / "01_dataset_analysis.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved: {report_path}")

    print("\n✅ Stage 1 complete.")
    return all_results


if __name__ == "__main__":
    main()
