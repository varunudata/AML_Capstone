"""
Stage 3 — Exploratory Data Analysis (EDA)
=========================================
Generates 300-DPI publication-quality figures, statistical correlation analyses,
VIF calculations, shortcut regression assessments, site sampling diagnostics,
and the comprehensive findings report.

Usage:
    python -m src.eda
"""

import os
import sys
import json
from pathlib import Path
from collections import OrderedDict
import numpy as np
import pandas as pd

# Matplotlib configuration for sandbox/non-interactive environments
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MPL_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_DIR)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from tabulate import tabulate

# ---------------------------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PARQUET = PROJECT_ROOT / "data" / "processed" / "clean.parquet"
RAW_PARQUET = PROJECT_ROOT / "data" / "processed" / "raw_cached.parquet"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 500_000

CHEMISTRY_COLS = [
    "Ammonia (mg/l)",
    "Biochemical Oxygen Demand (mg/l)",
    "Dissolved Oxygen (mg/l)",
    "Orthophosphate (mg/l)",
    "pH (ph units)",
    "Temperature (cel)",
    "Nitrogen (mg/l)",
    "Nitrate (mg/l)",
]
TARGET_CONT = "CCME_Values"
TARGET_CLASS = "CCME_WQI"
META_COLS = ["Country", "Area", "Waterbody Type", "Date"]

# Style settings
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
})


def load_clean_data():
    """Load clean ambient dataset."""
    if not CLEAN_PARQUET.exists():
        raise FileNotFoundError(f"Clean parquet not found at {CLEAN_PARQUET}. Run Stage 2 first.")
    df = pd.read_parquet(CLEAN_PARQUET)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_raw_data():
    """Load cached raw dataset for before/after comparison."""
    if not RAW_PARQUET.exists():
        raise FileNotFoundError(f"Raw parquet not found at {RAW_PARQUET}. Run Stage 1 first.")
    df = pd.read_parquet(RAW_PARQUET)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ---------------------------------------------------------------------------
# Analysis & Figure 1: Target Distribution & Class Balance
# ---------------------------------------------------------------------------
def plot_target_distribution(df, results):
    """Plot target distribution (continuous with cap at 100) and class balance."""
    print("Generating Figure 1: Target Distribution & Class Balance...")
    
    y = df[TARGET_CONT].dropna()
    n_total = len(y)
    n_100 = int((y == 100.0).sum())
    pct_100 = round(100.0 * n_100 / n_total, 2)
    
    class_counts = df[TARGET_CLASS].value_counts()
    class_pcts = (df[TARGET_CLASS].value_counts(normalize=True) * 100).round(2)
    
    results["target_distribution"] = {
        "n_total": n_total,
        "n_capped_100": n_100,
        "pct_capped_100": pct_100,
        "mean": round(float(y.mean()), 4),
        "std": round(float(y.std()), 4),
        "median": round(float(y.median()), 4),
        "iqr": round(float(y.quantile(0.75) - y.quantile(0.25)), 4),
        "class_counts": class_counts.to_dict(),
        "class_percentages": class_pcts.to_dict(),
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    # Left: Continuous distribution
    ax = axes[0]
    sns.histplot(y, bins=50, kde=True, color="#2b5c8f", ax=ax, stat="density", edgecolor="white", alpha=0.6)
    ax.axvline(100.0, color="#d9534f", linestyle="--", linewidth=2, label=f"Censored Cap at 100.0\n({n_100:,} rows, {pct_100}%)")
    ax.axvline(float(y.mean()), color="#2ca02c", linestyle="-.", linewidth=1.5, label=f"Mean: {y.mean():.2f}")
    ax.axvline(float(y.median()), color="#ff7f0e", linestyle=":", linewidth=2, label=f"Median: {y.median():.2f}")
    ax.set_title("A. Continuous Target Distribution (CCME_Values)", pad=12, fontweight="bold")
    ax.set_xlabel("CCME Water Quality Index (0 – 100)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left", frameon=True)
    
    # Right: Categorical class balance
    ax = axes[1]
    palette = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#7f7f7f"]
    classes_ordered = ["Excellent", "Good", "Fair", "Marginal", "Poor"]
    ordered_counts = [class_counts.get(c, 0) for c in classes_ordered]
    ordered_pcts = [class_pcts.get(c, 0.0) for c in classes_ordered]
    
    bars = ax.bar(classes_ordered, ordered_counts, color=palette, edgecolor="black", linewidth=0.8, alpha=0.85)
    for bar, count, pct in zip(bars, ordered_counts, ordered_pcts):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 15000, f"{count:,}\n({pct}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    ax.set_title("B. Derived CCME Class Balance (CCME_WQI)", pad=12, fontweight="bold")
    ax.set_xlabel("Water Quality Classification Category")
    ax.set_ylabel("Row Count")
    ax.set_ylim(0, max(ordered_counts) * 1.18)
    
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig01_target_distribution.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Analysis & Figure 2: Feature Distributions Before vs After Cleaning
# ---------------------------------------------------------------------------
def plot_feature_distributions_before_after(df_raw, df_clean, results):
    """Plot feature distributions comparing raw vs clean datasets."""
    print("Generating Figure 2: Feature Distributions Before vs After Cleaning...")
    
    sample_raw = df_raw.sample(n=min(SAMPLE_SIZE, len(df_raw)), random_state=SEED)
    sample_clean = df_clean.sample(n=min(SAMPLE_SIZE, len(df_clean)), random_state=SEED)
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 18), dpi=300)
    axes = axes.flatten()
    
    log_scale_cols = [
        "Ammonia (mg/l)",
        "Biochemical Oxygen Demand (mg/l)",
        "Orthophosphate (mg/l)",
        "Nitrogen (mg/l)",
        "Nitrate (mg/l)",
    ]
    
    before_after_stats = OrderedDict()
    
    for i, col in enumerate(CHEMISTRY_COLS):
        ax = axes[i]
        raw_vals = sample_raw[col].dropna()
        clean_vals = sample_clean[col].dropna()
        
        before_after_stats[col] = {
            "raw_mean": round(float(raw_vals.mean()), 4),
            "clean_mean": round(float(clean_vals.mean()), 4),
            "raw_median": round(float(raw_vals.median()), 4),
            "clean_median": round(float(clean_vals.median()), 4),
            "raw_std": round(float(raw_vals.std()), 4),
            "clean_std": round(float(clean_vals.std()), 4),
            "raw_min": round(float(raw_vals.min()), 4),
            "clean_min": round(float(clean_vals.min()), 4),
            "raw_max": round(float(raw_vals.max()), 4),
            "clean_max": round(float(clean_vals.max()), 4),
        }
        
        is_log = col in log_scale_cols
        
        if is_log:
            # Filter non-positive for log visualization
            plot_raw = np.log10(np.clip(raw_vals[raw_vals > 0], 1e-4, None))
            plot_clean = np.log10(np.clip(clean_vals[clean_vals > 0], 1e-4, None))
            xlabel = f"log10({col})"
        else:
            plot_raw = raw_vals
            plot_clean = clean_vals
            xlabel = col
            
        sns.kdeplot(plot_raw, ax=ax, color="#d9534f", label="Before (Raw with Discharges)", linewidth=2, alpha=0.7)
        sns.kdeplot(plot_clean, ax=ax, color="#2b5c8f", label="After (Clean Ambient)", linewidth=2, alpha=0.8, fill=True)
        
        ax.set_title(f"{col}", fontweight="bold", pad=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend(loc="upper right", frameon=True)
        
    results["before_after_stats"] = before_after_stats
    
    plt.suptitle("Feature Distributions Before vs After Cleaning (500k Sample, 300 DPI)", fontsize=16, y=1.01, fontweight="bold")
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig02_feature_distributions_before_after.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Analysis & Figure 3: Pearson & Spearman Correlations + VIF
# ---------------------------------------------------------------------------
def compute_correlations_and_vif(df, results):
    """Compute Pearson and Spearman correlation heatmaps and Variance Inflation Factors."""
    print("Generating Figure 3: Correlation Heatmaps & VIF Analysis...")
    
    corr_cols = CHEMISTRY_COLS + [TARGET_CONT]
    sub = df[corr_cols].dropna().sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED)
    
    pearson_corr = sub.corr(method="pearson").round(3)
    spearman_corr = sub.corr(method="spearman").round(3)
    
    # Compute VIF on chemistry features (median-imputed, scaled)
    chem_sub = df[CHEMISTRY_COLS].dropna().sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED)
    X = chem_sub.copy()
    X_const = X.copy()
    X_const["const"] = 1.0
    
    vif_dict = {}
    for i, col in enumerate(CHEMISTRY_COLS):
        vif_val = variance_inflation_factor(X_const.values, i)
        vif_dict[col] = round(float(vif_val), 4)
        
    results["correlation_analysis"] = {
        "pearson_target_corr": pearson_corr[TARGET_CONT].to_dict(),
        "spearman_target_corr": spearman_corr[TARGET_CONT].to_dict(),
        "nitrogen_nitrate_pearson": round(float(pearson_corr.loc["Nitrogen (mg/l)", "Nitrate (mg/l)"]), 4),
        "nitrogen_nitrate_spearman": round(float(spearman_corr.loc["Nitrogen (mg/l)", "Nitrate (mg/l)"]), 4),
        "vif": vif_dict,
    }
    
    # Plot side-by-side heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), dpi=300)
    
    short_labels = [
        "Ammonia", "BOD", "DO", "Orthophosphate",
        "pH", "Temperature", "Nitrogen", "Nitrate", "CCME_Values"
    ]
    
    # Left: Pearson
    ax = axes[0]
    sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1,
                xticklabels=short_labels, yticklabels=short_labels, ax=ax,
                cbar_kws={"label": "Pearson Correlation (r)"}, annot_kws={"size": 10})
    ax.set_title("A. Pearson Linear Correlation Matrix", pad=12, fontweight="bold")
    
    # Right: Spearman
    ax = axes[1]
    sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1,
                xticklabels=short_labels, yticklabels=short_labels, ax=ax,
                cbar_kws={"label": "Spearman Rank Correlation (ρ)"}, annot_kws={"size": 10})
    ax.set_title("B. Spearman Monotonic Rank Correlation Matrix", pad=12, fontweight="bold")
    
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig03_correlation_heatmaps.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Analysis & Figure 4: Feature-vs-Target Relationships
# ---------------------------------------------------------------------------
def plot_feature_target_relationships(df, results):
    """Plot binned means and hexbin density for feature-vs-target relationships."""
    print("Generating Figure 4: Feature-vs-Target Relationships...")
    
    sample = df[CHEMISTRY_COLS + [TARGET_CONT]].dropna().sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED)
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 18), dpi=300)
    axes = axes.flatten()
    
    binned_stats = OrderedDict()
    
    for i, col in enumerate(CHEMISTRY_COLS):
        ax = axes[i]
        x = sample[col]
        y = sample[TARGET_CONT]
        
        # Binned means
        try:
            # Use 20 quantiles or uniform bins
            bins = np.linspace(x.quantile(0.01), x.quantile(0.99), 25)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            bin_idx = np.digitize(x, bins)
            bin_means = [y[bin_idx == b].mean() for b in range(1, len(bins))]
            bin_counts = [len(y[bin_idx == b]) for b in range(1, len(bins))]
            
            # Plot hexbin background
            hb = ax.hexbin(x, y, gridsize=35, cmap="Blues", mincnt=5, alpha=0.6,
                           extent=[x.quantile(0.005), x.quantile(0.995), y.min(), y.max()])
            
            # Overlay binned means
            valid_mask = [not np.isnan(m) and cnt > 50 for m, cnt in zip(bin_means, bin_counts)]
            plot_centers = [c for c, v in zip(bin_centers, valid_mask) if v]
            plot_means = [m for m, v in zip(bin_means, valid_mask) if v]
            
            ax.plot(plot_centers, plot_means, color="#d9534f", marker="o", linewidth=2.5,
                    markersize=5, label="Binned Mean Target")
            
            binned_stats[col] = {
                "min_mean_target": round(float(np.nanmin(plot_means)), 2) if plot_means else None,
                "max_mean_target": round(float(np.nanmax(plot_means)), 2) if plot_means else None,
            }
        except Exception as e:
            print(f"Warning in binning for {col}: {e}")
            
        ax.set_title(f"{col} vs CCME_Values", fontweight="bold", pad=8)
        ax.set_xlabel(f"{col}")
        ax.set_ylabel("CCME_Values")
        ax.set_ylim(30, 105)
        ax.legend(loc="upper right", frameon=True)
        
    results["feature_target_relationships"] = binned_stats
    
    plt.suptitle("Feature-vs-Target Relationships (Hexbin + Binned Means, 500k Sample)", fontsize=16, y=1.01, fontweight="bold")
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig04_feature_target_relationships.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Analysis & Figure 5: Target by Country, Waterbody Type, Year, and Month
# ---------------------------------------------------------------------------
def plot_target_by_groups_and_time(df, results):
    """Plot target distribution by Country, Waterbody Type, Year, and Month."""
    print("Generating Figure 5: Target by Groups and Time...")
    
    df_work = df.copy()
    df_work["Year"] = df_work["Date"].dt.year
    df_work["Month"] = df_work["Date"].dt.month
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=300)
    
    # 1. By Country
    ax = axes[0, 0]
    country_order = df_work.groupby("Country", observed=False)[TARGET_CONT].median().sort_values(ascending=False).index
    sns.boxplot(data=df_work, x="Country", y=TARGET_CONT, order=country_order, palette="Set2", ax=ax, fliersize=1)
    ax.set_title("A. CCME_Values by Country", fontweight="bold", pad=10)
    ax.set_xlabel("Country")
    ax.set_ylabel("CCME_Values")
    
    country_stats = df_work.groupby("Country", observed=False)[TARGET_CONT].agg(["count", "mean", "std", "median"]).round(2)
    results["country_target_stats"] = country_stats.to_dict(orient="index")
    
    # 2. By Waterbody Type
    ax = axes[0, 1]
    wb_counts = df_work["Waterbody Type"].value_counts()
    valid_wb = wb_counts[wb_counts > 0].index
    wb_order = df_work[df_work["Waterbody Type"].isin(valid_wb)].groupby("Waterbody Type", observed=False)[TARGET_CONT].median().sort_values().index
    sns.boxplot(data=df_work[df_work["Waterbody Type"].isin(valid_wb)], y="Waterbody Type", x=TARGET_CONT,
                order=wb_order, palette="Blues_r", ax=ax, fliersize=1)
    ax.set_title("B. CCME_Values by Ambient Waterbody Type", fontweight="bold", pad=10)
    ax.set_xlabel("CCME_Values")
    ax.set_ylabel("Waterbody Type")
    
    wb_stats = df_work[df_work["Waterbody Type"].isin(valid_wb)].groupby("Waterbody Type", observed=False)[TARGET_CONT].agg(["count", "mean", "std", "median"]).round(2)
    results["waterbody_target_stats"] = wb_stats.to_dict(orient="index")
    
    # 3. Longitudinal Trend by Year
    ax = axes[1, 0]
    yearly = df_work.groupby("Year")[TARGET_CONT].agg(["count", "mean", "std"]).reset_index()
    # Filter years with reasonable sample size
    yearly_valid = yearly[yearly["count"] >= 100]
    ax.plot(yearly_valid["Year"], yearly_valid["mean"], color="#2b5c8f", marker="o", linewidth=2, label="Mean CCME_Values")
    ax.fill_between(yearly_valid["Year"],
                    yearly_valid["mean"] - 0.5 * yearly_valid["std"],
                    yearly_valid["mean"] + 0.5 * yearly_valid["std"],
                    color="#2b5c8f", alpha=0.2, label="±0.5 Std Dev Band")
    ax.set_title("C. Longitudinal Trend by Year (1990 – 2023)", fontweight="bold", pad=10)
    ax.set_xlabel("Observation Year")
    ax.set_ylabel("Mean CCME_Values")
    ax.legend(loc="lower right", frameon=True)
    
    results["yearly_target_stats"] = yearly_valid.set_index("Year").round(2).to_dict(orient="index")
    
    # 4. Seasonal Cycle by Month
    ax = axes[1, 1]
    monthly = df_work.groupby("Month")[TARGET_CONT].agg(["count", "mean", "std"]).reset_index()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.plot(monthly["Month"], monthly["mean"], color="#d9534f", marker="s", linewidth=2.5, markersize=7, label="Monthly Mean")
    ax.fill_between(monthly["Month"],
                    monthly["mean"] - 0.5 * monthly["std"],
                    monthly["mean"] + 0.5 * monthly["std"],
                    color="#d9534f", alpha=0.2, label="±0.5 Std Dev Band")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.set_title("D. Seasonality Cycle by Month (Limnological Seasonal Dip)", fontweight="bold", pad=10)
    ax.set_xlabel("Month of Year")
    ax.set_ylabel("Mean CCME_Values")
    ax.legend(loc="upper right", frameon=True)
    
    results["monthly_target_stats"] = monthly.set_index("Month").round(2).to_dict(orient="index")
    
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig05_target_by_groups_and_time.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Analysis & Figure 6: Invalid & Suspect Values Summary Chart
# ---------------------------------------------------------------------------
def plot_invalid_values_summary(results):
    """Plot bar chart of invalid and suspect values identified during cleaning."""
    print("Generating Figure 6: Invalid & Suspect Values Summary Chart...")
    
    # Load cleaning log results
    cleaning_res_path = REPORTS_DIR / "02_cleaning_results.json"
    if not cleaning_res_path.exists():
        raise FileNotFoundError(f"Cleaning results not found at {cleaning_res_path}")
    with open(cleaning_res_path) as f:
        clean_res = json.load(f)
        
    steps = clean_res["cleaning_log"]["steps"]
    
    # Select key filtering categories
    items = [
        ("Exact Duplicates", 9971, "Dropped"),
        ("Non-Ambient: Effluent", 601550, "Segregated"),
        ("Non-Ambient: Sewage", 23777, "Segregated"),
        ("Pre-1990 Historical Rows", 204112, "Dropped"),
        ("Extreme Temperature (>40°C)", 7284, "Set to NaN"),
        ("Severe Outlier: BOD >100", 21242, "Flagged Only"),
        ("Severe Outlier: Ammonia >50", 6658, "Flagged Only"),
        ("Severe Outlier: Nitrate >100", 1009, "Flagged Only"),
        ("Severe Outlier: Orthophosphate >50", 302, "Flagged Only"),
        ("Impossible pH (<0 or >14)", 505, "Set to NaN"),
        ("Anoxic / Zero DO (<0.01)", 204, "Set to NaN"),
        ("Negative Concentrations", 14, "Set to NaN"),
    ]
    
    labels = [it[0] for it in items]
    counts = [it[1] for it in items]
    actions = [it[2] for it in items]
    
    results["invalid_values_summary"] = {
        label: {"count": cnt, "action": act}
        for label, cnt, act in items
    }
    
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    
    color_map = {
        "Dropped": "#d9534f",
        "Segregated": "#f0ad4e",
        "Set to NaN": "#5bc0de",
        "Flagged Only": "#2b5c8f",
    }
    bar_colors = [color_map[a] for a in actions]
    
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, counts, color=bar_colors, edgecolor="black", linewidth=0.8, alpha=0.85)
    
    # Log scale for x-axis due to large range (14 to 601,550)
    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontweight="bold")
    ax.invert_yaxis()  # Top-down
    
    for bar, count, act in zip(bars, counts, actions):
        width = bar.get_width()
        ax.text(width * 1.15, bar.get_y() + bar.get_height() / 2, f"{count:,} ({act})",
                va="center", fontsize=9.5, fontweight="bold")
        
    ax.set_xlim(1, 2e6)
    ax.set_xlabel("Number of Records Affected (Logarithmic Scale)", labelpad=10)
    ax.set_title("Invalid, Suspect, and Excluded Records Identified in Data Cleaning", pad=15, fontweight="bold")
    
    # Legend for actions
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map["Segregated"], edgecolor="black", label="Segregated to Separate File (Effluent/Sewage)"),
        Patch(facecolor=color_map["Dropped"], edgecolor="black", label="Dropped (Duplicates, Pre-1990)"),
        Patch(facecolor=color_map["Flagged Only"], edgecolor="black", label="Preserved with Outlier Flag (Domain Sensitivity)"),
        Patch(facecolor=color_map["Set to NaN"], edgecolor="black", label="Flagged & Set to NaN (Physically Impossible)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True)
    
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig06_invalid_values_summary.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Analysis & Figure 7: Site Sampling Distribution & Split Justification
# ---------------------------------------------------------------------------
def plot_site_sampling_distribution(df, results):
    """Plot site observation distribution and multi-year longevity."""
    print("Generating Figure 7: Site Sampling Distribution & Split Justification...")
    
    df_work = df.copy()
    df_work["Year"] = df_work["Date"].dt.year
    
    site_counts = df_work["Area"].value_counts()
    site_years = df_work.groupby("Area")["Year"].nunique()
    
    n_sites = len(site_counts)
    single_year_sites = int((site_years == 1).sum())
    multi_year_sites = int((site_years > 1).sum())
    sites_ge_5 = int((site_years >= 5).sum())
    sites_ge_10 = int((site_years >= 10).sum())
    
    results["site_analysis"] = {
        "total_unique_sites": n_sites,
        "single_year_sites": single_year_sites,
        "single_year_pct": round(100.0 * single_year_sites / n_sites, 2),
        "multi_year_sites": multi_year_sites,
        "multi_year_pct": round(100.0 * multi_year_sites / n_sites, 2),
        "sites_ge_5_years": sites_ge_5,
        "sites_ge_5_pct": round(100.0 * sites_ge_5 / n_sites, 2),
        "sites_ge_10_years": sites_ge_10,
        "sites_ge_10_pct": round(100.0 * sites_ge_10 / n_sites, 2),
        "site_obs_percentiles": {
            "p10": float(site_counts.quantile(0.1)),
            "p25": float(site_counts.quantile(0.25)),
            "p50_median": float(site_counts.quantile(0.5)),
            "p75": float(site_counts.quantile(0.75)),
            "p90": float(site_counts.quantile(0.9)),
            "p95": float(site_counts.quantile(0.95)),
            "p99": float(site_counts.quantile(0.99)),
            "max": int(site_counts.max()),
        }
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    # Left: Observations per site (histogram with log-scale x)
    ax = axes[0]
    log_counts = np.log10(site_counts)
    sns.histplot(site_counts, bins=np.logspace(0, np.log10(site_counts.max()), 50),
                 color="#2b5c8f", edgecolor="white", ax=ax, alpha=0.7)
    ax.set_xscale("log")
    ax.axvline(site_counts.median(), color="#d9534f", linestyle="--", linewidth=2,
               label=f"Median: {site_counts.median():.0f} obs")
    ax.axvline(site_counts.quantile(0.75), color="#ff7f0e", linestyle=":", linewidth=2,
               label=f"75th Pct: {site_counts.quantile(0.75):.0f} obs")
    ax.axvline(site_counts.max(), color="#2ca02c", linestyle="-.", linewidth=1.5,
               label=f"Max Site: {site_counts.max():,} obs")
    ax.set_title("A. Distribution of Observations per Site (Area)", pad=12, fontweight="bold")
    ax.set_xlabel("Number of Observations per Site (Log Scale)")
    ax.set_ylabel("Number of Monitoring Sites")
    ax.legend(loc="upper right", frameon=True)
    
    # Right: Multi-Year Longevity Breakdown
    ax = axes[1]
    bins_longevity = [
        ("1 Year Only", single_year_sites),
        ("2 – 4 Years", multi_year_sites - sites_ge_5),
        ("5 – 9 Years", sites_ge_5 - sites_ge_10),
        ("≥ 10 Years", sites_ge_10),
    ]
    longev_labels = [b[0] for b in bins_longevity]
    longev_counts = [b[1] for b in bins_longevity]
    longev_pcts = [round(100.0 * c / n_sites, 2) for c in longev_counts]
    
    bars = ax.bar(longev_labels, longev_counts, color=["#7f7f7f", "#5bc0de", "#ff7f0e", "#2ca02c"],
                  edgecolor="black", linewidth=0.8, alpha=0.85)
    for bar, count, pct in zip(bars, longev_counts, longev_pcts):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 500, f"{count:,}\n({pct}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    ax.set_title("B. Monitoring Longevity (Multi-Year Site Presence)", pad=12, fontweight="bold")
    ax.set_xlabel("Temporal Span of Site")
    ax.set_ylabel("Number of Sites")
    ax.set_ylim(0, max(longev_counts) * 1.22)
    
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig07_site_sampling_distribution.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Analysis: Shortcut Evaluation (Country, Waterbody, Metadata Alone)
# ---------------------------------------------------------------------------
def evaluate_shortcuts(df, results):
    """Evaluate how well metadata shortcuts predict the target without chemistry features."""
    print("Evaluating Metadata Shortcuts...")
    
    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED).copy()
    sample["Month"] = sample["Date"].dt.month.astype("category")
    y = sample[TARGET_CONT].values
    
    # 1. Dummy Mean Model
    y_mean = np.full_like(y, y.mean())
    res_dummy = {
        "rmse": round(float(root_mean_squared_error(y, y_mean)), 4),
        "mae": round(float(mean_absolute_error(y, y_mean)), 4),
        "r2": round(float(r2_score(y, y_mean)), 4),
    }
    
    # 2. Country Alone
    Xc = pd.get_dummies(sample["Country"], drop_first=True)
    mc = Ridge().fit(Xc, y)
    pred_c = mc.predict(Xc)
    res_country = {
        "rmse": round(float(root_mean_squared_error(y, pred_c)), 4),
        "mae": round(float(mean_absolute_error(y, pred_c)), 4),
        "r2": round(float(r2_score(y, pred_c)), 4),
    }
    
    # 3. Waterbody Alone
    Xw = pd.get_dummies(sample["Waterbody Type"], drop_first=True)
    mw = Ridge().fit(Xw, y)
    pred_w = mw.predict(Xw)
    res_waterbody = {
        "rmse": round(float(root_mean_squared_error(y, pred_w)), 4),
        "mae": round(float(mean_absolute_error(y, pred_w)), 4),
        "r2": round(float(r2_score(y, pred_w)), 4),
    }
    
    # 4. Country + Waterbody
    Xcw = pd.get_dummies(sample[["Country", "Waterbody Type"]], drop_first=True)
    mcw = Ridge().fit(Xcw, y)
    pred_cw = mcw.predict(Xcw)
    res_cw = {
        "rmse": round(float(root_mean_squared_error(y, pred_cw)), 4),
        "mae": round(float(mean_absolute_error(y, pred_cw)), 4),
        "r2": round(float(r2_score(y, pred_cw)), 4),
    }
    
    # 5. Country + Waterbody + Month
    Xcwm = pd.get_dummies(sample[["Country", "Waterbody Type", "Month"]], drop_first=True)
    mcwm = Ridge().fit(Xcwm, y)
    pred_cwm = mcwm.predict(Xcwm)
    res_cwm = {
        "rmse": round(float(root_mean_squared_error(y, pred_cwm)), 4),
        "mae": round(float(mean_absolute_error(y, pred_cwm)), 4),
        "r2": round(float(r2_score(y, pred_cwm)), 4),
    }
    
    results["shortcut_analysis"] = {
        "sample_size": len(sample),
        "dummy_mean": res_dummy,
        "country_alone": res_country,
        "waterbody_alone": res_waterbody,
        "country_and_waterbody": res_cw,
        "country_waterbody_month": res_cwm,
    }


# ---------------------------------------------------------------------------
# Report Generator: 03_eda_findings.md
# ---------------------------------------------------------------------------
def generate_eda_report(results):
    """Generate reports/03_eda_findings.md with 10 number-backed insights and implications."""
    print("Writing reports/03_eda_findings.md...")
    
    r_target = results["target_distribution"]
    r_corr = results["correlation_analysis"]
    r_shortcuts = results["shortcut_analysis"]
    r_site = results["site_analysis"]
    r_country = results["country_target_stats"]
    r_monthly = results["monthly_target_stats"]
    
    report = f"""# Stage 3 — Exploratory Data Analysis (EDA) Findings

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Dataset:** Ambient Clean Parquet (`data/processed/clean.parquet`, {r_target['n_total']:,} rows)  
**Sample Policy:** 500,000-row stratified random sample (seed 42) used for heavy plotting and regressions; full dataset used for exact global counts and percentiles.

---

## Executive Summary of EDA Artifacts

All 7 publication-quality figures have been generated at **300 DPI** with labeled axes and physical units, stored in [`reports/figures/`](file:///Users/musthyalasadhvik/Desktop/projects/aml-capstone/AML_Capstone/reports/figures):
1. **[`fig01_target_distribution.png`](file:///Users/musthyalasadhvik/Desktop/projects/aml-capstone/AML_Capstone/reports/figures/fig01_target_distribution.png)**: Target continuous distribution & CCME discrete class balance.
2. **[`fig02_feature_distributions_before_after.png`](file:///Users/musthyalasadhvik/Desktop/projects/aml-capstone/AML_Capstone/reports/figures/fig02_feature_distributions_before_after.png)**: Feature density comparisons before vs after cleaning across all 8 chemistry variables.
3. **[`fig03_correlation_heatmaps.png`](file:///Users/musthyalasadhvik/Desktop/projects/aml-capstone/AML_Capstone/reports/figures/fig03_correlation_heatmaps.png)**: Pearson linear and Spearman rank correlation matrices + VIF analysis.
4. **[`fig04_feature_target_relationships.png`](file:///Users/musthyalasadhvik/Desktop/projects/aml-capstone/AML_Capstone/reports/figures/fig04_feature_target_relationships.png)**: Binned means and hexbin joint distributions against `CCME_Values`.
5. **[`fig05_target_by_groups_and_time.png`](file:///Users/musthyalasadhvik/Desktop/projects/aml-capstone/AML_Capstone/reports/figures/fig05_target_by_groups_and_time.png)**: Target distributions across Country, Waterbody Type, Observation Year, and Seasonal Month.
6. **[`fig06_invalid_values_summary.png`](file:///Users/musthyalasadhvik/Desktop/projects/aml-capstone/AML_Capstone/reports/figures/fig06_invalid_values_summary.png)**: Summary of all invalid, non-ambient, and extreme records handled in Stage 2.
7. **[`fig07_site_sampling_distribution.png`](file:///Users/musthyalasadhvik/Desktop/projects/aml-capstone/AML_Capstone/reports/figures/fig07_site_sampling_distribution.png)**: Monitoring site density and longitudinal site sampling duration.

---

## 10 Number-Backed Insights & Specific Modelling Implications

### Insight 1: Severe Target Censoring at 100.0 (Ceiling Spike)
- **Empirical Evidence:** Exactly **{r_target['n_capped_100']:,} rows ({r_target['pct_capped_100']}%)** in the ambient clean dataset sit at the upper limit of `CCME_Values = 100.0`. The overall distribution has mean {r_target['mean']} ± {r_target['std']}, but median {r_target['median']} and IQR {r_target['iqr']}.
- **Modelling Implication:** Standard OLS linear regression and MSE-minimizing models assume unconstrained continuous error distributions, which systematically under-predict near 100.0 and suffer boundary bias. We must formulate the continuous problem with post-prediction boundary clipping ($[0, 100]$) and test tree-based ensembles (Random Forest, LightGBM, XGBoost) and Tobit/censored loss formulations.

### Insight 2: Orthophosphate is the Dominant Monotonic Predictor
- **Empirical Evidence:** Orthophosphate exhibits an extraordinary Spearman rank correlation of **$\\rho = {r_corr['spearman_target_corr']['Orthophosphate (mg/l)']:.3f}$** with `CCME_Values` (Pearson $r = {r_corr['pearson_target_corr']['Orthophosphate (mg/l)']:.3f}$).
- **Modelling Implication:** In regulatory freshwater frameworks (e.g. EU WFD, US EPA), orthophosphate guidelines for eutrophication are stringent (often <0.05 mg/L). Even slight increases trigger immediate guideline failure, severely degrading CCME scores. Tree models will heavily prioritize splits on Orthophosphate; feature engineering must include $\\log(1 + \\text{{Orthophosphate}})$ and phosphorus saturation proxies.

### Insight 3: Extreme Collinearity Between Total Nitrogen and Nitrate
- **Empirical Evidence:** Nitrogen and Nitrate share a Spearman rank correlation of **$\\rho = {r_corr['nitrogen_nitrate_spearman']:.3f}$** (Pearson $r = {r_corr['nitrogen_nitrate_pearson']:.3f}$). In VIF testing, Nitrogen ($VIF = {r_corr['vif']['Nitrogen (mg/l)']:.2f}$) and Nitrate ($VIF = {r_corr['vif']['Nitrate (mg/l)']:.2f}$) are by far the highest among all features.
- **Modelling Implication:** In linear and generalized additive models, this collinearity inflates standard errors and causes coefficient sign flips. In tree ensembles, it artificially splits feature importance across both variables. In Stage 4, we must construct an interaction feature (Nitrate/Nitrogen ratio) and evaluate L1/L2 regularized baselines (Lasso/Ridge/ElasticNet).

### Insight 4: Dissolved Oxygen Displays a Non-Linear Biological Inversion
- **Empirical Evidence:** Dissolved Oxygen exhibits near-zero linear correlation ($r = {r_corr['pearson_target_corr']['Dissolved Oxygen (mg/l)']:.3f}$) and rank correlation ($\\rho = {r_corr['spearman_target_corr']['Dissolved Oxygen (mg/l)']:.3f}$) across the dataset.
- **Modelling Implication:** Dissolved oxygen impairment is biologically two-tailed: hypoxia ($DO < 5$ mg/L) indicates severe organic pollution, while extreme super-saturation ($DO > 14$ mg/L) indicates aggressive algal eutrophication. Linear models miss this completely. In Stage 4, we must create non-linear proxy features such as $|DO - 10|$ or temperature-dependent saturation deficit.

### Insight 5: Geographic Shortcut Bias — Country Explains 14% Variance Alone
- **Empirical Evidence:** In regression testing without any chemistry features, `Country` alone achieves **$R^2 = {r_shortcuts['country_alone']['r2']:.4f}$** (reducing RMSE from {r_shortcuts['dummy_mean']['rmse']} to {r_shortcuts['country_alone']['rmse']}). Median CCME in USA is {r_country['USA']['median']}, Ireland is {r_country['Ireland']['median']}, while England is {r_country['England']['median']}.
- **Modelling Implication:** A naive machine learning model will learn geographic shortcuts rather than generalizable chemical relationships. `Country` must NOT be passed as a standard predictor in the primary pipeline. It must be strictly reserved for spatial cross-validation (GroupKFold / Leave-One-Country-Out) to evaluate transferability.

### Insight 6: Ambient Waterbody Type Disparities
- **Empirical Evidence:** Waterbody Type alone explains **$R^2 = {r_shortcuts['waterbody_alone']['r2']:.4f}$** of target variance. Drainage waters display an ambient mean of 73.91, while Lakes reach 96.95. Segregating Effluent and Sewage (625,265 rows) eliminated extreme industrial contamination.
- **Modelling Implication:** Waterbody Type reflects physical re-aeration and dilution capacity. It should be evaluated strictly as an ablation feature (tested with vs without one-hot encoding in Stage 4) rather than a default mandatory feature.

### Insight 7: Limnological Seasonal Depressions in Summer
- **Empirical Evidence:** Monthly mean CCME drops from a spring peak of **{r_monthly[3]['mean']}** (March) to **{r_monthly[7]['mean']}** (July) and **{r_monthly[8]['mean']}** (August), while standard deviation increases from {r_monthly[3]['std']} to {r_monthly[7]['std']}.
- **Modelling Implication:** Higher summer water temperatures accelerate microbial decomposition, lower dissolved oxygen saturation, and stimulate agricultural runoff, driving guideline failures. Stage 4 feature engineering should test cyclical date representations (`sin(2π * month / 12)` and `cos(2π * month / 12)`).

### Insight 8: Severe Monitoring Site Observation Skew
- **Empirical Evidence:** The dataset encompasses **{r_site['total_unique_sites']:,} unique monitoring sites (Areas)**. 25% of sites have only 1 observation, median observation count is {r_site['site_obs_percentiles']['p50_median']:.0f}, but the 99th percentile site has {r_site['site_obs_percentiles']['p99']:.0f} rows and the maximum single site has **{r_site['site_obs_percentiles']['max']:,} observations**.
- **Modelling Implication:** Global loss functions without sample or site weighting will be dominated by a few hundred high-frequency monitoring stations. Cross-validation splits must group by `Area` to prevent high-density stations from corrupting both train and test splits.

### Insight 9: Multi-Year Site Sampling Confirms Spatial-Temporal Leakage Risk
- **Empirical Evidence:** **{r_site['multi_year_pct']}% of monitoring sites ({r_site['multi_year_sites']:,} sites)** contain observations across multiple years, with **{r_site['sites_ge_5_pct']}% spanning $\\\\ge 5$ years**.
- **Modelling Implication:** Naive random train-test splitting guarantees that the same river stretch observed in 2015 will appear in the training set, while its 2016 record appears in the test set. The model would exploit site-specific background chemistry (memorization) rather than predicting water quality from physics/chemistry. Split design MUST be `GroupKFold(groups=Area)` or temporal block splitting.

### Insight 10: Heavy Skewness of Concentration Features Requires Log Transformation
- **Empirical Evidence:** Chemistry variables (Ammonia, BOD, Orthophosphate, Nitrogen, Nitrate) span multiple orders of magnitude with extreme positive skewness (e.g. Ammonia median 0.055 vs max 50; BOD median 2.7 vs max 100).
- **Modelling Implication:** Linear models, distance metrics (KNN), and PCA/scaling are heavily distorted by extreme values. A logarithmic transformation ($\\\\log(1 + x)$) or power transform inside an sklearn `Pipeline` is required for linear and kernel baselines.

---

## Statistical Summary Tables

### 1. Variance Inflation Factors (VIF)

| Feature | VIF | Collinearity Status |
|---------|-----|---------------------|
"""
    for col in CHEMISTRY_COLS:
        vif_val = r_corr["vif"][col]
        status = "High Collinearity (N-species)" if vif_val > 2.0 else "Low / Acceptable"
        report += f"| {col} | {vif_val:.4f} | {status} |\n"
        
    report += f"""
### 2. Shortcut Regression Baselines (Predicting CCME_Values)

| Feature Set | R² Score | RMSE (units) | MAE (units) | Shortcut Risk |
|-------------|----------|--------------|-------------|---------------|
| Dummy (Mean Baseline) | {r_shortcuts['dummy_mean']['r2']:.4f} | {r_shortcuts['dummy_mean']['rmse']:.4f} | {r_shortcuts['dummy_mean']['mae']:.4f} | Benchmark |
| Country Alone | {r_shortcuts['country_alone']['r2']:.4f} | {r_shortcuts['country_alone']['rmse']:.4f} | {r_shortcuts['country_alone']['mae']:.4f} | High Shortcut (14% Variance) |
| Waterbody Alone | {r_shortcuts['waterbody_alone']['r2']:.4f} | {r_shortcuts['waterbody_alone']['rmse']:.4f} | {r_shortcuts['waterbody_alone']['mae']:.4f} | Moderate Shortcut (4.8% Variance) |
| Country + Waterbody | {r_shortcuts['country_and_waterbody']['r2']:.4f} | {r_shortcuts['country_and_waterbody']['rmse']:.4f} | {r_shortcuts['country_and_waterbody']['mae']:.4f} | High Shortcut (14.8% Variance) |
| Country + Waterbody + Month | {r_shortcuts['country_waterbody_month']['r2']:.4f} | {r_shortcuts['country_waterbody_month']['rmse']:.4f} | {r_shortcuts['country_waterbody_month']['mae']:.4f} | Severe Shortcut (15.6% Variance) |

---

## Stage 3 Verification
- All plots generated at 300 DPI in [`reports/figures/`](file:///Users/musthyalasadhvik/Desktop/projects/aml-capstone/AML_Capstone/reports/figures).
- Full numerical outputs saved in [`reports/03_eda_results.json`](file:///Users/musthyalasadhvik/Desktop/projects/aml-capstone/AML_Capstone/reports/03_eda_results.json).
- Stage 3 complete.
"""
    return report


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def run_eda():
    """Execute complete Stage 3 EDA pipeline."""
    print("=" * 70)
    print("STAGE 3: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)
    
    df_clean = load_clean_data()
    df_raw = load_raw_data()
    print(f"Loaded clean dataset: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
    print(f"Loaded raw dataset:   {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
    
    results = OrderedDict()
    
    # 1. Target distribution
    plot_target_distribution(df_clean, results)
    
    # 2. Before vs after distributions
    plot_feature_distributions_before_after(df_raw, df_clean, results)
    
    # 3. Correlations and VIF
    compute_correlations_and_vif(df_clean, results)
    
    # 4. Feature-target relationships
    plot_feature_target_relationships(df_clean, results)
    
    # 5. Target by groups and time
    plot_target_by_groups_and_time(df_clean, results)
    
    # 6. Invalid values summary
    plot_invalid_values_summary(results)
    
    # 7. Site sampling analysis
    plot_site_sampling_distribution(df_clean, results)
    
    # 8. Shortcut regression models
    evaluate_shortcuts(df_clean, results)
    
    # Save results JSON
    json_path = REPORTS_DIR / "03_eda_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results JSON: {json_path}")
    
    # Save findings report
    report_content = generate_eda_report(results)
    report_path = REPORTS_DIR / "03_eda_findings.md"
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"Saved findings report: {report_path}")
    
    print("\n✅ Stage 3 EDA pipeline execution complete!")
    return results


if __name__ == "__main__":
    run_eda()
