"""
Stage 5 — Data Splits Pipeline (Validity & Leakage Prevention)
==============================================================
Implements leakage-safe data splitting strategies for CCME water quality modeling:
  A. Site-held-out split: GroupShuffleSplit on Area (Train / Val / Test)
  B. Temporal split: Early (1990-2012) / Middle (2013-2017) / Late (2018-2023)
  C. Leave-One-Country-Out (LOCO) generalization stress test
  D. Plain random split: LABELED 'NOT FOR REPORTING' (demonstrates site leakage inflation)

Saves all split index arrays to data/processed/splits/ so that ALL downstream
models are evaluated on identical data slices.

Usage:
    python -m src.splits
"""

import os
import sys
import json
import logging
from pathlib import Path
from collections import OrderedDict
import numpy as np
import pandas as pd
from tabulate import tabulate

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------------------------
# Logging & Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PARQUET = PROJECT_ROOT / "data" / "processed" / "clean.parquet"
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
REPORTS_DIR = PROJECT_ROOT / "reports"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CONT = "CCME_Values"
TARGET_CLASS = "CCME_WQI"
GROUP_COL = "Area"
DATE_COL = "Date"
COUNTRY_COL = "Country"

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


# ---------------------------------------------------------------------------
# Split Implementations
# ---------------------------------------------------------------------------

def create_site_held_out_split(df: pd.DataFrame, train_size: float = 0.70, seed: int = SEED):
    """
    Two-stage GroupShuffleSplit ensuring zero overlap of Area between Train, Val, and Test.
    Train: ~70% groups
    Val:   ~15% groups
    Test:  ~15% groups
    """
    logger.info("Generating Site-held-out split (GroupShuffleSplit on %s)...", GROUP_COL)
    gss_outer = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_idx, temp_idx = next(gss_outer.split(df, groups=df[GROUP_COL]))

    temp_df = df.iloc[temp_idx]
    gss_inner = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed)
    val_sub_idx, test_sub_idx = next(gss_inner.split(temp_df, groups=temp_df[GROUP_COL]))

    val_idx = temp_idx[val_sub_idx]
    test_idx = temp_idx[test_sub_idx]

    # Validate zero site leakage
    train_sites = set(df.iloc[train_idx][GROUP_COL])
    val_sites = set(df.iloc[val_idx][GROUP_COL])
    test_sites = set(df.iloc[test_idx][GROUP_COL])

    assert len(train_sites & val_sites) == 0, "Leakage detected: Train and Val share sites!"
    assert len(train_sites & test_sites) == 0, "Leakage detected: Train and Test share sites!"
    assert len(val_sites & test_sites) == 0, "Leakage detected: Val and Test share sites!"

    logger.info("Site split complete: Train=%d, Val=%d, Test=%d (Zero site overlap verified)",
                len(train_idx), len(val_idx), len(test_idx))
    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "train_sites": len(train_sites),
        "val_sites": len(val_sites),
        "test_sites": len(test_sites),
    }


def create_temporal_split(df: pd.DataFrame):
    """
    Temporal split based on data collection years:
      Train: 1990 - 2012 (~60.4% data)
      Val:   2013 - 2017 (~22.2% data)
      Test:  2018 - 2023 (~17.4% data)
    """
    logger.info("Generating Temporal split (Train: <=2012, Val: 2013-2017, Test: >=2018)...")
    years = pd.to_datetime(df[DATE_COL]).dt.year

    train_mask = (years <= 2012)
    val_mask = (years >= 2013) & (years <= 2017)
    test_mask = (years >= 2018)

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]

    assert len(train_idx) + len(val_idx) + len(test_idx) == len(df), "Row count mismatch in temporal split!"
    logger.info("Temporal split complete: Train=%d, Val=%d, Test=%d",
                len(train_idx), len(val_idx), len(test_idx))
    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "cut_years": {"train": "1990-2012", "val": "2013-2017", "test": "2018-2023"},
    }


def create_loco_splits(df: pd.DataFrame):
    """
    Leave-One-Country-Out (LOCO) splits for spatial generalization stress tests.
    Generates folds holding out each country in turn.
    """
    logger.info("Generating Leave-One-Country-Out (LOCO) splits...")
    countries = df[COUNTRY_COL].unique()
    loco = {}
    for country in countries:
        test_mask = (df[COUNTRY_COL] == country)
        train_mask = ~test_mask
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        loco[str(country)] = {
            "train_idx": train_idx,
            "test_idx": test_idx,
            "held_out_country": str(country),
            "test_size": len(test_idx),
            "train_size": len(train_idx),
        }
    return loco


def create_random_split_leakage(df: pd.DataFrame, test_size: float = 0.20, seed: int = SEED):
    """
    Plain random row split: Strictly LABELED 'NOT FOR REPORTING'.
    Only used to demonstrate how repeating site monitoring across train/test
    artificially inflates model evaluation metrics.
    """
    logger.info("Generating Random split (NOT FOR REPORTING - Site Leakage Demonstration)...")
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=test_size, random_state=seed
    )
    train_sites = set(df.iloc[train_idx][GROUP_COL])
    test_sites = set(df.iloc[test_idx][GROUP_COL])
    overlap_sites = train_sites & test_sites
    overlap_pct = 100 * len(overlap_sites) / len(test_sites) if len(test_sites) > 0 else 0

    logger.warning("Random split has %d overlapping sites (%.2f%% of test sites appear in train)!",
                   len(overlap_sites), overlap_pct)
    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "overlapping_sites": len(overlap_sites),
        "overlap_pct": overlap_pct,
    }


# ---------------------------------------------------------------------------
# Distribution Analysis & Metric Calculation
# ---------------------------------------------------------------------------

def compute_split_distribution(df: pd.DataFrame, indices: np.ndarray, split_name: str):
    """Summarizes target and class distributions for a given split subset."""
    sub_df = df.iloc[indices]
    n_rows = len(sub_df)
    n_sites = sub_df[GROUP_COL].nunique()

    # Continuous target stats
    vals = sub_df[TARGET_CONT]
    mean_val = float(vals.mean())
    std_val = float(vals.std())
    median_val = float(vals.median())
    q25, q75 = float(vals.quantile(0.25)), float(vals.quantile(0.75))
    pct_100 = float(100 * (vals == 100.0).sum() / n_rows)

    # Class stats
    class_counts = sub_df[TARGET_CLASS].value_counts(normalize=True).to_dict()
    class_pcts = {k: round(v * 100, 2) for k, v in class_counts.items()}

    return {
        "split_name": split_name,
        "rows": n_rows,
        "pct_of_total": round(100 * n_rows / len(df), 2),
        "unique_sites": n_sites,
        "target_mean": round(mean_val, 3),
        "target_std": round(std_val, 3),
        "target_median": round(median_val, 3),
        "target_iqr": round(q75 - q25, 3),
        "target_pct_100": round(pct_100, 2),
        "class_proportions": class_pcts,
    }


def evaluate_leakage_impact(df: pd.DataFrame, site_split: dict, random_split: dict):
    """
    Trains a Ridge regression baseline on Site-Held-Out vs Random Split
    to quantify the empirical leakage inflation gap.
    """
    logger.info("Evaluating leakage impact with Ridge regression baseline...")
    rng = np.random.RandomState(SEED)
    
    n_train = 100_000
    n_test = 25_000

    # 1. Site-held-out evaluation
    sub_train_site = rng.choice(site_split["train_idx"], size=min(n_train, len(site_split["train_idx"])), replace=False)
    sub_test_site = rng.choice(site_split["test_idx"], size=min(n_test, len(site_split["test_idx"])), replace=False)

    pipe_site = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=SEED))
    ])

    X_train_site = df.iloc[sub_train_site][CHEMISTRY_COLS]
    y_train_site = df.iloc[sub_train_site][TARGET_CONT]
    X_test_site = df.iloc[sub_test_site][CHEMISTRY_COLS]
    y_test_site = df.iloc[sub_test_site][TARGET_CONT]

    pipe_site.fit(X_train_site, y_train_site)
    preds_site = pipe_site.predict(X_test_site)
    r2_site = float(r2_score(y_test_site, preds_site))
    rmse_site = float(root_mean_squared_error(y_test_site, preds_site))
    mae_site = float(mean_absolute_error(y_test_site, preds_site))

    # 2. Random split evaluation (NOT FOR REPORTING)
    sub_train_rand = rng.choice(random_split["train_idx"], size=min(n_train, len(random_split["train_idx"])), replace=False)
    sub_test_rand = rng.choice(random_split["test_idx"], size=min(n_test, len(random_split["test_idx"])), replace=False)

    pipe_rand = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=SEED))
    ])

    X_train_rand = df.iloc[sub_train_rand][CHEMISTRY_COLS]
    y_train_rand = df.iloc[sub_train_rand][TARGET_CONT]
    X_test_rand = df.iloc[sub_test_rand][CHEMISTRY_COLS]
    y_test_rand = df.iloc[sub_test_rand][TARGET_CONT]

    pipe_rand.fit(X_train_rand, y_train_rand)
    preds_rand = pipe_rand.predict(X_test_rand)
    r2_rand = float(r2_score(y_test_rand, preds_rand))
    rmse_rand = float(root_mean_squared_error(y_test_rand, preds_rand))
    mae_rand = float(mean_absolute_error(y_test_rand, preds_rand))

    logger.info("Leakage Evaluation Finished: Site-held-out R²=%.4f vs Random R²=%.4f", r2_site, r2_rand)
    return {
        "site_held_out": {"r2": round(r2_site, 4), "rmse": round(rmse_site, 4), "mae": round(mae_site, 4)},
        "random_split_not_for_reporting": {"r2": round(r2_rand, 4), "rmse": round(rmse_rand, 4), "mae": round(mae_rand, 4)},
        "r2_inflation_delta": round(r2_rand - r2_site, 4),
        "rmse_reduction_delta": round(rmse_site - rmse_rand, 4),
    }


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

def generate_markdown_report(site_stats, time_stats, loco_stats, random_stats, leakage_eval):
    """Generates the official Stage 5 documentation report."""
    md = []
    md.append("# Stage 5 — Data Splits & Validity Protocol Report\n")
    md.append("**Status:** COMPLETE — Zero-Leakage Ground Truth Enforced  ")
    md.append(f"**Target Analyzed:** `{TARGET_CONT}` (continuous) and `{TARGET_CLASS}` (derived)  ")
    md.append(f"**Grouping Unit:** `{GROUP_COL}` (Monitoring Station / Area)\n")
    md.append("---\n")

    md.append("## 1. Executive Summary & Validity Imperative\n")
    md.append("> **CRITICAL METHODOLOGICAL DIRECTIVE:**  \n")
    md.append("> Random row-level train-test splits are **STRICTLY INVALID** for reporting in water quality monitoring. ")
    md.append("Because identical monitoring sites repeat hundreds or thousands of times across years, a naive split causes ")
    md.append("**spatial and station-identity leakage**: the model memorizes site baseline chemistry rather than learning ")
    md.append("generalizable environmental physics. All reported results must use the **Site-Held-Out** (Grouped) split or the **Temporal** split.\n")

    md.append("## 2. Split A — Site-Held-Out Split (Primary Benchmark)\n")
    md.append("Partitioned via nested `GroupShuffleSplit` on `Area`:\n")
    headers = ["Split", "Rows", "% Data", "Unique Sites", "Target Mean", "Target Std", "% at 100.0"]
    rows = [
        [s["split_name"], f"{s['rows']:,}", f"{s['pct_of_total']}%", f"{s['unique_sites']:,}",
         s["target_mean"], s["target_std"], f"{s['target_pct_100']}%"]
        for s in site_stats
    ]
    md.append(tabulate(rows, headers=headers, tablefmt="github"))
    md.append("\n\n**Class Distributions (%):**\n")
    class_headers = ["Split"] + sorted(list(site_stats[0]["class_proportions"].keys()))
    class_rows = []
    for s in site_stats:
        r = [s["split_name"]] + [f"{s['class_proportions'].get(c, 0):.2f}%" for c in class_headers[1:]]
        class_rows.append(r)
    md.append(tabulate(class_rows, headers=class_headers, tablefmt="github"))

    md.append("\n\n## 3. Split B — Temporal Split (Longitudinal Generalization)\n")
    md.append("Partitioned chronologically into three eras:\n")
    t_rows = [
        [s["split_name"], f"{s['rows']:,}", f"{s['pct_of_total']}%", f"{s['unique_sites']:,}",
         s["target_mean"], s["target_std"], f"{s['target_pct_100']}%"]
        for s in time_stats
    ]
    md.append(tabulate(t_rows, headers=headers, tablefmt="github"))

    md.append("\n\n## 4. Split C — Leave-One-Country-Out (LOCO) Summary\n")
    loco_headers = ["Held-Out Country", "Test Samples", "Train Samples", "Test %"]
    loco_rows = [
        [c, f"{d['test_size']:,}", f"{d['train_size']:,}", f"{100 * d['test_size'] / (d['test_size'] + d['train_size']):.2f}%"]
        for c, d in loco_stats.items()
    ]
    md.append(tabulate(loco_rows, headers=loco_headers, tablefmt="github"))

    md.append("\n\n## 5. Site Leakage Quantification — Random Split (NOT FOR REPORTING ⚠️)\n")
    md.append("To rigorously quantify the deceptive score inflation caused by row-level splitting, ")
    md.append("we trained an identical baseline Ridge regression on both splits:\n")

    leak_headers = ["Splitting Strategy", "Test R²", "Test RMSE", "Test MAE", "Status"]
    leak_rows = [
        ["Site-Held-Out (Grouped Area)", leakage_eval["site_held_out"]["r2"],
         leakage_eval["site_held_out"]["rmse"], leakage_eval["site_held_out"]["mae"], "VALID / REPORTED"],
        ["Random Row Split (Leakage)", leakage_eval["random_split_not_for_reporting"]["r2"],
         leakage_eval["random_split_not_for_reporting"]["rmse"], leakage_eval["random_split_not_for_reporting"]["mae"],
         "⚠️ INVALID (NOT FOR REPORTING)"],
    ]
    md.append(tabulate(leak_rows, headers=leak_headers, tablefmt="github"))

    md.append(f"\n\n**Empirical Leakage Finding:**  \n")
    md.append(f"- **Overlapping Sites in Random Split:** {random_stats['overlapping_sites']:,} sites ")
    md.append(f"({random_stats['overlap_pct']:.2f}% of test monitoring sites are present in training!).  \n")
    md.append(f"- **Artificial $R^2$ Inflation:** $+{leakage_eval['r2_inflation_delta']:.4f}$  \n")
    md.append(f"- **Artificial RMSE Reduction:** $-{leakage_eval['rmse_reduction_delta']:.4f}$  \n")
    md.append("Random row splitting creates the dangerous illusion of higher model accuracy by leaking site geography and station idiosyncrasies.\n")

    md.append("## 6. Artifact Storage\n")
    md.append("All split indices have been saved to disk for exact multi-model replication:\n")
    md.append("- `data/processed/splits/site_held_out_split.npz`\n")
    md.append("- `data/processed/splits/time_split.npz`\n")
    md.append("- `data/processed/splits/loco_split.npz`\n")
    md.append("- `data/processed/splits/random_split_not_for_reporting.npz`\n")

    return "\n".join(md)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("STAGE 5 — LEAKAGE-SAFE DATA SPLITS PIPELINE")
    print("=" * 70)

    if not CLEAN_PARQUET.exists():
        raise FileNotFoundError(f"Clean parquet not found at {CLEAN_PARQUET}. Run Stage 2 first.")

    logger.info("Loading clean ambient dataset: %s", CLEAN_PARQUET)
    df = pd.read_parquet(CLEAN_PARQUET)
    logger.info("Dataset loaded: %d rows, %d unique sites", len(df), df[GROUP_COL].nunique())

    # A. Site-Held-Out Split
    site_split = create_site_held_out_split(df)
    np.savez_compressed(
        SPLITS_DIR / "site_held_out_split.npz",
        train_idx=site_split["train_idx"],
        val_idx=site_split["val_idx"],
        test_idx=site_split["test_idx"],
    )
    logger.info("Saved site_held_out_split.npz")

    site_stats = [
        compute_split_distribution(df, site_split["train_idx"], "Train (Site Held Out)"),
        compute_split_distribution(df, site_split["val_idx"], "Val (Site Held Out)"),
        compute_split_distribution(df, site_split["test_idx"], "Test (Site Held Out)"),
    ]

    # B. Temporal Split
    time_split = create_temporal_split(df)
    np.savez_compressed(
        SPLITS_DIR / "time_split.npz",
        train_idx=time_split["train_idx"],
        val_idx=time_split["val_idx"],
        test_idx=time_split["test_idx"],
    )
    logger.info("Saved time_split.npz")

    time_stats = [
        compute_split_distribution(df, time_split["train_idx"], "Train (1990-2012)"),
        compute_split_distribution(df, time_split["val_idx"], "Val (2013-2017)"),
        compute_split_distribution(df, time_split["test_idx"], "Test (2018-2023)"),
    ]

    # C. Leave-One-Country-Out Split
    loco_split = create_loco_splits(df)
    np.savez_compressed(
        SPLITS_DIR / "loco_split.npz",
        **{f"{c}_test_idx": d["test_idx"] for c, d in loco_split.items()},
        **{f"{c}_train_idx": d["train_idx"] for c, d in loco_split.items()},
    )
    logger.info("Saved loco_split.npz")

    # D. Random Split (Leakage Demonstration)
    random_split = create_random_split_leakage(df)
    np.savez_compressed(
        SPLITS_DIR / "random_split_not_for_reporting.npz",
        train_idx=random_split["train_idx"],
        test_idx=random_split["test_idx"],
    )
    logger.info("Saved random_split_not_for_reporting.npz")

    # Leakage evaluation
    leakage_eval = evaluate_leakage_impact(df, site_split, random_split)

    # Save Results JSON & Markdown Report
    full_results = {
        "site_held_out": site_stats,
        "temporal": time_stats,
        "loco": {c: {"train_size": d["train_size"], "test_size": d["test_size"]} for c, d in loco_split.items()},
        "random_split_leakage": {
            "overlapping_sites": random_split["overlapping_sites"],
            "overlap_pct": round(random_split["overlap_pct"], 2),
        },
        "leakage_evaluation": leakage_eval,
    }

    results_path = REPORTS_DIR / "05_splits_results.json"
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=4)
    logger.info("Saved %s", results_path)

    report_md = generate_markdown_report(site_stats, time_stats, loco_split, random_split, leakage_eval)
    report_path = REPORTS_DIR / "05_splits_report.md"
    with open(report_path, "w") as f:
        f.write(report_md)
    logger.info("Saved %s", report_path)

    print("\n" + report_md)
    print("\n✅ STAGE 5 COMPLETE.")


if __name__ == "__main__":
    main()
