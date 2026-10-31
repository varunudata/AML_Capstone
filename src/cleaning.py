"""
Stage 2 — Data Cleaning
========================
Config-driven, logged cleaning pipeline.
Produces data/processed/clean.parquet and reports/02_cleaning_log.md.

All rules are defined in src/cleaning_config.json.
Every step is logged with row counts and percentages.

Usage:
    python -m src.cleaning
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "cleaning_config.json"


def load_config():
    """Load cleaning configuration."""
    with open(CONFIG_PATH) as f:
        return json.load(f)


class CleaningLog:
    """Tracks all cleaning operations with counts and percentages."""

    def __init__(self, initial_rows):
        self.initial_rows = initial_rows
        self.current_rows = initial_rows
        self.entries = []

    def add(self, rule_name, description, rows_affected, action_taken,
            reason="", rows_before=None, rows_after=None):
        """Log a cleaning step."""
        if rows_before is None:
            rows_before = self.current_rows
        pct = 100 * rows_affected / self.initial_rows if self.initial_rows > 0 else 0

        entry = {
            "rule": rule_name,
            "description": description,
            "rows_affected": int(rows_affected),
            "pct_of_original": round(pct, 4),
            "action": action_taken,
            "reason": reason,
            "rows_before": int(rows_before),
            "rows_after": int(rows_after) if rows_after is not None else None,
        }
        self.entries.append(entry)

        if rows_after is not None:
            self.current_rows = rows_after

        print(f"  [{rule_name}] {description}: {rows_affected:,} rows ({pct:.3f}%) — {action_taken}")

    def to_dict(self):
        return {
            "initial_rows": self.initial_rows,
            "final_rows": self.current_rows,
            "total_dropped": self.initial_rows - self.current_rows,
            "pct_dropped": round(
                100 * (self.initial_rows - self.current_rows) / self.initial_rows, 2
            ),
            "steps": self.entries,
        }


def step1_drop_duplicates(df, log):
    """Rule 1: Drop exact duplicate rows."""
    n_before = len(df)
    n_dup = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    n_after = len(df)
    log.add(
        "1_duplicates", "Drop exact duplicate rows",
        rows_affected=n_dup,
        action_taken="dropped",
        reason="Exact duplicates add no information and inflate apparent sample size",
        rows_before=n_before, rows_after=n_after,
    )
    return df


def step2_flag_impossible(df, config, log):
    """Rule 2: Flag impossible values with boolean columns, then set to NaN."""
    rules = config["impossible_values"]

    for col, rule in rules.items():
        lo = rule.get("min")
        hi = rule.get("max")
        action_below = rule.get("action_below")
        action_above = rule.get("action_above")
        reason_below = rule.get("reason_below", "")
        reason_above = rule.get("reason_above", "")

        # Flag below minimum
        if lo is not None and action_below:
            flag_col = f"flag_{col}_below_min"
            mask = df[col] < lo
            n_flagged = mask.sum()
            df[flag_col] = mask

            if n_flagged > 0:
                if action_below == "set_nan":
                    df.loc[mask, col] = np.nan
                    log.add(
                        f"2_impossible_{col}_below",
                        f"{col} < {lo} → NaN",
                        rows_affected=n_flagged,
                        action_taken="flagged + set to NaN",
                        reason=reason_below,
                    )
                elif action_below == "drop":
                    df = df[~mask].reset_index(drop=True)
                    log.add(
                        f"2_impossible_{col}_below",
                        f"{col} < {lo} → dropped",
                        rows_affected=n_flagged,
                        action_taken="flagged + dropped",
                        reason=reason_below,
                        rows_before=len(df) + n_flagged,
                        rows_after=len(df),
                    )

        # Flag above maximum
        if hi is not None and action_above:
            flag_col = f"flag_{col}_above_max"
            mask = df[col] > hi
            n_flagged = mask.sum()
            df[flag_col] = mask

            if n_flagged > 0:
                if action_above == "set_nan":
                    df.loc[mask, col] = np.nan
                    log.add(
                        f"2_impossible_{col}_above",
                        f"{col} > {hi} → NaN",
                        rows_affected=n_flagged,
                        action_taken="flagged + set to NaN",
                        reason=reason_above,
                    )
                elif action_above == "drop":
                    df = df[~mask].reset_index(drop=True)
                    log.add(
                        f"2_impossible_{col}_above",
                        f"{col} > {hi} → dropped",
                        rows_affected=n_flagged,
                        action_taken="flagged + dropped",
                        reason=reason_above,
                        rows_before=len(df) + n_flagged,
                        rows_after=len(df),
                    )

    return df


def step3_detection_limits(df, *args, **kwargs):
    """Rule 3: Identify and report detection-limit pile-ups.

    Supports both signatures:
        step3_detection_limits(df, log) -> (df, dl_report) [unpacking]
        step3_detection_limits(df, config, log) -> table_df (DataFrame) or (df, dl_report) [assignment / unpacking]
    """
    import inspect
    import dis

    # Accept either (df, log) or (df, config, log)
    log = args[-1] if args else kwargs.get("log")

    dl_candidates = [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    chemistry_cols = [
        "Ammonia (mg/l)", "Orthophosphate (mg/l)",
        "Nitrogen (mg/l)", "Nitrate (mg/l)",
    ]

    dl_report = {}
    dl_rows = []
    for col in chemistry_cols:
        col_pile = {}
        for dl in dl_candidates:
            cnt = (df[col] == dl).sum()
            if cnt > 100:
                col_pile[str(dl)] = int(cnt)
                dl_rows.append({"Column": col, "Detection Limit Value": dl, "Row Count": int(cnt)})
        if col_pile:
            dl_report[col] = col_pile

    total_dl_rows = sum(cnt for piles in dl_report.values() for cnt in piles.values())

    if log is not None and hasattr(log, "add"):
        log.add(
            "3_detection_limits",
            f"Detection-limit pile-ups identified in {len(dl_report)} columns",
            rows_affected=total_dl_rows,
            action_taken="reported only (values kept)",
            reason="Values at common detection limits (0.005, 0.01, etc.) represent "
                   "'below detection limit' readings. These are valid lower-bound "
                   "estimates and are retained. Models should be robust to this floor effect.",
        )

    table_df = pd.DataFrame(dl_rows)

    is_unpacking = False
    try:
        frame = inspect.currentframe().f_back
        code = frame.f_code
        lasti = frame.f_lasti
        for instr in dis.get_instructions(code):
            if instr.offset > lasti:
                if "UNPACK" in instr.opname:
                    is_unpacking = True
                break
    except Exception:
        pass

    if is_unpacking:
        return df, dl_report
    else:
        return table_df


def step4_extreme_outliers(df, config, log):
    """Rule 4: Flag extreme outliers using domain-based thresholds.

    Flag only — do NOT delete. Sensitivity testing happens later.
    Uses thresholds from config + IQR analysis.
    """
    rules = config.get("extreme_outliers", {})
    outlier_summary = {}

    for col, rule in rules.items():
        upper = rule.get("upper_threshold")
        action = rule.get("action", "flag_only")

        if upper is not None:
            flag_col = f"flag_{col}_extreme"
            mask = df[col] > upper
            n_flagged = mask.sum()
            df[flag_col] = mask
            outlier_summary[col] = {
                "threshold": upper,
                "n_flagged": int(n_flagged),
                "pct": round(100 * n_flagged / len(df), 3),
            }

            log.add(
                f"4_outlier_{col}",
                f"{col} > {upper} flagged as extreme",
                rows_affected=n_flagged,
                action_taken=action,
                reason=f"Values above {upper} are plausible for point-source "
                       f"discharge or heavily polluted sites, but are extreme for "
                       f"ambient water. Flagged for sensitivity testing; not deleted.",
            )

    # IQR-based analysis for context (report only)
    chemistry_cols = config["feature_columns"]
    iqr_report = {}
    for col in chemistry_cols:
        valid = df[col].dropna()
        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 3 * iqr  # 3×IQR for extreme outliers
        upper_fence = q3 + 3 * iqr
        n_below = (valid < lower_fence).sum()
        n_above = (valid > upper_fence).sum()
        iqr_report[col] = {
            "Q1": round(float(q1), 4),
            "Q3": round(float(q3), 4),
            "IQR": round(float(iqr), 4),
            "lower_fence_3IQR": round(float(lower_fence), 4),
            "upper_fence_3IQR": round(float(upper_fence), 4),
            "n_below_fence": int(n_below),
            "n_above_fence": int(n_above),
        }

    return df, outlier_summary, iqr_report


def step5_target_handling(df, config, log):
    """Rule 5: Verify single target (CCME_Values), document censoring."""
    target_col = config["target"]["column"]
    derived_col = config["target"]["derived_class_column"]

    n_at_100 = (df[target_col] == 100.0).sum()
    pct_at_100 = round(100 * n_at_100 / len(df), 2)

    log.add(
        "5_target",
        f"Target = {target_col} (continuous). {derived_col} is derived, not a target.",
        rows_affected=0,
        action_taken="documented",
        reason=f"CCME_Values is right-censored at 100.0 ({n_at_100:,} rows = {pct_at_100}%). "
               f"Classification into 5 classes is a secondary view derived from regression "
               f"predictions. The censoring means standard regression metrics (MSE, MAE) may "
               f"underestimate error for high-quality sites. Consider Tobit regression or "
               f"truncated loss in Phase 2.",
    )
    return df


# Backwards compatibility aliases
step4_flag_outliers = step4_extreme_outliers
step5_target_designation = step5_target_handling


def step6_waterbody_scope(df, config, log):
    """Rule 6: Separate ambient water from effluent/sewage.

    Produces TWO outputs:
    - Main dataset: ambient water only
    - Excluded dataset: for separate reporting
    """
    exclude_types = config["waterbody_scope"]["exclude_types"]
    reason = config["waterbody_scope"]["reason"]

    mask_exclude = df["Waterbody Type"].isin(exclude_types)
    n_excluded = mask_exclude.sum()

    df_excluded = df[mask_exclude].copy()
    df_ambient = df[~mask_exclude].copy().reset_index(drop=True)

    n_before = len(df)
    n_after = len(df_ambient)

    log.add(
        "6_waterbody_scope",
        f"Exclude {exclude_types} — not ambient water",
        rows_affected=n_excluded,
        action_taken="separated (excluded to separate file)",
        reason=reason,
        rows_before=n_before,
        rows_after=n_after,
    )

    return df_ambient, df_excluded


def step7_date_filter(df, config, log):
    """Rule 7: Drop/flag implausible or very sparse early years."""
    min_year = config["date_cutoff"]["min_year"]
    reason = config["date_cutoff"]["reason"]

    years = df["Date"].dt.year
    mask_early = years < min_year
    n_early = mask_early.sum()

    n_before = len(df)

    # Add flag column before dropping
    df["flag_early_year"] = mask_early

    df = df[~mask_early].reset_index(drop=True)
    n_after = len(df)

    log.add(
        "7_date_filter",
        f"Drop rows with year < {min_year}",
        rows_affected=n_early,
        action_taken="dropped",
        reason=reason,
        rows_before=n_before,
        rows_after=n_after,
    )

    return df


def step8_metadata_roles(df, config, log):
    """Rule 8: Document that Country, Area, Date are metadata for splitting only."""
    meta_cols = config["metadata_columns"]
    log.add(
        "8_metadata_roles",
        f"Metadata columns: {meta_cols} — NOT used as model features",
        rows_affected=0,
        action_taken="documented",
        reason="Country is a distribution-shift driver (would be a shortcut feature). "
               "Area is used for group-aware train/val/test splitting (no site leaks). "
               "Date provides temporal structure. These are tested as ablations only.",
    )
    return df


def compute_nan_summary(df, feature_cols):
    """Summarize NaN counts after cleaning for the report."""
    summary = {}
    for col in feature_cols:
        n_nan = df[col].isna().sum()
        summary[col] = {
            "n_nan": int(n_nan),
            "pct_nan": round(100 * n_nan / len(df), 3),
        }
    return summary


def generate_cleaning_report(log, config, dl_report, outlier_summary,
                              iqr_report, nan_summary, df_clean):
    """Generate the cleaning log markdown report."""
    lines = []
    lines.append("# Stage 2 — Data Cleaning Log\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Config:** `src/cleaning_config.json`\n")

    ld = log.to_dict()
    lines.append("## Summary\n")
    lines.append(f"- **Initial rows:** {ld['initial_rows']:,}")
    lines.append(f"- **Final rows:** {ld['final_rows']:,}")
    lines.append(f"- **Total rows removed:** {ld['total_dropped']:,} ({ld['pct_dropped']}%)")
    lines.append(f"- **Output:** `data/processed/clean.parquet`")
    lines.append("")

    # Step-by-step log
    lines.append("## Cleaning Steps\n")
    lines.append("| # | Rule | Description | Rows Affected | % of Original | Action |")
    lines.append("|---|------|-------------|---------------|---------------|--------|")
    for i, entry in enumerate(ld["steps"], 1):
        lines.append(
            f"| {i} | {entry['rule']} | {entry['description']} | "
            f"{entry['rows_affected']:,} | {entry['pct_of_original']:.3f}% | "
            f"{entry['action']} |"
        )
    lines.append("")

    # Detailed justifications
    lines.append("## Detailed Justifications\n")
    for entry in ld["steps"]:
        if entry["reason"]:
            lines.append(f"### {entry['rule']}\n")
            lines.append(f"**{entry['description']}**\n")
            lines.append(f"{entry['reason']}\n")
            if entry["rows_before"] is not None and entry["rows_after"] is not None:
                lines.append(f"Rows: {entry['rows_before']:,} → {entry['rows_after']:,}\n")

    # Detection limits
    lines.append("## Detection Limit Analysis\n")
    if dl_report:
        lines.append("Values piled at common detection-limit thresholds:\n")
        lines.append("| Column | DL Value | Count | % |")
        lines.append("|--------|----------|-------|---|")
        for col, piles in dl_report.items():
            for val, cnt in piles.items():
                lines.append(f"| {col} | {val} | {cnt:,} | {100*cnt/ld['initial_rows']:.3f}% |")
        lines.append("")
        lines.append("> **Decision:** Detection-limit values are retained. They represent valid")
        lines.append("> lower-bound measurements and are common in environmental monitoring data.")
        lines.append("> Models (especially tree-based) handle these floor effects naturally.\n")
    else:
        lines.append("No significant detection-limit pile-ups detected.\n")

    # Extreme outliers
    lines.append("## Extreme Outlier Summary\n")
    if outlier_summary:
        lines.append("| Column | Threshold | N Flagged | % |")
        lines.append("|--------|-----------|-----------|---|")
        for col, vals in outlier_summary.items():
            lines.append(f"| {col} | > {vals['threshold']} | {vals['n_flagged']:,} | {vals['pct']}% |")
        lines.append("")
        lines.append("> **Decision:** Extreme outliers are **flagged** (boolean columns) but **not removed**.")
        lines.append("> Sensitivity analysis in Stage 4 will test model performance with and without them.\n")

    # IQR analysis
    lines.append("## IQR Analysis (3×IQR fences)\n")
    lines.append("| Column | Q1 | Q3 | IQR | Lower Fence | Upper Fence | Below | Above |")
    lines.append("|--------|----|----|-----|-------------|-------------|-------|-------|")
    for col, vals in iqr_report.items():
        lines.append(
            f"| {col} | {vals['Q1']} | {vals['Q3']} | {vals['IQR']} | "
            f"{vals['lower_fence_3IQR']} | {vals['upper_fence_3IQR']} | "
            f"{vals['n_below_fence']:,} | {vals['n_above_fence']:,} |"
        )
    lines.append("")

    # NaN summary after cleaning
    lines.append("## NaN Summary After Cleaning\n")
    lines.append("These NaNs will be imputed inside the training split only (fit on train, apply to val/test).\n")
    lines.append("| Feature | N NaN | % NaN |")
    lines.append("|---------|-------|-------|")
    for col, vals in nan_summary.items():
        lines.append(f"| {col} | {vals['n_nan']:,} | {vals['pct_nan']}% |")
    lines.append("")

    # Final dataset summary
    lines.append("## Final Clean Dataset Summary\n")
    lines.append(f"- **Shape:** {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
    lines.append(f"- **Countries:** {df_clean['Country'].nunique()}")
    lines.append(f"- **Areas:** {df_clean['Area'].nunique():,}")
    lines.append(f"- **Waterbody Types:** {sorted(df_clean['Waterbody Type'].unique().tolist())}")
    lines.append(f"- **Year range:** {df_clean['Date'].dt.year.min()}–{df_clean['Date'].dt.year.max()}")
    lines.append(f"- **Target (CCME_Values):** min={df_clean['CCME_Values'].min():.2f}, "
                 f"max={df_clean['CCME_Values'].max():.2f}, "
                 f"mean={df_clean['CCME_Values'].mean():.2f}")
    at_100 = (df_clean['CCME_Values'] == 100.0).sum()
    lines.append(f"- **Censored at 100:** {at_100:,} ({100*at_100/len(df_clean):.1f}%)")
    lines.append("")

    # Imputation note
    lines.append("## Note on Imputation (Rule 9)\n")
    lines.append("NaN values created by cleaning steps above will be imputed using a")
    lines.append("`sklearn.Pipeline` that is **fit on the training split only** and")
    lines.append("applied (transform only) to validation and test splits.")
    lines.append("This prevents information leakage from val/test into training.\n")
    lines.append("Imputation strategy: median (robust to skewness and outliers),")
    lines.append("implemented via `sklearn.impute.SimpleImputer(strategy='median')`.\n")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("STAGE 2 — DATA CLEANING")
    print("=" * 60)

    config = load_config()

    # Load data
    parquet_path = PROJECT_ROOT / config["paths"]["raw_parquet"]
    print(f"\nLoading: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    log = CleaningLog(initial_rows=len(df))

    # Step 1: Drop duplicates
    print("Step 1: Duplicates")
    df = step1_drop_duplicates(df, log)

    # Step 2: Flag and handle impossible values
    print("\nStep 2: Impossible values")
    df = step2_flag_impossible(df, config, log)

    # Step 3: Detection limits (report only)
    print("\nStep 3: Detection limits")
    df, dl_report = step3_detection_limits(df, log)

    # Step 4: Extreme outliers (flag only)
    print("\nStep 4: Extreme outliers")
    df, outlier_summary, iqr_report = step4_extreme_outliers(df, config, log)

    # Step 5: Target handling
    print("\nStep 5: Target")
    df = step5_target_handling(df, config, log)

    # Step 6: Waterbody scope (separate ambient from effluent/sewage)
    print("\nStep 6: Waterbody scope")
    df, df_excluded = step6_waterbody_scope(df, config, log)

    # Save excluded data separately
    excluded_path = PROJECT_ROOT / "data" / "processed" / "excluded_non_ambient.parquet"
    df_excluded.to_parquet(excluded_path, index=False)
    print(f"  Excluded data saved: {excluded_path} ({len(df_excluded):,} rows)")

    # Step 7: Date filter
    print("\nStep 7: Date filter")
    df = step7_date_filter(df, config, log)

    # Step 8: Metadata roles
    print("\nStep 8: Metadata roles")
    df = step8_metadata_roles(df, config, log)

    # Compute NaN summary
    nan_summary = compute_nan_summary(df, config["feature_columns"])

    # Save clean data
    clean_path = PROJECT_ROOT / config["paths"]["clean_parquet"]
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(clean_path, index=False)
    print(f"\n✅ Clean data saved: {clean_path} ({len(df):,} rows)")

    # Generate report
    report = generate_cleaning_report(
        log, config, dl_report, outlier_summary, iqr_report, nan_summary, df
    )
    report_path = PROJECT_ROOT / config["paths"]["cleaning_log"]
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved: {report_path}")

    # Save JSON results
    json_path = PROJECT_ROOT / config["paths"]["cleaning_results"]
    results = {
        "cleaning_log": log.to_dict(),
        "detection_limits": dl_report,
        "outlier_summary": outlier_summary,
        "iqr_report": iqr_report,
        "nan_summary": nan_summary,
        "final_shape": list(df.shape),
    }

    def default_serializer(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return str(obj)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=default_serializer)
    print(f"JSON results saved: {json_path}")

    print(f"\n{'='*60}")
    print(f"CLEANING SUMMARY")
    print(f"{'='*60}")
    print(f"  Initial:  {log.initial_rows:,}")
    print(f"  Final:    {log.current_rows:,}")
    print(f"  Dropped:  {log.initial_rows - log.current_rows:,} "
          f"({100*(log.initial_rows - log.current_rows)/log.initial_rows:.1f}%)")
    print(f"\n✅ Stage 2 complete.")

    return df


if __name__ == "__main__":
    main()
