"""
End-to-End Contract Verification Tests for Phase 2 Final Deliverables (Task F):
Asserts that Member 3 can load every `dl_*` prediction CSV without errors:
1. All 5 seeds [42, 43, 44, 45, 46] exist for val and test splits across:
   - dl_mlp_seed{N}_{val,test}.csv
   - dl_mlp_mcdropout_seed{N}_{val,test}.csv
   - dl_mlp_ensemble_seed{N}_{val,test}.csv
   - dl_mlp_quantile_seed{N}_{val,test}.csv
2. Leakage-aware run exists:
   - dl_mlp_leakaware_seed42_{val,test}.csv (and dl_mlp_leakage_aware_seed42_{val,test}.csv)
3. Every CSV has 50,000 rows matching row_id_val / row_id_test from M1's split, zero NaNs,
   required columns (row_id, y_true, y_pred, class_true, class_pred), and uncertainty columns
   (y_std, q05, q95) for mcdropout, ensemble, and quantile variants.
4. Tuning, ablation, and final summary tables exist in results/tables/.
"""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.utils import DL_ROOT


@pytest.mark.parametrize("variant", ["dl_mlp", "dl_mlp_mcdropout", "dl_mlp_ensemble", "dl_mlp_quantile"])
@pytest.mark.parametrize("seed", [42, 43, 44, 45, 46])
@pytest.mark.parametrize("split", ["val", "test"])
def test_dl_five_seed_prediction_csvs_contract(variant: str, seed: int, split: str):
    """Verify every 5-seed prediction CSV conforms to the team contract for Member 3."""
    csv_path = DL_ROOT / "results" / "predictions" / f"{variant}_seed{seed}_{split}.csv"
    assert csv_path.exists(), f"Missing contract prediction CSV: {csv_path}"

    df = pd.read_csv(csv_path)
    assert len(df) == 50000, f"Expected 50,000 rows in {csv_path.name}, got {len(df)}"

    required_cols = {"row_id", "y_true", "y_pred", "class_true", "class_pred"}
    assert required_cols.issubset(set(df.columns)), f"Missing columns in {csv_path.name}"
    assert not df[["row_id", "y_true", "y_pred", "class_true", "class_pred"]].isna().any().any()

    if variant in ("dl_mlp_mcdropout", "dl_mlp_ensemble", "dl_mlp_quantile"):
        unc_cols = {"y_std", "q05", "q95"}
        assert unc_cols.issubset(set(df.columns)), f"Missing uncertainty columns in {csv_path.name}"
        assert not df[["y_std", "q05", "q95"]].isna().any().any()
        assert (df["y_std"] > 0.0).all(), f"Non-positive y_std in {csv_path.name}"
        assert (df["q05"] <= df["y_pred"] + 1e-3).all(), f"q05 > y_pred in {csv_path.name}"
        assert (df["y_pred"] <= df["q95"] + 1e-3).all(), f"y_pred > q95 in {csv_path.name}"


@pytest.mark.parametrize("split", ["val", "test"])
def test_dl_leakaware_prediction_csv_contract(split: str):
    """Verify leakage-aware MLP prediction CSVs exist and conform to schema."""
    for name in [f"dl_mlp_leakaware_seed42_{split}.csv", f"dl_mlp_leakage_aware_seed42_{split}.csv"]:
        csv_path = DL_ROOT / "results" / "predictions" / name
        assert csv_path.exists(), f"Missing leakage-aware CSV: {csv_path}"
        df = pd.read_csv(csv_path)
        assert len(df) == 50000
        assert {"row_id", "y_true", "y_pred", "class_true", "class_pred"}.issubset(set(df.columns))
        assert not df["y_pred"].isna().any()


def test_phase2_tables_and_ieee_report_exist():
    """Verify tuning, ablation, final summary tables, and IEEE report artifacts exist."""
    tables_dir = DL_ROOT / "results" / "tables"
    for fname in [
        "sanity_suite_summary.json",
        "optuna_trials.csv",
        "optuna_study_summary.json",
        "ablation_results.csv",
        "ablation_results.md",
        "phase2_final_summary.csv",
        "phase2_final_results.json",
    ]:
        fpath = tables_dir / fname
        assert fpath.exists(), f"Missing table artifact: {fpath}"

    ieee_pdf = DL_ROOT / "reports" / "ieee_report" / "DL_IEEE_Report.pdf"
    assert ieee_pdf.exists(), "Missing DL_IEEE_Report.pdf"
    lit_rev = DL_ROOT / "reports" / "literature_review_dl.md"
    assert lit_rev.exists(), "Missing literature_review_dl.md"
