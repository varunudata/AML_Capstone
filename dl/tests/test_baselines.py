"""
Unit tests for Task E: AML Baselines Bridge & Contracts.
Asserts:
1. results/aml_baseline_metrics.json matches METRICS CONTRACT schema and values.
2. Prediction CSVs exist at results/predictions/{family}_{variant}_seed{N}_{split}.csv per PREDICTIONS CONTRACT.
3. Prediction CSVs have valid columns: row_id, y_true, y_pred, class_true, class_pred.
4. No NaNs in any exported prediction CSVs.
5. Recomputed metrics from saved CSVs match aml_baseline_metrics.json.
"""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from src.utils import load_config, resolve_path

DL_ROOT = Path(__file__).resolve().parent.parent


def test_aml_baseline_metrics_json_contract():
    """Verify metrics JSON schema and reasonable environmental metrics."""
    metrics_path = DL_ROOT / "results" / "aml_baseline_metrics.json"
    assert metrics_path.exists(), f"Missing {metrics_path}"

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Required top-level keys per contract
    assert "xgboost" in data
    assert "random_forest" in data
    assert "meta" in data

    # Sub-keys
    for model_name in ["xgboost", "random_forest"]:
        for split in ["val", "test"]:
            assert split in data[model_name]
            m = data[model_name][split]
            assert "MAE" in m
            assert "RMSE" in m
            assert "R2" in m
            assert m["R2"] > 0.99, f"{model_name} {split} R2 unexpectedly low: {m['R2']}"
            assert m["RMSE"] < 0.60, f"{model_name} {split} RMSE unexpectedly high: {m['RMSE']}"

    # Meta contract
    meta = data["meta"]
    assert meta["target"] == "CCME_Values"
    assert meta["split_type"] == "site_held_out (Area)"
    assert meta["n_val"] == 50000
    assert meta["n_test"] == 50000


def test_predictions_csv_contract():
    """Verify all 6 exported prediction CSVs conform to schema and have zero NaNs."""
    pred_dir = DL_ROOT / "results" / "predictions"
    assert pred_dir.exists(), f"Missing {pred_dir}"

    expected_files = [
        "aml_xgboost_seed42_val.csv",
        "aml_xgboost_seed42_test.csv",
        "aml_random_forest_seed42_val.csv",
        "aml_random_forest_seed42_test.csv",
        "aml_xgboost_leakage_aware_seed42_val.csv",
        "aml_xgboost_leakage_aware_seed42_test.csv"
    ]

    expected_cols = {"row_id", "y_true", "y_pred", "class_true", "class_pred"}

    for fname in expected_files:
        fpath = pred_dir / fname
        assert fpath.exists(), f"Missing expected prediction CSV: {fname}"

        df = pd.read_csv(fpath)
        assert expected_cols.issubset(set(df.columns)), f"Columns missing in {fname}: {expected_cols - set(df.columns)}"
        assert len(df) == 50000, f"Expected 50,000 rows in {fname}, got {len(df)}"

        # Check no NaNs
        assert not df["y_true"].isna().any(), f"NaN in y_true for {fname}"
        assert not df["y_pred"].isna().any(), f"NaN in y_pred for {fname}"
        assert not df["class_true"].isna().any(), f"NaN in class_true for {fname}"
        assert not df["class_pred"].isna().any(), f"NaN in class_pred for {fname}"


def test_metrics_match_csv_predictions():
    """Verify metrics recorded in JSON strictly match calculations from the CSV files."""
    metrics_path = DL_ROOT / "results" / "aml_baseline_metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    pred_dir = DL_ROOT / "results" / "predictions"

    for model, variant in [("xgboost", "aml_xgboost"), ("random_forest", "aml_random_forest")]:
        for split in ["val", "test"]:
            fpath = pred_dir / f"{variant}_seed42_{split}.csv"
            df = pd.read_csv(fpath)

            calc_rmse = round(float(np.sqrt(mean_squared_error(df["y_true"], df["y_pred"]))), 4)
            calc_mae = round(float(mean_absolute_error(df["y_true"], df["y_pred"])), 4)
            calc_r2 = round(float(r2_score(df["y_true"], df["y_pred"])), 4)

            recorded = metrics_data[model][split]
            assert abs(calc_rmse - recorded["RMSE"]) < 1e-3, f"RMSE mismatch for {model} {split}"
            assert abs(calc_mae - recorded["MAE"]) < 1e-3, f"MAE mismatch for {model} {split}"
            assert abs(calc_r2 - recorded["R2"]) < 1e-3, f"R2 mismatch for {model} {split}"
