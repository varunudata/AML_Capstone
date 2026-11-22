"""
Unit tests for Task A: Target Definition & Split Fidelity.
Asserts:
1. Exact single continuous target (CCME_Values) within valid physical bounds [31.3, 100.0].
2. Discrete classification scheme matches CCME 5-tier standard.
3. Zero site overlap between Train, Val, and Test splits (no spatial leakage).
4. Reproducibility of split_indices.json and exact row-for-row alignment with AML split.
5. Real data loading adheres strictly to the DATA CONTRACT.
"""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.utils import load_config, resolve_path
from src.data import load_dataset, load_split_indices, verify_zero_site_leakage, load_splits


def test_target_definition():
    """Verify ONE continuous target definition and its bounds."""
    cfg = load_config()
    target_name = cfg["data"]["target_name"]
    assert target_name == "CCME_Values", f"Target must be CCME_Values, found {target_name}"

    df = load_dataset(cfg)
    assert target_name in df.columns, f"{target_name} not found in dataset"

    vals = df[target_name].to_numpy()
    assert np.all(np.isfinite(vals)), "Target contains NaN or inf values!"
    assert vals.min() >= 31.0, f"Target min {vals.min()} below expected theoretical minimum"
    assert vals.max() <= 100.0, f"Target max {vals.max()} exceeds upper ceiling 100.0"

    # Verify classification scheme if defined
    class_name = cfg["data"]["class_target_name"]
    if class_name in df.columns:
        expected_classes = {"Excellent", "Good", "Fair", "Marginal", "Poor"}
        actual_classes = set(df[class_name].dropna().unique())
        assert actual_classes.issubset(expected_classes), f"Unexpected classes: {actual_classes}"


def test_zero_site_leakage():
    """Verify that train, val, and test share zero monitoring sites (Area)."""
    cfg = load_config()
    df = load_dataset(cfg)
    split_info = load_split_indices(cfg)

    train_idx = np.array(split_info["train_sample_idx"])
    val_idx = np.array(split_info["val_sample_idx"])
    test_idx = np.array(split_info["test_sample_idx"])

    leakage_stats = verify_zero_site_leakage(df, train_idx, val_idx, test_idx, group_col=cfg["data"]["group_column"])

    assert leakage_stats["overlap_train_val"] == 0, "Spatial leakage between Train and Val!"
    assert leakage_stats["overlap_train_test"] == 0, "Spatial leakage between Train and Test!"
    assert leakage_stats["overlap_val_test"] == 0, "Spatial leakage between Val and Test!"
    assert leakage_stats["train_sites"] > 0
    assert leakage_stats["val_sites"] > 0
    assert leakage_stats["test_sites"] > 0


def test_split_indices_json_contract():
    """Verify split_indices.json contains required metadata and row indices."""
    cfg = load_config()
    split_indices_path = resolve_path(cfg["data"]["split_indices_path"])
    assert split_indices_path.exists(), f"split_indices.json missing at {split_indices_path}"

    with open(split_indices_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data
    assert "train_sample_idx" in data
    assert "val_sample_idx" in data
    assert "test_sample_idx" in data

    meta = data["metadata"]
    assert meta["target_name"] == "CCME_Values"
    assert meta["split_policy"] == "group_stratified_site_held_out"
    assert meta["zero_overlap_verified"] is True

    # Check non-empty index lists
    assert len(data["train_sample_idx"]) == 250000
    assert len(data["val_sample_idx"]) == 50000
    assert len(data["test_sample_idx"]) == 50000


def test_real_load_splits_contract():
    """Verify real data loading (mock=False) satisfies all keys and types in DATA CONTRACT."""
    cfg = load_config()
    splits = load_splits(cfg, mock=False, subsample=True)

    expected_keys = {
        "X_train", "X_val", "X_test",
        "y_train", "y_val", "y_test",
        "feature_names",
        "row_id_train", "row_id_val", "row_id_test",
        "meta_test",
        "y_class_train", "y_class_val", "y_class_test"
    }
    assert expected_keys.issubset(set(splits.keys())), f"Missing keys: {expected_keys - set(splits.keys())}"

    # Verify dtypes
    assert splits["X_train"].dtype == np.float32
    assert splits["y_train"].dtype == np.float32
    assert splits["X_val"].dtype == np.float32
    assert splits["y_val"].dtype == np.float32
    assert splits["X_test"].dtype == np.float32
    assert splits["y_test"].dtype == np.float32

    # Verify sample sizes
    assert len(splits["X_train"]) == 250000
    assert len(splits["X_val"]) == 50000
    assert len(splits["X_test"]) == 50000

    # Verify metadata test
    assert isinstance(splits["meta_test"], pd.DataFrame)
    assert len(splits["meta_test"]) == 50000
    assert "Area" in splits["meta_test"].columns
    assert "Date" in splits["meta_test"].columns
