"""
Unit tests for Stage 5 Splits Protocol.
Verifies that no Area appears in both train and validation/test folds.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PARQUET = PROJECT_ROOT / "data" / "processed" / "clean.parquet"
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"


def test_site_held_out_zero_leakage():
    """Assertion: Absolutely zero site (Area) overlap between train, val, and test."""
    assert (SPLITS_DIR / "site_held_out_split.npz").exists(), "site_held_out_split.npz not found"
    
    splits = np.load(SPLITS_DIR / "site_held_out_split.npz")
    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]
    test_idx = splits["test_idx"]

    df = pd.read_parquet(CLEAN_PARQUET, columns=["Area"])

    train_sites = set(df.iloc[train_idx]["Area"])
    val_sites = set(df.iloc[val_idx]["Area"])
    test_sites = set(df.iloc[test_idx]["Area"])

    # Strict set intersection assertions
    overlap_train_val = train_sites.intersection(val_sites)
    overlap_train_test = train_sites.intersection(test_sites)
    overlap_val_test = val_sites.intersection(test_sites)

    assert len(overlap_train_val) == 0, f"Leakage: {len(overlap_train_val)} sites overlap between train and val"
    assert len(overlap_train_test) == 0, f"Leakage: {len(overlap_train_test)} sites overlap between train and test"
    assert len(overlap_val_test) == 0, f"Leakage: {len(overlap_val_test)} sites overlap between val and test"


def test_temporal_split_chronology():
    """Assertion: Temporal split strictly respects non-overlapping date boundaries."""
    assert (SPLITS_DIR / "time_split.npz").exists(), "time_split.npz not found"
    
    splits = np.load(SPLITS_DIR / "time_split.npz")
    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]
    test_idx = splits["test_idx"]

    df = pd.read_parquet(CLEAN_PARQUET, columns=["Date"])
    years = pd.to_datetime(df["Date"]).dt.year

    max_train_year = years.iloc[train_idx].max()
    min_val_year = years.iloc[val_idx].min()
    max_val_year = years.iloc[val_idx].max()
    min_test_year = years.iloc[test_idx].min()

    assert max_train_year <= 2012, f"Train year leakage: {max_train_year} > 2012"
    assert min_val_year >= 2013, f"Val year leakage: {min_val_year} < 2013"
    assert max_val_year <= 2017, f"Val year leakage: {max_val_year} > 2017"
    assert min_test_year >= 2018, f"Test year leakage: {min_test_year} < 2018"

