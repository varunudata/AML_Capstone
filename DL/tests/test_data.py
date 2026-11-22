"""
Comprehensive Unit Tests for Task D (DataLoaders, Preprocessing & Leakage Contract).
Asserts:
1. No row index overlap and zero group/site overlap between splits.
2. Scaler statistics equal train statistics strictly (no validation/test data leakage).
3. Zero NaNs remain after preprocessing in train, val, and test splits.
4. Deterministic output for fixed random seed.
5. Mock data contract keys match real data contract keys 100%.
6. PyTorch DataLoaders produce valid torch.Tensor batches with matching shapes and dtypes.
7. Leakage-aware ablation mode correctly drops specified features.
8. run_leakage_audit returns quantitative linear model baselines.
"""

import pytest
import numpy as np
import pandas as pd
import torch

from src.utils import load_config, seed_everything
from src.data import (
    load_splits,
    get_dataloaders,
    run_leakage_audit,
    FEATURE_NAMES_21
)


def test_no_row_and_group_overlap():
    """Assert zero row overlap and zero monitoring station overlap across all folds."""
    cfg = load_config()
    splits = load_splits(cfg, scaled=True, mock=False, subsample=True)

    r_train = set(splits["row_id_train"])
    r_val = set(splits["row_id_val"])
    r_test = set(splits["row_id_test"])

    # Row index disjointness
    assert len(r_train & r_val) == 0, "Train and Val share row indices!"
    assert len(r_train & r_test) == 0, "Train and Test share row indices!"
    assert len(r_val & r_test) == 0, "Val and Test share row indices!"

    # Spatial site disjointness
    meta_test = splits["meta_test"]
    assert "Area" in meta_test.columns

    # Load dataset to inspect train/val sites
    from src.data import load_dataset
    df = load_dataset(cfg)
    train_sites = set(df.iloc[splits["row_id_train"]]["Area"])
    val_sites = set(df.iloc[splits["row_id_val"]]["Area"])
    test_sites = set(meta_test["Area"])

    assert len(train_sites & val_sites) == 0, "Spatial site leakage between Train and Val!"
    assert len(train_sites & test_sites) == 0, "Spatial site leakage between Train and Test!"
    assert len(val_sites & test_sites) == 0, "Spatial site leakage between Val and Test!"


def test_scaler_statistics_equal_train_statistics():
    """Verify standard scaler statistics are derived strictly from the training split."""
    cfg = load_config()
    from src.data import get_or_fit_preprocessor
    preprocessor = get_or_fit_preprocessor(cfg)

    # Get unscaled training data
    splits_unscaled = load_splits(cfg, scaled=False, mock=False, subsample=True)
    X_train_u = splits_unscaled["X_train"]

    expected_mean = np.mean(X_train_u, axis=0)
    expected_std = np.std(X_train_u, axis=0)

    np.testing.assert_allclose(
        preprocessor.train_means_, expected_mean, atol=1e-2,
        err_msg="Preprocessor mean does not match train fold mean"
    )
    np.testing.assert_allclose(
        preprocessor.train_stds_, expected_std, atol=1e-2,
        err_msg="Preprocessor scale does not match train fold std"
    )


def test_no_nans_after_preprocessing():
    """Verify that both scaled and unscaled matrices have zero NaNs across all splits."""
    cfg = load_config()
    for scaled in [True, False]:
        splits = load_splits(cfg, scaled=scaled, mock=False, subsample=True)
        assert not np.isnan(splits["X_train"]).any(), f"NaN found in X_train (scaled={scaled})"
        assert not np.isnan(splits["X_val"]).any(), f"NaN found in X_val (scaled={scaled})"
        assert not np.isnan(splits["X_test"]).any(), f"NaN found in X_test (scaled={scaled})"


def test_deterministic_output_for_fixed_seed():
    """Verify that repeated calls with same seed produce identical matrices."""
    cfg = load_config()
    seed_everything(42)
    s1 = load_splits(cfg, scaled=True, mock=False, subsample=True)

    seed_everything(42)
    s2 = load_splits(cfg, scaled=True, mock=False, subsample=True)

    np.testing.assert_array_equal(s1["X_train"], s2["X_train"])
    np.testing.assert_array_equal(s1["y_train"], s2["y_train"])
    np.testing.assert_array_equal(s1["row_id_train"], s2["row_id_train"])


def test_mock_keys_match_real_keys():
    """Verify mock=True keys and real keys match 100%."""
    cfg = load_config()
    mock_data = load_splits(cfg, scaled=True, mock=True)
    real_data = load_splits(cfg, scaled=True, mock=False, subsample=True)

    assert set(mock_data.keys()) == set(real_data.keys()), "Mock keys and real keys differ!"


def test_dataloaders_batches_and_types():
    """Verify PyTorch DataLoaders yield valid float32 torch Tensors with expected shapes."""
    cfg = load_config()
    train_loader, val_loader, test_loader = get_dataloaders(cfg, batch_size=128, mock=False)

    for loader_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        batch_x, batch_y = next(iter(loader))
        assert isinstance(batch_x, torch.Tensor), f"{loader_name} X is not torch.Tensor"
        assert isinstance(batch_y, torch.Tensor), f"{loader_name} y is not torch.Tensor"
        assert batch_x.dtype == torch.float32
        assert batch_y.dtype == torch.float32
        assert batch_x.shape[0] == 128
        assert batch_x.shape[1] == len(FEATURE_NAMES_21)
        assert not torch.isnan(batch_x).any()
        assert not torch.isnan(batch_y).any()


def test_leakage_aware_feature_dropping():
    """Verify leakage_aware flag and drop_features list removes requested features."""
    cfg = load_config()
    cfg_copy = dict(cfg)
    cfg_copy["data"] = dict(cfg["data"])
    cfg_copy["data"]["leakage_aware"] = True
    cfg_copy["data"]["drop_features"] = ["Orthophosphate (mg/l)", "log_np_ratio"]

    splits = load_splits(cfg_copy, scaled=True, mock=False, subsample=True)
    assert "Orthophosphate (mg/l)" not in splits["feature_names"]
    assert "log_np_ratio" not in splits["feature_names"]
    assert splits["X_train"].shape[1] == len(FEATURE_NAMES_21) - 2
    assert splits["X_test"].shape[1] == len(FEATURE_NAMES_21) - 2


def test_leakage_audit_function():
    """Verify run_leakage_audit executes and returns valid metrics."""
    cfg = load_config()
    audit = run_leakage_audit(cfg)

    assert "raw_8_chemistry" in audit
    assert "all_21_engineered" in audit
    assert "finding" in audit

    r2_raw = audit["raw_8_chemistry"]["R2"]
    r2_all = audit["all_21_engineered"]["R2"]

    assert 0.40 <= r2_raw <= 0.60, f"Unexpected R2 for raw chemistry: {r2_raw}"
    assert 0.85 <= r2_all <= 0.98, f"Unexpected R2 for all features: {r2_all}"
    assert r2_all > r2_raw, "Engineered features should explain significantly more variance"
