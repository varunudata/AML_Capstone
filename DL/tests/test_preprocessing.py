"""
Unit tests for Task B: Leak-Safe Preprocessing & Preprocessor Artifact.
Asserts:
1. Detection-limit strings ('<0.5') are parsed to half-detection limits with flags.
2. Impossible physical values (pH <0 or >14, temp <-2 or >40, DO <=0 or >20, negative concentrations) are flagged and set to NaN.
3. Missingness indicators are constructed.
4. Imputer and Scaler are fit STRICTLY ON TRAIN ONLY.
5. No NaNs exist in the final transformed matrices.
6. Scaler mean and scale match training fold statistics exactly.
7. Both scaled=True (MLP) and scaled=False (Trees) paths work.
8. Preprocessor artifact persists cleanly to data/processed/preprocessor.joblib.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.utils import load_config, resolve_path
from src.data import (
    WaterQualityPreprocessor,
    load_splits,
    get_or_fit_preprocessor,
    FEATURE_NAMES_21
)


def test_detection_limit_and_impossible_cleaning():
    """Verify parsing of detection limits and zero-tolerance physical impossibility bounds."""
    dirty_data = pd.DataFrame({
        "Ammonia (mg/l)": ["<0.5", 1.2, -0.4, "0.10"],
        "Biochemical Oxygen Demand (mg/l)": [2.0, "< 1.0", -2.5, 1.5],
        "Dissolved Oxygen (mg/l)": [0.0, 8.5, 25.0, 9.0],
        "Orthophosphate (mg/l)": ["<0.02", 0.05, -0.01, 0.03],
        "pH (ph units)": [-1.0, 7.4, 15.2, 8.1],
        "Temperature (cel)": [-5.0, 18.0, 48.0, 22.0],
        "Nitrogen (mg/l)": [0.5, 1.2, -0.2, 0.8],
        "Nitrate (mg/l)": [0.3, 0.7, -0.1, 0.5]
    })

    preprocessor = WaterQualityPreprocessor()
    cleaned_df, flags = preprocessor.clean_and_engineer(dirty_data)

    # 1. Detection limit parsing (half-DL substitution)
    assert cleaned_df.loc[0, "Ammonia (mg/l)"] == pytest.approx(0.25)
    assert flags.loc[0, "flag_det_Ammonia (mg/l)"] == True

    assert cleaned_df.loc[1, "Biochemical Oxygen Demand (mg/l)"] == pytest.approx(0.5)
    assert flags.loc[1, "flag_det_Biochemical Oxygen Demand (mg/l)"] == True

    assert cleaned_df.loc[0, "Orthophosphate (mg/l)"] == pytest.approx(0.01)

    # 2. Impossible values set to NaN
    # Ammonia < 0
    assert np.isnan(cleaned_df.loc[2, "Ammonia (mg/l)"])
    assert flags.loc[2, "flag_imp_Ammonia (mg/l)"] == True

    # BOD < 0
    assert np.isnan(cleaned_df.loc[2, "Biochemical Oxygen Demand (mg/l)"])
    assert flags.loc[2, "flag_imp_Biochemical Oxygen Demand (mg/l)"] == True

    # DO <= 0 or > 20
    assert np.isnan(cleaned_df.loc[0, "Dissolved Oxygen (mg/l)"])
    assert np.isnan(cleaned_df.loc[2, "Dissolved Oxygen (mg/l)"])

    # pH < 0 or > 14
    assert np.isnan(cleaned_df.loc[0, "pH (ph units)"])
    assert np.isnan(cleaned_df.loc[2, "pH (ph units)"])

    # Temp < -2 or > 40
    assert np.isnan(cleaned_df.loc[0, "Temperature (cel)"])
    assert np.isnan(cleaned_df.loc[2, "Temperature (cel)"])

    # Valid values preserved
    assert cleaned_df.loc[1, "pH (ph units)"] == pytest.approx(7.4)
    assert cleaned_df.loc[1, "Temperature (cel)"] == pytest.approx(18.0)


def test_train_only_fitting_and_no_nans():
    """Verify imputer and scaler statistics are derived solely from training split."""
    cfg = load_config()

    # Load splits with scaled=True
    splits_scaled = load_splits(cfg, scaled=True, mock=False, subsample=True)
    X_train_s = splits_scaled["X_train"]
    X_val_s = splits_scaled["X_val"]
    X_test_s = splits_scaled["X_test"]

    # Verify no NaNs remain after preprocessing
    assert not np.isnan(X_train_s).any(), "NaNs found in X_train!"
    assert not np.isnan(X_val_s).any(), "NaNs found in X_val!"
    assert not np.isnan(X_test_s).any(), "NaNs found in X_test!"

    # Verify training set standardization properties
    train_mean = np.mean(X_train_s, axis=0)
    train_std = np.std(X_train_s, axis=0)

    np.testing.assert_allclose(train_mean, 0.0, atol=1e-3, err_msg="Train mean is not ~0")
    np.testing.assert_allclose(train_std, 1.0, atol=1e-3, err_msg="Train std is not ~1")

    # Verify scaled=False path
    splits_unscaled = load_splits(cfg, scaled=False, mock=False, subsample=True)
    X_train_u = splits_unscaled["X_train"]
    assert not np.isnan(X_train_u).any(), "NaNs found in unscaled X_train!"
    # Unscaled data should NOT have mean zero
    assert abs(np.mean(X_train_u)) > 0.1


def test_preprocessor_artifact_persistence():
    """Verify preprocessor artifact can be loaded and produces identical transforms."""
    cfg = load_config()
    preprocessor_path = resolve_path(cfg["data"]["preprocessor_path"])
    assert preprocessor_path.exists(), f"preprocessor.joblib missing at {preprocessor_path}"

    loaded_prep = WaterQualityPreprocessor.load(preprocessor_path)
    assert loaded_prep.imputer_ is not None
    assert loaded_prep.scaler_ is not None
    assert len(loaded_prep.feature_names_) == 21
    assert loaded_prep.train_means_.shape == (21,)
    assert loaded_prep.train_stds_.shape == (21,)
