"""
Data pipeline, preprocessing & DataLoaders for DL phase (Member 1).
Implements the DATA CONTRACT:
  load_splits(cfg, scaled=True, mock=False) -> dict
  get_dataloaders(cfg, batch_size) -> train_loader, val_loader, test_loader
  WaterQualityPreprocessor -> leak-safe cleaning, imputation, and scaling.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Set, Union
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .utils import get_logger, seed_everything, resolve_path

logger = get_logger("dl.data")

BASE_CHEMISTRY = [
    "Ammonia (mg/l)",
    "Biochemical Oxygen Demand (mg/l)",
    "Dissolved Oxygen (mg/l)",
    "Orthophosphate (mg/l)",
    "pH (ph units)",
    "Temperature (cel)",
    "Nitrogen (mg/l)",
    "Nitrate (mg/l)"
]

FEATURE_NAMES_21 = [
    "Ammonia (mg/l)",
    "Biochemical Oxygen Demand (mg/l)",
    "Dissolved Oxygen (mg/l)",
    "Orthophosphate (mg/l)",
    "pH (ph units)",
    "Temperature (cel)",
    "Nitrogen (mg/l)",
    "Nitrate (mg/l)",
    "log1p_ammonia",
    "log1p_biochemical",
    "log1p_orthophosphate",
    "log1p_nitrogen",
    "log1p_nitrate",
    "do_sat_ratio",
    "do_deficit",
    "log_np_ratio",
    "nitrate_to_n_ratio",
    "ph_dev_neutral",
    "ph_dist_buffer",
    "month_sin",
    "month_cos"
]


class WaterQualityPreprocessor(BaseEstimator, TransformerMixin):
    """
    Leak-safe preprocessing transformer for routine surface water chemistry:
    1. Detection-limit parsing (e.g. '<0.5' -> half-detection-limit 0.25 + indicator flag).
    2. Impossible physical value flagging & setting to NaN (pH outside [0,14], temp outside [-2,40], DO <= 0 or > 20).
    3. Missingness indicator columns for downstream modeling.
    4. Heavy-skew mitigation via log1p transforms.
    5. Stoichiometric and saturation domain interactions (DO deficit, N:P ratio, pH buffering).
    6. Median imputation FIT STRICTLY ON TRAIN SPLIT ONLY.
    7. StandardScaler FIT STRICTLY ON TRAIN SPLIT ONLY.
    """

    def __init__(self, add_missing_indicators: bool = False):
        self.add_missing_indicators = add_missing_indicators
        self.feature_names_ = FEATURE_NAMES_21.copy()
        self.imputer_: Optional[SimpleImputer] = None
        self.scaler_: Optional[StandardScaler] = None
        self.train_means_: Optional[np.ndarray] = None
        self.train_stds_: Optional[np.ndarray] = None
        self.train_medians_: Optional[np.ndarray] = None

    @staticmethod
    def clean_raw_series(series: pd.Series, col_name: str) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Parse detection-limit strings and enforce physical bounds.
        Returns: (cleaned_numeric_series, detection_limit_flag, impossible_value_flag)
        """
        # Detection limit flag
        det_flag = series.astype(str).str.contains("<", na=False)

        # Parse string expressions like '<0.5' into half detection limit
        def _parse_val(val: Any) -> float:
            if pd.isna(val):
                return np.nan
            if isinstance(val, (int, float)):
                return float(val)
            s = str(val).strip()
            if s.startswith("<"):
                try:
                    num = float(re.sub(r"[^\d.]", "", s))
                    return num / 2.0  # Analytical chemistry standard half-DL substitution
                except ValueError:
                    return np.nan
            try:
                return float(s)
            except ValueError:
                return np.nan

        num_series = series.apply(_parse_val).astype(np.float32)

        # Physical impossibility bounds
        imp_flag = pd.Series(False, index=series.index)

        if "pH" in col_name:
            imp_mask = (num_series < 0.0) | (num_series > 14.0)
            imp_flag = imp_mask
            num_series[imp_mask] = np.nan
        elif "Temperature" in col_name:
            imp_mask = (num_series < -2.0) | (num_series > 40.0)
            imp_flag = imp_mask
            num_series[imp_mask] = np.nan
        elif "Dissolved Oxygen" in col_name:
            imp_mask = (num_series <= 0.0) | (num_series > 20.0)
            imp_flag = imp_mask
            num_series[imp_mask] = np.nan
        else:
            # Nutrient concentrations cannot be physically negative
            imp_mask = num_series < 0.0
            imp_flag = imp_mask
            num_series[imp_mask] = np.nan

        return num_series, det_flag, imp_flag

    def clean_and_engineer(self, df_in: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Clean base chemistry, construct missing indicators, and engineer limnological features.
        """
        out = pd.DataFrame(index=df_in.index)
        flags = pd.DataFrame(index=df_in.index)

        # 1. Base chemistry with string/bound cleaning
        for col in BASE_CHEMISTRY:
            if col in df_in.columns:
                cleaned_col, det_f, imp_f = self.clean_raw_series(df_in[col], col)
                out[col] = cleaned_col
                flags[f"flag_det_{col}"] = det_f
                flags[f"flag_imp_{col}"] = imp_f
                flags[f"flag_missing_{col}"] = cleaned_col.isna()
            else:
                out[col] = np.nan
                flags[f"flag_missing_{col}"] = True

        # 2. Skewed features log1p
        out["log1p_ammonia"] = np.log1p(np.maximum(0.0, out["Ammonia (mg/l)"]))
        out["log1p_biochemical"] = np.log1p(np.maximum(0.0, out["Biochemical Oxygen Demand (mg/l)"]))
        out["log1p_orthophosphate"] = np.log1p(np.maximum(0.0, out["Orthophosphate (mg/l)"]))
        out["log1p_nitrogen"] = np.log1p(np.maximum(0.0, out["Nitrogen (mg/l)"]))
        out["log1p_nitrate"] = np.log1p(np.maximum(0.0, out["Nitrate (mg/l)"]))

        # 3. Limnological interactions & ratios
        t = out["Temperature (cel)"]
        do = out["Dissolved Oxygen (mg/l)"]
        do_sat = 468.0 / (31.6 + t)
        out["do_sat_ratio"] = do / (do_sat + 1e-4)
        out["do_deficit"] = do_sat - do

        out["log_np_ratio"] = np.log1p(
            np.maximum(0.0, out["Nitrogen (mg/l)"] / (out["Orthophosphate (mg/l)"] + 1e-4))
        )
        out["nitrate_to_n_ratio"] = np.clip(
            out["Nitrate (mg/l)"] / (out["Nitrogen (mg/l)"] + 1e-4),
            0.0,
            1.0
        )

        ph = out["pH (ph units)"]
        out["ph_dev_neutral"] = np.abs(ph - 7.0)
        out["ph_dist_buffer"] = np.maximum(0.0, 6.5 - ph) + np.maximum(0.0, ph - 8.5)

        # 4. Cyclical temporal encoding
        if "Date" in df_in.columns:
            dates = pd.to_datetime(df_in["Date"], errors="coerce")
            months = dates.dt.month.fillna(6).to_numpy()
            out["month_sin"] = np.sin(2.0 * np.pi * months / 12.0).astype(np.float32)
            out["month_cos"] = np.cos(2.0 * np.pi * months / 12.0).astype(np.float32)
        else:
            out["month_sin"] = np.float32(0.0)
            out["month_cos"] = np.float32(0.0)

        return out[FEATURE_NAMES_21], flags

    def fit(self, df_train: pd.DataFrame, y=None) -> "WaterQualityPreprocessor":
        """
        Fit imputer and scaler STRICTLY on training split.
        WHY: Prevents information leakage from validation and test sets into preprocessing statistics.
        """
        logger.info("Fitting WaterQualityPreprocessor strictly on training fold (N=%d)...", len(df_train))
        X_feats, _ = self.clean_and_engineer(df_train)

        # Fit median imputer
        self.imputer_ = SimpleImputer(strategy="median")
        X_imputed = self.imputer_.fit_transform(X_feats)
        self.train_medians_ = self.imputer_.statistics_.astype(np.float32)

        # Fit standard scaler
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X_imputed)
        self.train_means_ = self.scaler_.mean_.astype(np.float32)
        self.train_stds_ = self.scaler_.scale_.astype(np.float32)

        return self

    def transform(self, df: pd.DataFrame, scaled: bool = True) -> np.ndarray:
        """
        Transform dataset using frozen training parameters.
        - scaled=True: Imputed + StandardScaled (for MLP models).
        - scaled=False: Imputed but unscaled (for Tree models: XGBoost, Random Forest).
        """
        if self.imputer_ is None or self.scaler_ is None:
            raise RuntimeError("WaterQualityPreprocessor must be fit before calling transform.")

        X_feats, _ = self.clean_and_engineer(df)
        X_imputed = self.imputer_.transform(X_feats)

        if scaled:
            X_out = self.scaler_.transform(X_imputed)
        else:
            X_out = X_imputed

        return X_out.astype(np.float32)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save fitted preprocessor artifact to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved preprocessor artifact to: %s", path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "WaterQualityPreprocessor":
        """Load fitted preprocessor artifact from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at: {path}")
        obj = joblib.load(path)
        logger.info("Loaded preprocessor artifact from: %s", path)
        return obj


def load_dataset(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Load the cleaned, frozen parquet dataset.
    WHY: Reuses exact AML scope decisions (ambient waterbodies, year >= 1990, impossible values flagged).
    """
    parquet_path = resolve_path(cfg["data"]["clean_parquet_path"])
    if not parquet_path.exists():
        raise FileNotFoundError(f"Clean dataset not found at {parquet_path}")
    df = pd.read_parquet(parquet_path)
    return df


def load_split_indices(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load precomputed split indices ensuring exact row-for-row AML split reproduction.
    WHY: Preserves zero spatial leakage (0% station overlap between Train, Val, Test).
    """
    split_indices_path = resolve_path(cfg["data"]["split_indices_path"])
    if split_indices_path.exists():
        with open(split_indices_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Fallback: load directly from AML split npz
    aml_split_path = resolve_path(cfg["data"]["aml_split_path"])
    if aml_split_path.exists():
        npz = np.load(aml_split_path)
        return {
            "metadata": {"source": "site_held_out_split.npz"},
            "train_sample_idx": npz["train_idx"].tolist(),
            "val_sample_idx": npz["val_idx"].tolist(),
            "test_sample_idx": npz["test_idx"].tolist(),
        }

    raise FileNotFoundError(f"Neither {split_indices_path} nor {aml_split_path} could be located.")


def verify_zero_site_leakage(df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, group_col: str = "Area") -> Dict[str, int]:
    """Assert zero site overlap between train, val, and test splits."""
    train_sites = set(df.iloc[train_idx][group_col])
    val_sites = set(df.iloc[val_idx][group_col])
    test_sites = set(df.iloc[test_idx][group_col])

    overlap_train_val = len(train_sites & val_sites)
    overlap_train_test = len(train_sites & test_sites)
    overlap_val_test = len(val_sites & test_sites)

    if overlap_train_val > 0 or overlap_train_test > 0 or overlap_val_test > 0:
        raise ValueError(
            f"Spatial leakage detected! Overlaps: train-val={overlap_train_val}, "
            f"train-test={overlap_train_test}, val-test={overlap_val_test}"
        )

    return {
        "train_sites": len(train_sites),
        "val_sites": len(val_sites),
        "test_sites": len(test_sites),
        "overlap_train_val": overlap_train_val,
        "overlap_train_test": overlap_train_test,
        "overlap_val_test": overlap_val_test
    }


def get_or_fit_preprocessor(cfg: Dict[str, Any], df_train: Optional[pd.DataFrame] = None, force_refit: bool = False) -> WaterQualityPreprocessor:
    """
    Get existing preprocessor artifact or fit a new one on df_train.
    WHY: Imputer and Scaler are fit ONCE on train fold and persisted to data/processed/preprocessor.joblib.
    """
    preprocessor_path = resolve_path(cfg["data"]["preprocessor_path"])

    if preprocessor_path.exists() and not force_refit:
        return WaterQualityPreprocessor.load(preprocessor_path)

    if df_train is None:
        raise ValueError("preprocessor.joblib does not exist and df_train was not provided to fit it.")

    preprocessor = WaterQualityPreprocessor()
    preprocessor.fit(df_train)
    preprocessor.save(preprocessor_path)
    return preprocessor


def _generate_mock_splits(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Generate deterministic synthetic data adhering to the EXACT DATA CONTRACT."""
    seed = cfg.get("data", {}).get("split_seed", 42)
    rng = np.random.RandomState(seed)

    feature_names = FEATURE_NAMES_21
    n_features = len(feature_names)

    n_train = 128
    n_val = 32
    n_test = 32

    X_train = rng.randn(n_train, n_features).astype(np.float32)
    X_val = rng.randn(n_val, n_features).astype(np.float32)
    X_test = rng.randn(n_test, n_features).astype(np.float32)

    y_train = np.clip(rng.normal(loc=85.0, scale=12.0, size=n_train), 31.3, 100.0).astype(np.float32)
    y_val = np.clip(rng.normal(loc=85.0, scale=12.0, size=n_val), 31.3, 100.0).astype(np.float32)
    y_test = np.clip(rng.normal(loc=85.0, scale=12.0, size=n_test), 31.3, 100.0).astype(np.float32)

    row_id_train = np.arange(0, n_train, dtype=np.int64)
    row_id_val = np.arange(n_train, n_train + n_val, dtype=np.int64)
    row_id_test = np.arange(n_train + n_val, n_train + n_val + n_test, dtype=np.int64)

    def to_class(vals: np.ndarray) -> np.ndarray:
        classes = []
        for v in vals:
            if v >= 95.0:
                classes.append("Excellent")
            elif v >= 80.0:
                classes.append("Good")
            elif v >= 65.0:
                classes.append("Fair")
            elif v >= 45.0:
                classes.append("Marginal")
            else:
                classes.append("Poor")
        return np.array(classes, dtype=object)

    y_class_train = to_class(y_train)
    y_class_val = to_class(y_val)
    y_class_test = to_class(y_test)

    meta_test = pd.DataFrame({
        "row_id": row_id_test,
        "Country": ["Mockland"] * n_test,
        "Area": [f"Site_{i % 5}" for i in range(n_test)],
        "Waterbody Type": ["River"] * n_test,
        "Date": pd.date_range("2020-01-01", periods=n_test, freq="W")
    })

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names": list(feature_names),
        "row_id_train": row_id_train,
        "row_id_val": row_id_val,
        "row_id_test": row_id_test,
        "meta_test": meta_test,
        "y_class_train": y_class_train,
        "y_class_val": y_class_val,
        "y_class_test": y_class_test
    }


def load_splits(
    cfg: Dict[str, Any],
    scaled: bool = True,
    mock: bool = False,
    subsample: bool = True
) -> Dict[str, Any]:
    """
    DATA CONTRACT implementation:
    Returns dict with keys:
      - X_train, X_val, X_test: float32 arrays
      - y_train, y_val, y_test: float32 arrays
      - feature_names: list of feature column names
      - row_id_train, row_id_val, row_id_test: int64 arrays
      - meta_test: pd.DataFrame with site/time columns
      - y_class_train, y_class_val, y_class_test: class labels
    """
    if mock:
        return _generate_mock_splits(cfg)

    # 1. Load dataset & split indices
    df = load_dataset(cfg)
    split_info = load_split_indices(cfg)

    target_col = cfg.get("data", {}).get("target_name", "CCME_Values")
    class_col = cfg.get("data", {}).get("class_target_name", "CCME_WQI")
    group_col = cfg.get("data", {}).get("group_column", "Area")

    if subsample and "train_sample_idx" in split_info:
        train_idx = np.array(split_info["train_sample_idx"], dtype=np.int64)
        val_idx = np.array(split_info["val_sample_idx"], dtype=np.int64)
        test_idx = np.array(split_info["test_sample_idx"], dtype=np.int64)
    else:
        aml_split_path = resolve_path(cfg["data"]["aml_split_path"])
        npz = np.load(aml_split_path)
        train_idx = npz["train_idx"].astype(np.int64)
        val_idx = npz["val_idx"].astype(np.int64)
        test_idx = npz["test_idx"].astype(np.int64)

    # 2. Verify zero site leakage
    verify_zero_site_leakage(df, train_idx, val_idx, test_idx, group_col=group_col)

    # 3. Subset folds
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    df_test = df.iloc[test_idx]

    # 4. Fit or load preprocessor (fitted strictly on train fold!)
    preprocessor = get_or_fit_preprocessor(cfg, df_train=df_train)

    # 5. Transform feature matrices
    X_train = preprocessor.transform(df_train, scaled=scaled)
    X_val = preprocessor.transform(df_val, scaled=scaled)
    X_test = preprocessor.transform(df_test, scaled=scaled)
    feature_names = preprocessor.feature_names_

    # 6. Extract targets and classes
    y_train = df_train[target_col].to_numpy(dtype=np.float32)
    y_val = df_val[target_col].to_numpy(dtype=np.float32)
    y_test = df_test[target_col].to_numpy(dtype=np.float32)

    y_class_train = df_train[class_col].to_numpy(dtype=object) if class_col in df.columns else None
    y_class_val = df_val[class_col].to_numpy(dtype=object) if class_col in df.columns else None
    y_class_test = df_test[class_col].to_numpy(dtype=object) if class_col in df.columns else None

    # 7. Metadata DataFrame for test split
    meta_cols = ["Country", "Area", "Waterbody Type", "Date"]
    available_meta = [c for c in meta_cols if c in df_test.columns]
    meta_test = df_test[available_meta].copy().reset_index()
    meta_test = meta_test.rename(columns={"index": "row_id"})

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names": list(feature_names),
        "row_id_train": train_idx,
        "row_id_val": val_idx,
        "row_id_test": test_idx,
        "meta_test": meta_test,
        "y_class_train": y_class_train,
        "y_class_val": y_class_val,
        "y_class_test": y_class_test
    }


def get_dataloaders(
    cfg: Dict[str, Any],
    batch_size: Optional[int] = None,
    mock: bool = False,
    scaled: bool = True,
    num_workers: int = 0
) -> Tuple[Any, Any, Any]:
    """
    Convert split arrays into PyTorch DataLoaders.
    WHY: Standardized batching and device streaming for Member 2's MLP architecture.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for get_dataloaders. Please install torch.")

    splits = load_splits(cfg, scaled=scaled, mock=mock)
    bs = batch_size or cfg.get("mlp", {}).get("batch_size", 256)

    train_dataset = TensorDataset(
        torch.from_numpy(splits["X_train"]),
        torch.from_numpy(splits["y_train"])
    )
    val_dataset = TensorDataset(
        torch.from_numpy(splits["X_val"]),
        torch.from_numpy(splits["y_val"])
    )
    test_dataset = TensorDataset(
        torch.from_numpy(splits["X_test"]),
        torch.from_numpy(splits["y_test"])
    )

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
