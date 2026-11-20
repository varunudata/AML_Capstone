"""
Data pipeline & DataLoaders for DL phase (Member 1).
Implements the DATA CONTRACT:
  load_splits(cfg, scaled=True, mock=False) -> dict
  get_dataloaders(cfg, batch_size) -> train_loader, val_loader, test_loader
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Set
import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .utils import get_logger, seed_everything, resolve_path

logger = get_logger("dl.data")


def load_dataset(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Load the cleaned, frozen parquet dataset.
    WHY: Reuses exact AML scope decisions (ambient waterbodies, year >= 1990, impossible values flagged).
    """
    parquet_path = resolve_path(cfg["data"]["clean_parquet_path"])
    if not parquet_path.exists():
        raise FileNotFoundError(f"Clean dataset not found at {parquet_path}")
    logger.info("Loading clean dataset from: %s", parquet_path)
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
        logger.info("split_indices.json not found, falling back to: %s", aml_split_path)
        npz = np.load(aml_split_path)
        return {
            "metadata": {"source": "site_held_out_split.npz"},
            "train_sample_idx": npz["train_idx"].tolist(),
            "val_sample_idx": npz["val_idx"].tolist(),
            "test_sample_idx": npz["test_idx"].tolist(),
        }

    raise FileNotFoundError(f"Neither {split_indices_path} nor {aml_split_path} could be located.")


def verify_zero_site_leakage(df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, group_col: str = "Area") -> Dict[str, int]:
    """
    Assert zero site overlap between train, val, and test splits.
    WHY: Standard random splitting leaks >91% of monitoring stations, causing severe generalization overestimation.
    """
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


def _generate_mock_splits(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate deterministic synthetic data adhering to the EXACT DATA CONTRACT.
    WHY: Enables Member 2 (MLP) and Member 3 (Evaluation) to write and verify downstream code
    before real data processing pipeline lands.
    """
    seed = cfg.get("data", {}).get("split_seed", 42)
    rng = np.random.RandomState(seed)

    feature_names = cfg.get("data", {}).get("feature_list", [f"feat_{i}" for i in range(21)])
    n_features = len(feature_names)

    # Synthetic sizes
    n_train = 128
    n_val = 32
    n_test = 32

    # Synthetic features (standard normal)
    X_train = rng.randn(n_train, n_features).astype(np.float32)
    X_val = rng.randn(n_val, n_features).astype(np.float32)
    X_test = rng.randn(n_test, n_features).astype(np.float32)

    # Synthetic continuous targets in [31.3, 100.0]
    y_train = np.clip(rng.normal(loc=85.0, scale=12.0, size=n_train), 31.3, 100.0).astype(np.float32)
    y_val = np.clip(rng.normal(loc=85.0, scale=12.0, size=n_val), 31.3, 100.0).astype(np.float32)
    y_test = np.clip(rng.normal(loc=85.0, scale=12.0, size=n_test), 31.3, 100.0).astype(np.float32)

    # Row IDs
    row_id_train = np.arange(0, n_train, dtype=np.int64)
    row_id_val = np.arange(n_train, n_train + n_val, dtype=np.int64)
    row_id_test = np.arange(n_train + n_val, n_train + n_val + n_test, dtype=np.int64)

    # Discrete regulatory class mapping (CCME WQI standard)
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

    # Metadata DataFrame for test split
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
    scaled: bool = False,
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

    # Load dataset & split indices
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
        # Load from full split npz
        aml_split_path = resolve_path(cfg["data"]["aml_split_path"])
        npz = np.load(aml_split_path)
        train_idx = npz["train_idx"].astype(np.int64)
        val_idx = npz["val_idx"].astype(np.int64)
        test_idx = npz["test_idx"].astype(np.int64)

    # Verify zero site leakage
    verify_zero_site_leakage(df, train_idx, val_idx, test_idx, group_col=group_col)

    # Determine feature list
    feature_names = cfg.get("data", {}).get("raw_features", [
        "Ammonia (mg/l)", "Biochemical Oxygen Demand (mg/l)",
        "Dissolved Oxygen (mg/l)", "Orthophosphate (mg/l)",
        "pH (ph units)", "Temperature (cel)",
        "Nitrogen (mg/l)", "Nitrate (mg/l)"
    ])

    # Subset DataFrame folds
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    df_test = df.iloc[test_idx]

    # Extract target arrays
    y_train = df_train[target_col].to_numpy(dtype=np.float32)
    y_val = df_val[target_col].to_numpy(dtype=np.float32)
    y_test = df_test[target_col].to_numpy(dtype=np.float32)

    # Extract class arrays
    y_class_train = df_train[class_col].to_numpy(dtype=object) if class_col in df.columns else None
    y_class_val = df_val[class_col].to_numpy(dtype=object) if class_col in df.columns else None
    y_class_test = df_test[class_col].to_numpy(dtype=object) if class_col in df.columns else None

    # Extract feature matrices
    X_train = df_train[feature_names].to_numpy(dtype=np.float32)
    X_val = df_val[feature_names].to_numpy(dtype=np.float32)
    X_test = df_test[feature_names].to_numpy(dtype=np.float32)

    # Metadata DataFrame for test split
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
    scaled: bool = False,
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
