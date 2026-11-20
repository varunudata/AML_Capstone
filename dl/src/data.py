"""
Data pipeline & DataLoaders for DL phase (Member 1).
Implements the DATA CONTRACT:
  load_splits(cfg, scaled=True, mock=False) -> dict
  get_dataloaders(cfg, batch_size) -> train_loader, val_loader, test_loader
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .utils import get_logger, seed_everything

logger = get_logger("dl.data")


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


def load_splits(cfg: Dict[str, Any], scaled: bool = True, mock: bool = False) -> Dict[str, Any]:
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

    raise NotImplementedError(
        "Real data loading will be activated in Commit 2. Use mock=True for initial teammate development."
    )


def get_dataloaders(
    cfg: Dict[str, Any],
    batch_size: Optional[int] = None,
    mock: bool = False,
    num_workers: int = 0
) -> Tuple[Any, Any, Any]:
    """
    Convert split arrays into PyTorch DataLoaders.
    WHY: Standardized batching and device streaming for Member 2's MLP architecture.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for get_dataloaders. Please install torch.")

    splits = load_splits(cfg, scaled=True, mock=mock)

    bs = batch_size or cfg.get("mlp", {}).get("batch_size", 256)

    # Convert to PyTorch tensors
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
