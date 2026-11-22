"""
Unit tests verifying the MOCK DATA CONTRACT for Member 2 and Member 3.
Ensures that mock=True outputs:
1. All required keys exist.
2. Tensor dtypes and shapes are valid.
3. PyTorch DataLoaders instantiate and yield correct batch types.
"""

import pytest
import numpy as np
import pandas as pd
import torch

from src.utils import load_config, seed_everything
from src.data import load_splits, get_dataloaders


def test_mock_splits_contract_keys():
    """Verify all keys mandated by the team contract are present."""
    cfg = load_config()
    data = load_splits(cfg, scaled=True, mock=True)

    expected_keys = {
        "X_train", "X_val", "X_test",
        "y_train", "y_val", "y_test",
        "feature_names",
        "row_id_train", "row_id_val", "row_id_test",
        "meta_test",
        "y_class_train", "y_class_val", "y_class_test"
    }

    assert expected_keys.issubset(set(data.keys())), f"Missing keys: {expected_keys - set(data.keys())}"


def test_mock_splits_dtypes_and_shapes():
    """Verify data types and dimensional consistency."""
    cfg = load_config()
    data = load_splits(cfg, scaled=True, mock=True)

    # Dtypes
    assert data["X_train"].dtype == np.float32
    assert data["X_val"].dtype == np.float32
    assert data["X_test"].dtype == np.float32

    assert data["y_train"].dtype == np.float32
    assert data["y_val"].dtype == np.float32
    assert data["y_test"].dtype == np.float32

    # Dimensions
    n_features = len(data["feature_names"])
    assert data["X_train"].shape[1] == n_features
    assert data["X_val"].shape[1] == n_features
    assert data["X_test"].shape[1] == n_features

    assert len(data["X_train"]) == len(data["y_train"]) == len(data["row_id_train"])
    assert len(data["X_val"]) == len(data["y_val"]) == len(data["row_id_val"])
    assert len(data["X_test"]) == len(data["y_test"]) == len(data["row_id_test"])

    # Metadata DataFrame
    assert isinstance(data["meta_test"], pd.DataFrame)
    assert len(data["meta_test"]) == len(data["y_test"])
    assert "Area" in data["meta_test"].columns
    assert "Date" in data["meta_test"].columns


def test_mock_dataloaders():
    """Verify DataLoaders yield valid torch Tensors with expected shapes."""
    cfg = load_config()
    train_loader, val_loader, test_loader = get_dataloaders(cfg, batch_size=16, mock=True)

    batch_x, batch_y = next(iter(train_loader))
    assert isinstance(batch_x, torch.Tensor)
    assert isinstance(batch_y, torch.Tensor)
    assert batch_x.dtype == torch.float32
    assert batch_y.dtype == torch.float32
    assert batch_x.shape[0] == 16
    assert batch_x.shape[1] == len(cfg["data"]["feature_list"])
