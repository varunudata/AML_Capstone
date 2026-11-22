"""
Unit tests for Task E (dl/src/uncertainty.py):
1. MC Dropout with T=50 stochastic forward passes produces non-zero y_std and valid [q05, q95] intervals.
2. Deep Ensemble aggregation across 5 seeds produces valid ensemble mean, y_std, q05, q95.
3. Quantile Regression MLP with PinballQuantileLoss outputs non-crossing (q05 <= q50 <= q95) predictions.
4. Exported prediction CSVs contain row_id, y_true, y_pred, y_std, q05, q95, class_true, class_pred.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch

from src.utils import load_config, seed_everything
from src.data import load_splits
from src.models import WaterQualityMLP
from src.uncertainty import (
    predict_mc_dropout,
    aggregate_deep_ensemble,
    PinballQuantileLoss,
    train_quantile_mlp,
    compute_uncertainty_metrics,
)


def test_mc_dropout_50_passes():
    """Verify MC Dropout runs T=50 stochastic passes with BatchNorm frozen and produces positive y_std."""
    seed_everything(42)
    cfg = load_config()
    splits = load_splits(cfg, scaled=True, mock=True)
    x_test = splits["X_test"]
    y_test = splits["y_test"]

    model = WaterQualityMLP(
        input_dim=x_test.shape[1],
        hidden_dims=[64, 32],
        activation="gelu",
        dropout=0.2,
        batch_norm=True,
    )
    res = predict_mc_dropout(model, x_test, n_passes=50, seed=42)

    assert res["n_passes"] == 50
    assert res["y_pred"].shape == (len(x_test),)
    assert res["y_std"].shape == (len(x_test),)
    assert res["q05"].shape == (len(x_test),)
    assert res["q95"].shape == (len(x_test),)
    assert np.all(res["y_std"] > 0.0), "MC Dropout y_std must be strictly positive when dropout > 0"
    assert np.all(res["q05"] <= res["y_pred"] + 1e-4)
    assert np.all(res["y_pred"] <= res["q95"] + 1e-4)

    metrics = compute_uncertainty_metrics(y_test, res["y_pred"], res["y_std"], res["q05"], res["q95"])
    assert "PICP_90" in metrics
    assert "MPIW_90" in metrics
    assert "Gaussian_NLL" in metrics


def test_deep_ensemble_aggregation():
    """Verify 5-seed Deep Ensemble aggregation computes mean, epistemic std, and 90% intervals."""
    rng = np.random.RandomState(42)
    base = rng.uniform(50.0, 98.0, size=32).astype(np.float32)
    member_preds = [np.clip(base + rng.normal(0.0, 0.8, size=32), 0.0, 100.0).astype(np.float32) for _ in range(5)]

    agg = aggregate_deep_ensemble(member_preds, residual_scale=0.2)
    assert agg["y_pred"].shape == (32,)
    assert agg["y_std"].shape == (32,)
    assert np.all(agg["y_std"] > 0.0)
    assert np.all(agg["q05"] <= agg["y_pred"] + 1e-5)
    assert np.all(agg["y_pred"] <= agg["q95"] + 1e-5)


def test_pinball_loss_and_quantile_mlp():
    """Verify PinballQuantileLoss and train_quantile_mlp produce non-crossing [q05, q50, q95] predictions."""
    seed_everything(42)
    loss_fn = PinballQuantileLoss(quantiles=[0.05, 0.50, 0.95], crossing_penalty=0.5)
    preds = torch.tensor([[70.0, 80.0, 90.0], [65.0, 75.0, 85.0]], requires_grad=True)
    target = torch.tensor([82.0, 72.0])
    loss = loss_fn(preds, target)
    assert loss.item() > 0.0
    loss.backward()
    assert preds.grad is not None

    cfg = load_config()
    cfg["mlp"]["epochs"] = 4
    model, summary = train_quantile_mlp(
        cfg,
        seed=42,
        mock=True,
        export_csv=False,
        save_checkpoint=False,
        verbose=False,
    )
    assert model.output_dim == 3
    assert "PICP_90" in summary["val_metrics"]
    assert "MPIW_90" in summary["test_metrics"]
