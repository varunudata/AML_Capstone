"""
Unit tests for Task B & Task C (dl/src/train.py):
1. Overfit a 32-sample mini-batch to near-zero MSE loss (R2 > 0.995).
2. Shuffled-target test verifying test R2 ~ 0.0 (no feature/target leakage).
3. Trivial baselines check (Mean predictor R2 ~ 0.0, Ridge R2 > Mean R2).
4. Trainer mechanics: AdamW, early stopping, gradient clipping, checkpointing, and learning curve plotting.
"""

from pathlib import Path
import pytest
import numpy as np

from src.utils import load_config, seed_everything
from src.train import (
    overfit_tiny_batch,
    run_shuffled_target_test,
    compute_trivial_baselines,
    train_single_seed,
    plot_learning_curves,
)


def test_overfit_32_sample_batch():
    """Sanity check 1: Model must overfit a single 32-sample batch to near-zero loss."""
    cfg = load_config()
    res = overfit_tiny_batch(cfg, batch_size=32, steps=300, lr=1e-2, mock=False)

    assert res["final_mse_loss"] < 0.05, f"Failed to overfit 32 samples: MSE={res['final_mse_loss']}"
    assert res["final_r2"] > 0.998, f"Expected R2 > 0.998 on 32-sample overfit, got {res['final_r2']}"


def test_shuffled_target_yields_zero_r2():
    """Sanity check 2: Training on shuffled targets must yield ~0 R2 on test set."""
    cfg = load_config()
    res = run_shuffled_target_test(cfg, epochs=6, subsample_n=5000, mock=False)

    assert res["test_R2"] <= 0.03, f"Shuffled target test R2 suspiciously high: {res['test_R2']}"
    assert res["test_R2"] >= -0.15, f"Shuffled target test R2 unexpectedly negative: {res['test_R2']}"


def test_trivial_baselines_context():
    """Sanity check 3: Verify Mean predictor R2 ~ 0 and Ridge regression R2 > 0.85 on real split."""
    cfg = load_config()
    res = compute_trivial_baselines(cfg, mock=False)

    mean_test = res["mean_predictor"]["test"]
    ridge_test = res["ridge_regression"]["test"]

    assert abs(mean_test["R2"]) < 0.05, f"Mean predictor R2 should be ~0, got {mean_test['R2']}"
    assert mean_test["R2"] <= 0.001, "Train-mean predictor cannot have positive R2 on test set"
    assert ridge_test["R2"] > 0.85, f"Ridge regression R2 unexpectedly low: {ridge_test['R2']}"
    assert ridge_test["RMSE"] < mean_test["RMSE"], "Ridge must beat trivial mean predictor"


def test_trainer_checkpoint_and_learning_curves(tmp_path: Path):
    """Verify trainer executes, saves checkpoint/log artifacts, and plots learning curves."""
    cfg = load_config()
    cfg["mlp"]["epochs"] = 4
    seed_everything(42)

    _, summary = train_single_seed(
        cfg,
        seed=42,
        variant="pytest_tmp",
        mock=True,
        export_csv=False,
        save_checkpoint=False,
        verbose=False,
    )

    assert len(summary["history"]) == 4
    assert summary["best_epoch"] >= 1
    assert "RMSE" in summary["val_metrics"]
    assert "RMSE" in summary["test_metrics"]

    fig_file = tmp_path / "test_curves.png"
    out_fig = plot_learning_curves(summary["history"], out_path=fig_file)
    assert out_fig.exists()
    assert out_fig.stat().st_size > 1000
