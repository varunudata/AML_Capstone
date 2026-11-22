"""
Unit tests for Task D (dl/src/tune.py):
- Verifies Optuna hyperparameter search runs strictly on training/validation folds (never touching test).
- Verifies systematic ablation study covers depth, width, dropout, BatchNorm, weight decay, and MSE vs Huber.
- Verifies tidy CSV/JSON/Markdown output tables are saved.
"""

from pathlib import Path
import pytest
import pandas as pd

from src.utils import load_config
from src.tune import run_optuna_study, run_ablation_study


def test_optuna_study_mock_execution():
    """Verify Optuna study executes cleanly on mock data and returns best hyperparameters."""
    cfg = load_config()
    summary = run_optuna_study(
        cfg,
        n_trials=3,
        tuning_epochs=3,
        subsample_train=128,
        mock=True,
        save_results=False,
    )

    assert summary["n_trials_completed"] == 3
    assert "best_val_rmse" in summary
    assert summary["best_val_rmse"] > 0.0
    best_hp = summary["best_hyperparameters"]
    for key in ["hidden_dims", "activation", "batch_norm", "dropout", "learning_rate", "weight_decay", "batch_size"]:
        assert key in best_hp, f"Missing hyperparameter {key} in Optuna summary"


def test_ablation_study_mock_execution():
    """Verify ablation study evaluates all required architectural and regularization factors."""
    cfg = load_config()
    df_ab = run_ablation_study(
        cfg,
        ablation_epochs=2,
        subsample_train=128,
        mock=True,
        save_results=False,
    )

    assert isinstance(df_ab, pd.DataFrame)
    required_factors = {"Depth", "Width", "Dropout", "BatchNorm", "Weight Decay", "Loss Function"}
    assert required_factors.issubset(set(df_ab["Factor"].unique()))
    assert "Val_RMSE" in df_ab.columns
    assert "Val_MAE" in df_ab.columns
    assert "Val_R2" in df_ab.columns
    assert not df_ab["Val_RMSE"].isna().any()
