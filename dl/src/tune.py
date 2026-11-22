"""
Optuna Hyperparameter Optimization & Systematic Ablation Runner for WaterQualityMLP (Member 2).
Implements Task D:
- Optuna search over depth (1-4 layers), width (16-256), dropout (0.0-0.5),
  learning rate (1e-4..1e-2 log-scale), weight decay (1e-6..1e-2 log-scale),
  batch size (256, 512, 1024), activation (ReLU, GELU, SiLU), and BatchNorm (on/off).
- Objective = Validation RMSE ONLY (test split is strictly isolated and never accessed).
- Trial budget matched to AML baseline tuning budget (15 Optuna trials recorded in aml_baseline_metrics.json).
- Systematic ablation study over:
    1. Depth (1, 2, 3, 4 layers)
    2. Width (32, 128, 256 hidden units)
    3. Dropout (0.0, 0.1, 0.3)
    4. BatchNorm (On vs Off)
    5. Weight Decay (0.0, 1e-4, 1e-2)
    6. Loss Function (MSE vs Huber)
- Exports tidy CSV, Markdown, and JSON tables to results/tuning/ and results/tables/.
"""

import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import optuna
import torch
from torch.utils.data import DataLoader, TensorDataset

optuna.logging.set_verbosity(optuna.logging.WARNING)

_DL_ROOT_STR = str(Path(__file__).resolve().parent.parent)
if _DL_ROOT_STR not in sys.path:
    sys.path.insert(0, _DL_ROOT_STR)

try:
    from .utils import get_logger, seed_everything, load_config, DL_ROOT
    from .data import load_splits
    from .models import WaterQualityMLP, count_parameters
    from .train import MLPTrainer
except (ImportError, ValueError):
    from src.utils import get_logger, seed_everything, load_config, DL_ROOT
    from src.data import load_splits
    from src.models import WaterQualityMLP, count_parameters
    from src.train import MLPTrainer

logger = get_logger("dl.tune")


def _prepare_cached_tensors(
    cfg: Dict[str, Any],
    mock: bool = False,
    subsample_train: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Load training and validation tensors once into memory for rapid tuning/ablation.
    NOTE: X_test and y_test are intentionally NOT returned so tuning can never touch the test set.
    """
    splits = load_splits(cfg, scaled=True, mock=mock)
    x_tr = splits["X_train"]
    y_tr = splits["y_train"]
    x_val = splits["X_val"]
    y_val = splits["y_val"]

    if subsample_train is not None and len(x_tr) > subsample_train:
        rng = np.random.RandomState(cfg.get("mlp", {}).get("seed", 42))
        idx = rng.choice(len(x_tr), size=subsample_train, replace=False)
        x_tr = x_tr[idx]
        y_tr = y_tr[idx]

    return (
        torch.from_numpy(x_tr),
        torch.from_numpy(y_tr),
        torch.from_numpy(x_val),
        torch.from_numpy(y_val),
        int(x_tr.shape[1]),
    )


def run_optuna_study(
    cfg: Optional[Dict[str, Any]] = None,
    n_trials: Optional[int] = None,
    tuning_epochs: Optional[int] = None,
    subsample_train: Optional[int] = None,
    mock: bool = False,
    save_results: bool = True,
) -> Dict[str, Any]:
    """
    Run Optuna TPE hyperparameter optimization with budget matched to AML (15 trials).
    Optimizes validation RMSE exclusively.
    """
    if cfg is None:
        cfg = load_config()

    mlp_cfg = cfg.get("mlp", {})
    tune_cfg = mlp_cfg.get("tuning", {})
    seed = int(mlp_cfg.get("seed", 42))

    # Match AML baseline Optuna budget (15 trials in results/aml_baseline_metrics.json)
    aml_metrics_path = DL_ROOT / "results" / "aml_baseline_metrics.json"
    aml_budget_str = "15 Optuna GroupKFold trials on Area"
    if aml_metrics_path.exists():
        with open(aml_metrics_path, "r", encoding="utf-8") as f:
            aml_meta = json.load(f).get("meta", {})
            aml_budget_str = aml_meta.get("tuning_budget", aml_budget_str)

    budget_trials = int(n_trials if n_trials is not None else tune_cfg.get("n_trials", 15))
    epochs = int(tuning_epochs if tuning_epochs is not None else tune_cfg.get("tuning_epochs", 25))
    sub_n = int(subsample_train if subsample_train is not None else tune_cfg.get("subsample_train", 60000))

    logger.info(
        "Starting Optuna MLP search | budget=%d trials (matched to AML: '%s') | epochs=%d | train_subsample=%d",
        budget_trials,
        aml_budget_str,
        epochs,
        sub_n,
    )

    x_tr_t, y_tr_t, x_val_t, y_val_t, input_dim = _prepare_cached_tensors(
        cfg, mock=mock, subsample_train=sub_n
    )
    val_ds = TensorDataset(x_val_t, y_val_t)

    trial_records: List[Dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        seed_everything(seed + trial.number)

        n_layers = trial.suggest_int("n_layers", 1, 4)
        base_width = trial.suggest_categorical("width", [16, 32, 64, 128, 256])
        # Construct tapered pyramid or uniform hidden dimensions within [16, 256]
        taper = trial.suggest_categorical("taper", [True, False])
        if taper:
            hidden_dims = [max(16, base_width // (2 ** i)) for i in range(n_layers)]
        else:
            hidden_dims = [base_width] * n_layers

        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
        activation = trial.suggest_categorical("activation", ["relu", "gelu", "silu"])
        batch_norm = trial.suggest_categorical("batch_norm", [True, False])

        train_loader = DataLoader(TensorDataset(x_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)

        model = WaterQualityMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        trial_cfg = copy.deepcopy(cfg)
        trial_cfg["mlp"].update({
            "hidden_dims": hidden_dims,
            "activation": activation,
            "dropout": dropout,
            "batch_norm": batch_norm,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "early_stopping_patience": max(5, epochs // 3),
        })

        trainer = MLPTrainer(
            model=model,
            cfg=trial_cfg,
            run_name=f"optuna_trial_{trial.number:02d}",
        )
        summary = trainer.fit(train_loader, val_loader, save_checkpoint=False, verbose=False)
        val_m = summary["val_metrics"]

        rec = {
            "trial": trial.number,
            "n_layers": n_layers,
            "width": base_width,
            "taper": taper,
            "hidden_dims": str(hidden_dims),
            "activation": activation,
            "batch_norm": batch_norm,
            "dropout": round(dropout, 3),
            "learning_rate": round(lr, 6),
            "weight_decay": round(weight_decay, 7),
            "batch_size": batch_size,
            "n_parameters": summary["n_parameters"],
            "best_epoch": summary["best_epoch"],
            "val_RMSE": val_m["RMSE"],
            "val_MAE": val_m["MAE"],
            "val_R2": val_m["R2"],
            "val_Macro_F1": val_m.get("Macro_F1", 0.0),
            "train_time_sec": summary["train_time_sec"],
        }
        trial_records.append(rec)
        logger.info(
            "Trial %02d/%02d | dims=%s act=%s bn=%s drop=%.2f lr=%.4f bs=%d -> val_RMSE=%.4f (val_R2=%.4f)",
            trial.number + 1,
            budget_trials,
            hidden_dims,
            activation,
            batch_norm,
            dropout,
            lr,
            batch_size,
            val_m["RMSE"],
            val_m["R2"],
        )
        return float(val_m["RMSE"])

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name="water_quality_mlp_optuna")
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=budget_trials)
    total_time = time.perf_counter() - t0

    df_trials = pd.DataFrame(trial_records).sort_values("val_RMSE", ascending=True).reset_index(drop=True)
    best_row = df_trials.iloc[0].to_dict()

    # Parse hidden_dims list back from best trial
    best_hidden_dims = json.loads(best_row["hidden_dims"])

    study_summary = {
        "study_name": "water_quality_mlp_optuna",
        "aml_tuning_budget_reference": aml_budget_str,
        "n_trials_completed": budget_trials,
        "tuning_epochs_per_trial": epochs,
        "subsample_train": sub_n,
        "objective_metric": "val_RMSE (validation split only; test split untouched)",
        "total_tuning_time_sec": round(total_time, 3),
        "best_val_rmse": float(best_row["val_RMSE"]),
        "best_val_mae": float(best_row["val_MAE"]),
        "best_val_r2": float(best_row["val_R2"]),
        "best_hyperparameters": {
            "hidden_dims": best_hidden_dims,
            "n_layers": int(best_row["n_layers"]),
            "width": int(best_row["width"]),
            "taper": bool(best_row["taper"]),
            "activation": str(best_row["activation"]),
            "batch_norm": bool(best_row["batch_norm"]),
            "dropout": float(best_row["dropout"]),
            "learning_rate": float(best_row["learning_rate"]),
            "weight_decay": float(best_row["weight_decay"]),
            "batch_size": int(best_row["batch_size"]),
        },
        "trials": trial_records,
    }

    if save_results:
        for folder in [DL_ROOT / "results" / "tuning", DL_ROOT / "results" / "tables"]:
            folder.mkdir(parents=True, exist_ok=True)
            df_trials.to_csv(folder / "optuna_trials.csv", index=False)
            with open(folder / "optuna_study_summary.json", "w", encoding="utf-8") as f:
                json.dump(study_summary, f, indent=2)

        logger.info(
            "Optuna tuning complete! Best Val RMSE=%.4f | Best params=%s",
            study_summary["best_val_rmse"],
            study_summary["best_hyperparameters"],
        )

    return study_summary


def run_ablation_study(
    cfg: Optional[Dict[str, Any]] = None,
    base_params: Optional[Dict[str, Any]] = None,
    ablation_epochs: int = 25,
    subsample_train: int = 60000,
    mock: bool = False,
    save_results: bool = True,
) -> pd.DataFrame:
    """
    Systematic Ablation Runner (Task D):
    Evaluates one-factor-at-a-time variations across:
      1. Depth (1, 2, 3, 4 layers)
      2. Width (narrow [32, 16], medium [128, 64, 32], wide [256, 128, 64])
      3. Dropout (0.0, 0.1, 0.3)
      4. BatchNorm (On vs Off)
      5. Weight Decay (0.0, 1e-4, 1e-2)
      6. Loss Function (MSE vs Huber)
    Uses validation split ONLY and exports tidy CSV/Markdown/JSON tables.
    """
    if cfg is None:
        cfg = load_config()

    seed = int(cfg.get("mlp", {}).get("seed", 42))
    x_tr_t, y_tr_t, x_val_t, y_val_t, input_dim = _prepare_cached_tensors(
        cfg, mock=mock, subsample_train=subsample_train
    )

    # Anchor configuration for controlled single-factor ablations
    anchor = {
        "hidden_dims": [128, 64, 32],
        "activation": "gelu",
        "dropout": 0.1,
        "batch_norm": True,
        "learning_rate": 0.002,
        "weight_decay": 1e-4,
        "batch_size": 512,
        "loss_type": "huber",
    }
    if base_params is not None:
        anchor.update({k: v for k, v in base_params.items() if k in anchor})

    ablation_specs: List[Tuple[str, str, Dict[str, Any]]] = [
        # 1. Depth ablation
        ("Depth", "1 Layer [128]", {"hidden_dims": [128]}),
        ("Depth", "2 Layers [128, 64]", {"hidden_dims": [128, 64]}),
        ("Depth", "3 Layers [128, 64, 32] (Anchor)", {"hidden_dims": [128, 64, 32]}),
        ("Depth", "4 Layers [256, 128, 64, 32]", {"hidden_dims": [256, 128, 64, 32]}),
        # 2. Width ablation
        ("Width", "Narrow [32, 16]", {"hidden_dims": [32, 16]}),
        ("Width", "Medium [64, 64, 32]", {"hidden_dims": [64, 64, 32]}),
        ("Width", "Wide [256, 128, 64]", {"hidden_dims": [256, 128, 64]}),
        # 3. Dropout ablation
        ("Dropout", "Dropout p=0.0 (None)", {"dropout": 0.0}),
        ("Dropout", "Dropout p=0.1 (Anchor)", {"dropout": 0.1}),
        ("Dropout", "Dropout p=0.3 (Heavy)", {"dropout": 0.3}),
        # 4. BatchNorm ablation
        ("BatchNorm", "BatchNorm Off", {"batch_norm": False}),
        ("BatchNorm", "BatchNorm On (Anchor)", {"batch_norm": True}),
        # 5. Weight Decay ablation
        ("Weight Decay", "WD = 0.0", {"weight_decay": 0.0}),
        ("Weight Decay", "WD = 1e-4 (Anchor)", {"weight_decay": 1e-4}),
        ("Weight Decay", "WD = 1e-2 (Strong)", {"weight_decay": 1e-2}),
        # 6. Loss Function ablation
        ("Loss Function", "MSE Loss", {"loss_type": "mse"}),
        ("Loss Function", "Huber Loss (Anchor)", {"loss_type": "huber"}),
        # 7. Activation ablation
        ("Activation", "ReLU", {"activation": "relu"}),
        ("Activation", "GELU (Anchor)", {"activation": "gelu"}),
        ("Activation", "SiLU", {"activation": "silu"}),
    ]

    val_loader = DataLoader(TensorDataset(x_val_t, y_val_t), batch_size=1024, shuffle=False)
    rows: List[Dict[str, Any]] = []

    logger.info("Running systematic MLP ablation study (%d configurations)...", len(ablation_specs))
    for idx, (factor, label, overrides) in enumerate(ablation_specs, start=1):
        seed_everything(seed)
        spec = copy.deepcopy(anchor)
        spec.update(overrides)

        train_loader = DataLoader(
            TensorDataset(x_tr_t, y_tr_t),
            batch_size=int(spec["batch_size"]),
            shuffle=True,
        )

        model = WaterQualityMLP(
            input_dim=input_dim,
            hidden_dims=list(spec["hidden_dims"]),
            activation=str(spec["activation"]),
            dropout=float(spec["dropout"]),
            batch_norm=bool(spec["batch_norm"]),
        )

        ab_cfg = copy.deepcopy(cfg)
        ab_cfg["mlp"].update(spec)
        ab_cfg["mlp"]["epochs"] = ablation_epochs
        ab_cfg["mlp"]["early_stopping_patience"] = max(6, ablation_epochs // 3)

        trainer = MLPTrainer(
            model=model,
            cfg=ab_cfg,
            run_name=f"ablation_{idx:02d}",
        )
        summary = trainer.fit(train_loader, val_loader, save_checkpoint=False, verbose=False)
        val_m = summary["val_metrics"]

        row = {
            "Factor": factor,
            "Variant": label,
            "Hidden_Dims": str(spec["hidden_dims"]),
            "Activation": spec["activation"],
            "BatchNorm": spec["batch_norm"],
            "Dropout": spec["dropout"],
            "Weight_Decay": spec["weight_decay"],
            "Loss": spec["loss_type"].upper(),
            "Params": summary["n_parameters"],
            "Best_Epoch": summary["best_epoch"],
            "Val_RMSE": val_m["RMSE"],
            "Val_MAE": val_m["MAE"],
            "Val_R2": val_m["R2"],
            "Val_Macro_F1": val_m.get("Macro_F1", 0.0),
            "Train_Time_s": round(summary["train_time_sec"], 2),
        }
        rows.append(row)
        logger.info(
            "Ablation [%02d/%02d] %-13s | %-28s -> Val RMSE=%.4f | Val MAE=%.4f | Val R2=%.4f",
            idx,
            len(ablation_specs),
            factor,
            label,
            val_m["RMSE"],
            val_m["MAE"],
            val_m["R2"],
        )

    df_ablation = pd.DataFrame(rows)

    if save_results:
        for folder in [DL_ROOT / "results" / "tuning", DL_ROOT / "results" / "tables"]:
            folder.mkdir(parents=True, exist_ok=True)
            df_ablation.to_csv(folder / "ablation_results.csv", index=False)
            with open(folder / "ablation_results.json", "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2)
            try:
                md_table = df_ablation.to_markdown(index=False)
            except Exception:
                md_table = df_ablation.to_string(index=False)
            with open(folder / "ablation_results.md", "w", encoding="utf-8") as f:
                f.write("# Phase 2 MLP Hyperparameter & Architecture Ablation Table\n\n")
                f.write(md_table + "\n")

        logger.info("Saved ablation tables to results/tuning/ and results/tables/")

    return df_ablation


if __name__ == "__main__":
    cfg_loaded = load_config()
    study_res = run_optuna_study(cfg_loaded, mock=False, save_results=True)
    run_ablation_study(
        cfg_loaded,
        base_params=study_res["best_hyperparameters"],
        mock=False,
        save_results=True,
    )
