"""
Training Engine & Experiment Runner for Water Quality MLP Models (Member 2).
Implements Task B:
- AdamW optimizer with decoupled L2 weight decay.
- Learning rate scheduling (ReduceLROnPlateau on validation RMSE or CosineAnnealingLR).
- Early stopping strictly on validation RMSE (never touches test fold during training).
- Gradient norm clipping for training stability on skewed chemical inputs.
- Best-model checkpointing (.pt) + companion configuration JSON saved next to each checkpoint.
- Structured JSON training logs with epoch curves, parameter counts, and wall-clock timings.
- Deterministic seeding across Python, NumPy, and PyTorch (CPU-friendly execution).
- Loss selection: MSE or Huber loss (robust to tail outliers and boundary censoring at 100.0).
"""

import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

# Ensure dl/ is at the front of sys.path so joblib unpickling of src.data.WaterQualityPreprocessor
# works seamlessly whether invoked from repo root or dl/ directory.
_DL_ROOT_STR = str(Path(__file__).resolve().parent.parent)
if _DL_ROOT_STR not in sys.path:
    sys.path.insert(0, _DL_ROOT_STR)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score

try:
    from .utils import get_logger, seed_everything, load_config, DL_ROOT
    from .data import load_splits, get_dataloaders
    from .models import WaterQualityMLP, MultiTaskWaterQualityLoss, count_parameters, build_mlp_from_config
    from .aml_baselines import to_ccme_class
except (ImportError, ValueError):
    try:
        from dl.src.utils import get_logger, seed_everything, load_config, DL_ROOT
        from dl.src.data import load_splits, get_dataloaders
        from dl.src.models import WaterQualityMLP, MultiTaskWaterQualityLoss, count_parameters, build_mlp_from_config
        from dl.src.aml_baselines import to_ccme_class
    except ImportError:
        from src.utils import get_logger, seed_everything, load_config, DL_ROOT
        from src.data import load_splits, get_dataloaders
        from src.models import WaterQualityMLP, MultiTaskWaterQualityLoss, count_parameters, build_mlp_from_config
        from src.aml_baselines import to_ccme_class

logger = get_logger("dl.train")


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    include_macro_f1: bool = True,
) -> Dict[str, float]:
    """Compute MAE, RMSE, R2, and secondary discrete CCME Macro-F1."""
    y_t = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_p = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p))

    metrics = {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
    }
    if include_macro_f1:
        c_true = to_ccme_class(y_t)
        c_pred = to_ccme_class(y_p)
        metrics["Macro_F1"] = round(float(f1_score(c_true, c_pred, average="macro")), 4)
    return metrics


def export_dl_predictions_csv(
    filepath: Union[str, Path],
    row_id: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray] = None,
    q05: Optional[np.ndarray] = None,
    q95: Optional[np.ndarray] = None,
) -> Path:
    """
    Export predictions adhering strictly to the PREDICTIONS CONTRACT:
    Path: results/predictions/{family}_{variant}_seed{N}_{split}.csv
    Required columns: row_id, y_true, y_pred
    Optional columns: y_std, q05, q95, class_true, class_pred
    """
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_t = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_p = np.asarray(y_pred, dtype=np.float32).reshape(-1)

    payload: Dict[str, Any] = {
        "row_id": np.asarray(row_id, dtype=np.int64).reshape(-1),
        "y_true": np.round(y_t, 4),
        "y_pred": np.round(y_p, 4),
    }
    if y_std is not None:
        payload["y_std"] = np.round(np.asarray(y_std, dtype=np.float32).reshape(-1), 4)
    if q05 is not None:
        payload["q05"] = np.round(np.asarray(q05, dtype=np.float32).reshape(-1), 4)
    if q95 is not None:
        payload["q95"] = np.round(np.asarray(q95, dtype=np.float32).reshape(-1), 4)

    payload["class_true"] = to_ccme_class(y_t)
    payload["class_pred"] = to_ccme_class(y_p)

    df_out = pd.DataFrame(payload)
    df_out.to_csv(out_path, index=False)
    logger.info("Exported DL prediction CSV: %s (rows=%d)", out_path, len(df_out))
    return out_path


def _iter_batches(
    loader: DataLoader,
    shuffle: bool = False,
) -> List[Tuple[torch.Tensor, ...]]:
    """
    Fast, CPU-friendly batch iterator.
    WHY: Standard PyTorch DataLoader indexes TensorDataset row-by-row in Python (250,000 calls/epoch).
    When the underlying dataset is a TensorDataset in memory, vectorized tensor slicing in C++ is 8x faster
    while producing identical mini-batch tensor shapes and deterministic shuffling.
    """
    ds = loader.dataset
    if isinstance(ds, TensorDataset):
        tensors = ds.tensors
        n_samples = tensors[0].shape[0]
        bs = loader.batch_size or 256
        if shuffle:
            indices = torch.randperm(n_samples)
            return [
                tuple(t[indices[i : i + bs]] for t in tensors)
                for i in range(0, n_samples, bs)
            ]
        else:
            return [
                tuple(t[i : i + bs] for t in tensors)
                for i in range(0, n_samples, bs)
            ]
    return list(loader)


class MLPTrainer:
    """
    Config-driven Trainer for WaterQualityMLP.
    Handles AdamW optimization, learning rate scheduling, gradient clipping,
    validation RMSE early stopping, checkpoint persistence, and JSON logging.
    """

    def __init__(
        self,
        model: WaterQualityMLP,
        cfg: Dict[str, Any],
        run_name: str = "dl_mlp_seed42",
        device: Optional[Union[str, torch.device]] = None,
        save_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
    ) -> None:
        self.cfg = cfg
        self.mlp_cfg = cfg.get("mlp", {})
        self.run_name = run_name

        # CPU-friendly default device
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.save_dir = Path(save_dir) if save_dir is not None else (DL_ROOT / "results" / "models")
        self.log_dir = Path(log_dir) if log_dir is not None else (DL_ROOT / "results" / "logs")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Hyperparameters
        self.lr = float(self.mlp_cfg.get("learning_rate", 1e-3))
        self.weight_decay = float(self.mlp_cfg.get("weight_decay", 1e-4))
        self.epochs = int(self.mlp_cfg.get("epochs", 50))
        self.patience = int(self.mlp_cfg.get("early_stopping_patience", 10))
        self.max_grad_norm = float(self.mlp_cfg.get("max_grad_norm", 1.0))
        self.loss_type = str(self.mlp_cfg.get("loss_type", "huber")).lower()
        self.huber_delta = float(self.mlp_cfg.get("huber_delta", 1.0))
        self.scheduler_type = str(self.mlp_cfg.get("scheduler", "cosine")).lower()
        self.clip_predictions = bool(self.mlp_cfg.get("clip_predictions", True))
        self.clip_min = float(self.mlp_cfg.get("clip_min", 0.0))
        self.clip_max = float(self.mlp_cfg.get("clip_max", 100.0))

        # Loss function (MSE or Huber, with optional multi-task classification support)
        self.criterion = MultiTaskWaterQualityLoss(
            loss_type=self.loss_type,
            huber_delta=self.huber_delta,
        )

        # WHY AdamW: Decouples weight decay from adaptive gradient second moments,
        # providing cleaner L2 regularization on tabular MLP weights than standard Adam.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        if self.scheduler_type == "cosine":
            self.scheduler: Any = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, self.epochs), eta_min=1e-6
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=max(2, self.patience // 3)
            )

        self.history: List[Dict[str, float]] = []
        self.best_val_rmse: float = float("inf")
        self.best_epoch: int = 0
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.train_time_sec: float = 0.0

    def _extract_reg_output(
        self, outputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch with gradient clipping."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in _iter_batches(train_loader, shuffle=True):
            bx = batch[0].to(self.device)
            by = batch[1].to(self.device)
            b_cls = batch[2].to(self.device) if len(batch) > 2 else None

            # Guard against BatchNorm1d failing on single-sample final batch
            if self.model.batch_norm and bx.shape[0] < 2:
                continue

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(bx)
            loss = self.criterion(outputs, by, b_cls)
            loss.backward()

            # WHY Gradient Clipping: Prevents rare extreme nutrient spikes from causing destructive step jumps
            if self.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            bs = bx.shape[0]
            total_loss += float(loss.item()) * bs
            total_samples += bs

        return total_loss / max(1, total_samples)

    @torch.no_grad()
    def evaluate(
        self, loader: DataLoader
    ) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray, float]:
        """
        Evaluate model on a validation or test DataLoader.
        Returns: (avg_loss, metrics_dict, y_true_np, y_pred_np, inference_time_sec)
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        preds_list: List[np.ndarray] = []
        trues_list: List[np.ndarray] = []

        t0 = time.perf_counter()
        for batch in _iter_batches(loader, shuffle=False):
            bx = batch[0].to(self.device)
            by = batch[1].to(self.device)
            b_cls = batch[2].to(self.device) if len(batch) > 2 else None

            outputs = self.model(bx)
            loss = self.criterion(outputs, by, b_cls)
            reg_out = self._extract_reg_output(outputs)

            bs = bx.shape[0]
            total_loss += float(loss.item()) * bs
            total_samples += bs

            preds_list.append(reg_out.detach().cpu().numpy().reshape(-1))
            trues_list.append(by.detach().cpu().numpy().reshape(-1))

        infer_time = time.perf_counter() - t0
        y_pred = np.concatenate(preds_list, axis=0)
        y_true = np.concatenate(trues_list, axis=0)

        if self.clip_predictions:
            y_pred = np.clip(y_pred, self.clip_min, self.clip_max)

        metrics = compute_regression_metrics(y_true, y_pred)
        avg_loss = total_loss / max(1, total_samples)
        return avg_loss, metrics, y_true, y_pred, infer_time

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_checkpoint: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Full training loop with early stopping on validation RMSE,
        best checkpoint restoration, and JSON logging.
        """
        t_start = time.perf_counter()
        epochs_no_improve = 0
        self.best_val_rmse = float("inf")
        self.best_epoch = 0
        self.history = []

        # WHY: Initialize regression head bias to the training fold target mean (~88.5)
        # if it is currently zero. This starts the network at the trivial mean predictor
        # instead of 0.0, avoiding 15+ wasted epochs climbing from 0 to 88.5 under Huber loss.
        if isinstance(train_loader.dataset, TensorDataset) and len(train_loader.dataset.tensors) >= 2:
            y_tr_tensor = train_loader.dataset.tensors[1]
            if self.model.reg_head.bias is not None and torch.allclose(
                self.model.reg_head.bias.detach().cpu(),
                torch.zeros_like(self.model.reg_head.bias.detach().cpu()),
            ):
                train_mean_val = float(y_tr_tensor.float().mean().item())
                with torch.no_grad():
                    self.model.reg_head.bias.fill_(train_mean_val)

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics, _, _, _ = self.evaluate(val_loader)
            val_rmse = val_metrics["RMSE"]
            current_lr = float(self.optimizer.param_groups[0]["lr"])

            if self.scheduler_type == "cosine":
                self.scheduler.step()
            else:
                self.scheduler.step(val_rmse)

            record = {
                "epoch": epoch,
                "train_loss": round(float(train_loss), 5),
                "val_loss": round(float(val_loss), 5),
                "val_rmse": val_metrics["RMSE"],
                "val_mae": val_metrics["MAE"],
                "val_r2": val_metrics["R2"],
                "val_macro_f1": val_metrics.get("Macro_F1", 0.0),
                "lr": round(current_lr, 7),
            }
            self.history.append(record)

            if val_rmse < self.best_val_rmse:
                self.best_val_rmse = val_rmse
                self.best_epoch = epoch
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if verbose and (epoch == 1 or epoch % 5 == 0 or epochs_no_improve == 0):
                logger.info(
                    "[%s] Epoch %02d/%02d | train_loss=%.4f | val_RMSE=%.4f | val_MAE=%.4f | val_R2=%.4f | lr=%.2e",
                    self.run_name,
                    epoch,
                    self.epochs,
                    train_loss,
                    val_rmse,
                    val_metrics["MAE"],
                    val_metrics["R2"],
                    current_lr,
                )

            if epochs_no_improve >= self.patience:
                if verbose:
                    logger.info(
                        "[%s] Early stopping triggered at epoch %d (best epoch=%d, best val_RMSE=%.4f)",
                        self.run_name,
                        epoch,
                        self.best_epoch,
                        self.best_val_rmse,
                    )
                break

        self.train_time_sec = time.perf_counter() - t_start

        # Restore best weights
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        _, final_val_metrics, _, _, val_infer_time = self.evaluate(val_loader)
        n_params = count_parameters(self.model)

        summary = {
            "run_name": self.run_name,
            "n_parameters": n_params,
            "best_epoch": self.best_epoch,
            "total_epochs_run": len(self.history),
            "best_val_rmse": self.best_val_rmse,
            "val_metrics": final_val_metrics,
            "train_time_sec": round(self.train_time_sec, 4),
            "val_inference_time_sec": round(val_infer_time, 4),
            "hyperparameters": {
                "input_dim": self.model.input_dim,
                "hidden_dims": self.model.hidden_dims,
                "activation": self.model.activation_name,
                "dropout": self.model.dropout_p,
                "batch_norm": self.model.batch_norm,
                "learning_rate": self.lr,
                "weight_decay": self.weight_decay,
                "loss_type": self.loss_type,
                "scheduler": self.scheduler_type,
                "max_grad_norm": self.max_grad_norm,
                "epochs": self.epochs,
                "patience": self.patience,
            },
            "history": self.history,
        }

        if save_checkpoint:
            self.save_checkpoint(summary)
            self.save_training_log(summary)

        return summary

    def save_checkpoint(self, summary: Dict[str, Any]) -> Tuple[Path, Path]:
        """Save best model state_dict (.pt) and companion config JSON next to it."""
        ckpt_path = self.save_dir / f"{self.run_name}.pt"
        cfg_json_path = self.save_dir / f"{self.run_name}_config.json"

        torch.save(
            {
                "run_name": self.run_name,
                "state_dict": self.model.state_dict(),
                "hyperparameters": summary["hyperparameters"],
                "best_epoch": self.best_epoch,
                "best_val_rmse": self.best_val_rmse,
                "n_parameters": summary["n_parameters"],
            },
            ckpt_path,
        )

        config_meta = {k: v for k, v in summary.items() if k != "history"}
        with open(cfg_json_path, "w", encoding="utf-8") as f:
            json.dump(config_meta, f, indent=2)

        logger.info("Saved checkpoint: %s and companion config: %s", ckpt_path, cfg_json_path)
        return ckpt_path, cfg_json_path

    def save_training_log(self, summary: Dict[str, Any]) -> Path:
        """Persist full training log (including epoch-by-epoch learning curves) to JSON."""
        log_path = self.log_dir / f"{self.run_name}_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved training log JSON: %s", log_path)
        return log_path


@torch.no_grad()
def predict_array(
    model: WaterQualityMLP,
    X: np.ndarray,
    batch_size: int = 2048,
    device: Optional[Union[str, torch.device]] = None,
    clip_bounds: Optional[Tuple[float, float]] = (0.0, 100.0),
) -> Tuple[np.ndarray, float]:
    """Run deterministic inference on a NumPy feature array and return (predictions, elapsed_sec)."""
    dev = torch.device("cpu") if device is None else torch.device(device)
    model = model.to(dev)
    model.eval()

    x_tensor = torch.from_numpy(np.asarray(X, dtype=np.float32))
    n = x_tensor.shape[0]
    preds: List[np.ndarray] = []

    t0 = time.perf_counter()
    for i in range(0, n, batch_size):
        bx = x_tensor[i : i + batch_size].to(dev)
        out = model(bx)
        if isinstance(out, tuple):
            out = out[0]
        preds.append(out.detach().cpu().numpy())

    elapsed = time.perf_counter() - t0
    y_pred = np.concatenate(preds, axis=0)
    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred.reshape(-1)
    if clip_bounds is not None:
        y_pred = np.clip(y_pred, clip_bounds[0], clip_bounds[1])
    return y_pred.astype(np.float32), elapsed


def train_single_seed(
    cfg: Dict[str, Any],
    seed: int = 42,
    variant: str = "mlp",
    mock: bool = False,
    export_csv: bool = True,
    save_checkpoint: bool = True,
    verbose: bool = True,
) -> Tuple[WaterQualityMLP, Dict[str, Any]]:
    """
    Train a single seeded MLP run end-to-end, evaluate on validation and test sets,
    and optionally export contract prediction CSVs.
    """
    seed_everything(seed)
    mlp_cfg = cfg.get("mlp", {})
    bs = int(mlp_cfg.get("batch_size", 512))

    splits = load_splits(cfg, scaled=True, mock=mock)
    input_dim = splits["X_train"].shape[1]

    train_ds = TensorDataset(
        torch.from_numpy(splits["X_train"]),
        torch.from_numpy(splits["y_train"]),
    )
    val_ds = TensorDataset(
        torch.from_numpy(splits["X_val"]),
        torch.from_numpy(splits["y_val"]),
    )
    test_ds = TensorDataset(
        torch.from_numpy(splits["X_test"]),
        torch.from_numpy(splits["y_test"]),
    )

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

    model = build_mlp_from_config(cfg, input_dim=input_dim)
    run_name = f"dl_{variant}_seed{seed}"
    trainer = MLPTrainer(model=model, cfg=cfg, run_name=run_name)

    summary = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        save_checkpoint=save_checkpoint,
        verbose=verbose,
    )

    _, val_metrics, y_val_true, y_val_pred, val_infer_sec = trainer.evaluate(val_loader)
    _, test_metrics, y_test_true, y_test_pred, test_infer_sec = trainer.evaluate(test_loader)

    summary["val_metrics"] = val_metrics
    summary["test_metrics"] = test_metrics
    summary["val_inference_time_sec"] = round(val_infer_sec, 4)
    summary["test_inference_time_sec"] = round(test_infer_sec, 4)

    if save_checkpoint:
        trainer.save_checkpoint(summary)
        trainer.save_training_log(summary)

    if export_csv:
        pred_dir = DL_ROOT / "results" / "predictions"
        export_dl_predictions_csv(
            pred_dir / f"dl_{variant}_seed{seed}_val.csv",
            splits["row_id_val"],
            y_val_true,
            y_val_pred,
        )
        export_dl_predictions_csv(
            pred_dir / f"dl_{variant}_seed{seed}_test.csv",
            splits["row_id_test"],
            y_test_true,
            y_test_pred,
        )

    return model, summary


def overfit_tiny_batch(
    cfg: Dict[str, Any],
    batch_size: int = 32,
    steps: int = 300,
    lr: float = 1e-2,
    mock: bool = False,
) -> Dict[str, float]:
    """
    Sanity Check 1 (Task C): Overfit a single 32-sample mini-batch to near-zero loss.
    WHY: Verifies gradient flow, weight updates, and representation capacity before full training.
    """
    seed_everything(42)
    splits = load_splits(cfg, scaled=True, mock=mock)
    bx = torch.from_numpy(splits["X_train"][:batch_size])
    by = torch.from_numpy(splits["y_train"][:batch_size])

    model = WaterQualityMLP(
        input_dim=bx.shape[1],
        hidden_dims=[128, 64],
        activation="relu",
        dropout=0.0,
        batch_norm=False,
    )
    with torch.no_grad():
        model.reg_head.bias.fill_(float(by.mean().item()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    loss_fn = nn.MSELoss()

    model.train()
    initial_loss = float(loss_fn(model(bx), by).item())
    final_loss = initial_loss

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        preds = model(bx)
        loss = loss_fn(preds, by)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        final_preds = model(bx).numpy()
    r2 = float(r2_score(by.numpy(), final_preds))
    rmse = float(np.sqrt(mean_squared_error(by.numpy(), final_preds)))

    return {
        "batch_size": batch_size,
        "steps": steps,
        "initial_mse_loss": round(initial_loss, 5),
        "final_mse_loss": round(final_loss, 5),
        "final_rmse": round(rmse, 5),
        "final_r2": round(r2, 5),
    }


def run_shuffled_target_test(
    cfg: Dict[str, Any],
    epochs: int = 8,
    subsample_n: int = 10000,
    mock: bool = False,
) -> Dict[str, float]:
    """
    Sanity Check 2 (Task C): Shuffled-target permutation test.
    WHY: When training targets y_train are randomly shuffled, any legitimate model MUST achieve
    R2 ~ 0.0 on the unshuffled test set. Positive test R2 would indicate feature-target leakage.
    """
    seed_everything(42)
    splits = load_splits(cfg, scaled=True, mock=mock)
    rng = np.random.RandomState(42)

    n_tr = min(len(splits["X_train"]), subsample_n)
    x_tr = splits["X_train"][:n_tr]
    y_tr_shuffled = rng.permutation(splits["y_train"][:n_tr]).copy()

    x_val = splits["X_val"]
    y_val = splits["y_val"]
    x_test = splits["X_test"]
    y_test = splits["y_test"]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr_shuffled)),
        batch_size=256,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=512,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
        batch_size=512,
        shuffle=False,
    )

    cfg_shuf = copy.deepcopy(cfg)
    cfg_shuf.setdefault("mlp", {})["epochs"] = epochs
    cfg_shuf["mlp"]["early_stopping_patience"] = epochs

    model = build_mlp_from_config(cfg_shuf, input_dim=x_tr.shape[1])
    trainer = MLPTrainer(model=model, cfg=cfg_shuf, run_name="sanity_shuffled_target")
    trainer.fit(train_loader, val_loader, save_checkpoint=False, verbose=False)

    _, val_m, _, _, _ = trainer.evaluate(val_loader)
    _, test_m, _, _, _ = trainer.evaluate(test_loader)

    return {
        "val_R2": val_m["R2"],
        "val_RMSE": val_m["RMSE"],
        "test_R2": test_m["R2"],
        "test_RMSE": test_m["RMSE"],
        "test_MAE": test_m["MAE"],
    }


def compute_trivial_baselines(
    cfg: Dict[str, Any],
    mock: bool = False,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Sanity Check 3 (Task C): Trivial baselines (Mean Predictor and Ridge Regression)
    evaluated on the exact scaled train/val/test splits for reference context.
    """
    from sklearn.linear_model import Ridge

    splits = load_splits(cfg, scaled=True, mock=mock)
    x_tr, y_tr = splits["X_train"], splits["y_train"]
    x_val, y_val = splits["X_val"], splits["y_val"]
    x_test, y_test = splits["X_test"], splits["y_test"]

    # 1. Train-Mean Predictor
    y_mean = float(np.mean(y_tr))
    mean_val_preds = np.full_like(y_val, fill_value=y_mean)
    mean_test_preds = np.full_like(y_test, fill_value=y_mean)

    # 2. Ridge Regression (L2 linear baseline)
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(x_tr, y_tr)
    ridge_val_preds = np.clip(ridge.predict(x_val), 0.0, 100.0)
    ridge_test_preds = np.clip(ridge.predict(x_test), 0.0, 100.0)

    return {
        "mean_predictor": {
            "val": compute_regression_metrics(y_val, mean_val_preds),
            "test": compute_regression_metrics(y_test, mean_test_preds),
        },
        "ridge_regression": {
            "val": compute_regression_metrics(y_val, ridge_val_preds),
            "test": compute_regression_metrics(y_test, ridge_test_preds),
        },
    }


def plot_learning_curves(
    history: List[Dict[str, float]],
    out_path: Optional[Union[str, Path]] = None,
    title: str = "Baseline WaterQualityMLP Learning Curves",
) -> Path:
    """Plot and save epoch-by-epoch training/validation loss and validation RMSE curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_path is None:
        out_path = DL_ROOT / "results" / "figures" / "mlp_baseline_learning_curves.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_rmse = [h["val_rmse"] for h in history]
    val_mae = [h["val_mae"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)

    axes[0].plot(epochs, train_loss, label="Train Loss (Huber)", color="#1F6FD1", linewidth=2)
    axes[0].plot(epochs, val_loss, label="Val Loss (Huber)", color="#E05A47", linewidth=2, linestyle="--")
    axes[0].set_title(f"{title} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Huber Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, val_rmse, label="Val RMSE", color="#1F6FD1", linewidth=2)
    axes[1].plot(epochs, val_mae, label="Val MAE", color="#2E9CCA", linewidth=2, linestyle="-.")
    best_idx = int(np.argmin(val_rmse))
    axes[1].scatter(
        [epochs[best_idx]],
        [val_rmse[best_idx]],
        color="#D9381E",
        zorder=5,
        label=f"Best Val RMSE ({val_rmse[best_idx]:.4f} @ Ep {epochs[best_idx]})",
    )
    axes[1].set_title(f"{title} — Validation Error")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Error (CCME WQI Units)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved learning curves figure: %s", out_path)
    return out_path


def run_sanity_suite(
    cfg: Optional[Dict[str, Any]] = None,
    mock: bool = False,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Execute the complete Task C Sanity Suite BEFORE hyperparameter tuning:
    1. Overfit a 32-sample batch to near-zero loss.
    2. Shuffled-target permutation check (Test R2 ~ 0).
    3. Trivial baselines (Mean predictor & Ridge regression).
    4. First baseline MLP run + learning curve visualization.
    """
    if cfg is None:
        cfg = load_config()

    logger.info("1/4 Running 32-sample tiny-batch overfit check...")
    overfit_res = overfit_tiny_batch(cfg, batch_size=32, steps=300, lr=1e-2, mock=mock)
    logger.info("Tiny-batch overfit result: %s", overfit_res)

    logger.info("2/4 Running shuffled-target leakage check...")
    shuffled_res = run_shuffled_target_test(cfg, epochs=8, subsample_n=10000, mock=mock)
    logger.info("Shuffled-target check result: %s", shuffled_res)

    logger.info("3/4 Computing trivial baselines (Mean predictor, Ridge regression)...")
    baselines_res = compute_trivial_baselines(cfg, mock=mock)
    logger.info("Trivial baselines: %s", baselines_res)

    logger.info("4/4 Running initial baseline MLP training + learning curves...")
    _, mlp_summary = train_single_seed(
        cfg,
        seed=cfg.get("mlp", {}).get("seed", 42),
        variant="mlp_initial",
        mock=mock,
        export_csv=False,
        save_checkpoint=save_artifacts,
        verbose=True,
    )

    fig_path = None
    if save_artifacts:
        fig_path = str(
            plot_learning_curves(
                mlp_summary["history"],
                out_path=DL_ROOT / "results" / "figures" / "mlp_baseline_learning_curves.png",
            )
        )

    report = {
        "tiny_batch_overfit": overfit_res,
        "shuffled_target_test": shuffled_res,
        "trivial_baselines": baselines_res,
        "initial_mlp_run": {
            "n_parameters": mlp_summary["n_parameters"],
            "best_epoch": mlp_summary["best_epoch"],
            "train_time_sec": mlp_summary["train_time_sec"],
            "val_metrics": mlp_summary["val_metrics"],
            "test_metrics": mlp_summary["test_metrics"],
            "learning_curves_figure": fig_path,
        },
    }

    if save_artifacts:
        out_json = DL_ROOT / "results" / "tables" / "sanity_suite_summary.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Saved sanity suite summary JSON: %s", out_json)

    return report


if __name__ == "__main__":
    cfg_loaded = load_config()
    run_sanity_suite(cfg_loaded, mock=False, save_artifacts=True)
