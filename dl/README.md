# DL Phase: Water Quality Prediction via Deep Neural Networks

This directory contains the Deep Learning (DL) half of the Water Quality Prediction Capstone, expanding on the frozen Applied Machine Learning (AML) tree-based baselines.

---

## 👥 Team Responsibilities & Ownership

| Member | Role | Assigned Files | Status |
|---|---|---|---|
| **Member 1** | **Data Foundation & AML Bridge** | `src/data.py`, `src/utils.py`, `src/aml_baselines.py`, `notebooks/01_data_and_aml_bridge.ipynb`, `configs/config.yaml` | **Active (Phase 1)** |
| **Member 2** | **MLP Training & Uncertainty** | `src/models.py`, `src/train.py`, `src/tune.py`, `src/uncertainty.py`, `notebooks/02_mlp_training.ipynb` | Pending M1 Hand-off |
| **Member 3** | **Evaluation & Comparison Report** | `src/evaluate.py`, `src/calibration.py`, `src/explain.py`, `notebooks/03_comparison_report.ipynb` | Pending M2 Hand-off |

> **Contract Rule:** Each member edits ONLY their assigned files. Shared files (`config.yaml`, `requirements.txt`, `README.md`) are modified only within clearly designated member sections.

---

## 📋 Directory Layout

```
dl/
├── configs/
│   └── config.yaml                     # Shared configuration across all 3 phases
├── src/
│   ├── __init__.py
│   ├── utils.py                        # Logging, deterministic seeding, config loader (M1)
│   ├── data.py                         # Leak-safe data loading & PyTorch DataLoaders (M1)
│   ├── aml_baselines.py                # Bridge exporting AML predictions & metrics (M1)
│   ├── models.py                       # MLP architectures (M2)
│   ├── train.py                        # PyTorch training loops & checkpointing (M2)
│   ├── tune.py                         # Optuna hyperparameter optimization (M2)
│   ├── uncertainty.py                  # Monte Carlo Dropout & Deep Ensembles (M2)
│   ├── evaluate.py                     # Side-by-side DL vs AML benchmarking (M3)
│   ├── calibration.py                  # Uncertainty coverage & reliability diagrams (M3)
│   └── explain.py                      # SHAP & Integrated Gradients (M3)
├── notebooks/
│   ├── 01_data_and_aml_bridge.ipynb    # EDA, leakage audit, AML baseline validation (M1)
│   ├── 02_mlp_training.ipynb           # Training curves, tuning diagnostics (M2)
│   └── 03_comparison_report.ipynb      # Final comparison & viva report (M3)
├── data/
│   └── processed/                      # Preprocessor artifacts and split indices
├── results/
│   ├── predictions/                    # Standardized CSVs: {family}_{variant}_seed{N}_{split}.csv
│   ├── models/                         # PyTorch model checkpoints (.pt)
│   ├── logs/                           # Training & evaluation logs
│   ├── tuning/                         # Optuna sqlite databases & study summaries
│   ├── figures/                        # High-resolution benchmark figures
│   └── tables/                         # Formatted markdown / LaTeX tables
├── reports/                            # Technical reports and viva defense notes
├── tests/
│   ├── test_data.py                    # Unit tests for data contract, leakage, and preprocessor
│   └── test_mock.py                    # Unit tests for mock data contract
├── requirements.txt                    # Pinned dependencies
└── README.md
```

---

## 📑 Data Contract (M1 ➔ M2, M3)

In `src/data.py`:
- `load_splits(cfg, scaled=True, mock=False)`:
  - Returns dictionary with keys: `X_train`, `X_val`, `X_test` (float32 numpy arrays), `y_train`, `y_val`, `y_test` (float32), `feature_names` (list), `row_id_train`, `row_id_val`, `row_id_test` (int64), `meta_test` (DataFrame with site/time columns), and `y_class_train/val/test` (CCME discrete regulatory classes).
  - Passing `mock=True` returns synthetic data adhering to the identical interface so teammates can develop without waiting for large parquet files.
- `get_dataloaders(cfg, batch_size)`:
  - Returns PyTorch DataLoaders (`train_loader`, `val_loader`, `test_loader`).

---

## 📑 Predictions Contract (All Members)

All predictions are saved as CSV files under `results/predictions/`:
- Path format: `{family}_{variant}_seed{N}_{split}.csv`
  - `family` ∈ `{aml, dl}`
  - `variant` ∈ `{xgboost, random_forest, mlp_baseline, mlp_tuned, mlp_mc_dropout, ...}`
  - `split` ∈ `{val, test}`
- Columns: `row_id`, `y_true`, `y_pred` (optional uncertainty: `y_std`, `q05`, `q95`, `class_true`, `class_pred`).

---

## 📑 Metrics Contract

The baseline metrics are stored at `results/aml_baseline_metrics.json`:
```json
{
  "xgboost": {
    "val": {"MAE": 0.1303, "RMSE": 0.4659, "R2": 0.9988},
    "test": {"MAE": 0.1012, "RMSE": 0.3496, "R2": 0.9992}
  },
  "random_forest": {
    "val": {"MAE": 0.1278, "RMSE": 0.4099, "R2": 0.9991},
    "test": {"MAE": 0.0955, "RMSE": 0.3310, "R2": 0.9993}
  },
  "meta": {
    "target": "CCME_Values",
    "split_type": "site_held_out (Area)",
    "n_train": 250000,
    "n_val": 50000,
    "n_test": 50000,
    "tuning_budget": "15 Optuna GroupKFold trials"
  }
}
```

---

## 🛠️ Quickstart

Run unit tests for mock data:
```bash
PYTHONPATH=dl pytest dl/tests/test_mock.py -v
```
