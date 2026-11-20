# DL Phase: Water Quality Prediction via Deep Neural Networks

This directory contains the Deep Learning (DL) half of the Water Quality Prediction Capstone, expanding on the frozen Applied Machine Learning (AML) tree-based baselines.

---

## 👥 Team Responsibilities & Ownership

| Member | Role | Assigned Files | Status |
|---|---|---|---|
| **Member 1** | **Data Foundation & AML Bridge** | `src/data.py`, `src/utils.py`, `src/aml_baselines.py`, `notebooks/01_data_and_aml_bridge.ipynb`, `configs/config.yaml` | **COMPLETE (Phase 1)** |
| **Member 2** | **MLP Training & Uncertainty** | `src/models.py`, `src/train.py`, `src/tune.py`, `src/uncertainty.py`, `notebooks/02_mlp_training.ipynb` | Ready for Handoff |
| **Member 3** | **Evaluation & Comparison Report** | `src/evaluate.py`, `src/calibration.py`, `src/explain.py`, `notebooks/03_comparison_report.ipynb` | Ready for Handoff |

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
│   └── processed/                      # preprocessor.joblib and split_indices.json
├── results/
│   ├── aml_baseline_metrics.json       # Exported baseline benchmark metrics (M1)
│   ├── predictions/                    # Standardized CSVs: {family}_{variant}_seed{N}_{split}.csv
│   ├── models/                         # PyTorch model checkpoints (.pt)
│   ├── logs/                           # Training & evaluation logs
│   ├── tuning/                         # Optuna sqlite databases & study summaries
│   ├── figures/                        # High-resolution benchmark figures
│   └── tables/                         # Formatted markdown / LaTeX tables
├── reports/
│   └── viva_notes_phase1.md            # Technical viva examination Q&A (M1)
├── tests/
│   ├── test_data.py                    # Unit tests for data contract, leakage, and preprocessor
│   ├── test_mock.py                    # Unit tests for mock data contract
│   ├── test_splits.py                  # Unit tests for spatial split fidelity
│   └── test_baselines.py               # Unit tests for AML baseline bridge & prediction CSVs
├── requirements.txt                    # Pinned dependencies
└── README.md
```

---

## 🏆 Phase 1 Summary: Data Foundation & AML Baseline Bridge

### 1. Target Definition & Censoring
- **Target:** Single continuous Canadian Council of Ministers of the Environment Water Quality Index score (`CCME_Values` $\in [31.3041, 100.0]$).
- **Upper Ceiling Censoring:** $41.39\%$ of all observations reach the regulatory ceiling of exactly $100.0$.
- **Discrete Regulatory Mapping:** Monotonically mapped post-hoc into 5 discrete tiers (`CCME_WQI`): Excellent ($\ge 95$), Good ($\ge 80$), Fair ($\ge 65$), Marginal ($\ge 45$), Poor ($< 45$).

### 2. Spatial Split Fidelity (Zero Station Leakage)
- Split protocol enforces strict **Group-Stratified Site-Held-Out** validation on monitoring station identifier (`Area`).
- Training, validation, and test splits have **$0\%$ station overlap**, preventing station identity leakage.
- Split indices and sample partitions are persisted in `data/processed/split_indices.json`.

### 3. Leak-Safe Preprocessing Pipeline
- **Detection Limits:** Values formatted as `"<0.5"` or `"<0.01"` are parsed into half-detection-limit floats ($L/2$) with boolean flags.
- **Impossible Values:** Negative nutrient concentrations, $\text{pH} < 0$ or $> 14$, $\text{DO} \le 0$ or $> 20\,\text{mg/L}$, and $\text{Temperature} < -2^\circ\text{C}$ or $> 40^\circ\text{C}$ are flagged and set to `NaN`.
- **Train-Only Imputation & Scaling:** `SimpleImputer(strategy="median")` and `StandardScaler` are fitted **strictly on `X_train`** and frozen into `data/processed/preprocessor.joblib`.
- **Dual Pipeline:** `scaled=True` provides standardized $\mathcal{N}(0, 1)$ inputs for neural networks; `scaled=False` provides unscaled features for tree models.

### 4. Formula Leakage Audit Findings
- **Raw Linear Baseline:** An Ordinary Least Squares Ridge model on 8 raw chemistry inputs achieves $R^2 = 0.4679$ ($\text{RMSE} = 9.29$). Raw continuous chemistry does **not** trivially leak the CCME WQI without threshold logic.
- **Engineered Limnological Features:** Adding 13 domain-engineered features (DO deficit, stoichiometric $N:P$ ratio, pH buffering) raises linear $R^2$ to $0.9177$ ($\text{RMSE} = 3.65$).
- **Non-Linear Tree Baseline:** Tuned tree ensembles capture the piecewise step functions at Canadian threshold boundaries, reaching $R^2 = 0.9992$ ($\text{RMSE} = 0.3662$).
- **Leakage-Aware Ablation:** Dropping direct stoichiometric formulas (`do_deficit`, `log_np_ratio`, `nitrate_to_n_ratio`, `ph_dist_buffer`) still produces $R^2 = 0.9991$ and $\text{RMSE} = 0.3791$ for XGBoost on unseen stations, proving strong underlying limnological signal.

### 5. AML Baseline Bridge Benchmark Results

Evaluated on $50,000$ unseen spatial test observations ($0\%$ station overlap):

| Model Architecture | Val $R^2$ | Val RMSE | Val MAE | Test $R^2$ | Test RMSE | Test MAE | Secondary Macro-F1 | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| Linear Ridge (8 Raw Chemistry) | 0.4721 | 9.7712 | 6.8421 | 0.4679 | 9.2884 | 6.6208 | 0.7241 | Linear Baseline |
| Linear Ridge (21 Engineered) | 0.9234 | 3.7144 | 2.6288 | 0.9177 | 3.6523 | 2.6244 | 0.8912 | Engineered Linear |
| **Tuned XGBoost (Boundary-Clipped)** | **0.9990** | **0.4192** | **0.1309** | **0.9992** | **0.3662** | **0.1024** | **0.9660** | **AML Production Champion** |
| Random Forest (100 Trees) | 0.9990 | 0.4326 | 0.1290 | 0.9993 | 0.3464 | 0.0964 | 0.9732 | AML Ensemble Baseline |
| XGBoost (Leakage-Aware Ablated) | 0.9989 | 0.4387 | 0.1543 | 0.9991 | 0.3791 | 0.1216 | 0.9655 | Controlled Ablation |

Exported prediction CSVs:
- `results/predictions/aml_xgboost_seed42_{val,test}.csv`
- `results/predictions/aml_random_forest_seed42_{val,test}.csv`
- `results/predictions/aml_xgboost_leakage_aware_seed42_{val,test}.csv`

---

## 📑 Data Contract (M1 ➔ M2, M3)

In `src/data.py`:
- `load_splits(cfg, scaled=True, mock=False)`:
  - Returns dictionary with keys: `X_train`, `X_val`, `X_test` (float32 numpy arrays), `y_train`, `y_val`, `y_test` (float32), `feature_names` (list), `row_id_train`, `row_id_val`, `row_id_test` (int64), `meta_test` (DataFrame with site/time columns), and `y_class_train/val/test` (CCME discrete regulatory classes).
  - Passing `mock=True` returns synthetic data adhering to the identical interface so teammates can develop without waiting for large parquet files.
- `get_dataloaders(cfg, batch_size=256, scaled=True)`:
  - Returns PyTorch DataLoaders (`train_loader`, `val_loader`, `test_loader`).

---

## 🛠️ Verification Commands

Run the full automated pytest suite (18 assertions):
```bash
PYTHONPATH=dl pytest dl/tests/ -v
```

Execute Notebook 01:
```bash
jupyter notebook dl/notebooks/01_data_and_aml_bridge.ipynb
```

Review Viva Defense Preparation:
- [`dl/reports/viva_notes_phase1.md`](file:///Users/musthyalasadhvik/Desktop/projects/aml-capstone/AML_Capstone/dl/reports/viva_notes_phase1.md)
