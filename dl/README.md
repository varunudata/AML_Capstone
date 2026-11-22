# DL Phase: Water Quality Prediction via Deep Neural Networks

This directory contains the Deep Learning (DL) half of the Water Quality Prediction Capstone, expanding on the frozen Applied Machine Learning (AML) tree-based baselines.

---

## 👥 Team Responsibilities & Ownership

| Member | Role | Assigned Files | Status |
|---|---|---|---|
| **Member 1** | **Data Foundation & AML Bridge** | `src/data.py`, `src/utils.py`, `src/aml_baselines.py`, `notebooks/01_data_and_aml_bridge.ipynb`, `configs/config.yaml` | **COMPLETE (Phase 1)** |
| **Member 2** | **MLP Training & Uncertainty** | `src/models.py`, `src/train.py`, `src/tune.py`, `src/uncertainty.py`, `notebooks/02_mlp_training.ipynb` | **COMPLETE (Phase 2)** |
| **Member 3** | **Evaluation & Comparison Report** | `src/evaluate.py`, `src/calibration.py`, `src/explain.py`, `notebooks/03_comparison_report.ipynb` | **COMPLETE (Phase 3)** |

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

---

## 🧠 Phase 2 Summary: Regularized MLP Challenger, Tuning & Calibrated Uncertainty (Member 2)

### 1. Pre-Tuning Sanity Suite (`src/train.py`)
- **32-Sample Mini-Batch Overfit:** Driven from initial $\text{MSE} = 177.21$ to $\text{MSE} = 0.0000$ ($R^2 = 1.0000$) in 300 steps.
- **Shuffled-Target Permutation Check:** Randomly permuting `y_train` yields Test $R^2 = 0.0129$ ($\text{RMSE} = 12.65$), confirming zero feature-target leakage.
- **Trivial Baselines on Exact Scaled Split:**
  - Train-Mean Predictor: Test $\text{RMSE} = 12.8823$, $\text{MAE} = 9.9822$, $R^2 = -0.0235$.
  - Scaled Ridge Regression (21 features): Test $\text{RMSE} = 3.2880$, $\text{MAE} = 2.2614$, $R^2 = 0.9333$.
- **Initial Baseline MLP (`[128, 64, 32]`, 13,633 params):** Val $\text{RMSE} = 0.7404$ ($R^2 = 0.9970$), Test $\text{RMSE} = 0.6446$ ($R^2 = 0.9974$).

### 2. Optuna Tuning (Matched 15-Trial AML Budget) & 20-Configuration Ablation (`src/tune.py`)
- **Tuning Protocol:** 15 Optuna TPE trials optimizing **Validation RMSE exclusively** (test split strictly isolated).
- **Key Architectural Ablation Findings (`results/tables/ablation_results.md`):**
  - **BatchNorm Impact:** Enabling `BatchNorm1d` cuts validation RMSE in half from $1.7225 \to 0.8138$.
  - **Depth Impact:** Scaling from 1 layer (`[128]`, Val $\text{RMSE} = 1.5438$) to 3–4 layers (`[128, 64, 32]`, Val $\text{RMSE} = 0.9402$; `[256, 128, 64, 32]`, Val $\text{RMSE} = 0.8891$) captures multi-parameter threshold curvature.
  - **Loss Function:** Huber loss reduces validation MAE from $0.3548$ (MSE) to $0.3233$ by bounding gradient sensitivity to tail outliers and the 41.39% censored boundary at $100.0$.

### 3. Final 5-Seed Evaluation & Uncertainty Results (`src/uncertainty.py`)

Evaluated across 5 random seeds (`[42, 43, 44, 45, 46]`) on $50,000$ unseen spatial test observations ($0\%$ station overlap):

| Variant | Seeds | Params | Train Time (s) | Infer Time (s) | Val RMSE | Val MAE | Val $R^2$ | Test RMSE | Test MAE | Test $R^2$ | Test Macro-F1 | 90% PICP Coverage | 90% MPIW Width |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`dl_mlp` (5-Seed Mean ± Std)** | 42..46 | 13,633 | 13.07 | 0.0095 | 0.6522 | 0.3080 | 0.9976 | **0.5707 ± 0.0199** | **0.2715** | **0.9980** | 0.9512 | — | — |
| **`dl_mlp_mcdropout` ($T=50$ Passes)** | 42..46 | 13,633 | 13.07 | 1.9240 | 0.6873 | 0.3391 | 0.9974 | 0.6047 ± 0.0169 | 0.2986 | 0.9978 | 0.9478 | 0.9862 | 2.6331 |
| **`dl_mlp_ensemble` (5-Seed Deep Ensemble)** | 42..46 | 68,165 | 65.36 | 0.0474 | **0.6180** | **0.2855** | **0.9979** | **0.5381** | **0.2528** | **0.9982** | **0.9548** | **0.9043** | **1.0560** |
| **`dl_mlp_quantile` (Pinball $q_{05}/q_{50}/q_{95}$)** | 42..46 | 13,699 | 12.67 | 0.0064 | 0.6972 | 0.3539 | 0.9973 | 0.6148 ± 0.0386 | 0.3137 | 0.9976 | 0.9479 | 0.9912 | 3.4911 |
| **`dl_mlp_leakaware` (17 Features Ablated)** | 42 | 13,121 | 8.40 | 0.0076 | 0.8119 | 0.3550 | 0.9963 | 0.7124 | 0.2803 | 0.9969 | 0.9251 | — | — |

### 4. Exported Phase 2 Contract Artifacts for Member 3
- **Prediction CSVs (`results/predictions/`):**
  - `dl_mlp_seed{42..46}_{val,test}.csv`
  - `dl_mlp_mcdropout_seed{42..46}_{val,test}.csv` (includes `y_std`, `q05`, `q95`)
  - `dl_mlp_ensemble_seed{42..46}_{val,test}.csv` (includes `y_std`, `q05`, `q95`)
  - `dl_mlp_quantile_seed{42..46}_{val,test}.csv` (includes `y_std`, `q05`, `q95`)
  - `dl_mlp_leakaware_seed42_{val,test}.csv` (and alias `dl_mlp_leakage_aware_seed42_{val,test}.csv`)
- **Checkpoints & Configs (`results/models/`):** Regenerated via `python dl/src/uncertainty.py` (`.pt` files excluded from git via `.gitignore`, companion `*_config.json` saved alongside).
- **Tables & Figures:** `results/tables/phase2_final_summary.csv`, `results/tables/ablation_results.md`, `results/figures/mlp_baseline_learning_curves.png`, `results/figures/mlp_uncertainty_calibration.png`.
- **Notebook & Viva Defense:** `dl/notebooks/02_mlp_training.ipynb` and `dl/reports/viva_notes_phase2.md`.

## Phase 3 / Final report

- [Final IEEE Report (PDF)](reports/ieee_report/DL_IEEE_Report.pdf) · [IEEE LaTeX Source (`main.tex`)](reports/ieee_report/main.tex) · [DL Literature Review](reports/literature_review_dl.md)
- [Comparison notebook](notebooks/03_comparison_report.ipynb) · [Viva notes](reports/viva_notes_phase3.md)
- [Test comparison](results/tables/phase3_comparison.md) · [Site-block bootstrap](results/tables/phase3_paired_bootstrap.md) · [Interval coverage](results/tables/phase3_intervals.md)

The full reproduction uses the team's `Member3_Required_Artifacts.zip` as a read-only input. Extract it outside the clone. The required files are `data/processed/clean.parquet`, `data/processed/splits/site_held_out_split.npz`, `models/xgboost_production_model.joblib` (artifact size only), and `dl/results/models/dl_mlp_seed42.pt`. The repository supplies the frozen prediction CSVs, preprocessor, manifest and AML metrics.

From the repository root, using the shell that has the Python environment active:

```bash
python -m pip install -r dl/requirements.txt
dl/run_all.sh --artifact-root /path/to/extracted/Member3_Required_Artifacts
python -m pytest dl/tests/test_evaluate.py dl/tests/test_compare.py dl/tests/test_calibration.py dl/tests/test_explain.py -q
```

`run_all.sh` verifies the original AML split and reproduces the Phase 3 tables, figures and reports. It imports AML point metrics and uses the saved predictions for paired residuals and interval calibration. All model training and original Stage 6 evaluation stay frozen. The generated test-site table averages each model's per-seed metrics within site and records the number of runs. The PDF and plots may render slightly differently with different font libraries.

To run the notebook with the same inputs, set `AML_CAPSTONE_ARTIFACTS` to the extracted folder before launching Jupyter. Notebook cells write generated files to a temporary directory. If only the Git checkout is available, `python -m dl.src.reporting --output /path/to/scratch` builds the export-based subset with exploratory row bootstrap. The complete verified results are committed with the report.
