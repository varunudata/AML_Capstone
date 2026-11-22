# Water Quality Prediction Capstone: Applied Machine Learning (`AML/`) & Deep Learning (`DL/`)

This repository contains the complete, two-phase capstone investigation predicting the continuous **Canadian Council of Ministers of the Environment Water Quality Index (`CCME_Values` $\in [31.3041, 100.0]$)** across a global ambient surface water monitoring cohort ($1,988,629$ cleaned records across $46,575$ monitoring stations) under a strict **zero-leakage site-held-out spatial split (`Area`, $0\%$ station overlap)**.

---

## 📂 Repository Structure

```
AML_Capstone/
├── AML/                            # Phase 1: Applied Machine Learning (Ridge, Random Forest, Tuned XGBoost)
│   ├── configs/                    # Declarative data cleaning & model hyperparameter configs
│   ├── notebooks/                  # End-to-end AML notebooks (01_analysis .. 05_models)
│   ├── outputs/                    # Stage reports, JSON metrics, and 13 publication figures
│   ├── report/                     # Phase 1 IEEE LaTeX report (main.tex, references.bib, IEEEtran.cls)
│   ├── src/                        # Modular AML pipeline (cleaning, eda, features, splits, models)
│   ├── tests/                      # Automated pytest suite for cleaning rules & zero spatial leakage
│   ├── requirements.txt            # Pinned AML dependencies
│   └── README.md                   # Complete Phase 1 AML documentation
│
├── DL/                             # Phase 2: Deep Learning (Pyramidal MLP, Optuna, Uncertainty & SHAP)
│   ├── configs/config.yaml         # Shared DL configuration across M1, M2, and M3
│   ├── data/processed/             # Frozen train-only preprocessor.joblib and split_indices.json
│   ├── notebooks/                  # 01_data_and_aml_bridge, 02_mlp_training, 03_comparison_report
│   ├── reports/                    # Two-column IEEE report (DL_IEEE_Report.pdf + main.tex) & DL literature review
│   ├── results/                    # Standardized prediction CSVs, ablation tables, and diagnostic figures
│   ├── src/                        # Modular PyTorch DL pipeline (data, models, train, tune, uncertainty, evaluate, calibration, explain)
│   ├── tests/                      # 95 automated pytest unit and contract tests
│   ├── requirements.txt            # Pinned DL dependencies
│   └── README.md                   # Complete Phase 2 DL documentation
│
├── .gitignore
└── README.md                       # Master repository overview
```

---

## 🏆 Side-by-Side Benchmark Summary ($50,000$ Unseen Spatial Test Observations)

| Phase & Model Architecture | Runs | Params | Test MAE | Test RMSE | Test $R^2$ | Macro-F1 | 90% PICP Coverage | 90% MPIW Width |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AML:** Linear Ridge (8 Raw Chemistry) | 1 | 9 | 6.6208 | 9.2884 | 0.4679 | 0.7241 | — | — |
| **AML:** Linear Ridge (21 Engineered) | 1 | 22 | 2.2614 | 3.2880 | 0.9333 | 0.7894 | — | — |
| **AML:** Random Forest (100 Trees, Frozen) | 1 | — | **0.0964** | **0.3464** | **0.9993** | **0.9732** | — | — |
| **AML:** Tuned XGBoost (Conformal Wrapper) | 1 | — | **0.1024** | **0.3662** | **0.9992** | **0.9660** | **92.20%** | **0.466** |
| **DL:** Point MLP (`[128, 64, 32]`, ReLU+BN) | 5 | 13,633 | $0.2715 \pm 0.0266$ | $0.5707 \pm 0.0199$ | $0.9980 \pm 0.0001$ | $0.9512 \pm 0.0059$ | — | — |
| **DL:** MC Dropout ($T=50$ Stochastic Passes) | 5 | 13,633 | $0.2986 \pm 0.0219$ | $0.6047 \pm 0.0168$ | $0.9977 \pm 0.0001$ | $0.9478 \pm 0.0072$ | $98.61\% \pm 0.08\%$ | $2.633 \pm 0.079$ |
| **DL:** **5-Seed Deep Ensemble (DL Champion)** | 1 (5x) | 68,165 | **0.2528** | **0.5381** | **0.9982** | **0.9548** | **90.46%** | **1.056** |
| **DL:** Pinball Quantile MLP ($q_{05}/q_{50}/q_{95}$) | 5 | 13,699 | $0.3137 \pm 0.0233$ | $0.6149 \pm 0.0386$ | $0.9977 \pm 0.0003$ | $0.9479 \pm 0.0083$ | $99.12\% \pm 0.12\%$ | $3.491 \pm 0.051$ |

---

## 🛠️ Verification Commands

Run the **Applied Machine Learning (`AML/`)** test suite:
```bash
PYTHONPATH=AML pytest AML/tests/ -v
```

Run the **Deep Learning (`DL/`)** test suite (95 unit & contract assertions):
```bash
PYTHONPATH=DL pytest DL/tests/ -v
```
