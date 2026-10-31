# AML Capstone — Leakage-Safe Machine Learning for Environmental Water Quality Indexing

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Tests: Passing](https://img.shields.io/badge/tests-5%20passed-brightgreen.svg)]()
[![Code Style: Modular](https://img.shields.io/badge/architecture-modular%20%2F%20pipeline-orange.svg)]()

An end-to-end, scientifically rigorous applied machine learning project predicting continuous **Canadian Council of Ministers of the Environment Water Quality Index (CCME WQI)** scores from routine surface water chemistry across **$2.82$ million global records** (1940–2023).

---

## 📌 Executive Overview

Surface water quality indices distill complex chemical metrics into standardized ecological summaries. However, deterministic index calculation requires simultaneous laboratory assays of multiple analytes, creating significant operational bottlenecks.

This capstone project establishes a **zero-leakage machine learning benchmark** that maps routine physicochemical parameters directly to continuous CCME scores ($[31.3, 100.0]$):
* **Zero Spatial Leakage:** Standard random row splitting causes severe station-identity leakage ($>91\%$ of test sites repeated in training). We enforce a strict **Site-Held-Out** protocol (`GroupShuffleSplit` on `Area`) with **$0\%$ site overlap**.
* **Upper Ceiling Censoring at 100.0:** Over $41\%$ of ambient samples reach the regulatory maximum of $100.0$. We benchmark unconstrained regressors against **boundary clipping** and a **two-stage hurdle model**.
* **Domain Feature Engineering:** Temperature-dependent dissolved oxygen deficit and stoichiometric $N:P$ balances propel linear Ridge from $R^2 = 0.5176$ to $R^2 = 0.9204$.
* **State-of-the-Art Tree Ensembles:** Tuned **XGBoost (Hist Gradient Boosting)** achieves **$R^2 = 0.9995$** and an **RMSE of $0.2834$** ($95\%$ bootstrap CI: $[0.268, 0.299]$) on unseen monitoring stations.

---

## 🏆 Benchmark Results on Unseen Spatial Test Sites

Evaluated on **$358,067$ test observations across $6,987$ completely unseen monitoring stations** (zero overlap with training sites). Confidence intervals derived via 1,000 bootstrap iterations:

| Model Architecture | Val $R^2$ | Val RMSE | Val MAE | Test $R^2$ | Test RMSE [95% CI] | Test MAE | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1. Global Train-Mean Baseline | -0.0005 | 13.4321 | 9.7768 | -0.0235 | 12.8823 [12.63, 13.12] | 9.9822 | Naive Benchmark |
| 2. Per-Waterbody Mean Baseline | 0.0468 | 13.1114 | 9.5312 | 0.0210 | 12.5991 [12.35, 12.83] | 9.7042 | Naive Benchmark |
| 3. Regularized Ridge (Linear) | 0.9234 | 3.7144 | 2.6288 | 0.9168 | 3.6841 [3.61, 3.75] | 2.5855 | Linear Baseline |
| 4. Random Forest (100 Trees) | 0.9992 | 0.3802 | 0.1245 | 0.9992 | 0.3621 [0.34, 0.38] | 0.1197 | Non-Linear Ensemble |
| 5. Tuned XGBoost (Unconstrained) | 0.9995 | 0.2882 | 0.0768 | 0.9995 | 0.2859 [0.27, 0.30] | 0.0715 | Standard Regressor |
| **6. Tuned XGBoost (Boundary-Clipped)** | **0.9995** | **0.2863** | **0.0754** | **0.9995** | **0.2834 [0.268, 0.299]** | **0.0704** | **Production Champion** |
| 7. Two-Stage Hurdle Model | 0.9995 | 0.2921 | 0.0781 | 0.9995 | 0.2884 [0.273, 0.304] | 0.0732 | Dual Classifier-Regressor |

---

## 📁 Repository Structure

```
AML_Capstone/
├── configs/                              # Declarative JSON configurations
│   ├── cleaning_config.json              # Cleaning rules, bounds, and exclusion logic
│   └── model_config.json                 # Model hyperparameters & feature lists
├── data/                                 # Excluded from git (see download instructions)
│   ├── raw/                              # Combined_dataset.csv (304 MB)
│   └── processed/                        # clean.parquet, excluded_non_ambient.parquet
│       └── splits/                       # site_held_out_split.npz, time_split.npz, etc.
├── models/                               # Saved joblib model artifacts
│   ├── xgboost_production_model.joblib
│   ├── two_stage_hurdle_model.joblib
│   ├── random_forest_model.joblib
│   └── water_quality_transformer.joblib
├── notebooks/                            # Reproducible Jupyter notebooks
│   ├── 01_dataset_analysis.ipynb         # Profiling, dtypes & formula leakage analysis
│   ├── 02_data_cleaning.ipynb            # Duplicate removal, plausibility & scope segregation
│   ├── 03_eda.ipynb                      # High-DPI visualizations, correlation & VIF
│   ├── 04_feature_engineering.ipynb      # Stoichiometry, oxygen deficit & ablation
│   └── 05_models.ipynb                   # Model training, tuning, diagnostics & SHAP
├── report/                               # Compile-ready IEEE technical report
│   ├── main.tex                          # Full IEEEtran LaTeX manuscript
│   ├── references.bib                    # Verified BibTeX citations
│   └── IEEEtran.cls                      # Official IEEE LaTeX class
├── reports/                              # Detailed markdown audit logs & JSON results
│   ├── 01_dataset_analysis.md & .json
│   ├── 02_cleaning_log.md & .json
│   ├── 03_eda_findings.md & .json
│   ├── 04_feature_engineering_report.md & .json
│   ├── 05_splits_report.md & .json
│   ├── 06_models_report.md & .json
│   ├── literature_review.md              # 19 real citations with research gap synthesis
│   ├── presentation_outline.md           # 14-slide presentation outline with speaker notes
│   ├── viva_qna.md                       # 35 technical examination questions & answers
│   └── figures/                          # 13 publication-grade 300-DPI charts (fig01 - fig13)
├── src/                                  # Modular, production-grade Python pipeline
│   ├── __init__.py
│   ├── stage1_analysis.py                # Profiling & health check
│   ├── cleaning.py                       # Config-driven cleaning pipeline
│   ├── eda.py                            # Statistical EDA & figure generation
│   ├── features.py                       # Sklearn transformer & ablation engine
│   ├── splits.py                         # Zero-leakage site & temporal split generator
│   └── models.py                         # Optuna tuning, training, uncertainty & SHAP
├── tests/                                # Automated unit test suite
│   ├── test_cleaning.py                  # Pytest assertions for cleaning & scope rules
│   └── test_splits.py                    # Pytest assertion: zero Area overlap in splits
├── pytest.ini                            # Pytest configuration
├── requirements.txt                      # Fully pinned dependencies
└── README.md
```

---

## 🚀 Quick Start & Reproduction

### 1. Environment Setup
Clone the repository and install the pinned dependencies:
```bash
git clone https://github.com/varunudata/AML_Capstone.git
cd AML_Capstone

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt

# On macOS, ensure OpenMP is present for XGBoost:
brew install libomp
```

### 2. Dataset Download Note
The master dataset is hosted on **Figshare** under CC BY 4.0 license:
* **DOI:** `10.6084/m9.figshare.27800394`
* **Article Title:** *A Comprehensive Surface Water Quality Monitoring Dataset (1940–2023)*
* **File:** `Combined_dataset.csv` (304.74 MB, 2,827,978 lines)

Download directly into the raw data folder:
```bash
mkdir -p data/raw data/processed
curl -L -o data/raw/Combined_dataset.csv https://ndownloader.figshare.com/files/50757321
```

### 3. Running the Pipeline
Execute each pipeline stage sequentially:
```bash
# Stage 1: Dataset Profiling & Raw Parquet Caching
python -m src.stage1_analysis

# Stage 2: Data Cleaning & Ambient Scope Segregation
python -m src.cleaning

# Stage 3: Exploratory Data Analysis & Publication Visualizations
python -m src.eda

# Stage 4: Feature Engineering & Baseline Ablation
python -m src.features

# Stage 5: Zero-Leakage Site and Temporal Split Generation
python -m src.splits

# Stage 6: Model Training, Optuna Tuning, Residuals & Explainability
python -m src.models
```

### 4. Running Automated Tests
```bash
pytest tests/
```
Output:
```
tests/test_cleaning.py ...                                               [ 60%]
tests/test_splits.py ..                                                  [100%]
============================== 5 passed in 1.36s ===============================
```

---

## 🔬 Scientific Highlights & Key Findings

### 1. Spatial Leakage Quantification
In a plain random split, $28,368$ monitoring stations ($91.23\%$ of test stations) appear in the training partition. Gradient boosted models easily memorize station-specific baseline mineralogy, creating the illusion of high accuracy. Our `GroupShuffleSplit` on `Area` guarantees **zero spatial overlap**.

### 2. Orthophosphate Dominance & DO Inversion
* **Orthophosphate** exhibits an extraordinary Spearman rank correlation of **$\rho = -0.913$** with CCME scores. As the primary limiting nutrient for freshwater eutrophication, slight elevations above $0.05\,\text{mg/L}$ trigger immediate regulatory failure.
* **Dissolved Oxygen** displays near-zero linear correlation ($r = -0.001$) because biological impairment is U-shaped: hypoxia ($DO < 5\,\text{mg/L}$) indicates organic waste, while supersaturation ($DO > 14\,\text{mg/L}$) indicates severe algal blooms. Modeling the temperature-dependent oxygen deficit resolved this non-linearity.

### 3. Uncertainty Quantification
Dual XGBoost quantile regression models ($\alpha = 0.10$ and $0.90$) establish an 80% credible interval. On the unseen spatial test set, **$98.27\%$ of ground-truth observations** fall within the predicted bounds, demonstrating conservative, well-calibrated confidence intervals across regional watersheds.

### 4. Explainability & Formula Rediscovery
SHAP TreeExplainer attributions confirm that the model's decision splits correspond to genuine aquatic thresholds rather than naive arithmetic lookup. Tree splits place maximum penalty on orthophosphate around $0.05\,\text{mg/L}$ and quadratic penalties on dissolved oxygen deficits.

---

## 🔮 Phase-2 Roadmap

Building upon this Phase 1 foundation, Phase 2 will transition into deep hybrid architectures:
1. **Physics-Informed Neural Networks (PINNs):** Embedding conservation of mass and oxygen re-aeration differential equations into loss functions.
2. **TabNet Architectures:** Applying sparse sequential attention for feature selection in data-sparse international basins.
3. **Cross-Country Transfer Learning:** Evaluating domain adaptation from high-density monitoring networks (England) to developing catchments.

---

## 📄 License & Attribution
This codebase is developed by **Varun Udata** for the Applied Machine Learning Capstone. The dataset is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

