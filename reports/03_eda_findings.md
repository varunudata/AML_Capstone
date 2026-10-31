# Stage 3 — Exploratory Data Analysis (EDA) Findings

**Generated:** 2026-10-31 17:59:50  
**Dataset:** Ambient Clean Parquet (`data/processed/clean.parquet`, 1,988,629 rows)  
**Sample Policy:** 500,000-row stratified random sample (seed 42) used for heavy plotting and regressions; full dataset used for exact global counts and percentiles.

---

## Executive Summary of EDA Artifacts

All 7 publication-quality figures have been generated at **300 DPI** with labeled axes and physical units, stored in [`reports/figures/`](./reports/figures):
1. **[`fig01_target_distribution.png`](./reports/figures/fig01_target_distribution.png)**: Target continuous distribution & CCME discrete class balance.
2. **[`fig02_feature_distributions_before_after.png`](./reports/figures/fig02_feature_distributions_before_after.png)**: Feature density comparisons before vs after cleaning across all 8 chemistry variables.
3. **[`fig03_correlation_heatmaps.png`](./reports/figures/fig03_correlation_heatmaps.png)**: Pearson linear and Spearman rank correlation matrices + VIF analysis.
4. **[`fig04_feature_target_relationships.png`](./reports/figures/fig04_feature_target_relationships.png)**: Binned means and hexbin joint distributions against `CCME_Values`.
5. **[`fig05_target_by_groups_and_time.png`](./reports/figures/fig05_target_by_groups_and_time.png)**: Target distributions across Country, Waterbody Type, Observation Year, and Seasonal Month.
6. **[`fig06_invalid_values_summary.png`](./reports/figures/fig06_invalid_values_summary.png)**: Summary of all invalid, non-ambient, and extreme records handled in Stage 2.
7. **[`fig07_site_sampling_distribution.png`](./reports/figures/fig07_site_sampling_distribution.png)**: Monitoring site density and longitudinal site sampling duration.

---

## 10 Number-Backed Insights & Specific Modelling Implications

### Insight 1: Severe Target Censoring at 100.0 (Ceiling Spike)
- **Empirical Evidence:** Exactly **823,170 rows (41.39%)** in the ambient clean dataset sit at the upper limit of `CCME_Values = 100.0`. The overall distribution has mean 89.6209 ± 13.1363, but median 92.8696 and IQR 14.0867.
- **Modelling Implication:** Standard OLS linear regression and MSE-minimizing models assume unconstrained continuous error distributions, which systematically under-predict near 100.0 and suffer boundary bias. We must formulate the continuous problem with post-prediction boundary clipping ($[0, 100]$) and test tree-based ensembles (Random Forest, LightGBM, XGBoost) and Tobit/censored loss formulations.

### Insight 2: Orthophosphate is the Dominant Monotonic Predictor
- **Empirical Evidence:** Orthophosphate exhibits an extraordinary Spearman rank correlation of **$\rho = -0.913$** with `CCME_Values` (Pearson $r = -0.501$).
- **Modelling Implication:** In regulatory freshwater frameworks (e.g. EU WFD, US EPA), orthophosphate guidelines for eutrophication are stringent (often <0.05 mg/L). Even slight increases trigger immediate guideline failure, severely degrading CCME scores. Tree models will heavily prioritize splits on Orthophosphate; feature engineering must include $\log(1 + \text{Orthophosphate})$ and phosphorus saturation proxies.

### Insight 3: Extreme Collinearity Between Total Nitrogen and Nitrate
- **Empirical Evidence:** Nitrogen and Nitrate share a Spearman rank correlation of **$\rho = 0.907$** (Pearson $r = 0.765$). In VIF testing, Nitrogen ($VIF = 2.54$) and Nitrate ($VIF = 2.42$) are by far the highest among all features.
- **Modelling Implication:** In linear and generalized additive models, this collinearity inflates standard errors and causes coefficient sign flips. In tree ensembles, it artificially splits feature importance across both variables. In Stage 4, we must construct an interaction feature (Nitrate/Nitrogen ratio) and evaluate L1/L2 regularized baselines (Lasso/Ridge/ElasticNet).

### Insight 4: Dissolved Oxygen Displays a Non-Linear Biological Inversion
- **Empirical Evidence:** Dissolved Oxygen exhibits near-zero linear correlation ($r = -0.001$) and rank correlation ($\rho = 0.009$) across the dataset.
- **Modelling Implication:** Dissolved oxygen impairment is biologically two-tailed: hypoxia ($DO < 5$ mg/L) indicates severe organic pollution, while extreme super-saturation ($DO > 14$ mg/L) indicates aggressive algal eutrophication. Linear models miss this completely. In Stage 4, we must create non-linear proxy features such as $|DO - 10|$ or temperature-dependent saturation deficit.

### Insight 5: Geographic Shortcut Bias — Country Explains 14% Variance Alone
- **Empirical Evidence:** In regression testing without any chemistry features, `Country` alone achieves **$R^2 = 0.1403$** (reducing RMSE from 13.1585 to 12.2008). Median CCME in USA is 100.0, Ireland is 100.0, while England is 89.63.
- **Modelling Implication:** A naive machine learning model will learn geographic shortcuts rather than generalizable chemical relationships. `Country` must NOT be passed as a standard predictor in the primary pipeline. It must be strictly reserved for spatial cross-validation (GroupKFold / Leave-One-Country-Out) to evaluate transferability.

### Insight 6: Ambient Waterbody Type Disparities
- **Empirical Evidence:** Waterbody Type alone explains **$R^2 = 0.0480$** of target variance. Drainage waters display an ambient mean of 73.91, while Lakes reach 96.95. Segregating Effluent and Sewage (625,265 rows) eliminated extreme industrial contamination.
- **Modelling Implication:** Waterbody Type reflects physical re-aeration and dilution capacity. It should be evaluated strictly as an ablation feature (tested with vs without one-hot encoding in Stage 4) rather than a default mandatory feature.

### Insight 7: Limnological Seasonal Depressions in Summer
- **Empirical Evidence:** Monthly mean CCME drops from a spring peak of **90.99** (March) to **88.1** (July) and **88.12** (August), while standard deviation increases from 12.34 to 14.32.
- **Modelling Implication:** Higher summer water temperatures accelerate microbial decomposition, lower dissolved oxygen saturation, and stimulate agricultural runoff, driving guideline failures. Stage 4 feature engineering should test cyclical date representations (`sin(2π * month / 12)` and `cos(2π * month / 12)`).

### Insight 8: Severe Monitoring Site Observation Skew
- **Empirical Evidence:** The dataset encompasses **46,575 unique monitoring sites (Areas)**. 25% of sites have only 1 observation, median observation count is 8, but the 99th percentile site has 256 rows and the maximum single site has **49,149 observations**.
- **Modelling Implication:** Global loss functions without sample or site weighting will be dominated by a few hundred high-frequency monitoring stations. Cross-validation splits must group by `Area` to prevent high-density stations from corrupting both train and test splits.

### Insight 9: Multi-Year Site Sampling Confirms Spatial-Temporal Leakage Risk
- **Empirical Evidence:** **63.8% of monitoring sites (29,716 sites)** contain observations across multiple years, with **40.8% spanning $\\ge 5$ years**.
- **Modelling Implication:** Naive random train-test splitting guarantees that the same river stretch observed in 2015 will appear in the training set, while its 2016 record appears in the test set. The model would exploit site-specific background chemistry (memorization) rather than predicting water quality from physics/chemistry. Split design MUST be `GroupKFold(groups=Area)` or temporal block splitting.

### Insight 10: Heavy Skewness of Concentration Features Requires Log Transformation
- **Empirical Evidence:** Chemistry variables (Ammonia, BOD, Orthophosphate, Nitrogen, Nitrate) span multiple orders of magnitude with extreme positive skewness (e.g. Ammonia median 0.055 vs max 50; BOD median 2.7 vs max 100).
- **Modelling Implication:** Linear models, distance metrics (KNN), and PCA/scaling are heavily distorted by extreme values. A logarithmic transformation ($\\log(1 + x)$) or power transform inside an sklearn `Pipeline` is required for linear and kernel baselines.

---

## Statistical Summary Tables

### 1. Variance Inflation Factors (VIF)

| Feature | VIF | Collinearity Status |
|---------|-----|---------------------|
| Ammonia (mg/l) | 1.1185 | Low / Acceptable |
| Biochemical Oxygen Demand (mg/l) | 1.1123 | Low / Acceptable |
| Dissolved Oxygen (mg/l) | 1.1239 | Low / Acceptable |
| Orthophosphate (mg/l) | 1.1082 | Low / Acceptable |
| pH (ph units) | 1.0449 | Low / Acceptable |
| Temperature (cel) | 1.1291 | Low / Acceptable |
| Nitrogen (mg/l) | 2.5366 | High Collinearity (N-species) |
| Nitrate (mg/l) | 2.4238 | High Collinearity (N-species) |

### 2. Shortcut Regression Baselines (Predicting CCME_Values)

| Feature Set | R² Score | RMSE (units) | MAE (units) | Shortcut Risk |
|-------------|----------|--------------|-------------|---------------|
| Dummy (Mean Baseline) | 0.0000 | 13.1585 | 9.6606 | Benchmark |
| Country Alone | 0.1403 | 12.2008 | 8.5680 | High Shortcut (14% Variance) |
| Waterbody Alone | 0.0480 | 12.8386 | 9.3970 | Moderate Shortcut (4.8% Variance) |
| Country + Waterbody | 0.1477 | 12.1479 | 8.5089 | High Shortcut (14.8% Variance) |
| Country + Waterbody + Month | 0.1564 | 12.0859 | 8.4558 | Severe Shortcut (15.6% Variance) |

---

## Stage 3 Verification
- All plots generated at 300 DPI in [`reports/figures/`](./reports/figures).
- Full numerical outputs saved in [`reports/03_eda_results.json`](./reports/03_eda_results.json).
- Stage 3 complete.
