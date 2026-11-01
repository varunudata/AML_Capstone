# Stage 2 — Data Cleaning Log

**Generated:** 2026-10-31 16:31:18

**Config:** `src/cleaning_config.json`

## Summary

- **Initial rows:** 2,827,977
- **Final rows:** 1,988,629
- **Total rows removed:** 839,348 (29.68%)
- **Output:** `data/processed/clean.parquet`

## Cleaning Steps

| # | Rule | Description | Rows Affected | % of Original | Action |
|---|------|-------------|---------------|---------------|--------|
| 1 | 1_duplicates | Drop exact duplicate rows | 9,971 | 0.353% | dropped |
| 2 | 2_impossible_Ammonia (mg/l)_below | Ammonia (mg/l) < 0 → NaN | 2 | 0.000% | flagged + set to NaN |
| 3 | 2_impossible_Biochemical Oxygen Demand (mg/l)_below | Biochemical Oxygen Demand (mg/l) < 0 → NaN | 2 | 0.000% | flagged + set to NaN |
| 4 | 2_impossible_Dissolved Oxygen (mg/l)_below | Dissolved Oxygen (mg/l) < 0.01 → NaN | 204 | 0.007% | flagged + set to NaN |
| 5 | 2_impossible_Orthophosphate (mg/l)_below | Orthophosphate (mg/l) < 0 → NaN | 10 | 0.000% | flagged + set to NaN |
| 6 | 2_impossible_pH (ph units)_below | pH (ph units) < 0 → NaN | 1 | 0.000% | flagged + set to NaN |
| 7 | 2_impossible_pH (ph units)_above | pH (ph units) > 14 → NaN | 504 | 0.018% | flagged + set to NaN |
| 8 | 2_impossible_Temperature (cel)_below | Temperature (cel) < -2 → NaN | 1 | 0.000% | flagged + set to NaN |
| 9 | 2_impossible_Temperature (cel)_above | Temperature (cel) > 40 → NaN | 7,284 | 0.258% | flagged + set to NaN |
| 10 | 3_detection_limits | Detection-limit pile-ups identified in 4 columns | 359,419 | 12.709% | reported only (values kept) |
| 11 | 4_outlier_Ammonia (mg/l) | Ammonia (mg/l) > 50 flagged as extreme | 6,658 | 0.235% | flag_only |
| 12 | 4_outlier_Biochemical Oxygen Demand (mg/l) | Biochemical Oxygen Demand (mg/l) > 100 flagged as extreme | 21,242 | 0.751% | flag_only |
| 13 | 4_outlier_Orthophosphate (mg/l) | Orthophosphate (mg/l) > 50 flagged as extreme | 302 | 0.011% | flag_only |
| 14 | 4_outlier_Nitrate (mg/l) | Nitrate (mg/l) > 100 flagged as extreme | 1,009 | 0.036% | flag_only |
| 15 | 5_target | Target = CCME_Values (continuous). CCME_WQI is derived, not a target. | 0 | 0.000% | documented |
| 16 | 6_waterbody_scope | Exclude ['Effluent', 'Sewage'] — not ambient water | 625,265 | 22.110% | separated (excluded to separate file) |
| 17 | 7_date_filter | Drop rows with year < 1990 | 204,112 | 7.218% | dropped |
| 18 | 8_metadata_roles | Metadata columns: ['Country', 'Area', 'Waterbody Type', 'Date'] — NOT used as model features | 0 | 0.000% | documented |

## Detailed Justifications

### 1_duplicates

**Drop exact duplicate rows**

Exact duplicates add no information and inflate apparent sample size

Rows: 2,827,977 → 2,818,006

### 2_impossible_Ammonia (mg/l)_below

**Ammonia (mg/l) < 0 → NaN**

Negative ammonia is physically impossible; likely below-detection-limit code or instrument artefact

### 2_impossible_Biochemical Oxygen Demand (mg/l)_below

**Biochemical Oxygen Demand (mg/l) < 0 → NaN**

Negative BOD is physically impossible; likely instrument artefact

### 2_impossible_Dissolved Oxygen (mg/l)_below

**Dissolved Oxygen (mg/l) < 0.01 → NaN**

DO = 0 is implausible for water samples (instrument error or anoxic artefact)

### 2_impossible_Orthophosphate (mg/l)_below

**Orthophosphate (mg/l) < 0 → NaN**

Negative orthophosphate is physically impossible

### 2_impossible_pH (ph units)_below

**pH (ph units) < 0 → NaN**

pH < 0 is not possible for aqueous solutions

### 2_impossible_pH (ph units)_above

**pH (ph units) > 14 → NaN**

pH > 14 is not possible for aqueous solutions

### 2_impossible_Temperature (cel)_below

**Temperature (cel) < -2 → NaN**

Water temperature below -2°C is implausible (even salt water)

### 2_impossible_Temperature (cel)_above

**Temperature (cel) > 40 → NaN**

Water temperature above 40°C is implausible for environmental monitoring (likely sensor error or unit confusion)

### 3_detection_limits

**Detection-limit pile-ups identified in 4 columns**

Values at common detection limits (0.005, 0.01, etc.) represent 'below detection limit' readings. These are valid lower-bound estimates and are retained. Models should be robust to this floor effect.

### 4_outlier_Ammonia (mg/l)

**Ammonia (mg/l) > 50 flagged as extreme**

Values above 50 are plausible for point-source discharge or heavily polluted sites, but are extreme for ambient water. Flagged for sensitivity testing; not deleted.

### 4_outlier_Biochemical Oxygen Demand (mg/l)

**Biochemical Oxygen Demand (mg/l) > 100 flagged as extreme**

Values above 100 are plausible for point-source discharge or heavily polluted sites, but are extreme for ambient water. Flagged for sensitivity testing; not deleted.

### 4_outlier_Orthophosphate (mg/l)

**Orthophosphate (mg/l) > 50 flagged as extreme**

Values above 50 are plausible for point-source discharge or heavily polluted sites, but are extreme for ambient water. Flagged for sensitivity testing; not deleted.

### 4_outlier_Nitrate (mg/l)

**Nitrate (mg/l) > 100 flagged as extreme**

Values above 100 are plausible for point-source discharge or heavily polluted sites, but are extreme for ambient water. Flagged for sensitivity testing; not deleted.

### 5_target

**Target = CCME_Values (continuous). CCME_WQI is derived, not a target.**

CCME_Values is right-censored at 100.0 (1,012,105 rows = 35.92%). Classification into 5 classes is a secondary view derived from regression predictions. The censoring means standard regression metrics (MSE, MAE) may underestimate error for high-quality sites. Consider Tobit regression or truncated loss in Phase 2.

### 6_waterbody_scope

**Exclude ['Effluent', 'Sewage'] — not ambient water**

Effluent (601,550 rows) and Sewage (23,777 rows) are point-source discharges, not ambient water quality. They follow different regulatory standards and have drastically different chemistry distributions (mean CCME 66.6 and 53.7 vs 90.4 for ambient). Including them conflates water quality assessment with wastewater treatment performance.

Rows: 2,818,006 → 2,192,741

### 7_date_filter

**Drop rows with year < 1990**

Years 1940-1989 contain only ~206k sparse rows across few sites (mostly USA), with very different monitoring regimes. 1990+ provides 97.7% of data.

Rows: 2,192,741 → 1,988,629

### 8_metadata_roles

**Metadata columns: ['Country', 'Area', 'Waterbody Type', 'Date'] — NOT used as model features**

Country is a distribution-shift driver (would be a shortcut feature). Area is used for group-aware train/val/test splitting (no site leaks). Date provides temporal structure. These are tested as ablations only.

## Detection Limit Analysis

Values piled at common detection-limit thresholds:

| Column | DL Value | Count | % |
|--------|----------|-------|---|
| Ammonia (mg/l) | 0 | 1,389 | 0.049% |
| Ammonia (mg/l) | 0.001 | 1,362 | 0.048% |
| Ammonia (mg/l) | 0.002 | 2,324 | 0.082% |
| Ammonia (mg/l) | 0.005 | 1,837 | 0.065% |
| Ammonia (mg/l) | 0.01 | 10,316 | 0.365% |
| Ammonia (mg/l) | 0.02 | 12,821 | 0.453% |
| Ammonia (mg/l) | 0.05 | 14,449 | 0.511% |
| Ammonia (mg/l) | 0.1 | 7,675 | 0.271% |
| Orthophosphate (mg/l) | 0 | 690 | 0.024% |
| Orthophosphate (mg/l) | 0.001 | 6,276 | 0.222% |
| Orthophosphate (mg/l) | 0.002 | 2,318 | 0.082% |
| Orthophosphate (mg/l) | 0.005 | 5,661 | 0.200% |
| Orthophosphate (mg/l) | 0.01 | 71,766 | 2.538% |
| Orthophosphate (mg/l) | 0.02 | 172,530 | 6.101% |
| Orthophosphate (mg/l) | 0.05 | 14,537 | 0.514% |
| Orthophosphate (mg/l) | 0.1 | 8,763 | 0.310% |
| Nitrogen (mg/l) | 0 | 131 | 0.005% |
| Nitrogen (mg/l) | 0.001 | 147 | 0.005% |
| Nitrogen (mg/l) | 0.002 | 1,283 | 0.045% |
| Nitrogen (mg/l) | 0.005 | 1,634 | 0.058% |
| Nitrogen (mg/l) | 0.01 | 1,857 | 0.066% |
| Nitrogen (mg/l) | 0.02 | 1,024 | 0.036% |
| Nitrogen (mg/l) | 0.05 | 615 | 0.022% |
| Nitrogen (mg/l) | 0.1 | 3,650 | 0.129% |
| Nitrate (mg/l) | 0 | 1,175 | 0.042% |
| Nitrate (mg/l) | 0.001 | 320 | 0.011% |
| Nitrate (mg/l) | 0.002 | 940 | 0.033% |
| Nitrate (mg/l) | 0.005 | 908 | 0.032% |
| Nitrate (mg/l) | 0.01 | 2,303 | 0.081% |
| Nitrate (mg/l) | 0.02 | 1,897 | 0.067% |
| Nitrate (mg/l) | 0.05 | 866 | 0.031% |
| Nitrate (mg/l) | 0.1 | 5,955 | 0.211% |

> **Decision:** Detection-limit values are retained. They represent valid
> lower-bound measurements and are common in environmental monitoring data.
> Models (especially tree-based) handle these floor effects naturally.

## Extreme Outlier Summary

| Column | Threshold | N Flagged | % |
|--------|-----------|-----------|---|
| Ammonia (mg/l) | > 50 | 6,658 | 0.236% |
| Biochemical Oxygen Demand (mg/l) | > 100 | 21,242 | 0.754% |
| Orthophosphate (mg/l) | > 50 | 302 | 0.011% |
| Nitrate (mg/l) | > 100 | 1,009 | 0.036% |

> **Decision:** Extreme outliers are **flagged** (boolean columns) but **not removed**.
> Sensitivity analysis in Stage 4 will test model performance with and without them.

## IQR Analysis (3×IQR fences)

| Column | Q1 | Q3 | IQR | Lower Fence | Upper Fence | Below | Above |
|--------|----|----|-----|-------------|-------------|-------|-------|
| Ammonia (mg/l) | 0.03 | 0.32 | 0.29 | -0.84 | 1.19 | 0 | 322,561 |
| Biochemical Oxygen Demand (mg/l) | 1.6 | 2.85 | 1.25 | -2.15 | 6.6 | 0 | 291,485 |
| Dissolved Oxygen (mg/l) | 9.86 | 11.0 | 1.14 | 6.44 | 14.42 | 169,881 | 12,634 |
| Orthophosphate (mg/l) | 0.04 | 0.229 | 0.189 | -0.527 | 0.796 | 0 | 391,075 |
| pH (ph units) | 7.55 | 8.0 | 0.45 | 6.2 | 9.35 | 22,304 | 4,206 |
| Temperature (cel) | 8.91 | 14.2 | 5.29 | -6.96 | 30.07 | 0 | 1,631 |
| Nitrogen (mg/l) | 0.8 | 6.35 | 5.55 | -15.85 | 23.0 | 0 | 80,622 |
| Nitrate (mg/l) | 1.2 | 4.97 | 3.77 | -10.11 | 16.28 | 0 | 112,819 |

## NaN Summary After Cleaning

These NaNs will be imputed inside the training split only (fit on train, apply to val/test).

| Feature | N NaN | % NaN |
|---------|-------|-------|
| Ammonia (mg/l) | 2 | 0.0% |
| Biochemical Oxygen Demand (mg/l) | 2 | 0.0% |
| Dissolved Oxygen (mg/l) | 187 | 0.009% |
| Orthophosphate (mg/l) | 10 | 0.001% |
| pH (ph units) | 504 | 0.025% |
| Temperature (cel) | 224 | 0.011% |
| Nitrogen (mg/l) | 0 | 0.0% |
| Nitrate (mg/l) | 0 | 0.0% |

## Final Clean Dataset Summary

- **Shape:** 1,988,629 rows × 30 columns
- **Countries:** 5
- **Areas:** 46,575
- **Waterbody Types:** ['Bay', 'Canal', 'Coastal', 'Drainage', 'Estuarine', 'Lake', 'Marine', 'River', 'Sea Water', 'Transitional']
- **Year range:** 1990–2023
- **Target (CCME_Values):** min=32.15, max=100.00, mean=89.62
- **Censored at 100:** 823,170 (41.4%)

## Note on Imputation (Rule 9)

NaN values created by cleaning steps above will be imputed using a
`sklearn.Pipeline` that is **fit on the training split only** and
applied (transform only) to validation and test splits.
This prevents information leakage from val/test into training.

Imputation strategy: median (robust to skewness and outliers),
implemented via `sklearn.impute.SimpleImputer(strategy='median')`.
