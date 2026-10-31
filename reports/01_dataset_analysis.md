# Stage 1 — Dataset Analysis Report

**Generated:** 2026-10-31 16:06:44

**Dataset:** `data/raw/Combined_dataset.csv`

## 1. Basic Information

- **Shape:** 2,827,977 rows × 14 columns
- **Memory:** 328.49 MB
- **Total missing values:** 0
- **Exact duplicate rows:** 9,971

### Dtypes

| Column | Dtype |
|--------|-------|
| Country | category |
| Area | str |
| Waterbody Type | category |
| Date | datetime64[us] |
| Ammonia (mg/l) | float64 |
| Biochemical Oxygen Demand (mg/l) | float64 |
| Dissolved Oxygen (mg/l) | float64 |
| Orthophosphate (mg/l) | float64 |
| pH (ph units) | float64 |
| Temperature (cel) | float64 |
| Nitrogen (mg/l) | float64 |
| Nitrate (mg/l) | float64 |
| CCME_Values | float64 |
| CCME_WQI | category |

### Unique Counts

| Column | Unique Values |
|--------|---------------|
| Country | 5 |
| Area | 61,689 |
| Waterbody Type | 12 |
| Date | 23,876 |
| Ammonia (mg/l) | 13,987 |
| Biochemical Oxygen Demand (mg/l) | 4,717 |
| Dissolved Oxygen (mg/l) | 16,253 |
| Orthophosphate (mg/l) | 10,596 |
| pH (ph units) | 4,726 |
| Temperature (cel) | 11,536 |
| Nitrogen (mg/l) | 16,232 |
| Nitrate (mg/l) | 10,823 |
| CCME_Values | 221,931 |
| CCME_WQI | 5 |

## 2. Numeric Percentile Summary

|                                  |       count |      mean |       std |     min |      0% |      1% |      5% |     10% |     25% |     50% |     75% |      90% |       95% |    99% |   100% |   max |
|:---------------------------------|------------:|----------:|----------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|---------:|----------:|-------:|-------:|------:|
| Ammonia (mg/l)                   | 2.82798e+06 |  1.17136  |  5.66893  | -0.005  | -0.005  |  0.009  |  0.021  |  0.021  |  0.03   |  0.055  |   0.317 |   1.54   |   4.45816 |  26.7  |    200 |   200 |
| Biochemical Oxygen Demand (mg/l) | 2.82798e+06 |  4.88655  | 16.4136   | -2      | -2      |  0.7    |  1      |  1.1    |  1.6    |  2.7    |   2.83  |   6.85   |  11.7     |  68.2  |    255 |   255 |
| Dissolved Oxygen (mg/l)          | 2.82798e+06 | 10.008    |  1.85105  |  0      |  0      |  3.5505 |  5.7    |  8.1    |  9.86   | 10.2    |  11     |  11.6841 |  12.2     |  13.5  |     20 |    20 |
| Orthophosphate (mg/l)            | 2.82798e+06 |  0.710466 |  2.08944  | -0.004  | -0.004  |  0.004  |  0.01   |  0.02   |  0.04   |  0.107  |   0.227 |   1.82   |   4.9     |   9.14 |    100 |   100 |
| pH (ph units)                    | 2.82798e+06 |  7.73593  |  0.494687 | -1      | -1      |  6.3    |  7      |  7.2    |  7.55   |  7.78   |   8     |   8.2    |   8.3     |   8.7  |     30 |    30 |
| Temperature (cel)                | 2.82798e+06 | 11.8393   |  5.0102   | -5.26   | -5.26   |  2.97   |  5      |  6.3    |  8.99   | 11.46   |  14.2   |  17.37   |  19.2     |  25.7  |     98 |    98 |
| Nitrogen (mg/l)                  | 2.82798e+06 |  5.20953  |  6.20042  |  0      |  0      |  0.019  |  0.2    |  0.4    |  0.78   |  4      |   6.32  |  12.4    |  18.3     |  30.3  |     46 |    46 |
| Nitrate (mg/l)                   | 2.82798e+06 |  4.76679  |  6.07378  |  0      |  0      |  0.046  |  0.2    |  0.556  |  1.173  |  4.5    |   4.94  |   9.83   |  14.5     |  28.3  |    155 |   155 |
| CCME_Values                      | 2.82798e+06 | 85.0467   | 17.6466   | 31.3041 | 31.3041 | 40.8845 | 45.4589 | 51.8346 | 77.1535 | 90.5961 | 100     | 100      | 100       | 100    |    100 |   100 |

## 3. Row Counts by Groups

### By Country

| Country | Rows | % |
|---------|------|---|
| England | 2,129,198 | 75.3% |
| USA | 413,814 | 14.6% |
| Ireland | 235,019 | 8.3% |
| China | 45,997 | 1.6% |
| Canada | 3,949 | 0.1% |

### By Waterbody Type

| Waterbody Type | Rows | % |
|----------------|------|---|
| River | 1,852,579 | 65.5% |
| Effluent | 601,550 | 21.3% |
| Lake | 153,603 | 5.4% |
| Estuarine | 49,375 | 1.7% |
| Bay | 45,997 | 1.6% |
| Sea Water | 32,061 | 1.1% |
| Canal | 28,574 | 1.0% |
| Sewage | 23,777 | 0.8% |
| Marine | 23,162 | 0.8% |
| Drainage | 10,205 | 0.4% |
| Transitional | 4,275 | 0.2% |
| Coastal | 2,819 | 0.1% |

### Year Range: 1940–2023

Top 10 years by row count:

| Year | Rows |
|------|------|
| 2002 | 137,777 |
| 2003 | 136,686 |
| 2000 | 136,609 |
| 2004 | 136,037 |
| 2005 | 134,302 |
| 2007 | 132,699 |
| 2013 | 131,077 |
| 2006 | 130,829 |
| 2008 | 127,733 |
| 2012 | 120,690 |

### Area (Site) Statistics

- **Distinct areas:** 61,689
- **Max rows in one area:** 54,281 (`Lake Washington`)
- **Median rows per area:** 9.0
- **Mean rows per area:** 45.84
- **Areas with <10 rows:** 31,888
- **Areas with <5 rows:** 25,063
- **Areas with exactly 1 row:** 14,527

## 4. Plausibility Check

| Variable | Range | Actual Min | Actual Max | Below Min | Above Max | Exactly 0 | Negative |
|----------|-------|-----------|-----------|-----------|-----------|-----------|----------|
| Ammonia (mg/l) | [0, 50] | -0.0050 | 200.0000 | 2 | 6,658 | 1,390 | 2 |
| Biochemical Oxygen Demand (mg/l) | [0, 100] | -2.0000 | 255.0000 | 2 | 21,244 | 75 | 2 |
| Dissolved Oxygen (mg/l) | [0, 20] | 0.0000 | 20.0000 | 0 | 0 | 39 | 0 |
| Orthophosphate (mg/l) | [0, 50] | -0.0040 | 100.0000 | 10 | 302 | 690 | 10 |
| pH (ph units) | [0, 14] | -1.0000 | 30.0000 | 1 | 504 | 13 | 1 |
| Temperature (cel) | [-5, 45] | -5.2600 | 98.0000 | 1 | 6,818 | 266 | 32 |
| Nitrogen (mg/l) | [0, 200] | 0.0000 | 46.0000 | 0 | 0 | 131 | 0 |
| Nitrate (mg/l) | [0, 200] | 0.0000 | 155.0000 | 0 | 0 | 1,175 | 0 |

## 5. CCME_WQI Class Analysis (Reverse-Engineered)

- **Classes in data:** ['Excellent', 'Fair', 'Good', 'Marginal', 'Poor']
- **Rows with CCME_Values = 100.0:** 1,020,710 (36.09%)

| Class | Count | % of Total | Min Value | Max Value | Mean |
|-------|-------|-----------|-----------|-----------|------|
| Excellent | 1,020,710 | 36.09% | 100.00 | 100.00 | 100.00 |
| Fair | 317,856 | 11.24% | 65.00 | 80.00 | 73.27 |
| Good | 1,033,709 | 36.55% | 80.00 | 93.20 | 89.19 |
| Marginal | 328,734 | 11.62% | 45.00 | 65.00 | 53.42 |
| Poor | 126,968 | 4.49% | 31.30 | 45.00 | 42.43 |

### Derived Boundaries

| Class | Min | Max |
|-------|-----|-----|
| Excellent | 100.00 | 100.00 |
| Fair | 65.00 | 80.00 |
| Good | 80.00 | 93.20 |
| Marginal | 45.00 | 65.00 |
| Poor | 31.30 | 45.00 |

## 6. Leakage Analysis ⚠️

### Linear Regression on Raw Chemistry Features

- **R²:** 0.642
- **MAE:** 7.9086
- **N samples:** 500,000

### Exceedance-Count Feature Only

- **R²:** 0.5922
- **MAE:** 8.224

### Feature–Target Correlations

| Feature | Pearson | Spearman |
|---------|---------|----------|
| Ammonia (mg/l) | -0.392 | -0.7208 |
| Biochemical Oxygen Demand (mg/l) | -0.3204 | -0.5351 |
| Dissolved Oxygen (mg/l) | -0.0297 | -0.0234 |
| Orthophosphate (mg/l) | -0.6618 | -0.9167 |
| pH (ph units) | 0.0784 | 0.1077 |
| Temperature (cel) | -0.1082 | -0.1502 |
| Nitrogen (mg/l) | -0.6107 | -0.6013 |
| Nitrate (mg/l) | -0.3071 | -0.5003 |

> **Leakage Statement:**
> CCME WQI is computed from guideline exceedances (F1 scope, F2 frequency, F3 amplitude) of the SAME chemistry variables used as predictors. This means the ML task is partly 're-learning a deterministic formula'. A simple linear regression on raw chemistry features achieves R²=0.642. Even a single exceedance-count feature achieves R²=0.592. This inherent information overlap must be disclosed. The task still has value: (1) the exact guidelines/thresholds may vary by jurisdiction, (2) the CCME formula is non-linear, (3) we can study generalization across Countries and time.

## 7. Distribution Shift Analysis

### By Country

| Country | N | Mean | Median | Std | %=100 |
|---------|---|------|--------|-----|-------|
| Canada | 3,949 | 90.22 | 93.12 | 12.36 | 35.48% |
| China | 45,997 | 96.51 | 100.0 | 4.62 | 54.6% |
| England | 2,129,198 | 80.74 | 88.76 | 18.13 | 20.93% |
| Ireland | 235,019 | 98.11 | 100.0 | 6.41 | 84.24% |
| USA | 413,814 | 98.46 | 100.0 | 4.67 | 84.74% |

### Class Distribution by Country

**Canada:** Good: 52.19%, Excellent: 35.48%, Fair: 5.93%, Marginal: 4.81%, Poor: 1.6%
**China:** Excellent: 54.6%, Good: 44.4%, Fair: 0.73%, Marginal: 0.26%, Poor: 0.0%
**England:** Good: 43.18%, Excellent: 20.93%, Marginal: 15.29%, Fair: 14.71%, Poor: 5.89%
**Ireland:** Excellent: 84.24%, Good: 14.07%, Fair: 0.69%, Poor: 0.61%, Marginal: 0.39%
**USA:** Excellent: 84.74%, Good: 14.2%, Fair: 0.58%, Marginal: 0.46%, Poor: 0.02%

### By Waterbody Type

| Waterbody Type | N | Mean | Median | Std |
|----------------|---|------|--------|-----|
| Bay | 45,997 | 96.51 | 100.0 | 4.62 |
| Canal | 28,574 | 86.53 | 89.63 | 13.48 |
| Coastal | 2,819 | 93.7 | 100.0 | 16.15 |
| Drainage | 10,205 | 73.94 | 74.34 | 16.82 |
| Effluent | 601,550 | 66.56 | 67.59 | 19.11 |
| Estuarine | 49,375 | 85.12 | 89.63 | 12.6 |
| Lake | 153,603 | 97.11 | 100.0 | 7.75 |
| Marine | 23,162 | 97.09 | 100.0 | 4.31 |
| River | 1,852,579 | 90.03 | 93.0 | 12.97 |
| Sea Water | 32,061 | 84.37 | 89.63 | 15.78 |
| Sewage | 23,777 | 53.66 | 47.88 | 13.76 |
| Transitional | 4,275 | 93.76 | 100.0 | 15.56 |

### By Era

| Era | N | Mean | Median | Std | %=100 |
|-----|---|------|--------|-----|-------|
| 1940-1970 | 96,867 | 98.55 | 100.0 | 4.55 | 86.1% |
| 1970-1990 | 108,816 | 97.44 | 100.0 | 6.61 | 78.32% |
| 1990-2000 | 37,406 | 99.18 | 100.0 | 3.28 | 91.23% |
| 2000-2010 | 1,306,155 | 78.3 | 86.13 | 20.2 | 23.74% |
| 2010-2015 | 574,484 | 88.13 | 90.8 | 13.73 | 34.93% |
| 2015-2020 | 451,276 | 89.94 | 92.9 | 12.65 | 42.01% |
| 2020-2024 | 252,973 | 91.54 | 93.15 | 11.17 | 46.49% |

## 8. Data Dictionary

| # | Column | Type | Role | Description |
|---|--------|------|------|-------------|
| 1 | Country | Categorical | Metadata (split) | Country of origin. NOT a model feature (leakage risk). |
| 2 | Area | String | Metadata (split) | Site/station identifier. Used for group-aware splitting. |
| 3 | Waterbody Type | Categorical | Metadata/Feature | Type of water body (River, Lake, etc.). Ablation candidate. |
| 4 | Date | Datetime | Metadata | Measurement date (dd-mm-yyyy). Month extracted for seasonality. |
| 5 | Ammonia (mg/l) | Float | Feature | Ammonia concentration. Positively skewed. |
| 6 | Biochemical Oxygen Demand (mg/l) | Float | Feature | BOD. Organic pollution indicator. |
| 7 | Dissolved Oxygen (mg/l) | Float | Feature | DO. Essential for aquatic life. Inversely related to pollution. |
| 8 | Orthophosphate (mg/l) | Float | Feature | Phosphate concentration. Eutrophication indicator. |
| 9 | pH (ph units) | Float | Feature | Acidity/alkalinity. Expected 6–9 for ambient water. |
| 10 | Temperature (cel) | Float | Feature | Water temperature. Affects DO saturation. |
| 11 | Nitrogen (mg/l) | Float | Feature | Total nitrogen. Likely collinear with Nitrate. |
| 12 | Nitrate (mg/l) | Float | Feature | Nitrate concentration. Component of total nitrogen. |
| 13 | CCME_Values | Float | **TARGET** | Continuous WQI score (31.3–100). Censored at 100. |
| 14 | CCME_WQI | Categorical | Derived | 5-class label derived from CCME_Values. NOT a separate target. |

> **Note on missing features:** The dataset does NOT contain turbidity, 
> conductivity, or dissolved solids, which may be listed in the project brief. 
> We use the 8 available chemistry variables and document this deviation.
