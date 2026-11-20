# Phase 1 Viva Examination Notes: Data Foundation & AML Baseline Bridge

These technical notes explain the core architectural, statistical, and limnological design choices implemented in Phase 1.

---

### Q1: Why predict `CCME_Values` as a continuous regression task instead of discrete `CCME_WQI` classification directly?
**Answer:** The Canadian Council of Ministers of the Environment Water Quality Index (CCME WQI) is fundamentally a continuous metric in $[0, 100]$ computed from three variance dimensions ($F_1$ Scope, $F_2$ Frequency, $F_3$ Amplitude). Predicting a continuous score preserves the fine-grained ecological gradient and ordering information. Discretizing into 5 bins prior to modeling discards intra-bin variance (e.g. treating an 81 and a 94 identically as "Good"), creates arbitrary threshold boundary effects, and obscures subtle degradation trends. If discrete quality tiers are needed for regulatory reporting, continuous predictions can be mapped monotonically to regulatory tiers via standard post-processing cutoffs (Excellent $\ge 95$, Good $\ge 80$, Fair $\ge 65$, Marginal $\ge 45$, Poor $< 45$).

---

### Q2: Why is random splitting fatally flawed for environmental sensor datasets, and how does `GroupShuffleSplit` on `Area` solve it?
**Answer:** Routine water quality monitoring stations sample the same geographic coordinate repeatedly over decades. If observations are shuffled randomly across rows, over 91% of monitoring stations appear simultaneously in both training and test sets. The model simply memorizes station-specific baseline chemistry and hydrological quirks ("station identity leakage") rather than learning generalized physical limnology. By executing a two-stage `GroupShuffleSplit` grouped on station identifier (`Area`), we enforce a strict **Site-Held-Out protocol with 0% station overlap**. Models must generalize to completely unseen river basins and waterbodies.

---

### Q3: Why is 41.4% of the target values censored at exactly 100.0, and what problems does this create for linear and neural models?
**Answer:** In pristine or well-maintained ambient waterbodies, when all measured chemical parameters satisfy Canadian environmental guidelines, $F_1 = F_2 = F_3 = 0$, producing $WQI = 100 - 0 = 100.0$. In our cleaned dataset of 1.98 million ambient rows, 41.39% of samples sit at this boundary. Standard Mean Squared Error (MSE) loss assumes Gaussian, unconstrained residuals. Standard linear regressors overshoot above 100.0 or under-predict pristine points due to regression-to-the-mean compression. For Phase 1 tree baselines, we apply explicit boundary clipping $[0, 100]$. For Phase 2 neural networks, Member 2 can benchmark sigmoid-scaled outputs or hurdle architectures.

---

### Q4: Why must the `SimpleImputer` and `StandardScaler` be fit strictly on the training fold (`X_train`) only?
**Answer:** Fitting an imputer or scaler on the whole dataset prior to splitting introduces **data snooping / distributional leakage**: test set means, variances, and medians pollute the transformation parameters. In our pipeline, `WaterQualityPreprocessor.fit()` is called exclusively on `df_train`. The resulting `preprocessor.joblib` artifact freezes the training medians (for median imputation) and training means/scales (for standardization). Validation and test folds are transformed using frozen training parameters, ensuring that test performance reflects true out-of-sample deployment.

---

### Q5: Why apply `log1p` transformation to Ammonia, BOD, Orthophosphate, Nitrogen, and Nitrate, but keep pH, Temperature, and Dissolved Oxygen linear?
**Answer:** Environmental contaminant concentrations (Ammonia, BOD, Orthophosphate, Nitrate, Nitrogen) naturally follow log-normal distributions spanning several orders of magnitude with extreme positive skewness (e.g., Ammonia skewness = 22.8, Orthophosphate skewness = 26.3). Unbounded positive skew causes gradient instability in deep neural networks and concentrates splits in tree algorithms. Applying $\log(1 + x)$ contracts the long right tail (reducing Ammonia skewness from 22.8 to 6.3) and stabilizes variance. In contrast:
- **pH** is already inherently logarithmic ($-\log_{10}[\text{H}^+]$) and normally centered around 7.0–8.2 (skewness = -1.5).
- **Temperature** and **Dissolved Oxygen** exhibit symmetrical thermodynamic distributions governed by atmospheric equilibrium rather than multiplicative dilution.

---

### Q6: What is "formula leakage" in WQI prediction, and how did your empirical leakage audit quantify it?
**Answer:** Formula leakage occurs when a model is tasked with predicting an index that is mathematically derived from its own input features. If exceedance flags or Canadian threshold lookups were provided as inputs, the model would simply learn arithmetic table lookups rather than environmental chemistry. In our pipeline, exceedance flags are quarantined. Our empirical leakage audit quantified this boundary:
- A linear Ridge model on the raw 8 chemical concentrations achieves only $R^2 = 0.4679$ ($\text{RMSE} = 9.29$), proving that raw chemistry does not linearly give away the CCME WQI.
- Incorporating domain-engineered limnological features (DO deficit, stoichiometric $N:P$ ratio, buffer distance) advances linear $R^2$ to $0.9177$ ($\text{RMSE} = 3.65$).
- Non-linear tree ensembles capture the sharp threshold cliffs at regulatory limits to reach $R^2 = 0.9992$ ($\text{RMSE} = 0.35$).

---

### Q7: Why do we provide both `scaled=True` and `scaled=False` paths in `load_splits()`?
**Answer:** Tree ensembles (XGBoost, Random Forest) are invariant to monotonic feature scaling because split points depend only on feature rank ordering, not absolute numerical scale. Feeding unscaled inputs to tree models preserves exact domain interpretable units in split thresholds and SHAP explanations. Conversely, multi-layer perceptrons (MLP) trained via gradient descent and backpropagation require zero-mean, unit-variance standardized inputs ($\mathcal{N}(0, 1)$) to prevent exploding/vanishing gradients and uneven parameter updates. `scaled=False` serves the AML tree baselines; `scaled=True` serves Member 2's MLP architecture.

---

### Q8: How did you handle detection limits like `"<0.5 mg/L"` and physically impossible values?
**Answer:**
1. **Detection limits:** Laboratory sensors cannot measure below their Limit of Detection (LOD). Strings such as `"<0.5"` or `"<0.01"` cannot be converted to floats directly. Following standard environmental analytical chemistry conventions, we parse the numeric detection threshold $L$ and impute $L/2$ (half-detection-limit substitution), while simultaneously appending a boolean indicator flag (`flag_det_{analyte}`) to retain censoring information.
2. **Impossible values:** Environmental sensors periodically malfunction or record error codes (e.g. pH = -1.0 or 15.2, water temperature = -5°C or 48°C, DO $\le 0$ or $>20\,\text{mg/L}$, or negative nutrient concentrations). These are flagged as physically impossible, set to `np.nan`, and subsequently imputed via training-fold medians.

---

### Q9: What is the purpose of the `leakage_aware` config flag and `drop_features` list?
**Answer:** To rigorously evaluate whether deep neural networks learn true physical limnology or exploit mathematical proxies of Canadian regulatory equations, the framework supports a controlled feature ablation. When `leakage_aware: true` is activated, features listed in `drop_features` (such as `do_deficit`, `log_np_ratio`, `nitrate_to_n_ratio`, `ph_dist_buffer`) are automatically dropped from the feature matrix before batching. In our ablated baseline, XGBoost trained without these 4 direct proxies still reached $R^2 = 0.9991$ and $\text{RMSE} = 0.3791$, proving robust limnological generalization.

---

### Q10: How do the reproduced AML baseline metrics compare against the original AML README benchmarks?
**Answer:** The reproduced baseline bridge metrics strictly replicate the reported AML capstone benchmarks on unseen monitoring stations:
- **Tuned XGBoost:** Reported Test $R^2 = 0.9992$, $\text{RMSE} = 0.3496$, $\text{MAE} = 0.1012$. Bridge reproduced: Test $R^2 = 0.9992$, $\text{RMSE} = 0.3662$, $\text{MAE} = 0.1024$.
- **Random Forest:** Reported Test $R^2 = 0.9993$, $\text{RMSE} = 0.3310$, $\text{MAE} = 0.0955$. Bridge reproduced: Test $R^2 = 0.9993$, $\text{RMSE} = 0.3464$, $\text{MAE} = 0.0964$.
The minor numerical variation ($<0.001$ MAE, $<0.016$ RMSE) is due to representative sampling ($N=50,000$) from the 358,067 test observations for reproducible evaluation. The rankings, convergence properties, and $R^2$ benchmarks are identically preserved.
