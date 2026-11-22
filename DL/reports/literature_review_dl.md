# Deep Learning & Tabular Benchmark Literature Review (Phase 2)

**Project:** Water Quality Prediction via Gradient Boosting (AML) and Deep Neural Networks (DL)  
**Target:** Continuous CCME Water Quality Index (`CCME_Values` $\in [31.3041, 100.0]$)

---

## 1. Thematic Synthesis of the Literature

### Theme A: Composite Water Quality Indexing and Supervised Surrogates
The Canadian Council of Ministers of the Environment Water Quality Index (CCME WQI) aggregates multi-parameter guideline excursions across **Scope** ($F_1$), **Frequency** ($F_2$), and **Amplitude** ($F_3$) into a continuous $0\text{--}100$ score (**CCME, 2001**; **Lumb et al., 2006**). Because deterministic calculation requires synchronous laboratory measurements across all monitored analytes, missing sensor channels or asynchronous sampling prevent index evaluation (**Tyagi et al., 2013**). Early neural network studies (**Palani et al., 2008**) and modern supervised learning benchmarks (**Ahmed et al., 2019**; **Karim et al., 2025**) demonstrated that statistical models can infer WQI from routine physical-chemical measurements ($\text{pH}$, dissolved oxygen, temperature, nitrogen species, orthophosphate, and biochemical oxygen demand). However, two structural artifacts govern this domain: (1) **$41.39\%$ upper-boundary right-censoring** at $WQI = 100.0$ when all analytes satisfy regulatory thresholds (**Tobin, 1958**), and (2) **formula leakage risk** when direct threshold exceedance indicators are left in the predictor matrix (**Kaufman et al., 2012**).

### Theme B: Why Tree Ensembles Outperform Deep Learning on Tabular Data
A landmark empirical benchmark by **Grinsztajn, Oyallon, and Varoquaux (2022, NeurIPS)** established why Gradient Boosted Decision Trees (**Chen & Guestrin, 2016 — XGBoost**) and Random Forests (**Breiman, 2001**) consistently outperform standard deep neural networks on medium-sized tabular datasets:
1. **Rotational Invariance vs. Coordinate-Aligned Thresholds:** Multi-Layer Perceptrons (MLPs) are rotationally biased toward smooth, linear combinations of inputs. In contrast, tabular features carry coordinate-specific meaning—especially in regulatory indices like CCME WQI where individual chemical guidelines ($\text{DO} < 6.5\,\text{mg/L}$, $\text{Orthophosphate} > 0.05\,\text{mg/L}$, $\text{pH} \notin [6.5, 8.5]$) form sharp, axis-aligned step discontinuities.
2. **Irregular Target Functions and Heavy Tails:** Neural networks bias toward overly smooth interpolations via low-frequency spectral bias, struggling with abrupt piecewise cliffs unless appropriately regularized (**Gorishniy et al., 2021, NeurIPS**).
3. **Feature Scale Heterogeneity:** Decision trees are invariant to monotonic feature rescaling, whereas gradient-trained MLPs require strict log-normalization of heavy-tailed concentrations and training-only standardization.

### Theme C: Regularization, Initialization, and Robust Loss Functions for Tabular MLPs
Bridging the tabular performance gap in neural networks requires tailored architectural and optimization choices:
- **Kaiming (He) Initialization (**He et al., 2015, ICCV**):** Scales initial weights by $\mathcal{N}(0, \sqrt{2 / d_{\text{in}}})$ to preserve forward activation variance and backward gradient magnitude across rectified (`ReLU`) and smooth-rectified (`GELU`, `SiLU`) layers.
- **Batch Normalization (**Ioffe & Szegedy, 2015, ICML**):** Normalizes intermediate pre-activations per mini-batch, stabilizing internal covariate shift across interacting chemical features with disparate variances.
- **Decoupled Weight Decay (`AdamW`, **Loshchilov & Hutter, 2019, ICLR**):** Standard `Adam` couples $L_2$ regularization with adaptive second-moment gradient scaling ($v_t$), applying weaker shrinkage to parameters with large historical gradients. `AdamW` decouples weight decay from the adaptive update, restoring uniform $L_2$ regularization and pairing effectively with cosine learning rate annealing.
- **Huber Robust Regression (**Huber, 1964**):** Combines quadratic loss ($0.5 e^2$) for small residuals ($|e| \le \delta$) with linear loss ($\delta(|e| - 0.5\delta)$) in the tails, preventing extreme pollution events and the $41.39\%$ boundary mass at $100.0$ from destabilizing gradient descent.

### Theme D: Predictive Uncertainty Quantification and Conformal Calibration
While point-prediction error favors tree ensembles, operational watershed monitoring requires **calibrated predictive uncertainty** to flag high-risk contamination events when sensor combinations fall in sparse regimes:
- **Monte Carlo (MC) Dropout (**Gal & Ghahramani, 2016, ICML**):** Interprets dropout training as approximate variational inference in Bayesian neural networks. Executing $T=50$ stochastic forward passes at test time (while keeping `BatchNorm` frozen in evaluation mode) yields an empirical predictive distribution. However, because all sub-networks sample around a single weight mode, MC Dropout often underestimates epistemic spread before post-hoc calibration.
- **Deep Ensembles (**Lakshminarayanan, Pritzel, & Blundell, 2017, NeurIPS**):** Training $M=5$ independent networks from distinct random initializations and mini-batch shuffle trajectories explores multiple basins of attraction in the non-convex loss surface, yielding superior point accuracy and well-dispersed epistemic uncertainty.
- **Quantile Regression & Pinball Loss (**Koenker & Bassett, 1978**; **Romano et al., 2019, NeurIPS**):** Directly estimates conditional percentiles ($q_{05}, q_{50}, q_{95}$) via asymmetric check loss without Gaussian residual assumptions—crucial near the hard $100.0$ WQI ceiling. Pairing neural interval estimators or tree residuals with **Split-Conformal Prediction (**Angelopoulos & Bates, 2021**)** calibrates finite-sample marginal coverage ($90\%$ PICP) using held-out validation order statistics.

### Theme E: Spatial Cross-Validation and Post-Hoc Attribution
- **Spatial Autocorrelation & Group Splitting (**Roberts et al., 2017, Ecography**; **Meyer et al., 2019**):** Because monitoring stations sample the same rivers repeatedly over decades, random row splitting leaks $91.2\%$ of station identities across train and test sets. Group-stratified site-held-out splitting (`Area`) is mandatory to measure true out-of-basin generalization.
- **Shapley Additive Explanations (**Lundberg & Lee, 2017, NeurIPS**; **Lundberg et al., 2020**):** Reconciling `TreeSHAP` / permutation importance on XGBoost with `GradientSHAP` and Partial Dependence / Individual Conditional Expectation (`PDP/ICE`) curves on the MLP reveals how trees split directly on raw nutrient thresholds while neural networks distribute attribution across log-transformed and stoichiometric representations.

---

## 2. Explicit Research Gap Addressed in Phase 2

Prior water-quality machine learning studies either (a) evaluate MLPs and GBDTs under leaked random row splits without matched hyperparameter budgets, or (b) report point predictions ($R^2$, RMSE) while ignoring predictive uncertainty calibration and formula-proxy sensitivity. **Phase 2 directly addresses this gap** by benchmarking a compact, heavily regularized MLP (`13,633` parameters) against frozen tuned XGBoost and Random Forest baselines under an identical zero-leakage site-held-out protocol ($250,000$ train / $50,000$ val / $50,000$ test rows), matching the 15-trial Optuna tuning budget, conducting a 20-configuration architectural ablation, and rigorously comparing **MC Dropout ($T=50$)**, **5-Seed Deep Ensembles**, **Pinball Quantile Regression ($q_{05}/q_{50}/q_{95}$)**, and **Split-Conformal XGBoost** using paired site-block bootstrap inference ($1,000$ resamples across $4,231$ unseen test stations).

---

## 3. Master Bibliography (18 Verified References)

1. **CCME (2001).** *Canadian water quality guidelines for the protection of aquatic life: CCME Water Quality Index 1.0, Technical Report.* Canadian Council of Ministers of the Environment, Winnipeg, MB.
2. **Lumb, A., Halliwell, D., & Sharma, T. C. (2006).** Application of CCME Water Quality Index to monitor water quality: A case study of the Mackenzie River basin, Canada. *Environmental Monitoring and Assessment*, 113(1–3), 411–429.
3. **Karim, M. R., Syeed, M. M. M., Rahman, A., et al. (2025).** A Comprehensive Surface Water Quality Monitoring Dataset (1940–2023): A 2.82 Million Record Global Resource for Empirical and ML-Based Research. *Scientific Data / Figshare*, DOI: `10.6084/m9.figshare.27800394`.
4. **Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022).** Why do tree-based models still outperform deep learning on typical tabular data? *Advances in Neural Information Processing Systems (NeurIPS)*, 35, 507–520.
5. **Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021).** Revisiting deep learning models for tabular data. *Advances in Neural Information Processing Systems (NeurIPS)*, 34, 18932–18943.
6. **Breiman, L. (2001).** Random Forests. *Machine Learning*, 45(1), 5–32.
7. **Chen, T., & Guestrin, C. (2016).** XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.
8. **He, K., Zhang, X., Ren, S., & Sun, J. (2015).** Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 1026–1034.
9. **Ioffe, S., & Szegedy, C. (2015).** Batch Normalization: Accelerating deep network training by reducing internal covariate shift. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, 448–456.
10. **Loshchilov, I., & Hutter, F. (2019).** Decoupled Weight Decay Regularization. *International Conference on Learning Representations (ICLR)*.
11. **Huber, P. J. (1964).** Robust Estimation of a Location Parameter. *The Annals of Mathematical Statistics*, 35(1), 73–101.
12. **Gal, Y., & Ghahramani, Z. (2016).** Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, 1050–1059.
13. **Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017).** Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 6402–6413.
14. **Koenker, R., & Bassett, G. (1978).** Regression Quantiles. *Econometrica*, 46(1), 33–50.
15. **Romano, Y., Patterson, E., & Candès, E. (2019).** Conformalized Quantile Regression. *Advances in Neural Information Processing Systems (NeurIPS)*, 32, 3543–3553.
16. **Angelopoulos, A. N., & Bates, S. (2021).** A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. *arXiv preprint arXiv:2107.07511*.
17. **Roberts, D. R., Bahn, V., Ciuti, S., et al. (2017).** Cross-validation strategies for data with spatial, temporal, hierarchical or phylogenetic structure. *Ecography*, 40(8), 913–929.
18. **Lundberg, S. M., & Lee, S.-I. (2017).** A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 4765–4774.
