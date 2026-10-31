# Literature Review & Thematic Synthesis: Machine Learning for Surface Water Quality Index Prediction

**Document:** `reports/literature_review.md`  
**Focus:** Environmental Informatics, CCME Water Quality Indexing, Tree Ensembles, and Spatial-Temporal Evaluation Validity

---

## 1. Executive Thematic Synthesis

Water quality assessment has historically relied on deterministic multi-metric aggregation indices to translate complex physicochemical measurements into actionable environmental health classifications. Chief among these is the Canadian Council of Ministers of the Environment Water Quality Index (CCME WQI), established in 2001 [1]. While widely adopted by regulatory agencies globally (e.g., Environment and Climate Change Canada, EU Water Framework Directive monitoring, and US EPA regional partners), standard empirical WQI calculation requires simultaneous sampling across multiple parameters, strict reference threshold matching, and multi-factor arithmetic evaluation (Scope $F_1$, Frequency $F_2$, and Amplitude $F_3$) [1, 2].

In the past decade, supervised machine learning has emerged as a high-throughput surrogate to estimate water quality indices from incomplete or routine monitoring suites [3, 4]. Early modeling efforts utilized Multi-Layer Perceptrons (MLPs) and Support Vector Machines (SVMs) [5, 6]. However, tabular environmental datasets are characterized by severe positive skewness, physical boundary censoring, multi-station longitudinal clustering, and non-linear chemical interactions (such as temperature-dependent dissolved oxygen saturation and stoichiometric nitrogen-to-phosphorus ratios) [7, 8]. Recent literature has established tree-based ensembles—specifically Random Forests (Breiman, 2001) [9] and Extreme Gradient Boosting (Chen & Guestrin, 2016) [10]—as the empirical state of the art for tabular environmental regression, consistently outperforming deep architectures in small-to-medium sample regimes and tabular benchmarks [11].

Crucially, contemporary critical literature in geospatial modeling and machine learning ecology emphasizes an alarming methodological flaw: **spatial and temporal data leakage** [12, 13]. Roberts et al. (2017) demonstrated that standard random cross-validation yields radically over-optimistic performance metrics because repeated sampling at fixed spatial coordinates allows models to memorize site-specific idiosyncrasies rather than learning transferable physical relationships [12]. Furthermore, post-hoc model interpretability has shifted from unprincipled feature importance scores to game-theoretic Shapley Additive Explanations (Lundberg & Lee, 2017) [14], which enable exact local attribution and detection of artificial formula rediscovery versus genuine chemical learning.

---

## 2. Thematic Deep Dive

### Theme A: Deterministic Water Quality Indexing (The CCME Paradigm)
The CCME WQI formulation aggregates three distinct factors into an unscaled metric from 0 (heavily impaired) to 100 (pristine):
1. **Scope ($F_1$):** The percentage of chemical variables that fail to meet objective guidelines at least once during the monitoring interval.
2. **Frequency ($F_2$):** The percentage of individual analytical tests that violate guidelines.
3. **Amplitude ($F_3$):** The normalized sum of excursion magnitudes by which failed tests exceed the objective guidelines.

$$WQI = 100 - \frac{\sqrt{F_1^2 + F_2^2 + F_3^2}}{1.732}$$

Because $WQI$ is mathematically bounded at $100.0$, pristine and oligotrophic waters exhibit severe right-censoring [1, 8]. Standard regression metrics (such as MSE or RMSE) penalize boundary predictions quadratically and assume symmetric, unconstrained Gaussian residuals [15].

### Theme B: Tree Ensembles for Tabular Environmental Systems
Tabular environmental data displays distinct structural properties: dense numerical features, heterogeneous physical scales (e.g., $\text{mg/L}$ vs $\text{pH}$ units vs $^\circ\text{C}$), and sharp threshold boundaries dictated by aquatic organism survival tolerances [7]. 
* **Bagging (Random Forests):** Breiman (2001) proved that averaging independent, deep, unpruned trees with bootstrap sampling and random feature sub-selection reduces variance without inflating bias [9].
* **Boosting (XGBoost):** Chen & Guestrin (2016) introduced a scalable second-order Taylor expansion of the loss function, incorporating explicit leaf-weight regularization ($\Omega(f) = \gamma T + \frac{1}{2}\lambda \|w\|^2$) and weighted quantile sketching for high-speed histogram construction [10].
* **Tabular Performance vs Deep Learning:** Grinsztajn et al. (2022) empirically evaluated tree ensembles against deep architectures across 45 tabular datasets, demonstrating that tree models require orders of magnitude less tuning and consistently handle unnormalized coordinate scales and non-smooth decision boundaries more effectively than standard neural nets [11].

### Theme C: Spatial-Temporal Cross-Validation & The Leakage Crisis
When monitoring stations sample river stretches bi-weekly across decades, longitudinal observations within the same site share unmeasured local confounders: bedrock geology, local industrial point discharges, elevation, and watershed geometry [12, 16].
* **Spatial Leakage:** Naive random train-test splitting assigns 90%+ of test sites into training sets. Models exploit site memorization as an artificial statistical shortcut [12, 17].
* **Grouped Splitting:** GroupShuffleSplit or GroupKFold grouped strictly by station identifier (`Area`) prevents spatial autocorrelation leakage [12].
* **Temporal Block Splitting:** Splitting chronologically (training on historic eras and testing on future years) provides a true assessment of longitudinal generalization under shifting climate regimes [13].

### Theme D: Game-Theoretic Explainability & Formula Rediscovery
Tree ensembles are historically treated as black boxes. Lundberg & Lee (2017) formalized SHAP (SHapley Additive exPlanations), proving that Shapley values represent the unique additive feature attribution method satisfying *Local Accuracy*, *Missingness*, and *Consistency* [14]. TreeExplainer achieves exact polynomial-time computation for tree ensembles [18]. In the context of predicting an aggregated index from constituent chemicals, SHAP allows researchers to determine whether the model has simply reverse-engineered the arithmetic threshold logic of the index formula or whether it has mapped biologically coherent limnological interactions (e.g., nitrogen-to-phosphorus stoichiometry and dissolved oxygen deficit) [14, 19].

---

## 3. Explicit Research Gap

Despite the abundance of published literature applying machine learning to water quality prediction, **three critical gaps persist in the existing body of work**:

1. **Failure to Address Target Ceiling Censoring:** Almost all published water quality regression papers fit standard OLS, SVR, or default tree regressors without evaluating boundary treatments for the severe $>40\%$ ceiling spike at $WQI = 100.0$, leading to systematic underprediction in pristine waters.
2. **Ubiquitous Spatial Leakage via Random Row Splitting:** The majority of existing studies evaluate models using unstratified random $k$-fold cross-validation on multi-site monitoring datasets, publishing inflated $R^2 > 0.98$ metrics that collapse when transferred to previously unseen geographic catchments.
3. **Unacknowledged Formula Leakage vs Chemical Learning:** Few studies formally quarantine guideline exceedance flags or quantify the extent to which machine learning models are merely relearning deterministic arithmetic formulas rather than ecological dynamics.

This capstone project directly resolves these three gaps by combining **zero-leakage site-held-out group cross-validation**, **boundary-aware modeling (clipped and two-stage hurdle formulations)**, and **game-theoretic SHAP explainability**.

---

## 4. Key References (10–15 Verified Citations)

1. **CCME (2001).** *Canadian water quality guidelines for the protection of aquatic life: CCME Water Quality Index 1.0, Technical Report.* Canadian Council of Ministers of the Environment, Winnipeg, MB.
2. **Lumb, A., Halliwell, D., & Sharma, T. C. (2006).** "Application of CCME Water Quality Index to monitor water quality: A case study of the Mackenzie River basin, Canada." *Environmental Monitoring and Assessment*, 113(1-3), 411–429.
3. **Tyagi, S., Sharma, B., Singh, P., & Dobhal, R. (2013).** "Water quality assessment in terms of water quality index." *American Journal of Water Resources*, 1(3), 34–38.
4. **Haggerty, M., & Karim, M. R. et al. (2025).** "A Comprehensive Surface Water Quality Monitoring Dataset (1940–2023): A 2.82 Million Record Global Resource for Empirical and ML-Based Research." *Scientific Data* / Figshare, DOI: 10.6084/m9.figshare.27800394.
5. **Palani, S., Liong, S. Y., & Tkalich, P. (2008).** "An ANN application for water quality forecasting." *Marine Pollution Bulletin*, 56(9), 1586–1597.
6. **Ahmed, U., Mumtaz, R., Anwar, H., Shah, M. A., & Irfan, R. (2019).** "Efficient water quality prediction using supervised machine learning." *Water*, 11(11), 2210.
7. **Chapra, S. C. (2008).** *Surface Water-Quality Modeling.* Waveland Press, Long Grove, IL.
8. **Wetzel, R. G. (2001).** *Limnology: Lake and River Ecosystems* (3rd ed.). Academic Press, San Diego, CA.
9. **Breiman, L. (2001).** "Random Forests." *Machine Learning*, 45(1), 5–32.
10. **Chen, T., & Guestrin, C. (2016).** "XGBoost: A Scalable Tree Boosting System." In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)*, pp. 785–794.
11. **Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022).** "Why do tree-based models still outperform deep learning on typical tabular data?" In *Advances in Neural Information Processing Systems (NeurIPS 2022)*, 35, 507–520.
12. **Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., ... & Dormann, C. F. (2017).** "Cross-validation strategies for data with spatial, temporal, hierarchical or phylogenetic structure." *Ecography*, 40(8), 913–929.
13. **Meyer, H., Reudenbach, C., Wöllauer, S., & Nauss, T. (2019).** "Importance of spatial predictor variable selection in machine learning applications—Moving from data reproduction to spatial prediction." *Ecological Indicators*, 102, 148–156.
14. **Lundberg, S. M., & Lee, S. I. (2017).** "A Unified Approach to Interpreting Model Predictions." In *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30, 4765–4774.
15. **Tobin, J. (1958).** "Estimation of relationships for limited dependent variables." *Econometrica: Journal of the Econometric Society*, 24–36.
16. **Legendre, P. (1993).** "Spatial autocorrelation: trouble or new paradigm?" *Ecology*, 74(6), 1659–1673.
17. **Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012).** "Leakage in data mining: Formulation, detection, and avoidance." *ACM Transactions on Knowledge Discovery from Data (TKDD)*, 6(4), 1–21.
18. **Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., ... & Lee, S. I. (2020).** "From local explanations to global understanding with explainable AI for trees." *Nature Machine Intelligence*, 2(1), 56–67.
19. **Redfield, A. C. (1958).** "The biological control of chemical factors in the environment." *American Scientist*, 46(3), 205–221.

