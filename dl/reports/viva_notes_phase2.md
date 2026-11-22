# Phase 2 Viva Examination Notes: MLP Architecture, Training, Tuning & Calibrated Uncertainty

These technical viva notes explain the architectural, optimization, regularization, and uncertainty quantification design decisions implemented by **Member 2** in Phase 2 (`src/models.py`, `src/train.py`, `src/tune.py`, `src/uncertainty.py`, `notebooks/02_mlp_training.ipynb`).

---

### Q1: Why did we design a small, heavily regularized MLP (`~12k–20k` parameters) instead of a very deep or wide neural network?
**Answer:** Our tabular feature matrix has $D = 21$ continuous physical-chemical and stoichiometric features across $N = 250,000$ training observations. On low-dimensional tabular data, over-parameterized deep networks easily memorize station-specific noise and suffer from severe overfitting when evaluated on unseen monitoring stations (`GroupShuffleSplit` on `Area`). A compact pyramidal MLP (e.g., 2 to 3 hidden layers of width 64–256 with BatchNorm, Dropout, and decoupled $L_2$ weight decay) provides enough capacity to approximate smooth multi-sensor chemical interactions (temperature-dependent oxygen saturation and pH-ammonia toxicity curves) while keeping generalization gap minimal.

---

### Q2: Why use Kaiming (He) weight initialization together with training-mean output bias initialization?
**Answer:**
1. **Kaiming / He Normal Initialization (`fan_in`):** Rectified and smooth-rectified activations (ReLU, GELU, SiLU) zero out or attenuate negative pre-activations, halving activation variance at each layer under standard Xavier initialization. He initialization scales weights by $\sqrt{2 / d_{\text{in}}}$, preserving unit activation variance in the forward pass and preventing vanishing or exploding gradients in the backward pass.
2. **Training-Mean Output Bias Initialization:** Continuous `CCME_Values` lie in $[31.30, 100.0]$ with a training mean of $\approx 88.5$. If the final linear regression head starts with bias $0.0$, initial predictions are $\approx 0.0$ ($\text{RMSE} \approx 89.0$), wasting the first 15+ epochs simply shifting the output offset up to $88.5$. Initializing `reg_head.bias` strictly to `mean(y_train)` starts the network at the trivial mean predictor ($R^2 = 0.0$, $\text{RMSE} \approx 13.4$) at step 0, allowing gradient updates to immediately refine hidden feature interactions.

---

### Q3: Why did we choose `AdamW` over standard `Adam` or `SGD`?
**Answer:** In standard `Adam`, $L_2$ regularization is added directly to the loss gradient ($\nabla \mathcal{L} + \lambda \theta$), which then gets divided by the adaptive second-moment estimate $\sqrt{v_t}$. As a result, weights with large historical gradients receive **less** effective weight decay than weights with small gradients, coupling learning rate and regularization. **`AdamW` decouples weight decay** from the adaptive gradient step ($\theta_{t+1} = (1 - \eta \lambda)\theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$), applying uniform proportional shrinkage across all weights and significantly improving tabular generalization during cosine learning rate decay.

---

### Q4: Why compare Huber loss against Mean Squared Error (MSE), and why is Huber effective on `CCME_Values`?
**Answer:** `CCME_Values` exhibits two structural challenges:
1. **41.39% Right-Censoring at 100.0:** Nearly half the dataset sits at the hard regulatory ceiling of $100.0$.
2. **Heavy-Tailed Pollution Events:** Episodic agricultural or sewage runoff produces extreme low-WQI outliers ($WQI \in [31.3, 45.0]$).
MSE squares every residual ($e^2$), causing rare tail outliers to dominate gradient updates and pulling predictions away from the dense high-WQI mode. **Huber loss** behaves quadratically ($0.5 e^2$) for small errors ($|e| \le \delta$) to preserve smooth convergence near the optimum, and linearly ($\delta(|e| - 0.5\delta)$) for large errors ($|e| > \delta$), bounding the gradient contribution of any single extreme observation and improving Mean Absolute Error (MAE).

---

### Q5: Why is early stopping based strictly on grouped validation RMSE, accompanied by gradient norm clipping?
**Answer:**
- **Early Stopping on Validation RMSE:** Training loss decreases monotonically with additional epochs, whereas validation error on unseen monitoring stations (`Area`) reaches a minimum and then degrades as the network begins memorizing training stations. Monitoring validation RMSE with a patience window and restoring the best checkpoint weights (`best_state_dict`) acts as a non-parametric regularizer in time. Critically, the test split is **never** touched during early stopping or model selection.
- **Gradient Clipping (`max_grad_norm = 1.0`):** Even after `log1p` transformation and `StandardScaler`, multi-parameter chemical spikes can create steep local loss surfaces. Clipping the global $L_2$ norm of parameter gradients prevents a single anomalous mini-batch from destabilizing learned weights.

---

### Q6: Why did we run the 4-part Sanity Suite (32-sample overfit, shuffled-target test, trivial baselines, learning curves) BEFORE running Optuna tuning?
**Answer:** Tuning hyperparameters on a buggy pipeline wastes compute and masks silent defects:
1. **32-Sample Mini-Batch Overfit:** Confirms that gradients flow properly through all layers and the architecture has sufficient capacity to drive training MSE to $\approx 0.0$ ($R^2 = 1.0$).
2. **Shuffled-Target Permutation Test:** When `y_train` is randomly permuted, test $R^2$ collapses to $\approx 0.0$ ($0.0129$). If test $R^2$ had remained significantly positive, it would have exposed fatal feature-target or train-test preprocessing leakage.
3. **Trivial Baselines (Mean & Ridge):** Establishes the performance floor on the exact split (Mean predictor Test $\text{RMSE} = 12.88, R^2 = -0.0235$; Linear Ridge Test $\text{RMSE} = 3.29, R^2 = 0.9333$). Any useful MLP must substantially outperform Ridge's $\text{RMSE} = 3.29$.

---

### Q7: How did we ensure a fair hyperparameter tuning comparison against the AML tree baselines?
**Answer:** A common pitfall in tabular ML vs. DL comparisons is giving one model family 100 tuning trials and the other default hyperparameters. We inspected `results/aml_baseline_metrics.json` from Phase 1 and matched the exact AML Optuna budget (**15 Optuna trials**). Every trial trained on the training split and evaluated **exclusively on the site-held-out validation split (`val_RMSE`)**, exploring depth ($1\text{--}4$ layers), width ($16\text{--}256$), activation (`relu`, `gelu`, `silu`), `BatchNorm` (on/off), dropout ($0.0\text{--}0.5$), learning rate ($10^{-4}\text{--}10^{-2}$), and weight decay ($10^{-6}\text{--}10^{-2}$).

---

### Q8: During Monte Carlo (MC) Dropout inference ($T=50$ passes), why must `BatchNorm` layers stay in `.eval()` mode while `Dropout` layers are placed in `.train()` mode?
**Answer:** Following Gal & Ghahramani (2016), MC Dropout approximates Bayesian posterior inference by sampling $T=50$ stochastic dropout masks at test time. However, if one simply calls `model.train()` during test inference, **`BatchNorm1d` layers also switch to training mode**—updating their running mean/variance using test data and normalizing each test batch by its own batch statistics! That violates test isolation and corrupts predictions. Our helper `_enable_mc_dropout_only()` keeps the network in `model.eval()` (freezing BatchNorm running statistics from training) and selectively toggles only `nn.Dropout` modules to `.train()`.

---

### Q9: How do our three uncertainty quantification methods (MC Dropout, 5-Seed Deep Ensemble, and Pinball Quantile MLP) differ in what uncertainty they capture?
**Answer:**
1. **MC Dropout ($T=50$ passes):** Samples sub-networks from a single trained weight basin via Bernoulli dropout masks. It is computationally cheap (1 training run, 50 fast forward passes) but tends to underestimate epistemic uncertainty because all sub-networks share the same local minimum.
2. **5-Seed Deep Ensemble (Lakshminarayanan et al., 2017):** Trains $M=5$ independent networks from distinct Kaiming random initializations and mini-batch shuffle orders (`seeds 42..46`). Different seeds converge to distinct basins of attraction in the non-convex loss landscape, providing stronger **epistemic uncertainty** on out-of-distribution stations and lower point-prediction RMSE via variance reduction.
3. **Quantile Regression MLP (`PinballQuantileLoss` for $q_{05}, q_{50}, q_{95}$):** Directly models **aleatoric (data) uncertainty** and asymmetric residual skew without assuming symmetric Gaussian errors. Because `CCME_Values` is right-censored at $100.0$, prediction intervals near $98\text{--}100$ are naturally asymmetric (tight upper bound at $100$, wider lower tail). Combining Pinball loss with non-crossing monotonicity regularization and split-conformal validation calibration achieves $\approx 90\%$ empirical coverage (`PICP_90`).

---

### Q10: What did the `leakage_aware=true` run (`dl_mlp_leakaware`) reveal about the MLP vs. XGBoost?
**Answer:** When `leakage_aware=true` drops the 4 direct stoichiometric formula proxies (`do_deficit`, `log_np_ratio`, `nitrate_to_n_ratio`, `ph_dist_buffer`), reducing the input dimension from 21 to 17 features, the MLP must learn oxygen saturation and nutrient stoichiometry directly from raw temperature, dissolved oxygen, nitrogen,orthophosphate, and pH. The ablated MLP still achieves $R^2 > 0.996$ on unseen test stations (within $\sim 0.05\text{--}0.10$ RMSE of the full 21-feature MLP), confirming that the neural network learns genuine multi-sensor physical-chemical interactions rather than relying solely on pre-computed domain ratios.
