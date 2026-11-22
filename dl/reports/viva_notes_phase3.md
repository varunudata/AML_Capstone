# Viva notes

1. **What exactly did we predict?**

   We predicted the supplied continuous CCME Water Quality Index. It summarizes how many parameters exceed their guidelines, how often they exceed them and by how much. The quality tiers are derived from the continuous score.

2. **How do we know the comparison used the same rows?**

   We replayed the original sampling rule on the frozen AML split archive. Every sampled partition matched the DL manifest in order. We then verified that the saved prediction files agree on validation and test row IDs and observed scores.

3. **Why use a paired site bootstrap?**

   The models see the same test observations, so the same resample should be applied to both. Observations from one site can be correlated. Resampling whole sites preserves that dependence within sites.

4. **How did the MLP compare with the trees?**

   Both tree baselines had lower test RMSE on the matched exports. The deep ensemble was the strongest DL variant for point accuracy, and its error was still higher than XGBoost's.

5. **Why give XGBoost a conformal interval?**

   A point regressor can still provide an interval. We found a radius from the saved validation errors and applied it to the saved test predictions. It gives a direct coverage-width comparison with the neural intervals.

6. **What do coverage and width measure?**

   Coverage counts how often the observed WQI lands inside the predicted interval. Width measures how informative the interval is. A very wide interval can achieve high coverage without offering a precise prediction.

7. **What did SHAP and permutation importance tell us?**

   SHAP attributes changes in a prediction to its inputs relative to a reference. Permutation importance measures the drop in score after shuffling one input. The tree ranked raw orthophosphate highest by permutation, and the MLP relied strongly on its log transform. Correlated features can divide the signal.

8. **What do PDP and ICE show?**

   ICE shows how a model's prediction changes for each selected row when one input varies. A PDP averages those traces. Holding engineered features fixed can create combinations that would rarely occur together in the field.

9. **What does the leakage ablation establish?**

   Removing four engineered proxies hurt both models, especially the seed-matched MLP. Raw chemistry can still encode parts of the index formula. A site-held-out evaluation tests new sites within the sampled setting. Forecasting future periods would need a separate test.
