| model            | reference   | unit              |   delta_rmse |   ci_low |   ci_high |
|:-----------------|:------------|:------------------|-------------:|---------:|----------:|
| dl_mlp           | aml_xgboost | row (exploratory) |       0.2045 |   0.1856 |    0.2249 |
| dl_mlp_mcdropout | aml_xgboost | row (exploratory) |       0.2385 |   0.2194 |    0.2585 |
| dl_mlp_ensemble  | aml_xgboost | row (exploratory) |       0.1719 |   0.1528 |    0.1926 |
| dl_mlp_quantile  | aml_xgboost | row (exploratory) |       0.2487 |   0.2301 |    0.2691 |
