| model             | parameters   | train_seconds   | test_inference_seconds   |
|:------------------|:-------------|:----------------|:-------------------------|
| aml_xgboost       | n/a          | n/a             | n/a                      |
| aml_random_forest | n/a          | n/a             | n/a                      |
| dl_mlp            | 13633.0      | 13.07           | 0.0095                   |
| dl_mlp_mcdropout  | 13633.0      | 13.07           | 1.924                    |
| dl_mlp_ensemble   | 68165.0      | 65.36           | 0.0474                   |
| dl_mlp_quantile   | 13699.0      | 12.67           | 0.0064                   |
| dl_mlp_leakaware  | 13121.0      | 8.4             | 0.0076                   |
