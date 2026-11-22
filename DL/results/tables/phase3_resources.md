| model             | checkpoint_bytes   | parameters   | train_seconds   | test_inference_seconds   |
|:------------------|:-------------------|:-------------|:----------------|:-------------------------|
| aml_xgboost       | 559528.0           | n/a          | n/a             | n/a                      |
| aml_random_forest | n/a                | n/a          | n/a             | n/a                      |
| dl_mlp            | 64780.0            | 13633.0      | 13.07           | 0.0095                   |
| dl_mlp_mcdropout  | n/a                | 13633.0      | 13.07           | 1.924                    |
| dl_mlp_ensemble   | n/a                | 68165.0      | 65.36           | 0.0474                   |
| dl_mlp_quantile   | n/a                | 13699.0      | 12.67           | 0.0064                   |
| dl_mlp_leakaware  | n/a                | 13121.0      | 8.4             | 0.0076                   |
