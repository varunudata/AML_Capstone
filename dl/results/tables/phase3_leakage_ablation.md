| split   | model       |   seed | metric   |   full |   ablated |   delta_ablated_minus_full | dropped_features                                             |
|:--------|:------------|-------:|:---------|-------:|----------:|---------------------------:|:-------------------------------------------------------------|
| val     | aml_xgboost |     42 | MAE      | 0.1309 |    0.1543 |                     0.0234 | do_deficit, log_np_ratio, nitrate_to_n_ratio, ph_dist_buffer |
| val     | aml_xgboost |     42 | RMSE     | 0.4192 |    0.4387 |                     0.0195 | do_deficit, log_np_ratio, nitrate_to_n_ratio, ph_dist_buffer |
| val     | aml_xgboost |     42 | R2       | 0.9990 |    0.9989 |                    -0.0001 | do_deficit, log_np_ratio, nitrate_to_n_ratio, ph_dist_buffer |
| val     | aml_xgboost |     42 | Macro_F1 | 0.9707 |    0.9674 |                    -0.0033 | do_deficit, log_np_ratio, nitrate_to_n_ratio, ph_dist_buffer |
| val     | dl_mlp      |     42 | MAE      | 0.2847 |    0.3550 |                     0.0703 | do_deficit, log_np_ratio, nitrate_to_n_ratio, ph_dist_buffer |
| val     | dl_mlp      |     42 | RMSE     | 0.6294 |    0.8119 |                     0.1824 | do_deficit, log_np_ratio, nitrate_to_n_ratio, ph_dist_buffer |
| val     | dl_mlp      |     42 | R2       | 0.9978 |    0.9963 |                    -0.0015 | do_deficit, log_np_ratio, nitrate_to_n_ratio, ph_dist_buffer |
| val     | dl_mlp      |     42 | Macro_F1 | 0.9685 |    0.9423 |                    -0.0262 | do_deficit, log_np_ratio, nitrate_to_n_ratio, ph_dist_buffer |
| test    | aml_xgboost |     42 | MAE      | 0.1024 |    0.1216 |                     0.0192 | do_deficit, log_np_ratio, nitrate_to_n_ratio, ph_dist_buffer |
| test    | aml_xgboost |     42 | RMSE     | 0.3662 |    0.3791 |                     0.0129 | do_deficit, log_np_ratio, nitrate_to_n_ratio, ph_dist_buffer |
| test    | aml_xgboost |     42 | R2       | 0.9992 |    0.9991 |                    -0.0001 | do_deficit, log_np_ratio, nitrate_to_n_ratio, ph_dist_buffer |
| test    | aml_xgboost |     42 | Macro_F1 | 0.9660 |    0.9655 |                    -0.0005 | do_deficit, log_np_ratio, nitrate_to_n_ratio, ph_dist_buffer |
| test    | dl_mlp      |     42 | MAE      | 0.2497 |    0.2803 |                     0.0306 | do_deficit, log_np_ratio, nitrate_to_n_ratio, ph_dist_buffer |
| test    | dl_mlp      |     42 | RMSE     | 0.5470 |    0.7124 |                     0.1654 | do_deficit, log_np_ratio, nitrate_to_n_ratio, ph_dist_buffer |
| test    | dl_mlp      |     42 | R2       | 0.9982 |    0.9969 |                    -0.0013 | do_deficit, log_np_ratio, nitrate_to_n_ratio, ph_dist_buffer |
| test    | dl_mlp      |     42 | Macro_F1 | 0.9582 |    0.9251 |                    -0.0332 | do_deficit, log_np_ratio, nitrate_to_n_ratio, ph_dist_buffer |
