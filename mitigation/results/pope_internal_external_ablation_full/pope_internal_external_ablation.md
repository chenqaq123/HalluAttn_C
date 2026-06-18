# POPE Internal vs External Ablation

| Method | Score | Selection | Detector calls | MCC | TPR | FPR | Yes rate |
|---|---|---:|---:|---:|---:|---:|---:|
| vanilla_base | anchor | 0.00 | 0 | 0.730 | 0.813 | 0.087 | 0.450 |
| lh_alone_base_yes_suppress | lh_shape_pope_layers_22_31 | 0.50 | 0 | 0.495 | 0.432 | 0.018 | 0.225 |
| lh_to_tdev_base_yes | lh_shape_pope_layers_22_31 | 0.50 | 2025 | 0.754 | 0.801 | 0.056 | 0.428 |
| lh_to_tdev_all | lh_shape_pope_layers_22_31 | 0.50 | 4500 | 0.746 | 0.812 | 0.071 | 0.442 |
| lh_to_tdev_base_yes | prompt_token_pos | 0.50 | 2025 | 0.741 | 0.805 | 0.070 | 0.438 |
| lh_to_tdev_base_yes | target_char_len | 0.50 | 2025 | 0.743 | 0.802 | 0.066 | 0.434 |
| full_tdev | anchor | 1.00 | 9000 | 0.763 | 0.806 | 0.051 | 0.429 |
