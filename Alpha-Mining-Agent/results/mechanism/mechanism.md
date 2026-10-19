# Failure-mechanism analysis (exploratory, not pre-specified)

Uses only training / validation data and the logs; the Group C triggers were checked round by round against the diagnosis text in the logged prompts.
Group A / B triggers are a counterfactual replay (what a diagnosis would have flagged), used as a control.

## 1. Rule triggers (mean per run, 20 rounds)

| config | R1 | R2 | R3 | distinct_formulas |
| --- | --- | --- | --- | --- |
| deepseek_A | 5.0 | 4.6 | 15.4 | 17.6 |
| deepseek_A_think | 0.3 | 0.7 | 21.0 | 18.3 |
| deepseek_B | 4.4 | 14.2 | 9.6 | 13.4 |
| deepseek_C | 5.4 | 12.0 | 12.0 | 16.4 |
| deepseek_C_think | 8.0 | 2.0 | 14.3 | 15.3 |
| qwen_A | 6.6 | 4.4 | 12.8 | 16.6 |
| qwen_B | 14.0 | 18.2 | 6.6 | 16.8 |
| qwen_C | 16.4 | 7.0 | 11.0 | 16.8 |

## 2. Next-round response after a trigger

kids: next-round candidates that contain the body of the diagnosed formula; fixed: those modified as advised (R1 adds ts_std, R2 adds ts_decay_linear/ts_mean, R3 adds volume/dollar_vol); d_*: mean change of the valid fixed candidates relative to the parent; fixed_better_share: share of fixed candidates with higher fitness than the parent; below_r2_share: share whose turnover is below the R2 threshold; repeat: share of triggers with the same formula and rule as in the previous round; parent_at_depth_limit: share of diagnosed formulas already at the depth limit of 6. A candidate is derived from a formula if they share a subtree covering more than half of the formula's nodes (ignoring windows, constants and +c offsets).

| config | rule | triggers | kids_per | fixed_per | any_fixed | d_turnover | d_absicir | d_fitness | d_maxcorr | fixed_ok | repeat | parent_at_depth_limit | fixed_invalid_share | fixed_better_share | below_r2_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek_A | R1 | 25 | 1.68 | 0.08 | 0.08 | -0.241 | -0.053 | -0.049 | 0.094 | 2 | 0.24 | 0.08 | 0.0 | 0.0 | 1.0 |
| deepseek_A | R2 | 23 | 1.565 | 0.478 | 0.217 | -0.182 | -0.02 | -0.008 | 0.145 | 8 | 0.217 | 0.174 | 0.0 | 0.375 | 0.75 |
| deepseek_A | R3 | 77 | 3.649 | 0.494 | 0.286 | 0.055 | -0.047 | -0.052 | -0.103 | 30 | 0.156 | 0.429 | 0.053 | 0.133 | 0.633 |
| deepseek_A_think | R1 | 1 | 0.0 | 0.0 | 0.0 | nan | nan | nan | nan | 0 | 0.0 | 0.0 | nan | nan | nan |
| deepseek_A_think | R2 | 2 | 1.0 | 0.5 | 0.5 | -0.388 | -0.031 | -0.004 | -0.402 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| deepseek_A_think | R3 | 63 | 1.317 | 0.048 | 0.032 | 0.022 | -0.029 | -0.029 | 0.02 | 3 | 0.175 | 0.429 | 0.0 | 0.667 | 1.0 |
| deepseek_B | R1 | 22 | 0.591 | 0.0 | 0.0 | nan | nan | nan | nan | 0 | 0.5 | 0.227 | nan | nan | nan |
| deepseek_B | R2 | 71 | 1.507 | 1.014 | 0.761 | -0.177 | -0.052 | -0.039 | 0.159 | 30 | 0.634 | 0.479 | 0.181 | 0.233 | 0.633 |
| deepseek_B | R3 | 48 | 0.604 | 0.021 | 0.021 | -0.044 | -0.076 | -0.072 | 0.294 | 1 | 0.0 | 0.292 | 0.0 | 0.0 | 0.0 |
| deepseek_C | R1 | 27 | 0.852 | 0.407 | 0.407 | 0.137 | -0.075 | -0.08 | -0.164 | 2 | 0.63 | 0.815 | 0.818 | 0.0 | 0.5 |
| deepseek_C | R2 | 60 | 0.983 | 0.767 | 0.617 | -0.239 | -0.043 | -0.029 | 0.001 | 29 | 0.45 | 0.633 | 0.217 | 0.241 | 0.966 |
| deepseek_C | R3 | 60 | 0.417 | 0.117 | 0.1 | 0.138 | -0.026 | -0.03 | -0.303 | 3 | 0.1 | 0.5 | 0.571 | 0.333 | 1.0 |
| deepseek_C_think | R1 | 24 | 0.333 | 0.292 | 0.292 | -0.042 | -0.089 | -0.089 | 0.105 | 2 | 0.583 | 0.958 | 0.714 | 0.0 | 1.0 |
| deepseek_C_think | R2 | 6 | 0.333 | 0.333 | 0.333 | -0.347 | -0.103 | -0.079 | -0.065 | 2 | 0.333 | 0.167 | 0.0 | 0.0 | 1.0 |
| deepseek_C_think | R3 | 43 | 0.256 | 0.116 | 0.093 | 0.116 | -0.035 | -0.042 | -0.32 | 5 | 0.163 | 0.279 | 0.0 | 0.0 | 0.8 |
| qwen_A | R1 | 33 | 0.182 | 0.0 | 0.0 | nan | nan | nan | nan | 0 | 0.303 | 0.0 | nan | nan | nan |
| qwen_A | R2 | 22 | 0.545 | 0.5 | 0.318 | -0.17 | -0.008 | 0.004 | 0.08 | 9 | 0.227 | 0.0 | 0.0 | 0.667 | 0.889 |
| qwen_A | R3 | 64 | 0.953 | 0.0 | 0.0 | nan | nan | nan | nan | 0 | 0.109 | 0.016 | nan | nan | nan |
| qwen_B | R1 | 70 | 0.529 | 0.029 | 0.029 | -0.056 | -0.057 | -0.051 | -0.216 | 2 | 0.514 | 0.586 | 0.0 | 0.0 | 0.0 |
| qwen_B | R2 | 91 | 0.549 | 0.143 | 0.143 | -0.123 | -0.002 | 0.007 | 0.31 | 8 | 0.604 | 0.725 | 0.231 | 0.625 | 0.25 |
| qwen_B | R3 | 33 | 0.424 | 0.0 | 0.0 | nan | nan | nan | nan | 0 | 0.03 | 0.212 | nan | nan | nan |
| qwen_C | R1 | 82 | 0.549 | 0.183 | 0.183 | -0.126 | -0.056 | -0.05 | 0.007 | 9 | 0.585 | 0.488 | 0.4 | 0.111 | 0.667 |
| qwen_C | R2 | 35 | 0.943 | 0.4 | 0.371 | -0.147 | -0.022 | -0.012 | 0.105 | 13 | 0.286 | 0.429 | 0.071 | 0.385 | 0.538 |
| qwen_C | R3 | 55 | 0.164 | 0.055 | 0.055 | -0.205 | -0.068 | -0.068 | -0.287 | 1 | 0.2 | 0.182 | 0.667 | 0.0 | 1.0 |

## 3. Trajectories by round (mean per 5 rounds; depth of valid candidates)

| config | phase | turnover | high_turnover | absicir | depth | invalid | smooth_ops |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek_A | 1-5 | 0.542 | 0.39 | 0.067 | 3.935 | 0.004 | 0.352 |
| deepseek_A | 6-10 | 0.491 | 0.295 | 0.096 | 4.889 | 0.016 | 0.467 |
| deepseek_A | 11-15 | 0.478 | 0.202 | 0.097 | 5.22 | 0.052 | 0.46 |
| deepseek_A | 16-20 | 0.453 | 0.128 | 0.105 | 5.416 | 0.152 | 0.528 |
| deepseek_A_think | 1-5 | 0.415 | 0.16 | 0.07 | 3.774 | 0.0 | 0.76 |
| deepseek_A_think | 6-10 | 0.418 | 0.1 | 0.095 | 4.904 | 0.013 | 1.547 |
| deepseek_A_think | 11-15 | 0.407 | 0.053 | 0.089 | 4.788 | 0.06 | 1.493 |
| deepseek_A_think | 16-20 | 0.378 | 0.087 | 0.094 | 5.018 | 0.02 | 0.94 |
| deepseek_B | 1-5 | 0.552 | 0.362 | 0.058 | 4.806 | 0.112 | 1.052 |
| deepseek_B | 6-10 | 0.558 | 0.354 | 0.062 | 4.955 | 0.18 | 1.038 |
| deepseek_B | 11-15 | 0.493 | 0.231 | 0.059 | 4.933 | 0.192 | 0.935 |
| deepseek_B | 16-20 | 0.56 | 0.375 | 0.058 | 5.036 | 0.144 | 0.972 |
| deepseek_C | 1-5 | 0.512 | 0.295 | 0.059 | 5.066 | 0.232 | 1.124 |
| deepseek_C | 6-10 | 0.479 | 0.221 | 0.064 | 5.35 | 0.384 | 1.293 |
| deepseek_C | 11-15 | 0.529 | 0.321 | 0.057 | 5.452 | 0.324 | 1.228 |
| deepseek_C | 16-20 | 0.463 | 0.205 | 0.059 | 5.35 | 0.32 | 1.259 |
| deepseek_C_think | 1-5 | 0.459 | 0.222 | 0.055 | 4.544 | 0.073 | 0.893 |
| deepseek_C_think | 6-10 | 0.444 | 0.109 | 0.06 | 4.908 | 0.053 | 1.167 |
| deepseek_C_think | 11-15 | 0.436 | 0.11 | 0.06 | 5.119 | 0.087 | 1.267 |
| deepseek_C_think | 16-20 | 0.477 | 0.154 | 0.055 | 5.0 | 0.087 | 1.0 |
| qwen_A | 1-5 | 0.5 | 0.388 | 0.061 | 3.626 | 0.008 | 0.6 |
| qwen_A | 6-10 | 0.399 | 0.204 | 0.061 | 3.578 | 0.0 | 0.708 |
| qwen_A | 11-15 | 0.403 | 0.197 | 0.07 | 3.504 | 0.0 | 0.476 |
| qwen_A | 16-20 | 0.321 | 0.052 | 0.064 | 3.48 | 0.0 | 0.336 |
| qwen_B | 1-5 | 0.654 | 0.685 | 0.044 | 4.554 | 0.18 | 0.363 |
| qwen_B | 6-10 | 0.641 | 0.634 | 0.045 | 4.743 | 0.156 | 0.435 |
| qwen_B | 11-15 | 0.63 | 0.606 | 0.051 | 4.833 | 0.164 | 0.561 |
| qwen_B | 16-20 | 0.63 | 0.603 | 0.054 | 4.998 | 0.256 | 0.56 |
| qwen_C | 1-5 | 0.603 | 0.589 | 0.046 | 4.729 | 0.14 | 0.617 |
| qwen_C | 6-10 | 0.567 | 0.51 | 0.045 | 4.803 | 0.188 | 0.774 |
| qwen_C | 11-15 | 0.525 | 0.366 | 0.057 | 4.973 | 0.252 | 0.805 |
| qwen_C | 16-20 | 0.462 | 0.239 | 0.049 | 5.18 | 0.28 | 0.985 |

## 4. Local refinement or broad exploration: share of next-round candidates derived from the shown formulas

| config | derived_share | invalid | depth_all |
| --- | --- | --- | --- |
| deepseek_A | 0.498 | 0.056 | 4.822 |
| deepseek_A_think | 0.305 | 0.023 | 4.65 |
| deepseek_B | 0.218 | 0.157 | 5.154 |
| deepseek_C | 0.137 | 0.315 | 5.849 |
| deepseek_C_think | 0.068 | 0.075 | 4.947 |
| qwen_A | 0.184 | 0.002 | 3.461 |
| qwen_B | 0.105 | 0.189 | 5.042 |
| qwen_C | 0.114 | 0.215 | 5.249 |

## 5. Rounds with an R1/R2 trigger minus rounds without (next-round candidates)

For Group C this is the effect of the actual diagnosis; for A/B it is the difference under the same trigger condition without any diagnosis text (counterfactual).

| config | invalid | depth_all | smooth_ops | derived_share | rounds_with |
| --- | --- | --- | --- | --- | --- |
| deepseek_A | -0.069 | -0.58 | -0.154 | -0.19 | 25 |
| deepseek_A_think | -0.026 | -0.717 | -0.746 | -0.198 | 3 |
| deepseek_B | -0.171 | -0.628 | 0.109 | 0.031 | 62 |
| deepseek_C | 0.002 | 0.096 | 0.515 | 0.12 | 55 |
| deepseek_C_think | 0.024 | 0.483 | 0.122 | 0.069 | 28 |
| qwen_A | -0.002 | -0.391 | 0.05 | -0.12 | 39 |
| qwen_B | 0.003 | 0.209 | -0.134 | 0.036 | 71 |
| qwen_C | 0.134 | 0.814 | 0.318 | 0.069 | 68 |

## 6. Reasons for invalid formulas (share of all candidates)

| config | depth | other | syntax | unknown name |
| --- | --- | --- | --- | --- |
| deepseek_A | 0.054 | 0.002 | 0.0 | 0.0 |
| deepseek_A_think | 0.023 | 0.0 | 0.0 | 0.0 |
| deepseek_B | 0.154 | 0.0 | 0.0 | 0.003 |
| deepseek_C | 0.307 | 0.007 | 0.0 | 0.001 |
| deepseek_C_think | 0.075 | 0.0 | 0.0 | 0.0 |
| qwen_A | 0.0 | 0.001 | 0.0 | 0.001 |
| qwen_B | 0.126 | 0.001 | 0.0 | 0.062 |
| qwen_C | 0.171 | 0.0 | 0.004 | 0.04 |

## 8. Validation-period retention of the training IC sign, all valid candidates

| config | n | same_sign | train_absicir | valid_absicir |
| --- | --- | --- | --- | --- |
| alpha101 | 38.0 | 0.763 | 0.086 | 0.073 |
| deepseek_A | 102.4 | 0.682 | 0.106 | 0.061 |
| deepseek_A_think | 120.0 | 0.535 | 0.102 | 0.046 |
| deepseek_B | 70.2 | 0.567 | 0.078 | 0.052 |
| deepseek_C | 67.6 | 0.533 | 0.08 | 0.053 |
| deepseek_C_think | 83.667 | 0.523 | 0.081 | 0.05 |
| gp1000 | 282.0 | 0.546 | 0.138 | 0.053 |
| gp200 | 43.8 | 0.719 | 0.099 | 0.071 |
| qwen_A | 77.2 | 0.607 | 0.088 | 0.057 |
| qwen_B | 58.6 | 0.564 | 0.077 | 0.055 |
| qwen_C | 65.6 | 0.588 | 0.076 | 0.058 |

## 9. Group contrasts (five seeds per group; Welch t-test and exact permutation test, unpaired)

| model | contrast | metric | diff | t_welch | p_welch | p_perm |
| --- | --- | --- | --- | --- | --- | --- |
| deepseek | B-A | P1 | -0.055 | -1.171 | 0.305 | 0.333 |
| deepseek | B-A | invalid_share | 0.101 | 1.258 | 0.244 | 0.302 |
| deepseek | B-A | top10_train_turnover | 0.136 | 4.009 | 0.01 | 0.008 |
| deepseek | B-A | top10_valid_turnover | 0.136 | 4.022 | 0.01 | 0.008 |
| deepseek | B-A | valid_icir | -0.055 | -1.863 | 0.133 | 0.024 |
| deepseek | B-A | test1_icir | -0.027 | -0.772 | 0.477 | 0.492 |
| deepseek | B-A | test1_sr_net | 0.131 | 0.634 | 0.548 | 0.532 |
| deepseek | B-A | test1_turnover | 0.167 | 3.078 | 0.031 | 0.016 |
| deepseek | B-A | test2_icir | 0.019 | 0.437 | 0.679 | 0.659 |
| deepseek | C-A | P1 | -0.079 | -1.632 | 0.168 | 0.127 |
| deepseek | C-A | invalid_share | 0.259 | 3.259 | 0.012 | 0.032 |
| deepseek | C-A | top10_train_turnover | 0.112 | 2.915 | 0.023 | 0.008 |
| deepseek | C-A | top10_valid_turnover | 0.111 | 2.868 | 0.023 | 0.008 |
| deepseek | C-A | valid_icir | -0.052 | -1.764 | 0.148 | 0.048 |
| deepseek | C-A | test1_icir | -0.028 | -0.806 | 0.458 | 0.468 |
| deepseek | C-A | test1_sr_net | 0.098 | 0.515 | 0.629 | 0.611 |
| deepseek | C-A | test1_turnover | 0.133 | 2.275 | 0.065 | 0.056 |
| deepseek | C-A | test2_icir | 0.011 | 0.244 | 0.817 | 0.849 |
| deepseek | C-B | P1 | -0.024 | -1.676 | 0.149 | 0.151 |
| deepseek | C-B | invalid_share | 0.158 | 1.883 | 0.096 | 0.103 |
| deepseek | C-B | top10_train_turnover | -0.024 | -0.996 | 0.357 | 0.349 |
| deepseek | C-B | top10_valid_turnover | -0.025 | -0.989 | 0.361 | 0.357 |
| deepseek | C-B | valid_icir | 0.003 | 0.407 | 0.695 | 0.698 |
| deepseek | C-B | test1_icir | -0.001 | -0.09 | 0.93 | 0.937 |
| deepseek | C-B | test1_sr_net | -0.033 | -0.272 | 0.793 | 0.754 |
| deepseek | C-B | test1_turnover | -0.034 | -1.195 | 0.276 | 0.27 |
| deepseek | C-B | test2_icir | -0.009 | -0.361 | 0.727 | 0.73 |
| qwen | B-A | P1 | -0.042 | -1.316 | 0.246 | 0.206 |
| qwen | B-A | invalid_share | 0.187 | 11.029 | 0.0 | 0.008 |
| qwen | B-A | top10_train_turnover | 0.165 | 4.176 | 0.005 | 0.008 |
| qwen | B-A | top10_valid_turnover | 0.17 | 4.217 | 0.005 | 0.008 |
| qwen | B-A | valid_icir | 0.009 | 1.046 | 0.329 | 0.349 |
| qwen | B-A | test1_icir | -0.022 | -1.021 | 0.343 | 0.349 |
| qwen | B-A | test1_sr_net | -0.164 | -0.664 | 0.527 | 0.508 |
| qwen | B-A | test1_turnover | 0.119 | 4.604 | 0.002 | 0.008 |
| qwen | B-A | test2_icir | 0.005 | 0.331 | 0.749 | 0.746 |
| qwen | C-A | P1 | -0.013 | -0.4 | 0.705 | 0.897 |
| qwen | C-A | invalid_share | 0.213 | 5.145 | 0.007 | 0.008 |
| qwen | C-A | top10_train_turnover | 0.09 | 2.015 | 0.08 | 0.095 |
| qwen | C-A | top10_valid_turnover | 0.093 | 2.039 | 0.077 | 0.087 |
| qwen | C-A | valid_icir | 0.009 | 0.762 | 0.469 | 0.444 |
| qwen | C-A | test1_icir | -0.018 | -0.984 | 0.356 | 0.357 |
| qwen | C-A | test1_sr_net | -0.018 | -0.083 | 0.937 | 0.929 |
| qwen | C-A | test1_turnover | 0.055 | 1.399 | 0.21 | 0.183 |
| qwen | C-A | test2_icir | 0.02 | 1.377 | 0.21 | 0.175 |
| qwen | C-B | P1 | 0.029 | 1.805 | 0.109 | 0.103 |
| qwen | C-B | invalid_share | 0.026 | 0.582 | 0.585 | 0.595 |
| qwen | C-B | top10_train_turnover | -0.074 | -2.12 | 0.071 | 0.071 |
| qwen | C-B | top10_valid_turnover | -0.077 | -2.21 | 0.063 | 0.048 |
| qwen | C-B | valid_icir | -0.0 | -0.025 | 0.981 | 0.984 |
| qwen | C-B | test1_icir | 0.004 | 0.174 | 0.866 | 0.873 |
| qwen | C-B | test1_sr_net | 0.146 | 0.861 | 0.421 | 0.421 |
| qwen | C-B | test1_turnover | -0.063 | -1.638 | 0.154 | 0.111 |
| qwen | C-B | test2_icir | 0.015 | 1.047 | 0.328 | 0.373 |
