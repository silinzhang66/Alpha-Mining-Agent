# Supplementary analyses (post hoc, exploratory)

## 1. Share of repair candidates and other candidates that beat the diagnosed formula (R1/R2)

| config | rule | fixed_ok | fixed_better | others | others_better | fixed_better_share | others_better_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek_A | R1 | 2 | 0 | 186 | 40 | 0.0 | 0.215 |
| deepseek_A | R2 | 8 | 3 | 167 | 52 | 0.375 | 0.311 |
| deepseek_A_think | R1 | 0 | 0 | 10 | 5 | nan | 0.5 |
| deepseek_A_think | R2 | 1 | 0 | 15 | 6 | 0.0 | 0.4 |
| deepseek_B | R1 | 0 | 0 | 142 | 19 | nan | 0.134 |
| deepseek_B | R2 | 30 | 7 | 380 | 41 | 0.233 | 0.108 |
| deepseek_C | R1 | 2 | 0 | 142 | 13 | 0.0 | 0.092 |
| deepseek_C | R2 | 29 | 7 | 281 | 46 | 0.241 | 0.164 |
| deepseek_C_think | R1 | 2 | 0 | 159 | 10 | 0.0 | 0.063 |
| deepseek_C_think | R2 | 2 | 0 | 44 | 3 | 0.0 | 0.068 |
| qwen_A | R1 | 0 | 0 | 253 | 31 | nan | 0.123 |
| qwen_A | R2 | 9 | 6 | 186 | 30 | 0.667 | 0.161 |
| qwen_B | R1 | 2 | 0 | 419 | 48 | 0.0 | 0.115 |
| qwen_B | R2 | 8 | 5 | 577 | 54 | 0.625 | 0.094 |
| qwen_C | R1 | 9 | 1 | 506 | 53 | 0.111 | 0.105 |
| qwen_C | R2 | 13 | 5 | 247 | 47 | 0.385 | 0.19 |

## 2. Test period I ICIR: LLM runs without thinking vs. baselines

| contrast | n | diff | p_welch | p_perm |
| --- | --- | --- | --- | --- |
| LLM - GP | 30 vs 10 | -0.0572 | 0.0007 | 0.0003 |
| LLM - Alpha101 | 30 vs 1 | -0.0596 | 0.0 | nan |

## 3. Range of IC t-statistics in test period II

| config | icir | ic_t_min | ic_t_max | sr_gross |
| --- | --- | --- | --- | --- |
| alpha101 | 0.16 | 1.534 | 1.534 | 0.101 |
| deepseek_A | 0.085 | -0.197 | 1.588 | 0.2 |
| deepseek_A_think | 0.103 | 0.636 | 1.114 | -0.184 |
| deepseek_B | 0.104 | 0.559 | 1.371 | -0.202 |
| deepseek_C | 0.095 | 0.28 | 1.162 | -0.082 |
| deepseek_C_think | 0.142 | 0.825 | 1.57 | 0.219 |
| gp1000 | 0.022 | -0.217 | 0.853 | -0.266 |
| gp200 | -0.003 | -1.462 | 0.901 | -0.081 |
| qwen_A | 0.076 | 0.382 | 1.015 | -0.304 |
| qwen_B | 0.081 | 0.411 | 0.983 | -0.247 |
| qwen_C | 0.096 | 0.733 | 1.241 | -0.088 |

## 4. Welch 95% confidence intervals for C - A and C - B

| model | metric | contrast | diff | ci_low | ci_high | p_welch |
| --- | --- | --- | --- | --- | --- | --- |
| deepseek | P1 | C - A | -0.079 | -0.206 | 0.048 | 0.168 |
| deepseek | P1 | C - B | -0.024 | -0.06 | 0.012 | 0.149 |
| deepseek | test1_icir | C - A | -0.028 | -0.118 | 0.062 | 0.458 |
| deepseek | test1_icir | C - B | -0.001 | -0.035 | 0.032 | 0.93 |
| deepseek | test1_sr_net | C - A | 0.098 | -0.392 | 0.589 | 0.629 |
| deepseek | test1_sr_net | C - B | -0.033 | -0.321 | 0.255 | 0.793 |
| qwen | P1 | C - A | -0.013 | -0.095 | 0.069 | 0.705 |
| qwen | P1 | C - B | 0.029 | -0.008 | 0.066 | 0.109 |
| qwen | test1_icir | C - A | -0.018 | -0.062 | 0.025 | 0.356 |
| qwen | test1_icir | C - B | 0.004 | -0.051 | 0.06 | 0.866 |
| qwen | test1_sr_net | C - A | -0.018 | -0.557 | 0.522 | 0.937 |
| qwen | test1_sr_net | C - B | 0.146 | -0.264 | 0.555 | 0.421 |

## 5. P1 over valid, non-duplicate candidates only

| config | P1 | invalid | duplicate | P1_valid | P1_valid_sd |
| --- | --- | --- | --- | --- | --- |
| alpha101 | 0.244 | 0.0 | 0.0 | 0.244 | nan |
| deepseek_A | 0.171 | 0.056 | 0.256 | 0.244 | 0.13 |
| deepseek_A_think | 0.165 | 0.023 | 0.203 | 0.217 | 0.097 |
| deepseek_B | 0.116 | 0.157 | 0.278 | 0.208 | 0.023 |
| deepseek_C | 0.092 | 0.315 | 0.164 | 0.183 | 0.059 |
| deepseek_C_think | 0.097 | 0.075 | 0.17 | 0.128 | 0.01 |
| gp1000 | 0.039 | 0.0 | 0.466 | 0.074 | 0.021 |
| gp200 | 0.086 | 0.0 | 0.428 | 0.149 | 0.116 |
| qwen_A | 0.125 | 0.002 | 0.322 | 0.193 | 0.125 |
| qwen_B | 0.083 | 0.189 | 0.151 | 0.126 | 0.038 |
| qwen_C | 0.112 | 0.215 | 0.097 | 0.163 | 0.037 |

## 6. Break-even one-way cost (bps)

| config | test1 | test2 |
| --- | --- | --- |
| alpha101 | -0.0 | 0.9 |
| deepseek_A | -1.8 | 2.2 |
| deepseek_A_think | -3.5 | -2.1 |
| deepseek_B | -3.9 | -1.9 |
| deepseek_C | -4.1 | -1.0 |
| deepseek_C_think | -4.4 | 2.1 |
| gp1000 | 3.3 | -2.5 |
| gp200 | 6.5 | -1.8 |
| qwen_A | -0.7 | -3.1 |
| qwen_B | -0.3 | -2.4 |
| qwen_C | -0.9 | -0.8 |
