# Case study (deepseek_C_s1, exploratory)

## R1: `ts_decay_linear(ts_mean((high - max(open, close)) / (high - low), 20), 5)`

- First proposed in round 2; depth 6; diagnosed in 18 rounds (rounds 3–20), always in the shown list.
- Training: |ICIR| 0.1527, turnover 0.319, fitness 0.1527; regime IC (sign-aligned) down +0.0016 (t=0.3), range +0.0204 (t=3.2), up +0.0161 (t=2.8).
- Validation: |ICIR| 0.0636, IC sign flipped, turnover 0.329; regime IC (aligned to the training sign) down -0.0063 (t=-0.7), range -0.0093 (t=-0.5), up -0.0025 (t=-0.3).
- Diagnosis text: [R1 regime failure] Evidence: up-market IC=+0.0161 (t=2.8), down-pressure IC=+0.0016 (sign-aligned). The signal is too noisy in down-pressure markets. Add a volatility penalty such as dividing by ts_std(returns, d), or an amplitude filter.

18 derived next-round candidates, 9 modified as advised; 14 invalid (14 too deep), 0 duplicates; 0 valid candidates with higher fitness than the parent.

| round | formula | follows | status | depth | train_absicir | train_turnover | fitness | valid_absicir | valid_turnover | valid_same_sign |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | ts_decay_linear(ts_mean((high - max(open, close)) / (high - low), 20) / (ts_std(returns, 20) + 0.001), 5) | True | invalid | 7 | nan | nan | nan | nan | nan | nan |
| 4 | ts_decay_linear((high - max(open, close)) / (high - low) / (ts_std(returns, 20) + 0.001), 5) | True | ok | 6 | 0.003 | 0.5934 | -0.0064 | 0.0392 | 0.5997 | True |
| 5 | ts_mean((high - max(open, close)) / (high - low + 0.001), 10) | False | ok | 5 | 0.0835 | 0.532 | 0.0803 | 0.0054 | 0.5401 | False |
| 6 | ts_decay_linear(ts_mean((high - max(open, close)) / (high - low + 0.001), 10), 10) | False | ok | 6 | 0.1042 | 0.4198 | 0.1042 | 0.0606 | 0.4266 | False |
| 6 | ts_decay_linear(ts_mean((min(open, close) - low) / (high - low + 0.001) - (high - max(open, close)) / (high - low + 0.001), 10), 5) | False | invalid | 7 | nan | nan | nan | nan | nan | nan |
| 8 | ts_decay_linear(ts_mean((high - max(open, close)) / (high - low + 0.001) * ts_rank(volume, 20), 10), 5) | False | invalid | 7 | nan | nan | nan | nan | nan | nan |
| 9 | ts_decay_linear(ts_mean((high - max(open, close)) / (high - low) / (ts_std(returns, 20) + 0.001), 10), 5) | True | invalid | 7 | nan | nan | nan | nan | nan | nan |
| 10 | ts_decay_linear(ts_mean((high - max(open, close)) / (high - low) / (ts_std(returns, 20) + 0.001), 10), 5) | True | invalid | 7 | nan | nan | nan | nan | nan | nan |
| 11 | ts_decay_linear(ts_mean((high - max(open, close)) / (high - low + 0.001), 10) / (ts_std(returns, 20) + 0.01), 5) | True | invalid | 7 | nan | nan | nan | nan | nan | nan |
| 12 | ts_decay_linear(ts_mean((high - max(open, close)) / (high - low), 30), 10) | False | ok | 6 | 0.1111 | 0.2134 | 0.1111 | 0.1543 | 0.2228 | False |
| 13 | ts_decay_linear(ts_mean((high - max(open, close)) / (high - low), 20), 5) / (0.01 + ts_std(returns, 20)) | True | invalid | 7 | nan | nan | nan | nan | nan | nan |
| 14 | ts_decay_linear(ts_mean((close - low) / (high - low) - (high - max(open, close)) / (high - low), 10), 5) | False | invalid | 7 | nan | nan | nan | nan | nan | nan |
| 15 | ts_decay_linear(ts_mean((high - max(open, close)) / (high - low) / ts_std(returns, 20), 10), 5) | True | invalid | 7 | nan | nan | nan | nan | nan | nan |
| 16 | ts_decay_linear(ts_mean((high - max(open, close)) / (high - low) * (ts_rank(volume, 20) - 0.5), 20), 5) | False | invalid | 7 | nan | nan | nan | nan | nan | nan |
| 16 | ts_decay_linear(ts_mean((min(open, close) - low) / (high - low) - (high - max(open, close)) / (high - low), 10), 5) | False | invalid | 7 | nan | nan | nan | nan | nan | nan |
| 17 | ts_decay_linear(ts_mean((high - max(open, close)) / (high - low), 10) / (1 + ts_std(returns, 20)), 5) | True | invalid | 7 | nan | nan | nan | nan | nan | nan |
| 18 | ts_decay_linear(ts_mean((high - max(open, close)) / (high - low), 20) / (1 + ts_std(returns, 20)), 5) | True | invalid | 7 | nan | nan | nan | nan | nan | nan |
| 19 | ts_decay_linear(ts_mean((high - max(open, close)) / (high - low) - (min(open, close) - low) / (high - low), 20), 10) | False | invalid | 7 | nan | nan | nan | nan | nan | nan |

## R2: `ts_decay_linear(ts_mean(ts_rank(volume, 10) - ts_rank(abs(returns), 10), 10), 5)`

- First proposed in round 10; depth 6; diagnosed in 10 rounds (rounds 11–20), always in the shown list.
- Training: |ICIR| 0.1387, turnover 0.613, fitness 0.1273; regime IC (sign-aligned) down +0.0059 (t=1.2), range +0.0113 (t=2.8), up +0.0071 (t=1.8).
- Validation: |ICIR| 0.0044, IC sign kept, turnover 0.611; regime IC (aligned to the training sign) down +0.0045 (t=0.8), range +0.0105 (t=1.0), up -0.0099 (t=-1.2).
- Diagnosis text: [R2 high turnover] Evidence: weekly one-way turnover=0.61, lag-1 rank autocorrelation=0.96. The signal jumps too much and trading costs erode it. Wrap the outer expression in ts_decay_linear(x, d) or ts_mean(x, d) to smooth it over time; if it is already at the depth limit, smooth an inner term instead of adding a layer.

11 derived next-round candidates, 7 modified as advised; 1 invalid (1 too deep), 4 duplicates; 0 valid candidates with higher fitness than the parent.

| round | formula | follows | status | depth | train_absicir | train_turnover | fitness | valid_absicir | valid_turnover | valid_same_sign |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | ts_decay_linear(ts_mean(ts_rank(volume, 10) - ts_rank(abs(returns), 10), 20), 10) | True | ok | 6 | 0.0502 | 0.3856 | 0.0502 | 0.0264 | 0.389 | False |
| 12 | ts_decay_linear(ts_mean(ts_rank(volume, 10) - ts_rank(abs(returns), 10), 20), 10) | True | duplicate | 6 | nan | nan | nan | nan | nan | nan |
| 13 | ts_decay_linear(ts_mean(ts_rank(volume, 10) - ts_rank(abs(returns), 10), 10), 10) | True | ok | 6 | 0.1177 | 0.5537 | 0.1124 | 0.0195 | 0.5556 | False |
| 14 | ts_decay_linear(ts_mean(ts_rank(volume, 20) - ts_rank(abs(returns), 20), 20), 10) | True | ok | 6 | 0.0383 | 0.3443 | 0.0383 | 0.0966 | 0.3455 | False |
| 15 | ts_decay_linear(ts_mean(ts_rank(volume, 20) - ts_rank(abs(returns), 20), 20), 10) | True | duplicate | 6 | nan | nan | nan | nan | nan | nan |
| 17 | ts_decay_linear(ts_rank(volume, 20) - ts_rank(abs(returns), 20), 10) | False | ok | 5 | 0.1056 | 0.6075 | 0.0948 | 0.0954 | 0.6032 | True |
| 18 | ts_mean(ts_rank(volume, 20) - ts_rank(abs(returns), 20), 10) | False | ok | 5 | 0.1063 | 0.5408 | 0.1022 | 0.0289 | 0.5369 | True |
| 18 | ts_decay_linear((ts_rank(volume, 20) - ts_rank(abs(returns), 20)) / (1 + ts_std(returns, 20)), 10) | False | ok | 6 | 0.1045 | 0.6074 | 0.0938 | 0.0948 | 0.6033 | True |
| 19 | ts_decay_linear(ts_mean(ts_rank(volume, 20) - ts_rank(abs(returns), 20), 10) / (1 + ts_std(returns, 20)), 5) | False | invalid | 7 | nan | nan | nan | nan | nan | nan |
| 20 | ts_decay_linear(ts_mean(ts_rank(volume, 10) - ts_rank(abs(returns), 10), 10), 10) | True | duplicate | 6 | nan | nan | nan | nan | nan | nan |
| 20 | ts_decay_linear(ts_mean(ts_rank(volume, 20) - ts_rank(abs(returns), 20), 20), 10) | True | duplicate | 6 | nan | nan | nan | nan | nan | nan |
