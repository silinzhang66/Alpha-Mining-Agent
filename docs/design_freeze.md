# Design freeze record (2026-10-17)

> Written at the design freeze and committed together with the frozen code. **No part of the design was changed after the freeze.** The test set was unlocked (`UNLOCK_TEST=1`) only after all formal runs had finished, and was evaluated once. All calibration before the freeze used only the training period (2016–2021) and the validation period (2022–2023).

## 1. Frozen parameters (`configs/experiment.json`)

| Item | Value | Changed during the pilot? |
| --- | --- | --- |
| Horizon / return definition | 5 days, open(t+1) → open(t+6) | No (a 1-day horizon was compared and not adopted) |
| Cross-sectional neutralization | MAD winsorization → residual of a regression on FF12 industry + log ADV20 → z-score | No (the pilot confirmed that neutralization raises the training→validation sign agreement: 79% vs. 67%) |
| Fitness | \|ICIR\| − λ·max(0, Turnover − τ), λ = 0.1, τ = 0.5 | No |
| Pool-entry gate | Training-period medians of Alpha101 (82 formulas): \|IC\| ≥ 0.004327, \|ICIR\| ≥ 0.044838 (`results/alpha101_gates.json`) | Recomputed on the formal panel |
| Pool correlation cap | ρ ≤ 0.6, greedy replacement | No |
| R1 regime failure | NW t of the up-regime IC > **1.65** (lag 4), and range or down-pressure IC < 1/3 of the up-regime IC | **Yes**, lowered from 2 (Group C was almost never triggered at the original value) |
| R2 high turnover | Weekly one-way turnover > **0.6**, or lag-1 rank autocorrelation < 0.5; advice text adds "if the depth limit is reached, smooth the inner term" | **Yes**, raised from 0.5 (about half of all candidates triggered it at the original value) |
| R3 homogeneity | ρ > 0.6 and rejected by greedy replacement | No |
| Diagnoses per round | At most 3, assigned in turn across the shown factors | No |
| DSL | Windows {3, 5, 10, 20, 30, 60}, plus lags {1, 2}; depth ≤ 6, nodes ≤ 30; **added element-wise `max(x, y)`, `min(x, y)`** | **Yes** (the theory priors of Groups B/C use max/min themselves, which produced invalid formulas; all three groups and GP share the same DSL) |
| Rounds × candidates per round | 20 × 10 = 200; temperature 0.7; seeds 0–4 | No |
| GP baselines | Budget 200 (population 10), 1000 (population 50) | No |

## 2. Pre-specified metrics

* **Primary metric P1:** share of the 200 candidates that satisfy quality conditions 1–3.
* **Primary metric P2:** net-of-cost Sharpe ratio of the quintile long-short portfolio of the Top-10 composite in the test periods.
* **Auxiliary (diversity) metrics:** size of the final de-duplicated qualified pool; final size of the training-period pool; share of duplicate candidates. The last two were added at the freeze and are reported as auxiliary only.

## 3. Models and API

| Model | API name | Endpoint | Parameters that disable thinking / web search | Model name returned by the server in the pilot |
| --- | --- | --- | --- | --- |
| DeepSeek-V4.1-Flash-1017 | `deepseek-flash` | `https://api.deepseek.com` | `{"thinking": {"type": "disabled"}}` | `deepseek-flash` |
| Qwen3.8-Flash | `qwen3.8-flash` | `https://maas.qianwenaiapi.com/compatible-mode/v1` | `{"enable_thinking": false, "enable_search": false}` | `qwen3.8-flash` |

Neither model has a dated snapshot. The client checks the model name returned by the server during a run and aborts if it changes.

## 4. Data versions

* Constituents: fja05680/sp500, commit `a2430f2` (2026-09-07); `data/raw/sp500_hist.csv`, sha256 prefix `36326709d46d6cd2`.
* Prices: yfinance 1.7.0 with gaps filled from Tiingo, downloaded up to 2023-12-29 (test periods not downloaded). 641 entities usable; daily constituent coverage over training + validation: mean 98.41%, minimum 95.24%.
* Panel `data/cache/panel_2023-12-31.npz` (not distributed; rebuilt by the code): 2014-06-02 to 2023-12-29, 2413 days × 641 entities, sha256 prefix `a29e2c45afa1fd15`.
* Mappings and corrections: `data/ticker_map.csv` (`08571b22c35b1aa9`); industries `data/sic_overrides.csv` (`41e4765bb8287f50`, 32 manual corrections, checked by hand); the adjustment-repair rule applies only to DHR on 2016-07-05.
* Environment: Python 3.11.0, pandas 3.0.6, numpy 2.4.6, deap 1.4, openai 3.33.0; full dependencies in `uv.lock`.

## 5. Checks passed before the freeze

Look-ahead detection; reproduction of a known effect (short-term reversal has a positive training IC); adjustment spot checks (6 split days); constituent spot checks (10 dates); cross-sectional mask; two-level pool isolation (`scripts/check_pool_isolation.py`: with all data after the training period replaced by random numbers, all records of 200 GP candidates and 5 LLM rounds, the training pool and the prompts are identical); coverage statistics. All 58 unit tests passed.

## 6. Pilot findings (recorded as observed; not used to adjust the design)

In 8 pilot runs (2 models × Groups A/C × two rounds, budget 50 each, seed 0; logs in `results/runs/pilot/` and `logs/llm/pilot/`), Group A reached a higher best training fitness than Group C in every case; P1 was the same in both groups; Group C had a larger training pool (13–15 vs. 8–10) and fewer duplicates. About half of the pooled factors kept their sign in the validation period, which points to selection bias in the training period. With one seed and a small budget these are not conclusions; the formal runs follow the design above without adjustment to the pilot results.

## Addendum: changes after the freeze

### 2026-10-18: output cap for the thinking-mode supplementary runs (S1/S2)

- **Problem:** in the first S1 run (`deepseek_A_think_s0`), 52 of 57 calls ended with `finish_reason = length`: the reasoning used up the whole 4096-token output cap and the answer was empty, so in 15 of 20 rounds all 10 candidates were recorded as invalid. This reflects the cap, not the model, and S1/S2 could not be measured this way.
- **Change:** the output cap was raised to 32768 for thinking calls only (`MAX_TOKENS_THINKING` in `alpha_mining/llm/client.py`); one test call confirmed that the API accepts this value (`logs/llm/pilot/pilot_max_tokens_check.jsonl`). Calls without thinking keep 4096, covered by a unit test. The output cap is not among the frozen parameters in Section 1; it is an implementation parameter.
- **Not affected:** the 30 main runs had finished before the change, and all 600 of their calls ended with `finish_reason = stop`, with no truncation.
- **Handling:** the void run was moved to `results/runs/aborted/` and `logs/llm/aborted/` and kept; all six S1/S2 runs were re-run from scratch with the new cap.

### 2026-10-18: test-set unlock (one-time evaluation)

- All formal runs and Top-10 selections (`results/final/`, `results/final_sensitivity/`) were committed before the unlock; no selection, parameter or code was changed afterwards.
- **Data:** `UNLOCK_TEST=1 scripts/download_prices.py --refresh` downloaded prices up to 2026-09-30 (yfinance 1.7.0). The pre-unlock price cache was kept as a backup.
- **A step added during execution:** the trading calendar is taken from the SPY cache, but SPY is not a constituent, so `download_all` does not refresh it. The first rebuilt panel therefore still ended on 2023-12-29; SPY was refreshed separately (`fetch('SPY', refresh=True)`) and the panel rebuilt. This was an execution omission, not a design change; after the refresh, all trading days up to 2023 are identical to the original.
- **Panel:** `data/cache/panel_2026-09-30.npz`, 2014-06-02 to 2026-09-30, 3102 days × 685 entities, sha256 prefix `fe70dc4b920beeed`. For the 634 entities shared with the frozen panel, the masks up to 2023 are identical and daily returns differ by at most 2.4e-6 (rounding of adjustment factors).
- **Validation rules applied unchanged:** after the extension to 2026, Tiingo returned data of a re-used ticker for 7 delisted entities (ANSS, CAM, CHK, FL, HBI, PCL, SIVBQ); the frozen validation rules flagged them as suspect / missing and they were left out of the new panel. Only **ANSS** was still a constituent in the test periods (until 2025-07-17), so it is missing from test period I. This is a false positive caused by a 204-day gap from the ticker re-use; to avoid changing code after the unlock it was not fixed, and the paper states it as a limitation. The other 6 had left the index before 2023-03.
- **Coverage (`data/daily_coverage.csv`):** test period I mean 99.20%, minimum 98.81%; test period II mean 99.54%, minimum 99.20%. Recomputed for 2016–2023: mean 97.83%, minimum 94.44%, below the 98.41% / 95.24% at the freeze because of the 7 entities above; the frozen panel and the training / validation results are not affected.
- **Constituents:** the snapshot ends on 2026-09-07 (the frozen version); constituents from 2026-09-08 to 09-30 are carried forward from 09-07.
- **Evaluation:** `evaluate_test.py` (once for the main table and once with `--src final_sensitivity`), `summarize.py`, `make_figures.py`; results recorded as obtained.
