# Pre-specified post-review experiments, review round 1 (2026-10-21)

> Written and committed before any of these runs, in response to reviewer comments 1–3: depth-compatible repair advice, a stronger turnover penalty, and statistics without instructions. Sections 1–4 were not changed after the runs; Section 5 was added afterwards. The paper reports these experiments in Section 7 ("Post-review experiments").

## 1. Scope and constraints

- Training (2016–2021) and validation (2022–2023) data only. The data loader stays locked (`UNLOCK_TEST` is not set), `scripts/evaluate_test.py` is not run, and the test periods are not evaluated again.
- The frozen design of the main study ([design_freeze.md](design_freeze.md)) is unchanged. These experiments are a post-review supplement, labelled "post-review, development data only" in the paper, and do not replace any main result.
- Model: DeepSeek-V4.1-Flash (API name `deepseek-flash`), with thinking and search off and temperature 0.7. R = 20, K = 10 and the budget is 200. The DSL, gates, training pool, Top-10 selection and R1–R3 thresholds are those of the main study (`configs/experiment.json`).
- Only DeepSeek is used. The depth conflict is strongest in DeepSeek C (81% of R1 diagnoses concern depth-6 formulas, and 82% of R1 repairs are too deep), and the reviewer's comment notes that one model is enough.
- The API model has no dated snapshot and may differ from the one used in the main runs (2026-10-18). Groups B and C are therefore rerun in the same batch as controls, and all contrasts are computed within this batch. Differences between the main-study DeepSeek B/C and the reruns are reported only as a drift check.

## 2. Groups (five runs each; runs are not seeded and not paired)

| Group | Run name | Content | Used in |
| --- | --- | --- | --- |
| B | `deepseek_B_s*` | Same as main-study B | E3 control, drift check |
| C | `deepseek_C_s*` | Same as main-study C; replaying the main study's 260 Group C prompts with this code reproduces every diagnosis block exactly | E1 and E3 control |
| Cd | `deepseek_Cd_s*` | C, but the R1 advice does not add depth: "…without adding a layer: if the expression is already at the depth limit, divide an inner term or replace an inner operator instead of wrapping the whole expression." R2, R3 and all trigger conditions are unchanged | E1 |
| Cs | `deepseek_Cs_s*` | B's prompt plus the statistics that C's diagnoses quote under the same triggers: regime ICs with t, turnover with autocorrelation, pool correlation. **No** rule names, interpretation or repair instructions. Header: "ADDITIONAL STATISTICS (training sample)" | E3 |
| B-λ | `deepseek_B_lam1_s*` | B with λ = 1.0 (τ = 0.5 unchanged). The prompt's fitness formula, the evaluator and the training pool all use λ = 1.0 | E2 |
| C-λ | `deepseek_C_lam1_s*` | C with λ = 1.0. The R2 threshold stays at 0.6 | E2 |

Why λ = 1.0: cutting turnover from 0.61 to 0.50 then gains 0.11 in fitness, the same order as a typical |ICIR| (0.10–0.15). Only one parameter is changed.

## 3. Metrics and contrasts

All metrics use training and validation data only. A formula is "too deep" if it was rejected with `expression depth … exceeds the limit`. "Following R1/R2" uses the heuristic of `scripts/analyze_mechanism.py`: the candidate reuses a subtree covering at least half of the parent formula's nodes and adds the advised operator.

- **E1 (depth conflict): Cd − C.** Primary metric: share of too-deep formulas among all candidates. Secondary: share of invalid formulas; R1 repair success (share of R1-following candidates that are valid and beat the parent's fitness); share of R1-following candidates that are too deep; P1; validation ICIR of the composite.
- **E2 (objective alignment): C-λ − B-λ.** Metrics: validation turnover of the composite, P1, validation ICIR of the composite. Also R2 repair success, defined as above. Reported alongside C − B at λ = 0.1 from the same batch (descriptive; no interaction test).
- **E3 (information vs. instructions): C − Cs and Cs − B.** Metrics: P1, validation ICIR of the composite, share of invalid formulas, validation turnover of the composite.

P1 is defined as in the main study: qualified candidates divided by 200, with invalid and duplicate candidates counted in the denominator.

## 4. Statistics and reporting

- For each contrast: difference in means, Welch 95% confidence interval, Welch p and exact permutation p (252 splits; smallest attainable p = 0.008). No correction for multiple comparisons.
- Five runs per group give limited power (the main study's MDE for P1 is about 0.05–0.17). Results are reported whatever their direction, and a non-significant difference is not read as "no effect".
- No runs are added, and no threshold or λ is changed. A run is repeated under the same name only if it is interrupted by an API or technical failure, and any such repeat is recorded below.
- The paper reports the experiments in a short paragraph labelled as a post-review supplement on development data only; full results are in this repository.

## 5. Execution record (added after the runs)

- 2026-10-21: all 30 runs completed in one batch, six in parallel at 2–3.5 minutes each. There were no failures and no repeats. The served model name was `deepseek-flash` in every response. `scripts/build_final.py --review 1` produced the validation results, and `scripts/analyze_review1.py` wrote `results/review1/summary/`. The replayed diagnoses match the prompts actually sent in every round.
- Pre-specified results: `results/review1/summary/tests.csv`; per-run values: `per_run.csv`. Apart from one secondary measure, no contrast is significant (p > 0.1): C − B R2 repair success at λ = 0.1 has permutation p = 0.048 (Welch p = 0.065, uncorrected).
- Post hoc description (not pre-specified):
  - At λ = 1.0 the formulas shown have lower training turnover (0.37–0.39, against 0.45–0.50 at λ = 0.1).
  - R2 fires only 0.2 times per run (12–19 at λ = 0.1) and R1 0.8–1.2 times, so C-λ's feedback is almost always silent.
  - Mean candidate turnover (0.51–0.53) and composite validation turnover (0.69–0.73) are close to their values at λ = 0.1, because the composite selects factors by validation |ICIR|.
- Drift and replication check:
  - Rerun C has 15.3% too-deep formulas (main study 30.7%), and rerun B 20.6% (main study 15.4%).
  - The main-study DeepSeek C − B difference in invalid share was +0.158 (p ≈ 0.10); in the rerun it is −0.049. The group-level depth spillover is not reproduced.
  - The share of R1-following repairs that are too deep is reproduced (rerun C 67%, main study 82%).
