# Why Rule-Based Diagnostic Feedback Fails in LLM Alpha Mining

Code, prompts, complete LLM logs and saved results for the paper *Why Rule-Based Diagnostic Feedback Fails in LLM Alpha Mining: A Controlled Evaluation with a Single-Use Test Set*.

The study runs an LLM factor-mining loop on S&P 500 stocks with three prompt groups:

- **A**: blind symbolic search with performance feedback only, without financial vocabulary;
- **B**: A plus price-volume theory priors and a short hypothesis per formula;
- **C**: B plus rule-based diagnostic feedback (rules R1–R3, translated into repair instructions).

Two LLMs (DeepSeek-V4.1-Flash, Qwen3.8-Flash) are compared with GP baselines at equal and five-fold evaluation budgets and with Alpha101. All methods share one operator library (DSL), one fitness function and one evaluator. The design was frozen before the main runs ([docs/design_freeze.md](docs/design_freeze.md)), and each of the two test periods was evaluated exactly once.

| Period | Dates | Use |
| --- | --- | --- |
| Training | 2016-01 – 2021-12 | mining, fitness, training pool |
| Validation | 2022-01 – 2023-12 | qualification and Top-10 selection |
| Test I | 2024-01 – 2025-12 | evaluated once |
| Test II | 2026-01 – 2026-09 | evaluated once |

## Repository layout

```
alpha_mining/            library code
  data/                  price download, S&P 500 membership, SEC SIC industries, panel
  dsl/                   expression language: grammar, parser, AST sandbox, operators
  evaluation/            preprocessing, IC/ICIR, turnover, backtest, market regimes, robustness
  mining/                training pool, diagnostic rules R1-R3, mining session
  llm/                   API client (with request logging), prompts for groups A/B/C, mining loop
  baselines/             GP (DEAP) and Alpha101
  analysis/              final pool, Top-10 and composite
  probe.py               knowledge-cutoff probe
  config.py              paths, period splits, test-set lock
configs/experiment.json  frozen experiment parameters
scripts/                 entry points (see "Pipeline")
tests/                   unit tests
data/                    derived tables and the probe question bank (see "Data")
results/                 all saved results (see "Results")
logs/llm/                every LLM request and response
docs/                    design freeze record, pre-specified robustness reporting, probe results
```

## Setup

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                         # installs the exact versions in uv.lock
cp .env.example .env            # then fill in the keys you need
uv run pytest                   # unit tests
```

`.env` holds `DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY` (Qwen), `TIINGO_API_KEY` (price gap filling) and `SEC_USER_AGENT` (SEC EDGAR requires a contact string, e.g. `"Name email@example.com"`). The analysis scripts that only read saved results need no keys. The figure scripts need matplotlib, which is not a project dependency: run them with `uv run --with matplotlib==3.10.0 ...`.

## Data

**Not included:** raw prices. yfinance and Tiingo data may not be redistributed. `scripts/download_prices.py` downloads them into `data/cache/` (a Tiingo key is needed for the gap-filled tickers), and the panel `data/cache/panel_<end>.npz` is built from that cache on first use. Prices of the same tickers downloaded at a later date can differ slightly from ours (e.g. revised adjustment factors).

**Included:**

| File | Content | Source |
| --- | --- | --- |
| `data/raw/sp500_hist.csv`, `sp500_ticker_start_end.csv` | daily S&P 500 constituents | [fja05680/sp500](https://github.com/fja05680/sp500), commit `a2430f2` (2026-09-07); see that repository for its terms |
| `data/alpha101_formulas.csv` | the 101 formulas with an eligibility flag (82 computable from our data) | extracted by `scripts/extract_alpha101.py` from Kakushadze (2016), *101 Formulaic Alphas*, arXiv:1601.00991 |
| `data/sic_codes.csv`, `sic_overrides.csv`, `tiingo_sic_review.csv` | SIC codes per entity and 32 manual corrections | SEC EDGAR |
| `data/ticker_map.csv`, `tiingo_map.csv` | index ticker → price entity; tickers served by Tiingo | built by us |
| `data/coverage_report*.csv`, `daily_coverage.csv` | download status and daily constituent coverage | built by us |
| `data/probe/` | the 230 probe questions (`questions.jsonl`), their sources and evidence (`questions_review.csv`) and raw questions by month (`raw/`) | built by us |

`scripts/extract_alpha101.py` expects the arXiv text at `data/raw/alpha101/alpha101.txt` (not included; only needed to re-extract the formulas).

## Test-set lock

Every loader refuses data from 2024-01-01 onwards unless `UNLOCK_TEST=1` is set (`alpha_mining/config.py`). Mining, selection and all checks ran with the lock on. The one-time test evaluation was done with `scripts/evaluate_test.py` after all selections had been committed; its outputs are in `results/test/` and `results/test_sensitivity/`. The paper's out-of-sample claim rests on that single evaluation; running the script again with new selections would not be a held-out test.

## Pipeline

Steps in the order they were run. All `uv run python` commands are run from the repository root.

1. **Data.** `scripts/download_prices.py` (prices); `scripts/build_tiingo_map.py` (proposes Tiingo entities for tickers yfinance cannot serve); `scripts/check_data.py` (adjustment, membership and coverage checks → `results/checks/`).
2. **Alpha101 and gates.** `scripts/extract_alpha101.py`; `scripts/run_alpha101.py` (writes the entry gates `results/alpha101_gates.json`).
3. **Pre-freeze checks.** `scripts/check_pool_isolation.py` (nothing after the training period reaches the miner); `scripts/analyze_pilot.py` (pilot runs in `results/runs/pilot/`).
4. **Knowledge-cutoff probe.** `scripts/run_probe.py build | run --model {deepseek,qwen} | analyze` → `results/probe/`, see [docs/probe_results.md](docs/probe_results.md).
5. **Mining.** `scripts/run_formal.sh {deepseek,qwen,gp}` runs every configuration through `scripts/run_mining.py`, e.g.
   ```bash
   uv run python scripts/run_mining.py llm --model deepseek --group C --seed 0
   uv run python scripts/run_mining.py gp --budget 200 --seed 0
   ```
   Each run writes `results/runs/<run>.jsonl` (every candidate with its training metrics), `<run>.meta.json`, and for LLM runs `logs/llm/<run>.jsonl`.
6. **Selection (train + validation only).** `scripts/build_final.py` → `results/final/`; `scripts/build_sensitivity.py` → `results/final_sensitivity/` ([docs/robustness_reporting.md](docs/robustness_reporting.md)).
7. **Test evaluation (done once).** `UNLOCK_TEST=1 uv run python scripts/evaluate_test.py [--src final_sensitivity]` → `results/test/`, `results/test_sensitivity/`.
8. **Tables, analyses and figures** (read saved results only):
   ```bash
   uv run python scripts/summarize.py                                   # results/summary/
   uv run python scripts/make_cost_table.py                             # results/summary/cost*.csv
   uv run python scripts/analyze_mechanism.py                           # results/mechanism/mechanism.md
   uv run python scripts/make_case_study.py                             # Table 2
   PYTHONPATH=scripts uv run python scripts/analyze_supplement.py       # results/mechanism/supplement.md
   uv run --with matplotlib==3.10.0 python scripts/make_figure1.py
   uv run --with matplotlib==3.10.0 python scripts/make_figures.py
   ```
   `make_case_study.py` recomputes validation metrics and needs the panel up to 2023.

Runs are unseeded at the API level, so re-running the LLM steps gives new runs, not our logs; the logs in `logs/llm/` are the record of the runs reported in the paper.

## Results

| Path | Content |
| --- | --- |
| `results/runs/` | every candidate of every formal run, with training metrics and pool decisions; `pilot/`: pre-freeze pilot runs; `aborted/`: the thinking run voided by the output cap (see the addendum in `docs/design_freeze.md`) |
| `logs/llm/` | every LLM request and response of the formal runs and of the probe, with token usage; `pilot/`, `aborted/` as above |
| `results/final/`, `results/final_sensitivity/` | final qualified pool, frozen Top-10 and composite per run (train + validation) |
| `results/test/`, `results/test_sensitivity/` | the one-time test evaluation |
| `results/summary/` | per-run and per-config tables (`tables.md`; `table2_main.csv` is Table 1 of the paper; full robustness and cost tables) |
| `results/mechanism/` | failure-mechanism analysis of Section 6 (exploratory) and post hoc supplementary numbers |
| `results/probe/` | probe answers and per-month accuracy |
| `results/figures/` | Figures 1–3; `fig2_convergence_correlations.*` and `fig3_test_periods.*` contain the panels referred to as "in the repository" (correlation of Group C factors with known factors; IC by market regime) |
| `results/checks/` | data-check outputs |

Run names: `<model>_<group>[_think]_s<seed>` for LLM runs, `gp<budget>_s<seed>` for GP.

## License

The code is released under the MIT License ([LICENSE](LICENSE)). Third-party data keep the terms of their sources listed under "Data".
