# Knowledge-cutoff probe: results (run 2026-10-18)

Setup: 23 calls per model, 10 questions per call (months mixed within a batch), temperature 0, thinking and web search off. The question bank is `data/probe/questions.jsonl`, with sources and evidence in `data/probe/questions_review.csv`; full requests and responses are in `logs/llm/probe_*.jsonl`. The server returned the model names `deepseek-flash` and `qwen3.8-flash`, matching the freeze record. All 230 answers were parsed.

## 1. Decision by the pre-specified rule

| Model | Release date | Effective cutoff C | Controls (2023) | 2025 accuracy |
| --- | --- | --- | --- | --- |
| DeepSeek-V4.1-Flash-1017 | 2026-10-17 | **2026-05** | 19/20 | 108/120 (90%) |
| Qwen3.8-Flash | 2026-08-26 | **2026-05** | 15/20 | 74/120 (62%) |

Monthly accuracy and p-values are in `results/probe/summary.json` (`uv run python scripts/run_probe.py analyze`).

**Pre-specified interpretation:** C falls in 2026, so test period II (2026-01 to 2026-09) overlaps the effective knowledge of both models for five months (2026-01 to 2026-05), which the paper reports. Four months (2026-06 to 2026-09) remain after C, at least the three required, so the ICIR and net Sharpe ratio of the 2026-06 to 2026-09 sub-period are **reported in addition**, without replacing the main table.

## 2. Descriptive observations (not part of the pre-specified rule)

1. **DeepSeek stays above chance after C.** June–September 2026 combined: 16/40 (40%), one-sided binomial p = 0.026; by month 4, 4, 5 and 3 correct, none significant on its own. Possible reasons: partial knowledge of these months, or questions that can be guessed without knowing the event. So it is not certain that the months after C are fully unseen, and the sub-period results should be read with care.
2. **Qwen after C: 11/40 (27.5%), p = 0.42**, indistinguishable from chance. Its monthly results vary a lot: 2025-10 has only 3/10, and 4 months from 2025-12 to 2026-04 are not significant. C = 2026-05 rests on a single month (6/10, p = 0.0197) at the edge of significance, so this conclusion is fragile. The 15/20 on the controls shows the question format itself works.
3. That both models give the same C is a coincidence; Qwen's much lower overall accuracy looks more like knowing less than like a different cutoff.
