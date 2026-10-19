"""Supplementary numbers for the paper (post hoc, exploratory, not pre-registered).

Reads only mining logs and already-saved results; no evaluation is re-run and
the test periods are read from results/test/*.json.

  beat_parent  for every R1/R2 trigger, the share of advice-following repairs and
               of all other valid next-round candidates whose training fitness
               exceeds the diagnosed parent's (baseline for "repairs beat parent")
  llm_vs_base  test-I ICIR of the 30 non-thinking LLM runs against the 10 GP runs
               (Welch t, Monte Carlo permutation) and against Alpha101 (one-sample t)
  test2_t      range of the test-II IC t statistic per configuration
  contrast_ci  Welch 95% confidence intervals for the C - A and C - B contrasts
  p1_valid     P1 with invalid and duplicate candidates removed from the denominator
  breakeven    one-way cost (bps) at which the mean weekly net return of the
               composite is zero: mean gross / mean traded weight (= 4 x turnover)

  uv run python scripts/analyze_supplement.py
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from alpha_mining.context import thresholds
from alpha_mining.mining.diagnostics import diagnose
from analyze_mechanism import (OUT, RES, RUN_RE, applies, config_of, derived, llm_runs,
                               load_candidates, markdown, select_shown, tree)


def beat_parent() -> pd.DataFrame:
    th = thresholds()
    rows = []
    for run in llm_runs():
        cands = load_candidates(run)
        for rnd in range(1, max(c.round for c in cands) + 1):
            nxt = [c for c in cands if c.round == rnd]
            shown = select_shown(cands, rnd)
            by_expr = {c.expr: c for c in shown}
            for d in diagnose(shown, th):
                if d.rule == "R3":
                    continue
                parent = by_expr[d.formula]
                pt = tree(parent.expr)
                fixed = [c for c in nxt if derived(pt, c.tree) and applies(d.rule, pt, c.tree)]
                ok_fixed = [c for c in fixed if c.result is not None]
                others = [c for c in nxt if c.result is not None and all(c is not f for f in fixed)]
                rows.append({"config": config_of(run), "rule": d.rule,
                             "fixed_ok": len(ok_fixed),
                             "fixed_better": sum(c.fitness > parent.fitness for c in ok_fixed),
                             "others": len(others),
                             "others_better": sum(c.fitness > parent.fitness for c in others)})
    t = pd.DataFrame(rows).groupby(["config", "rule"]).sum()
    t["fixed_better_share"] = t.fixed_better / t.fixed_ok.replace(0, np.nan)
    t["others_better_share"] = t.others_better / t.others.replace(0, np.nan)
    return t.round(3).reset_index()


def llm_vs_base(per_run: pd.DataFrame, n_perm: int = 200_000) -> pd.DataFrame:
    is_llm = per_run.config.str.match(r"^(deepseek|qwen)_[ABC]$")
    llm = per_run.loc[is_llm, "test1_icir"].to_numpy(float)
    gp = per_run.loc[per_run.config.isin(["gp200", "gp1000"]), "test1_icir"].to_numpy(float)
    a101 = float(per_run.loc[per_run.config == "alpha101", "test1_icir"].iloc[0])
    rng = np.random.default_rng(0)
    pooled = np.concatenate([llm, gp])
    obs = llm.mean() - gp.mean()
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(pooled)
        null[i] = p[:len(llm)].mean() - p[len(llm):].mean()
    return pd.DataFrame([
        {"contrast": "LLM - GP", "n": f"{len(llm)} vs {len(gp)}", "diff": obs,
         "p_welch": stats.ttest_ind(llm, gp, equal_var=False).pvalue,
         "p_perm": np.mean(np.abs(null) >= abs(obs) - 1e-12)},
        {"contrast": "LLM - Alpha101", "n": f"{len(llm)} vs 1", "diff": llm.mean() - a101,
         "p_welch": stats.ttest_1samp(llm, a101).pvalue, "p_perm": np.nan},
    ]).round(4)


def test2_t() -> pd.DataFrame:
    rows = []
    for p in sorted((RES / "test").glob("*.json")):
        d = json.load(open(p))
        rows.append({"config": config_of(d["run"]), "icir": d["test2"]["icir"],
                     "ic_t": d["test2"]["ic_t"], "sr_gross": d["test2"]["sharpe_gross"]})
    t = pd.DataFrame(rows)
    return (t.groupby("config").agg(icir=("icir", "mean"), ic_t_min=("ic_t", "min"),
                                    ic_t_max=("ic_t", "max"), sr_gross=("sr_gross", "mean"))
            .round(3).reset_index())


def contrast_ci(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ["deepseek", "qwen"]:
        for metric in ["P1", "test1_icir", "test1_sr_net"]:
            for a, b in [("C", "A"), ("C", "B")]:
                x = per_run.loc[per_run.config == f"{model}_{a}", metric].to_numpy(float)
                y = per_run.loc[per_run.config == f"{model}_{b}", metric].to_numpy(float)
                r = stats.ttest_ind(x, y, equal_var=False)
                ci = r.confidence_interval()
                rows.append({"model": model, "metric": metric, "contrast": f"{a} - {b}",
                             "diff": x.mean() - y.mean(), "ci_low": ci.low, "ci_high": ci.high,
                             "p_welch": r.pvalue})
    return pd.DataFrame(rows).round(3)


def p1_valid(per_run: pd.DataFrame) -> pd.DataFrame:
    t = per_run.copy()
    t["valid_unique"] = 1 - t.duplicate_share.fillna(0) - t.invalid_share.fillna(0)
    t["P1_valid"] = t.P1 / t.valid_unique
    return (t.groupby("config").agg(P1=("P1", "mean"), invalid=("invalid_share", "mean"),
                                    duplicate=("duplicate_share", "mean"),
                                    P1_valid=("P1_valid", "mean"), P1_valid_sd=("P1_valid", "std"))
            .round(3).reset_index())


def breakeven() -> pd.DataFrame:
    rows = []
    for p in sorted((RES / "test").glob("*.json")):
        d = json.load(open(p))
        for period in ["test1", "test2"]:
            m = d[period]
            gross = np.nanmean(np.asarray(m["weekly"]["gross"], float))
            rows.append({"config": config_of(d["run"]), "period": period,
                         "bps": gross / (4 * m["turnover"]) * 1e4})
    t = pd.DataFrame(rows)
    return (t.pivot_table(index="config", columns="period", values="bps", aggfunc="mean")
            .round(1).reset_index())


def main() -> None:
    per_run = pd.read_csv(RES / "summary" / "per_run.csv")
    bp, lb, t2 = beat_parent(), llm_vs_base(per_run), test2_t()
    ci, pv, be = contrast_ci(per_run), p1_valid(per_run), breakeven()
    with open(OUT / "supplement.md", "w") as f:
        f.write("# Supplementary analyses (post hoc, exploratory)\n\n")
        f.write("## 1. Share of repair candidates and other candidates that beat the diagnosed formula (R1/R2)\n\n")
        f.write(markdown(bp, index=False) + "\n\n")
        f.write("## 2. Test period I ICIR: LLM runs without thinking vs. baselines\n\n")
        f.write(markdown(lb, index=False) + "\n\n")
        f.write("## 3. Range of IC t-statistics in test period II\n\n")
        f.write(markdown(t2, index=False) + "\n\n")
        f.write("## 4. Welch 95% confidence intervals for C - A and C - B\n\n")
        f.write(markdown(ci, index=False) + "\n\n")
        f.write("## 5. P1 over valid, non-duplicate candidates only\n\n")
        f.write(markdown(pv, index=False) + "\n\n")
        f.write("## 6. Break-even one-way cost (bps)\n\n")
        f.write(markdown(be, index=False) + "\n")
    print(open(OUT / "supplement.md").read())


if __name__ == "__main__":
    main()
