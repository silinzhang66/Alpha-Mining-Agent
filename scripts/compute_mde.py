"""Minimum detectable effects for the group contrasts, Limitation (v).

Post hoc, from the existing five runs per group in results/summary/per_run.csv;
no new runs and no test-set evaluation. Two-sided Welch test at alpha = 0.05,
power 0.8, with the observed variances and Welch degrees of freedom, solved
exactly with the noncentral t distribution (MDE_approx is the usual
(t_{1-alpha/2} + t_{power}) * SE shortcut).

  uv run python scripts/compute_mde.py
"""
import numpy as np
import pandas as pd
from scipy import optimize, stats

from alpha_mining.config import ROOT

OUT = ROOT / "results" / "summary"
METRICS = ["P1", "valid_icir", "test1_icir", "test1_sr_net"]
CONTRASTS = [("C", "A"), ("C", "B"), ("B", "A")]


def mde(a, b, alpha=0.05, power=0.8):
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    nu = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    tc = stats.t.ppf(1 - alpha / 2, nu)
    # the lower tail is negligible and returns NaN for large noncentrality
    pw = lambda d: stats.nct.sf(tc, nu, d / se) + np.nan_to_num(stats.nct.cdf(-tc, nu, d / se))
    d = optimize.brentq(lambda d: pw(d) - power, 1e-9, 10 * se)
    approx = (tc + stats.t.ppf(power, nu)) * se
    return d, approx, nu


def main():
    df = pd.read_csv(OUT / "per_run.csv")
    rows = []
    for model in ["deepseek", "qwen"]:
        for g1, g2 in CONTRASTS:
            a = df[df["config"] == f"{model}_{g1}"]
            b = df[df["config"] == f"{model}_{g2}"]
            assert len(a) == 5 and len(b) == 5
            for col in METRICS:
                d, approx, nu = mde(a[col], b[col])
                rows.append({"model": model, "contrast": f"{g1}-{g2}", "metric": col,
                             "observed": a[col].mean() - b[col].mean(), "MDE": d,
                             "MDE_approx": approx, "welch_df": nu, "ref_mean": b[col].mean()})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "mde.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
