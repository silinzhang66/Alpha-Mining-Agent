"""Diagnostic Translator (Group C): turn training-period metric shortfalls into
operator-level repair instructions.

It only re-expresses numbers every group already sees (plus the regime split of
the same training IC series); it never adds validation or test information.
Step 0 aligns signs so every factor is diagnosed as a positive factor:
f~ = sign(IC_train) * f, which flips regime ICs and leaves turnover,
autocorrelation and |rho| unchanged.
"""
from dataclasses import dataclass

import numpy as np

from alpha_mining.mining.session import Candidate


@dataclass(frozen=True)
class Thresholds:
    """Initial values before the pilot; the frozen values are in configs/experiment.json."""
    r1_up_t: float = 2.0            # up-state IC must be significant (Newey-West t)
    r1_ratio: float = 1 / 3         # weak state IC below this fraction of the up-state IC
    r2_turnover: float = 0.5        # weekly one-way turnover of the quintile legs
    r2_autocorr: float = 0.5        # lag-1 rank autocorrelation
    max_per_round: int = 3


@dataclass
class Diagnosis:
    formula: str
    rule: str
    text: str
    evidence: str = ""   # the numbers alone, for the stats-only control (review round 1)


def _fmt(x: float) -> str:
    return f"{x:+.4f}"


# Review round 1, experiment E1: R1 advice that does not add a layer at the depth limit.
R1_ADVICE = "Add a volatility penalty such as dividing by ts_std(returns, d), or an amplitude filter."
R1_ADVICE_DEPTH = ("Add a volatility penalty such as dividing by ts_std(returns, d), or an amplitude "
                   "filter, without adding a layer: if the expression is already at the depth limit, "
                   "divide an inner term or replace an inner operator instead of wrapping the "
                   "whole expression.")


def diagnose_one(c: Candidate, th: Thresholds, r1_advice: str = R1_ADVICE) -> list[Diagnosis]:
    r = c.result
    out = []
    sign = 1.0 if r.ic >= 0 else -1.0
    reg = {k: (sign * ic, sign * t) for k, (ic, t) in r.regime_ic.items()}
    up_ic, up_t = reg.get("up", (np.nan, np.nan))
    if np.isfinite(up_ic) and up_ic > 0 and up_t > th.r1_up_t:
        weak = [k for k in ("range", "down_pressure")
                if np.isfinite(reg[k][0]) and reg[k][0] < th.r1_ratio * up_ic]
        if weak:
            names = {"range": "range-bound", "down_pressure": "down-pressure"}
            evidence = ", ".join([f"up-market IC={_fmt(up_ic)} (t={up_t:.1f})"] +
                                 [f"{names[k]} IC={_fmt(reg[k][0])}" for k in weak])
            out.append(Diagnosis(c.expr, "R1", (
                f"[R1 regime failure] Evidence: {evidence} (sign-aligned). The signal is too noisy "
                f"in {' and '.join(names[k] for k in weak)} markets. {r1_advice}"),
                f"{evidence} (sign-aligned)"))
    if r.turnover > th.r2_turnover or r.autocorr < th.r2_autocorr:
        out.append(Diagnosis(c.expr, "R2", (
            f"[R2 high turnover] Evidence: weekly one-way turnover={r.turnover:.2f}, "
            f"lag-1 rank autocorrelation={r.autocorr:.2f}. The signal jumps too much and trading "
            f"costs erode it. Wrap the outer expression in ts_decay_linear(x, d) or ts_mean(x, d) "
            f"to smooth it over time; if it is already at the depth limit, smooth an inner "
            f"term instead of adding a layer."),
            f"weekly one-way turnover={r.turnover:.2f}, lag-1 rank autocorrelation={r.autocorr:.2f}"))
    if c.pool is not None and c.pool.action == "rejected_corr":
        out.append(Diagnosis(c.expr, "R3", (
            f"[R3 redundancy] Evidence: correlation with the factor pool={c.pool.max_corr:.2f} "
            f"(> 0.6) and its fitness does not beat the correlated pool factor(s): "
            f"{'; '.join(c.pool.corr_with[:2])}. Replace the core operator or bring in a "
            f"volume dimension to make it orthogonal."),
            f"correlation with the factor pool={c.pool.max_corr:.2f}; its fitness does not beat the "
            f"correlated pool factor(s): {'; '.join(c.pool.corr_with[:2])}"))
    return out


def diagnose(shown: list[Candidate], th: Thresholds = Thresholds(),
             r1_advice: str = R1_ADVICE) -> list[Diagnosis]:
    """Diagnoses for the factors shown this round, at most `max_per_round`,
    taken round-robin across factors so each shown factor gets a turn."""
    per_factor = [diagnose_one(c, th, r1_advice) for c in shown if c.result is not None]
    out: list[Diagnosis] = []
    for k in range(max((len(d) for d in per_factor), default=0)):
        for d in per_factor:
            if k < len(d) and len(out) < th.max_per_round:
                out.append(d[k])
    return out
