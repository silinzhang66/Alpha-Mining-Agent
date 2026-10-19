"""Post-hoc robustness reporting (not part of the frozen design).

Discrete factors -- comparisons, ternaries, sign, short-window ts_rank -- leave
large blocks of tied raw values. After neutralization each block moves through
the ranking as a unit, so their IC is fragile to small perturbations. P1 and
the final factor pool are additionally reported separately for discrete and
continuous factors, using the rule below fixed before the formal experiments.
"""
import numpy as np

from alpha_mining.evaluation.metrics import MIN_STOCKS

DISCRETE_RATIO = 0.5


def distinct_ratio(raw: np.ndarray, rows: np.ndarray) -> float:
    """Median over `rows` of (#distinct finite raw values / #finite raw values)."""
    out = []
    for x in raw[rows]:
        x = x[np.isfinite(x)]
        if len(x) >= MIN_STOCKS:
            out.append(len(np.unique(x)) / len(x))
    return float(np.median(out)) if out else np.nan


def is_discrete(raw: np.ndarray, rows: np.ndarray) -> bool:
    """Raw (pre-preprocessing) factor on the training rows; NaN ratio counts as discrete."""
    r = distinct_ratio(raw, rows)
    return not (r >= DISCRETE_RATIO)
