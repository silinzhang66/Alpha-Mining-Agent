"""Second-level (final) factor pool, Top-10 selection and the composite factor
(paper Section 4).

Runs after mining ends and is the first place validation data is read. The
selection (final pool, Top-10, composite signs) uses training and validation
only, and is written to disk before the test set is unlocked; test-period
evaluation then only reads that frozen selection.

  qualified (conditions 1-3): |IC| and |ICIR| >= the Alpha101 training medians
      on BOTH train and validation, and the same IC sign on both.
  P1: qualified candidates / all candidates (budget), so invalid, duplicate and
      failed formulas count in the denominator.
  final pool: qualified factors by validation |ICIR| descending, skipping any
      with rho > 0.6 (validation period) against a factor already in the pool.
  Top-10: the first 10 of the final pool (all of them if fewer).
  composite: each Top-10 factor signed by its validation IC, equal-weight mean
      of the processed (z-scored) values; mean over the factors available for a
      stock that day, NaN if none.
"""
import warnings
from dataclasses import dataclass, field

import numpy as np

from alpha_mining.config import TRAIN, VALID
from alpha_mining.evaluation.evaluator import Evaluator, factor_ranks
from alpha_mining.evaluation.metrics import factor_rho
from alpha_mining.evaluation.preprocess import preprocess
from alpha_mining.evaluation.robustness import distinct_ratio, DISCRETE_RATIO

TOP_K = 10


def passes_gates(m: dict, gates: dict) -> bool:
    return (abs(m["ic"]) >= gates["train_abs_ic_median"]
            and abs(m["icir"]) >= gates["train_abs_icir_median"])


def qualified(train: dict, valid: dict | None, gates: dict) -> bool:
    if valid is None or not (passes_gates(train, gates) and passes_gates(valid, gates)):
        return False
    return bool(np.sign(train["ic"]) == np.sign(valid["ic"]) != 0)


def summary(r) -> dict:
    return {"ic": r.ic, "icir": r.icir, "ic_t": r.ic_t, "turnover": r.turnover,
            "fitness": r.fitness, **{k: v for k, v in r.backtest.items() if k != "turnover"}}


@dataclass
class Factor:
    key: str                     # canonical formula (or Alpha101 id)
    expr: str
    train: dict
    discrete: bool
    distinct_ratio: float
    valid: dict | None = None
    is_qualified: bool = False
    values: np.ndarray | None = field(default=None, repr=False)  # processed, full panel

    def to_json(self) -> dict:
        return {"key": self.key, "expr": self.expr, "train": self.train, "valid": self.valid,
                "qualified": self.is_qualified, "discrete": self.discrete,
                "distinct_ratio": self.distinct_ratio}


def assess(key: str, expr: str, raw: np.ndarray, ev: Evaluator, gates: dict) -> Factor:
    """Train metrics for every candidate; validation only if the training gates pass."""
    train_rows = ev.period_rows(*TRAIN)
    ratio = distinct_ratio(raw, train_rows)
    f = preprocess(raw, ev.neutralizer)
    tr = summary(ev.metrics(f, *TRAIN))
    fac = Factor(key, expr, tr, bool(not (ratio >= DISCRETE_RATIO)), float(ratio))
    if np.isfinite(tr["icir"]) and passes_gates(tr, gates):
        fac.valid = summary(ev.metrics(f, *VALID))
        fac.is_qualified = qualified(tr, fac.valid, gates)
        if fac.is_qualified:
            fac.values = f.astype(np.float32)
    return fac


def final_pool(factors: list[Factor], ev: Evaluator, rho_max: float) -> list[Factor]:
    rows = ev.period_rows(*VALID)
    pool, ranks = [], []
    for fac in sorted((f for f in factors if f.is_qualified), key=lambda f: -abs(f.valid["icir"])):
        r = factor_ranks(fac.values[rows])
        if all(not (factor_rho(r, q) > rho_max) for q in ranks):
            pool.append(fac)
            ranks.append(r)
    return pool


def mean_pairwise_rho(values: list[np.ndarray], rows: np.ndarray) -> float:
    ranks = [factor_ranks(v[rows]) for v in values]
    rho = [factor_rho(ranks[i], ranks[j]) for i in range(len(ranks)) for j in range(i + 1, len(ranks))]
    return float(np.nanmean(rho)) if rho else np.nan


def composite(values: list[np.ndarray], signs: list[float]) -> np.ndarray:
    if not values:
        raise ValueError("empty Top-10")
    stack = np.stack([s * v for s, v in zip(signs, values)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells stay NaN
        return np.nanmean(stack, axis=0)


def selection_values(sel: dict, ev: Evaluator) -> list[np.ndarray]:
    """Recompute the frozen Top-10 on the evaluator's panel (DSL formulas or Alpha101)."""
    from alpha_mining.baselines.alpha101 import Alpha101
    from alpha_mining.dsl.evaluate import evaluate
    p = ev.panel
    eng = None
    out = []
    for f in sel["top10"]:
        if f["key"].startswith("alpha"):
            eng = eng or Alpha101(p.features, p.mask)
            raw = eng.compute(f["expr"])
        else:
            raw = evaluate(f["expr"], p.features, p.mask)
        out.append(preprocess(raw, ev.neutralizer))
    return out


def evaluate_selection(sel: dict, ev: Evaluator, periods: dict) -> dict:
    """Composite (frozen signs) on each named period: metrics, regime IC, weekly net returns."""
    from alpha_mining.evaluation import metrics as M
    comp = composite(selection_values(sel, ev), [f["sign"] for f in sel["top10"]])
    out = {}
    for name, (start, end) in periods.items():
        r = ev.metrics(comp, start, end)
        rows = ev.period_rows(start, end)
        bt = M.quintile_backtest(comp[rows], ev.panel.fields["adj_open"][rows], ev.panel.dates[rows])
        out[name] = {**summary(r), "regime_ic": r.regime_ic,
                     "weekly": {"dates": [str(d.date()) for d in bt.dates], "net": bt.net.tolist(),
                                "gross": bt.gross.tolist()}}
    return out


def ic_day_share(values: np.ndarray, ev: Evaluator, period) -> float:
    """Share of the period's trading days on which the daily IC is defined."""
    from alpha_mining.evaluation import metrics as M
    rows = ev.period_rows(*period)
    ic = M.spearman_rows(values[rows], ev.period_label(rows))
    return float(np.isfinite(ic).sum() / len(rows))
