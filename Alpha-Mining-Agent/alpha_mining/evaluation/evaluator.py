"""Expression -> processed factor -> metrics on one period.

Factors are always computed on the full panel (so rolling windows at the start
of a period see their history) and only the metric rows are restricted to the
period. Labels whose forward window crosses the period end are dropped, so no
metric of a period uses prices from after it.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alpha_mining.data.panel import HORIZON, Panel
from alpha_mining.dsl.evaluate import evaluate
from alpha_mining.dsl.ops import ts_mean
from alpha_mining.dsl.parse import Node, canonical, parse
from alpha_mining.evaluation import metrics as M
from alpha_mining.evaluation.preprocess import Neutralizer, preprocess
from alpha_mining.evaluation.regimes import NAMES as REGIME_NAMES


@dataclass
class FactorResult:
    expr: str
    canonical: str
    ic: float
    icir: float
    ic_t: float
    turnover: float
    autocorr: float
    fitness: float
    backtest: dict
    regime_ic: dict = field(default_factory=dict)    # name -> (mean IC, NW t)
    values: np.ndarray | None = field(default=None, repr=False)  # processed (T_period, N)

    def numbers(self) -> dict:
        return {"IC": self.ic, "ICIR": self.icir, "Turnover": self.turnover,
                "Fitness": self.fitness}


class Evaluator:
    def __init__(self, panel: Panel, industry: np.ndarray, regimes: np.ndarray | None,
                 lam: float, tau: float):
        self.panel = panel
        self.lam, self.tau = lam, tau
        self.regimes = regimes
        log_adv = np.log(ts_mean(panel.fields["dollar_vol"], 20))
        self.neutralizer = Neutralizer(industry, log_adv, panel.mask)

    def factor(self, expr: str | Node) -> np.ndarray:
        """Processed (winsorized, neutralized, standardized) factor on the full panel."""
        raw = evaluate(expr, self.panel.features, self.panel.mask)
        return preprocess(raw, self.neutralizer)

    def period_rows(self, start, end) -> np.ndarray:
        d = self.panel.dates
        return np.flatnonzero((d >= start) & (d <= end))

    def period_label(self, rows: np.ndarray) -> np.ndarray:
        y = self.panel.label[rows].copy()
        y[-(HORIZON + 1):] = np.nan  # forward window would leave the period
        return y

    def metrics(self, f: np.ndarray, start, end, keep_values: bool = False,
                expr: str = "") -> FactorResult:
        rows = self.period_rows(start, end)
        fp = f[rows]
        ic = M.spearman_rows(fp, self.period_label(rows))
        stats = M.ic_stats(ic)
        bt = M.quintile_backtest(fp, self.panel.fields["adj_open"][rows], self.panel.dates[rows])
        summ = bt.summary()
        regime_ic = {}
        if self.regimes is not None:
            reg = self.regimes[rows]
            for k, name in enumerate(REGIME_NAMES):
                x = ic[reg == k]
                regime_ic[name] = (float(np.nanmean(x)) if np.isfinite(x).any() else np.nan,
                                   M.newey_west_t(x))
        return FactorResult(
            expr=expr, canonical=canonical(parse(expr)) if expr else "",
            ic=stats["ic"], icir=stats["icir"], ic_t=stats["ic_t"],
            turnover=summ["turnover"], autocorr=M.rank_autocorr(fp),
            fitness=M.fitness(stats["icir"], summ["turnover"], self.lam, self.tau),
            backtest=summ, regime_ic=regime_ic, values=fp if keep_values else None,
        )

    def run(self, expr: str, start, end, keep_values: bool = False) -> FactorResult:
        return self.metrics(self.factor(expr), start, end, keep_values, expr)


def factor_ranks(values: np.ndarray) -> np.ndarray:
    """Cross-sectional ranks cached for rho computations."""
    return M.rank_rows(values)


def to_frame(results: list[FactorResult]) -> pd.DataFrame:
    return pd.DataFrame([{**r.numbers(), "expr": r.expr, "ic_t": r.ic_t,
                          "autocorr": r.autocorr, **r.backtest} for r in results])
