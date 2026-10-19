"""Training-period factor pool with greedy replacement.

Only training-period values and metrics ever enter this pool. rho(f, g) is the
absolute time-mean of daily cross-sectional Spearman correlations, computed
from cached ranks.
"""
from dataclasses import dataclass

import numpy as np

from alpha_mining.evaluation.evaluator import FactorResult, factor_ranks
from alpha_mining.evaluation.metrics import factor_rho

RHO_MAX = 0.6


@dataclass
class PoolMember:
    canonical: str
    expr: str
    fitness: float
    ranks: np.ndarray  # float32 cross-sectional ranks, training rows


@dataclass
class PoolDecision:
    action: str               # added | replaced | rejected_quality | rejected_corr
    max_corr: float           # max rho against the pool before the decision
    corr_with: list[str]      # canonicals of members with rho > RHO_MAX
    removed: list[str]


class TrainingPool:
    def __init__(self, min_abs_ic: float, min_abs_icir: float, rho_max: float = RHO_MAX):
        self.min_abs_ic, self.min_abs_icir, self.rho_max = min_abs_ic, min_abs_icir, rho_max
        self.members: list[PoolMember] = []

    def correlations(self, ranks: np.ndarray) -> np.ndarray:
        return np.array([factor_rho(ranks, m.ranks) for m in self.members])

    def max_corr(self, ranks: np.ndarray) -> float:
        rho = self.correlations(ranks)
        return float(np.nanmax(rho)) if len(rho) and np.isfinite(rho).any() else 0.0

    def offer(self, res: FactorResult) -> PoolDecision:
        """Greedy replacement: join if uncorrelated, or if fitter than every
        correlated member (which are then removed); otherwise reject (rule R3)."""
        ranks = factor_ranks(res.values).astype(np.float32)
        rho = self.correlations(ranks)
        max_corr = float(np.nanmax(rho)) if len(rho) and np.isfinite(rho).any() else 0.0
        close = [i for i, r in enumerate(rho) if r > self.rho_max]
        corr_with = [self.members[i].canonical for i in close]
        if not (abs(res.ic) >= self.min_abs_ic and abs(res.icir) >= self.min_abs_icir):
            return PoolDecision("rejected_quality", max_corr, corr_with, [])
        member = PoolMember(res.canonical, res.expr, res.fitness, ranks)
        if not close:
            self.members.append(member)
            return PoolDecision("added", max_corr, [], [])
        if all(res.fitness > self.members[i].fitness for i in close):
            removed = [self.members[i].canonical for i in close]
            self.members = [m for i, m in enumerate(self.members) if i not in set(close)]
            self.members.append(member)
            return PoolDecision("replaced", max_corr, corr_with, removed)
        return PoolDecision("rejected_corr", max_corr, corr_with, [])
