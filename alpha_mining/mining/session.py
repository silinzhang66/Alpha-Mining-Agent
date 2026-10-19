"""Budgeted mining session shared by GP and the LLM groups.

Every proposed formula costs one unit of budget, including formulas that fail
validation or raise at evaluation, and exact duplicates (after AST
canonicalization), which are not re-evaluated. All metrics are training-period.
"""
import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from alpha_mining.config import TRAIN
from alpha_mining.dsl.parse import DSLError, canonical, parse
from alpha_mining.evaluation.evaluator import Evaluator, FactorResult
from alpha_mining.mining.pool import PoolDecision, TrainingPool


@dataclass
class Candidate:
    round: int
    expr: str
    status: str                      # ok | invalid | duplicate | error
    message: str = ""
    canonical: str = ""
    result: FactorResult | None = None
    pool: PoolDecision | None = None
    meta: dict = field(default_factory=dict)  # e.g. the LLM's hypothesis

    @property
    def fitness(self) -> float:
        return self.result.fitness if self.result else -np.inf

    def feedback_numbers(self) -> dict:
        """The raw numbers every group sees: Formula, Fitness, IC, ICIR, Turnover, Max_Pool_Corr."""
        r = self.result
        return {"Formula": self.expr, "Fitness": r.fitness, "IC": r.ic, "ICIR": r.icir,
                "Turnover": r.turnover, "Max_Pool_Corr": self.pool.max_corr}

    def to_json(self) -> dict:
        d = {"round": self.round, "expr": self.expr, "status": self.status,
             "message": self.message, "canonical": self.canonical, "meta": self.meta}
        if self.result:
            r = asdict(self.result)
            r.pop("values")
            d["result"] = r
        if self.pool:
            d["pool"] = asdict(self.pool)
        return d


class MiningSession:
    def __init__(self, evaluator: Evaluator, pool: TrainingPool, budget: int):
        self.evaluator, self.pool, self.budget = evaluator, pool, budget
        self.candidates: list[Candidate] = []
        self.seen: dict[str, Candidate] = {}

    @property
    def used(self) -> int:
        return len(self.candidates)

    @property
    def remaining(self) -> int:
        return self.budget - self.used

    def submit(self, expr: str, round_: int, meta: dict | None = None) -> Candidate:
        if self.remaining <= 0:
            raise RuntimeError("evaluation budget exhausted")
        c = Candidate(round_, expr, "ok", meta=meta or {})
        try:
            node = parse(expr)
            c.canonical = canonical(node)
        except DSLError as e:
            c.status, c.message = "invalid", str(e)
            self.candidates.append(c)
            return c
        if c.canonical in self.seen:
            c.status, c.message = "duplicate", "this formula has already been evaluated"
            self.candidates.append(c)
            return c
        try:
            t0 = time.perf_counter()
            res = self.evaluator.run(expr, *TRAIN, keep_values=True)
            c.meta["eval_seconds"] = round(time.perf_counter() - t0, 3)
        except Exception as e:  # numerical failure inside an operator
            c.status, c.message = "error", f"evaluation failed: {type(e).__name__}: {e}"
            self.candidates.append(c)
            return c
        if not np.isfinite(res.icir):
            c.status, c.message, c.result = "error", "factor has no valid IC (all NaN or constant)", None
            self.candidates.append(c)
            self.seen[c.canonical] = c
            return c
        c.result = res
        c.pool = self.pool.offer(res)
        res.values = None  # the pool keeps its own ranks; free the panel
        self.candidates.append(c)
        self.seen[c.canonical] = c
        return c

    def evaluated(self) -> list[Candidate]:
        return [c for c in self.candidates if c.result is not None]

    def save(self, path) -> None:
        with open(path, "w") as f:
            for c in self.candidates:
                f.write(json.dumps(c.to_json(), default=float) + "\n")
