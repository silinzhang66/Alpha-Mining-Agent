"""Final pool, Top-10 and composite for each finished run (train + validation only).

Writes results/final/<run>.json, including the frozen Top-10 selection that
scripts/evaluate_test.py later evaluates on the test periods. Pilot runs are
skipped. Existing outputs are kept unless --force.

  uv run python scripts/build_final.py              # all runs + Alpha101
  uv run python scripts/build_final.py gp200_s0 deepseek_C_s0
  uv run python scripts/build_final.py --review 1   # review-round runs, no Alpha101

Fitness fields always use the frozen lambda; P1 and the selection do not use fitness.
"""
import argparse
import json
import time

import numpy as np

from alpha_mining.analysis.final import (TOP_K, Factor, assess, composite, final_pool,
                                         mean_pairwise_rho, summary)
from alpha_mining.baselines.alpha101 import Alpha101, eligible_formulas
from alpha_mining.config import ROOT, TRAIN, VALID
from alpha_mining.context import config, evaluator
from alpha_mining.dsl.evaluate import evaluate


p = argparse.ArgumentParser()
p.add_argument("runs", nargs="*")
p.add_argument("--force", action="store_true")
p.add_argument("--review", type=int, default=None, help="review round: results/review<N>/")
args = p.parse_args()
base = ROOT / "results" / (f"review{args.review}" if args.review else "")
RUNS, OUT = base / "runs", base / "final"

cfg = config()
gates = json.loads((ROOT / cfg["pool_gate_file"]).read_text())
ev = evaluator()
panel = ev.panel
OUT.mkdir(parents=True, exist_ok=True)

names = args.runs or ([] if args.review else ["alpha101"]) + sorted(p.stem for p in RUNS.glob("*.jsonl") if "_pilot" not in p.stem)


def candidates(name: str):
    """(records for status counts, [(key, expr, raw)] for evaluated formulas, meta)."""
    if name == "alpha101":
        eng = Alpha101(panel.features, panel.mask)
        rows = list(eligible_formulas().itertuples())
        recs = [{"status": "ok"} for _ in rows]
        return recs, ((f"alpha{a.alpha:03d}", a.formula, eng.compute(a.formula)) for a in rows), \
            {"method": "alpha101", "pool_size": None}
    recs = [json.loads(l) for l in (RUNS / f"{name}.jsonl").read_text().splitlines()]
    meta = json.loads((RUNS / f"{name}.meta.json").read_text())
    ok = [r for r in recs if r["status"] == "ok"]
    return recs, ((r["canonical"], r["expr"], evaluate(r["expr"], panel.features, panel.mask)) for r in ok), meta


for name in names:
    out = OUT / f"{name}.json"
    if out.exists() and not args.force:
        print(f"{name}: exists, skipped")
        continue
    t0 = time.time()
    recs, cands, meta = candidates(name)
    factors: list[Factor] = [assess(k, e, raw, ev, gates) for k, e, raw in cands]
    budget = len(recs)
    pool = final_pool(factors, ev, cfg["pool_rho_max"])
    top = pool[:TOP_K]
    signs = [float(np.sign(f.valid["ic"])) for f in top]
    res = {"run": name, "meta": {k: v for k, v in meta.items() if k != "config"}, "budget": budget,
           "status_counts": {s: sum(r["status"] == s for r in recs) for s in ("ok", "invalid", "duplicate", "error")},
           "n_qualified": sum(f.is_qualified for f in factors),
           "training_pool_size": meta.get("pool_size"),
           "final_pool_size": len(pool), "top_k": len(top)}
    res["P1"] = res["n_qualified"] / budget
    res["duplicate_share"] = res["status_counts"]["duplicate"] / budget
    # Pre-registered robustness reporting (docs/robustness_reporting.md): P1 by discrete / continuous.
    for label, flag in (("discrete", True), ("continuous", False)):
        grp = [f for f in factors if f.discrete == flag]
        res[f"{label}_n"] = len(grp)
        res[f"{label}_qualified"] = sum(f.is_qualified for f in grp)
        res[f"{label}_P1"] = res[f"{label}_qualified"] / len(grp) if grp else None
    res["final_pool_discrete"] = sum(f.discrete for f in pool)
    res["top10"] = [{**f.to_json(), "sign": s} for f, s in zip(top, signs)]
    if top:
        comp = composite([f.values for f in top], signs)
        res["rho_bar"] = mean_pairwise_rho([f.values for f in top], ev.period_rows(*VALID))
        res["composite"] = {"train": summary(ev.metrics(comp, *TRAIN)), "valid": summary(ev.metrics(comp, *VALID))}
    res["factors"] = [f.to_json() for f in factors]
    out.write_text(json.dumps(res, indent=1, default=float))
    print(f"{name}: P1 {res['P1']:.3f} ({res['n_qualified']}/{budget}), final pool {len(pool)}, "
          f"top {len(top)}, {time.time() - t0:.0f}s")
