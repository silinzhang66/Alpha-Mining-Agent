"""Sensitivity analysis: IC valid-day rule (docs/robustness_reporting.md, part 2).

A qualified factor must additionally have a defined daily IC on at least 50% of
the trading days of both the training and the validation period. P1, final
pool, Top-10 and composite are rebuilt exactly as in build_final.py with this
one extra filter, and written in the same format to results/final_sensitivity/
so that evaluate_test.py --src final_sensitivity evaluates them the same way.
Only previously qualified factors can change status, so only they are recomputed.

  uv run python scripts/build_sensitivity.py
"""
import json

import numpy as np

from alpha_mining.analysis.final import (TOP_K, Factor, composite, final_pool, ic_day_share,
                                         mean_pairwise_rho, selection_values, summary)
from alpha_mining.config import ROOT, TRAIN, VALID
from alpha_mining.context import config, evaluator

MIN_DAY_SHARE = 0.5
SRC, OUT = ROOT / "results" / "final", ROOT / "results" / "final_sensitivity"

ev = evaluator()
rho_max = config()["pool_rho_max"]
OUT.mkdir(parents=True, exist_ok=True)
for path in sorted(SRC.glob("*.json")):
    res = json.loads(path.read_text())
    qual = [x for x in res["factors"] if x["qualified"]]
    vals = selection_values({"top10": qual}, ev)
    kept = []
    for x, v in zip(qual, vals):
        x["ic_day_share"] = {"train": ic_day_share(v, ev, TRAIN), "valid": ic_day_share(v, ev, VALID)}
        if min(x["ic_day_share"].values()) >= MIN_DAY_SHARE:
            kept.append(Factor(x["key"], x["expr"], x["train"], x["discrete"], x["distinct_ratio"],
                               x["valid"], True, v.astype(np.float32)))
        else:
            x["qualified"] = False
    pool = final_pool(kept, ev, rho_max)
    top = pool[:TOP_K]
    signs = [float(np.sign(f.valid["ic"])) for f in top]
    keep = {f.key for f in kept}
    res.update({"sensitivity": {"min_ic_day_share": MIN_DAY_SHARE, "dropped": len(qual) - len(kept)},
                "n_qualified": len(kept), "P1": len(kept) / res["budget"],
                "final_pool_size": len(pool), "top_k": len(top),
                "final_pool_discrete": sum(f.discrete for f in pool),
                "top10": [{**f.to_json(), "sign": s} for f, s in zip(top, signs)]})
    for label, flag in (("discrete", True), ("continuous", False)):
        res[f"{label}_qualified"] = sum(f.discrete == flag for f in kept)
        res[f"{label}_P1"] = res[f"{label}_qualified"] / res[f"{label}_n"] if res[f"{label}_n"] else None
    res.pop("rho_bar", None), res.pop("composite", None)
    if top:
        comp = composite([f.values for f in top], signs)
        res["rho_bar"] = mean_pairwise_rho([f.values for f in top], ev.period_rows(*VALID))
        res["composite"] = {"train": summary(ev.metrics(comp, *TRAIN)), "valid": summary(ev.metrics(comp, *VALID))}
    (OUT / path.name).write_text(json.dumps(res, indent=1, default=float))
    print(f"{res['run']}: dropped {len(qual) - len(kept)}, P1 {res['P1']:.3f}, top {len(top)}")
