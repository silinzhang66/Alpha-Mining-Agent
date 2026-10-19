"""One-time test-period evaluation of the frozen Top-10 composites.

Requires UNLOCK_TEST=1 and a panel rebuilt to 2026-09-30. Reads only the
selections in results/final/*.json (made on train + validation before the
unlock); nothing here feeds back into any selection.

Periods: Test I, Test II, and the Test II sub-period after the effective
knowledge cutoff C from the probe (results/probe/summary.json), reported as a
supplement because C + 1 .. 2026-09 spans at least 3 months.

  UNLOCK_TEST=1 uv run python scripts/evaluate_test.py
  UNLOCK_TEST=1 uv run python scripts/evaluate_test.py --src final_sensitivity
"""
import argparse
import json

import numpy as np
import pandas as pd

from alpha_mining.analysis.final import evaluate_selection
from alpha_mining.config import ROOT, TEST1, TEST2, assert_allowed, test_unlocked
from alpha_mining.context import evaluator

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="final", choices=["final", "final_sensitivity"])
args = ap.parse_args()
FINAL = ROOT / "results" / args.src
OUT = ROOT / "results" / ("test" if args.src == "final" else "test_sensitivity")

if not test_unlocked():
    raise SystemExit("set UNLOCK_TEST=1 (only after all formal experiments are finished)")
assert_allowed(TEST2[1])
ev = evaluator()
if ev.panel.dates[-1] < TEST2[1] - pd.Timedelta(days=7):
    raise SystemExit(f"panel ends {ev.panel.dates[-1].date()}; rebuild it with load_panel(rebuild=True)")

probe = json.loads((ROOT / "results" / "probe" / "summary.json").read_text())
subs = {}
for model, s in probe.items():
    start = (pd.Period(s["cutoff"], "M") + 1).start_time if s["cutoff"] else TEST2[0]
    if (TEST2[1].to_period("M") - start.to_period("M")).n + 1 >= 3:
        subs[model] = (start, TEST2[1])

OUT.mkdir(parents=True, exist_ok=True)
for path in sorted(FINAL.glob("*.json")):
    sel = json.loads(path.read_text())
    if not sel["top10"]:
        continue
    model = sel["meta"].get("args", {}).get("model")
    periods = {"test1": TEST1, "test2": TEST2}
    # LLM runs use their own model's cutoff; baselines get every model's sub-period.
    for m, per in subs.items():
        if model in (None, m):
            periods[f"test2_after_cutoff_{m}"] = per
    res = evaluate_selection(sel, ev, periods)
    train_icir = sel["composite"]["train"]["icir"]
    for k in ("test1", "test2"):
        res[k]["decay_ratio"] = res[k]["icir"] / train_icir if train_icir else np.nan
    (OUT / path.name).write_text(json.dumps({"run": sel["run"], "periods": {k: [str(a.date()), str(b.date())]
                                             for k, (a, b) in periods.items()}, **res}, indent=1, default=float))
    print(f"{sel['run']}: Test I SR_net {res['test1']['sharpe_net']:.2f}, Test II SR_net {res['test2']['sharpe_net']:.2f}")
