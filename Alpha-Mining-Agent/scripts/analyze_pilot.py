"""Summarize pilot runs for calibration (before the design freeze).

Per run: candidate status counts, quality-gate hit rate, pool size, best
training fitness by round, and how often diagnostic rules R1-R3 would fire on
all evaluated candidates versus how often they were actually shown (Group C).

  uv run python scripts/analyze_pilot.py deepseek_A_s0_pilot deepseek_C_s0_pilot
"""
import json
import sys

import numpy as np
import pandas as pd

from alpha_mining.config import ROOT
from alpha_mining.context import config, thresholds

# Pilot runs are archived in pilot/ subfolders, apart from the formal runs.
RUNS, LOGS = ROOT / "results" / "runs" / "pilot", ROOT / "logs" / "llm" / "pilot"
gates = json.loads((ROOT / config()["pool_gate_file"]).read_text())
th = thresholds()


def rules(r: dict, pool: dict) -> set:
    fired = set()
    sign = 1.0 if r["ic"] >= 0 else -1.0
    reg = {k: (sign * v[0], sign * v[1]) for k, v in r["regime_ic"].items()}
    up_ic, up_t = reg["up"]
    if np.isfinite(up_ic) and up_ic > 0 and up_t > th.r1_up_t and any(
            np.isfinite(reg[k][0]) and reg[k][0] < th.r1_ratio * up_ic for k in ("range", "down_pressure")):
        fired.add("R1")
    if r["turnover"] > th.r2_turnover or r["autocorr"] < th.r2_autocorr:
        fired.add("R2")
    if pool and pool["action"] == "rejected_corr":
        fired.add("R3")
    return fired


def shown_rules(run: str) -> pd.DataFrame:
    """Diagnoses actually put in each round's prompt, from the LLM request log."""
    rows = []
    path = LOGS / f"{run}.jsonl"
    if not path.exists():
        return pd.DataFrame(columns=["round", "R1", "R2", "R3"])
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("attempt", 0):
            continue
        text = rec["request"]["messages"][-1]["content"]
        rows.append({"round": rec["round"], **{k: text.count(f"[{k} ") for k in ("R1", "R2", "R3")}})
    return pd.DataFrame(rows).set_index("round")


for run in sys.argv[1:]:
    recs = [json.loads(l) for l in (RUNS / f"{run}.jsonl").read_text().splitlines()]
    ok = [r for r in recs if r["status"] == "ok"]
    print(f"\n######## {run}: {len(recs)} candidates")
    print("status:", pd.Series([r["status"] for r in recs]).value_counts().to_dict())
    for r in recs:
        if r["status"] in ("invalid", "error"):
            print(f"  [{r['status']}] r{r['round']} {r['expr'][:70]!r}: {r['message'][:90]}")
    df = pd.DataFrame([{"round": r["round"], "expr": r["expr"], **{k: r["result"][k] for k in
                        ("ic", "icir", "ic_t", "turnover", "autocorr", "fitness")},
                        "pool": r["pool"]["action"], "max_corr": r["pool"]["max_corr"],
                        "rules": ",".join(sorted(rules(r["result"], r["pool"])))} for r in ok])
    df["quality"] = (df.ic.abs() >= gates["train_abs_ic_median"]) & (df.icir.abs() >= gates["train_abs_icir_median"])
    print(f"quality hit rate {df.quality.mean():.0%} ({df.quality.sum()}/{len(recs)} of all candidates: "
          f"{df.quality.sum() / len(recs):.0%}); pool actions {df.pool.value_counts().to_dict()}")
    by = df.groupby("round").agg(n_ok=("fitness", "size"), best_fit=("fitness", "max"),
                                 med_fit=("fitness", "median"), med_abs_ic=("ic", lambda s: s.abs().median()),
                                 med_turnover=("turnover", "median"), quality=("quality", "sum"))
    by["best_so_far"] = by.best_fit.cummax()
    for k in ("R1", "R2", "R3"):
        by[f"{k}_would_fire"] = df.groupby("round").rules.apply(lambda s: s.str.contains(k).sum())
    by = by.join(shown_rules(run).add_suffix("_shown"))
    print(by.round(4).to_string())
    print("would-fire share over all evaluated:", {k: f"{df.rules.str.contains(k).mean():.0%}" for k in ("R1", "R2", "R3")})
    print("top 5 by fitness:")
    print(df.sort_values("fitness", ascending=False).head(5)[["round", "fitness", "ic", "icir", "turnover", "pool", "expr"]]
          .round(4).to_string(index=False, max_colwidth=80))
