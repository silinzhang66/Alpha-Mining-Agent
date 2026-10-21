"""Review round 1 supplementary experiments (docs/review1_prespecification.md).

Training and validation data only: reads results/review1/runs, results/review1/final
(scripts/build_final.py --review 1) and logs/llm/review1. Diagnoses are replayed
with the frozen thresholds and, for the diagnosed groups, checked against the
prompts actually sent.

  E1  Cd - C          depth conflict
  E2  C_lam1 - B_lam1 alignment with a stronger turnover penalty (C - B at lambda 0.1 alongside)
  E3  C - Cs, Cs - B  verbal instructions vs. the statistics alone

  uv run python scripts/analyze_review1.py
"""
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_mechanism as am  # noqa: E402

from alpha_mining.config import ROOT  # noqa: E402
from alpha_mining.context import thresholds  # noqa: E402
from alpha_mining.mining.diagnostics import diagnose  # noqa: E402

RES = ROOT / "results" / "review1"
LOGS = ROOT / "logs" / "llm" / "review1"
OUT = RES / "summary"
RUN_RE = re.compile(r"^deepseek_(B|C|Cd|Cs)(_lam1)?_s(\d)$")
CONTRASTS = [("E1", "Cd", "C"), ("E2", "C_lam1", "B_lam1"), ("E2", "C", "B"),
             ("E3", "C", "Cs"), ("E3", "Cs", "B")]
METRICS = ["too_deep_share", "invalid_share", "P1", "valid_icir", "valid_turnover",
           "r1_success", "r1_fixed_too_deep", "r2_success"]

am.RES = RES  # load_candidates reads <RES>/runs


def logged_rules(run: str, group: str) -> dict[int, int]:
    """Number of diagnoses (C, Cd) or statistics entries (Cs) in each sent prompt."""
    out = {}
    for line in open(LOGS / f"{run}.jsonl"):
        d = json.loads(line)
        user = d["request"]["messages"][-1]["content"]
        block = user.split("ADDITIONAL STATISTICS" if group == "Cs" else "DIAGNOSIS", 1)[-1]
        out[d["round"]] = len(re.findall(r"^\* ", block, re.M))
    return out


def analyse_run(run: str, th) -> dict:
    m = RUN_RE.match(run)
    group, lam = m.group(1), m.group(2) or ""
    cands = am.load_candidates(run)
    logged = logged_rules(run, group) if group != "B" else {}
    rep = {"R1": [0, 0, 0, 0], "R2": [0, 0, 0, 0]}  # fixed, valid, beat parent, too deep
    for rnd in range(1, max(c.round for c in cands) + 1):
        shown = am.select_shown(cands, rnd)
        diags = diagnose(shown, th)
        if group != "B" and len(diags) != logged.get(rnd, 0):
            raise RuntimeError(f"{run} round {rnd}: replay {len(diags)} != sent {logged.get(rnd)}")
        by_expr = {c.expr: c for c in shown}
        nxt = [c for c in cands if c.round == rnd]
        for d in diags:
            if d.rule not in rep:
                continue
            parent = by_expr[d.formula]
            pt = am.tree(parent.expr)
            fixed = [c for c in nxt if am.derived(pt, c.tree) and am.applies(d.rule, pt, c.tree)]
            r = rep[d.rule]
            r[0] += len(fixed)
            r[1] += sum(c.result is not None for c in fixed)
            r[2] += sum(c.result is not None and c.fitness > parent.fitness for c in fixed)
            r[3] += sum(c.status == "invalid" and "depth" in c.message for c in fixed)
    fin = json.load(open(RES / "final" / f"{run}.json"))
    n = len(cands)
    comp = fin.get("composite", {}).get("valid", {})
    return {"run": run, "config": f"{group}{lam}", "n": n,
            "too_deep_share": sum(c.status == "invalid" and "depth" in c.message for c in cands) / n,
            "invalid_share": sum(c.status == "invalid" for c in cands) / n,
            "duplicate_share": sum(c.status == "duplicate" for c in cands) / n,
            "P1": fin["P1"], "valid_icir": comp.get("icir", np.nan),
            "valid_turnover": comp.get("turnover", np.nan), "top_k": fin["top_k"],
            "r1_fixed": rep["R1"][0], "r1_success": rep["R1"][2] / rep["R1"][0] if rep["R1"][0] else np.nan,
            "r1_fixed_too_deep": rep["R1"][3] / rep["R1"][0] if rep["R1"][0] else np.nan,
            "r2_fixed": rep["R2"][0], "r2_success": rep["R2"][2] / rep["R2"][0] if rep["R2"][0] else np.nan}


def welch(x: np.ndarray, y: np.ndarray) -> dict:
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = np.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1)) if se > 0 else np.nan
    obs = x.mean() - y.mean()
    half = stats.t.ppf(0.975, df) * se if se > 0 else np.nan
    pooled = np.concatenate([x, y])
    null = [pooled[list(i)].mean() - np.delete(pooled, list(i)).mean()
            for i in itertools.combinations(range(len(pooled)), len(x))]
    return {"n": f"{len(x)}v{len(y)}", "diff": obs, "ci_lo": obs - half, "ci_hi": obs + half,
            "p_welch": 2 * stats.t.sf(abs(obs) / se, df) if se > 0 else np.nan,
            "p_perm": np.mean(np.abs(null) >= abs(obs) - 1e-12)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    th = thresholds()
    runs = sorted(p.stem for p in (RES / "runs").glob("*.jsonl") if RUN_RE.match(p.stem))
    per_run = pd.DataFrame([analyse_run(r, th) for r in runs])
    per_run.to_csv(OUT / "per_run.csv", index=False)
    means = per_run.drop(columns="run").groupby("config").agg(["mean", "std"]).round(3)
    print(per_run.groupby("config")[["n"]].size().to_string(), "\n")
    print(per_run.groupby("config")[METRICS + ["r1_fixed", "r2_fixed", "duplicate_share"]]
          .mean().round(3).T.to_string(), "\n")
    rows = []
    for exp, g, h in CONTRASTS:
        for m in METRICS:
            x = per_run.loc[per_run.config == g, m].to_numpy(float)
            y = per_run.loc[per_run.config == h, m].to_numpy(float)
            if np.isfinite(x).sum() >= 2 and np.isfinite(y).sum() >= 2:
                rows.append({"exp": exp, "contrast": f"{g}-{h}", "metric": m, **welch(x, y)})
    tests = pd.DataFrame(rows)
    tests.to_csv(OUT / "tests.csv", index=False)
    means.to_csv(OUT / "means.csv")
    print(tests.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
