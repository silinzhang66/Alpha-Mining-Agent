"""Collect per-run results into the paper tables (mean ± sd over seeds).

Reads results/final/*.json, results/test/*.json (if the test set has been
evaluated), results/runs/*.meta.json and logs/llm/*.jsonl. Writes CSVs and a
Markdown version to results/summary/.

  Table 2   main results (P1, rho_bar, train/valid ICIR, Test I, Test II)
  aux       final/training pool sizes, duplicate share, decay ratio, gross SR,
            Test II after the probe cutoff (supplement)
  discrete  pre-registered P1 by discrete / continuous factors
  cost      tokens, API time and wall time per LLM run (Table 4 inputs)
  sensitivity  Table 2 columns under the IC valid-day rule (results/final_sensitivity)

  uv run python scripts/summarize.py
"""
import json
import re

import numpy as np
import pandas as pd

from alpha_mining.config import ROOT

RES = ROOT / "results"
OUT = RES / "summary"
ORDER = ["alpha101", "gp200", "gp1000", "deepseek_A", "deepseek_B", "deepseek_C",
         "qwen_A", "qwen_B", "qwen_C", "deepseek_A_think", "deepseek_C_think"]


def config_of(run: str) -> str:
    return re.sub(r"_s\d+$", "", run)


def g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return np.nan
        d = d[k]
    return d


def collect(final_dir: str, test_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted((RES / final_dir).glob("*.json")):
        f = json.loads(path.read_text())
        t = json.loads((RES / test_dir / path.name).read_text()) if (RES / test_dir / path.name).exists() else {}
        sub = next((k for k in t if k.startswith("test2_after_cutoff")), None)
        rows.append({
            "config": config_of(f["run"]), "run": f["run"],
            "P1": f["P1"], "rho_bar": f.get("rho_bar", np.nan),
            "train_icir": g(f, "composite", "train", "icir"), "valid_icir": g(f, "composite", "valid", "icir"),
            "test1_icir": g(t, "test1", "icir"), "test1_sr_net": g(t, "test1", "sharpe_net"),
            "test1_mdd": g(t, "test1", "max_drawdown"), "test1_turnover": g(t, "test1", "turnover"),
            "test2_icir": g(t, "test2", "icir"), "test2_sr_net": g(t, "test2", "sharpe_net"),
            # auxiliary
            "final_pool": f["final_pool_size"], "training_pool": f["training_pool_size"],
            "top_k": f["top_k"], "duplicate_share": f["duplicate_share"],
            "invalid_share": f["status_counts"]["invalid"] / f["budget"],
            "test1_sr_gross": g(t, "test1", "sharpe_gross"), "test2_sr_gross": g(t, "test2", "sharpe_gross"),
            "test1_ann_ret": g(t, "test1", "ann_return_net"),
            "test1_decay": g(t, "test1", "decay_ratio"), "test2_decay": g(t, "test2", "decay_ratio"),
            "test2_sub_icir": g(t, sub, "icir") if sub else np.nan,
            "test2_sub_sr_net": g(t, sub, "sharpe_net") if sub else np.nan,
            # pre-registered discrete / continuous reporting
            "discrete_share": f["discrete_n"] / max(1, f["discrete_n"] + f["continuous_n"]),
            "discrete_P1": f["discrete_P1"], "continuous_P1": f["continuous_P1"],
            "final_pool_discrete_share": f["final_pool_discrete"] / f["final_pool_size"] if f["final_pool_size"] else np.nan,
        })
    return pd.DataFrame(rows)


df = collect("final", "test")
if df.empty:
    raise SystemExit("no results/final/*.json yet; run scripts/build_final.py first")


def table(cols: list[str], df: pd.DataFrame = df) -> pd.DataFrame:
    def fmt(s: pd.Series) -> str:
        s = s.astype(float).dropna()
        if s.empty:
            return "–"
        return f"{s.mean():.3f}" if len(s) == 1 else f"{s.mean():.3f} ± {s.std(ddof=1):.3f}"
    out = df.groupby("config").agg(seeds=("run", "size"), **{c: (c, fmt) for c in cols})
    return out.reindex([c for c in ORDER if c in out.index] + sorted(set(out.index) - set(ORDER)))


tables = {
    "table2_main": table(["P1", "rho_bar", "train_icir", "valid_icir", "test1_icir", "test1_sr_net",
                          "test1_mdd", "test1_turnover", "test2_icir", "test2_sr_net"]),
    "aux": table(["final_pool", "training_pool", "top_k", "duplicate_share", "invalid_share",
                  "test1_sr_gross", "test2_sr_gross", "test1_ann_ret", "test1_decay", "test2_decay",
                  "test2_sub_icir", "test2_sub_sr_net"]),
    "discrete": table(["discrete_share", "discrete_P1", "continuous_P1", "final_pool_discrete_share"]),
}
MAIN_COLS = ["P1", "rho_bar", "train_icir", "valid_icir", "test1_icir", "test1_sr_net",
             "test1_mdd", "test1_turnover", "test2_icir", "test2_sr_net"]
if (RES / "final_sensitivity").exists():
    tables["sensitivity"] = table(MAIN_COLS + ["final_pool"], collect("final_sensitivity", "test_sensitivity"))

cost = []
for path in sorted((RES / "runs").glob("*.meta.json")):
    meta = json.loads(path.read_text())
    run = meta["run"]
    log = ROOT / "logs" / "llm" / f"{run}.jsonl"
    if "_pilot" in run or meta["args"]["method"] != "llm" or not log.exists():
        continue
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    cost.append({"config": config_of(run), "run": run, "calls": len(recs),
                 "prompt_tokens": sum(r["response"]["usage"]["prompt_tokens"] for r in recs),
                 "completion_tokens": sum(r["response"]["usage"]["completion_tokens"] for r in recs),
                 "api_seconds": sum(r["seconds"] for r in recs), "wall_seconds": meta["seconds"]})
for path in sorted((RES / "runs").glob("gp*.meta.json")):
    meta = json.loads(path.read_text())
    cost.append({"config": config_of(meta["run"]), "run": meta["run"], "wall_seconds": meta["seconds"]})
if cost:
    tables["cost"] = pd.DataFrame(cost).groupby("config").mean(numeric_only=True).round(1)



def markdown(t: pd.DataFrame) -> str:
    t = t.reset_index()
    lines = ["| " + " | ".join(map(str, t.columns)) + " |", "|" + " --- |" * len(t.columns)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in t.itertuples(index=False)]
    return "\n".join(lines)


OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / "per_run.csv", index=False)
md = []
for name, t in tables.items():
    t.to_csv(OUT / f"{name}.csv")
    md += [f"## {name}", "", markdown(t), ""]
(OUT / "tables.md").write_text("\n".join(md))
print("\n".join(md))
