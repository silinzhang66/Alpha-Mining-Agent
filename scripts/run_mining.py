"""Run one mining configuration (GP or an LLM group) for one seed.

Examples:
  uv run python scripts/run_mining.py gp --budget 200 --seed 0
  uv run python scripts/run_mining.py llm --model deepseek --group C --seed 0
  uv run python scripts/run_mining.py llm --model qwen --group A --seed 0 --rounds 5   # pilot
  uv run python scripts/run_mining.py llm --model deepseek --group Cd --seed 0 --review 1  # review round 1
  uv run python scripts/run_mining.py llm --model deepseek --group B --lam 1 --seed 0 --review 1

Review-round runs (docs/review1_prespecification.md) use train/validation only and are
written to results/review<N>/runs and logs/llm/review<N>/, apart from the main runs.
"""
import argparse
import json
import time

from alpha_mining.baselines.gp import run_gp
from alpha_mining.config import ROOT
from alpha_mining.context import config, evaluator, new_pool, thresholds
from alpha_mining.llm.client import ChatClient
from alpha_mining.llm.loop import run_llm
from alpha_mining.mining.session import MiningSession

p = argparse.ArgumentParser()
p.add_argument("method", choices=["gp", "llm"])
p.add_argument("--seed", type=int, required=True)
p.add_argument("--budget", type=int, default=None, help="GP budget (default rounds*k)")
p.add_argument("--model", choices=["deepseek", "qwen"])
p.add_argument("--group", choices=["A", "B", "C", "Cd", "Cs"], help="Cd, Cs: review round 1 only")
p.add_argument("--rounds", type=int, default=None)
p.add_argument("--thinking", action="store_true", help="supplementary runs S1/S2 only")
p.add_argument("--tag", default="", help="suffix for pilot runs")
p.add_argument("--lam", type=float, default=None, help="turnover penalty override (review round 1, E2)")
p.add_argument("--review", type=int, default=None, help="review round: separate output folders")
args = p.parse_args()

cfg = config()
rounds = args.rounds or cfg["rounds"]
k = cfg["k"]
if args.method == "gp":
    budget = args.budget or rounds * k
    run_name = f"gp{budget}_s{args.seed}"
else:
    budget = rounds * k
    lam_tag = f"_lam{args.lam:g}" if args.lam is not None else ""
    run_name = f"{args.model}_{args.group}{lam_tag}{'_think' if args.thinking else ''}_s{args.seed}"
run_name += args.tag
if (args.lam is not None or args.group in ("Cd", "Cs")) and args.review is None:
    raise SystemExit("--lam and groups Cd/Cs are for review-round runs only (--review)")
lam = cfg["lambda"] if args.lam is None else args.lam
sub = f"review{args.review}" if args.review else ""

out_dir = ROOT / "results" / sub / "runs"
out_dir.mkdir(parents=True, exist_ok=True)
session = MiningSession(evaluator(args.lam), new_pool(), budget)
t0 = time.time()
if args.method == "gp":
    run_gp(session, pop_size=cfg["gp_pop_size"].get(str(budget), 10), seed=args.seed)
else:
    client = ChatClient(args.model, args.thinking, cfg["temperature"], f"{sub}/{run_name}" if sub else run_name)
    run_llm(session, client, args.group, rounds, k, lam, cfg["tau"], thresholds())
session.save(out_dir / f"{run_name}.jsonl")
meta = {"run": run_name, "args": vars(args), "config": cfg, "lambda": lam, "budget": budget,
        "seconds": round(time.time() - t0, 1), "pool_size": len(session.pool.members),
        "status_counts": {s: sum(c.status == s for c in session.candidates)
                          for s in ("ok", "invalid", "duplicate", "error")}}
(out_dir / f"{run_name}.meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta["status_counts"]), "pool", meta["pool_size"], f"{meta['seconds']}s")
