"""API cost per mining run (paper Section 7, "Cost"), from the token usage in logs/llm/*.jsonl.

Every call is priced from its own usage record, splitting cached and uncached
input tokens. Prices are list prices per 1M tokens, checked 2026-10-18:

  DeepSeek (api-docs.deepseek.com/quick_start/pricing), model deepseek-flash
  = DeepSeek-V4.1-Flash-1017, USD; off-peak is half the peak price. Peak hours
  are 01:00-04:00 and 06:00-10:00 UTC, Monday to Friday (Chinese public
  holidays excluded; none fall on our call dates).

  Qwen (www.qianwenai.com/models/qwen3.8-flash#pricing, the platform whose
  example code uses our endpoint maas.qianwenaiapi.com), qwen3.8-flash, CNY:
  input 0.8, output 2.7, cache hit 0.1; no separate thinking-mode price.
  Converted to USD at 6.6918 CNY/USD (16 Oct 2026 close, poundsterlinglive.com).

  uv run python scripts/make_cost_table.py
"""
import json
import re
from datetime import datetime, timezone

import pandas as pd

from alpha_mining.config import ROOT

OUT = ROOT / "results" / "summary"
RUN_RE = re.compile(r"^(deepseek|qwen)_([ABC])(_think)?_s\d$")

DEEPSEEK_PEAK = {"hit": 0.006, "miss": 0.3, "out": 1.2}  # USD
QWEN_CNY = {"hit": 0.1, "miss": 0.8, "out": 2.7}
CNY_PER_USD = 6.6918


def deepseek_peak(t: datetime) -> bool:
    u = t.astimezone(timezone.utc)
    return u.weekday() < 5 and (1 <= u.hour < 4 or 6 <= u.hour < 10)


def usage(rec: dict) -> tuple[int, int, int]:
    u = rec["response"]["usage"]
    hit = u.get("prompt_cache_hit_tokens")
    if hit is None:
        hit = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return hit, u["prompt_tokens"] - hit, u["completion_tokens"]


def price(p: dict, hit: int, miss: int, out: int) -> float:
    return (hit * p["hit"] + miss * p["miss"] + out * p["out"]) / 1e6


def main() -> None:
    rows = []
    for path in sorted((ROOT / "logs" / "llm").glob("*.jsonl")):
        run = path.stem
        if not RUN_RE.match(run):
            continue
        row = {"config": re.sub(r"_s\d$", "", run), "run": run, "calls": 0,
               "hit": 0, "miss": 0, "out": 0, "peak_calls": 0, "usd": 0.0, "cny": float("nan")}
        for line in open(path):
            rec = json.loads(line)
            hit, miss, out = usage(rec)
            row["calls"] += 1
            row["hit"] += hit
            row["miss"] += miss
            row["out"] += out
            if run.startswith("deepseek"):
                peak = deepseek_peak(datetime.strptime(rec["time"], "%Y-%m-%dT%H:%M:%S%z"))
                row["peak_calls"] += peak
                row["usd"] += price(DEEPSEEK_PEAK, hit, miss, out) * (1 if peak else 0.5)
        if run.startswith("deepseek"):
            row["usd_peak"] = price(DEEPSEEK_PEAK, row["hit"], row["miss"], row["out"])
        else:
            row["cny"] = price(QWEN_CNY, row["hit"], row["miss"], row["out"])
            row["usd"] = row["usd_peak"] = row["cny"] / CNY_PER_USD
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cost_per_run.csv", index=False)
    cols = ["calls", "hit", "miss", "out", "peak_calls", "usd", "usd_peak", "cny"]
    t = df.groupby("config")[cols].mean()
    t["cache_share"] = t.hit / (t.hit + t.miss)
    t["total_usd_all_runs"] = df.groupby("config").usd.sum()
    t.to_csv(OUT / "table4_cost.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(t.round(4))
        print("total USD, all formal runs:", round(df.usd.sum(), 4))


if __name__ == "__main__":
    main()
