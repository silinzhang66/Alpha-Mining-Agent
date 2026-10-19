"""Two-level pool isolation check (pre-freeze check).

Mining is run twice: on the real panel, and on a copy whose raw data after the
training period (prices, volumes, membership, SPY) is replaced by random
numbers, with features, mask and labels rebuilt from it. The training pool and
everything fed back to the miner (candidate records, LLM prompts) must be
identical, while validation-period metrics must change.
"""
import json
import time

import numpy as np
import pandas as pd

from alpha_mining.baselines.gp import run_gp
from alpha_mining.config import TRAIN, VALID
from alpha_mining.context import config, new_pool, thresholds
from alpha_mining.data.panel import assemble, benchmark, load_panel
from alpha_mining.data.sectors import industry_codes
from alpha_mining.evaluation.evaluator import Evaluator
from alpha_mining.evaluation.regimes import regime_labels
from alpha_mining.llm.client import Reply
from alpha_mining.llm.loop import run_llm
from alpha_mining.mining.session import MiningSession

GP_BUDGET, LLM_ROUNDS = 200, 5


def scrambled(panel, spy: pd.Series, seed: int = 0):
    rng = np.random.default_rng(seed)
    after = panel.dates > TRAIN[1]
    shape = (after.sum(), len(panel.entities))
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.03, shape), axis=0))
    open_ = close * np.exp(rng.normal(0, 0.01, shape))
    fields = {k: v.copy() for k, v in panel.fields.items()}
    fields["adj_close"][after], fields["adj_open"][after] = close, open_
    fields["adj_high"][after] = np.maximum(open_, close) * 1.01
    fields["adj_low"][after] = np.minimum(open_, close) * 0.99
    fields["volume"][after] = rng.lognormal(14, 1, shape)
    fields["dollar_vol"][after] = close * fields["volume"][after]
    # Membership after the training period is randomized too: keep the real mask
    # on training rows and draw the rest.
    p = assemble(panel.dates, panel.entities, fields, np.ones(panel.mask.shape, bool))
    p.mask[~after] = panel.mask[~after]
    p.mask[after] = rng.random(shape) < 0.8
    spy = spy.copy()
    late = spy.index > TRAIN[1]
    spy[late] = spy[~late].iloc[-1] * np.exp(np.cumsum(rng.normal(0, 0.02, late.sum())))
    return p, spy


class ScriptedClient:
    """Fixed replies; records every prompt so prompts can be compared across runs."""

    def __init__(self, k: int):
        feats = ["close", "volume", "returns", "high", "low", "open", "dollar_vol", "typical"]
        wins = [3, 5, 10, 20, 30, 60]
        forms = [f"{op}({x}, {w})" for op in ("ts_mean", "ts_std", "ts_rank", "ts_delta")
                 for x in feats for w in wins]
        rng = np.random.default_rng(1)
        rng.shuffle(forms)
        self.replies = [json.dumps([{"hypothesis": "", "formula": f} for f in forms[i * k:(i + 1) * k]])
                        for i in range(LLM_ROUNDS)]
        self.prompts = []

    def chat(self, messages, **meta):
        self.prompts.append(messages)
        return Reply(self.replies[len(self.prompts) - 1], "", "scripted", 0, 0, None, 0.0)


def run_all(panel, spy) -> dict:
    cfg = config()
    ev = Evaluator(panel, industry_codes(panel.entities), regime_labels(spy, panel.dates),
                   cfg["lambda"], cfg["tau"])
    out = {}
    s = MiningSession(ev, new_pool(), GP_BUDGET)
    run_gp(s, pop_size=cfg["gp_pop_size"][str(GP_BUDGET)], seed=0)
    out["gp"] = s
    s = MiningSession(ev, new_pool(), LLM_ROUNDS * cfg["k"])
    client = ScriptedClient(cfg["k"])
    run_llm(s, client, "C", LLM_ROUNDS, cfg["k"], cfg["lambda"], cfg["tau"], thresholds())
    out["llm"], out["prompts"] = s, client.prompts
    out["valid_ic"] = ev.run("-ts_delta(close, 5)", *VALID).ic
    return out


def records(s: MiningSession) -> str:
    """Candidate records without wall-clock timings, which differ between any two runs."""
    recs = [c.to_json() for c in s.candidates]
    for r in recs:
        r["meta"].pop("eval_seconds", None)
    return json.dumps(recs, sort_keys=True, default=str)


def same_pool(a: MiningSession, b: MiningSession) -> bool:
    ma, mb = a.pool.members, b.pool.members
    return len(ma) == len(mb) and all(
        x.canonical == y.canonical and x.fitness == y.fitness and np.array_equal(x.ranks, y.ranks, equal_nan=True)
        for x, y in zip(ma, mb))


t0 = time.time()
panel, spy = load_panel(), benchmark()["adj_close"]
real = run_all(panel, spy)
fake = run_all(*scrambled(panel, spy))
checks = {
    "gp_records_identical": records(real["gp"]) == records(fake["gp"]),
    "gp_pool_identical": same_pool(real["gp"], fake["gp"]),
    "llm_records_identical": records(real["llm"]) == records(fake["llm"]),
    "llm_pool_identical": same_pool(real["llm"], fake["llm"]),
    "llm_prompts_identical": real["prompts"] == fake["prompts"],
    "validation_changed": real["valid_ic"] != fake["valid_ic"],
}
print(json.dumps(checks, indent=2))
print(f"GP candidates {len(real['gp'].candidates)}, pool {len(real['gp'].pool.members)}; "
      f"LLM candidates {len(real['llm'].candidates)}, pool {len(real['llm'].pool.members)}, "
      f"prompts {len(real['prompts'])}; rev5 valid IC real {real['valid_ic']:.5f} vs scrambled "
      f"{fake['valid_ic']:.5f}; {time.time() - t0:.0f}s")
print("PASS" if all(checks.values()) else "FAIL")
