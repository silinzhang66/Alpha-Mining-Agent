"""End-to-end mining tests on a synthetic panel with a fake LLM."""
import json

import numpy as np
import pandas as pd
import pytest

from alpha_mining.baselines.gp import run_gp
from alpha_mining.config import TRAIN
from alpha_mining.data.panel import Panel
from alpha_mining.evaluation.evaluator import Evaluator
from alpha_mining.llm import loop
from alpha_mining.llm.client import Reply
from alpha_mining.llm.prompts import build_messages, parse_reply
from alpha_mining.mining.pool import TrainingPool
from alpha_mining.mining.session import MiningSession

T, N = 700, 60


@pytest.fixture(scope="module")
def evaluator():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range(TRAIN[0] - pd.Timedelta(days=200), periods=T)
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0))
    open_ = close * (1 + rng.normal(0, 0.005, (T, N)))
    high, low = np.maximum(open_, close) * 1.01, np.minimum(open_, close) * 0.99
    volume = rng.lognormal(13, 0.5, (T, N))
    prev = np.vstack([np.full((1, N), np.nan), close[:-1]])
    feats = {"open": open_, "high": high, "low": low, "close": close, "volume": volume,
             "returns": close / prev - 1, "ret_overnight": open_ / prev - 1,
             "ret_intraday": close / open_ - 1, "dollar_vol": close * volume,
             "typical": (high + low + close) / 3}
    label = np.full((T, N), np.nan)
    label[:-6] = open_[6:] / open_[1:-5] - 1
    fields = {"adj_open": open_, "adj_close": close, "dollar_vol": close * volume}
    panel = Panel(dates, [f"S{i}" for i in range(N)], fields, feats, np.ones((T, N), bool), label)
    regimes = rng.integers(0, 3, T)
    return Evaluator(panel, rng.integers(0, 4, N), regimes, lam=0.1, tau=0.5)


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, **meta):
        self.calls.append(messages)
        return Reply(self.replies.pop(0), "", "fake", 0, 0, None, 0.0)


def test_session_budget_duplicates_and_invalid(evaluator):
    s = MiningSession(evaluator, TrainingPool(0.0, 0.0), budget=4)
    a = s.submit("ts_mean(returns, 5)", 1)
    b = s.submit("ts_mean(returns, 5) ", 1)
    c = s.submit("ts_mean(returns, 7)", 1)
    d = s.submit("rank(volume) * rank(close)", 1)
    assert (a.status, b.status, c.status, d.status) == ("ok", "duplicate", "invalid", "ok")
    assert s.remaining == 0 and a.pool.action == "added"
    with pytest.raises(RuntimeError):
        s.submit("close", 2)


def test_pool_greedy_replacement(evaluator):
    pool = TrainingPool(0.0, 0.0)
    s = MiningSession(evaluator, pool, budget=3)
    s.submit("ts_mean(returns, 5)", 1)
    child = s.submit("ts_decay_linear(ts_mean(returns, 5), 3)", 1)  # near-identical child
    assert child.pool.max_corr > 0.6
    assert child.pool.action in ("replaced", "rejected_corr")
    assert len(pool.members) == 1


def test_llm_loop_feedback_and_budget(evaluator):
    r1 = "```json\n" + json.dumps([{"hypothesis": "h", "formula": f"ts_mean(returns, {d})"}
                                    for d in (3, 5, 10)] + [{"formula": "bad(close)"}]) + "\n```"
    r2 = "not json at all"
    client = FakeClient([r1, r2, r2, r2])
    s = MiningSession(evaluator, TrainingPool(0.0, 0.0), budget=8)
    loop.run_llm(s, client, "C", rounds=2, k=4, lam=0.1, tau=0.5)
    assert s.used == 8
    second_prompt = client.calls[1][1]["content"]
    assert "bad(close)" in second_prompt and "unknown operator" in second_prompt
    assert "DIAGNOSIS" in second_prompt and "Fitness=" in second_prompt
    assert sum(c.status == "invalid" for c in s.candidates if c.round == 2) == 4


def test_group_a_prompt_has_no_theory():
    a = build_messages("A", 10, 0.1, 0.5, "fb")[0]["content"]
    b = build_messages("B", 10, 0.1, 0.5, "fb")[0]["content"]
    assert "THEORY" not in a and "stock" not in a.lower() and "hypothesis" not in a
    assert "THEORY" in b and "hypothesis" in b
    assert parse_reply('[{"formula": "close"}]') == [{"formula": "close", "hypothesis": ""}]


def test_gp_spends_exact_budget(evaluator):
    s = MiningSession(evaluator, TrainingPool(0.0, 0.0), budget=30)
    run_gp(s, pop_size=10, seed=1)
    assert s.used == 30
    assert {c.round for c in s.candidates} == {1, 2, 3}


def test_max_tokens_by_thinking_mode(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from alpha_mining.llm import client as C
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    monkeypatch.setattr(C, "LOG_DIR", tmp_path)
    seen = []

    def create(messages, **params):
        seen.append(params["max_tokens"])
        msg = SimpleNamespace(content="ok", reasoning_content="r" if params["extra_body"]["thinking"]["type"] == "enabled" else None,
                              model_extra={})
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, completion_tokens_details=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], model="deepseek-flash", usage=usage,
                               model_dump=lambda: {})
    for thinking in (False, True):
        cl = C.ChatClient("deepseek", thinking, 0.7, f"t{thinking}")
        cl.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        cl.chat([{"role": "user", "content": "hi"}])
    assert seen == [C.MAX_TOKENS, C.MAX_TOKENS_THINKING] and C.MAX_TOKENS == 4096
