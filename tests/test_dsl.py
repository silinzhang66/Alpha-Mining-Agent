import numpy as np
import pytest

from alpha_mining.dsl import ops, spec
from alpha_mining.dsl.evaluate import evaluate
from alpha_mining.dsl.parse import DSLError, canonical, parse

T, N = 120, 8


@pytest.fixture
def panel():
    rng = np.random.default_rng(0)
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0))
    open_ = close * (1 + rng.normal(0, 0.005, (T, N)))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = rng.lognormal(13, 0.5, (T, N))
    prev = np.vstack([np.full((1, N), np.nan), close[:-1]])
    feats = {
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
        "returns": close / prev - 1, "ret_overnight": open_ / prev - 1,
        "ret_intraday": close / open_ - 1, "dollar_vol": close * volume,
        "typical": (high + low + close) / 3,
    }
    mask = np.ones((T, N), dtype=bool)
    return feats, mask


# --- parsing / validation ----------------------------------------------------

@pytest.mark.parametrize("expr", [
    "rank(ts_corr(close, volume, 10))",
    "-ts_delta(close, 5)",
    "ts_decay_linear(ret_overnight, 5) / (ts_std(returns, 20) + 0.01)",
    "log(dollar_vol) * sign(ret_intraday)",
    "ts_delay(close, 1)",
])
def test_valid_expressions(expr):
    parse(expr)


@pytest.mark.parametrize("expr,msg", [
    ("ts_delay(close, -1)", "d must be one of"),
    ("ts_delta(close, 0)", "d must be one of"),
    ("ts_mean(close, 7)", "d must be one of"),
    ("ts_mean(close, 5.0)", "d must be one of"),
    ("ts_mean(close, volume)", "d must be one of"),
    ("vwap / close", "unknown feature"),
    ("ts_argmax(close, 5)", "unknown operator"),
    ("__import__('os').system('ls')", "unsupported call"),
    ("close.shift(-1)", "unsupported call"),
    ("close ** 2", "unsupported syntax"),
    ("close if volume else open", "unsupported syntax"),
    ("rank(close, 5)", "exactly 1 argument"),
    ("ts_corr(close, 5)", "takes 3 arguments"),
    ("3 + 4", "no feature"),
    ("rank(close", "syntax error"),
    ("close\nvolume", "single line"),
])
def test_rejected_expressions(expr, msg):
    with pytest.raises(DSLError, match=msg):
        parse(expr)


def test_depth_and_size_limits():
    deep = "close"
    for _ in range(spec.MAX_DEPTH):
        deep = f"abs({deep})"
    with pytest.raises(DSLError, match="depth"):
        parse(deep)
    def balanced(depth):  # 2**depth - 1 nodes
        return "close" if depth == 1 else f"({balanced(depth - 1)} + {balanced(depth - 1)})"
    wide = balanced(5) + " + volume"  # 33 nodes, depth 6
    with pytest.raises(DSLError, match="nodes"):
        parse(wide)


@pytest.mark.parametrize("a,b", [
    ("close + volume", "volume + close"),
    ("(close + volume) + open", "open + (volume + close)"),
    ("close * volume * open", "open * (close * volume)"),
    ("ts_corr(close, volume, 10)", "ts_corr(volume, close, 10)"),
    ("rank(close + 1)", "rank(1 + close)"),
])
def test_canonical_equivalent(a, b):
    assert canonical(parse(a)) == canonical(parse(b))


@pytest.mark.parametrize("a,b", [
    ("close - volume", "volume - close"),
    ("close / volume", "volume / close"),
    ("ts_mean(close, 5)", "ts_mean(close, 10)"),
])
def test_canonical_distinct(a, b):
    assert canonical(parse(a)) != canonical(parse(b))


def test_canonical_round_trip():
    for e in ["-(close + volume) / ts_std(returns, 20)", "rank(-ts_delta(close, 5)) * -2"]:
        c = canonical(parse(e))
        assert canonical(parse(c)) == c


# --- evaluation --------------------------------------------------------------

def test_ts_ops_match_naive(panel):
    feats, mask = panel
    x = feats["close"]
    d = 5
    t, i = 50, 3
    win = x[t - d + 1:t + 1, i]
    assert np.isclose(ops.ts_mean(x, d)[t, i], win.mean())
    assert np.isclose(ops.ts_std(x, d)[t, i], win.std(ddof=1))
    assert np.isclose(ops.ts_delay(x, d)[t, i], x[t - d, i])
    assert np.isclose(ops.ts_delta(x, d)[t, i], x[t, i] - x[t - d, i])
    w = np.arange(1, d + 1)
    assert np.isclose(ops.ts_decay_linear(x, d)[t, i], (win * w).sum() / w.sum())
    assert np.isclose(ops.ts_rank(x, d)[t, i], (win <= win[-1]).mean())
    y = feats["volume"]
    assert np.isclose(ops.ts_corr(x, y, d)[t, i], np.corrcoef(win, y[t - d + 1:t + 1, i])[0, 1])
    assert np.isnan(ops.ts_mean(x, d)[d - 2, i])


@pytest.mark.parametrize("expr", [
    "rank(ts_corr(close, volume, 10))",
    "ts_decay_linear(zscore(returns), 20) - ts_delay(close, 1)",
    "ts_rank(scale(dollar_vol), 30) * ts_cov(high, low, 5)",
    "ts_delta(log(volume), 3) / ts_max(abs(ret_intraday), 60)",
])
def test_no_lookahead(panel, expr):
    """Changing data after day t must not change factor values on or before t."""
    feats, mask = panel
    base = evaluate(expr, feats, mask)
    t = 80
    shocked = {k: v.copy() for k, v in feats.items()}
    for v in shocked.values():
        v[t + 1:] *= 3.0
    after = evaluate(expr, shocked, mask)
    np.testing.assert_array_equal(base[:t + 1], after[:t + 1])


def test_mask_excludes_nonmembers(panel):
    """A non-member is NaN in rank(close) and ts_mean(rank(close), 5), and does not
    affect the ranks of the other stocks (cross-section mask check)."""
    feats, mask = panel
    mask = mask.copy()
    mask[:100, 0] = False  # stock 0 joins on day 100
    r = evaluate("rank(close)", feats, mask)
    nested = evaluate("ts_mean(rank(close), 5)", feats, mask)
    assert np.isnan(r[:100, 0]).all() and np.isnan(nested[:104, 0]).all()
    assert not np.isnan(nested[104, 0])
    others = evaluate("rank(close)", {k: v[:, 1:] for k, v in feats.items()}, mask[:, 1:])
    np.testing.assert_allclose(r[:100, 1:], others[:100])


def test_protected_ops_finite(panel):
    feats, mask = panel
    zero = dict(feats, volume=np.zeros((T, N)))
    out = evaluate("close / volume + log(volume)", zero, mask)
    assert np.isfinite(out).all()


def test_elementwise_max_min(panel):
    feats, mask = panel
    up = evaluate("max(returns, 0)", feats, mask)
    assert np.allclose(up, np.maximum(feats["returns"], 0), equal_nan=True)
    shadow = evaluate("high - max(open, close)", feats, mask)
    assert np.nanmin(shadow) >= 0
    assert canonical(parse("min(open, close)")) == canonical(parse("min(close, open)"))
    assert canonical(parse("max(open, close)")) != canonical(parse("min(open, close)"))
    with pytest.raises(DSLError, match="takes exactly 2 arguments"):
        parse("max(close)")
