import numpy as np
import pytest

from alpha_mining.baselines.alpha101 import Alpha101, _ts_arg, eligible_formulas, parse

T, N = 300, 40


@pytest.fixture(scope="module")
def engine():
    rng = np.random.default_rng(0)
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0))
    open_ = close * (1 + rng.normal(0, 0.005, (T, N)))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = rng.lognormal(13, 0.5, (T, N))
    prev = np.vstack([np.full((1, N), np.nan), close[:-1]])
    feats = {"open": open_, "high": high, "low": low, "close": close, "volume": volume,
             "returns": close / prev - 1, "dollar_vol": close * volume,
             "typical": (high + low + close) / 3}
    return Alpha101(feats, np.ones((T, N), dtype=bool))


def test_precedence_and_ternary():
    assert parse("a + b * c") == ("bin", "+", ("var", "a"), ("bin", "*", ("var", "b"), ("var", "c")))
    assert parse("a - b - c")[2][0] == "bin"  # left-associative
    assert parse("x < 1 ? 2 : y < 3 ? 4 : 5")[0] == "if"
    assert parse("-x ^ 2") == ("neg", ("bin", "^", ("var", "x"), ("num", 2.0)))


def test_ts_argmax_position():
    x = np.array([[1.0], [5.0], [2.0], [3.0]])
    out = _ts_arg(x, 3, np.argmax)
    assert out[2, 0] == 2 and out[3, 0] == 1  # windows [1,5,2] and [5,2,3]


def test_all_eligible_alphas_evaluate(engine):
    df = eligible_formulas()
    assert len(df) == 82
    for _, row in df.iterrows():
        out = engine.compute(row["formula"])
        assert out.shape == (T, N)
        assert np.isfinite(out[-1]).sum() > 0, f"Alpha#{row['alpha']} all NaN"


def test_known_alpha_values(engine):
    # Alpha#101 = (close - open) / ((high - low) + .001)
    v = engine.vars
    out = engine.compute("((close - open) / ((high - low) + .001))")
    np.testing.assert_allclose(out, (v["close"] - v["open"]) / (v["high"] - v["low"] + 0.001))
    # Alpha#12 = sign(delta(volume, 1)) * (-1 * delta(close, 1))
    out = engine.compute("(sign(delta(volume, 1)) * (-1 * delta(close, 1)))")
    exp = np.sign(v["volume"][1:] - v["volume"][:-1]) * -(v["close"][1:] - v["close"][:-1])
    np.testing.assert_allclose(out[1:], exp)
