import numpy as np
import pandas as pd
import pytest

from alpha_mining.evaluation import metrics as M
from alpha_mining.evaluation.preprocess import Neutralizer, preprocess, winsorize_mad

T, N = 300, 120


@pytest.fixture
def data():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=T)
    ret = rng.normal(0, 0.02, (T, N))
    open_ = 100 * np.exp(np.cumsum(ret, axis=0))
    mask = np.ones((T, N), dtype=bool)
    sector = rng.integers(0, 5, N)
    log_adv = rng.normal(17, 1, (T, N))
    return dates, open_, mask, sector, log_adv


def test_spearman_identity_and_sign():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(10, 50))
    assert np.allclose(M.spearman_rows(a, a), 1)
    assert np.allclose(M.spearman_rows(a, -a ** 3), -1)


def test_newey_west_reduces_to_iid_t_without_lags():
    rng = np.random.default_rng(0)
    x = rng.normal(0.1, 1, 2000)
    t0 = M.newey_west_t(x, lags=0)
    assert np.isclose(t0, x.mean() / (x.std() / np.sqrt(len(x))))
    # Positively autocorrelated series: HAC t is smaller than the naive t.
    y = np.convolve(x, np.ones(5) / 5, mode="valid")
    assert M.newey_west_t(y, 4) < M.newey_west_t(y, 0)


def test_residual_orthogonal_to_design(data):
    dates, open_, mask, sector, log_adv = data
    rng = np.random.default_rng(2)
    f = rng.normal(size=(T, N)) + sector[None, :] * 0.5 + 0.3 * log_adv
    neu = Neutralizer(sector, log_adv, mask)
    r = neu.residual(f)
    t = 10
    for c in np.unique(sector):
        assert abs(np.nanmean(r[t, sector == c])) < 1e-6
    assert abs(np.nansum(r[t] * log_adv[t])) < 1e-3 * N


def test_projection_matches_lstsq_with_missing(data):
    dates, open_, mask, sector, log_adv = data
    rng = np.random.default_rng(3)
    f = rng.normal(size=(T, N))
    f[5, :3] = np.nan          # 2.5% missing -> projection path
    f[6, :20] = np.nan         # 17% missing -> exact regression path
    neu = Neutralizer(sector, log_adv, mask)
    r = neu.residual(f)
    for t in (5, 6):
        ok = np.isfinite(f[t])
        X = neu.X[t][ok]
        exact = f[t, ok] - X @ np.linalg.lstsq(X, f[t, ok], rcond=None)[0]
        assert np.isnan(r[t, ~ok]).all()
        tol = 1e-4 if t == 6 else 0.05  # mean-filling is an approximation
        assert np.allclose(r[t, ok], exact, atol=tol)


def test_winsorize_clips_outliers():
    x = np.tile(np.r_[np.linspace(-1, 1, 99), 100.0], (2, 1))
    w = winsorize_mad(x)
    assert w[0, -1] < 5 and np.allclose(w[0, :99], x[0, :99])


def test_backtest_perfect_foresight_and_turnover(data):
    dates, open_, mask, *_ = data
    reb = M.week_ends(dates)
    fut = np.full((T, N), np.nan)
    for i, t in enumerate(reb[:-1]):
        if reb[i + 1] + 1 < T:
            fut[t] = open_[reb[i + 1] + 1] / open_[t + 1] - 1
    bt = M.quintile_backtest(fut, open_, dates)
    assert np.nanmin(bt.gross) > 0
    # A constant signal never trades after the first period.
    const = np.tile(np.arange(N, dtype=float), (T, 1))
    bt2 = M.quintile_backtest(const, open_, dates)
    assert np.allclose(bt2.turnover[1:], 0)
    assert np.isclose(bt2.turnover[0], 0.5)  # 0.5 * (1 + 1) per leg, averaged


def test_preprocess_output_standardized(data):
    dates, open_, mask, sector, log_adv = data
    f = np.random.default_rng(4).normal(size=(T, N))
    z = preprocess(f, Neutralizer(sector, log_adv, mask))
    assert np.allclose(np.nanmean(z, axis=1), 0, atol=1e-8)
    assert np.allclose(np.nanstd(z, axis=1), 1)


def test_discrete_classification():
    from alpha_mining.evaluation.robustness import distinct_ratio, is_discrete
    rng = np.random.default_rng(0)
    cont = rng.normal(size=(20, 100))
    disc = (cont > 0).astype(float)
    rows = np.arange(20)
    assert distinct_ratio(cont, rows) == 1.0
    assert not is_discrete(cont, rows)
    assert distinct_ratio(disc, rows) == 0.02
    assert is_discrete(disc, rows)
    cont[:, :50] = np.nan  # NaNs are ignored, not counted as ties
    assert not is_discrete(cont, rows)
    assert is_discrete(np.full((20, 100), np.nan), rows)


def test_final_pool_rules():
    from alpha_mining.analysis.final import qualified, composite
    g = {"train_abs_ic_median": 0.004, "train_abs_icir_median": 0.04}
    good = {"ic": 0.01, "icir": 0.1}
    assert qualified(good, {"ic": 0.005, "icir": 0.05}, g)
    assert not qualified(good, {"ic": -0.01, "icir": -0.1}, g)      # sign flip
    assert not qualified(good, {"ic": 0.003, "icir": 0.1}, g)       # weak validation IC
    assert not qualified({"ic": 0.01, "icir": 0.03}, good, g)       # weak training ICIR
    assert not qualified(good, None, g)
    a = np.array([[1.0, np.nan, np.nan], [2.0, 2.0, np.nan]])
    b = np.array([[3.0, 1.0, np.nan], [np.nan, 4.0, np.nan]])
    c = composite([a, b], [1.0, -1.0])
    np.testing.assert_allclose(c[0, :2], [-1.0, -1.0])
    np.testing.assert_allclose(c[1, :2], [2.0, -1.0])
    assert np.isnan(c[:, 2]).all()


def test_final_pool_dedup_by_valid_icir():
    from alpha_mining.analysis.final import Factor, final_pool

    class Ev:
        def period_rows(self, start, end):
            return np.arange(40)

    rng = np.random.default_rng(1)
    base = rng.normal(size=(40, 60))
    other = rng.normal(size=(40, 60))
    def fac(key, vals, icir, q=True):
        return Factor(key, key, {}, False, 1.0, {"icir": icir, "ic": icir / 10}, q, vals)
    fs = [fac("weak_twin", base + 0.05 * rng.normal(size=base.shape), 0.10),
          fac("strong", base, -0.30),                     # |ICIR| ranks first
          fac("indep", other, 0.20),
          fac("unqualified", rng.normal(size=base.shape), 0.90, q=False)]
    assert [f.key for f in final_pool(fs, Ev(), 0.6)] == ["strong", "indep"]
