"""Factor metrics: daily rank IC, ICIR, Newey-West t, weekly quintile backtest,
turnover, fitness and the cross-sectional factor correlation rho.

All functions take (T, N) arrays aligned with Panel.dates.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_STOCKS = 30
NW_LAGS = 4
COST_BPS = 10.0      # one-way, per unit of traded notional
WEEKS_PER_YEAR = 52


def rank_rows(x: np.ndarray) -> np.ndarray:
    return pd.DataFrame(x).rank(axis=1).to_numpy()


def rowcorr(a: np.ndarray, b: np.ndarray, min_n: int = MIN_STOCKS) -> np.ndarray:
    """Per-row Pearson correlation over entries finite in both rows."""
    ok = np.isfinite(a) & np.isfinite(b)
    n = ok.sum(axis=1)
    a0, b0 = np.where(ok, a, 0.0), np.where(ok, b, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = a0.sum(1, keepdims=True) / n[:, None]
        mb = b0.sum(1, keepdims=True) / n[:, None]
        da, db = np.where(ok, a0 - ma, 0.0), np.where(ok, b0 - mb, 0.0)
        r = (da * db).sum(1) / np.sqrt((da ** 2).sum(1) * (db ** 2).sum(1))
    r[n < min_n] = np.nan
    return r


def spearman_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact per-row Spearman: rank both on their joint support, then Pearson."""
    ok = np.isfinite(a) & np.isfinite(b)
    return rowcorr(rank_rows(np.where(ok, a, np.nan)), rank_rows(np.where(ok, b, np.nan)))


def newey_west_t(x: np.ndarray, lags: int = NW_LAGS) -> float:
    """t-statistic of the mean with Bartlett-kernel HAC standard error."""
    x = x[np.isfinite(x)]
    n = len(x)
    if n <= lags + 1:
        return np.nan
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else np.nan


def ic_stats(ic: np.ndarray) -> dict:
    v = ic[np.isfinite(ic)]
    sd = v.std(ddof=1) if len(v) > 1 else np.nan
    return {"ic": float(v.mean()) if len(v) else np.nan,
            "icir": float(v.mean() / sd) if sd and sd > 0 else np.nan,
            "ic_t": newey_west_t(ic), "n_days": int(len(v))}


def factor_rho(rank_f: np.ndarray, rank_g: np.ndarray) -> float:
    """|time-mean of daily cross-sectional Spearman| from cached rank matrices."""
    r = rowcorr(rank_f, rank_g)
    return float(abs(np.nanmean(r))) if np.isfinite(r).any() else np.nan


def rank_autocorr(f: np.ndarray, lag: int = 1) -> float:
    """Mean cross-sectional rank correlation of the factor with itself `lag` days earlier."""
    return float(np.nanmean(spearman_rows(f[lag:], f[:-lag])))


def week_ends(dates: pd.DatetimeIndex) -> np.ndarray:
    """Row index of the last trading day of each calendar week."""
    s = pd.Series(np.arange(len(dates)), index=dates)
    return s.groupby(dates.to_period("W")).max().to_numpy()


@dataclass
class Backtest:
    dates: pd.DatetimeIndex   # rebalance (signal) dates
    gross: np.ndarray         # long-short return per period, before costs
    net: np.ndarray
    turnover: np.ndarray      # one-way turnover per period, average of the two legs

    def summary(self) -> dict:
        ok = np.isfinite(self.net)
        g, n = self.gross[ok], self.net[ok]
        ann = np.sqrt(WEEKS_PER_YEAR)
        curve = np.cumprod(1 + n)
        mdd = float((1 - curve / np.maximum.accumulate(curve)).max()) if len(n) else np.nan
        sr = lambda r: float(r.mean() / r.std(ddof=1) * ann) if len(r) > 1 and r.std() > 0 else np.nan
        return {"sharpe_gross": sr(g), "sharpe_net": sr(n),
                "ann_return_net": float(curve[-1] ** (WEEKS_PER_YEAR / len(n)) - 1) if len(n) else np.nan,
                "max_drawdown": mdd, "turnover": float(np.nanmean(self.turnover[1:]))}


def quintile_backtest(f: np.ndarray, open_: np.ndarray, dates: pd.DatetimeIndex,
                      cost_bps: float = COST_BPS) -> Backtest:
    """Weekly equal-weight top-minus-bottom quintile portfolio.

    Signal at the close of each week's last trading day t; trade at open(t+1) and
    hold to the open after the next rebalance day. A stock whose prices stop
    mid-period (delisting, acquisition) is marked at its last open.
    """
    open_ = pd.DataFrame(open_).ffill(limit=5).to_numpy()
    reb = week_ends(dates)
    reb = reb[reb + 1 < len(dates)]
    T, N = f.shape
    w_prev_l = np.zeros(N)
    w_prev_s = np.zeros(N)
    gross, net, turn = [], [], []
    for i, t in enumerate(reb[:-1]):
        t_next = reb[i + 1]
        r = open_[t_next + 1] / open_[t + 1] - 1
        x = f[t]
        ok = np.isfinite(x) & np.isfinite(r)
        if ok.sum() < MIN_STOCKS:
            gross.append(np.nan), net.append(np.nan), turn.append(np.nan)
            continue
        q = pd.Series(x[ok]).rank(pct=True).to_numpy()
        idx = np.flatnonzero(ok)
        w_l, w_s = np.zeros(N), np.zeros(N)
        w_l[idx[q > 0.8]] = 1.0
        w_s[idx[q <= 0.2]] = 1.0
        w_l /= w_l.sum()
        w_s /= w_s.sum()
        traded = np.abs(w_l - w_prev_l).sum() + np.abs(w_s - w_prev_s).sum()
        g = np.nansum(w_l * np.where(ok, r, 0)) - np.nansum(w_s * np.where(ok, r, 0))
        gross.append(g)
        net.append(g - traded * cost_bps / 1e4)
        turn.append(traded / 4)  # 0.5 * sum|dw| per leg, averaged over 2 legs
        w_prev_l, w_prev_s = w_l, w_s
    return Backtest(dates[reb[:-1]], np.array(gross), np.array(net), np.array(turn))


def fitness(icir: float, turnover: float, lam: float, tau: float) -> float:
    if not np.isfinite(icir):
        return -np.inf
    return abs(icir) - lam * max(0.0, (turnover if np.isfinite(turnover) else 1.0) - tau)
