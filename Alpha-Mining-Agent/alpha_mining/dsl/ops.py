"""Vectorized operator implementations on (T, N) float arrays.

Time-series operators run on the full, unmasked panel and need a complete window
(min_periods = d). Cross-sectional operators take the day's membership mask and
only rank/standardize among that day's members; non-members come out as NaN.
"""
import warnings

import numpy as np
import pandas as pd

from alpha_mining.dsl.spec import EPS


def _clean(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return np.where(np.isfinite(a), a, np.nan)


def _df(a: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(a)


# --- time series -----------------------------------------------------------

def ts_mean(x, d):
    return _clean(_df(x).rolling(d, min_periods=d).mean().to_numpy())


def ts_std(x, d):
    return _clean(_df(x).rolling(d, min_periods=d).std().to_numpy())


def ts_max(x, d):
    return _clean(_df(x).rolling(d, min_periods=d).max().to_numpy())


def ts_min(x, d):
    return _clean(_df(x).rolling(d, min_periods=d).min().to_numpy())


def ts_rank(x, d):
    """Percentile rank of today's value within the trailing d-day window."""
    return _clean(_df(x).rolling(d, min_periods=d).rank(pct=True).to_numpy())


def ts_delay(x, d):
    out = np.full_like(x, np.nan, dtype=np.float64)
    out[d:] = x[:-d]
    return out


def ts_delta(x, d):
    return _clean(x - ts_delay(x, d))


def ts_decay_linear(x, d):
    """Weighted mean with weights d, d-1, ..., 1 (today weighted most)."""
    w = np.arange(d, 0, -1, dtype=np.float64)
    w /= w.sum()
    out = np.zeros_like(x, dtype=np.float64)
    for lag in range(d):
        out += w[lag] * ts_delay(x, lag) if lag else w[0] * x
    return _clean(out)


def ts_corr(x, y, d):
    return _clean(_df(x).rolling(d, min_periods=d).corr(_df(y)).to_numpy())


def ts_cov(x, y, d):
    return _clean(_df(x).rolling(d, min_periods=d).cov(_df(y)).to_numpy())


# --- cross section ---------------------------------------------------------

def _masked(x, mask):
    return np.where(mask, x, np.nan)


def rank(x, mask):
    """Percentile rank among the day's members, in (0, 1]."""
    return _df(_masked(x, mask)).rank(axis=1, pct=True).to_numpy()


def zscore(x, mask):
    m = _masked(x, mask)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN days -> NaN
        mu = np.nanmean(m, axis=1, keepdims=True)
        sd = np.nanstd(m, axis=1, keepdims=True)
        return _clean((m - mu) / sd)


def scale(x, mask):
    """Rescale so the day's absolute values sum to 1."""
    m = _masked(x, mask)
    with np.errstate(invalid="ignore", divide="ignore"):
        return _clean(m / np.nansum(np.abs(m), axis=1, keepdims=True))


# --- elementwise / algebra ---------------------------------------------------

def protected_div(x, y):
    """x / y with |y| floored at EPS (sign kept), so near-zero denominators stay finite."""
    y = np.where(np.abs(y) < EPS, np.where(y < 0, -EPS, EPS), y)
    return _clean(x / y)


def protected_log(x):
    return _clean(np.log(np.abs(x) + EPS))
