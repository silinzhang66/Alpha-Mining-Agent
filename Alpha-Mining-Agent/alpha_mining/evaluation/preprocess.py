"""Cross-sectional preprocessing: 3x MAD winsorization -> neutralization -> z-score.

Neutralization regresses each day's factor on sector dummies and log(ADV20) and
keeps the residual. The design matrix is the same for every factor, so its QR
basis is computed once per day and a residual is f - Q (Q^T f).
"""
import warnings

import numpy as np

MAD_K = 3.0
MAX_MISSING_FOR_PROJECTION = 0.05  # above this, fall back to an exact regression


def _rowwise(fn, x):
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN days -> NaN
        return fn(x)


def winsorize_mad(x: np.ndarray, k: float = MAD_K) -> np.ndarray:
    med = _rowwise(lambda a: np.nanmedian(a, axis=1, keepdims=True), x)
    mad = _rowwise(lambda a: np.nanmedian(np.abs(a - med), axis=1, keepdims=True), x)
    # 1.4826 * MAD estimates the standard deviation under normality.
    lo, hi = med - k * 1.4826 * mad, med + k * 1.4826 * mad
    return np.clip(x, lo, hi)


def zscore(x: np.ndarray) -> np.ndarray:
    mu = _rowwise(lambda a: np.nanmean(a, axis=1, keepdims=True), x)
    sd = _rowwise(lambda a: np.nanstd(a, axis=1, keepdims=True), x)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (x - mu) / sd
    return np.where(np.isfinite(out), out, np.nan)


class Neutralizer:
    """Per-day orthonormal basis of [sector dummies, log ADV20] over eligible stocks.

    sector: (N,) integer codes (or (T, N) for point-in-time codes), -1 = unknown.
    log_adv: (T, N) log 20-day average dollar volume.
    mask: (T, N) bool universe; stocks with an unknown sector or no ADV are
    left out of the regression and come out as NaN.
    """

    def __init__(self, sector: np.ndarray, log_adv: np.ndarray, mask: np.ndarray):
        T, N = mask.shape
        sector = np.broadcast_to(sector, (T, N))
        self.eligible = mask & (sector >= 0) & np.isfinite(log_adv)
        self.idx: list[np.ndarray] = []
        self.X: list[np.ndarray] = []
        self.Q: list[np.ndarray] = []
        for t in range(T):
            cols = np.flatnonzero(self.eligible[t])
            s = sector[t, cols]
            codes = np.unique(s)
            # All sector dummies, no separate intercept (the dummies span it).
            X = np.column_stack([(s == c).astype(np.float64) for c in codes] + [log_adv[t, cols]])
            self.idx.append(cols)
            self.X.append(X)
            self.Q.append(np.linalg.qr(X)[0].astype(np.float32) if len(cols) > X.shape[1] else None)

    def residual(self, f: np.ndarray) -> np.ndarray:
        out = np.full(f.shape, np.nan)
        for t, cols in enumerate(self.idx):
            Q = self.Q[t]
            if Q is None:
                continue
            y = f[t, cols]
            ok = np.isfinite(y)
            if ok.sum() <= Q.shape[1]:
                continue
            if ok.mean() >= 1 - MAX_MISSING_FOR_PROJECTION:
                # Fill gaps with the day's mean, project, then put the gaps back.
                y_fill = np.where(ok, y, y[ok].mean()).astype(np.float64)
                r = y_fill - Q.astype(np.float64) @ (Q.T.astype(np.float64) @ y_fill)
            else:
                X = self.X[t][ok]
                beta = np.linalg.lstsq(X, y[ok], rcond=None)[0]
                r = np.full(len(cols), np.nan)
                r[ok] = y[ok] - X @ beta
            out[t, cols] = np.where(ok, r, np.nan)
        return out


def preprocess(f: np.ndarray, neutralizer: Neutralizer | None) -> np.ndarray:
    x = winsorize_mad(f)
    if neutralizer is not None:
        x = neutralizer.residual(x)
    return zscore(x)
