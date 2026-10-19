"""Market states for diagnostic rule R1.

Each day is labelled by SPY's trailing 20-day return relative to the tertiles of
that return over the training period: 0 = down-pressure, 1 = range, 2 = up.
The label for day t uses prices up to t only; the tertile cut points use the
training period only and are applied to training-period diagnostics.
"""
import numpy as np
import pandas as pd

from alpha_mining.config import TRAIN

LOOKBACK = 20
NAMES = ("down_pressure", "range", "up")


def trailing_return(close: pd.Series, lookback: int = LOOKBACK) -> pd.Series:
    return close / close.shift(lookback) - 1


def regime_labels(spy_close: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """Integer state per date in `dates` (-1 where the lookback is unavailable)."""
    r = trailing_return(spy_close)
    train = r[(r.index >= TRAIN[0]) & (r.index <= TRAIN[1])].dropna()
    cuts = train.quantile([1 / 3, 2 / 3]).to_numpy()
    r = r.reindex(dates)
    out = np.digitize(r.to_numpy(), cuts)
    out[~np.isfinite(r.to_numpy())] = -1
    return out
