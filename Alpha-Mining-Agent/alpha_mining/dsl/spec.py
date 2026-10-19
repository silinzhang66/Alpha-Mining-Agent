"""DSL whitelist: features, operators and allowed window parameters (Table 1)."""

BASE_FEATURES = ("open", "high", "low", "close", "volume", "returns")
DERIVED_FEATURES = ("ret_overnight", "ret_intraday", "dollar_vol", "typical")
FEATURES = BASE_FEATURES + DERIVED_FEATURES

WINDOWS = (3, 5, 10, 20, 30, 60)
LAG_WINDOWS = (1, 2) + WINDOWS  # ts_delay / ts_delta also allow short lags; all positive

CROSS_SECTIONAL = ("rank", "zscore", "scale")
ELEMENTWISE = ("abs", "log", "sign")
ELEMENTWISE_BINARY = ("max", "min")  # elementwise max(x, y) / min(x, y)
# name -> (number of series arguments, allowed windows)
TIME_SERIES = {
    "ts_mean": (1, WINDOWS),
    "ts_std": (1, WINDOWS),
    "ts_max": (1, WINDOWS),
    "ts_min": (1, WINDOWS),
    "ts_rank": (1, WINDOWS),
    "ts_decay_linear": (1, WINDOWS),
    "ts_delta": (1, LAG_WINDOWS),
    "ts_delay": (1, LAG_WINDOWS),
    "ts_corr": (2, WINDOWS),
    "ts_cov": (2, WINDOWS),
}
SYMMETRIC = ("ts_corr", "ts_cov") + ELEMENTWISE_BINARY

MAX_DEPTH = 6
MAX_NODES = 30
EPS = 1e-8
