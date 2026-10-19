"""Point-in-time S&P 500 membership from fja05680/sp500, keyed by price entity.

Index tickers change over time (FB -> META) and are sometimes reused by other
companies (SE). Membership is therefore resolved to *entities* -- a yfinance
symbol, or ti_<SYMBOL> for a Tiingo series -- through the date-ranged maps in
data/ticker_map.csv (hand-curated) and data/tiingo_map.csv (generated; only
used for tickers the hand-curated map does not mention).
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from alpha_mining.config import DATA_RAW, ROOT, TRAIN

HIST_FILE = DATA_RAW / "sp500_hist.csv"
TICKER_MAP_FILE = ROOT / "data" / "ticker_map.csv"
TIINGO_MAP_FILE = ROOT / "data" / "tiingo_map.csv"


@lru_cache
def load_snapshots() -> pd.DataFrame:
    """One row per change date with the full member list (as sets)."""
    df = pd.read_csv(HIST_FILE, parse_dates=["date"])
    df["tickers"] = df["tickers"].str.split(",").map(frozenset)
    return df.sort_values("date").reset_index(drop=True)


@lru_cache
def load_ticker_map() -> pd.DataFrame:
    read = lambda f: pd.read_csv(f, comment="#", dtype=str, keep_default_na=False)
    m = read(TICKER_MAP_FILE)
    if TIINGO_MAP_FILE.exists():
        ti = read(TIINGO_MAP_FILE)
        m = pd.concat([m, ti[~ti["index_ticker"].isin(m["index_ticker"])]], ignore_index=True)
    m["start"] = pd.to_datetime(m["start"].replace("", None)).fillna(pd.Timestamp.min)
    m["end"] = pd.to_datetime(m["end"].replace("", None)).fillna(pd.Timestamp.max)
    return m


def resolve(ticker: str, date: pd.Timestamp) -> str | None:
    """Entity holding `ticker`'s prices on `date` (None = no usable data)."""
    m = load_ticker_map()
    hit = m[(m["index_ticker"] == ticker) & (m["start"] <= date) & (m["end"] >= date)]
    if len(hit):
        return hit["entity"].iloc[0] or None
    return ticker.replace(".", "-")


def index_membership(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Boolean (T, N) frame over index tickers; each snapshot holds until the next change."""
    snaps = load_snapshots()
    idx = snaps["date"].searchsorted(dates, side="right") - 1
    if (idx < 0).any():
        raise ValueError("dates precede the first membership snapshot")
    tickers = sorted(set().union(*snaps["tickers"].iloc[idx.min():idx.max() + 1]))
    col = {t: i for i, t in enumerate(tickers)}
    arr = np.zeros((len(dates), len(tickers)), dtype=bool)
    for row, s in enumerate(idx):
        arr[row, [col[t] for t in snaps["tickers"].iloc[s]]] = True
    return pd.DataFrame(arr, index=dates, columns=tickers)


def membership(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Boolean (T, N) frame over entities.

    Index tickers without usable data are dropped, so a False entry can mean
    either "not a member" or "member without a price feed"; see `index_membership`
    for the latter when measuring coverage.
    """
    raw = index_membership(dates)
    m = load_ticker_map()
    cols: dict[str, pd.Series] = {}

    def add(sym: str, on: pd.Series) -> None:
        cols[sym] = cols[sym] | on if sym in cols else on

    for t in raw.columns:
        covered = pd.Series(False, index=dates)
        for _, row in m[m["index_ticker"] == t].iterrows():
            in_range = pd.Series((dates >= row["start"]) & (dates <= row["end"]), index=dates)
            if row["entity"]:
                add(row["entity"], raw[t] & in_range)
            covered |= in_range
        add(t.replace(".", "-"), raw[t] & ~covered)
    out = pd.DataFrame(cols).sort_index(axis=1)
    return out.loc[:, out.any()]
    return pd.DataFrame(cols).sort_index(axis=1)


def entities_since(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    dates = pd.bdate_range(start, end)
    m = membership(dates)
    return sorted(m.columns[m.any()])
