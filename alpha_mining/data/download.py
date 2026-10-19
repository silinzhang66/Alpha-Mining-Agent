"""Price download (yfinance, Tiingo for gaps) with per-entity cache and validation.

Every entity is cached in one schema, built from each source's raw fields:
  adj_open/adj_high/adj_low/adj_close  total-return adjusted prices
  volume                               split-adjusted share volume
  dollar_vol                           unadjusted close x unadjusted volume
  close_raw_split                      split-adjusted (not dividend-adjusted) close
Entities named ti_<SYMBOL> come from Tiingo; all others from yfinance
(`auto_adjust=False`, so Close/Volume are split-adjusted and Adj Close is
total-return adjusted).
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

from alpha_mining.config import HISTORY_START, PRICE_CACHE, ROOT, TRAIN, assert_allowed, data_end
from alpha_mining.data.universe import index_membership, membership, resolve

COVERAGE_REPORT = ROOT / "data" / "coverage_report.csv"
DAILY_COVERAGE = ROOT / "data" / "daily_coverage.csv"
TOLERANCE = pd.Timedelta(days=7)
MIN_DOLLAR_VOLUME = 5e6  # S&P 500 members trade far more; less suggests a wrong feed
MAX_GAP_DAYS = 30        # a reused ticker shows up as a long hole in the series


def cache_path(entity: str):
    return PRICE_CACHE / f"{entity}.parquet"


def _yfinance(symbol: str, end: pd.Timestamp) -> pd.DataFrame:
    h = yf.Ticker(symbol).history(
        start=HISTORY_START, end=end + pd.Timedelta(days=1), auto_adjust=False, actions=True,
    )
    if not len(h):
        return pd.DataFrame()
    h.index = h.index.tz_localize(None).normalize()
    f = h["Adj Close"] / h["Close"]
    return pd.DataFrame({
        "adj_open": h["Open"] * f, "adj_high": h["High"] * f, "adj_low": h["Low"] * f,
        "adj_close": h["Adj Close"], "volume": h["Volume"],
        "dollar_vol": h["Close"] * h["Volume"],  # split factors cancel: raw dollar volume
        "close_raw_split": h["Close"],
    })


def _tiingo(symbol: str, end: pd.Timestamp) -> pd.DataFrame:
    load_dotenv(ROOT / ".env")
    token = os.environ.get("TIINGO_API_KEY")
    if not token:
        raise RuntimeError("TIINGO_API_KEY is not set in .env")
    r = requests.get(
        f"https://api.tiingo.com/tiingo/daily/{symbol}/prices",
        params={"startDate": HISTORY_START.date().isoformat(),
                "endDate": end.date().isoformat(), "token": token},
        timeout=60,
    )
    if r.status_code == 404:
        return pd.DataFrame()
    r.raise_for_status()
    h = pd.DataFrame(r.json())
    if h.empty:
        return h
    h.index = pd.DatetimeIndex(pd.to_datetime(h.pop("date"))).tz_localize(None).normalize()
    # Split-adjust the raw close by the product of split factors on *later* days.
    later_splits = h["splitFactor"][::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    return pd.DataFrame({
        "adj_open": h["adjOpen"], "adj_high": h["adjHigh"], "adj_low": h["adjLow"],
        "adj_close": h["adjClose"], "volume": h["adjVolume"],
        "dollar_vol": h["close"] * h["volume"],
        "close_raw_split": h["close"] / later_splits,
    })


def fetch(entity: str, refresh: bool = False, retries: int = 8) -> pd.DataFrame:
    """Daily history for one entity in the common schema; empty results are not cached."""
    end = data_end()
    assert_allowed(end)
    path = cache_path(entity)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    for attempt in range(retries):
        try:
            h = _tiingo(entity[3:], end) if entity.startswith("ti_") else _yfinance(entity, end)
            break
        except requests.HTTPError as e:
            if attempt == retries - 1 or e.response.status_code not in (429, 500, 502, 503):
                raise
            # Tiingo's free tier caps requests per hour; wait for the window to roll.
            time.sleep(600 if e.response.status_code == 429 else 2 ** attempt)
        except RuntimeError:
            raise  # configuration problem (missing key): retrying cannot help
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    if len(h):
        h = h[h.index <= end]
        PRICE_CACHE.mkdir(parents=True, exist_ok=True)
        h.to_parquet(path)
    return h


def validate(entity: str, h: pd.DataFrame, member: pd.Series) -> dict:
    """Check the feed covers the membership span and looks like a large-cap stock."""
    on = member.index[member]
    rec = {"entity": entity, "rows": len(h), "need_start": on.min(), "need_end": on.max(),
           "first": h.index.min() if len(h) else pd.NaT,
           "last": h.index.max() if len(h) else pd.NaT}
    if not len(h):
        rec["status"] = "missing"
        return rec
    inside = h[(h.index >= on.min()) & (h.index <= on.max())]
    rec["med_dollar_vol"] = float(inside["dollar_vol"].median()) if len(inside) else np.nan
    rec["zero_ret_frac"] = float((inside["adj_close"].diff() == 0).mean()) if len(inside) else np.nan
    span = h[h.index >= on.min() - pd.Timedelta(days=120)]  # warm-up window plus membership
    rec["max_gap_days"] = int(span.index.to_series().diff().dt.days.max()) if len(span) > 1 else 0
    # Quality first: a partial feed (late start / early end) still counts towards
    # coverage, so it must pass the same checks as a full one.
    if (not rec["med_dollar_vol"] >= MIN_DOLLAR_VOLUME or rec["zero_ret_frac"] > 0.2
            or rec["max_gap_days"] > MAX_GAP_DAYS):
        rec["status"] = "suspect_feed"
    elif rec["first"] > on.min() + TOLERANCE:
        rec["status"] = "late_start"
    elif rec["last"] < on.max() - TOLERANCE:
        rec["status"] = "early_end"
    else:
        rec["status"] = "ok"
    return rec


ADJ_PRICES = ("adj_open", "adj_high", "adj_low", "adj_close")


def repair_adjustment(h: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Undo spurious jumps in the total-return adjustment factor.

    A corporate action (spin-off, special dividend) moves the split-adjusted
    close while the total-return close stays smooth. The reverse -- a large
    total-return move on a quiet split-adjusted day -- is a broken factor (DHR at
    the 2016 Fortive spin-off: +61% adjusted, +3.6% raw). On such days the
    adjusted history before the day is rescaled so the day's adjusted return
    equals the split-adjusted one. Returns the repaired frame and the fixes made.
    """
    a = h["adj_close"].pct_change()
    r = h["close_raw_split"].pct_change()
    bad = h.index[(a.abs() > 0.4) & (r.abs() < 0.1)]
    if not len(bad):
        return h, []
    h = h.copy()
    fixes = []
    for d in bad:
        m = (1 + a[d]) / (1 + r[d])
        h.loc[h.index < d, list(ADJ_PRICES)] *= m
        fixes.append((d, float(a[d]), float(r[d])))
    return h, fixes


def daily_coverage(dates: pd.DatetimeIndex, have: dict[str, pd.DatetimeIndex]) -> pd.DataFrame:
    """Per day: index members, members with a price bar that day, and the ratio."""
    raw = index_membership(dates)
    hit = np.zeros(len(dates), dtype=int)
    for t in raw.columns:
        for i in np.flatnonzero(raw[t].to_numpy()):
            ent = resolve(t, dates[i])
            hit[i] += ent is not None and ent in have and dates[i] in have[ent]
    n = raw.sum(axis=1).to_numpy()
    return pd.DataFrame({"members": n, "with_data": hit, "coverage": hit / n}, index=dates)


def download_all(workers: int = 4, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    end = data_end()
    dates = pd.bdate_range(TRAIN[0], end)
    mem = membership(dates)

    def one(ent):
        try:
            return validate(ent, fetch(ent, refresh=refresh), mem[ent])
        except Exception as e:
            return {"entity": ent, "status": f"error: {e}"}

    yf_ents = [e for e in mem.columns if not e.startswith("ti_")]
    ti_ents = [e for e in mem.columns if e.startswith("ti_")]
    with ThreadPoolExecutor(workers) as ex:
        recs = list(ex.map(one, yf_ents))
    recs += [one(e) for e in ti_ents]  # sequential: Tiingo rate limits
    report = pd.DataFrame(recs)
    report.to_csv(COVERAGE_REPORT, index=False)

    # Only validated feeds count towards coverage; a late start or early end is a
    # genuine gap in the feed, so the days it does cover still count.
    good = set(report.loc[report["status"].isin(["ok", "late_start", "early_end"]), "entity"])
    have = {e: pd.read_parquet(cache_path(e)).index for e in good}
    trading = pd.DatetimeIndex(sorted(set().union(*have.values()))).intersection(dates)
    cov = daily_coverage(trading, have)
    cov.to_csv(DAILY_COVERAGE, index_label="date")
    return report, cov
