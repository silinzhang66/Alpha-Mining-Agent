"""Aligned (T, N) price panel, DSL features, membership mask and forward labels.

Time-series features are built on the full panel; `mask` marks the stocks that
are index members on a day, have a clean bar that day and at least
MIN_HISTORY clean bars so far. Labels are open(t+1) -> open(t+1+HORIZON)
returns, so a signal computed after the close of t is traded at the next open.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_mining.config import DATA_CACHE, HISTORY_START, assert_allowed, data_end
from alpha_mining.data.download import COVERAGE_REPORT, cache_path, fetch, repair_adjustment
from alpha_mining.data.universe import membership

MIN_HISTORY = 60
HORIZON = 5
USABLE = ("ok", "late_start", "early_end")
FIELDS = ("adj_open", "adj_high", "adj_low", "adj_close", "volume", "dollar_vol")


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    entities: list[str]
    fields: dict[str, np.ndarray]    # raw aligned fields, NaN on unusable bars
    features: dict[str, np.ndarray]  # DSL features (spec.FEATURES)
    mask: np.ndarray                 # bool (T, N)
    label: np.ndarray                # forward HORIZON-day open-to-open return

    def slice(self, start, end) -> "Panel":
        """Rows in [start, end]; time-series history before `start` is lost, so
        evaluate factors on the full panel and slice the results instead."""
        sel = (self.dates >= start) & (self.dates <= end)
        cut = lambda d: {k: v[sel] for k, v in d.items()}
        return Panel(self.dates[sel], self.entities, cut(self.fields), cut(self.features),
                     self.mask[sel], self.label[sel])


BENCHMARK = "SPY"  # trading calendar and market-state series


def benchmark() -> pd.DataFrame:
    return fetch(BENCHMARK)


def build_panel() -> Panel:
    end = data_end()
    assert_allowed(end)
    report = pd.read_csv(COVERAGE_REPORT)
    dates = benchmark().index
    dates = dates[(dates >= HISTORY_START) & (dates <= end)]
    member_all = membership(dates)
    usable = set(report.loc[report["status"].isin(USABLE), "entity"])
    entities = sorted(usable & set(member_all.columns))
    frames = {e: repair_adjustment(pd.read_parquet(cache_path(e)))[0] for e in entities}

    fields = {f: np.full((len(dates), len(entities)), np.nan) for f in FIELDS}
    for j, e in enumerate(entities):
        h = frames[e].reindex(dates)
        bad = (h["volume"] <= 0) | h[["adj_open", "adj_high", "adj_low", "adj_close"]].le(0).any(axis=1)
        h[bad] = np.nan
        for f in FIELDS:
            fields[f][:, j] = h[f].to_numpy()
    return assemble(dates, entities, fields, member_all[entities].to_numpy())


def assemble(dates: pd.DatetimeIndex, entities: list[str], fields: dict[str, np.ndarray],
             member: np.ndarray) -> Panel:
    """Features, mask and labels from aligned raw fields and index membership."""
    o, hi, lo, c = (fields[k] for k in ("adj_open", "adj_high", "adj_low", "adj_close"))
    prev_c = np.vstack([np.full((1, c.shape[1]), np.nan), c[:-1]])
    features = {
        "open": o, "high": hi, "low": lo, "close": c, "volume": fields["volume"],
        "returns": c / prev_c - 1,
        "ret_overnight": o / prev_c - 1,
        "ret_intraday": c / o - 1,
        "dollar_vol": fields["dollar_vol"],
        "typical": (hi + lo + c) / 3,
    }

    valid = np.isfinite(c) & np.isfinite(o)
    history = np.cumsum(valid, axis=0)
    mask = member & valid & (history >= MIN_HISTORY)

    label = np.full_like(o, np.nan)
    label[:-(HORIZON + 1)] = o[HORIZON + 1:] / o[1:-HORIZON] - 1
    return Panel(dates, entities, fields, features, mask, label)


def panel_path():
    return DATA_CACHE / f"panel_{data_end().date()}.npz"


def load_panel(rebuild: bool = False) -> Panel:
    path = panel_path()
    if path.exists() and not rebuild:
        z = np.load(path, allow_pickle=False)
        fields = {k[2:]: z[k] for k in z.files if k.startswith("f_")}
        features = {k[2:]: z[k] for k in z.files if k.startswith("x_")}
        return Panel(pd.DatetimeIndex(z["dates"]), list(z["entities"]), fields, features,
                     z["mask"], z["label"])
    p = build_panel()
    np.savez(path, dates=p.dates.values, entities=np.array(p.entities), mask=p.mask,
             label=p.label, **{f"f_{k}": v for k, v in p.fields.items()},
             **{f"x_{k}": v for k, v in p.features.items()})
    return p
