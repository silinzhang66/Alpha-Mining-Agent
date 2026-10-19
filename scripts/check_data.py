"""Data-pipeline correctness checks on the formal panel (pre-freeze check).

1. Adjustment: known split days show no price or volume jump; panel-wide list
   of extreme daily moves among index members, for manual review.
2. Membership: on 10 random dates, the panel's members equal the constituent
   snapshot in effect that day, and every member missing from the mask has a
   recorded reason.
3. Coverage: daily share of index members present in the mask.
"""
import numpy as np
import pandas as pd

from alpha_mining.config import DATA_RAW, ROOT, TRAIN, VALID
from alpha_mining.data.panel import MIN_HISTORY, load_panel
from alpha_mining.data.universe import resolve

OUT = ROOT / "results" / "checks"
SPLITS = [("AAPL", "2020-08-31", 4), ("TSLA", "2020-08-31", 5), ("TSLA", "2022-08-25", 3),
          ("NVDA", "2021-07-20", 4), ("GOOGL", "2022-07-18", 20), ("AMZN", "2022-06-06", 20)]
EXTREME = 0.4

panel = load_panel()
OUT.mkdir(parents=True, exist_ok=True)
ents = {e: j for j, e in enumerate(panel.entities)}
in_sample = (panel.dates >= TRAIN[0]) & (panel.dates <= VALID[1])
ret = panel.features["returns"]
vol = panel.features["volume"]

# 1. Adjustment ------------------------------------------------------------
print("== split days: adjusted return and volume vs trailing 20-day median ==")
rows = []
for sym, day, factor in SPLITS:
    j, i = ents[sym], panel.dates.get_loc(pd.Timestamp(day))
    med = np.nanmedian(vol[i - 20:i, j])
    rows.append({"entity": sym, "date": day, "split": f"{factor}:1",
                 "ret_prev": ret[i - 1, j], "ret_split_day": ret[i, j], "ret_next": ret[i + 1, j],
                 "volume_ratio": vol[i, j] / med})
splits = pd.DataFrame(rows)
splits["pass"] = (splits[["ret_prev", "ret_split_day", "ret_next"]].abs().max(axis=1) < 0.15) \
    & splits["volume_ratio"].between(1 / 3, 3)
print(splits.round(4).to_string(index=False))

ti, tj = np.nonzero(in_sample[:, None] & panel.mask & (np.abs(ret) > EXTREME))
extreme = pd.DataFrame({"entity": [panel.entities[j] for j in tj], "date": panel.dates[ti].date,
                        "return": ret[ti, tj],
                        "volume_ratio": [vol[i, j] / np.nanmedian(vol[max(i - 20, 0):i, j]) for i, j in zip(ti, tj)]})
extreme = extreme.sort_values("return")
extreme.to_csv(OUT / "extreme_moves.csv", index=False)
print(f"\n== daily |return| > {EXTREME:.0%} among members, {TRAIN[0].year}-{VALID[1].year}: {len(extreme)} ==")
print(extreme.round(3).to_string(index=False))

# 2. Membership ------------------------------------------------------------
hist = pd.read_csv(DATA_RAW / "sp500_hist.csv", parse_dates=["date"]).sort_values("date")
rng = np.random.default_rng(2026)
sample = np.sort(rng.choice(panel.dates[in_sample], 10, replace=False))
history = np.cumsum(np.isfinite(panel.fields["adj_close"]) & np.isfinite(panel.fields["adj_open"]), axis=0)
summary, missing = [], []
print("\n== membership on 10 random dates ==")
for d in pd.DatetimeIndex(sample):
    snap = sorted(set(hist.loc[hist["date"] <= d, "tickers"].iloc[-1].split(",")))
    i = panel.dates.get_loc(d)
    n_missing = len(missing)
    in_mask = {panel.entities[j] for j in np.flatnonzero(panel.mask[i])}
    expected = set()
    for t in snap:
        e = resolve(t, d)
        if e in in_mask:
            expected.add(e)
            continue
        if e is None:
            why = "no usable source (ticker_map)"
        elif e not in ents:
            why = "feed failed validation"
        elif not np.isfinite(panel.fields["adj_close"][i, ents[e]]):
            why = "no bar that day"
        elif history[i, ents[e]] < MIN_HISTORY:
            why = f"fewer than {MIN_HISTORY} bars"
        else:
            why = "UNEXPLAINED"
        missing.append({"date": d.date(), "ticker": t, "entity": e, "reason": why})
    extra = in_mask - {resolve(t, d) for t in snap}
    summary.append({"date": d.date(), "index_tickers": len(snap), "in_mask": len(in_mask),
                    "missing": len(missing) - n_missing,
                    "not_in_index": len(extra)})
summary = pd.DataFrame(summary)
missing = pd.DataFrame(missing)
print(summary.to_string(index=False))
print(missing.to_string(index=False))
missing.to_csv(OUT / "membership_sample_missing.csv", index=False)

# 3. Coverage --------------------------------------------------------------
cov = pd.read_csv(ROOT / "data" / "daily_coverage.csv", parse_dates=["date"]).set_index("date")["coverage"]
cov = cov[(cov.index >= TRAIN[0]) & (cov.index <= VALID[1])]
print(f"\n== coverage {TRAIN[0].date()} to {VALID[1].date()}: mean {cov.mean():.2%}, min {cov.min():.2%} "
      f"on {cov.idxmin().date()} ==")
print(cov.groupby(cov.index.year).agg(["mean", "min"]).map(lambda x: f"{x:.1%}").T.to_string())

ok = splits["pass"].all() and (missing["reason"] != "UNEXPLAINED").all() and (summary["not_in_index"] == 0).all()
print("\nPASS" if ok else "\nCHECK FAILURES ABOVE")
