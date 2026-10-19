import numpy as np
import pandas as pd

from alpha_mining.data.download import validate


def feed(dates, dollar_vol, flat=False):
    close = np.ones(len(dates)) if flat else np.linspace(10, 20, len(dates))
    return pd.DataFrame({"adj_close": close, "dollar_vol": dollar_vol}, index=dates)


def test_partial_feeds_still_get_quality_checks():
    days = pd.bdate_range("2016-01-01", "2018-12-31")
    member = pd.Series(True, index=days)
    late = days[days >= "2017-06-01"]
    assert validate("x", feed(late, 1e8), member)["status"] == "late_start"
    # Junk data (tiny volume, flat prices) must not pass just because it starts late.
    assert validate("x", feed(late, 2e4), member)["status"] == "suspect_feed"
    assert validate("x", feed(late, 1e8, flat=True), member)["status"] == "suspect_feed"
    # A reused ticker whose data starts after membership ended has no overlap at all.
    member[days >= "2017-01-01"] = False
    assert validate("x", feed(late, 1e8), member)["status"] == "suspect_feed"


def test_repair_adjustment_only_fixes_broken_factor():
    from alpha_mining.data.download import repair_adjustment
    days = pd.bdate_range("2020-01-01", periods=6)
    raw = pd.Series([100, 101, 102, 104, 105, 106.0], index=days)
    adj = raw * 0.9
    adj.iloc[:3] *= 0.6  # broken factor: +70% jump on day 3 while raw moves +2%
    h = pd.DataFrame({"close_raw_split": raw, **{c: adj for c in
                      ("adj_open", "adj_high", "adj_low", "adj_close")}})
    fixed, fixes = repair_adjustment(h)
    assert len(fixes) == 1 and fixes[0][0] == days[3]
    assert np.allclose(fixed["adj_close"].pct_change().iloc[1:], raw.pct_change().iloc[1:])
    # A spin-off (raw drops, adjusted smooth) is a correct adjustment and is left alone.
    spin = pd.DataFrame({"close_raw_split": raw.where(raw.index < days[3], raw * 0.6),
                         **{c: raw * 0.9 for c in ("adj_open", "adj_high", "adj_low", "adj_close")}})
    assert repair_adjustment(spin)[1] == []
