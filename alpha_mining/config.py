"""Global constants: paths, period splits, and the test-set lock."""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_CACHE = ROOT / "data" / "cache"
PRICE_CACHE = DATA_CACHE / "prices"

HISTORY_START = pd.Timestamp("2014-06-01")  # warm-up so 2016 constituents have 60d history
TRAIN = (pd.Timestamp("2016-01-01"), pd.Timestamp("2021-12-31"))
VALID = (pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31"))
TEST1 = (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31"))
TEST2 = (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-09-30"))

TEST_START = TEST1[0]


class TestLockedError(RuntimeError):
    pass


def test_unlocked() -> bool:
    return os.environ.get("UNLOCK_TEST") == "1"


def data_end() -> pd.Timestamp:
    """Last date any loader may touch."""
    return TEST2[1] if test_unlocked() else TEST_START - pd.Timedelta(days=1)


def assert_allowed(end: pd.Timestamp) -> None:
    if pd.Timestamp(end) >= TEST_START and not test_unlocked():
        raise TestLockedError(
            f"Access to data on/after {TEST_START.date()} requires UNLOCK_TEST=1"
        )
