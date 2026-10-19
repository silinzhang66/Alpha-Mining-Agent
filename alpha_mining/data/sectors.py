"""SIC codes from SEC EDGAR and Fama-French 12 industries, for neutralization.

CIK lookup per entity:
  * yfinance entities (still trading): SEC's current ticker -> CIK table.
  * Tiingo entities (mostly delisted; their tickers may now belong to someone
    else): company name from Tiingo's metadata, then EDGAR's entity search.
The SIC code then comes from the EDGAR submissions record of that CIK. Every
lookup records its method and the EDGAR name so the result can be reviewed;
manual corrections go in data/sic_overrides.csv.
"""
import os
import re
import time

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from alpha_mining.config import ROOT

SIC_FILE = ROOT / "data" / "sic_codes.csv"
OVERRIDES_FILE = ROOT / "data" / "sic_overrides.csv"
SEC_PAUSE = 0.15  # SEC fair-access limit is 10 requests/second


def _env(name: str) -> str:
    load_dotenv(ROOT / ".env")
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"{name} is not set in .env")
    return v


def _sec_get(url: str, retries: int = 6, **params) -> requests.Response:
    for attempt in range(retries):
        time.sleep(SEC_PAUSE)
        r = requests.get(url, params=params, headers={"User-Agent": _env("SEC_USER_AGENT")}, timeout=30)
        if r.status_code not in (429, 500, 502, 503, 504):
            break
        time.sleep(2 ** attempt)
    r.raise_for_status()
    return r


def current_tickers() -> dict[str, int]:
    data = _sec_get("https://www.sec.gov/files/company_tickers.json").json()
    return {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in data.values()}


def tiingo_name(symbol: str, retries: int = 8) -> str | None:
    for _ in range(retries):
        r = requests.get(f"https://api.tiingo.com/tiingo/daily/{symbol}",
                         params={"token": _env("TIINGO_API_KEY")}, timeout=30)
        if r.status_code != 429:
            break
        time.sleep(600)  # free-tier hourly cap; wait for the window to roll
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json().get("name")


def search_entity(text: str) -> tuple[int, str] | None:
    """Top EDGAR entity-search hit for a company name: (CIK, EDGAR entity label)."""
    text = re.sub(r"\s+-\s+Class\s+\w+$", "", text)  # Tiingo appends share class
    for attempt in range(5):
        body = _sec_get("https://efts.sec.gov/LATEST/search-index", keysTyped=text).json()
        if "hits" in body:
            break
        time.sleep(2 ** attempt)  # backend errors come back as HTTP 200 without hits
    else:
        raise RuntimeError(f"EDGAR entity search failed for {text!r}: {body.get('errorMessage')}")
    hits = body["hits"]["hits"]
    return (int(hits[0]["_id"]), hits[0]["_source"]["entity"]) if hits else None


def submissions(cik: int) -> dict:
    return _sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()


def lookup(entities: list[str]) -> pd.DataFrame:
    """CIK and SIC for each entity; results are cached in SIC_FILE."""
    done = pd.read_csv(SIC_FILE) if SIC_FILE.exists() else pd.DataFrame(columns=["entity"])
    todo = [e for e in entities if e not in set(done["entity"])]
    cur = current_tickers() if todo else {}
    rows = []
    for e in todo:
        rec = {"entity": e, "cik": None, "method": None, "query": None}
        if e.startswith("ti_"):
            name = tiingo_name(e[3:])
            rec["query"] = name
            hit = search_entity(name) if name else None
            if hit:
                rec["cik"], rec["method"] = hit[0], "tiingo_name"
        elif e in cur:
            rec["cik"], rec["method"] = cur[e], "sec_ticker"
        if rec["cik"]:
            s = submissions(rec["cik"])
            rec.update(edgar_name=s.get("name"), sic=s.get("sic") or None,
                       sic_desc=s.get("sicDescription"), edgar_tickers=",".join(s.get("tickers", [])))
        rows.append(rec)
        # Save as we go: Tiingo rate limits make a full pass take hours.
        out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        out.to_csv(SIC_FILE, index=False)
    return pd.concat([done, pd.DataFrame(rows)], ignore_index=True)


def sic_codes() -> pd.Series:
    """entity -> 4-digit SIC, with manual overrides applied."""
    s = pd.read_csv(SIC_FILE).set_index("entity")["sic"]
    if OVERRIDES_FILE.exists():
        o = pd.read_csv(OVERRIDES_FILE, comment="#").set_index("entity")["sic"]
        s = o.combine_first(s)
    return s.dropna().astype(int)


# Fama-French 12-industry classification (Ken French data library, Siccodes12).
FF12_NAMES = ("NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq",
              "Telcm", "Utils", "Shops", "Hlth", "Money", "Other")
_FF12_RANGES = {
    0: [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)],
    1: [(2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716),
        (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999)],
    2: [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569), (3580, 3629),
        (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799),
        (3830, 3839), (3860, 3899)],
    3: [(1200, 1399), (2900, 2999)],
    4: [(2800, 2829), (2840, 2899)],
    5: [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)],
    6: [(4800, 4899)],
    7: [(4900, 4949)],
    8: [(5000, 5999), (7200, 7299), (7600, 7699)],
    9: [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)],
    10: [(6000, 6999)],
}


def ff12(sic: int) -> int:
    """Fama-French 12 industry index (0-11) for a 4-digit SIC code."""
    for k, ranges in _FF12_RANGES.items():
        if any(lo <= sic <= hi for lo, hi in ranges):
            return k
    return 11


def industry_codes(entities: list[str]) -> np.ndarray:
    """FF12 index per entity, -1 where no SIC code is known."""
    sic = sic_codes()
    return np.array([ff12(int(sic[e])) if e in sic.index else -1 for e in entities])
