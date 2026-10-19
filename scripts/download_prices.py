"""Download (or refresh) price history for every entity in the universe."""
import argparse
import logging

from alpha_mining.data.download import download_all

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

p = argparse.ArgumentParser()
p.add_argument("--refresh", action="store_true")
args = p.parse_args()

report, cov = download_all(refresh=args.refresh)
print(report["status"].value_counts().to_string())
print(report[~report["status"].isin(["ok", "missing"])].to_string())
print(cov["coverage"].describe().to_string())
print(cov["coverage"].resample("YE").agg(["mean", "min"]).to_string())
