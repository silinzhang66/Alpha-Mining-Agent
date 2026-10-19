"""Extract the 101 formulas from the arXiv text of Kakushadze (2016), 1601.00991.

Writes data/alpha101_formulas.csv with one row per alpha and an `eligible` flag:
an alpha is eligible for the baseline if it needs no industry neutralization
(IndNeutralize / IndClass) and no market cap -- the only inputs we lack.
"""
import re

import pandas as pd

from alpha_mining.config import DATA_RAW, ROOT

text = (DATA_RAW / "alpha101" / "alpha101.txt").read_text()
body = text[text.find("Alpha#1:"):text.find("A.1. Functions")]
body = re.sub(r"\n\s*\d+\s*\n", "\n", body)  # page numbers
rows = []
for m in re.finditer(r"Alpha#(\d+)\s*:(.*?)(?=Alpha#\d+\s*:|\Z)", body, re.S):
    formula = " ".join(m.group(2).split())
    low = formula.lower()
    reason = "industry neutralization" if "indneutralize" in low or "indclass" in low else (
        "market cap" if re.search(r"\bcap\b", low) else "")
    rows.append({"alpha": int(m.group(1)), "formula": formula,
                 "eligible": not reason, "excluded_because": reason})
df = pd.DataFrame(rows)
assert list(df["alpha"]) == list(range(1, 102)), "missing or duplicated alphas"
df.to_csv(ROOT / "data" / "alpha101_formulas.csv", index=False)
print(df["eligible"].sum(), "eligible;", df.loc[~df["eligible"], "alpha"].tolist(), "excluded")
