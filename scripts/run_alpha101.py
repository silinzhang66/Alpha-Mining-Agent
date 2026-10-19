"""Alpha101 baseline on train and validation; writes the relative entry gates.

The gates (median |IC| and |ICIR| of the eligible Alpha101 factors on the
training period) define both the training-pool quality gate and conditions 1-2
of a qualified factor.
"""
import json

import numpy as np
import pandas as pd

from alpha_mining.baselines.alpha101 import Alpha101, eligible_formulas
from alpha_mining.config import ROOT, TRAIN, VALID
from alpha_mining.context import evaluator
from alpha_mining.evaluation.preprocess import preprocess

ev = evaluator()
engine = Alpha101(ev.panel.features, ev.panel.mask)
rows = []
for _, a in eligible_formulas().iterrows():
    f = preprocess(engine.compute(a["formula"]), ev.neutralizer)
    for name, (start, end) in {"train": TRAIN, "valid": VALID}.items():
        r = ev.metrics(f, start, end)
        rows.append({"alpha": a["alpha"], "period": name, "ic": r.ic, "icir": r.icir, "ic_t": r.ic_t,
                     "turnover": r.turnover, "autocorr": r.autocorr, **r.backtest})
df = pd.DataFrame(rows)
out = ROOT / "results"
out.mkdir(exist_ok=True)
df.to_csv(out / "alpha101_train_valid.csv", index=False)
train = df[df["period"] == "train"]
gates = {"train_abs_ic_median": float(np.nanmedian(train["ic"].abs())),
         "train_abs_icir_median": float(np.nanmedian(train["icir"].abs())),
         "n_alphas": int(train["alpha"].nunique())}
(out / "alpha101_gates.json").write_text(json.dumps(gates, indent=2))
print(json.dumps(gates, indent=2))
print(df.groupby("period")[["ic", "icir", "turnover", "sharpe_net"]].agg(lambda s: s.abs().median()).round(4))
