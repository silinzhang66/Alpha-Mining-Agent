"""Prompts for Groups A (blind symbolic search), B (price-volume anomaly priors)
and C (priors + diagnostic feedback).

All groups share the DSL definition, the metric definitions, the feedback
numbers and the output budget. The DSL is described in purely mathematical
terms so that Group A receives no financial vocabulary beyond the variable
names themselves. Group B/C add the research framing, the five theory
directions and a <= 50-word hypothesis per formula; Group C additionally gets
the Diagnostic Translator's repair instructions.
"""
import json
import re

from alpha_mining.dsl import spec

DSL_SPEC = f"""\
EXPRESSION LANGUAGE
Write each expression as a single line using only the items below.

Variables (each is a panel indexed by day t and item i):
  open, high, low, close, volume
  returns       = close / ts_delay(close, 1) - 1
  ret_overnight = open / ts_delay(close, 1) - 1
  ret_intraday  = close / open - 1
  dollar_vol    = close * volume
  typical       = (high + low + close) / 3

Cross-sectional operators (computed across items i on each day t):
  rank(x)    percentile rank in (0, 1]
  zscore(x)  (x - mean) / std
  scale(x)   x / sum(|x|)

Time-series operators (computed over the past d days of each item, today included):
  ts_mean(x, d), ts_std(x, d), ts_max(x, d), ts_min(x, d),
  ts_rank(x, d)          percentile rank of today's value within the window
  ts_decay_linear(x, d)  weighted mean with weights d, d-1, ..., 1 (today largest)
  ts_corr(x, y, d), ts_cov(x, y, d)
  with d in {list(spec.WINDOWS)}
  ts_delay(x, d)  value d days ago;  ts_delta(x, d) = x - ts_delay(x, d)
  with d in {list(spec.LAG_WINDOWS)}

Arithmetic: +, -, *, / (division is protected against zero), unary minus, numeric constants
Elementwise: abs(x), log(x) (= log(|x| + eps)), sign(x),
  max(x, y), min(x, y)  (y may be a constant, e.g. max(returns, 0))

Limits: nesting depth <= {spec.MAX_DEPTH}, at most {spec.MAX_NODES} nodes (variables, constants and
operators; window arguments are not counted). Any other name, syntax or window is rejected."""

METRICS = """\
SCORING (computed on a fixed training sample; you never see the data)
Each expression is cross-sectionally standardized every day. Then:
  IC        = mean over days of the rank correlation between the expression and the
              target value 5 days ahead
  ICIR      = mean(IC) / std(IC) over days
  Turnover  = weekly one-way turnover of the top-20% and bottom-20% groups
  Fitness   = |ICIR| - {lam} * max(0, Turnover - {tau})        <- to be maximized
  Max_Pool_Corr = highest absolute correlation with the expressions already accepted
              into the pool (accepted expressions need low correlation with each other)
The sign does not matter: a strongly negative IC is as good as a strongly positive one."""

ROLE_BLIND = """\
You are searching a space of symbolic expressions. Your goal is to propose new
expressions that achieve a higher Fitness score. Use the scores of previous
expressions to guide the search."""

ROLE_RESEARCH = """\
You are a quantitative equity researcher mining daily price-volume alpha factors
for U.S. large-cap stocks (S&P 500 members). Each expression is a factor value per
stock per day; the target is the stock's return over the next 5 trading days.
Your goal is to propose new factors that achieve a higher Fitness score."""

THEORY = """\
THEORY DIRECTIONS (price-volume anomaly priors)
Ground each factor in one of these directions:
 1. Volume-price divergence: price moves that volume does not confirm (or volume
    surges without price moves) tend to reverse.
 2. Overnight vs. intraday returns (Lou, Polk & Skouras, 2019, "A tug of war"):
    different investor clienteles trade overnight and intraday, so the two
    return components carry different persistence and reversal patterns.
 3. Illiquidity (Amihud proxy): price impact per dollar traded, |returns| /
    dollar_vol, and shocks to it relate to future returns.
 4. Shadows and intraday extremes: upper/lower shadows (high vs. max(open, close),
    low vs. min(open, close)) and where the close sits in the day's range reflect
    intraday buying and selling pressure.
 5. Volatility asymmetry: downside vs. upside volatility and return skewness
    relate to future returns."""

FORMAT_BLIND = """\
OUTPUT
Return exactly {k} new expressions as a JSON array inside a ```json code block:
```json
[{{"formula": "..."}}, ...]
```
No other text."""

FORMAT_RESEARCH = """\
OUTPUT
Return exactly {k} new factors as a JSON array inside a ```json code block. For each
factor first state a financial hypothesis (at most 50 words, naming the theory
direction it uses), then the formula:
```json
[{{"hypothesis": "...", "formula": "..."}}, ...]
```
No other text."""


def _fmt(v) -> str:
    return f"{v:+.4f}" if isinstance(v, float) else str(v)


def feedback_block(shown: list[dict], failures: list[dict], round_: int, rounds: int) -> str:
    lines = [f"ROUND {round_} of {rounds}."]
    if shown:
        lines.append("Best expressions so far (top 2 overall + best of the previous round):")
        for i, row in enumerate(shown, 1):
            nums = ", ".join(f"{k}={_fmt(v)}" for k, v in row.items() if k != "Formula")
            lines.append(f"  {i}. {row['Formula']}\n     {nums}")
    else:
        lines.append("No expressions have been scored yet.")
    if failures:
        lines.append("Problems in the previous round (each still used one evaluation):")
        for f in failures:
            lines.append(f"  - {f['formula']}  ->  {f['message']}")
    return "\n".join(lines)


def diagnosis_block(diagnoses: list) -> str:
    if not diagnoses:
        return "DIAGNOSIS\nNo rule was triggered for the expressions shown above."
    lines = ["DIAGNOSIS (backtest medical chart for the expressions shown above)"]
    for d in diagnoses:
        lines.append(f"* {d.formula}\n  {d.text}")
    return "\n".join(lines)


def stats_block(diagnoses: list) -> str:
    """Review round 1, experiment E3 (group Cs): the statistics quoted by the
    diagnoses, without rule names, interpretation or repair instructions."""
    if not diagnoses:
        return "ADDITIONAL STATISTICS\nNone for the expressions shown above."
    lines = ["ADDITIONAL STATISTICS (training sample) for the expressions shown above"]
    for d in diagnoses:
        lines.append(f"* {d.formula}\n  {d.evidence}")
    return "\n".join(lines)


# Groups of review round 1: Cd = C with depth-compatible R1 advice (E1),
# Cs = B plus the diagnoses' statistics without instructions (E3).
DIAGNOSED = ("C", "Cd", "Cs")


def build_messages(group: str, k: int, lam: float, tau: float, feedback: str,
                   diagnosis: str | None = None) -> list[dict]:
    if group not in ("A", "B", *DIAGNOSED):
        raise ValueError(group)
    blind = group == "A"
    system = "\n\n".join([
        ROLE_BLIND if blind else ROLE_RESEARCH,
        DSL_SPEC,
        METRICS.format(lam=lam, tau=tau),
        *([] if blind else [THEORY]),
        (FORMAT_BLIND if blind else FORMAT_RESEARCH).format(k=k),
    ])
    user = feedback
    if group in DIAGNOSED:
        user += "\n\n" + (diagnosis or (stats_block([]) if group == "Cs" else diagnosis_block([])))
    user += f"\n\nPropose {k} new expressions now."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_reply(text: str) -> list[dict]:
    """Extract the JSON array of {formula[, hypothesis]} objects from a reply."""
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S) or re.search(r"(\[.*\])", text, re.S)
    if not m:
        raise ValueError("no JSON array found")
    items = json.loads(m.group(1))
    out = []
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("formula"), str):
            out.append({"formula": it["formula"].strip(),
                        "hypothesis": str(it.get("hypothesis", "")).strip()})
        elif isinstance(it, str):
            out.append({"formula": it.strip(), "hypothesis": ""})
    if not out:
        raise ValueError("JSON array holds no formulas")
    return out
