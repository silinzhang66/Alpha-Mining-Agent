"""Table 2 case studies: what the model did after a diagnosis (exploratory).

Two diagnosed formulas from deepseek_C_s1, chosen in the mechanism analysis
(paper Section 6) as the clearest instances of the R1 depth conflict and the R2
objective mismatch. For every round in which the formula was diagnosed, list
the next-round candidates derived from it, with training metrics from the
mining record and validation metrics re-computed here (validation period
only; the test set stays locked).

  uv run python scripts/make_case_study.py
"""
import sys
from pathlib import Path

import pandas as pd

from alpha_mining.config import ROOT, VALID
from alpha_mining.context import evaluator, thresholds

sys.path.insert(0, str(Path(__file__).parent))
from analyze_mechanism import applies, derived, diagnose, load_candidates, markdown, select_shown, tree  # noqa: E402

OUT = ROOT / "results" / "mechanism"
RUN = "deepseek_C_s1"
CASES = {
    "R1": "ts_decay_linear(ts_mean((high - max(open, close)) / (high - low), 20), 5)",
    "R2": "ts_decay_linear(ts_mean(ts_rank(volume, 10) - ts_rank(abs(returns), 10), 10), 5)",
}


def regime_text(regime_ic: dict, sign: float) -> str:
    names = {"up": "up", "range": "range", "down_pressure": "down"}
    return ", ".join(f"{names[k]} {sign * ic:+.4f} (t={sign * t:.1f})" for k, (ic, t) in regime_ic.items())


def main() -> None:
    th, ev = thresholds(), evaluator()
    cands = load_candidates(RUN)
    rows, lines = [], [f"# Case study ({RUN}, exploratory)", ""]
    for rule, formula in CASES.items():
        parent = next(c for c in cands if c.expr == formula)
        pt = tree(formula)
        sign = 1.0 if parent.result.ic >= 0 else -1.0
        pv = ev.run(formula, *VALID)
        diagnosed, text = [], ""
        for rnd in range(1, 21):
            shown = select_shown(cands, rnd)
            hit = [d for d in diagnose(shown, th) if d.formula == formula and d.rule == rule]
            if not hit:
                continue
            diagnosed.append(rnd)
            text = text or hit[0].text
            for c in cands:
                if c.round != rnd or not derived(pt, c.tree):
                    continue
                row = {"case": rule, "round": rnd, "formula": c.expr, "follows": applies(rule, pt, c.tree),
                       "status": c.status, "message": c.message, "depth": c.tree.depth}
                if c.result is not None:
                    v = ev.run(c.expr, *VALID)
                    row |= {"train_absicir": abs(c.result.icir), "train_turnover": c.result.turnover,
                            "fitness": c.result.fitness, "valid_absicir": abs(v.icir),
                            "valid_turnover": v.turnover, "valid_same_sign": (v.ic >= 0) == (c.result.ic >= 0)}
                rows.append(row)
        sub = pd.DataFrame([r for r in rows if r["case"] == rule])
        lines += [f"## {rule}: `{formula}`", "",
                  f"- First proposed in round {parent.round}; depth {pt.depth}; diagnosed in {len(diagnosed)} rounds"
                  f" (rounds {diagnosed[0]}–{diagnosed[-1]}), always in the shown list.",
                  f"- Training: |ICIR| {abs(parent.result.icir):.4f}, turnover {parent.result.turnover:.3f}, "
                  f"fitness {parent.fitness:.4f}; regime IC (sign-aligned) {regime_text(parent.result.regime_ic, sign)}.",
                  f"- Validation: |ICIR| {abs(pv.icir):.4f}, IC sign {'kept' if (pv.ic >= 0) == (parent.result.ic >= 0) else 'flipped'}, "
                  f"turnover {pv.turnover:.3f}; regime IC (aligned to the training sign) {regime_text(pv.regime_ic, sign)}.",
                  f"- Diagnosis text: {text}", "",
                  f"{len(sub)} derived next-round candidates, {int(sub.follows.sum())} modified as advised; "
                  f"{int((sub.status == 'invalid').sum())} invalid ({int((sub.depth > 6).sum())} too deep), {int((sub.status == 'duplicate').sum())} duplicates; "
                  f"{int((sub.get('fitness', pd.Series(dtype=float)) > parent.fitness).sum())} valid candidates with higher fitness than the parent.", "",
                  markdown(sub.drop(columns=["case", "message"]).round(4), index=False), ""]
    pd.DataFrame(rows).to_csv(OUT / "case_study.csv", index=False)
    (OUT / "case_study.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
