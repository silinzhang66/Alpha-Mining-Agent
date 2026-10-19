"""Failure-mechanism analysis of the LLM groups (exploratory, not pre-registered).

Uses only mining logs (training-period metrics), the LLM request logs and the
final-pool files (training + validation). Test-period numbers are read from the
already-saved results/summary/per_run.csv for the group-difference tests; no
evaluation is re-run.

  triggers   R1-R3 triggers per run, replayed from the mining records with the
             frozen thresholds. For Group C this is checked against the
             diagnosis text actually sent in the prompt; for A/B it is the
             counterfactual (would have fired) used as a placebo.
  response   next-round candidates that derive from a diagnosed formula, whether
             they apply the suggested repair, and the metric change
  rounds     per-round turnover, depth, invalid share by configuration
  invalid    invalid-formula reasons by configuration
  sign       training -> validation IC sign retention of training-pool factors
  tests      per-seed C-A and B-A differences: t test and exact permutation

  uv run python scripts/analyze_mechanism.py
"""
import ast
import itertools
import json
import re
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats

from alpha_mining.config import ROOT
from alpha_mining.context import thresholds
from alpha_mining.dsl import spec
from alpha_mining.dsl.parse import DSLError, _convert, parse
from alpha_mining.mining.diagnostics import diagnose

RES = ROOT / "results"
OUT = RES / "mechanism"
RUN_RE = re.compile(r"^(deepseek|qwen)_([ABC])(_think)?_s(\d)$")
SMOOTH = {"ts_decay_linear", "ts_mean"}
VOLUME = {"volume", "dollar_vol"}


def llm_runs() -> list[str]:
    return sorted(p.name[:-6] for p in (RES / "runs").glob("*.jsonl") if RUN_RE.match(p.name[:-6]))


def config_of(run: str) -> str:
    return re.sub(r"_s\d+$", "", run)


def depth(expr: str) -> float:
    try:
        return parse(expr).depth
    except DSLError:
        return np.nan


def load_candidates(run: str) -> list[SimpleNamespace]:
    out = []
    for line in open(RES / "runs" / f"{run}.jsonl"):
        d = json.loads(line)
        r = d.get("result")
        res = pool = None
        if r is not None:
            res = SimpleNamespace(ic=r["ic"], icir=r["icir"], turnover=r["turnover"],
                                  autocorr=r["autocorr"], fitness=r["fitness"],
                                  regime_ic={k: tuple(v) for k, v in r["regime_ic"].items()})
            pool = SimpleNamespace(**d["pool"])
        out.append(SimpleNamespace(round=d["round"], expr=d["expr"], status=d["status"],
                                   message=d["message"], result=res, pool=pool, meta=d["meta"],
                                   fitness=res.fitness if res else -np.inf, tree=tree(d["expr"])))
    return out


def select_shown(cands, rnd):
    """Same rule as alpha_mining.llm.loop.select_shown, on the state before round `rnd`."""
    scored = sorted((c for c in cands if c.result is not None and c.round < rnd),
                    key=lambda c: c.fitness, reverse=True)
    top = scored[:2]
    prev = [c for c in scored if c.round == rnd - 1 and all(c is not t for t in top)]
    return top + prev[:1]


def logged_diagnoses(run: str) -> dict[int, list[str]]:
    """Rule tags in the diagnosis block of the accepted prompt per round (Group C)."""
    out = {}
    path = ROOT / "logs" / "llm" / f"{run}.jsonl"
    for line in open(path):
        d = json.loads(line)
        user = d["request"]["messages"][-1]["content"]
        out[d["round"]] = re.findall(r"\[(R\d) ", user.split("DIAGNOSIS", 1)[-1]) if "DIAGNOSIS" in user else []
    return out


def tree(expr: str):
    """Parse without the depth / size limits, so over-deep repairs still compare."""
    try:
        return _convert(ast.parse(expr.strip(), mode="eval").body)
    except (DSLError, SyntaxError, ValueError, AttributeError):
        return None


def strip_eps(n):
    """Drop `x + c` / `x - c` offsets (division guards such as dollar_vol + 1)."""
    if n.op in ("add", "sub") and len(n.args) == 2:
        a, b = n.args
        if b.op == "const":
            return strip_eps(a)
        if a.op == "const" and n.op == "add":
            return strip_eps(b)
    return replace(n, args=tuple(strip_eps(a) for a in n.args))


def skeleton(n) -> str:
    """Canonical text with windows dropped and constants masked."""
    if n.op == "const":
        return "c"
    if not n.args:
        return n.op
    args = sorted(skeleton(a) for a in n.args) if n.op in ("add", "mul", *spec.SYMMETRIC) \
        else [skeleton(a) for a in n.args]
    return f"{n.op}({','.join(args)})"


def subtrees(n) -> dict[str, int]:
    out = {skeleton(n): n.size}
    for a in n.args:
        out.update(subtrees(a))
    return out


def derived(parent, child) -> bool:
    """The child reuses a subtree covering at least half of the parent's nodes
    (windows and constants ignored), and is not the parent itself."""
    if parent is None or child is None or parent == child:
        return False
    parent, child = strip_eps(parent), strip_eps(child)
    sp = subtrees(parent)
    best = max((sp[k] for k in set(sp) & set(subtrees(child))), default=0)
    return best >= max(3, parent.size / 2)


def count_ops(n, ops) -> int:
    return (n.op in ops) + sum(count_ops(a, ops) for a in n.args)


def smooth_windows(n) -> int:
    return (n.window if n.op in SMOOTH else 0) + sum(smooth_windows(a) for a in n.args)


def applies(rule: str, parent, child) -> bool:
    """R1: more ts_std; R2: more smoothing operators or longer smoothing windows;
    R3: more volume / dollar_vol terms."""
    if rule == "R1":
        return count_ops(child, {"ts_std"}) > count_ops(parent, {"ts_std"})
    if rule == "R2":
        return (count_ops(child, SMOOTH) > count_ops(parent, SMOOTH)
                or smooth_windows(child) > smooth_windows(parent))
    return count_ops(child, VOLUME) > count_ops(parent, VOLUME)


def analyse_run(run: str, th) -> tuple[list[dict], list[dict], list[dict]]:
    cands = load_candidates(run)
    m = RUN_RE.match(run)
    model, group, think, seed = m.group(1), m.group(2), bool(m.group(3)), int(m.group(4))
    base = {"run": run, "config": config_of(run), "model": model, "group": group,
            "think": think, "seed": seed}
    rounds = max(c.round for c in cands)
    logged = logged_diagnoses(run) if group == "C" else {}

    trig, resp, per_round, prev = [], [], [], set()
    for rnd in range(1, rounds + 1):
        nxt = [c for c in cands if c.round == rnd]
        shown = select_shown(cands, rnd)
        diags = diagnose(shown, th)
        if group == "C" and sorted(d.rule for d in diags) != sorted(logged.get(rnd, [])):
            raise RuntimeError(f"{run} round {rnd}: replay {[d.rule for d in diags]} "
                               f"!= logged {logged.get(rnd)}")
        by_expr = {c.expr: c for c in shown}
        for d in diags:
            parent = by_expr[d.formula]
            pt = tree(parent.expr)
            kids = [c for c in nxt if derived(pt, c.tree)]
            fixed = [c for c in kids if applies(d.rule, pt, c.tree)]
            ok = [c for c in fixed if c.result is not None]
            pr = parent.result
            trig.append({**base, "round": rnd, "rule": d.rule, "formula": parent.expr})
            resp.append({
                **base, "round": rnd, "rule": d.rule, "formula": parent.expr,
                "repeat": (parent.expr, d.rule) in prev, "parent_depth": pt.depth,
                "parent_turnover": pr.turnover, "parent_absicir": abs(pr.icir),
                "parent_fitness": pr.fitness, "parent_maxcorr": parent.pool.max_corr,
                "n_kids": len(kids), "n_fixed": len(fixed), "n_fixed_ok": len(ok),
                "n_fixed_invalid": sum(c.status == "invalid" for c in fixed),
                "d_turnover": np.mean([c.result.turnover - pr.turnover for c in ok]) if ok else np.nan,
                "d_absicir": np.mean([abs(c.result.icir) - abs(pr.icir) for c in ok]) if ok else np.nan,
                "d_fitness": np.mean([c.result.fitness - pr.fitness for c in ok]) if ok else np.nan,
                "d_maxcorr": np.mean([c.pool.max_corr - parent.pool.max_corr for c in ok]) if ok else np.nan,
                "fixed_better": sum(c.result.fitness > pr.fitness for c in ok),
                "fixed_lower_turnover": sum(c.result.turnover < th.r2_turnover for c in ok),
            })
        prev = {(d.formula, d.rule) for d in diags}
        ok = [c for c in nxt if c.result is not None]
        shown_trees = [tree(c.expr) for c in shown]
        hyp = " ".join(c.meta.get("hypothesis", "") for c in nxt).lower()
        per_round.append({
            **base, "round": rnd, "n": len(nxt),
            "invalid": sum(c.status == "invalid" for c in nxt) / len(nxt),
            "duplicate": sum(c.status == "duplicate" for c in nxt) / len(nxt),
            "turnover": np.mean([c.result.turnover for c in ok]) if ok else np.nan,
            "high_turnover": np.mean([c.result.turnover > th.r2_turnover for c in ok]) if ok else np.nan,
            "absicir": np.mean([abs(c.result.icir) for c in ok]) if ok else np.nan,
            "depth": np.mean([depth(c.expr) for c in ok]) if ok else np.nan,
            "smooth_ops": np.mean([count_ops(c.tree, SMOOTH) for c in nxt if c.tree]),
            "depth_all": np.mean([c.tree.depth for c in nxt if c.tree]),
            "derived_share": np.mean([any(derived(t, c.tree) for t in shown_trees) for c in nxt]),
            "n_diag": len(diags), "r1": any(d.rule == "R1" for d in diags),
            "r2": any(d.rule == "R2" for d in diags), "r3": any(d.rule == "R3" for d in diags),
            "mentions_turnover": int(bool(re.search(r"turnover|smooth", hyp))),
        })
    return trig, resp, per_round


def markdown(t: pd.DataFrame, index: bool = True) -> str:
    t = t.reset_index() if index else t
    lines = ["| " + " | ".join(map(str, t.columns)) + " |", "|" + " --- |" * len(t.columns)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in t.itertuples(index=False)]
    return "\n".join(lines)


def invalid_reason(msg: str) -> str:
    for key, pat in [("depth", "depth"), ("nodes", "nodes"), ("window", "window"),
                     ("unknown name", "unknown|not allowed|unsupported"),
                     ("syntax", "syntax|parse"), ("missing", "fewer formulas|unparseable")]:
        if re.search(pat, msg, re.I):
            return key
    return "other"


def sign_retention() -> pd.DataFrame:
    rows = []
    for p in sorted((RES / "final").glob("*.json")):
        d = json.load(open(p))
        fac = [f for f in d["factors"] if f.get("valid")]
        run = d["run"]
        rows.append({"run": run, "config": config_of(run), "n": len(fac),
                     "same_sign": np.mean([np.sign(f["train"]["ic"]) == np.sign(f["valid"]["ic"]) for f in fac]),
                     "train_absicir": np.mean([abs(f["train"]["icir"]) for f in fac]),
                     "valid_absicir": np.mean([abs(f["valid"]["icir"]) for f in fac])})
    return pd.DataFrame(rows)


def top10_turnover() -> pd.DataFrame:
    rows = []
    for p in sorted((RES / "final").glob("*.json")):
        d = json.load(open(p))
        rows.append({"run": d["run"],
                     "top10_train_turnover": np.mean([f["train"]["turnover"] for f in d["top10"]]),
                     "top10_valid_turnover": np.mean([f["valid"]["turnover"] for f in d["top10"]])})
    return pd.DataFrame(rows)


def group_tests(per_run: pd.DataFrame) -> pd.DataFrame:
    """Group differences over 5 seeds per configuration. The seed index does not
    pair LLM runs (sampling is not seeded), so the tests are unpaired: Welch t
    and the exact permutation test over all C(10, 5) splits."""
    metrics = ["P1", "invalid_share", "top10_train_turnover", "top10_valid_turnover",
               "valid_icir", "test1_icir", "test1_sr_net", "test1_turnover", "test2_icir"]
    rows = []
    for model in ("deepseek", "qwen"):
        for g, h in (("B", "A"), ("C", "A"), ("C", "B")):
            x_df = per_run[per_run.config == f"{model}_{g}"]
            y_df = per_run[per_run.config == f"{model}_{h}"]
            for m in metrics:
                x, y = x_df[m].to_numpy(float), y_df[m].to_numpy(float)
                obs = x.mean() - y.mean()
                pooled = np.concatenate([x, y])
                null = [pooled[list(i)].mean() - np.delete(pooled, list(i)).mean()
                        for i in itertools.combinations(range(len(pooled)), len(x))]
                rows.append({"model": model, "contrast": f"{g}-{h}", "metric": m,
                             "diff": obs, "t_welch": stats.ttest_ind(x, y, equal_var=False).statistic,
                             "p_welch": stats.ttest_ind(x, y, equal_var=False).pvalue,
                             "p_perm": np.mean(np.abs(null) >= abs(obs) - 1e-12)})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    th = thresholds()
    trig, resp, rounds = [], [], []
    for run in llm_runs():
        t, r, pr = analyse_run(run, th)
        trig += t
        resp += r
        rounds += pr
    trig, resp, rounds = pd.DataFrame(trig), pd.DataFrame(resp), pd.DataFrame(rounds)
    resp.to_csv(OUT / "response.csv", index=False)
    rounds.to_csv(OUT / "rounds.csv", index=False)

    lines = ["# Failure-mechanism analysis (exploratory, not pre-specified)", "",
             "Uses only training / validation data and the logs; the Group C triggers were checked round by round against the diagnosis text in the logged prompts.",
             "Group A / B triggers are a counterfactual replay (what a diagnosis would have flagged), used as a control.", ""]

    # 1. trigger counts
    tc = trig.groupby(["config", "rule"]).size().unstack(fill_value=0)
    nrun = trig.groupby("config").run.nunique()
    tc = tc.div(nrun, axis=0).round(1)
    tc["distinct_formulas"] = (trig.groupby(["config", "run"]).formula.nunique()
                               .groupby("config").mean().round(1))
    lines += ["## 1. Rule triggers (mean per run, 20 rounds)", "", markdown(tc), ""]

    # 2. response to R2 / R1 / R3
    agg = (resp.groupby(["config", "rule"])
           .agg(triggers=("run", "size"), kids_per=("n_kids", "mean"),
                fixed_per=("n_fixed", "mean"), any_fixed=("n_fixed", lambda s: (s > 0).mean()),
                fixed_invalid=("n_fixed_invalid", "sum"), fixed_total=("n_fixed", "sum"),
                d_turnover=("d_turnover", "mean"), d_absicir=("d_absicir", "mean"),
                d_fitness=("d_fitness", "mean"), d_maxcorr=("d_maxcorr", "mean"),
                fixed_better=("fixed_better", "sum"), fixed_ok=("n_fixed_ok", "sum"),
                fixed_below_tau=("fixed_lower_turnover", "sum"),
                repeat=("repeat", "mean"), parent_at_depth_limit=("parent_depth", lambda s: (s >= spec.MAX_DEPTH).mean())))
    agg["fixed_invalid_share"] = agg.fixed_invalid / agg.fixed_total.replace(0, np.nan)
    agg["fixed_better_share"] = agg.fixed_better / agg.fixed_ok.replace(0, np.nan)
    agg["below_r2_share"] = agg.fixed_below_tau / agg.fixed_ok.replace(0, np.nan)
    agg = agg.drop(columns=["fixed_invalid", "fixed_total", "fixed_better", "fixed_below_tau"])
    agg.to_csv(OUT / "response_summary.csv")
    lines += ["## 2. Next-round response after a trigger", "",
              "kids: next-round candidates that contain the body of the diagnosed formula; fixed: those modified as advised (R1 adds ts_std, "
              "R2 adds ts_decay_linear/ts_mean, R3 adds volume/dollar_vol); d_*: mean change of the valid fixed candidates relative to the parent; "
              "fixed_better_share: share of fixed candidates with higher fitness than the parent; below_r2_share: share whose turnover is below the R2 threshold; "
              "repeat: share of triggers with the same formula and rule as in the previous round; parent_at_depth_limit: share of diagnosed formulas already at the depth limit of 6. "
              "A candidate is derived from a formula if they share a subtree covering more than half of the formula's nodes (ignoring windows, constants and +c offsets).",
              "", markdown(agg.round(3)), ""]

    # 3. per-round trajectories
    traj = (rounds.assign(phase=pd.cut(rounds["round"], [0, 5, 10, 15, 20], labels=["1-5", "6-10", "11-15", "16-20"]))
            .groupby(["config", "phase"], observed=True)[["turnover", "high_turnover", "absicir", "depth", "invalid", "smooth_ops"]]
            .mean().round(3))
    traj.to_csv(OUT / "trajectory.csv")
    lines += ["## 3. Trajectories by round (mean per 5 rounds; depth of valid candidates)", "", markdown(traj), ""]

    # 4. local refinement vs. broad exploration, and what a diagnosis does to the next round
    expl = rounds.groupby("config")[["derived_share", "invalid", "depth_all"]].mean().round(3)
    lines += ["## 4. Local refinement or broad exploration: share of next-round candidates derived from the shown formulas", "", markdown(expl), ""]
    rounds["r12"] = rounds.r1 | rounds.r2
    cond = (rounds[rounds["round"] > 1]
            .groupby(["config", "r12"])[["invalid", "depth_all", "smooth_ops", "derived_share"]].mean()
            .unstack("r12"))
    diff = pd.concat({k: cond[k][True] - cond[k][False] for k in ["invalid", "depth_all", "smooth_ops", "derived_share"]}, axis=1)
    diff["rounds_with"] = rounds[rounds["round"] > 1].groupby("config").r12.sum()
    lines += ["## 5. Rounds with an R1/R2 trigger minus rounds without (next-round candidates)", "",
              "For Group C this is the effect of the actual diagnosis; for A/B it is the difference under the same trigger condition without any diagnosis text (counterfactual).", "",
              markdown(diff.round(3)), ""]

    # 4. invalid reasons
    inv = []
    for run in llm_runs():
        for c in load_candidates(run):
            if c.status == "invalid" or (c.status == "error"):
                inv.append({"config": config_of(run), "reason": invalid_reason(c.message)})
    inv = pd.DataFrame(inv)
    n_total = rounds.groupby("config").n.sum()
    it = inv.groupby(["config", "reason"]).size().unstack(fill_value=0).div(n_total, axis=0).round(3)
    lines += ["## 6. Reasons for invalid formulas (share of all candidates)", "", markdown(it), ""]

    # 5. sign retention
    sr = sign_retention()
    sr.to_csv(OUT / "sign_retention.csv", index=False)
    srs = sr.groupby("config")[["n", "same_sign", "train_absicir", "valid_absicir"]].mean().round(3)
    lines += ["## 8. Validation-period retention of the training IC sign, all valid candidates", "", markdown(srs), ""]

    # 6. group tests
    per_run = pd.read_csv(RES / "summary" / "per_run.csv").merge(top10_turnover(), on="run", how="left")
    gt = group_tests(per_run)
    gt.to_csv(OUT / "group_tests.csv", index=False)
    lines += ["## 9. Group contrasts (five seeds per group; Welch t-test and exact permutation test, unpaired)", "",
              markdown(gt.round(3).set_index(["model", "contrast", "metric"])), ""]

    (OUT / "mechanism.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
