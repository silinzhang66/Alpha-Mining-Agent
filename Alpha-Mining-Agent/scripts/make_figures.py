"""Paper figures 2 and 3 (SVG + 600-dpi TIFF in results/figures/, PNG preview).

The ACM template takes one image per figure number, vector (SVG) preferred or
TIFF, at most the 6.08 in text width, so each figure is a single 6.0 in image
with two rows of panels:

  Figure 2 (a)    best-so-far training fitness by round (1-20), one panel per
                  model; mean over seeds with a min-max band. GP(200) is shown
                  per 10 evaluations; GP(1000)'s final level as a reference line.
                  Degenerate candidates (Newey-West t undefined: IC exists on too
                  few days, e.g. 2 of 1511 in gp1000_s2) are left out of this
                  plot only; no metric or selection changes.
  Figure 2 (b)    |rho| (validation period) between each model's leading
                  Group C factors and Alpha101 / three classic factors.
  Figure 3 (a)    cumulative net value of the Top-10 composite over Test I + II
                  (needs results/test, i.e. after the one-time unlock).
  Figure 3 (b)    regime IC of the composite over Test I + II (same).

  fig2_convergence, fig3_net_value
                  panel (a) of Figures 2 and 3 alone, as used in the 8-page
                  paper (--paper writes only these two).

matplotlib is not part of the frozen environment; run it as an overlay so
uv.lock stays untouched:
  uv run --with matplotlib==3.10.0 python scripts/make_figures.py [--paper]

Figure 3 only reads the saved results/test/*.json; it never re-evaluates.
"""
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alpha_mining.config import ROOT, VALID

RES, OUT = ROOT / "results", ROOT / "results" / "figures"
MODELS = {"deepseek": "DeepSeek-V4.1-Flash", "qwen": "Qwen3.8-Flash"}

# Validated categorical slots (dataviz reference palette, light mode); colour
# follows the entity in every figure. GP baselines recede in grey, dashed.
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e6e5e0", "#fcfcfb"
STYLE = {
    "C": dict(color="#2a78d6", ls="-", label="Group C"),
    "B": dict(color="#eb6834", ls="-", label="Group B"),
    "A": dict(color="#1baf7a", ls="-", label="Group A"),
    "gp200": dict(color=INK2, ls="--", label="GP(200)"),
    "gp1000": dict(color=MUTED, ls=":", label="GP(1000)"),
    "alpha101": dict(color=MUTED, ls="-.", label="Alpha101"),
}
REGIMES = [("up", "Up"), ("range", "Range"), ("down_pressure", "Down pressure")]

plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "lines.linewidth": 2,
    "legend.frameon": False, "font.family": "Arial", "svg.fonttype": "path",
})
WIDTH = 6.0  # in; ACM text width is 6.08 in


def runs(pattern: str) -> list[str]:
    return sorted(p.stem for p in (RES / "runs").glob("*.jsonl")
                  if re.fullmatch(pattern, p.stem) and "_pilot" not in p.stem)


def best_so_far(run: str, per: int = 10) -> np.ndarray:
    """Best training fitness after each block of `per` candidates (a round)."""
    recs = [json.loads(l) for l in (RES / "runs" / f"{run}.jsonl").read_text().splitlines()]
    ok = lambda r: r.get("result") and np.isfinite(r["result"]["ic_t"] if r["result"]["ic_t"] is not None else np.nan)
    fit = np.array([r["result"]["fitness"] if ok(r) else -np.inf for r in recs])
    best = np.maximum.accumulate(fit)[per - 1::per]
    return np.where(np.isfinite(best), best, np.nan)


def save(fig, name: str) -> None:
    """SVG (glyphs as paths, so fonts are embedded), 600-dpi TIFF, PNG preview."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.svg")


def panel_label(sub, text: str, top: float = 0.78) -> None:
    """Panel title on the first line; the shared legend (if any) goes on the second."""
    sub.suptitle(text, x=0.0, y=0.99, ha="left", va="top", fontsize=8, weight="bold", color=INK)
    sub.subplots_adjust(top=top)


def shared_legend(sub, ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    sub.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=len(labels),
               fontsize=7, handlelength=2.2, columnspacing=1.4)


def convergence(axes) -> None:
    for ax, (model, title) in zip(axes, MODELS.items()):
        series = {g: runs(rf"{model}_{g}_s\d+") for g in "ABC"} | {"gp200": runs(r"gp200_s\d+")}
        for key, names in series.items():
            if not names:
                continue
            curves = np.array([best_so_far(n) for n in names])
            x = np.arange(1, curves.shape[1] + 1)
            st = STYLE[key]
            ax.plot(x, np.nanmean(curves, 0), color=st["color"], ls=st["ls"], label=st["label"])
            if len(names) > 1:
                ax.fill_between(x, np.nanmin(curves, 0), np.nanmax(curves, 0), color=st["color"],
                                alpha=0.12, lw=0)
        gp1000 = runs(r"gp1000_s\d+")
        if gp1000:
            level = np.nanmean([best_so_far(n)[-1] for n in gp1000])
            ax.axhline(level, color=MUTED, ls=":", lw=1.2)
            ax.annotate("GP(1000) final", (1, level), xytext=(0, 3), textcoords="offset points",
                        fontsize=7, color=INK2)
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.set_xlabel("Round (10 candidates each)")
    axes[0].set_ylabel("Best training fitness")



def classic_correlations(sub, axes, top_n: int = 8) -> None:
    from alpha_mining.analysis.final import selection_values
    from alpha_mining.baselines.alpha101 import Alpha101, eligible_formulas
    from alpha_mining.context import evaluator
    from alpha_mining.dsl.evaluate import evaluate
    from alpha_mining.evaluation.evaluator import factor_ranks
    from alpha_mining.evaluation.metrics import factor_rho
    from alpha_mining.evaluation.preprocess import preprocess

    finals = {p.stem: json.loads(p.read_text()) for p in (RES / "final").glob("*.json")}
    if not finals:
        raise SystemExit("no results/final yet")
    ev = evaluator()
    rows = ev.period_rows(*VALID)
    proc = lambda raw: factor_ranks(preprocess(raw, ev.neutralizer)[rows])
    eng = Alpha101(ev.panel.features, ev.panel.mask)
    a101 = [proc(eng.compute(f)) for f in eligible_formulas()["formula"]]
    classic = {"Reversal": "-ts_delta(close, 5)", "Momentum": "ts_mean(returns, 60)",
               "Volatility": "ts_std(returns, 20)"}
    classic = {k: proc(evaluate(e, ev.panel.features, ev.panel.mask)) for k, e in classic.items()}
    cols = ["Alpha101\n(max)", "Rev.", "Mom.", "Vol."]
    for ax, model in zip(axes, MODELS):
        # The model's Group C Top-10 factors pooled over seeds, best validation |ICIR| first.
        pool = {}
        for name, f in finals.items():
            if re.fullmatch(rf"{model}_C_s\d+", name):
                for t in f["top10"]:
                    pool.setdefault(t["key"], t)
        top = sorted(pool.values(), key=lambda t: -abs(t["valid"]["icir"]))[:top_n]
        if not top:
            ax.set_visible(False)
            continue
        vals = [factor_ranks(v[rows]) for v in selection_values({"top10": top}, ev)]
        mat = np.array([[max(factor_rho(v, a) for a in a101), *[factor_rho(v, c) for c in classic.values()]]
                        for v in vals])
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(len(cols)), cols, fontsize=7)
        ax.set_yticks(range(len(top)), [f"F{i + 1}" for i in range(len(top))], fontsize=7)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color=SURFACE if mat[i, j] > 0.55 else INK)
        ax.set_title(f"{MODELS[model]} (Group C)", fontsize=8, color=INK, loc="left")
        pd.DataFrame({"factor": [f"F{i + 1}" for i in range(len(top))], "expr": [t["expr"] for t in top]}
                     ).to_csv(OUT / f"fig2b_{model}_factors.csv", index=False)
    sub.colorbar(im, ax=axes, shrink=0.85, label="|ρ| (validation)")


def net_value(axes, tests: dict) -> None:
    for ax, model in zip(axes, MODELS):
        keys = {"A": rf"{model}_A_s\d+", "B": rf"{model}_B_s\d+", "C": rf"{model}_C_s\d+",
                "gp200": r"gp200_s\d+", "gp1000": r"gp1000_s\d+", "alpha101": "alpha101"}
        for key, pat in keys.items():
            names = [n for n in tests if re.fullmatch(pat, n)]
            if not names:
                continue
            # Equal-weight average of the seeds' weekly net returns, Test I then Test II.
            weekly = []
            for n in names:
                s = pd.concat([pd.Series(tests[n][p]["weekly"]["net"],
                                         index=pd.to_datetime(tests[n][p]["weekly"]["dates"])) for p in ("test1", "test2")])
                weekly.append(s)
            r = pd.concat(weekly, axis=1).mean(axis=1).fillna(0)
            st = STYLE[key]
            nav = (1 + r).cumprod()
            ax.plot(nav.index, nav.values, color=st["color"], ls=st["ls"], label=st["label"])
        ax.axvline(pd.Timestamp("2026-01-01"), color=MUTED, lw=0.8)
        ax.set_title(MODELS[model], fontsize=8, color=INK, loc="left")
    axes[0].set_ylabel("Net value (after 10 bps one-way)")
    for ax in axes:
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))


def regime_ic(axes, tests: dict) -> None:
    width = 0.2
    for ax, model in zip(axes, MODELS):
        keys = [("A", rf"{model}_A_s\d+"), ("B", rf"{model}_B_s\d+"), ("C", rf"{model}_C_s\d+"),
                ("gp1000", r"gp1000_s\d+")]
        for k, (key, pat) in enumerate(keys):
            names = [n for n in tests if re.fullmatch(pat, n)]
            if not names:
                continue
            means = [np.nanmean([tests[n][p]["regime_ic"][reg][0] for n in names for p in ("test1", "test2")])
                     for reg, _ in REGIMES]
            st = STYLE[key]
            ax.bar(np.arange(3) + (k - 1.5) * width, means, width * 0.9, color=st["color"],
                   hatch="//" if key.startswith("gp") else None, edgecolor=SURFACE, label=st["label"])
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(range(3), [lab for _, lab in REGIMES])
        ax.set_title(MODELS[model], fontsize=8, color=INK, loc="left")
    axes[0].set_ylabel("Mean daily IC (Test I + II)")


def figure2() -> None:
    fig = plt.figure(figsize=(WIDTH, 5.4))
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.15], hspace=0.04)
    panel_label(top, "(a) Best training fitness by round")
    axes = top.subplots(1, 2, sharey=True)
    convergence(axes)
    shared_legend(top, axes[1])
    panel_label(bottom, "(b) Correlation of leading Group C factors with known factors", top=0.86)
    classic_correlations(bottom, bottom.subplots(1, 2, gridspec_kw={"wspace": 0.35}))
    save(fig, "fig2_convergence_correlations")


def figure3() -> None:
    tests = {p.stem: json.loads(p.read_text()) for p in (RES / "test").glob("*.json")} \
        if (RES / "test").exists() else {}
    if not tests:
        raise SystemExit("no results/test yet (test set still locked)")
    fig = plt.figure(figsize=(WIDTH, 5.0))
    top, bottom = fig.subfigures(2, 1, hspace=0.04)
    panel_label(top, "(a) Net value of the Top-10 composite, Test I + II")
    axes = top.subplots(1, 2, sharey=True)
    net_value(axes, tests)
    shared_legend(top, axes[0])
    panel_label(bottom, "(b) Regime IC of the composite, Test I + II")
    axes = bottom.subplots(1, 2, sharey=True)
    regime_ic(axes, tests)
    shared_legend(bottom, axes[0])
    save(fig, "fig3_test_periods")


def single_row(name: str, height: float, draw, legend_from: int) -> None:
    """One row of two model panels with a shared legend on top (8-page paper version)."""
    fig = plt.figure(figsize=(WIDTH, height))
    axes = fig.subplots(1, 2, sharey=True)
    draw(axes)
    handles, labels = axes[legend_from].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(labels),
               fontsize=7, handlelength=2.2, columnspacing=1.4)
    fig.subplots_adjust(top=0.80, bottom=0.16)
    save(fig, name)


def paper_figures() -> None:
    """Single-panel versions used in the paper: panel (a) of Figures 2 and 3. The
    two-row figures above stay in the repository with panels (b)."""
    single_row("fig2_convergence", 2.3, convergence, 1)
    tests = {p.stem: json.loads(p.read_text()) for p in (RES / "test").glob("*.json")}
    single_row("fig3_net_value", 2.3, lambda axes: net_value(axes, tests), 0)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] != ["--paper"]:
        figure2()
        figure3()
    paper_figures()
