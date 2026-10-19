"""Figure 1: system and evaluation protocol (SVG + TIFF in results/figures/).

Top: one mining round (prompt -> LLM -> AST sandbox -> training-period
evaluator -> training pool), the feedback every group gets and the Diagnostic
Translator only Group C gets; GP replaces prompt + LLM under the same DSL,
fitness and budget. Bottom: the period split (not to scale) and where each
period is used.

  uv run --with matplotlib==3.10.0 python scripts/make_figure1.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent))
from make_figures import GRID, INK, INK2, MUTED, STYLE, SURFACE, WIDTH, save  # noqa: E402

C = STYLE["C"]["color"]
FS = 6.5


def box(ax, x0, y0, x1, y1, title, body="", edge=INK2, dashed=False, fill=SURFACE, tcolor=INK):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.8",
                                lw=0.9, ec=edge, fc=fill, ls="--" if dashed else "-"))
    cx = (x0 + x1) / 2
    if body:
        ax.text(cx, y1 - 1.5, title, ha="center", va="top", fontsize=FS + 0.5, weight="bold", color=tcolor)
        ax.text(cx, y1 - 4.8, body, ha="center", va="top", fontsize=FS, color=INK2, linespacing=1.3)
    else:
        ax.text(cx, (y0 + y1) / 2, title, ha="center", va="center", fontsize=FS + 0.5, weight="bold", color=tcolor)


def arrow(ax, pts, color=INK2, label=None, label_at=None, ls="-"):
    """Polyline arrow through `pts`; the head sits on the last point."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=0.9, ls=ls, solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9, ls=ls, mutation_scale=7,
                                shrinkA=0, shrinkB=0))
    if label:
        ax.text(*label_at, label, fontsize=FS - 0.3, color=color, ha="center", va="bottom")


def main() -> None:
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    # ---- one mining round (top row) ----
    top0, top1, mid = 30.5, 42.5, 36.5
    box(ax, 1, top0, 19.5, top1, "Prompt", "A: DSL + scoring\nB: + theory priors\nC: + diagnosis")
    box(ax, 22, top0, 39, top1, "LLM", "DeepSeek, Qwen\n10 per round")
    box(ax, 41.5, top0, 57, top1, "AST sandbox", "whitelisted ops\ndepth ≤ 6, dedup")
    box(ax, 59.5, top0, 78.5, top1, "Evaluator", "train period only\nIC, turnover, fitness")
    box(ax, 81, top0, 99, top1, "Training pool", "replace if fitter\nρ ≤ 0.6")
    for x0, x1 in ((19.5, 22), (39, 41.5), (57, 59.5), (78.5, 81)):
        arrow(ax, [(x0, mid), (x1, mid)])

    # GP baseline replaces prompt + LLM
    ax.add_patch(FancyBboxPatch((1, 45), 42, 4.5, boxstyle="round,pad=0,rounding_size=0.8",
                                lw=0.9, ec=MUTED, fc=SURFACE, ls="--"))
    ax.text(22, 47.25, "GP baseline (DEAP) in place of Prompt + LLM", ha="center", va="center",
            fontsize=FS, color=INK2)
    arrow(ax, [(43, 47.25), (48.75, 47.25), (48.75, top1)], color=MUTED, ls="--")

    # ---- feedback (bottom row) ----
    bot0, bot1 = 14, 23.5
    box(ax, 62, bot0, 99, bot1, "Feedback (all groups)", "metrics of best 2 + last round's best\n+ last round's errors")
    box(ax, 18, bot0, 59, bot1, "Diagnostic Translator (Group C)",
        "R1 regime failure · R2 high turnover\nR3 redundancy → repair instructions", edge=C, tcolor=C)
    arrow(ax, [(91, top0), (91, bot1)])                                 # round results -> feedback
    arrow(ax, [(62, 18.75), (59, 18.75)], color=C)                      # feedback -> translator
    arrow(ax, [(18, 18.75), (5, 18.75), (5, top0)], color=C)            # translator -> prompt
    arrow(ax, [(68, bot1), (68, 27.5), (13, 27.5), (13, top0)])             # feedback -> prompt
    ax.text(40, 26.9, "20 rounds × 10 = 200 evaluations; no chat history",
            fontsize=FS - 0.5, color=INK2, ha="center", va="top")

    # ---- period split (bottom strip, not to scale) ----
    spans = [("Train 2016–21: mining", 1, 36, GRID), ("Valid 2022–23: selection", 36, 64, GRID),
             ("Test I 2024–25", 64, 82, "#d9e6f6"), ("Test II 2026/1–9", 82, 99, "#d9e6f6")]
    for label, a, b, fc in spans:
        ax.add_patch(plt.Rectangle((a, 5), b - a, 5, fc=fc, ec=SURFACE, lw=1.5))
        ax.text((a + b) / 2, 7.5, label, ha="center", va="center", fontsize=FS - 0.5, color=INK)
    ax.text(64.5, 10.6, "evaluated once, after design freeze", fontsize=FS - 0.8, color=INK2, ha="left", va="bottom")
    cut = 82 + (99 - 82) * 5 / 9
    ax.plot([cut, cut], [2.5, 4.6], color=C, lw=1)
    ax.plot(cut, 4.6, marker="^", color=C, ms=3.5)
    ax.text(cut - 0.8, 1.8, "probed knowledge cutoff 2026-05", fontsize=FS - 0.8, color=C, ha="right", va="center")

    save(fig, "fig1_architecture")


if __name__ == "__main__":
    main()
