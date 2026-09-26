#!/usr/bin/env python3
"""Appendix B1: triggered events are a real measurement instrument, model
level. Restyle of plots_activation_poc/poc_model_level.png with paper terms.

(a) Triggered events per episode vs task success — more capable policies
    trigger FEWER events per episode (pooled r ~ -0.97***),
(b) property coverage vs success — while covering MORE of the demonstrated
    structure (pooled r ~ +0.97***). Coverage is the DEMO-UNION definition
    from activation_poc.coverage_frame: |union-set properties triggered| /
    |union set|, where the union is what any human demo of the task
    triggered, WITH the virtual terminal-release credit for LIBERO episodes
    that succeed while still grasping (LIBERO truncates at the success flag).
Together: the count tracks how a policy behaves, not monitor noise.
Data: appendix_episodes.csv (+ appendix_poc_model_level.csv, computed once
from the activation shards via activation_poc.coverage_frame; --recompute
rebuilds it).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from appendix_style import (MODEL_COLORS, MODEL_DISPLAY, MODEL_ORDER,
                            SUITE_DISPLAY, SUITE_FIT_COLORS, SUITE_MARKERS,
                            SUITE_DASHES, SUITES, apply_style, fit_label,
                            fit_stats, panel_shade, save)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "appendix", "plots")
os.makedirs(OUT, exist_ok=True)
EP = os.path.join(HERE, "appendix_episodes.csv")
CSV = os.path.join(HERE, "appendix_poc_model_level.csv")
PC = ("/path/to/SafeManip/eval/"
      "saved_eval_rollouts/monitor_files/0920/plot_combined")
POOLED_STYLE = dict(color="#111111", linestyle="-", linewidth=2.0)


def recompute_table(ep):
    """(suite, model) means; coverage via the original demo-union definition
    (activation_poc.coverage_frame), fed with this cache's episode table."""
    sys.path.insert(0, PC)
    from activation_poc import coverage_frame  # noqa: E402
    rows = []
    for suite in SUITES:
        sub = ep[ep.suite == suite]
        cov = coverage_frame(suite, sub[["model", "task", "episode",
                                         "task_success"]].copy())
        cov_m = cov.groupby("model")["coverage"].mean()
        g = (sub.groupby("model", observed=True)
             .agg(succ=("task_success", "mean"), events=("A", "mean")))
        for m in g.index:
            rows.append(dict(suite=suite, model=m, succ=g.loc[m, "succ"],
                             events=g.loc[m, "events"],
                             coverage=float(cov_m.get(m, np.nan))))
    return pd.DataFrame(rows)


def scatter_panel(ax, tab, ycol, ylab, title, legend_loc):
    for suite in SUITES:
        sub = tab[tab.suite == suite]
        for _, r in sub.iterrows():
            ax.scatter(r["succ"] * 100, r[ycol], s=150,
                       color=MODEL_COLORS[r["model"]],
                       marker=SUITE_MARKERS[suite], edgecolor="white",
                       linewidth=1.4, zorder=3)
    handles = []
    for suite in SUITES:
        sub = tab[tab.suite == suite]
        st = fit_stats(sub["succ"] * 100, sub[ycol])
        xs = np.array([sub["succ"].min(), sub["succ"].max()]) * 100
        ax.plot(xs, st["intercept"] + st["slope"] * xs,
                color=SUITE_FIT_COLORS[suite],
                linestyle=SUITE_DASHES[suite], linewidth=2.0, zorder=2)
        handles.append(mlines.Line2D(
            [], [], color=SUITE_FIT_COLORS[suite],
            linestyle=SUITE_DASHES[suite], linewidth=2.0,
            label=fit_label(SUITE_DISPLAY[suite], st)))
    st = fit_stats(tab["succ"] * 100, tab[ycol])
    xs = np.array([tab["succ"].min(), tab["succ"].max()]) * 100
    ax.plot(xs, st["intercept"] + st["slope"] * xs, zorder=2, **POOLED_STYLE)
    handles.append(mlines.Line2D([], [], label=fit_label("Pooled", st),
                                 **POOLED_STYLE))
    ax.legend(handles=handles, loc=legend_loc, fontsize=12.5, frameon=True,
              framealpha=0.95, edgecolor="#AAAAAA")
    ax.set_xlabel("Task success rate (%)", fontsize=17)
    ax.set_ylabel(ylab, fontsize=17)
    ax.set_title(title, fontsize=18, pad=8)
    ax.tick_params(labelsize=15)
    ax.margins(x=0.08, y=0.10)
    ax.set_ylim(bottom=0)
    panel_shade(ax, grid_axis="both")


def main():
    apply_style()
    if "--recompute" in sys.argv or not os.path.exists(CSV):
        ep = pd.read_csv(EP, dtype={"episode": str})
        tab = recompute_table(ep)
        tab.to_csv(CSV, index=False)
        print(f"[csv] {CSV}")
    else:
        tab = pd.read_csv(CSV)

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.6))
    scatter_panel(axes[0], tab, "events", "Triggered events per episode",
                  "(a) Triggered events per episode", "upper right")
    scatter_panel(axes[1], tab, "coverage", "Property coverage",
                  "(b) Property coverage", "lower right")
    axes[1].set_ylim(0, 1.0)

    handles = [mlines.Line2D([], [], marker="o", linestyle="none",
                             markersize=9, color=MODEL_COLORS[m],
                             markeredgecolor="white", markeredgewidth=1.2,
                             label=MODEL_DISPLAY[m]) for m in MODEL_ORDER]
    handles += [mlines.Line2D([], [], marker=SUITE_MARKERS[s],
                              linestyle="none", markersize=9,
                              markerfacecolor="#BBBBBB",
                              markeredgecolor="#333333",
                              label=SUITE_DISPLAY[s]) for s in SUITES]
    fig.legend(handles=handles, loc="lower center", ncol=9, frameon=True,
               framealpha=0.95, edgecolor="#AAAAAA", fontsize=12.5,
               handletextpad=0.25, columnspacing=0.9,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=2.2)
    save(fig, OUT, "poc_model_level")


if __name__ == "__main__":
    main()
