#!/usr/bin/env python3
"""Appendix B2: triggered events scale with task structure (RoboCasa365).
Restyle of plots_activation_poc/poc1_activation_vs_complexity.png.

Task-level points, successful vs failed episodes fitted separately:
(a) triggered events per episode vs number of subtasks (execution depth),
(b) unique properties triggered per episode (conceptual breadth).
Successful episodes traverse the whole task, so their count tracks intrinsic
complexity; failed ones truncate where the policy died. RoboCasa only — its
ground-truth numberSubtask spans 1-15; LIBERO tops out at 4 primitives.
Data: appendix_episodes.csv (+ n_subtasks from analysis/taskDiff.csv).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sps

from appendix_style import apply_style, panel_shade, save, stars

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "appendix", "plots")
os.makedirs(OUT, exist_ok=True)
EP = os.path.join(HERE, "appendix_episodes.csv")
SUCC_C, FAIL_C = "#2E7D32", "#C44E52"


def panel(ax, tab, ycol, ylab, title):
    handles = []
    for succ, c, mk, lab in ((1, SUCC_C, "o", "success"),
                             (0, FAIL_C, "^", "fail")):
        sub = tab[tab.task_success == succ].dropna(subset=["n_subtasks", ycol])
        ax.scatter(sub.n_subtasks, sub[ycol], s=64, color=c, marker=mk,
                   alpha=0.75, edgecolor="#333333", linewidth=0.6, zorder=3)
        rho, p = sps.spearmanr(sub.n_subtasks, sub[ycol])
        lr = sps.linregress(sub.n_subtasks, sub[ycol])
        xs = np.array([sub.n_subtasks.min(), sub.n_subtasks.max()])
        ls = (0, (6, 2)) if succ else (0, (1.6, 1.6))
        ax.plot(xs, lr.intercept + lr.slope * xs, color=c, linestyle=ls,
                linewidth=2.2, zorder=2)
        handles.append(mlines.Line2D(
            [], [], color=c, linestyle=ls, linewidth=2.2, marker=mk,
            markersize=8, markeredgecolor="#333333", markeredgewidth=0.6,
            label=f"{lab}: " + r"$\rho$=" + f"{rho:+.2f} {stars(p)}"))
    ax.legend(handles=handles, loc="upper left" if "unique" in ylab.lower()
              else "upper left", fontsize=13.5, frameon=True,
              framealpha=0.95, edgecolor="#AAAAAA")
    ax.set_xlabel("Number of subtasks", fontsize=17)
    ax.set_ylabel(ylab, fontsize=17)
    ax.set_title(title, fontsize=18, pad=8)
    ax.tick_params(labelsize=15)
    ax.set_ylim(bottom=0)
    panel_shade(ax, grid_axis="both")


def main():
    apply_style()
    ep = pd.read_csv(EP, dtype={"episode": str})
    ep = ep[ep.suite == "RoboCasa"]
    tab = (ep.groupby(["task", "task_success"], observed=True)
           .agg(n_subtasks=("n_subtasks", "first"), events=("A", "mean"),
                unique=("act_unique", "mean"), n=("episode", "size"))
           .reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4))
    panel(axes[0], tab, "events", "Triggered events per episode",
          "(a) Execution depth")
    panel(axes[1], tab, "unique", "Unique properties per episode",
          "(b) Conceptual breadth")
    fig.tight_layout(w_pad=2.4)
    save(fig, OUT, "poc_complexity")


if __name__ == "__main__":
    main()
