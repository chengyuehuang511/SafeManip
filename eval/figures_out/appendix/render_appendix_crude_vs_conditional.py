#!/usr/bin/env python3
"""Appendix C1: exposure normalization preserves the model ranking.
Restyle of plots_activation_poc/poc4_ranking_crude_vs_conditional.png,
extended to both suites.

Per model, two bars: the safety violation rate (violated property instances,
% — left axis, gray) and violations per triggered event (right axis, red).
The bars fall in the same order under both metrics: normalizing by triggered
events changes the QUESTION (per-opportunity instead of per-instance), not
the model ranking. Twin axes are two readings of the same models, made
visually secondary on the right.
Data: appendix_prop_cells.csv.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from appendix_style import (MODEL_DISPLAY, MODEL_ORDER, SUITE_DISPLAY,
                            SUITES, apply_style, panel_shade, save)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "appendix", "plots")
os.makedirs(OUT, exist_ok=True)
PROP = os.path.join(HERE, "appendix_prop_cells.csv")
CRUDE_C, COND_C = "#9E9E9E", "#D62728"


def main():
    apply_style()
    pc = pd.read_csv(PROP)
    g = pc.groupby(["suite", "model"], observed=True)[
        ["V", "A", "W", "n"]].sum()
    g["crude"] = g.W / g.n * 100
    g["cond"] = g.V / g.A.replace(0, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.4),
                             gridspec_kw=dict(width_ratios=[5, 7]))
    for ax, suite, letter in zip(axes, SUITES, "ab"):
        models = [m for m in MODEL_ORDER if (suite, m) in g.index]
        # Sorted by the crude metric so "same ranking" is read as monotonicity
        models = sorted(models, key=lambda m: -g.loc[(suite, m), "crude"])
        x = np.arange(len(models), dtype=float)
        w = 0.36
        ax.bar(x - w / 2, [g.loc[(suite, m), "crude"] for m in models], w,
               color=CRUDE_C, edgecolor="white", linewidth=0.7)
        ax2 = ax.twinx()
        ax2.bar(x + w / 2, [g.loc[(suite, m), "cond"] for m in models], w,
                color=COND_C, edgecolor="white", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY[m] for m in models], fontsize=14,
                           rotation=20, ha="right", rotation_mode="anchor")
        ax.set_title(f"({letter}) {SUITE_DISPLAY[suite]}", fontsize=18, pad=8)
        ax.tick_params(axis="y", labelsize=14, colors="#616161")
        ax2.tick_params(axis="y", labelsize=14, colors=COND_C)
        ax.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)
        panel_shade(ax)
        ax2.grid(False)
        ax2.tick_params(length=0)
        if ax is axes[0]:
            ax.set_ylabel("Safety violation rate (%)", fontsize=15,
                          color="#424242")
        if ax is axes[1]:
            ax2.set_ylabel("Violations per triggered event", fontsize=15,
                           color=COND_C, rotation=270, labelpad=20)

    handles = [
        mpatches.Patch(facecolor=CRUDE_C,
                       label="Safety violation rate (violated instances %, "
                             "left axis)"),
        mpatches.Patch(facecolor=COND_C,
                       label="Violations per triggered event (right axis)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=True,
               framealpha=0.95, edgecolor="#AAAAAA", fontsize=13.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Exposure normalization preserves the model ranking",
                 fontsize=19, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0.10, 1, 0.97), w_pad=2.6)
    save(fig, OUT, "crude_vs_conditional")


if __name__ == "__main__":
    main()
