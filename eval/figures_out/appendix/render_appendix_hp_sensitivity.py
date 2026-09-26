#!/usr/bin/env python3
"""Appendix F3: predicate-threshold sensitivity, restyled to the paper style.

One FEATURED panel for the only threshold that moves anything
(FORBIDDEN_CONTACT_TOLERANCE_FRAMES) plus a compact grid of the six flat
knobs — the flatness is the result (those thresholds do not matter on these
corpora), so the flat panels earn only sparkline-sized space.

Two curves per panel, as in the original: agreement with the 100
human-confirmed audit instances (left axis) and label drift vs the committed
default (right axis, visually secondary). The committed default is the dotted
vertical line; its agreement is 1.0 by definition (it IS the reference), so
the content is the shape around it.

Data: SafeManip/monitor/hp_sweep/scores/cell_scores.csv (written by
hp_sweep_score.py) — copied to appendix_hp_cell_scores.csv beside this file
so the render never depends on the monitor tree.
"""
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

from appendix_style import apply_style, panel_shade, save

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "appendix", "plots")
os.makedirs(OUT, exist_ok=True)
CSV = os.path.join(HERE, "appendix_hp_cell_scores.csv")
SRC = ("/path/to/SafeManip/SafeManip/monitor"
       "/hp_sweep/scores/cell_scores.csv")

AGREE_C, DRIFT_C = "#2E7D32", "#C44E52"
KNOB_DISPLAY = {
    "FORBIDDEN_CONTACT_TOLERANCE_FRAMES": "Forbidden-contact\ntolerance (frames)",
    "SKILL_ONSET_FRAMES": "Skill onset\n(frames)",
    "SETTLE_TIMEOUT_FRAMES": "Settle timeout\n(frames)",
    "CLUTTER_THRESHOLD": "Clutter\nthreshold",
    "GRASP_BILATERAL_MIN_CONTACT_BODIES": "Grasp min contact\nbodies",
    "PERSISTENCE_FRAMES": "Persistence\n(frames)",
    "GRASP_SLIP_LINEAR_THRESHOLD": "Grasp slip\nthreshold",
}
FEATURED = "FORBIDDEN_CONTACT_TOLERANCE_FRAMES"


def draw_knob(ax, sub, default, featured=False):
    xs = sub["value"].astype(float).values
    ax.plot(xs, sub["agreement"] * 100, color=AGREE_C, linewidth=2.2,
            marker="o", markersize=7 if featured else 5, zorder=4)
    ax2 = ax.twinx()
    ax2.plot(xs, sub["drift"] * 100, color=DRIFT_C, linewidth=1.6,
             linestyle="--", marker="s", markersize=6 if featured else 4,
             zorder=3, alpha=0.9)
    ax.axvline(float(default), color="#555555", linewidth=1.2,
               linestyle=(0, (2, 2)))
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:g}" for v in xs],
                       fontsize=15 if featured else 11)
    ax.minorticks_off()
    ax.set_ylim(-4, 104)
    ax2.set_ylim(-0.028, 0.72)   # drift tops out at 0.63%
    panel_shade(ax, grid_axis="y")
    ax2.grid(False)
    ax2.tick_params(length=0)
    # The drift axis is deliberately secondary: same sweep, second reading.
    ax2.tick_params(axis="y", labelsize=13 if featured else 9,
                    colors=DRIFT_C)
    ax.tick_params(axis="y", labelsize=15 if featured else 10,
                   colors=AGREE_C)
    if not featured:
        ax2.set_yticks([0.0, 0.6])
        ax.set_yticks([75, 100])
    return ax2


def main():
    apply_style()
    if not os.path.exists(CSV):
        shutil.copy(SRC, CSV)
        print(f"[csv] {CSV} (copied from monitor tree)")
    d = pd.read_csv(CSV)
    d = d[d["cell"] != "default"]
    knobs = [FEATURED] + [k for k in KNOB_DISPLAY if k != FEATURED
                          and (d.knob == k).any()]

    fig = plt.figure(figsize=(13.6, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.6], wspace=0.30,
                          left=0.075, right=0.935, top=0.855, bottom=0.235)
    axf = fig.add_subplot(gs[0])
    sub = d[d.knob == FEATURED].sort_values("value")
    draw_knob(axf, sub, sub["default_value"].iloc[0], featured=True)
    axf.set_title("(a) Forbidden-contact\ntolerance (frames)",
                  fontsize=16, pad=8)
    axf.set_ylabel("Agreement with human (%)", fontsize=15,
                   color=AGREE_C)
    axf.set_xlabel("Threshold value", fontsize=15)

    grid = gs[1].subgridspec(2, 3, hspace=0.52, wspace=0.42)
    for i, k in enumerate(knobs[1:]):
        ax = fig.add_subplot(grid[i // 3, i % 3])
        sub = d[d.knob == k].sort_values("value")
        draw_knob(ax, sub, sub["default_value"].iloc[0])
        ax.set_title(KNOB_DISPLAY[k].replace("\n", " "), fontsize=10.5,
                     pad=4, fontweight="normal")
    fig.text(0.415, 0.93, "(b) Flat thresholds (no verdict changes "
             "across the sweep)", fontsize=17, fontweight="bold",
             ha="left")

    handles = [
        mlines.Line2D([], [], color=AGREE_C, linewidth=2.2, marker="o",
                      markersize=6, label="Agreement with human (left axis)"),
        mlines.Line2D([], [], color=DRIFT_C, linewidth=1.6, linestyle="--",
                      marker="s", markersize=5,
                      label="Label drift vs default (right axis, %)"),
        mlines.Line2D([], [], color="#555555", linewidth=1.2,
                      linestyle=(0, (2, 2)), label="Committed default"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True,
               framealpha=0.95, edgecolor="#AAAAAA", fontsize=13,
               bbox_to_anchor=(0.5, 0.008))
    save(fig, OUT, "hp_sensitivity")


if __name__ == "__main__":
    main()
