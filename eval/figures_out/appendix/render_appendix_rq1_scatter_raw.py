#!/usr/bin/env python3
"""Appendix E1: RQ1 scatter under the RAW metric — safety violation rate
(% of rollouts violating >= 1 property) vs task success. The robustness
companion to the main text's violations-per-triggered-event scatter: the
benchmark-dependent capability-safety pattern is not an artifact of the
per-trigger normalization.

Layout and style mirror render_rq1_scatter_single.py (labels at the points,
leaders where dense, per-suite fits in a framed legend).
Data: appendix_episodes.csv (violated episode := V > 0).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from appendix_style import (MODEL_COLORS, MODEL_DISPLAY_BOLD, SUITE_DASHES,
                            SUITE_DISPLAY, SUITE_FIT_COLORS, SUITE_MARKERS,
                            SUITES, apply_style, fit_label, fit_stats, save)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "appendix", "plots")
os.makedirs(OUT, exist_ok=True)
EP = os.path.join(HERE, "appendix_episodes.csv")

# (dx pt, dy pt, leader?) — same grammar as the main RQ1 scatter, re-fanned
# for this metric's geometry (RoboCasa sits at 25-40%, LIBERO at 3-30%).
OFFSETS = {
    ("RoboCasa", "grootn15"): (12, 2, False),   # right of the dot: centred
    ("RoboCasa", "grootn16"): (-10, 2, False),  # above it overflowed the box
    ("RoboCasa", "openpi_pi0"): (-13, -4, False),
    ("RoboCasa", "openpi_pi05"): (13, -6, False),
    ("RoboCasa", "rldx1"): (13, 2, False),
    # Every LIBERO policy except OpenVLA gets a leader (explicit user
    # decision): the cluster is too tight to bind labels by adjacency alone.
    ("LIBERO", "openvla"): (14, 4, False),
    ("LIBERO", "grootn15"): (-20, 5, True),
    ("LIBERO", "openpi_pi0"): (6, 15, True),
    ("LIBERO", "openpi_pi05"): (13, 12, True),
    ("LIBERO", "grootn16"): (-30, -4, True),
    ("LIBERO", "rldx1"): (17, -5, True),
    ("LIBERO", "cosmos_policy"): (-16, -20, True),
}
LABEL_TEXT = {
    (s, m): "GR00T\n" + MODEL_DISPLAY_BOLD[m][-4:]
    for s in SUITES for m in ("grootn15", "grootn16")
}
# Single line in the tight LIBERO cluster, as in the main RQ1 figure.
del LABEL_TEXT[("LIBERO", "grootn16")]


def main():
    apply_style()
    matplotlib.rcParams.update({
        "axes.grid": True, "grid.color": "#CCCCCC",
        "grid.linewidth": 0.8, "grid.linestyle": "--"})
    ep = pd.read_csv(EP, dtype={"episode": str})
    tab = (ep.assign(viol=(ep.V > 0).astype(float))
           .groupby(["suite", "model"], observed=True)
           .agg(succ=("task_success", "mean"), rate=("viol", "mean"))
           .reset_index())
    tab["rate"] *= 100
    tab.to_csv(os.path.join(HERE, "appendix_rq1_scatter_raw.csv"),
               index=False)

    stats = {s: fit_stats(tab.loc[tab.suite == s, "succ"] * 100,
                          tab.loc[tab.suite == s, "rate"]) for s in SUITES}

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for suite in SUITES:
        sub = tab[tab.suite == suite]
        st = stats[suite]
        xs = np.array([sub.succ.min(), sub.succ.max()]) * 100
        ax.plot(xs, st["intercept"] + st["slope"] * xs,
                color=SUITE_FIT_COLORS[suite], linestyle=SUITE_DASHES[suite],
                linewidth=2.0, alpha=0.9, zorder=1)
        for _, r in sub.sort_values("rate", ascending=False).iterrows():
            ax.scatter(r.succ * 100, r.rate, s=150,
                       color=MODEL_COLORS[r.model],
                       marker=SUITE_MARKERS[suite], edgecolor="white",
                       linewidth=1.4, zorder=3)
    for _, r in tab.iterrows():
        off = OFFSETS.get((r.suite, r.model), (12, 6, False))
        dx, dy, leader = off[:3]
        ha = "center" if dx == 0 else ("left" if dx > 0 else "right")
        va = ("bottom" if dy >= 0 else "top") if dx == 0 else "center"
        kw = dict(textcoords="offset points", xytext=(dx, dy), fontsize=15,
                  fontweight="bold", ha=ha, multialignment="center",
                  linespacing=0.95, va=va, zorder=4)
        if leader:
            kw["arrowprops"] = dict(arrowstyle="-", lw=1.0,
                                    color=MODEL_COLORS[r.model],
                                    shrinkA=0, shrinkB=8)
        ax.annotate(LABEL_TEXT.get((r.suite, r.model),
                                   MODEL_DISPLAY_BOLD[r.model]),
                    (r.succ * 100, r.rate), **kw)

    handles = [mlines.Line2D([], [], color=SUITE_FIT_COLORS[s],
                             linestyle=SUITE_DASHES[s], linewidth=2.0,
                             label=fit_label(SUITE_DISPLAY[s], stats[s]))
               for s in SUITES]
    ax.legend(handles=handles, loc="upper right", fontsize=15,
              frameon=True, framealpha=0.95, edgecolor="#AAAAAA")
    ax.margins(x=0.10, y=0.14)
    # Label room only (Cosmos Policy's label sits below its low point) --
    # no data or tick lives below 0.
    ax.set_ylim(bottom=-2.5)
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + 6.0)
    ax.set_xlabel("Task success rate (%)", fontsize=21)
    ax.set_ylabel("Safety violation rate (%)", fontsize=17)
    ax.tick_params(labelsize=18)
    fig.tight_layout()
    save(fig, OUT, "RQ1_raw")


if __name__ == "__main__":
    main()
