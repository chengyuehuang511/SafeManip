#!/usr/bin/env python3
"""Appendix E3: RQ3 under the RAW metrics — safety violation rate (% of
rollouts violating >= 1 property) and unsafe-state exposure rate (% of
rollout timesteps in violation) by task horizon and category. Robustness
companion to the main text's per-trigger RQ3 figure: reuses that figure's
exact stacked-pairs layout and paired-bar grammar by importing
render_rq3_conditional from figures_out (bar geometry, panel treatment,
legend), swapping only the metric.

Episode-level rates: total = share of episodes with V > 0 (violation rate)
or mean of D (exposure); succ = the same over successful episodes only.
Data: the parent RQ3 cache (RQ3_conditional_episodes.csv) — same episodes,
same tier/category maps.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "appendix", "plots")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.dirname(HERE))
import render_rq3_conditional as R3  # noqa: E402

from appendix_style import save  # noqa: E402

CSV = os.path.join(os.path.dirname(HERE), "RQ3_conditional_episodes.csv")


def rate_raw(ep, by, num):
    """num='viol' -> % episodes violated; num='D' -> mean exposure (% of
    rollout). Episode means, pooled over policies (same pooling stance as the
    conditional figure's default)."""
    d = ep.assign(val=(ep.V > 0).astype(float) if num == "viol" else ep.D)
    keys = ["suite"] + by
    out = pd.DataFrame({"total": d.groupby(keys, observed=True)["val"].mean()})
    s = (d[d.task_success == 1].groupby(keys, observed=True)
         .agg(v=("val", "mean"), n=("episode", "size"))
         .reindex(out.index))
    out["succ"] = s.v.where(s.n >= R3.MIN_SUCC)
    return out * 100.0


def main():
    R3.apply_style()
    ep = pd.read_csv(CSV, dtype={"episode": str})
    tiers = [t for t in R3.TIER_ORDER if (ep.tier == t).any()]
    cats = sorted([c for c in R3.CATEGORY_ORDER if (ep.category == c).any()],
                  key=lambda c: -int(ep.loc[(ep.category == c) &
                                            (ep.suite == "RoboCasa"),
                                            "task"].nunique()))

    fig = plt.figure(figsize=(9.5, 9.0))
    outer = fig.add_gridspec(2, 1, left=0.148, right=0.995, top=0.956,
                             bottom=0.097, hspace=0.32)
    axes = []
    for gi in range(2):
        sub = outer[gi].subgridspec(2, 1, hspace=0.14)
        axes.append(fig.add_subplot(sub[0]))
        axes.append(fig.add_subplot(sub[1]))
    tier_labs = R3.tick_labels(tiers, R3.counts(ep, ["tier"]))
    cat_labs = R3.tick_labels(cats, R3.counts(ep, ["category"]),
                              R3.CATEGORY_DISPLAY)
    span_t = 0.84 * len(tiers) / len(cats) * 1.6
    for gi, (by, keys, labs, span, fs, gtitle) in enumerate((
            ("tier", tiers, tier_labs, span_t, 16, "Task horizon"),
            ("category", cats, cat_labs, 0.84, 12.0, "Task category"))):
        for pi, (num, ylab) in enumerate((
                ("viol", "Safety\nviolation\nrate (%)"),
                ("D", "Unsafe-state\nexposure\nrate (%)"))):
            ax = axes[gi * 2 + pi]
            R3.suite_bars(ax, rate_raw(ep, [by], num), keys,
                          mode="bars", succ=True, span=span)
            ax.set_ylabel(ylab, fontsize=14, fontweight="bold")
            ax.tick_params(axis="y", labelsize=15)
            if pi == 0:
                ax.set_title(f"({'ab'[gi]}) {gtitle}", pad=8,
                             fontsize=19, fontweight="bold")
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels(labs, fontsize=fs)
    handles = []
    for s in R3.SUITES:
        handles.append(Patch(
            facecolor=R3.SUITE_LIGHT[s], edgecolor="#888888",
            linewidth=0.5, label=f"{R3.SUITE_DISPLAY[s]} (all rollouts)"))
        handles.append(Patch(
            facecolor=R3.SUITE_COLORS[s], hatch=R3.SUCC_HATCH,
            edgecolor="white", linewidth=0.0,
            label=f"{R3.SUITE_DISPLAY[s]} (successful only)"))
    # ONE y scale per metric across the two groupings, as in the conditional
    # figure; the violation pair gets legend headroom.
    for pair, pad in (((0, 2), 1.52), ((1, 3), 1.0)):
        ytop = max(axes[i].get_ylim()[1] for i in pair) * pad
        for i in pair:
            axes[i].set_ylim(0, ytop)
    axes[0].legend(handles=handles, loc="upper left", ncol=2,
                   frameon=True, framealpha=0.95, edgecolor="#AAAAAA",
                   fontsize=13, handlelength=1.5, handletextpad=0.5,
                   columnspacing=1.2)
    save(fig, OUT, "RQ3_raw")


if __name__ == "__main__":
    main()
