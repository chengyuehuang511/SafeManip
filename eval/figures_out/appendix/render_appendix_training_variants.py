#!/usr/bin/env python3
"""Appendix G: does more (or different) adaptation data buy safety, or only
success? The three GR00T N1.5 Foundation-Model-Learning recipes, all evaluated
on the SAME RoboCasa365 target split, so the only thing that varies across the
three points is what the policy was adapted on: pretraining data only
(GR00T-pt), target demonstrations only (GR00T-to), or both in sequence
(GR00T-tpt). The Multitask-Learning checkpoint is not a point here -- its
target-split episodes were never monitored (compute_variant_cache scope note).

Panel (a) capability against safety: task success rate (x) vs violations per
triggered event (y), one point per variant, with bootstrap CIs over episodes.
A safety dividend would show as points sliding down-right along the ladder;
a capability-only dividend as points sliding right at constant height.

Panel (b) where the change lands: violations per triggered event by safety
category, grouped bars over variants. The pooled number in (a) can stay flat
while the composition underneath it moves, and the categories differ by two
orders of magnitude in trigger frequency, so this is a log axis.

Data: appendix_variant_episodes.csv + appendix_variant_prop_cells.csv
(compute_variant_cache.py). NOT appendix_episodes.csv -- that cache is the
five-policy PRETRAIN-split corpus and the two must never be pooled.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from appendix_style import (CATEGORY_DISPLAY, CATEGORY_ORDER, apply_style,
                            panel_shade, save)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "appendix", "plots")
os.makedirs(OUT, exist_ok=True)
EP = os.path.join(HERE, "appendix_variant_episodes.csv")
CELLS = os.path.join(HERE, "appendix_variant_prop_cells.csv")

# The adaptation-data ladder, in the order the text reads it.
VARIANT_ORDER = ["GR00T-pt", "GR00T-to", "GR00T-tpt"]
VARIANT_COLORS = {"GR00T-pt": "#2CA02C", "GR00T-to": "#1F77B4",
                  "GR00T-tpt": "#D62728"}
VARIANT_MARKERS = {"GR00T-pt": "^", "GR00T-to": "o", "GR00T-tpt": "s"}
# Adaptation data behind each point, spelled out on the legend so the figure is
# readable without the caption.
VARIANT_SUB = {
    "GR00T-pt": "Human300+Synthetic60",
    "GR00T-to": "Target50 only",
    "GR00T-tpt": "both, then Target50",
}
# (dx pt, dy pt) label offsets, re-fanned once the points were placed.
OFFSETS = {"GR00T-pt": (12, -10), "GR00T-to": (12, 6), "GR00T-tpt": (12, 6)}
N_BOOT = 2000
RNG_SEED = 0  # fixed: the CIs must not move between renders of the same cache


def boot_ci(num, den, n_boot=N_BOOT, seed=RNG_SEED):
    """Percentile CI for sum(num)/sum(den) resampling EPISODES, not events.

    The ratio's unit of independence is the episode: one episode contributes
    both a violation count and a trigger count, and they are correlated within
    it. Resampling events would treat the denominator as fixed and give
    intervals that are far too tight.
    """
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    if den.sum() <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    d = den[idx].sum(axis=1)
    r = np.where(d > 0, num[idx].sum(axis=1) / np.where(d > 0, d, 1), np.nan)
    return np.nanpercentile(r, 2.5), np.nanpercentile(r, 97.5)


def panel_a(ax, ep):
    rows = []
    for v in VARIANT_ORDER:
        sub = ep[ep.variant_display == v]
        if sub.empty:
            continue
        lo, hi = boot_ci(sub.V.values, sub.A.values)
        # Success from the eval harness's own stats.json, not the monitor's
        # replay flag, which is biased ~5-6 points low on this corpus (see
        # compute_variant_cache.harness_success). It is a per-task rate, so its
        # CI resamples TASKS -- 50 of them, each carrying 50 episodes.
        s = (sub.groupby("task", observed=True)["success_harness"].first()
             .values.astype(float))
        rng = np.random.default_rng(RNG_SEED)
        sboot = s[rng.integers(0, len(s), size=(N_BOOT, len(s)))].mean(axis=1)
        rows.append(dict(
            variant=v, succ=100 * s.mean(), rate=sub.V.sum() / sub.A.sum(),
            lo=lo, hi=hi,
            slo=100 * np.percentile(sboot, 2.5),
            shi=100 * np.percentile(sboot, 97.5),
            episodes=len(sub), V=sub.V.sum(), A=sub.A.sum()))
    tab = pd.DataFrame(rows)

    for _, r in tab.iterrows():
        c = VARIANT_COLORS[r.variant]
        ax.plot([r.slo, r.shi], [r.rate, r.rate], color=c, lw=1.6, alpha=0.65,
                zorder=2)
        ax.plot([r.succ, r.succ], [r.lo, r.hi], color=c, lw=1.6, alpha=0.65,
                zorder=2)
        ax.scatter(r.succ, r.rate, s=190, color=c,
                   marker=VARIANT_MARKERS[r.variant], edgecolor="white",
                   linewidth=1.4, zorder=3)
        dx, dy = OFFSETS.get(r.variant, (12, 6))
        ax.annotate(r.variant, (r.succ, r.rate), textcoords="offset points",
                    xytext=(dx, dy), fontsize=14, fontweight="bold",
                    ha="left" if dx > 0 else "right", va="center", zorder=4)

    ax.set_xlabel("Task success rate (%)", fontsize=17)
    ax.set_ylabel("Violations per triggered event", fontsize=15)
    ax.tick_params(labelsize=14)
    ax.margins(x=0.22, y=0.22)
    ax.set_ylim(bottom=max(0.0, ax.get_ylim()[0]))
    handles = [mlines.Line2D([], [], color=VARIANT_COLORS[v], marker=VARIANT_MARKERS[v],
                             linestyle="none", markersize=9,
                             markeredgecolor="white",
                             label=f"{v}: {VARIANT_SUB[v]}")
               for v in VARIANT_ORDER if v in set(tab.variant)]
    ax.legend(handles=handles, loc="best", fontsize=11, frameon=True,
              framealpha=0.95, edgecolor="#AAAAAA")
    # No LaTeX escapes here: matplotlib renders the backslash literally.
    ax.set_title("(a) Capability vs. safety on the target split", fontsize=16)
    return tab


def panel_b(ax, cells):
    present = [v for v in VARIANT_ORDER if v in set(cells.variant_display)]
    cats = [c for c in CATEGORY_ORDER if c in set(cells.category)]
    g = (cells.groupby(["variant_display", "category"], observed=True)
         [["V", "A"]].sum().reset_index())
    g["rate"] = g.V / g.A.replace(0, np.nan)
    wide = g.pivot(index="category", columns="variant_display",
                   values="rate").reindex(cats)[present]

    x = np.arange(len(cats), dtype=float)
    w = 0.8 / len(present)
    # Zeros are real (no violation of that category was ever recorded) but
    # invisible on a log axis, so they get an explicit floor tick instead of a
    # silently missing bar.
    floor = np.nanmin(wide.values[wide.values > 0]) / 3 if (wide.values > 0).any() else 1e-4
    for i, v in enumerate(present):
        vals = wide[v].values.astype(float)
        drawn = np.where((vals > 0) & np.isfinite(vals), vals, floor)
        ax.bar(x + (i - (len(present) - 1) / 2) * w, drawn, width=w * 0.92,
               color=VARIANT_COLORS[v], edgecolor="white", linewidth=0.7,
               label=v, zorder=3)
        for xi, (val, dv) in zip(x, zip(vals, drawn)):
            if not (val > 0) or not np.isfinite(val):
                ax.text(xi + (i - (len(present) - 1) / 2) * w, dv * 1.15, "0",
                        ha="center", va="bottom", fontsize=8.5, color="#555555")
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor / 1.6)
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_DISPLAY[c] for c in cats], rotation=30,
                       ha="right", fontsize=12)
    ax.set_ylabel("Violations per triggered event", fontsize=15)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(loc="upper left", fontsize=11, ncol=2, frameon=True,
              framealpha=0.95, edgecolor="#AAAAAA")
    ax.set_title("(b) Same metric, by safety category", fontsize=16)
    return wide


def main():
    apply_style()
    ep = pd.read_csv(EP, dtype={"episode": str})
    cells = pd.read_csv(CELLS)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0),
                             gridspec_kw=dict(width_ratios=[1.0, 1.25]))
    panel_shade(axes[0], grid_axis="both")
    panel_shade(axes[1], grid_axis="y")
    tab = panel_a(axes[0], ep)
    wide = panel_b(axes[1], cells)
    fig.tight_layout()
    save(fig, OUT, "training_variants")

    tab.to_csv(os.path.join(HERE, "appendix_variant_summary.csv"), index=False)
    print(tab.round(5).to_string(index=False))
    print("\nper-category violations per triggered event:")
    print(wide.round(5).to_string())


if __name__ == "__main__":
    main()
