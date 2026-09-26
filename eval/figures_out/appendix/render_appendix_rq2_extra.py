#!/usr/bin/env python3
"""Appendix D2 + E2: the remaining RQ2-family heatmaps.

  appendix_rq2_counts    — violations per violated instance (conditional on
                           the instance violating at all): how many times a
                           failing obligation is re-violated within one
                           episode. (Old make_plots.py panel (b).)
  appendix_rq2_raw       — the RAW metrics: (a) safety violation rate
                           (% of applicable instances violating) and
                           (b) unsafe-state exposure rate (% of rollout
                           timesteps in violation), the robustness companion
                           to the per-trigger RQ2 figure in the main text.

Estimators are the pooled ratio of sums per (suite, model, category), same
family as render_rq2_conditional.py. Cells that are never applicable render
as the grey n/a. Rows/columns/display exactly as the main RQ2 figures.
Data: appendix_prop_cells.csv.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from appendix_style import (CATEGORY_DISPLAY, CATEGORY_ORDER, MODEL_DISPLAY,
                            MODEL_ORDER, SUITES, apply_style, draw_heatmap,
                            save, soft_cmap, suite_divider)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "appendix", "plots")
os.makedirs(OUT, exist_ok=True)
PROP = os.path.join(HERE, "appendix_prop_cells.csv")


def matrix(series, cols):
    out = pd.DataFrame(index=CATEGORY_ORDER,
                       columns=pd.MultiIndex.from_tuples(cols), dtype=float)
    for suite, m in cols:
        for cat in CATEGORY_ORDER:
            try:
                out.loc[cat, (suite, m)] = series.loc[(suite, m, cat)]
            except KeyError:
                pass
    return out


def _vmax(mat, floor=0.01):
    v = np.nanmax(mat.values) if mat.size else np.nan
    return max(floor, v) if np.isfinite(v) else floor


def main():
    apply_style()
    pc = pd.read_csv(PROP)
    g = pc.groupby(["suite", "model", "category"], observed=True)[
        ["V", "A", "D", "W", "n"]].sum()
    counts = g.V / g.W.replace(0, np.nan)          # per violated instance
    vrate = g.W / g.n.replace(0, np.nan) * 100     # % instances violated
    erate = g.D / g.n.replace(0, np.nan) * 100     # % of rollout, mean

    cols = []
    for suite in SUITES:
        have = set(pc.loc[pc.suite == suite, "model"])
        cols += [(suite, m) for m in MODEL_ORDER if m in have]
    col_labels = [MODEL_DISPLAY[m] for _, m in cols]
    row_labels = [CATEGORY_DISPLAY[c] for c in CATEGORY_ORDER]
    reds = soft_cmap("Reds", "Reds_soft_x")

    # -- D2: counts per violated instance -------------------------------------
    fig, ax = plt.subplots(figsize=(10.6, 6.4))
    cmat = matrix(counts, cols)
    im = draw_heatmap(ax, cmat, row_labels, col_labels, reds, fmt="{:.1f}",
                      vmin=1, gamma=0.5, vmax=_vmax(cmat, 2.0),
                      fontsize=13, ticksize=15)
    ax.set_title("Violations per violated instance", pad=34, fontsize=19)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(
        labelsize=13)
    suite_divider(ax, cols, band_fs=16)
    fig.tight_layout()
    save(fig, OUT, "RQ2_counts")

    # -- E2: raw metrics -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(17.6, 6.4))
    vmat = matrix(vrate, cols)
    im0 = draw_heatmap(axes[0], vmat, row_labels, col_labels, reds,
                       fmt="{:.1f}", vmin=0, gamma=0.6, vmax=_vmax(vmat, 1.0),
                       fontsize=12.5, ticksize=15)
    axes[0].set_title("(a) Safety violation rate (%)", pad=34, fontsize=19)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04).ax.tick_params(
        labelsize=13)
    emat = matrix(erate, cols)
    im1 = draw_heatmap(axes[1], emat, row_labels, col_labels, reds,
                       fmt="{:.2f}", vmin=0, gamma=0.5, vmax=_vmax(emat, 0.05),
                       fontsize=12.5, ticksize=15)
    axes[1].set_title("(b) Unsafe-state exposure rate (% of rollout)",
                      pad=34, fontsize=19)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04).ax.tick_params(
        labelsize=13)
    axes[1].set_yticklabels([])
    for ax in axes:
        suite_divider(ax, cols, band_fs=16)
    fig.tight_layout()
    save(fig, OUT, "RQ2_raw")


if __name__ == "__main__":
    main()
