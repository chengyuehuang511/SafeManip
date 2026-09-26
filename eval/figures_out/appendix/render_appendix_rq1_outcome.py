#!/usr/bin/env python3
"""Appendix A1 (app:rq1_outcomes): rollout outcomes, raw and per opportunity.

(a)/(b) Outcome decomposition per suite — every rollout falls in one of
    {Safe, Unsafe} x {Successful, Failed}; models sorted by task success.
(c) Violations per triggered event split by rollout outcome — failed episodes
    carry 1.7-7x the per-trigger risk of successful ones, yet successful
    rollouts still violate.

Merges the old plots/RQ1_stacked_bar_combined.png and
plots_conditional/RQ1_outcome_conditional_combined.png into the single figure
the tex caption at app:rq1_outcomes already promises.
Data: appendix_episodes.csv (violated episode := V > 0).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from appendix_style import (MODEL_DISPLAY, MODEL_ORDER, OUTCOME_COLORS,
                            OUTCOME_DISPLAY, SUITE_DISPLAY, SUITES,
                            apply_style, panel_shade, save)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "appendix", "plots")
os.makedirs(OUT, exist_ok=True)
EP = os.path.join(HERE, "appendix_episodes.csv")
SUCC_C, FAIL_C = "#2E7D32", "#C62828"


def main():
    apply_style()
    ep = pd.read_csv(EP, dtype={"episode": str})
    ep["violated"] = (ep.V > 0).astype(int)

    fig = plt.figure(figsize=(14.8, 9.2))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.92], hspace=0.52,
                             left=0.082, right=0.99, top=0.94, bottom=0.12)
    top = outer[0].subgridspec(1, 2, wspace=0.16, width_ratios=[5, 7])
    ax_a = fig.add_subplot(top[0])
    ax_b = fig.add_subplot(top[1])
    ax_c = fig.add_subplot(outer[1])

    # -- (a)/(b) outcome decomposition ---------------------------------------
    for ax, suite, letter in ((ax_a, "RoboCasa", "a"), (ax_b, "LIBERO", "b")):
        sub = ep[ep.suite == suite]
        g = (sub.groupby("model", observed=True)
             .agg(succ=("task_success", "mean"), n=("episode", "size")))
        shares = (sub.assign(outcome=np.select(
            [sub.task_success.eq(1) & sub.violated.eq(0),
             sub.task_success.eq(1) & sub.violated.eq(1),
             sub.task_success.eq(0) & sub.violated.eq(0)],
            ["safe_success", "unsafe_success", "safe_fail"], "unsafe_fail"))
            .groupby(["model", "outcome"], observed=True).size()
            .unstack(fill_value=0))
        shares = shares.div(shares.sum(axis=1), axis=0) * 100
        models = [m for m in g.sort_values("succ", ascending=False).index]
        x = np.arange(len(models))
        bottom = np.zeros(len(models))
        for oc in ("safe_success", "unsafe_success", "safe_fail",
                   "unsafe_fail"):
            v = np.array([shares.loc[m, oc] if oc in shares.columns else 0.0
                          for m in models])
            ax.bar(x, v, 0.62, bottom=bottom, color=OUTCOME_COLORS[oc],
                   edgecolor="white", linewidth=0.8)
            bottom += v
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{MODEL_DISPLAY[m]}\n({g.loc[m, 'succ'] * 100:.1f}%)"
             for m in models], fontsize=13, rotation=20, ha="right",
            rotation_mode="anchor")
        ax.set_ylim(0, 100)
        ax.set_title(f"({letter}) Rollout outcomes — {SUITE_DISPLAY[suite]}",
                     fontsize=18, pad=8)
        ax.tick_params(axis="y", labelsize=14)
        panel_shade(ax)
    ax_a.set_ylabel("Share of rollouts (%)", fontsize=16)

    # -- (c) violations per triggered event by outcome -----------------------
    risk = (ep.groupby(["suite", "model", "task_success"], observed=True)
            .agg(V=("V", "sum"), A=("A", "sum")))
    risk["r"] = risk.V / risk.A.replace(0, np.nan)
    cols = []
    for suite in SUITES:
        have = set(ep.loc[ep.suite == suite, "model"])
        cols += [(suite, m) for m in MODEL_ORDER if m in have]
    x = np.arange(len(cols), dtype=float)
    n_rc = sum(1 for s, _ in cols if s == "RoboCasa")
    x[n_rc:] += 0.7   # gap between the suites
    w = 0.34
    for succ, c, lab in ((1, SUCC_C, "Successful episodes"),
                         (0, FAIL_C, "Failed episodes")):
        v = np.array([float(risk.loc[(s, m, succ), "r"])
                      if (s, m, succ) in risk.index else np.nan
                      for s, m in cols])
        ax_c.bar(x + (w / 2 if succ == 0 else -w / 2), np.nan_to_num(v), w,
                 color=c, edgecolor="white", linewidth=0.7, label=lab)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([MODEL_DISPLAY[m] for _, m in cols], fontsize=14,
                         rotation=20, ha="right", rotation_mode="anchor")
    ax_c.set_ylabel("Violations per\ntriggered event", fontsize=16)
    ax_c.set_title("(c) Violations per triggered event, by rollout outcome",
                   fontsize=18, pad=8)
    ax_c.tick_params(axis="y", labelsize=14)
    edge = (x[n_rc - 1] + x[n_rc]) / 2
    ax_c.axvline(edge, color="#333333", linewidth=1.0, linestyle=(0, (4, 3)))
    # Suite names INSIDE the panel (above the axes they collide with the
    # panel title); the top band is empty at these bar heights.
    # RoboCasa's label starts after the in-panel legend (upper left).
    for lo, hi, name in ((x[2], x[n_rc - 1], "RoboCasa"),
                         (x[n_rc], x[-1], "LIBERO")):
        ax_c.text((lo + hi) / 2, 0.94, SUITE_DISPLAY[name], ha="center",
                  va="top", fontsize=15,
                  transform=ax_c.get_xaxis_transform())
    ax_c.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax_c.set_ylim(0, ax_c.get_ylim()[1] * 1.14)
    # Panel (c)'s own key, inside: the outcome legend below belongs to (a)/(b).
    ax_c.legend(loc="upper left", fontsize=13, frameon=True,
                framealpha=0.95, edgecolor="#AAAAAA")
    panel_shade(ax_c)

    handles = [mpatches.Patch(facecolor=OUTCOME_COLORS[k],
                              label=OUTCOME_DISPLAY[k])
               for k in ("safe_success", "unsafe_success", "safe_fail",
                         "unsafe_fail")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True,
               framealpha=0.95, edgecolor="#AAAAAA", fontsize=13,
               handlelength=1.5, columnspacing=1.2,
               bbox_to_anchor=(0.5, 0.005))
    save(fig, OUT, "RQ1_outcome")


if __name__ == "__main__":
    main()
