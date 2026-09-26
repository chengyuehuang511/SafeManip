#!/usr/bin/env python3
"""Appendix F2 (-> 12_appendix_agreement.tex): the monitor is calibrated —
quiet on human demonstrations, and where it fires a human agrees.

Restyle of plots/training_alignment.png. Panels as the original:
 (a) property instances violated (%) on human demos vs VLA eval rollouts,
     log scale — the demo positive rate is 1-2 orders of magnitude lower;
 (b) WHICH properties fire on the demos — concentrated in the contact family,
     the expected signature of imperfect teleoperation;
 (c) the human audit: a census of all violated instances in the final corpora
     plus satisfied spot-checks, all confirmed.

Data: computed ONCE from the original pipeline's loaders (audit CSVs +
viewer annotation store + reanalysis estimands) and cached to
appendix_training_alignment.json; pass --recompute after re-annotating or
re-auditing.
"""
import json
import os
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from appendix_style import SUITE_DISPLAY, apply_style, panel_shade, save

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "appendix", "plots")
os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(HERE, "appendix_training_alignment.json")
PC = ("/path/to/SafeManip/eval/saved_eval_rollouts"
      "/monitor_files/0920/plot_combined")
SUITES = ("RoboCasa", "LIBERO")
TRAIN_C, EVAL_C = "#2E7D32", "#C62828"
SUITE_STACK = {"RoboCasa": "#455A64", "LIBERO": "#8C6BB1"}


def recompute():
    sys.path.insert(0, PC)
    import plot_training_alignment as TA  # noqa: E402
    train, ev, human = TA.load_training(), TA.load_eval(), TA.load_human()
    a = {}
    for suite in SUITES:
        d = train[suite]
        a[suite] = dict(train_rate=float(d["violated"].mean()),
                        eval_rate=float(ev[suite]),
                        n_demos=int(d.groupby(["task", "episode"]).ngroups))
    counts = {s: Counter() for s in SUITES}
    for suite in SUITES:
        d = train[suite]
        for p, g in d.groupby("property_name"):
            counts[suite][p] = int(g["violated"].sum())
    props = [p for p, _ in
             (counts["RoboCasa"] + counts["LIBERO"]).most_common()
             if counts["RoboCasa"][p] + counts["LIBERO"][p] > 0]
    b = {p: {s: counts[s][p] for s in SUITES} for p in props}
    c = defaultdict(Counter)
    for suite, section, verdict in human:
        c[f"{suite}|{section}"][verdict] += 1
    return dict(a=a, b=b, c={k: dict(v) for k, v in c.items()})


def short_prop(p):
    import re
    import textwrap
    return "\n".join(textwrap.wrap(
        re.sub(r"^rc_", "", p).replace("_", " "), 26))


def main():
    apply_style()
    if "--recompute" in sys.argv or not os.path.exists(CACHE):
        data = recompute()
        with open(CACHE, "w") as f:
            json.dump(data, f, indent=1)
        print(f"[json] {CACHE}")
    else:
        data = json.load(open(CACHE))

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2),
                             gridspec_kw={"width_ratios": [1, 1.45, 1.1],
                                          "wspace": 0.74})

    # (a) demo vs eval violation rate, log scale
    ax = axes[0]
    x = np.arange(len(SUITES))
    w = 0.36
    for i, suite in enumerate(SUITES):
        d = data["a"][suite]
        ax.bar(i - w / 2, d["train_rate"] * 100, w, color=TRAIN_C,
               edgecolor="white", lw=0.7)
        ax.bar(i + w / 2, d["eval_rate"] * 100, w, color=EVAL_C,
               edgecolor="white", lw=0.7)
        ax.annotate(f"{d['train_rate'] * 100:.2f}%",
                    (i - w / 2, d["train_rate"] * 100),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=12)
        ax.annotate(f"{d['eval_rate'] * 100:.1f}%",
                    (i + w / 2, d["eval_rate"] * 100),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{SUITE_DISPLAY[s]}\n({data['a'][s]['n_demos']} "
                        "demos)" for s in SUITES], fontsize=12)
    ax.set_xlim(-0.65, len(SUITES) - 0.35)
    ax.set_yscale("log")
    ax.set_ylim(top=max(d["eval_rate"] for d in data["a"].values()) * 100 * 7)
    ax.set_ylabel("Property instances violated (%)", fontsize=15)
    ax.set_title("(a) Violation rate\non human demos", fontsize=16, pad=8)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(handles=[mpatches.Patch(color=TRAIN_C, label="Human demos"),
                       mpatches.Patch(color=EVAL_C,
                                      label="VLA policies (eval)")],
              fontsize=12, loc="upper center", frameon=True,
              framealpha=0.95, edgecolor="#AAAAAA")
    panel_shade(ax)

    # (b) which properties fire on the demos
    ax = axes[1]
    props = list(data["b"])
    y = np.arange(len(props))
    left = np.zeros(len(props))
    for suite in SUITES:
        v = np.array([data["b"][p][suite] for p in props], float)
        ax.barh(y, v, left=left, color=SUITE_STACK[suite],
                edgecolor="white", lw=0.6, label=SUITE_DISPLAY[suite])
        left += v
    for i in range(len(props)):
        ax.annotate(f" {int(left[i])}", (left[i], i), va="center",
                    fontsize=12, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels([short_prop(p) for p in props], fontsize=11.5)
    ax.invert_yaxis()
    ax.set_xlim(0, left.max() * 1.12)
    ax.set_xlabel("Violated instances in the training corpora", fontsize=15)
    ax.set_title("(b) Violations in human demos", fontsize=17, pad=8)
    ax.tick_params(axis="x", labelsize=13)
    ax.legend(fontsize=12, loc="lower right", frameon=True, framealpha=0.95,
              edgecolor="#AAAAAA")
    panel_shade(ax, grid_axis="x")

    # (c) human audit
    ax = axes[2]
    order = [k for s in SUITES for k in (f"{s}|violations", f"{s}|satisfied")
             if k in data["c"]]
    y = np.arange(len(order))
    colors = {"confirmed": "#2E7D32", "disputed": "#C62828",
              "unsure": "#F9A825", "unverifiable": "#90A4AE",
              "unlabelled": "#CFD8DC"}
    left = np.zeros(len(order))
    seen = []
    for v in colors:
        vals = np.array([data["c"][k].get(v, 0) for k in order], float)
        if not vals.any():
            continue
        ax.barh(y, vals, left=left, color=colors[v], edgecolor="white",
                lw=0.6)
        seen.append(v)
        left += vals
    total = conf = 0
    for i, k in enumerate(order):
        n = sum(data["c"][k].values())
        cnf = data["c"][k].get("confirmed", 0)
        total += n
        conf += cnf
        ax.annotate(f" {cnf}/{n}", (left[i], i), va="center", fontsize=12,
                    color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels([k.replace("|", "\n").replace(
        "RoboCasa", SUITE_DISPLAY["RoboCasa"]) for k in order], fontsize=12.5)
    ax.invert_yaxis()
    ax.set_xlim(0, left.max() * 1.30)
    ax.set_xlabel("Audited instances\n(human-confirmed)", fontsize=15)
    ax.set_title(f"(c) Human audit:\n{conf}/{total} confirmed", fontsize=16,
                 pad=8)
    ax.tick_params(axis="x", labelsize=13)
    if len(seen) > 1:
        ax.legend(handles=[mpatches.Patch(color=colors[v], label=v)
                           for v in seen], fontsize=12, loc="lower right",
                  frameon=True, framealpha=0.95, edgecolor="#AAAAAA")
    panel_shade(ax, grid_axis="x")

    # Explicit margins: tight_layout centres the wide titles over the
    # narrow outer panels and pushes them off the canvas.
    fig.subplots_adjust(left=0.060, right=0.988, top=0.845, bottom=0.155)
    save(fig, OUT, "training_alignment")


if __name__ == "__main__":
    main()
