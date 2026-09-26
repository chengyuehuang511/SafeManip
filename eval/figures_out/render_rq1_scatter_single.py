#!/usr/bin/env python3
"""Single-panel RQ1 conditional scatter (per explicit user decision):
no combined/zoom gridspec, no model legend -- model names annotated at the
points with leader lines where the LIBERO high-success cluster is tight --
one fit per simulator identified in the legend, large type, y-axis labelled
"Violations per triggered event".

SELF-CONTAINED RENDER: the 12 (suite, model) points live in
RQ1_scatter_conditional_combined.csv next to this file, and the style
constants below are copied verbatim from plot_combined/load_both.load_style()
and make_plots.py (2026-09-23). Rendering therefore takes seconds and needs
nothing from the flash share. Only `--recompute` reloads both suites through
the pipeline (minutes) to regenerate the CSV; run it after any monitor or
corpus change, since the CSV does not know it is stale.

Lives here because the original share is full (2026-09-23). Once the share
has space again, fold this back into plot_combined/make_plots_conditional.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "RQ1_scatter_conditional_combined.csv")
PC = ("/path/to/SafeManip/eval/saved_eval_rollouts"
      "/monitor_files/0920/plot_combined")

SUITES = ("RoboCasa", "LIBERO")

# --- copied from plot_combined (load_both.load_style / make_plots.py) -------
MODEL_ORDER = ["grootn15", "grootn16", "openpi_pi0", "openpi_pi05",
               "openvla", "cosmos_policy", "rldx1"]
MODEL_DISPLAY = {
    "grootn15": "GR00T N1.5", "grootn16": "GR00T N1.6",
    "openpi_pi0": r"$\mathbf{\pi_0}$",
    "openpi_pi05": r"$\mathbf{\pi_{0.5}}$",
    "openvla": "OpenVLA", "cosmos_policy": "Cosmos Policy", "rldx1": "RLDX-1",
}
MODEL_COLORS = {  # tab10 (ood-culture-tasks.pdf's saturated family; the
    # Okabe-Ito palette is retired -- RQ1 is the only remaining figure with
    # per-model colours, so this cannot desync another figure)
    "grootn15": "#1F77B4", "grootn16": "#2CA02C", "openpi_pi0": "#D62728",
    "openpi_pi05": "#E377C2", "rldx1": "#FF7F0E", "openvla": "#17BECF",
    "cosmos_policy": "#7F7F7F",
}
SUITE_DISPLAY = {"RoboCasa": "RoboCasa365", "LIBERO": "LIBERO"}
SUITE_MARKERS = {"RoboCasa": "o", "LIBERO": "s"}  # square: reads larger
# than the old triangle at the same s (user 2026-09-24)
SUITE_COLORS = {"RoboCasa": "#444444", "LIBERO": "#8B4513"}  # dark brown
# for the LIBERO fit (user 2026-09-24; the purple was retired)
SUITE_DASHES = {"RoboCasa": (0, (7, 2.5)), "LIBERO": (0, (1.8, 1.8))}


def apply_style():
    # Reference layout style (ref/fig/high quality/ood-culture-tasks.pdf):
    # white background, light dashed grid, full black axes box, and the
    # reference's sans-serif face (user 2026-09-24, reversing the earlier
    # Times New Roman call) -- bold stays reserved for titles, axis labels,
    # and model-name annotations only; ticks stay regular.
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "mathtext.fontset": "dejavusans",
        "axes.labelweight": "bold", "axes.titleweight": "bold",
        "axes.grid": True, "grid.color": "#CCCCCC",
        "grid.linewidth": 0.8, "grid.linestyle": "--",
        "axes.edgecolor": "black", "axes.linewidth": 1.2,
        "axes.axisbelow": True,
    })


def fit_stats(x, y):
    """OLS + Pearson/Spearman; None below 3 points (copied from make_plots)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.ptp(x) == 0:
        return None
    lr = sps.linregress(x, y)
    rho, rho_p = sps.spearmanr(x, y)
    return {"n": len(x), "slope": lr.slope, "intercept": lr.intercept,
            "pearson_r": lr.rvalue, "pearson_p": lr.pvalue,
            "spearman_rho": float(rho), "spearman_p": float(rho_p)}


def _stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else \
        "*" if p < 0.05 else "n.s."


def _fit_label(suite, st):
    star = _stars(st["pearson_p"])
    star = star if star != "n.s." else " n.s."
    return (f"{SUITE_DISPLAY[suite]}: $r$={st['pearson_r']:+.2f}{star} "
            f"($n$={st['n']})")


def recompute_table():
    """Reload both suites through the flash pipeline (minutes) -> DataFrame."""
    sys.path.insert(0, PC)
    from activation_data import episode_table, load_suite  # noqa: E402
    rows = []
    for suite in SUITES:
        df, _meta = load_suite(suite)
        ep = episode_table(df)
        g = (df.groupby("model", observed=True)
             .agg(V=("violations", "sum"), A=("activations", "sum")))
        g["risk"] = np.where(g.A > 0, g.V / g.A, np.nan)
        succ = ep.groupby("model").task_success.mean()
        for m in MODEL_ORDER:
            if m not in g.index:
                continue
            rows.append(dict(suite=suite, model=m, success_rate=succ[m],
                             risk=g.loc[m, "risk"]))
    return pd.DataFrame(rows)


# (dx pt, dy pt, leader?, shrinkB, anchor (ax,ay) pt) fanned out so no label
# touches another point, no leader crosses another leader, and each label
# hugs its own marker. shrinkB is how far the leader stops from the anchor.
# The optional 5th element shifts the leader's ANCHOR off the marker centre,
# in points: LIBERO grootn16 is covered by the rldx1 square, so its leader
# targets the BOTTOM-LEFT VERTEX of the green square (half a side = ~6 pt at
# s=150) instead of the hidden centre (user 2026-09-24).
OFFSETS = {
    ("RoboCasa", "grootn15"): (-9, 0, False),
    ("RoboCasa", "grootn16"): (-10, 6, False),
    ("RoboCasa", "openpi_pi0"): (-13, -4, False),
    ("RoboCasa", "openpi_pi05"): (13, -8, False),
    ("RoboCasa", "rldx1"): (13, 2, False),
    ("LIBERO", "openvla"): (14, 6, False),
    ("LIBERO", "grootn15"): (-13, -4, False),
    ("LIBERO", "openpi_pi0"): (0, 8, False),
    ("LIBERO", "openpi_pi05"): (0, 7, False),
    ("LIBERO", "grootn16"): (-28, -13, True, 1, (-6, -6)),
    # rldx1's label sits directly beside its marker; a very short leader in
    # the model's yellow runs from the MIDDLE OF THE RIGHT EDGE of its square
    # (anchor +6 pt, half a side) to the label (user 2026-09-24).
    ("LIBERO", "rldx1"): (12, 2, True, 1, (6, 0)),
    # cosmos_policy's square is fully visible, but its leader approaches from
    # the lower left, so it too targets the bottom-left vertex (user
    # 2026-09-24) rather than stopping short of the centre.
    ("LIBERO", "cosmos_policy"): (-2, -24, True, 1, (-6, -6)),
}

# GR00T names stack in two lines (kept on the LEFT of their points, per
# explicit user decision) so large labels stay clear of axis and fit lines.
LABEL_TEXT = {
    (s, m): "GR00T\n" + MODEL_DISPLAY[m][-4:]
    for s in SUITES for m in ("grootn15", "grootn16")
}
# LIBERO's grootn16 stays single-line: green is boxed in by yellow (upper
# right) and gray (lower right), so its label must approach from the lower
# left -- the only free band there (between GR00T N1.5's label and Cosmos
# Policy's) is one line tall at this figure height.
del LABEL_TEXT[("LIBERO", "grootn16")]


def main():
    apply_style()
    if "--recompute" in sys.argv or not os.path.exists(CSV):
        table = recompute_table()
        table.to_csv(CSV, index=False)
        print(f"[csv] {CSV}")
    else:
        table = pd.read_csv(CSV)

    stats = {s: fit_stats(table.loc[table.suite == s, "success_rate"] * 100,
                          table.loc[table.suite == s, "risk"])
             for s in SUITES}
    with open(os.path.join(HERE, "RQ1_scatter_fit_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    # White background + dashed gray grid + full box, per the reference
    # figures (ref/fig/high quality) -- replaces the gray-panel treatment.
    for suite in SUITES:
        sub = table[table["suite"] == suite]
        st = stats[suite]
        xs = np.array([sub["success_rate"].min(), sub["success_rate"].max()])
        xs *= 100  # fit drawn across the suite's own span, never extrapolated
        ax.plot(xs, st["intercept"] + st["slope"] * xs,
                color=SUITE_COLORS[suite], linestyle=SUITE_DASHES[suite],
                linewidth=2.0, alpha=0.9, zorder=1)
        # Descending risk so the lowest marker in an overlapping stack is
        # drawn last (on top); rldx1 is then forced above grootn16 (their
        # points nearly coincide -- green keeps its lower-left corner and has
        # its own leader, while yellow would otherwise vanish entirely).
        ordered = sub.sort_values("risk", ascending=False)
        idx = list(ordered["model"])
        if "rldx1" in idx and "grootn16" in idx:
            idx.remove("rldx1")
            idx.insert(idx.index("grootn16") + 1, "rldx1")
        ordered = ordered.set_index("model").loc[idx].reset_index()
        for _, r in ordered.iterrows():
            # White edge = surface ring so the LIBERO high-success cluster's
            # overlapping triangles stay individually readable.
            ax.scatter(r["success_rate"] * 100, r["risk"], s=150,
                       color=MODEL_COLORS[r["model"]],
                       marker=SUITE_MARKERS[suite], edgecolor="white",
                       linewidth=1.4, zorder=3)

    for _, r in table.iterrows():
        off = OFFSETS.get((r["suite"], r["model"]), (12, 6, False))
        dx, dy, leader = off[:3]
        shrink = off[3] if len(off) > 3 else 9
        anchor = off[4] if len(off) > 4 else None
        ha = "center" if dx == 0 else ("left" if dx > 0 else "right")
        va = ("bottom" if dy >= 0 else "top") if dx == 0 else "center"
        kw = dict(textcoords="offset points", xytext=(dx, dy), fontsize=15,
                  fontweight="bold", ha=ha, multialignment="center",
                  linespacing=0.95, va=va, zorder=4)
        if anchor is not None:
            kw["xycoords"] = matplotlib.transforms.offset_copy(
                ax.transData, fig, anchor[0] / 72, anchor[1] / 72,
                units="inches")
        if leader:
            # Leader in the MODEL's colour (user 2026-09-24): the line then
            # names its marker even where the cluster is dense.
            kw["arrowprops"] = dict(arrowstyle="-", lw=1.0,
                                    color=MODEL_COLORS[r["model"]],
                                    shrinkA=0, shrinkB=shrink)
        ax.annotate(LABEL_TEXT.get((r["suite"], r["model"]),
                                   MODEL_DISPLAY[r["model"]]),
                    (r["success_rate"] * 100, r["risk"]), **kw)

    fit_handles = [mlines.Line2D([], [], color=SUITE_COLORS[s],
                                 linestyle=SUITE_DASHES[s], linewidth=2.0,
                                 label=_fit_label(s, stats[s]))
                   for s in SUITES]
    # Framed legend with a plain gray edge, regular-weight text (reference
    # legend grammar; the old all-bold legend text is retired).
    ax.legend(handles=fit_handles, loc="lower left", fontsize=15,
              frameon=True, framealpha=0.95, edgecolor="#AAAAAA")
    # Extra y headroom: at reduced height the offset labels (top GR00T N1.5,
    # bottom Cosmos Policy) otherwise poke past the axes box. The slightly
    # negative floor is label room only -- no data or tick lives below 0.
    ax.margins(x=0.10, y=0.14)
    ax.set_ylim(bottom=-0.005)
    # Extra room on the RIGHT only (user 2026-09-24): the RLDX-1 and Cosmos
    # Policy labels sit right of the LIBERO cluster and were touching the
    # axes box; a symmetric margin bump wasted space on the left instead.
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + 6.0)
    # Sans runs wider than the retired serif: sizes sit between the serif
    # values and the bold-sans ones so nothing clips or collides.
    ax.set_xlabel("Task success rate (%)", fontsize=21, fontweight="bold")
    # Same size as the identical y label in RQ3's conditional figure (user
    # 2026-09-24): the two figures carry the same metric, so its axis label must
    # not change size between them.
    ax.set_ylabel("Violations per triggered event", fontsize=17,
                  fontweight="bold")
    ax.tick_params(labelsize=18)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"RQ1_scatter_conditional_combined.{ext}")
        fig.savefig(out, dpi=300 if ext == "png" else None)
        print(f"[fig] {out}")


if __name__ == "__main__":
    main()
