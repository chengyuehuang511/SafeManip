#!/usr/bin/env python3
"""RQ3 conditional figure: violations per triggered event, one row, two panels.

Replaces the two surviving RQ3 figures (explicit user requests 2026-09-23):

  (a) Task horizon -- was RQ3_horizon_conditional_combined.png's top row, with
      both simulators now on ONE horizon axis. The old bottom row (triggered
      events per episode) is dropped: only the violation rate is reported.
  (b) Task category -- was RQ3_task_suite_conditional_combined.png's two
      heatmaps, now a single bar panel: the bar is the all-episode rate and the
      dark tick across it is the rate among SUCCESSFUL episodes only
      (V_succ / A_succ, the old heatmap (b), drawn where the cell has at least
      MIN_SUCC successful episodes). Overlaying instead of stacking is what the
      "among successes" metric allows -- it is a different conditioning of the
      same ratio, not a part of it.

Style follows render_rq2_conditional.py: task counts in the tick labels,
categories sorted by RoboCasa count, bold panel titles, plain ticks. No x10^-2
rescaling, unlike RQ2: nothing is printed in a cell here, so the axis can carry
the magnitude itself. Model = colour; the two suites are separate blocks inside each group,
split by a gap, LIBERO outlined -- no hatching.

LIBERO horizons: the pipeline only ever had a coarse Short/Long split for
LIBERO (Short = spatial/object/goal, Long = libero_10/90), which is not
comparable to RoboCasa's tiers. Its tasks are re-binned onto RoboCasa's own
tier rule from total_primitives -- see TIER_RULE. LIBERO tops out at 4
primitives, so it has no Long tier; that is a finding, not a gap.

LIBERO task categories: ground truth from
libero_task_skill_sequences_categorized.csv (user-supplied 2026-09-23), which
carries the RoboCasa task-suite taxonomy per LIBERO task directly.

Caching follows render_rq2_conditional.py: episode-level sums live in
RQ3_conditional_episodes.csv next to this file, so a plain run renders in
seconds. `--recompute` reloads both suites through the pipeline (minutes).
`--validate` checks the estimators against cells read off the old PNGs.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects  # noqa: F401  (registers the submodule)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "RQ3_conditional_episodes.csv")
LB_SKILLS = os.path.join(HERE, "libero_task_skill_sequences_categorized.csv")
PC = ("/path/to/SafeManip/eval/saved_eval_rollouts"
      "/monitor_files/0920/plot_combined")

SUITES = ("RoboCasa", "LIBERO")
SUITE_DISPLAY = {"RoboCasa": "RoboCasa365", "LIBERO": "LIBERO"}
MODEL_ORDER = ["grootn15", "grootn16", "openpi_pi0", "openpi_pi05",
               "openvla", "cosmos_policy", "rldx1"]
MODEL_DISPLAY = {
    "grootn15": "GR00T N1.5", "grootn16": "GR00T N1.6",
    "openpi_pi0": r"$\pi_0$", "openpi_pi05": r"$\pi_{0.5}$",
    "openvla": "OpenVLA", "cosmos_policy": "Cosmos", "rldx1": "RLDX-1",
}
MODEL_COLORS = {  # plot_libero/style.py, verbatim (Okabe-Ito)
    "grootn15": "#0072B2", "grootn16": "#009E73", "openpi_pi0": "#D55E00",
    "openpi_pi05": "#CC79A7", "openvla": "#56B4E9",
    "cosmos_policy": "#666666", "rldx1": "#E69F00",
}
TIER_ORDER = ["Atomic", "Short", "Medium", "Long"]
CATEGORY_ORDER = ["AtomicFixture", "StorageOrganization",
                  "CookingIngredientPreparation", "BeveragePreparationServing",
                  "BreadBreakfastReheating", "CleaningWashingSanitation",
                  "PlatingServingPortioning"]
# Tick names exactly as the surviving RQ3 heatmap printed them, wrapped so that
# seven of them fit across one panel.
CATEGORY_DISPLAY = {
    "AtomicFixture": "Atomic/\nFixture",
    "BeveragePreparationServing": "Beverage\nPreparation/\nServing",
    "BreadBreakfastReheating": "Bread/\nBreakfast\nReheating",
    "CleaningWashingSanitation": "Cleaning/\nWashing\nSanitation",
    "CookingIngredientPreparation": "Cooking/\nIngredient\nPreparation",
    "PlatingServingPortioning": "Plating/\nServing\nPortioning",
    "StorageOrganization": "Storage/\nOrganization",
}
# The categorized LIBERO CSV spells categories the way the figure prints them.
GT_CATEGORY_KEY = {
    "Atomic/Fixture": "AtomicFixture",
    "Beverage Preparation/Serving": "BeveragePreparationServing",
    "Cooking/Ingredient Preparation": "CookingIngredientPreparation",
    "Plating/Serving/Portioning": "PlatingServingPortioning",
    "Storage/Organization": "StorageOrganization",
}
# Simulator colours, the exact paired-bar grammar of
# ref/fig/high quality/ood-culture-tasks.pdf (user 2026-09-24): the ALL-
# rollouts bar is the LIGHT solid (like the reference's Vanilla), and the
# successful-only bar is the SATURATED fill with a white "//" hatch (like the
# reference's CARE-aligned). Pairs from tab20: red for RoboCasa, blue LIBERO.
SUITE_COLORS = {"RoboCasa": "#D62728", "LIBERO": "#1F77B4"}
SUITE_LIGHT = {"RoboCasa": "#FF9896", "LIBERO": "#AEC7E8"}
SUCC_HATCH = "//"
# Pair geometry (in units of one dark-bar width): equal-width pair members
# touching, like the reference's Vanilla/aligned pairs; the suite gap alone
# structures the group (pair < suite gap < group gap).
LIGHT_W, PAIR_GAP, SUITE_GAP = 1.0, 0.0, 0.30
PANEL_BG = "white"     # reference style: white panels, dashed gray grid
MIN_SUCC = 5      # the old heatmap (b)'s n_succ >= 5 rule
# The among-successes tick: yellow with NO outline (user 2026-09-24 -- the
# black halo read as the mark being truncated at its ends). Gold is light
# enough to pop on both saturated bar colours and still prints on white.
SUCC_COLOR = "#FFD700"
SUCC_FX = None


def apply_style():
    # Reference layout style (ood-culture-tasks.pdf) with the reference's
    # sans-serif face (user 2026-09-24, reversing the earlier Times New
    # Roman call): white panels, light dashed grid, full black box; bold
    # only on titles and axis/row labels, ticks regular.
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "mathtext.fontset": "dejavusans",
        "axes.labelweight": "bold", "axes.titleweight": "bold",
        "axes.edgecolor": "black", "axes.linewidth": 1.2,
        "axes.axisbelow": True,
        "hatch.linewidth": 1.8,
    })


# RoboCasa's own horizon rule, read straight off analysis/taskDiff.csv
# (numberSubtask -> difficulty_horizon): 1 Atomic, 2 Short, 3-4 Medium,
# >=5 Long. Applied unchanged to LIBERO's total_primitives so the two suites
# land on ONE axis.
def TIER_RULE(n):
    if n is None or not np.isfinite(n):
        return None
    n = int(n)
    return ("Atomic" if n <= 1 else "Short" if n == 2 else
            "Medium" if n <= 4 else "Long")


def libero_meta():
    """LIBERO task -> (tier, n_primitives, RoboCasa task category).

    The CSV spells task names with spaces, the monitor corpus with
    underscores; join on the spaced form."""
    d = pd.read_csv(LB_SKILLS)
    key = d["task"].astype(str)
    return (dict(zip(key, d["total_primitives"].map(TIER_RULE))),
            dict(zip(key, d["total_primitives"])),
            dict(zip(key, d["category"].map(GT_CATEGORY_KEY))))


def recompute():
    """Episode-level sums + task metadata for both suites (minutes)."""
    sys.path.insert(0, PC)
    from activation_data import episode_table, load_suite  # noqa: E402
    lb_tier, lb_prim, lb_cat = libero_meta()
    parts = []
    for suite in SUITES:
        df, meta = load_suite(suite)
        ep = episode_table(df)
        # Episode exposure D = sum over the episode's applicable properties of
        # violation_duration / num_frames -- RQ2's quantity, summed to episode
        # level so D/A pools exactly like V/A.
        df = df.assign(duration_frac=df["violation_duration"]
                       / df["num_frames"].replace(0, np.nan))
        d = (df.groupby(["model", "task", "episode"], observed=True)
             ["duration_frac"].sum().rename("D").reset_index())
        ep = ep.merge(d, on=["model", "task", "episode"], how="left")
        ep.insert(0, "suite", suite)
        if suite == "RoboCasa":
            ep["category"] = ep["task"].map(meta["suite_map"])
            ep["tier"] = ep["task"].map(meta["horizon_map"])
            ep["n_prim"] = np.nan
        else:
            spaced = ep["task"].str.replace("_", " ", regex=False)
            ep["category"] = spaced.map(lb_cat)
            ep["tier"] = spaced.map(lb_tier)
            ep["n_prim"] = spaced.map(lb_prim)
            for col in ("tier", "category"):
                miss = sorted(set(ep.loc[ep[col].isna(), "task"]))
                print(f"[libero] {ep.task.nunique()} eval tasks, "
                      f"{len(miss)} without a {col}")
                for t in miss:
                    print(f"  UNMAPPED {col}: {t!r}")
        parts.append(ep)
    out = pd.concat(parts, ignore_index=True)
    return out.rename(columns={"violations": "V", "act_total": "A"})[
        ["suite", "model", "task", "episode", "category", "tier", "n_prim",
         "task_success", "V", "A", "D"]]


# ---------------------------------------------------------------------------
# Estimators. Same pooled ratio-of-sums as RQ2: total violations over total
# triggered events in the group, so frequently-triggering tasks are not
# up-weighted by an episode count. `succ` re-runs the same ratio over the
# successful episodes only (the old heatmap (b)), suppressed below MIN_SUCC
# successes so a 1-success cell cannot show a spike.
# ---------------------------------------------------------------------------
def rate(ep, by, pool=False, num="V"):
    """pool=True drops the model split (user 2026-09-24: separating by model is
    not necessary here), pooling every policy's numerator and triggered events
    into one ratio per (suite, group). Still a ratio of sums, so the pooled
    value is the suite's overall rate, not a mean of per-policy rates.

    num="V" -> violations per triggered event; num="D" -> exposure per
    triggered event (fraction of rollout, RQ2's panel-(b) quantity)."""
    keys = ["suite"] + ([] if pool else ["model"]) + by
    tot = ep.groupby(keys, observed=True)[[num, "A"]].sum()
    out = pd.DataFrame({"total": tot[num] / tot.A.replace(0, np.nan)})
    s = (ep[ep.task_success == 1].groupby(keys, observed=True)
         .agg(num=(num, "sum"), A=("A", "sum"), n=("episode", "size"))
         .reindex(tot.index))
    out["succ"] = (s.num / s.A.replace(0, np.nan)).where(s.n >= MIN_SUCC)
    return out


def events(ep, by):
    """Triggered events per episode. Not plotted any more;
    kept because `--validate` uses it to check this file's loader against the
    old figure's bottom row."""
    g = ep.groupby(["suite", "model"] + by, observed=True).agg(
        A=("A", "sum"), n=("episode", "size"))
    return pd.DataFrame({"total": g.A / g.n.replace(0, np.nan)})


def counts(ep, by):
    """(RoboCasa, LIBERO) task counts per group, for the tick labels."""
    n = ep.pivot_table(index=by[0], columns="suite", values="task",
                       aggfunc="nunique")

    def cnt(k, suite):
        v = n.loc[k, suite] if (k in n.index and suite in n.columns) else np.nan
        return "–" if not np.isfinite(v) else f"{int(v)}"

    return {k: (cnt(k, "RoboCasa"), cnt(k, "LIBERO")) for k in n.index}


def validate(ep):
    """Cells read off the two surviving RQ3 PNGs -- the LIBERO category rows
    double as a check on the ground-truth category join. The LIBERO horizon
    rows are NOT checked: the old figure used the coarse suite-derived
    Short/Long, which this figure deliberately replaces."""
    r_t, e_t = rate(ep, ["tier"]), events(ep, ["tier"])["total"]
    r_c = rate(ep, ["category"])
    refs = [
        ("RoboCasa", "grootn15", "Atomic", r_t["total"], 0.036),
        ("RoboCasa", "grootn15", "Short", r_t["total"], 0.0737),
        ("RoboCasa", "grootn15", "Medium", r_t["total"], 0.0566),
        ("RoboCasa", "grootn15", "Long", r_t["total"], 0.0995),
        ("RoboCasa", "grootn15", "Atomic", e_t, 8.3),
        ("RoboCasa", "grootn15", "Long", e_t, 34.8),
        ("RoboCasa", "grootn15", "AtomicFixture", r_c["total"], 0.036),
        ("RoboCasa", "rldx1", "StorageOrganization", r_c["total"], 0.088),
        ("LIBERO", "grootn15", "AtomicFixture", r_c["total"], 0.050),
        ("LIBERO", "grootn15", "BeveragePreparationServing",
         r_c["total"], 0.011),
        ("LIBERO", "grootn15", "PlatingServingPortioning", r_c["total"], 0.027),
        ("LIBERO", "grootn15", "StorageOrganization", r_c["total"], 0.014),
        ("LIBERO", "openvla", "StorageOrganization", r_c["total"], 0.048),
        # among successes -- the old figure's right-hand heatmap
        ("RoboCasa", "grootn15", "AtomicFixture", r_c["succ"], 0.025),
        ("RoboCasa", "grootn15", "PlatingServingPortioning",
         r_c["succ"], 0.118),
        ("LIBERO", "openvla", "StorageOrganization", r_c["succ"], 0.022),
        ("LIBERO", "grootn15", "AtomicFixture", r_c["succ"], 0.000),
    ]
    bad = 0
    for suite, m, key, series, ref in refs:
        try:
            v = float(series.loc[(suite, m, key)])
        except KeyError:
            v = np.nan
        ok = np.isfinite(v) and abs(v - ref) <= max(0.0015, 0.03 * ref)
        bad += not ok
        print(f"  {'ok ' if ok else 'BAD'} {suite:8s} {m:13s} {key:28s} "
              f"got {v:8.3f}  png {ref}")
    print(f"[validate] {len(refs) - bad}/{len(refs)} match")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def bar_positions(cols, span=0.86, gap=0.55):
    """x offsets within a group: the RoboCasa models, a `gap`-wide bar of empty
    space, then the LIBERO models. The gap, not a hatch, is what separates the
    suites (user 2026-09-23: the hatching was ugly)."""
    n_rc = sum(1 for s, _ in cols if s == "RoboCasa")
    if n_rc in (0, len(cols)):   # single-suite axes (the faceted variant)
        gap = 0.0
    slots = len(cols) + gap
    w = span / slots
    out, k = [], 0.0
    for j in range(len(cols)):
        if j == n_rc and gap:
            k += gap
        out.append(-span / 2 + w * (k + 0.5))
        k += 1.0
    return out, w


def group_bars(ax, tab, keys, cols, scale=1.0, succ=False):
    offs, w = bar_positions(cols)
    for (suite, m), off in zip(cols, offs):
        c = MODEL_COLORS[m]
        x = np.array([i + off for i in range(len(keys))], float)

        def col(field):
            return np.array([
                (lambda v: v * scale if np.isfinite(v) else np.nan)(
                    float(tab.loc[(suite, m, k), field])
                    if (suite, m, k) in tab.index else np.nan)
                for k in keys])

        # LIBERO gets a dark outline, RoboCasa none -- the only difference in
        # fill, so the model colours stay fully saturated in both suites.
        ax.bar(x, np.nan_to_num(col("total")), width=w * 0.9, color=c,
               edgecolor="none" if suite == "RoboCasa" else "#1A1A1A",
               linewidth=0.0 if suite == "RoboCasa" else 0.9)
        if succ:
            v, t = col("succ"), col("total")
            ok = np.isfinite(v)
            # Where the success-only rate is HIGHER than the all-episode one the
            # tick floats above its bar; a hairline stem keeps it attached to it.
            up = ok & (v > np.nan_to_num(t))
            ax.vlines(x[up], np.nan_to_num(t)[up], v[up], color="#555555",
                      linewidth=0.7, linestyle=(0, (1, 1.2)), zorder=4)
            ax.hlines(v[ok], x[ok] - w * 0.45, x[ok] + w * 0.45,
                      color=SUCC_COLOR, linewidth=1.7, zorder=5)
    ax.set_xticks(range(len(keys)))
    ax.set_xlim(-0.5, len(keys) - 0.5)
    ax.grid(axis="y", color="#D5D5D5", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def group_dumbbell(ax, tab, keys, cols, scale=1.0, succ=True):
    """Panel-(b) alternative: one dumbbell per series instead of a bar -- a
    stem from the among-successes rate to the all-episode rate, an open marker
    at the former and a filled one at the latter. Same information as the
    bar+tick, far less ink, and the length of the stem IS the success discount.
    """
    offs, w = bar_positions(cols)
    for (suite, m), off in zip(cols, offs):
        c = MODEL_COLORS[m]
        x = np.array([i + off for i in range(len(keys))], float)
        t = np.array([float(tab.loc[(suite, m, k), "total"])
                      if (suite, m, k) in tab.index else np.nan
                      for k in keys]) * scale
        v = np.array([float(tab.loc[(suite, m, k), "succ"])
                      if (suite, m, k) in tab.index else np.nan
                      for k in keys]) * scale
        both = np.isfinite(t) & np.isfinite(v)
        ax.vlines(x[both], np.minimum(t, v)[both], np.maximum(t, v)[both],
                  color=c, linewidth=2.4, alpha=0.55, zorder=3)
        ok = np.isfinite(t)
        ax.plot(x[ok], t[ok], marker="o", linestyle="none", markersize=6.0,
                color=c, markeredgecolor="#1A1A1A" if suite == "LIBERO"
                else "none", markeredgewidth=0.8, zorder=5)
        ok = np.isfinite(v)
        ax.plot(x[ok], v[ok], marker="o", linestyle="none", markersize=5.5,
                markerfacecolor="white", markeredgecolor=c,
                markeredgewidth=1.6, zorder=4)
    ax.set_xticks(range(len(keys)))
    ax.set_xlim(-0.5, len(keys) - 0.5)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#D5D5D5", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def group_hbars(ax, tab, keys, cols, scale=1.0, succ=True):
    """Panel-(b) alternative: the same bars rotated. Categories read left to
    right along the y axis, so the seven long names need no wrapping, and the
    12 series stack downwards instead of crowding sideways."""
    offs, w = bar_positions(cols)
    for (suite, m), off in zip(cols, offs):
        c = MODEL_COLORS[m]
        y = np.array([i - off for i in range(len(keys))], float)
        t = np.array([float(tab.loc[(suite, m, k), "total"])
                      if (suite, m, k) in tab.index else np.nan
                      for k in keys]) * scale
        v = np.array([float(tab.loc[(suite, m, k), "succ"])
                      if (suite, m, k) in tab.index else np.nan
                      for k in keys]) * scale
        ax.barh(y, np.nan_to_num(t), height=w * 0.9, color=c,
                edgecolor="none" if suite == "RoboCasa" else "#1A1A1A",
                linewidth=0.0 if suite == "RoboCasa" else 0.9)
        ok = np.isfinite(v)
        right = ok & (v > np.nan_to_num(t))
        ax.hlines(y[right], np.nan_to_num(t)[right], v[right], color="#555555",
                  linewidth=0.7, linestyle=(0, (1, 1.2)), zorder=4)
        ax.vlines(v[ok], y[ok] - w * 0.45, y[ok] + w * 0.45, color=SUCC_COLOR,
                  linewidth=1.7, zorder=5)
    ax.set_yticks(range(len(keys)))
    ax.set_ylim(len(keys) - 0.5, -0.5)
    ax.grid(axis="x", color="#D5D5D5", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# Macaron red, copied from plot_combined/activation_data.py (the registered
# colormap the RQ2 conditional heatmaps use), so the split-cell variant below
# is stylistically continuous with them.
MACARON_RED = ["#FDF5F3", "#FBDCD5", "#F6BBAE", "#EE9184",
               "#DE6560", "#BB3F4A", "#8C2B39"]


def group_split_heatmap(ax, tab, keys, cols, scale=1.0, succ=True):
    """Panel-(b) alternative in the RQ2 heatmap family: ONE grid of cells, each
    cut on the diagonal -- upper-left triangle = all episodes, lower-right =
    among successes. That is literally the two old heatmaps superimposed, and it
    scales to 12 columns without any of the bars' ink."""
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "macaron_red", MACARON_RED)
    vals = np.array([[tab.loc[(s, m, k), f] if (s, m, k) in tab.index else np.nan
                      for f in ("total", "succ")] for k in keys
                     for s, m in cols], float) * scale
    norm = matplotlib.colors.PowerNorm(0.6, vmin=0,
                                       vmax=np.nanpercentile(vals, 97))
    for i, k in enumerate(keys):
        for j, (s, m) in enumerate(cols):
            for field, tri in (("total", [(-.5, -.5), (.5, -.5), (-.5, .5)]),
                               ("succ", [(.5, -.5), (.5, .5), (-.5, .5)])):
                v = (float(tab.loc[(s, m, k), field]) * scale
                     if (s, m, k) in tab.index else np.nan)
                fc = "#DDDDDD" if not np.isfinite(v) else cmap(norm(v))
                ax.add_patch(matplotlib.patches.Polygon(
                    [(j + dx, i + dy) for dx, dy in tri], closed=True,
                    facecolor=fc, edgecolor="white", linewidth=0.6))
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(len(keys) - 0.5, -0.5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([MODEL_DISPLAY[m] for _, m in cols], rotation=45,
                       ha="right", rotation_mode="anchor", fontsize=15)
    ax.set_yticks(range(len(keys)))
    ax.grid(False)
    n_rc = sum(1 for s, _ in cols if s == "RoboCasa")
    ax.axvline(n_rc - 0.5, color="black", linewidth=1.6)
    for lo, hi, name in ((0, n_rc, "RoboCasa"), (n_rc, len(cols), "LIBERO")):
        ax.text((lo + hi) / 2 / len(cols), 1.01, SUITE_DISPLAY[name],
                transform=ax.transAxes, ha="center", va="bottom", fontsize=17)
    return matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)


def group_lines(ax, tab, keys, cols, scale=1.0, succ=True):
    """Panel-(b) alternative: a profile line per series across the categories,
    filled markers = all episodes, open markers = among successes. Compact, but
    the connecting lines imply an order the categories do not have -- they are
    sorted by applicable-task count, nothing more."""
    for s, m in cols:
        c = MODEL_COLORS[m]
        x = np.arange(len(keys), dtype=float)
        t = np.array([float(tab.loc[(s, m, k), "total"])
                      if (s, m, k) in tab.index else np.nan
                      for k in keys]) * scale
        v = np.array([float(tab.loc[(s, m, k), "succ"])
                      if (s, m, k) in tab.index else np.nan
                      for k in keys]) * scale
        ax.plot(x, t, color=c, linewidth=1.6,
                linestyle="-" if s == "RoboCasa" else (0, (3, 1.4)),
                marker="o" if s == "RoboCasa" else "s", markersize=5.5,
                markeredgecolor="none", zorder=3)
        ok = np.isfinite(v)
        ax.plot(x[ok], v[ok], linestyle="none",
                marker="o" if s == "RoboCasa" else "s", markersize=5.5,
                markerfacecolor="white", markeredgecolor=c,
                markeredgewidth=1.4, zorder=4)
    ax.set_xticks(range(len(keys)))
    ax.set_xlim(-0.4, len(keys) - 0.6)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#D5D5D5", linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# Category palette for the transposed panels, where colour carries the task
# category instead of the model (seaborn "deep" -- deliberately NOT the
# Okabe-Ito model palette, so the two encodings can never be confused).
CATEGORY_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
                   "#937860", "#DA8BC3"]


def policy_panel(ax, tab, keys, cols, scale=1.0, mode="dumbbell"):
    """Panel-(b) transposed: the policies (x, blocked by simulator) carry the
    comparison and the task category becomes colour.

    mode="dumbbell": stem from the among-successes rate to the all-episode rate.
    mode="delta":    a signed bar of (all - successes), i.e. how much of the
                     violation rate succeeding buys back. Above zero = the
                     violations sit in the failures; below zero = the policy
                     violates MORE on the episodes it completes.
    """
    # Group centres: one per policy, with a gap between the two simulators.
    n_rc = sum(1 for s, _ in cols if s == "RoboCasa")
    centres = np.array([j + (0.8 if j >= n_rc else 0.0)
                        for j in range(len(cols))], float)
    span, n = 0.88, len(keys)
    w = span / n
    for i, k in enumerate(keys):
        c = CATEGORY_COLORS[i % len(CATEGORY_COLORS)]
        off = -span / 2 + w * (i + 0.5)
        x = centres + off
        t = np.array([float(tab.loc[(s, m, k), "total"])
                      if (s, m, k) in tab.index else np.nan
                      for s, m in cols]) * scale
        v = np.array([float(tab.loc[(s, m, k), "succ"])
                      if (s, m, k) in tab.index else np.nan
                      for s, m in cols]) * scale
        if mode == "delta":
            d = t - v
            ok = np.isfinite(d)
            ax.bar(x[ok], d[ok], width=w * 0.88, color=c, linewidth=0.0)
        else:
            both = np.isfinite(t) & np.isfinite(v)
            ax.vlines(x[both], np.minimum(t, v)[both], np.maximum(t, v)[both],
                      color=c, linewidth=2.2, alpha=0.6, zorder=3)
            ok = np.isfinite(t)
            ax.plot(x[ok], t[ok], marker="o", linestyle="none", markersize=5.8,
                    color=c, markeredgecolor="none", zorder=5)
            ok = np.isfinite(v)
            ax.plot(x[ok], v[ok], marker="o", linestyle="none", markersize=5.4,
                    markerfacecolor="white", markeredgecolor=c,
                    markeredgewidth=1.5, zorder=4)
    if mode == "delta":
        ax.axhline(0, color="#333333", linewidth=1.0)
    else:
        ax.set_ylim(bottom=0)
    ax.set_xticks(centres)
    ax.set_xticklabels([MODEL_DISPLAY[m] for _, m in cols], rotation=30,
                       ha="right", rotation_mode="anchor", fontsize=16)
    ax.set_xlim(centres[0] - 0.7, centres[-1] + 0.7)
    ax.grid(axis="y", color="#D5D5D5", linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    edge = (centres[n_rc - 1] + centres[n_rc]) / 2
    ax.axvline(edge, color="#333333", linewidth=1.0, linestyle=(0, (4, 3)))
    for lo, hi, name in ((centres[0], centres[n_rc - 1], "RoboCasa"),
                         (centres[n_rc], centres[-1], "LIBERO")):
        ax.text((lo + hi) / 2, 1.012, SUITE_DISPLAY[name], ha="center",
                va="bottom", fontsize=17,
                transform=ax.get_xaxis_transform())


def tick_labels(keys, cnts, display=None):
    out = []
    for k in keys:
        rc, lb = cnts.get(k, ("–", "–"))
        out.append(f"{display[k] if display else k}\n({rc}/{lb})")
    return out


def svals(tab, suite, keys, field, scale=1.0):
    """Column of a model-pooled table, NaN where the group is absent."""
    return np.array([float(tab.loc[(suite, k), field])
                     if (suite, k) in tab.index else np.nan
                     for k in keys]) * scale


def suite_bars(ax, tab, keys, scale=1.0, succ=False, span=0.66, mode="bars",
               per_model=None):
    """Model-pooled panel: one series per simulator, colour = simulator.

    mode="bars"     grouped bars; among-successes is its OWN lighter bar next
                    to its suite's all-episode bar (user 2026-09-24: the
                    in-bar tick read as the bar being split into parts)
    mode="lollipop" the same numbers as stem+dot: same comparison, a tenth of
                    the ink, and the ticks stop fighting the bar fill
    mode="strip"    bars plus one small dot per POLICY, so the reader can see
                    how much disagreement the pooling hides (needs per_model)
    mode="dumbbell" all-episode dot to among-successes dot, stem = the discount
    mode="slope"    the two simulators on ONE x slot joined by a line, so the
                    RoboCasa-vs-LIBERO gap is read as a slope, not a pair
    """
    suites = [s for s in SUITES if any((s, k) in tab.index for k in keys)]
    w = span / len(suites)
    top = 0.0
    if mode == "slope":
        xs = np.arange(len(keys), dtype=float)
        vals = {s: svals(tab, s, keys, "total", scale) for s in suites}
        if len(suites) == 2:
            a, b = (vals[s] for s in suites)
            both = np.isfinite(a) & np.isfinite(b)
            ax.vlines(xs[both], np.minimum(a, b)[both], np.maximum(a, b)[both],
                      color="#999999", linewidth=1.6, zorder=2)
        for s in suites:
            v = vals[s]
            ok = np.isfinite(v)
            ax.plot(xs[ok], v[ok], marker="o", linestyle="none", markersize=11,
                    color=SUITE_COLORS[s], zorder=4)
            top = max(top, np.nanmax(np.append(v, 0.0)))
        if succ:
            for s in suites:
                v = svals(tab, s, keys, "succ", scale)
                ok = np.isfinite(v)
                ax.plot(xs[ok], v[ok], marker="o", linestyle="none",
                        markersize=10, markerfacecolor="white",
                        markeredgecolor=SUITE_COLORS[s], markeredgewidth=2.0,
                        zorder=5)
                top = max(top, np.nanmax(np.append(v, 0.0)))
    else:
        # In bars mode with succ, each suite is a dark all-episode bar with a
        # NARROWER light among-successes sibling hugged against it (the light
        # bar is a conditional re-read of the same metric, not a fourth primary
        # series); the gap between the two suites is wider than the one inside
        # a pair, so pairing is read before benchmark.
        paired = succ and mode == "bars"
        if paired:
            units = (len(suites) * (1.0 + PAIR_GAP + LIGHT_W)
                     + (len(suites) - 1) * SUITE_GAP)
            w = span / units
        for j, suite in enumerate(suites):
            c = SUITE_COLORS[suite]
            if paired:
                k = j * (1.0 + PAIR_GAP + LIGHT_W + SUITE_GAP)
                dark_off = -span / 2 + w * (k + 0.5)
                light_off = -span / 2 + w * (k + 1.0 + PAIR_GAP + LIGHT_W / 2)
                x = np.array([i + dark_off for i in range(len(keys))], float)
            else:
                x = np.array([i - span / 2 + w * (j + 0.5)
                              for i in range(len(keys))], float)
            t = svals(tab, suite, keys, "total", scale)
            v = svals(tab, suite, keys, "succ", scale)
            top = max(top, np.nanmax(np.append(t, 0.0)))
            if mode in ("bars", "strip"):
                # ood-culture-tasks grammar: all rollouts = LIGHT solid,
                # successful only = SATURATED with a white hatch.
                ax.bar(x, np.nan_to_num(t), width=w * (1.0 if paired
                                                       else 0.88),
                       color=SUITE_LIGHT[suite] if paired else c,
                       edgecolor="none" if paired else "white",
                       linewidth=0.0 if paired else 0.7)
                if paired:
                    top = max(top, np.nanmax(np.append(v, 0.0)))
                    ax.bar(x - dark_off + light_off, np.nan_to_num(v),
                           width=w * LIGHT_W, color=c,
                           hatch=SUCC_HATCH, edgecolor="white",
                           linewidth=0.0)
            elif mode == "lollipop":
                ok = np.isfinite(t)
                ax.vlines(x[ok], 0, t[ok], color=c, linewidth=2.6, zorder=3)
                ax.plot(x[ok], t[ok], marker="o", linestyle="none",
                        markersize=9.5, color=c, zorder=4)
            elif mode == "dumbbell":
                both = np.isfinite(t) & np.isfinite(v)
                ax.vlines(x[both], np.minimum(t, v)[both],
                          np.maximum(t, v)[both], color=c, linewidth=2.6,
                          alpha=0.55, zorder=3)
                ok = np.isfinite(t)
                ax.plot(x[ok], t[ok], marker="o", linestyle="none",
                        markersize=9.5, color=c, zorder=5)
                ok = np.isfinite(v)
                ax.plot(x[ok], v[ok], marker="o", linestyle="none",
                        markersize=9.0, markerfacecolor="white",
                        markeredgecolor=c, markeredgewidth=2.0, zorder=4)
                top = max(top, np.nanmax(np.append(v, 0.0)))
            if mode == "strip" and per_model is not None:
                # Fixed (not random) offsets inside the bar: reproducible, and
                # a policy always sits in the same slot across groups.
                ms = [m for m in MODEL_ORDER
                      if any((suite, m, k) in per_model.index for k in keys)]
                for q, m in enumerate(ms):
                    pv = np.array([float(per_model.loc[(suite, m, k), "total"])
                                   if (suite, m, k) in per_model.index
                                   else np.nan for k in keys]) * scale
                    ok = np.isfinite(pv)
                    dx = (-0.30 + 0.60 * (q + 0.5) / len(ms)) * w
                    ax.plot(x[ok] + dx, pv[ok], marker="o", linestyle="none",
                            markersize=4.6, markerfacecolor="white",
                            markeredgecolor="#1A1A1A", markeredgewidth=0.9,
                            zorder=6)
                    top = max(top, np.nanmax(np.append(pv, 0.0)))
            if succ and mode in ("strip", "lollipop"):
                # hlines do not autoscale, and an among-successes rate can
                # exceed its bar (RoboCasa Plating) -- grow the axis or the tick
                # lands outside it.
                top = max(top, np.nanmax(np.append(v, 0.0)))
                ok = np.isfinite(v)
                up = ok & (v > np.nan_to_num(t))
                ax.vlines(x[up], np.nan_to_num(t)[up], v[up], color="#555555",
                          linewidth=0.8, linestyle=(0, (1, 1.2)), zorder=4)
                ax.hlines(v[ok], x[ok] - w * 0.44, x[ok] + w * 0.44,
                          color=SUCC_COLOR, linewidth=2.6, zorder=7,
                          path_effects=SUCC_FX)
    for b in range(len(keys) - 1):
        ax.axvline(b + 0.5, color="#C4C4C4", linewidth=0.8,
                   linestyle=(0, (4, 3)), zorder=1.5)
    ax.set_xticks(range(len(keys)))
    ax.set_xlim(-0.5, len(keys) - 0.5)
    ax.set_ylim(0, max(ax.get_ylim()[1], top * 1.08))
    panel_shade(ax, grid_axis="y")


def panel_shade(ax, grid_axis):
    """Reference panel treatment (ref/fig/high quality): white background,
    light dashed gridlines, full black axes box."""
    ax.set_facecolor(PANEL_BG)
    ax.grid(False)
    ax.grid(axis=grid_axis, color="#CCCCCC", linewidth=0.8, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("black")
        s.set_linewidth(1.2)


def suite_hbars(ax, tab, keys, scale=1.0, succ=False, span=0.66):
    """Horizontal twin: groups down the y axis, so the seven category names are
    read as plain left-to-right text instead of three stacked lines."""
    suites = [s for s in SUITES if any((s, k) in tab.index for k in keys)]
    if succ:  # same pair geometry as the vertical bars
        h = span / (len(suites) * (1.0 + PAIR_GAP + LIGHT_W)
                    + (len(suites) - 1) * SUITE_GAP)
    else:
        h = span / len(suites)
    right = 0.0
    for j, suite in enumerate(suites):
        c = SUITE_COLORS[suite]
        k = j * ((1.0 + PAIR_GAP + LIGHT_W + SUITE_GAP) if succ else 1.0)
        # First key at the TOP, matching the vertical panels' left-to-right.
        y = np.array([(len(keys) - 1 - i) + span / 2 - h * (k + 0.5)
                      for i in range(len(keys))], float)
        t = svals(tab, suite, keys, "total", scale)
        v = svals(tab, suite, keys, "succ", scale)
        # ood-culture-tasks grammar: all rollouts = LIGHT solid, successful
        # only = SATURATED fill with a white hatch (hatch colour = edgecolor).
        ax.barh(y, np.nan_to_num(t), height=h * (1.0 if succ else 0.88),
                color=SUITE_LIGHT[suite] if succ else c,
                edgecolor="white" if not succ else "none",
                linewidth=0.7 if not succ else 0.0)
        right = max(right, np.nanmax(np.append(t, 0.0)))
        if succ:
            right = max(right, np.nanmax(np.append(v, 0.0)))
            ax.barh(y - h * (0.5 + PAIR_GAP + LIGHT_W / 2),
                    np.nan_to_num(v), height=h * LIGHT_W,
                    color=c, hatch=SUCC_HATCH,
                    edgecolor="white", linewidth=0.0)
    # Tiny dashed divider on each group boundary (user 2026-09-24): one more
    # cue that the four bars between two dashes belong together.
    for b in range(len(keys) - 1):
        ax.axhline(b + 0.5, color="#C4C4C4", linewidth=0.8,
                   linestyle=(0, (4, 3)), zorder=1.5)
    ax.set_yticks([len(keys) - 1 - i for i in range(len(keys))])
    ax.set_ylim(-0.5, len(keys) - 0.5)
    ax.set_xlim(0, max(ax.get_xlim()[1], right * 1.08))
    panel_shade(ax, grid_axis="x")


def pooled_legend(fig, mode):
    """One row under both panels (user 2026-09-24: inside a panel looked odd).
    A single row costs little height and never lands on the data, whichever
    variant is drawn."""
    if mode in ("bars", "hbars"):
        # ood-culture-tasks legend grammar (user 2026-09-24): one FRAMED box
        # with a light gray edge, one entry per suite x conditioning pair --
        # "<suite> (all)" light solid, "<suite> (successful only)" saturated
        # with the white hatch -- replacing the two-section titled strip.
        handles = []
        for s in SUITES:
            handles.append(Patch(
                facecolor=SUITE_LIGHT[s], edgecolor="#888888", linewidth=0.5,
                label=f"{SUITE_DISPLAY[s]} (all rollouts)"))
            handles.append(Patch(
                facecolor=SUITE_COLORS[s], hatch=SUCC_HATCH,
                edgecolor="white", linewidth=0.0,
                label=f"{SUITE_DISPLAY[s]} (successful only)"))
        fig.legend(handles=handles, loc="lower center", ncol=4,
                   frameon=True, framealpha=0.95, edgecolor="#AAAAAA",
                   fontsize=16, handlelength=1.6, handletextpad=0.55,
                   columnspacing=1.3, bbox_to_anchor=(0.5, -0.004))
        return
    handles = [Patch(facecolor=SUITE_COLORS[s], label=SUITE_DISPLAY[s])
               for s in SUITES]
    if mode in ("dumbbell", "slope"):
        handles += [
            Line2D([], [], marker="o", linestyle="none", color="#777777",
                   markersize=9, label="All episodes"),
            Line2D([], [], marker="o", linestyle="none",
                   markerfacecolor="white", markeredgecolor="#777777",
                   markeredgewidth=2.0, markersize=9,
                   label="Among successes"),
        ]
    else:
        # No n_succ >= 5 qualifier (user 2026-09-24): every pooled cell has at
        # least 34 successful episodes, so the MIN_SUCC rule never fires here --
        # it only mattered for the per-model cells.
        handles.append(Line2D([], [], color=SUCC_COLOR, linewidth=2.6,
                              path_effects=SUCC_FX,
                              label="Among successes"))
    if mode == "strip":
        handles.append(Line2D([], [], marker="o", linestyle="none",
                              markerfacecolor="white",
                              markeredgecolor="#1A1A1A", markeredgewidth=0.9,
                              markersize=7, label="Individual policy"))
    # Top RIGHT in the vertical panels (user 2026-09-24: it was sitting on the
    # bars on the left). Horizontal panel (a): bottom-right instead, the only
    # free corner there (the long Long bar still stops at 0.075).
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=18, handlelength=1.6,
               handletextpad=0.6, columnspacing=1.8,
               bbox_to_anchor=(0.5, 0.0))


def pooled_figure(ep, tiers, cats, metric_label, mode="bars"):
    """The figure once the model split is dropped (user 2026-09-24): colour
    carries the simulator, so both panels are two series per group and panel (b)
    keeps its among-successes overlay without a twelve-series legend. `mode`
    selects the rendering -- see suite_bars, plus mode="hbars"."""
    t_tab, c_tab = rate(ep, ["tier"], pool=True), rate(ep, ["category"],
                                                       pool=True)
    per_model = rate(ep, ["category"]) if mode == "strip" else None
    # Condensed, RQ1/RQ2 proportions (user 2026-09-24): a smaller canvas with
    # larger type, the key moved inside panel (a), and almost no padding, so the
    # figure survives a two-column shrink.
    if mode == "stack":
        # THE DEFAULT (user 2026-09-24): four stacked short full-width panels
        # of VERTICAL grouped bars -- ood-culture-tasks.pdf's own layout (its
        # three stacked "Alignment in ..." panels). Rows: (a)/(b) violations
        # per triggered event over horizon/category, (c)/(d) the same two
        # groupings for exposure. Title carries the grouping, the y label the
        # metric, and the framed legend sits INSIDE panel (a) like the
        # reference's in-panel keys.
        # Condensed height (user 2026-09-24: "a lot of white space"), rows
        # PAIRED BY GROUPING (user 2026-09-24): the two horizon panels stack
        # under ONE "(a) Task horizon" title and share ONE set of x tick
        # labels (printed only under the lower panel), likewise the two
        # category panels under "(b)". The metric is carried by each panel's
        # own y label.
        # Nested gridspecs, NOT tight_layout: tight_layout uses ONE hspace for
        # every row gap, so the big gap (tick labels + next title, between the
        # pairs) was forced inside each pair too. Here the pair-internal gap
        # is a thin 0.14 and only the between-group gap carries the labels.
        # 9.5 wide ("even more thinner", user 2026-09-24) with a 0.32
        # between-group gap; the width floor is the seven wrapped category
        # tick labels staying separated.
        fig = plt.figure(figsize=(9.5, 9.0))
        # Margins (user 2026-09-24): left 0.135 -- at 0.125 the rotated y
        # labels touched the canvas edge; the other three sides trimmed to
        # just clear their content (four-line 12pt category labels at the
        # bottom, the "(a)" title at the top).
        outer = fig.add_gridspec(2, 1, left=0.135, right=0.995, top=0.956,
                                 bottom=0.097, hspace=0.32)
        axes = []
        for _gi in range(2):
            sub = outer[_gi].subgridspec(2, 1, hspace=0.14)
            axes.append(fig.add_subplot(sub[0]))
            axes.append(fig.add_subplot(sub[1]))
        tier_labs = tick_labels(tiers, counts(ep, ["tier"]))
        cat_labs = tick_labels(cats, counts(ep, ["category"]),
                               CATEGORY_DISPLAY)
        # Wider bars in the 4-group panels (1.6x, not the width-matching
        # 1.35x) so they do not float in the full-width axes.
        span_t = 0.84 * len(tiers) / len(cats) * 1.6
        for gi, (by, keys, labs, span, fs, gtitle) in enumerate((
                ("tier", tiers, tier_labs, span_t, 16, "Task horizon"),
                ("category", cats, cat_labs, 0.84, 12.0, "Task category"))):
            # Three-line y labels at 14pt: rotated, the label's height is its
            # longest LINE, and the two-line 16pt version was taller than
            # these short panels -- it spilled onto the neighbouring panel's
            # label and ticks.
            for pi, (num, ylab) in enumerate((
                    ("V", "Violations per\ntriggered\nevent"),
                    ("D", "Exposure per\ntriggered\nevent (%)"))):
                ax = axes[gi * 2 + pi]
                suite_bars(ax, rate(ep, [by], pool=True, num=num), keys,
                           mode="bars", succ=True, span=span,
                           scale=1.0 if num == "V" else 100.0)
                ax.set_ylabel(ylab, fontsize=14, fontweight="bold")
                ax.tick_params(axis="y", labelsize=15)
                if pi == 0:   # upper pair member: title, no x labels
                    ax.set_title(f"({'ab'[gi]}) {gtitle}", pad=8,
                                 fontsize=19, fontweight="bold")
                    ax.set_xticklabels([])
                    ax.yaxis.set_major_locator(
                        matplotlib.ticker.MultipleLocator(0.02))
                    ax.yaxis.set_major_formatter(
                        matplotlib.ticker.FormatStrFormatter("%.2f"))
                else:
                    ax.set_xticklabels(labs, fontsize=fs)
        # In-panel framed legend (reference grammar): upper left of (a),
        # 2x2, with y headroom so it clears the Atomic/Short bars.
        handles = []
        for s in SUITES:
            handles.append(Patch(
                facecolor=SUITE_LIGHT[s], edgecolor="#888888",
                linewidth=0.5, label=f"{SUITE_DISPLAY[s]} (all rollouts)"))
            handles.append(Patch(
                facecolor=SUITE_COLORS[s], hatch=SUCC_HATCH,
                edgecolor="white", linewidth=0.0,
                label=f"{SUITE_DISPLAY[s]} (successful only)"))
        # ONE y scale per metric (user 2026-09-24): the horizon and category
        # panels of a metric carry the same quantity, so their bars must be
        # comparable across panels. Sharing also hands panel (a) its legend
        # headroom for free -- the category violations max (~0.10) is well
        # above the horizon bars (<=0.075), so the box clears the bars.
        for pair, pad in (((0, 2), 1.10), ((1, 3), 1.0)):
            # The violations pair gets 10% extra shared headroom: panel (a)'s
            # in-panel legend box otherwise grazes the Short group's bar.
            ytop = max(axes[i].get_ylim()[1] for i in pair) * pad
            for i in pair:
                axes[i].set_ylim(0, ytop)
        axes[0].legend(handles=handles, loc="upper left", ncol=2,
                       frameon=True, framealpha=0.95, edgecolor="#AAAAAA",
                       fontsize=13, handlelength=1.5, handletextpad=0.5,
                       columnspacing=1.2)
        save(fig, "pooled")
        return
    if mode == "hbars":
        # THE DEFAULT (user 2026-09-24), 2x2 like the vertical layout: row 1 is
        # violations per triggered event, row 2 is RQ2's exposure per triggered
        # event (% of rollout). Groups run down the y axis, so every category
        # name is plain one-line text. span=0.94: thick bars, thin group gaps.
        cat_disp = {k: " ".join(v.replace("/\n", "/").split())
                    for k, v in CATEGORY_DISPLAY.items()}
        # 8.2, not 10.2: without per-panel x labels the rows pack tighter
        # (user 2026-09-24: "make the whole figure even shorter").
        fig, axes = plt.subplots(2, 2, figsize=(15.4, 8.2),
                                 gridspec_kw=dict(width_ratios=[1.0, 1.0]))
        letters = iter("abcd")
        for r, (num, xlab) in enumerate((
                ("V", "Violations per triggered event"),
                ("D", "Exposure per triggered event (%)"))):
            scale = 1.0 if num == "V" else 100.0
            suite_hbars(axes[r][0], rate(ep, ["tier"], pool=True, num=num),
                        tiers, succ=True, scale=scale, span=0.88)
            axes[r][0].set_yticklabels(
                tick_labels(tiers, counts(ep, ["tier"])), fontsize=17)
            suite_hbars(axes[r][1], rate(ep, ["category"], pool=True, num=num),
                        cats, succ=True, scale=scale, span=0.88)
            axes[r][1].set_yticklabels(
                tick_labels(cats, counts(ep, ["category"]), cat_disp),
                fontsize=17)
            # One x scale per metric row, so (a)/(b) and (c)/(d) compare.
            xm = max(ax.get_xlim()[1] for ax in axes[r])
            for c, gtitle in enumerate(("Task horizon", "Task category")):
                ax = axes[r][c]
                ax.set_xlim(0, xm)
                ax.tick_params(axis="x", labelsize=19)
                # Row/column semantics (user 2026-09-24): the grouping is a
                # COLUMN property, so it is titled once, on the top row; the
                # metric is a ROW property, carried entirely by the rotated
                # row label added after layout (no per-panel x labels). Panel
                # letters sit ABOVE the axes, clear of the plots.
                if r == 0:
                    ax.set_title(gtitle, pad=8, fontsize=24,
                                 fontweight="bold")
                # Panel letters in the upper-RIGHT corner, inside the axes
                # (user 2026-09-24) -- the top-right is empty in every panel,
                # since the first group's bars start from the left edge.
                ax.text(0.985, 0.965, f"({next(letters)})",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=22, fontweight="bold")
        pooled_legend(fig, mode)
        fig.tight_layout(rect=(0.050, 0.058, 1, 1), pad=0.3, w_pad=1.0,
                         h_pad=1.9)
        # Rotated row labels carrying the metric (user 2026-09-24: the metric
        # moved off the x axes and onto the row), centred on each row's axes
        # (positions are only final after tight_layout). Two lines so the
        # rotated text stays shorter than the row is tall.
        for r, headline in enumerate(("Violations per\ntriggered event",
                                      "Exposure per\ntriggered event (%)")):
            pos = axes[r][0].get_position()
            fig.text(0.024, (pos.y0 + pos.y1) / 2, headline, rotation=90,
                     ha="center", va="center", fontsize=22, fontweight="bold")
        save(fig, "pooled")
        return
    elif mode == "bars":
        # 2x2 (user 2026-09-24: "add the exposure below"): row 1 is violations
        # per triggered event, row 2 is RQ2's exposure per triggered event (% of
        # rollout), both rows over the same two groupings with the same pairing.
        # Columns share x so the group labels print once, at the bottom.
        fig, axes = plt.subplots(2, 2, figsize=(17.2, 8.8), sharey="row",
                                 sharex="col",
                                 gridspec_kw=dict(width_ratios=[1.0, 1.0],
                                                  hspace=0.24))
        span_a = 0.84 * len(tiers) / len(cats) * 1.35
        letters = iter("abcd")
        for r, (num, ylab) in enumerate((
                ("V", "Violations per\ntriggered event"),
                ("D", "Exposure per\ntriggered event (%)"))):
            scale = 1.0 if num == "V" else 100.0
            suite_bars(axes[r][0], rate(ep, ["tier"], pool=True, num=num),
                       tiers, mode="bars", succ=True, span=span_a, scale=scale)
            suite_bars(axes[r][1], rate(ep, ["category"], pool=True, num=num),
                       cats, mode="bars", succ=True, span=0.84, scale=scale)
            axes[r][0].set_ylabel(ylab, fontsize=20, fontweight="bold")
            for c, gtitle in enumerate(("Task horizon", "Task category")):
                ax = axes[r][c]
                ax.set_title(f"({next(letters)}) {gtitle}", pad=8,
                             fontsize=24, fontweight="bold")
                ax.tick_params(axis="y", labelsize=20, labelleft=True)
            if num == "V":
                # Two-decimal ticks on a 0.02 grid, not the default 0.025 with
                # its ragged third decimal; the exposure row autoscales.
                for ax in axes[r]:
                    ax.yaxis.set_major_locator(
                        matplotlib.ticker.MultipleLocator(0.02))
                    ax.yaxis.set_major_formatter(
                        matplotlib.ticker.FormatStrFormatter("%.2f"))
        # ONE tick-label size on both panels (user 2026-09-24): 15 is the
        # largest at which the seven wrapped category names stay separated.
        axes[1][0].set_xticklabels(tick_labels(tiers, counts(ep, ["tier"])),
                                   fontsize=15)
        axes[1][1].set_xticklabels(
            tick_labels(cats, counts(ep, ["category"]), CATEGORY_DISPLAY),
            fontsize=15)
        pooled_legend(fig, mode)
        # Explicit geometry, not tight_layout: it does not budget for
        # fig.legend, so the legend kept landing on panel (d)'s three-line tick
        # labels no matter the rect. bottom=0.175 = labels (~0.10) + legend row.
        fig.subplots_adjust(left=0.075, right=0.992, top=0.945, bottom=0.175,
                            hspace=0.52, wspace=0.10)
        save(fig, "pooled_bars")
        return
    else:
        fig, axes = plt.subplots(1, 2, figsize=(17.2, 5.0), sharey=True,
                                 gridspec_kw=dict(width_ratios=[1.0, 1.0]))
        # succ=True here too (user 2026-09-24): the horizon panel gets the same
        # among-successes overlay as the category panel.
        suite_bars(axes[0], t_tab, tiers, mode=mode, succ=True,
                   span=0.66 * len(tiers) / len(cats) * 1.35)
        axes[0].set_xticklabels(tick_labels(tiers, counts(ep, ["tier"])),
                                fontsize=15)
        # Size and weight match RQ1's identical y label exactly. supylabel, not
        # set_ylabel: at 21 pt the rotated text is taller than the axes box
        # (which lost height to the tick and legend rows), so an axes-anchored
        # label gets truncated -- the figure-anchored one has the full canvas
        # height to centre in.
        fig.supylabel(metric_label, fontsize=21, fontweight="bold", x=0.008)
        suite_bars(axes[1], c_tab, cats, succ=True, mode=mode,
                   per_model=per_model)
        axes[1].set_xticklabels(tick_labels(cats, counts(ep, ["category"]),
                                            CATEGORY_DISPLAY), fontsize=15)
    for ax, title in zip(axes, ("(a) Task horizon", "(b) Task category")):
        ax.set_title(title, pad=8, fontsize=24, fontweight="bold")
        if mode != "hbars":
            # labelleft on BOTH panels (user 2026-09-24: the two subfigures'
            # y-axis type must match) -- sharey otherwise leaves (b) unlabelled.
            ax.tick_params(axis="y", labelsize=20, labelleft=True)
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FormatStrFormatter("%.2f"))
    pooled_legend(fig, mode)
    # rect bottom 0.115, not 0.085: the extra sliver is the breathing room
    # between the tick labels and the legend row (user 2026-09-24). Small left
    # inset so the bold y label is not trimmed at the canvas edge.
    fig.tight_layout(rect=(0.028, 0.115, 1, 1), pad=0.25, w_pad=0.9)
    save(fig, f"pooled_{mode}")


def save(fig, variant):
    stem = "RQ3_conditional_ab" + ("" if variant == "pooled" else f"_{variant}")
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"{stem}.{ext}")
        fig.savefig(out, dpi=300 if ext == "png" else None)
        print(f"[fig] {out}")


def cat_legend(fig, cats, mode, y=0.0):
    handles = [Patch(facecolor=CATEGORY_COLORS[i % len(CATEGORY_COLORS)],
                     label=" ".join(CATEGORY_DISPLAY[c].replace("/\n", "/")
                                    .split()))
               for i, c in enumerate(cats)]
    if mode == "dumbbell":
        handles += [
            Line2D([], [], marker="o", linestyle="none", color="#777777",
                   markersize=7, label="All episodes"),
            Line2D([], [], marker="o", linestyle="none",
                   markerfacecolor="white", markeredgecolor="#777777",
                   markeredgewidth=1.5, markersize=7,
                   label=r"Among successes ($n_{\rm succ}\geq$"f"{MIN_SUCC})"),
        ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=16, handlelength=1.6, columnspacing=1.4,
               bbox_to_anchor=(0.5, y))


def policy_figure(ep, cols, tiers, cats, metric_label, mode):
    """Transposed panel (b): policies on the x axis, task category as colour."""
    fig, axes = plt.subplots(1, 2, figsize=(20.5, 6.6),
                             gridspec_kw=dict(width_ratios=[4.2, 7.0]))
    group_bars(axes[0], rate(ep, ["tier"]), tiers, cols, scale=1.0)
    axes[0].set_xticklabels(tick_labels(tiers, counts(ep, ["tier"])),
                            fontsize=16)
    axes[0].set_ylabel(metric_label, fontsize=19)
    policy_panel(axes[1], rate(ep, ["category"]), cats, cols, scale=1.0,
                 mode=mode)
    axes[1].set_ylabel(metric_label if mode == "dumbbell"
                       else r"$\Delta$ violations per triggered event "
                            r"(all $-$ successes)", fontsize=19)
    for ax, title in zip(axes, ("(a) Task horizon",
                                "(b) Task category, by policy")):
        ax.set_title(title, pad=32 if ax is axes[1] else 14, fontsize=22,
                     fontweight="bold")
        ax.tick_params(axis="y", labelsize=17)
    cat_legend(fig, cats, mode)
    fig.tight_layout(rect=(0, 0.115 if mode == "dumbbell" else 0.09, 1, 1),
                     pad=0.4, w_pad=1.4)
    save(fig, f"{mode}_policy")


def facet_figure(ep, cols, tiers, cats, metric_label):
    """2x2 facets: rows = the two groupings, columns = the two simulators. Each
    facet only carries its own suite's policies (5 vs 7), so nothing has to be
    crammed 12-wide; y is shared within a row so the suites stay comparable."""
    rc = [c for c in cols if c[0] == "RoboCasa"]
    lb = [c for c in cols if c[0] == "LIBERO"]
    fig, axes = plt.subplots(2, 2, figsize=(17.0, 11.0), sharey="row",
                             gridspec_kw=dict(width_ratios=[1.0, 1.0],
                                              hspace=0.46, wspace=0.06))
    letters = iter("abcd")
    for row, (by, keys, disp, draw) in enumerate(
            (("tier", tiers, None, group_bars),
             ("category", cats, CATEGORY_DISPLAY, group_dumbbell))):
        tab, cnt = rate(ep, [by]), counts(ep, [by])
        for col, (suite, sub) in enumerate((("RoboCasa", rc), ("LIBERO", lb))):
            ax = axes[row][col]
            present = [k for k in keys
                       if ((ep.suite == suite) & (ep[by] == k)).any()]
            draw(ax, tab, present, sub, scale=1.0, succ=(by == "category"))
            # One suite per facet, so the tick carries just that suite's count.
            labs = [lab.split("\n(")[0] + "\n(" +
                    (cnt[k][0] if suite == "RoboCasa" else cnt[k][1]) + ")"
                    for k, lab in zip(present, tick_labels(present, cnt, disp))]
            if row == 1:   # seven long category names in a half-width facet
                labs = [" ".join(l.replace("/\n", "/").split()) for l in labs]
                ax.set_xticklabels(labs, fontsize=14, rotation=28, ha="right",
                                   rotation_mode="anchor")
            else:
                ax.set_xticklabels(labs, fontsize=16)
            ax.set_title(f"({next(letters)}) "
                         f"{'Task horizon' if row == 0 else 'Task category'}"
                         f" — {SUITE_DISPLAY[suite]}",
                         pad=14, fontsize=21, fontweight="bold")
            ax.tick_params(axis="y", labelsize=17)
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=18)
    handles = [Patch(facecolor=MODEL_COLORS[m], label=MODEL_DISPLAY[m])
               for m in MODEL_ORDER if any(mm == m for _, mm in cols)]
    handles += [
        Line2D([], [], marker="o", linestyle="none", color="#777777",
               markersize=7, label="All episodes (rows c, d)"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor="#777777", markeredgewidth=1.5, markersize=7,
               label=r"Among successes ($n_{\rm succ}\geq$"f"{MIN_SUCC})"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=16, handlelength=1.6, columnspacing=1.4,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.085, 1, 1), pad=0.5)
    save(fig, "facet")


def main():
    apply_style()
    if "--recompute" in sys.argv or not os.path.exists(CSV):
        ep = recompute()
        ep.to_csv(CSV, index=False)
        print(f"[csv] {CSV}  ({len(ep):,} episodes)")
    else:
        ep = pd.read_csv(CSV, dtype={"episode": str})
    if "--validate" in sys.argv:
        validate(ep)
        return

    cols = []
    for suite in SUITES:
        have = set(ep.loc[ep.suite == suite, "model"])
        cols += [(suite, m) for m in MODEL_ORDER if m in have]
    tiers = [t for t in TIER_ORDER if (ep.tier == t).any()]
    cats = sorted([c for c in CATEGORY_ORDER if (ep.category == c).any()],
                  key=lambda c: -int(ep.loc[(ep.category == c) &
                                            (ep.suite == "RoboCasa"),
                                            "task"].nunique()))

    # Variants. `pooled` is the default (user 2026-09-24: no model split) --
    # colour = simulator, two bars per group. The per-model renderings are kept
    # behind the flag: bars (tick = among successes), dumbbell, hbars, split,
    # lines, dumbbell_policy, delta_policy, facet.
    variant = "pooled"
    for a in sys.argv[1:]:
        if a.startswith("--variant="):
            variant = a.split("=", 1)[1]
    # No x10^-2 here (user 2026-09-24): the axis carries the numbers, so the
    # scaling only earns its keep when values are printed in cells, as in RQ2.
    metric_label = "Violations per triggered event"

    if variant == "pooled" or variant.startswith("pooled_"):
        # stack (four stacked short vertical-bar panels, the ood-culture
        # layout) is the default rendering (user 2026-09-24); hbars and the
        # vertical 2x2 remain available as --variant=pooled_hbars /
        # pooled_bars.
        pooled_figure(ep, tiers, cats, metric_label,
                      mode=variant[len("pooled_"):] if "_" in variant
                      else "stack")
        return
    if variant == "facet":
        facet_figure(ep, cols, tiers, cats, metric_label)
        return
    if variant in ("dumbbell_policy", "delta_policy"):
        policy_figure(ep, cols, tiers, cats, metric_label,
                      mode=variant.split("_")[0])
        return

    horiz = variant in ("hbars", "split")   # categories down the y axis
    fig, axes = plt.subplots(
        1, 2, figsize=(20.5, 6.9 if horiz else 6.0), sharey=not horiz,
        gridspec_kw=dict(width_ratios=[len(tiers), len(cats)] if not horiz
                         else [4.0, 7.0]))
    group_bars(axes[0], rate(ep, ["tier"]), tiers, cols, scale=1.0)
    axes[0].set_xticklabels(tick_labels(tiers, counts(ep, ["tier"])),
                            fontsize=16)
    axes[0].set_ylabel(metric_label, fontsize=19)

    draw = {"bars": group_bars, "dumbbell": group_dumbbell,
            "hbars": group_hbars, "split": group_split_heatmap,
            "lines": group_lines}[variant]
    sm = draw(axes[1], rate(ep, ["category"]), cats, cols, scale=1.0,
              succ=True)
    lab = tick_labels(cats, counts(ep, ["category"]),
                      {k: " ".join(v.replace("/\n", "/").split())
                       for k, v in CATEGORY_DISPLAY.items()}
                      if horiz else CATEGORY_DISPLAY)
    if horiz:
        axes[1].set_yticklabels(lab, fontsize=16)
        if variant != "split":
            axes[1].set_xlabel(metric_label, fontsize=19)
            axes[1].tick_params(axis="x", labelsize=17)
    else:
        axes[1].set_xticklabels(lab, fontsize=16)
    if variant == "split":
        fig.colorbar(sm, ax=axes[1], fraction=0.030, pad=0.015,
                     extend="max").ax.tick_params(labelsize=15)
    for ax, title in zip(axes, ("(a) Task horizon", "(b) Task category")):
        ax.set_title(title, pad=14, fontsize=22, fontweight="bold")
        ax.tick_params(axis="y", labelsize=17)

    handles = [Patch(facecolor=MODEL_COLORS[m], label=MODEL_DISPLAY[m])
               for m in MODEL_ORDER if any(mm == m for _, mm in cols)]
    if variant == "split":
        # The colour key is the colorbar; the legend only has to say which
        # triangle is which metric (and that panel (a) is still bars).
        handles += [
            Patch(facecolor="#BBBBBB", edgecolor="#1A1A1A", linewidth=0.9,
                  label="LIBERO (outlined, panel a)"),
            Patch(facecolor="#BB3F4A", label="Upper-left: all episodes"),
            Patch(facecolor="#F6BBAE",
                  label=r"Lower-right: among successes ($n_{\rm succ}\geq$"
                        f"{MIN_SUCC})"),
            Patch(facecolor="#DDDDDD", label="n/a"),
        ]
    elif variant == "lines":
        handles += [
            Line2D([], [], color="#777777", linewidth=1.6, marker="o",
                   markersize=6, label="RoboCasa (solid)"),
            Line2D([], [], color="#777777", linewidth=1.6, marker="s",
                   linestyle=(0, (3, 1.4)), markersize=6,
                   label="LIBERO (dashed)"),
            Line2D([], [], color="#777777", linestyle="none", marker="o",
                   markerfacecolor="white", markeredgecolor="#777777",
                   markeredgewidth=1.4, markersize=6,
                   label=r"Among successes ($n_{\rm succ}\geq$"f"{MIN_SUCC})"),
        ]
    elif variant == "dumbbell":
        handles += [
            Line2D([], [], marker="o", linestyle="none", color="#777777",
                   markeredgecolor="#1A1A1A", markersize=7,
                   label="LIBERO (outlined)"),
            Line2D([], [], marker="o", linestyle="none", color="#777777",
                   markersize=7, label="All episodes"),
            Line2D([], [], marker="o", linestyle="none",
                   markerfacecolor="white", markeredgecolor="#777777",
                   markeredgewidth=1.6, markersize=7,
                   label=r"Among successes ($n_{\rm succ}\geq$"f"{MIN_SUCC})"),
        ]
    else:
        handles += [
            Patch(facecolor="#BBBBBB", edgecolor="#1A1A1A", linewidth=0.9,
                  label="LIBERO (outlined)"),
            Line2D([], [], color=SUCC_COLOR, linewidth=1.9,
                   label=r"Among successes ($n_{\rm succ}\geq$"f"{MIN_SUCC})"),
        ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=17, handlelength=1.7,
               columnspacing=1.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.065 if horiz else 0.075, 1, 1), pad=0.4,
                     w_pad=1.2)
    stem = "RQ3_conditional_ab" + ("" if variant == "bars" else f"_{variant}")
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"{stem}.{ext}")
        fig.savefig(out, dpi=300 if ext == "png" else None)
        print(f"[fig] {out}")


if __name__ == "__main__":
    main()
