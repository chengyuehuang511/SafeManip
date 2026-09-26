#!/usr/bin/env python3
"""RQ2 conditional heatmaps, split per explicit user decision (2026-09-23):

  Figure 1 (RQ2_conditional_ab):  (a) Violations per triggered event
                                  (b) Exposure rate per triggered event
                                      (% of rollout)
  Figure 2 (RQ2_conditional_exposure): Triggered events per episode
                                  (the old panel (c), now standalone)

"Triggered event" is the new user-facing name for what the pipeline calls an
activation. The original generator (plot_combined/make_plots_conditional.py)
was TRUNCATED TO 0 BYTES when the original share hit 100% -- this file recreates it
self-contained, with the estimator validated cell-by-cell against the
surviving plots_conditional/RQ2_conditional_combined.png.

Same caching pattern as render_rq1_scatter_single.py: granular sums live in
RQ2_conditional_cells.csv next to this file, so a plain run renders in
seconds. `--recompute` reloads both suites through the flash pipeline
(minutes); run it after any monitor/corpus change. `--validate` prints
candidate estimators against reference cells read off the surviving PNG.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "RQ2_conditional_cells.csv")
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
CATEGORY_ORDER = ["CollisionContact", "GraspStability", "ReleaseStability",
                  "CrossContam", "PreconditionSafe", "Mechanism",
                  "Containment", "EAS"]
# Sentence-case names matching the reference safety-rate figure's row labels
# (its "Action-onset" is this pipeline's PreconditionSafe -- confirmed by its
# applicable-task count of 49, and "Enclosure/access" is EAS at 24).
CATEGORY_DISPLAY = {
    "CollisionContact": "Collision/contact", "GraspStability": "Grasp stability",
    "ReleaseStability": "Release stability", "CrossContam": "Cross-contamination",
    "PreconditionSafe": "Action-onset", "Mechanism": "Mechanism",
    "Containment": "Containment", "EAS": "Enclosure/access",
}
# Desaturated Reds/Blues (user 2026-09-24: "make the color saturation
# lower"): the standard colormaps with HSV saturation scaled by DESAT and
# the near-black top trimmed at CMAP_TOP.
HEAT_CMAP, EXPO_CMAP = "Reds_soft", "Blues_soft"
DESAT, CMAP_TOP = 0.70, 0.90


def register_soft_cmaps():
    for base, name in (("Reds", "Reds_soft"), ("Blues", "Blues_soft")):
        rgba = plt.get_cmap(base)(np.linspace(0.0, CMAP_TOP, 256))
        hsv = matplotlib.colors.rgb_to_hsv(rgba[:, :3])
        hsv[:, 1] *= DESAT
        rgba[:, :3] = matplotlib.colors.hsv_to_rgb(hsv)
        try:
            matplotlib.colormaps.register(
                matplotlib.colors.ListedColormap(rgba, name=name), name=name)
        except ValueError:
            pass  # already registered


def apply_style():
    register_soft_cmaps()
    # Reference layout style with the reference's sans-serif face
    # (user 2026-09-24, reversing the earlier Times New Roman call):
    # bold only on titles and cell values; ticks/band labels regular.
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "mathtext.fontset": "dejavusans",
        "axes.labelweight": "bold", "axes.titleweight": "bold",
        "axes.edgecolor": "black", "axes.linewidth": 1.2,
    })


def recompute_cells():
    """Reload both suites (minutes) -> per-(property, task) sums. Everything
    any estimator needs, so estimator changes never re-trigger this."""
    sys.path.insert(0, PC)
    from activation_data import load_suite  # noqa: E402
    parts = []
    for suite in SUITES:
        df, _meta = load_suite(suite)
        df = df[df["applicable"] == 1].copy()
        df["duration_frac"] = (df["violation_duration"]
                               / df["num_frames"].replace(0, np.nan))
        g = (df.groupby(["model", "category", "task", "property_name"],
                        observed=True)
             .agg(V=("violations", "sum"), A=("activations", "sum"),
                  D=("duration_frac", "sum"), n_ep=("episode", "nunique"))
             .reset_index())
        g.insert(0, "suite", suite)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Estimators, VALIDATED cell-by-cell against the surviving conditional PNG
# (2026-09-23): panels (a)/(b) are the pooled ratio of sums per
# (suite, model, category); the exposure panel is the per-episode CATEGORY
# TOTAL (activations summed over the category's properties within a task,
# divided by that task's episode count), averaged over tasks. A per-property
# mean instead undercounts every multi-property category (Grasp x2,
# Precondition, EAS) -- that mismatch is how the sum was identified. Cells
# that never trigger come out NaN and render as the grey "n/a", never 0.
# ---------------------------------------------------------------------------
def pooled(cells, num, den):
    g = (cells.groupby(["suite", "model", "category"], observed=True)
         [[num, den]].sum())
    return (g[num] / g[den].replace(0, np.nan)).rename("val")


def exposure(cells):
    g = (cells.groupby(["suite", "model", "category", "task"], observed=True)
         .agg(S=("A", "sum"), N=("n_ep", "max")))
    per_task = g.S / g.N.replace(0, np.nan)
    return (per_task.groupby(["suite", "model", "category"], observed=True)
            .mean())


def category_rows(cells):
    """Row order and labels in the reference figure's grammar: name plus the
    number of tasks the category is APPLICABLE to, rows sorted by that count
    descending. The reference is RoboCasa-only so it carries one number; this
    figure spans both suites, so both are shown ("50/40"), with an en dash
    where a category has no LIBERO counterpart. Ties (Grasp/Release at 38) keep
    CATEGORY_ORDER, matching the reference's own ordering."""
    n = cells.pivot_table(index="category", columns="suite", values="task",
                          aggfunc="nunique")

    def cnt(cat, suite):
        v = n.loc[cat, suite] if (cat in n.index and suite in n.columns) \
            else np.nan
        return "–" if not np.isfinite(v) else f"{int(v)}"

    def rc(cat):
        v = n.loc[cat, "RoboCasa"] if cat in n.index else np.nan
        return v if np.isfinite(v) else 0

    order = sorted(CATEGORY_ORDER, key=lambda c: -rc(c))
    labels = [f"{CATEGORY_DISPLAY[c]} ({cnt(c, 'RoboCasa')}/"
              f"{cnt(c, 'LIBERO')})" for c in order]
    return order, labels


def matrix(series, cols, rows=None):
    rows = list(rows) if rows is not None else CATEGORY_ORDER
    out = pd.DataFrame(index=rows,
                       columns=pd.MultiIndex.from_tuples(cols), dtype=float)
    for suite, m in cols:
        for cat in rows:
            try:
                out.loc[cat, (suite, m)] = series.loc[(suite, m, cat)]
            except KeyError:
                pass
    return out


def validate(cells):
    """Print candidate estimators against cells read off the surviving PNG."""
    refs = [  # (suite, model, category, metric, value_in_png)
        ("RoboCasa", "grootn15", "CollisionContact", "risk", 0.078),
        ("RoboCasa", "grootn16", "CrossContam", "risk", 1.964),
        ("LIBERO", "openvla", "CollisionContact", "risk", 0.128),
        ("RoboCasa", "grootn15", "CollisionContact", "dur", 0.18),
        ("LIBERO", "openvla", "CollisionContact", "dur", 1.98),
        ("RoboCasa", "grootn15", "CollisionContact", "expo", 11.48),
        ("RoboCasa", "openpi_pi0", "CollisionContact", "expo", 18.80),
        ("LIBERO", "openvla", "CollisionContact", "expo", 1.21),
    ]
    e = {"risk": pooled(cells, "V", "A"),
         "dur": pooled(cells, "D", "A") * 100,
         "expo": exposure(cells)}
    for suite, m, cat, metric, ref in refs:
        try:
            v = e[metric].loc[(suite, m, cat)]
        except KeyError:
            v = np.nan
        print(f"  {suite:8s} {m:12s} {cat:18s} {metric}: "
              f"got {v:8.3f}  png {ref}")


def draw_heatmap(ax, mat, row_labels, col_labels, cmap, fmt="{:.2f}",
                 vmin=0, vmax=None, gamma=None, fontsize=13.0,
                 ticksize=18.0):
    data = mat.astype(float)
    masked = np.ma.masked_invalid(data.values)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#DDDDDD")
    ax.grid(False)
    if gamma is not None:
        norm = matplotlib.colors.PowerNorm(gamma, vmin=vmin, vmax=vmax)
        im = ax.imshow(masked, cmap=cmap_obj, aspect="auto", norm=norm)
    else:
        im = ax.imshow(masked, cmap=cmap_obj, aspect="auto",
                       vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    # 45 deg, not 35: with 12 columns the shallower angle spread each label far
    # enough horizontally that neighbours collided across the suite divider.
    ax.set_xticklabels(col_labels, rotation=45, ha="right",
                       rotation_mode="anchor", fontsize=ticksize - 2)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=ticksize)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.values[i, j]
            if np.isnan(v):
                continue
            color = "white" if im.norm(v) > 0.6 else "black"
            txt = fmt(v) if callable(fmt) else fmt.format(v)
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=fontsize, fontweight="bold", color=color)
    return im


def _vmax(mat, scale=1.0, floor=0.01):
    v = np.nanmax(mat.values) if mat.size else np.nan
    return max(floor, v * scale) if np.isfinite(v) else floor


def bulk_vmax(mat, gap=3.0, floor=0.01):
    """Color range for the BULK of a panel, not its outliers (explicit user
    decision 2026-09-23: the panels are different quantities, so their scales
    need not agree -- each should be chosen to make its own comparison legible).

    Scan the upper half of the sorted values for the largest multiplicative
    step; if some top group is separated from the rest by >= `gap`x, clip there
    and let that group saturate (its numbers are printed in the cells anyway).
    With no such step the full range is returned unchanged, so this only fires
    where an outlier actually flattens the panel -- panel (a), whose
    Cross-Contamination row is ~9x the next-largest cell. The gap search is
    restricted to values above the median so that ratios among near-zero cells
    (0.05 -> 0.01 is also 5x) cannot trigger it."""
    v = np.sort(mat.values[np.isfinite(mat.values)])[::-1]
    v = v[v > 0]
    if v.size < 4:
        return max(floor, v[0]) if v.size else floor
    med, cut, best = np.median(v), None, gap
    for i in range(v.size - 1):
        if v[i + 1] < med:
            break
        r = v[i] / v[i + 1]
        if r > best:
            best, cut = r, v[i + 1]
    return max(floor, v[0] if cut is None else cut)


def decorate(ax, cols, n_rc, band_fs=18):
    ax.axvline(n_rc - 0.5, color="black", linewidth=1.8)
    # Regular weight, like the tick labels: only the panel titles and the cell
    # values are bold (RQ1's convention).
    ax.text(((n_rc - 1) / 2 + 0.5) / len(cols), 1.015,
            SUITE_DISPLAY["RoboCasa"], transform=ax.transAxes, ha="center",
            va="bottom", fontsize=band_fs)
    ax.text((n_rc + (len(cols) - n_rc) / 2) / len(cols), 1.015,
            SUITE_DISPLAY["LIBERO"], transform=ax.transAxes, ha="center",
            va="bottom", fontsize=band_fs)


def main():
    apply_style()
    if "--recompute" in sys.argv or not os.path.exists(CSV):
        cells = recompute_cells()
        cells.to_csv(CSV, index=False)
        print(f"[csv] {CSV}")
    else:
        cells = pd.read_csv(CSV)
    if "--validate" in sys.argv:
        validate(cells)
        return

    cols = []
    for suite in SUITES:
        have = set(cells.loc[cells.suite == suite, "model"])
        cols += [(suite, m) for m in MODEL_ORDER if m in have]
    col_labels = [MODEL_DISPLAY[m] for _, m in cols]
    rows, row_labels = category_rows(cells)
    n_rc = sum(1 for s, _ in cols if s == "RoboCasa")

    # Panel (a) is reported in units of 1e-2 (explicit user decision
    # 2026-09-23): the raw ratios are 0.001-1.96, so three decimals were the
    # only way to keep resolution. Scaling the DATA (not just the label) keeps
    # the colorbar in the same units as the printed cells.
    risk = matrix(pooled(cells, "V", "A"), cols, rows) * 100
    dur = matrix(pooled(cells, "D", "A"), cols, rows) * 100
    expo = matrix(exposure(cells), cols, rows)

    # -- Figure 1: (a) risk + (b) exposure-rate, shared category rows --------
    # Compact layout in the reference figure's proportions: short panels, thin
    # colorbars hugging them, minimal gutter between (a) and (b), and type
    # scaled up relative to the cells.
    fig, axes = plt.subplots(1, 2, figsize=(18.2, 5.5))
    # Three sig figs, so the 1e-2-scaled Cross-Contamination row (>100) drops
    # its decimal instead of colliding with its neighbours.
    v0 = bulk_vmax(risk, floor=5.0)
    im0 = draw_heatmap(axes[0], risk, row_labels, col_labels, HEAT_CMAP,
                       fmt=lambda v: f"{v:.0f}" if v >= 100 else f"{v:.1f}",
                       vmin=0, gamma=0.6, vmax=v0)
    axes[0].set_title("(a) Violations per triggered event "
                      r"($\times 10^{-2}$)", pad=36,
                      fontsize=19, fontweight="bold")
    fig.colorbar(im0, ax=axes[0], fraction=0.035, pad=0.022,
                 extend="max" if v0 < np.nanmax(risk.values) else "neither")

    v1 = bulk_vmax(dur, floor=0.05)
    im1 = draw_heatmap(axes[1], dur, row_labels, col_labels, HEAT_CMAP,
                       fmt="{:.2f}", vmin=0, gamma=0.5, vmax=v1)
    axes[1].set_title("(b) Exposure rate per triggered event (% of rollout)",
                      pad=36, fontsize=19, fontweight="bold")
    fig.colorbar(im1, ax=axes[1], fraction=0.035, pad=0.022,
                 extend="max" if v1 < np.nanmax(dur.values) else "neither")
    print(f"[scale] (a) vmax={v0:.2f} of max {np.nanmax(risk.values):.2f}; "
          f"(b) vmax={v1:.2f} of max {np.nanmax(dur.values):.2f}")
    axes[1].set_yticklabels([])
    for ax in axes:
        decorate(ax, cols, n_rc)
    for im, ax in ((im0, axes[0]), (im1, axes[1])):
        pass
    for cb_ax in fig.axes:
        if cb_ax not in list(axes):
            cb_ax.tick_params(labelsize=16)
    fig.tight_layout(pad=0.3, w_pad=1.6)
    fig.subplots_adjust(wspace=0.17)
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"RQ2_conditional_ab.{ext}")
        fig.savefig(out, dpi=300 if ext == "png" else None)
        print(f"[fig] {out}")

    # -- Figure 2: standalone exposure (old panel (c)) -----------------------
    fig2, ax2 = plt.subplots(figsize=(11.6, 5.8))
    v2 = bulk_vmax(expo, floor=0.5)
    im2 = draw_heatmap(ax2, expo, row_labels, col_labels, EXPO_CMAP,
                       fmt="{:.2f}", vmin=0, gamma=0.5, vmax=v2)
    ax2.set_title("Triggered events per episode", pad=36,
                  fontsize=19, fontweight="bold")
    fig2.colorbar(im2, ax=ax2, fraction=0.035, pad=0.022,
                  extend="max" if v2 < np.nanmax(expo.values) else "neither"
                  ).ax.tick_params(labelsize=16)
    decorate(ax2, cols, n_rc)
    fig2.tight_layout(pad=0.3)
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"RQ2_conditional_exposure.{ext}")
        fig2.savefig(out, dpi=300 if ext == "png" else None)
        print(f"[fig] {out}")


if __name__ == "__main__":
    main()
