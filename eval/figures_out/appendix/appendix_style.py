"""Shared paper style for every appendix figure — copied from the main
figures' scripts (render_rq1_scatter_single.py / render_rq2_conditional.py /
render_rq3_conditional.py, restyled 2026-09-24) so the appendix cannot drift
from the main text. Regular-weight sans, sparse bold (titles/axis labels
only), white panels, light dashed grid, full black axes box.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

SUITES = ("RoboCasa", "LIBERO")
SUITE_DISPLAY = {"RoboCasa": "RoboCasa365", "LIBERO": "LIBERO"}
MODEL_ORDER = ["grootn15", "grootn16", "openpi_pi0", "openpi_pi05",
               "openvla", "cosmos_policy", "rldx1"]
MODEL_DISPLAY = {
    "grootn15": "GR00T N1.5", "grootn16": "GR00T N1.6",
    "openpi_pi0": r"$\pi_0$", "openpi_pi05": r"$\pi_{0.5}$",
    "openvla": "OpenVLA", "cosmos_policy": "Cosmos Policy", "rldx1": "RLDX-1",
}
MODEL_DISPLAY_BOLD = dict(MODEL_DISPLAY,
                          openpi_pi0=r"$\mathbf{\pi_0}$",
                          openpi_pi05=r"$\mathbf{\pi_{0.5}}$")
MODEL_COLORS = {  # tab10, matching the restyled RQ1 scatter
    "grootn15": "#1F77B4", "grootn16": "#2CA02C", "openpi_pi0": "#D62728",
    "openpi_pi05": "#E377C2", "rldx1": "#FF7F0E", "openvla": "#17BECF",
    "cosmos_policy": "#7F7F7F",
}
SUITE_MARKERS = {"RoboCasa": "o", "LIBERO": "s"}
SUITE_FIT_COLORS = {"RoboCasa": "#444444", "LIBERO": "#8B4513"}
SUITE_DASHES = {"RoboCasa": (0, (7, 2.5)), "LIBERO": (0, (1.8, 1.8))}
# Paired-bar grammar (ood-culture-tasks.pdf): all rollouts = light solid,
# successful only = saturated + white // hatch.
SUITE_COLORS = {"RoboCasa": "#D62728", "LIBERO": "#1F77B4"}
SUITE_LIGHT = {"RoboCasa": "#FF9896", "LIBERO": "#AEC7E8"}
SUCC_HATCH = "//"
# Safety-category rows, order and display exactly as render_rq2_conditional.py
CATEGORY_ORDER = ["CollisionContact", "PreconditionSafe", "GraspStability",
                  "ReleaseStability", "EAS", "Mechanism", "CrossContam",
                  "Containment"]
CATEGORY_DISPLAY = {
    "CollisionContact": "Collision/contact", "PreconditionSafe": "Action-onset",
    "GraspStability": "Grasp stability", "ReleaseStability": "Release stability",
    "EAS": "Enclosure/access", "Mechanism": "Mechanism",
    "CrossContam": "Cross-contamination", "Containment": "Containment",
}
# Outcome colors, as in the old RQ1 stacked bar (kept: semantic green/red).
OUTCOME_COLORS = {
    "safe_success": "#2E7D32", "unsafe_success": "#F9A825",
    "safe_fail": "#607D8B", "unsafe_fail": "#C62828",
}
OUTCOME_DISPLAY = {
    "safe_success": "Safe & Successful", "unsafe_success": "Unsafe & Successful",
    "safe_fail": "Safe & Failed", "unsafe_fail": "Unsafe & Failed",
}


def apply_style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "mathtext.fontset": "dejavusans",
        "axes.labelweight": "bold", "axes.titleweight": "bold",
        "axes.edgecolor": "black", "axes.linewidth": 1.2,
        "axes.axisbelow": True,
        "hatch.linewidth": 1.8,
    })


def panel_shade(ax, grid_axis="y"):
    """White panel, light dashed grid, full black box (reference treatment)."""
    ax.set_facecolor("white")
    ax.grid(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#CCCCCC", linewidth=0.8,
                linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("black")
        s.set_linewidth(1.2)


def soft_cmap(base, name):
    """Desaturated heatmap ramps (Reds_soft/Blues_soft): HSV saturation x0.70,
    top trimmed at 0.90 so the dark end never goes near-black."""
    import matplotlib.colors as mcolors
    xs = np.linspace(0.0, 0.90, 256)
    rgba = plt.get_cmap(base)(xs)
    hsv = mcolors.rgb_to_hsv(rgba[:, :3])
    hsv[:, 1] *= 0.70
    rgba[:, :3] = mcolors.hsv_to_rgb(hsv)
    return mcolors.ListedColormap(rgba, name=name)


def draw_heatmap(ax, mat, row_labels, col_labels, cmap, fmt="{:.2f}",
                 vmin=0, vmax=None, gamma=None, fontsize=13.0,
                 ticksize=16.0):
    """RQ2-style annotated heatmap (bold cell values, regular ticks)."""
    data = mat.astype(float)
    masked = np.ma.masked_invalid(data.values)
    cmap = cmap.copy()
    cmap.set_bad("#DDDDDD")
    ax.grid(False)
    if gamma is not None:
        norm = matplotlib.colors.PowerNorm(gamma, vmin=vmin, vmax=vmax)
        im = ax.imshow(masked, cmap=cmap, aspect="auto", norm=norm)
    else:
        im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=ticksize)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=ticksize)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.values[i, j]
            if np.isnan(v):
                continue
            color = "white" if im.norm(v) > 0.62 else "black"
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=fontsize, fontweight="bold", color=color)
    return im


def suite_divider(ax, cols, band_fs=17):
    """Black divider + suite band labels above an RQ2-style heatmap."""
    n_rc = sum(1 for s, _ in cols if s == "RoboCasa")
    ax.axvline(n_rc - 0.5, color="black", linewidth=1.8)
    for lo, hi, name in ((0, n_rc, "RoboCasa"), (n_rc, len(cols), "LIBERO")):
        ax.text((lo + hi) / 2 / len(cols), 1.015, SUITE_DISPLAY[name],
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=band_fs)


def fit_stats(x, y):
    import scipy.stats as sps
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.ptp(x) == 0:
        return None
    lr = sps.linregress(x, y)
    rho, rho_p = sps.spearmanr(x, y)
    return {"n": len(x), "slope": lr.slope, "intercept": lr.intercept,
            "pearson_r": lr.rvalue, "pearson_p": lr.pvalue,
            "spearman_rho": float(rho), "spearman_p": float(rho_p)}


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else \
        "*" if p < 0.05 else "n.s."


def fit_label(name, st):
    s = stars(st["pearson_p"])
    s = s if s != "n.s." else " n.s."
    return f"{name}: $r$={st['pearson_r']:+.2f}{s} ($n$={st['n']})"


def save(fig, here, stem):
    import os
    for ext in ("png", "pdf"):
        out = os.path.join(here, f"{stem}.{ext}")
        fig.savefig(out, dpi=300 if ext == "png" else None)
        print(f"[fig] {out}")
