#!/usr/bin/env python3
"""Predicate-hierarchy companion figure for qualitative_monitor.{png,pdf}.

Sits LEFT of the qualitative figure on one paper line: hierarchy 0.3 of the
line, qualitative 0.7, same rendered height (user 2026-09-24; ratio moved
1/3 -> 2/5 -> 0.3). Qualitative is 11.0x9.0, so this is 11*0.3/0.7 x 9 =
4.71x9.0 -- at that width ratio and the same height the two print at the
SAME physical scale. PNG at dpi=300 + vector PDF.

At 0.3 of the line the figure is a SINGLE-COLUMN stack (two side-by-side
chains no longer fit at print-matched fonts): full-width property cards,
stacked predicate groups with one-line boxes (name left, definition right),
and a compact atomic-chip grid. Chain identity is carried by the accent
colors (blue grasp / purple containment, the qualitative figure's
enabled/end colors) and the twin colored arrows between bands. Tight
margins and small inter-band gaps (user 2026-09-24: "gaps between blocks
are too large ... margin ... too large"), fonts/icons enlarged in the same
pass.

ICONS are hand-drawn vector mini-glyphs (user 2026-09-24: Unicode glyphs
"not cute enough" vs hierarchy_reference.png) -- each icon is a tiny
square axes with rounded two-tone patches in the element's accent color:
gripper-holding-a-ball for grasp, a trapezoid cup with a falling drop for
containment (the earlier flat-rounded mug read as flipped, user
2026-09-24), burst / arrow-into-box / germ / play / gear / door for the
collapsed categories, and crosshair / cube / tag / grid / contact-circles
for the atomic signals.

No header row -- the figure opens directly with the LTLf band. Content
mirrors the reference hierarchy in the paper's style + naming
(3a_tab_formulas.tex): three bands, flow bottom -> top,
    Atomic signals & properties  ->  Compositional predicates
                                 ->  LTLf safety properties (phi_1..phi_10)
The two categories UNFOLDED are the two the qualitative figure shows --
Grasp Stability (phi_2) and Containment (phi_7, bounded F_{<=100}) -- with
paper atom names; the other six categories are collapsed chips. Predicate
definitions come from the monitor code (SafeManip/monitor/sim/robocasa/
predicates.py): object_grasped = bilateral finger contact ^ gripper closed;
object_sync = object slip w.r.t. rigid eef attachment < eps;
object_grasped_raw ends = grasp contact ends; liquid_transfer_event =
dispense/dump onset ^ content is liquid; liquid_settled = supported ^
stable ^ support-type-matches (InIntendedReceiver in the qualitative cell).

Run with ~/miniconda3/bin/python (never /bin/python3).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                Polygon, Rectangle)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_W, FIG_H = 4.71, 9.0

VIOL_RED = "#D64545"
ENAB_BLUE = "#3E78B2"       # grasp chain accent (qualitative's enabled)
END_PURPLE = "#7A5AA0"      # containment chain accent (qualitative's end)
TRUE_GREEN = "#4C9A62"
CHIP_BG = "#D6E2F5"         # figure-intro's light blue header chip
TEXT_DARK = "#333333"
BAND_BG = "#F5F5F5"         # layer container fill
BAND_EC = "#BBBBBB"
ATOM_C = "#5A6E85"          # atomic icon color


def apply_style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "mathtext.fontset": "dejavusans",
    })


def tint(c, f):
    """Mix color c with white: f=0 -> white, f=1 -> c (no alpha, so the
    band background never shows through)."""
    r, g, b = mcolors.to_rgb(c)
    return (1 - f * (1 - r), 1 - f * (1 - g), 1 - f * (1 - b))


def rbox(fig, x, y, w, h, fc, ec, lw=0.9, rs=0.008, z=2, ls="-"):
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={rs}",
        transform=fig.transFigure, facecolor=fc, edgecolor=ec,
        linewidth=lw, linestyle=ls, mutation_aspect=FIG_W / FIG_H,
        zorder=z))


def hexbox(fig, x0, y0, w, h, fc, ec, lw=1.0, z=3, cut=0.016):
    """Flat-top hexagon (condition shape) -- the sub-predicate
    silhouette, distinct from the stadium parent pills."""
    ym = y0 + h / 2
    y1 = y0 + h
    pts = [(x0 + cut, y0), (x0 + w - cut, y0), (x0 + w, ym),
           (x0 + w - cut, y1), (x0 + cut, y1), (x0, ym)]
    fig.patches.append(Polygon(
        pts, closed=True, transform=fig.transFigure, facecolor=fc,
        edgecolor=ec, linewidth=lw, joinstyle="round", zorder=z))


def arrow(fig, p0, p1, color, lw=2.2, z=4):
    fig.add_artist(FancyArrowPatch(
        p0, p1, transform=fig.transFigure, color=color, linewidth=lw,
        arrowstyle="-|>", mutation_scale=14, shrinkA=0, shrinkB=0,
        zorder=z, connectionstyle="arc3,rad=0.0"))


def band_arrows(fig, y0, y1, xg, xc):
    """Twin colored arrows between bands: the two unfolded chains,
    aligned with the activation-tree centers."""
    arrow(fig, (xg, y0), (xg, y1), ENAB_BLUE)
    arrow(fig, (xc, y0), (xc, y1), END_PURPLE)


# ------------------------------ icons ---------------------------------
# Each icon draws into a tiny SQUARE axes (unit coords) placed at a figure
# position; two-tone accent-colored patches, reference-figure style.

def icon_axes(fig, cx, cy, w):
    h = w * FIG_W / FIG_H          # square in inches
    ax = fig.add_axes([cx - w / 2, cy - h / 2, w, h], frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_visible(False)
    ax.set_zorder(8)               # above the fig.patches chips (z<=4)
    return ax


def i_gripper(ax, c):
    """Parallel-jaw gripper holding a ball."""
    ax.add_patch(Rectangle((0.43, 0.80), 0.14, 0.18, facecolor=c))
    ax.add_patch(FancyBboxPatch(
        (0.14, 0.66), 0.72, 0.17,
        boxstyle="round,pad=0,rounding_size=0.06", facecolor=c,
        edgecolor="none"))
    for x0 in (0.14, 0.70):
        ax.add_patch(FancyBboxPatch(
            (x0, 0.24), 0.16, 0.46,
            boxstyle="round,pad=0,rounding_size=0.06", facecolor=c,
            edgecolor="none"))
    ax.add_patch(Circle((0.5, 0.36), 0.155, facecolor=tint(c, 0.25),
                        edgecolor=c, linewidth=1.6))


def i_mug(ax, c):
    """Cup (trapezoid, unmistakably upright) with a drop falling in --
    the earlier flat-rounded mug read as flipped (user 2026-09-24)."""
    ax.add_patch(Circle((0.665, 0.35), 0.145, facecolor="none",
                        edgecolor=c, linewidth=2.6))
    ax.add_patch(Polygon([(0.12, 0.58), (0.62, 0.58), (0.55, 0.08),
                          (0.19, 0.08)], facecolor=c, joinstyle="round"))
    ax.add_patch(Polygon([(0.155, 0.545), (0.585, 0.545), (0.565, 0.44),
                          (0.175, 0.44)], facecolor=tint(c, 0.30)))
    ax.add_patch(Circle((0.37, 0.76), 0.075, facecolor=c))
    ax.add_patch(Polygon([(0.302, 0.785), (0.438, 0.785), (0.37, 0.99)],
                         facecolor=c))


def i_burst(ax, c):
    ang = np.linspace(0, 2 * np.pi, 17)[:-1] + np.pi / 16
    r = np.where(np.arange(16) % 2 == 0, 0.46, 0.20)
    pts = np.c_[0.5 + r * np.cos(ang), 0.5 + r * np.sin(ang)]
    ax.add_patch(Polygon(pts, facecolor=c))
    ax.add_patch(Circle((0.5, 0.5), 0.085, facecolor=tint(c, 0.15)))


def i_release(ax, c):
    """Arrow dropping into an open box, settling safely."""
    ax.add_patch(Polygon([(0.43, 0.95), (0.57, 0.95), (0.57, 0.62),
                          (0.68, 0.62), (0.5, 0.40), (0.32, 0.62),
                          (0.43, 0.62)], facecolor=c))
    ax.add_patch(Polygon([(0.18, 0.44), (0.18, 0.08), (0.82, 0.08),
                          (0.82, 0.44)], closed=False, facecolor="none",
                         edgecolor=c, linewidth=2.4,
                         joinstyle="round"))


def i_germ(ax, c):
    """Contamination germ: spiky blob with pores."""
    for k in range(8):
        a = k * np.pi / 4
        ax.plot([0.5 + 0.28 * np.cos(a), 0.5 + 0.44 * np.cos(a)],
                [0.5 + 0.28 * np.sin(a), 0.5 + 0.44 * np.sin(a)],
                color=c, lw=2.4, solid_capstyle="round")
    ax.add_patch(Circle((0.5, 0.5), 0.30, facecolor=c))
    ax.add_patch(Circle((0.42, 0.56), 0.06, facecolor=tint(c, 0.20)))
    ax.add_patch(Circle((0.58, 0.44), 0.045, facecolor=tint(c, 0.20)))


def i_play(ax, c):
    ax.add_patch(Circle((0.5, 0.5), 0.42, facecolor=c))
    ax.add_patch(Polygon([(0.40, 0.29), (0.40, 0.71), (0.74, 0.5)],
                         facecolor=tint(c, 0.10)))


def i_gear(ax, c):
    for k in range(8):
        tr = (mtransforms.Affine2D().rotate_deg_around(0.5, 0.5, k * 45)
              + ax.transData)
        ax.add_patch(Rectangle((0.435, 0.10), 0.13, 0.25, facecolor=c,
                               transform=tr))
    ax.add_patch(Circle((0.5, 0.5), 0.29, facecolor=c))
    ax.add_patch(Circle((0.5, 0.5), 0.115, facecolor=tint(c, 0.15)))


def i_door(ax, c):
    ax.add_patch(Rectangle((0.18, 0.08), 0.64, 0.84, facecolor="none",
                           edgecolor=c, linewidth=2.2))
    ax.add_patch(Polygon([(0.30, 0.115), (0.30, 0.885), (0.64, 0.80),
                          (0.64, 0.20)], facecolor=c))
    ax.add_patch(Circle((0.57, 0.50), 0.04, facecolor=tint(c, 0.15)))


def i_check(ax, c):
    """Stable hold: check in a circle."""
    ax.add_patch(Circle((0.5, 0.5), 0.42, facecolor=c))
    ax.plot([0.30, 0.44, 0.72], [0.50, 0.34, 0.66], color=tint(c, 0.10),
            lw=3.2, solid_capstyle="round", solid_joinstyle="round")


def i_drop(ax, c):
    ax.add_patch(Circle((0.5, 0.34), 0.27, facecolor=c))
    ax.add_patch(Polygon([(0.246, 0.42), (0.754, 0.42), (0.5, 0.96)],
                         facecolor=c))
    ax.add_patch(Circle((0.40, 0.30), 0.065, facecolor=tint(c, 0.25)))


def i_contact(ax, c):
    ax.add_patch(Circle((0.37, 0.5), 0.25, facecolor=tint(c, 0.45)))
    ax.add_patch(Circle((0.63, 0.5), 0.25, facecolor="none",
                        edgecolor=c, linewidth=2.2))


def i_uncontact(ax, c):
    """Contact broken: the two contact circles pulled apart."""
    ax.add_patch(Circle((0.26, 0.5), 0.19, facecolor=tint(c, 0.45)))
    ax.add_patch(Circle((0.74, 0.5), 0.19, facecolor="none",
                        edgecolor=c, linewidth=2.0))
    ax.plot([0.46, 0.54], [0.30, 0.42], color=c, lw=1.8,
            solid_capstyle="round")
    ax.plot([0.46, 0.54], [0.58, 0.70], color=c, lw=1.8,
            solid_capstyle="round")


def i_open(ax, c):
    """Gripper with jaws splayed wide, ball released below."""
    ax.add_patch(Rectangle((0.43, 0.80), 0.14, 0.18, facecolor=c))
    ax.add_patch(FancyBboxPatch(
        (0.10, 0.66), 0.80, 0.17,
        boxstyle="round,pad=0,rounding_size=0.06", facecolor=c,
        edgecolor="none"))
    for x0 in (0.06, 0.78):
        ax.add_patch(FancyBboxPatch(
            (x0, 0.30), 0.16, 0.40,
            boxstyle="round,pad=0,rounding_size=0.06", facecolor=c,
            edgecolor="none"))
    ax.add_patch(Circle((0.5, 0.20), 0.13, facecolor="none",
                        edgecolor=c, linewidth=1.8))


def i_crosshair(ax, c):
    ax.add_patch(Circle((0.5, 0.5), 0.26, facecolor="none", edgecolor=c,
                        linewidth=2.2))
    for p0, p1 in [((0.5, 0.90), (0.5, 0.68)), ((0.5, 0.32), (0.5, 0.10)),
                   ((0.10, 0.5), (0.32, 0.5)), ((0.68, 0.5), (0.90, 0.5))]:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=c, lw=2.2,
                solid_capstyle="round")
    ax.add_patch(Circle((0.5, 0.5), 0.055, facecolor=c))


def i_cube(ax, c):
    ax.add_patch(Polygon([(0.5, 0.92), (0.88, 0.70), (0.5, 0.48),
                          (0.12, 0.70)], facecolor=tint(c, 0.45)))
    ax.add_patch(Polygon([(0.12, 0.70), (0.5, 0.48), (0.5, 0.06),
                          (0.12, 0.28)], facecolor=tint(c, 0.75)))
    ax.add_patch(Polygon([(0.88, 0.70), (0.5, 0.48), (0.5, 0.06),
                          (0.88, 0.28)], facecolor=c))


def i_grid(ax, c):
    for i in range(3):
        for j in range(3):
            f = 1.0 if (i + j) % 2 == 0 else 0.40
            ax.add_patch(Rectangle((0.13 + i * 0.26, 0.13 + j * 0.26),
                                   0.22, 0.22, facecolor=tint(c, f)))


def i_tag(ax, c):
    ax.add_patch(Polygon([(0.14, 0.78), (0.52, 0.78), (0.88, 0.5),
                          (0.52, 0.22), (0.14, 0.22)], facecolor=c,
                         joinstyle="round"))
    ax.add_patch(Circle((0.30, 0.5), 0.065, facecolor=tint(c, 0.15)))


# -----------------------------------------------------------------------

def measure(fig, text, fontsize, weight="normal", style="normal"):
    """Width of a text string in figure fraction."""
    t = fig.text(0, 0, text, fontsize=fontsize, fontweight=weight,
                 fontstyle=style)
    w = (t.get_window_extent(renderer=fig.canvas.get_renderer()).width
         / fig.bbox.width)
    t.remove()
    return w


def icon_label(fig, xc, y, icon_fn, iw, text, fs, color, weight="bold",
               style="normal", icolor=None, gap=0.007):
    """Icon + label pair, centered as a UNIT at xc: total width =
    icon + gap + measured text, so a long label can never push the icon
    out of its chip (the old text-half-width offset did)."""
    tw = measure(fig, text, fs, weight, style)
    x = xc - (iw + gap + tw) / 2
    icon_fn(icon_axes(fig, x + iw / 2, y, iw), icolor or color)
    fig.text(x + iw + gap, y, text, fontsize=fs, fontweight=weight,
             fontstyle=style, ha="left", va="center", color=color,
             zorder=6)


def main():
    apply_style()
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    L, R = 0.020, 0.980
    W = R - L

    # saturated status colors (user 2026-09-25: reference ticks/crosses
    # pop more than the paper's muted green/red)
    # "p" = pending: the obligation is still open at the snapshot (an
    # Until not yet discharged) -- neither accept nor reject
    ST_COL = {"t": "#1FA054", "f": "#E03131", "n": "#8E959D",
              "p": "#E8A23D"}

    def badge_ax(fig_, cx, cy, status, w=0.026):
        col = ST_COL[status]
        s = w / 0.026               # scale strokes with badge size, or
        ax = icon_axes(fig_, cx, cy, w)  # small badges wash out
        ax.set_zorder(9)
        # reference-mock style (user 2026-09-25): the colored disc
        # dominates, the white glyph stays compact -- big radius, thin
        # ring, mark confined to the middle ~35%
        ax.add_patch(Circle((0.5, 0.5), 0.49, facecolor=col,
                            edgecolor="white", linewidth=0.7 * s))
        if status == "f":
            ax.plot([0.34, 0.66], [0.34, 0.66], color="white",
                    lw=2.1 * s, solid_capstyle="round")
            ax.plot([0.34, 0.66], [0.66, 0.34], color="white",
                    lw=2.1 * s, solid_capstyle="round")
        elif status == "p":            # pending: white ellipsis dots
            for dx in (-0.20, 0.0, 0.20):
                ax.add_patch(Circle((0.5 + dx, 0.5), 0.065,
                                    facecolor="white",
                                    edgecolor="none"))
        else:
            ax.plot([0.30, 0.455, 0.71], [0.50, 0.355, 0.645],
                    color="white", lw=2.1 * s, solid_capstyle="round",
                    solid_joinstyle="round")

    def halo(x, y, w, h, col):
        rbox(fig, x - 0.006, y - 0.005, w + 0.012, h + 0.010,
             tint(col, 0.30), tint(col, 0.30), lw=0, rs=0.012, z=2)

    def formula_row(xc, y, segs, fs, pad=0.0045):
        """Formula with per-atom activation colors: place measured
        mathtext segments left-to-right on one shared baseline. pad
        re-adds the inter-segment space mathtext trims from trailing
        `\\ ` when a segment is measured alone."""
        widths = [measure(fig, s, fs) for s, _ in segs]
        total = sum(widths) + pad * (len(segs) - 1)
        x = xc - total / 2
        for (s, c), w in zip(segs, widths):
            fig.text(x, y, s, fontsize=fs, ha="left", va="baseline",
                     color=c, zorder=6)
            x += w + pad

    # ---------------- band 3 (top): LTLf safety properties ---------------
    b3_y0, b3_y1 = 0.606, 0.988
    rbox(fig, L, b3_y0, W, b3_y1 - b3_y0, "white", BAND_EC, rs=0.010, z=1)
    fig.text(L + 0.018, b3_y1 - 0.010, "LTL$_f$ safety properties",
             fontsize=17, fontweight="bold", ha="left", va="top",
             color=TEXT_DARK, zorder=5)
    fig.text(R - 0.018, b3_y1 - 0.015,
             r"$\varphi_1$–$\varphi_{10}$",
             fontsize=12, ha="right", va="top", color="#777777",
             zorder=5)
    # collapsed categories: the six NOT unfolded, pastel chips with icons
    rows = [[(i_burst, "Collision & Contact", "#F8E0E0", "#A94442"),
             (i_release, "Release Stability", "#E2F0E6", "#3E7D50")],
            [(i_germ, "Cross-Contamination", "#EFE6F5", "#6C4E93"),
             (i_play, "Action-Onset", "#F9EFD8", "#9A7B24")],
            [(i_gear, "Mechanism", "#DFF0EF", "#357E79"),
             (i_door, "Enclosure & Access", "#FBE8DA", "#B0642B")]]
    cy = b3_y1 - 0.080
    for r, row in enumerate(rows):
        cw = (W - 0.026) / 2
        for c, (icon, name, fc, tc) in enumerate(row):
            x0 = L + 0.013 + c * cw
            y0 = cy - r * 0.047
            rbox(fig, x0 + 0.004, y0, cw - 0.008, 0.038, fc, tc,
                 lw=0.9, rs=0.006, z=2)
            icon_label(fig, x0 + cw / 2, y0 + 0.019, icon, 0.038,
                       name, 11.2, tc)
    # the two UNFOLDED property cards (the qualitative figure's cells),
    # full-width, stacked; ACTIVATION style (user 2026-09-25): overall
    # accept/reject badge in the header, formula atoms colored by their
    # truth value at the snapshot
    card_h, card_gap = 0.090, 0.010
    card_w = W - 0.026
    cx0 = L + 0.013
    T, F = ST_COL["t"], ST_COL["f"]
    cards = [
        (b3_y0 + 0.010 + card_h + card_gap, ENAB_BLUE, i_gripper,
         "Grasp Stability", "p",   # Until still open mid-grasp: PENDING
         [(r"$\varphi_2:\ \mathbf{G}($", TEXT_DARK),
          (r"$\mathsf{ObjGrasped}$", T),
          (r"$\ \rightarrow\ ($", TEXT_DARK),
          (r"$\mathsf{StableGrasp}$", T),
          (r"$\ \mathbf{U}\ $", TEXT_DARK),
          (r"$\mathsf{ObjReleased}$", F),
          (r"$))$", TEXT_DARK)]),
        (b3_y0 + 0.010, END_PURPLE, i_mug, "Containment", "f",
         [(r"$\varphi_7:\ \mathbf{G}($", TEXT_DARK),
          (r"$\mathsf{LiqTransfer}$", T),
          (r"$\ \rightarrow\ \mathbf{F}_{\leq 100}\ $", TEXT_DARK),
          (r"$\mathsf{LiqContained}$", F),
          (r"$)$", TEXT_DARK)]),
    ]
    for y0, accent, icon, name, verdict, segs in cards:
        rbox(fig, cx0, y0, card_w, card_h, "white", accent, lw=1.8,
             rs=0.008, z=3)
        xc = cx0 + card_w / 2
        rbox(fig, cx0 + 0.014, y0 + card_h - 0.042, card_w - 0.028,
             0.034, CHIP_BG, "#7C8DB0", lw=0.8, rs=0.005, z=4)
        # icon + title + accept/reject badge, centered as one unit
        iw, bw_, gp = 0.046, 0.036, 0.009
        tw = measure(fig, name, 14.5, "bold", "italic")
        x = xc - (iw + gp + tw + gp + bw_) / 2
        icon(icon_axes(fig, x + iw / 2, y0 + card_h - 0.025, iw), accent)
        fig.text(x + iw + gp, y0 + card_h - 0.025, name, fontsize=14.5,
                 fontweight="bold", fontstyle="italic", ha="left",
                 va="center", zorder=6)
        badge_ax(fig, x + iw + gp + tw + gp + bw_ / 2,
                 y0 + card_h - 0.025, verdict, w=bw_)
        formula_row(xc, y0 + 0.017, segs, 11.4)

    # ---------------- band 2 (middle): compositional predicates ----------
    # ACTIVATION TREES (user 2026-09-25, hierarchy_reference_2.png): the
    # leaf predicates carry live truth badges from the atomic signals at a
    # snapshot (the qualitative figure's stories), fan into an event node,
    # and light the per-chain status pill -- grasp satisfied, containment
    # violated (LiqContained shown false-red, not the mockup's gray: the
    # pill's X IS LiqContained staying false).
    b2_y0, b2_y1 = 0.248, 0.576
    rbox(fig, L, b2_y0, W, b2_y1 - b2_y0, "white", BAND_EC, rs=0.010, z=1)
    fig.text(L + 0.018, b2_y1 - 0.010, "Compositional predicates",
             fontsize=17, fontweight="bold", ha="left", va="top",
             color=TEXT_DARK, zorder=5)


    # TWO ROWS of decomposition TREES (user 2026-09-25): row 1 = the
    # grasp-stability atoms (phi_2), row 2 = the containment atoms
    # (phi_7). Each atom is a parent NODE pill; its REAL sub-predicates
    # from the monitor hang below as leaf chips with arrows up -- a
    # forest, not a flat list. (ST_COL / badge_ax / halo defined above,
    # shared with the property cards.)
    def tarrow(p0, p1, color):
        fig.add_artist(FancyArrowPatch(
            p0, p1, transform=fig.transFigure, color=color,
            linewidth=1.6, arrowstyle="-|>", mutation_scale=10,
            shrinkA=0, shrinkB=0, zorder=4))

    def small_leaf(x0, y0, w, h, icon_fn, name, status):
        # lower-tier style: SOLID but thinner border, sharper corners,
        # lighter fill, no halo (dashes read as "pending", user
        # 2026-09-25)
        col = ST_COL[status]
        rbox(fig, x0, y0, w, h, tint(col, 0.09), col, lw=1.1,
             rs=0.010, z=3)
        lines = name.split("\n")
        iw, gp = 0.026, 0.005
        fs = 8.8            # shrink until icon+text fits inside the chip
        tmax = max(measure(fig, ln, fs, "bold") for ln in lines)
        while fs > 7 and iw + gp + tmax > w - 0.020:
            fs -= 0.3
            tmax = max(measure(fig, ln, fs, "bold") for ln in lines)
        xs = x0 + w / 2 - (iw + gp + tmax) / 2
        icon_fn(icon_axes(fig, xs + iw / 2, y0 + h / 2, iw), col)
        xt = xs + iw + gp + tmax / 2
        yc = y0 + h / 2 + (len(lines) - 1) * 0.0078
        for i, line in enumerate(lines):
            fig.text(xt, yc - i * 0.0155, line, fontsize=fs,
                     fontweight="bold", ha="center", va="center",
                     color=col, zorder=6)
        badge_ax(fig, x0 + w - 0.008, y0 + h - 0.006, status,
                 w=0.030)
        return x0 + w / 2

    def node_pill(xc, y0, h, icon_fn, name, status, accent):
        col = ST_COL[status]
        iw, bw_, gp = 0.022, 0.024, 0.004
        tw = measure(fig, name, 9.8, "bold")
        total = iw + gp + tw + gp + bw_
        x0 = xc - total / 2 - 0.007
        rbox(fig, x0 - 0.003, y0 - 0.004, total + 0.020, h + 0.008,
             tint(col, 0.30), tint(col, 0.30), lw=0, rs=0.022, z=2)
        rbox(fig, x0, y0, total + 0.014, h, tint(col, 0.13), col,
             lw=1.5, rs=0.020, z=3)
        x = xc - total / 2
        icon_fn(icon_axes(fig, x + iw / 2, y0 + h / 2, iw), accent)
        fig.text(x + iw + gp, y0 + h / 2, name, fontsize=9.8,
                 fontweight="bold", ha="left", va="center",
                 color=TEXT_DARK, zorder=6)
        badge_ax(fig, x + iw + gp + tw + gp + bw_ / 2, y0 + h / 2,
                 status, w=bw_)

    def forest_row(leaf_y, pill_y, groups, accent):
        """One row of atom trees: parent pills over their leaf chips."""
        lw_, wg, lh, ph = 0.172, 0.008, 0.064, 0.044
        n_leaves = sum(len(g[3]) for g in groups)
        n_within = sum(len(g[3]) - 1 for g in groups)
        span = n_leaves * lw_ + n_within * wg
        bg = (W - 0.050 - span) / (len(groups) - 1)   # between-group gap
        x = L + 0.025
        for icon_fn, atom, a_st, leaves in groups:
            centers = []
            for l_icon, lname, l_st in leaves:
                centers.append(small_leaf(x, leaf_y, lw_, lh, l_icon,
                                          lname, l_st))
                x += lw_ + wg
            x += bg - wg
            gc = sum(centers) / len(centers)
            node_pill(gc, pill_y, ph, icon_fn, atom, a_st, accent)
            for lx in centers:
                tarrow((lx, leaf_y + lh + 0.006),
                       (gc + 0.4 * (lx - gc), pill_y), accent)

    # sub-predicates from monitor/sim/robocasa/predicates.py; states =
    # the qualitative snapshot (mid-grasp holding / coffee timeout f620:
    # ONLY the receiver conjunct of LiqContained fails)
    forest_row(0.406, 0.492, [
        (i_gripper, "ObjGrasped", "t",
         [(i_contact, "2-Finger\nContact", "t"),
          (i_gripper, "Gripper\nClosed", "t")]),
        (i_check, "StableGrasp", "t",
         [(i_crosshair, "Slip < ε", "t")]),
        (i_release, "ObjReleased", "f",
         [(i_uncontact, "Grasp\nEnds", "f"),
          (i_open, "Gripper\nOpening", "f")]),
    ], ENAB_BLUE)
    forest_row(0.260, 0.346, [
        (i_drop, "LiqTransfer", "t",
         [(i_play, "Dispense\nOnset", "t"),
          (i_drop, "Is Liquid", "t")]),
        (i_mug, "LiqContained", "f",
         [(i_tag, "InIntended\nReceiver", "f"),
          (i_cube, "Supported", "t"),
          (i_check, "Stable", "t")]),
    ], END_PURPLE)
    xg, xc_t = 0.5 - 0.06, 0.5 + 0.06   # band-gap flow arrows

    band_arrows(fig, b2_y1, b3_y0, xg, xc_t)   # predicates -> properties

    # ---------------- band 1 (bottom): atomic signals ---------------------
    b1_y0, b1_y1 = 0.012, 0.218
    rbox(fig, L, b1_y0, W, b1_y1 - b1_y0, BAND_BG, BAND_EC, rs=0.010, z=1)
    fig.text(L + 0.018, b1_y1 - 0.010, "Atomic signals & properties",
             fontsize=17, fontweight="bold", ha="left", va="top",
             color=TEXT_DARK, zorder=5)
    atoms = [[(i_contact, "Contact"), (i_gripper, "Gripper"),
              (i_crosshair, "EE pose"), (i_cube, "Obj pose·vel")],
             [(i_gear, "Fixtures"), (i_grid, "Regions"),
              (i_tag, "Semantics"), (None, "⋯")]]
    aw = (W - 0.030) / 4
    ah = 0.046
    ay = b1_y1 - 0.050 - ah
    for r, row in enumerate(atoms):
        for c, (icon, name) in enumerate(row):
            x0 = L + 0.015 + c * aw
            y0 = ay - r * (ah + 0.009)
            rbox(fig, x0 + 0.004, y0, aw - 0.008, ah,
                 "white", "#AAAAAA", lw=0.8, rs=0.002, z=2)
            if icon is not None:
                # center the icon+name PAIR -- left-aligning left a big
                # one-sided white margin in each chip (user 2026-09-24)
                icon_label(fig, x0 + aw / 2, y0 + ah / 2, icon, 0.034,
                           name, 10, "#444444", weight="normal",
                           icolor=ATOM_C, gap=0.005)
            else:
                fig.text(x0 + aw / 2, y0 + ah / 2, name, fontsize=12,
                         ha="center", va="center", color="#444444",
                         zorder=5)
    fig.text((L + R) / 2, b1_y0 + 0.007,
             "raw sim state & object attributes, per frame",
             fontsize=9.5, fontstyle="italic", ha="center", va="bottom",
             color="#888888", zorder=5)

    band_arrows(fig, b1_y1, b2_y0, xg, xc_t)   # signals -> predicates

    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"predicate_hierarchy.{ext}")
        fig.savefig(out, dpi=300 if ext == "png" else None)
        print(f"[fig] {out}")


if __name__ == "__main__":
    main()
