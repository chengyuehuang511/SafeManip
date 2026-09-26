#!/usr/bin/env python3
"""Compact standalone ManipVerse figure for a paper SUBSECTION
-> ../manipverse_small.{png,pdf}.

COMPREHENSIVE tree (user 2026-09-25: "include the ones in gray text as
well to show everything ... avoid (liquid)"): every sense of libero:milk
from manipverse_wordnet.json is a NODE -- no gray "also:" tail, no
"(liquid)" annotation (not in the WordNet view's data):
    kind:       food -> {dairy, drink}
    hazard:     spill (direct)
    affordance: thermal_treatment -> {cookable, freezable, fridgable}
                manipulation      -> {graspable, pourable}
                cleaning          -> {washable}
    role:       movable (direct)
VERTICAL left-to-right layout (user 2026-09-25: the staggered same-level
row read confusingly): milk on the LEFT fans to a single COLUMN of all
ten leaf synsets, then hypernym groups, then dimension roots on the
right -- one row per leaf, no stagger.

Same visual language as render_manipverse_example.py: flat borderless
pastel pills, teal/magenta/gold/indigo accents, WordNet is-a arrows
(hyponym -> hypernym), milk carton icon, stats footer.

Run with ~/testnvme/miniconda3/bin/python (never /bin/python3).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
FIG_W, FIG_H = 4.36, 3.22

KIND_TEAL = "#00A6A0"
HAZARD_MAGENTA = "#9E3E69"
AFFORD_GOLD = "#9B8420"
ROLE_INDIGO = "#4D5BB5"
TEXT_DARK = "#333333"
TEXT_MUTED = "#8A8A8A"
CARTON_EDGE = "#6B7A8F"


def apply_style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "mathtext.fontset": "dejavusans",
    })


def tint(c, f):
    r, g, b = mcolors.to_rgb(c)
    return (1 - f * (1 - r), 1 - f * (1 - g), 1 - f * (1 - b))


def shade(c, f):
    r, g, b = mcolors.to_rgb(c)
    return (r * (1 - f), g * (1 - f), b * (1 - f))


def icon_milk(fig, cx, cy, w):
    """cx/cy in inches on the W x H canvas."""
    ax = fig.add_axes([(cx - w / 2) / FIG_W, (cy - w / 2) / FIG_H,
                       w / FIG_W, w / FIG_H], frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_visible(False)
    ax.set_zorder(8)
    ax.add_patch(Polygon([(0.26, 0.08), (0.74, 0.08), (0.74, 0.62),
                          (0.26, 0.62)], closed=True, facecolor="white",
                         edgecolor=CARTON_EDGE, linewidth=1.5))
    ax.add_patch(Polygon([(0.26, 0.62), (0.5, 0.86), (0.74, 0.62)],
                         closed=True, facecolor=tint(CARTON_EDGE, 0.22),
                         edgecolor=CARTON_EDGE, linewidth=1.5,
                         joinstyle="round"))
    ax.plot([0.5, 0.5], [0.66, 0.82], color=CARTON_EDGE, linewidth=1.1)
    ax.add_patch(Polygon([(0.26, 0.30), (0.74, 0.30), (0.74, 0.46),
                          (0.26, 0.46)], closed=True,
                         facecolor=tint(KIND_TEAL, 0.30), edgecolor="none"))
    ax.add_patch(Circle((0.5, 0.375), 0.055, facecolor="white",
                        edgecolor=CARTON_EDGE, linewidth=0.9))


def main():
    apply_style()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.canvas.draw()

    def pill(cx, cy, text, accent, bold=False, fs=9.6):
        t = ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                    color=shade(accent, 0.22),
                    fontweight="bold" if bold else "normal", zorder=6)
        bb = t.get_window_extent(fig.canvas.get_renderer())
        inv = ax.transData.inverted()
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        px, py = 0.055, 0.045
        ax.add_patch(FancyBboxPatch(
            (x0 - px, y0 - py), (x1 - x0) + 2 * px, (y1 - y0) + 2 * py,
            boxstyle="round,pad=0.0,rounding_size=0.05",
            facecolor=tint(accent, 0.15), edgecolor="none", linewidth=0,
            zorder=5))
        return (x0 - px, x1 + px, y0 - py, y1 + py)

    def edge(p_from, p_to, color, lw=1.0, rad=0.0):
        ax.add_patch(FancyArrowPatch(
            p_from, p_to, color=color, linewidth=lw, arrowstyle="-|>",
            mutation_scale=6, shrinkA=0, shrinkB=0, zorder=3,
            connectionstyle=f"arc3,rad={rad}"))

    def top(bb):
        return ((bb[0] + bb[1]) / 2, bb[3])

    def bot(bb):
        return ((bb[0] + bb[1]) / 2, bb[2])

    # columns: milk | leaves | hypernym groups | dimension roots
    X_LEAF, X_MID, X_ROOT = 1.38, 2.55, 3.80

    def lft(bb):
        return (bb[0], (bb[2] + bb[3]) / 2)

    def rgt(bb):
        return (bb[1], (bb[2] + bb[3]) / 2)

    def lft_at(bb, dy):
        """Left-edge anchor offset vertically -- converging arrowheads
        collide when they all land on the exact same point."""
        return (bb[0], (bb[2] + bb[3]) / 2 + dy)

    roots = {}
    for cy, name, a in ((2.90, "KIND", KIND_TEAL),
                        (2.42, "HAZARD", HAZARD_MAGENTA),
                        (1.13, "AFFORDANCE", AFFORD_GOLD),
                        (0.22, "ROLE", ROLE_INDIGO)):
        t = ax.text(X_ROOT, cy, name, ha="center", va="center",
                    fontsize=10, color=shade(a, 0.15), fontweight="bold",
                    zorder=6)
        bb = t.get_window_extent(fig.canvas.get_renderer())
        inv = ax.transData.inverted()
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        roots[name] = (x0 - 0.04, x1, y0, y1)

    # hypernym groups (from manipverse_wordnet.json)
    food = pill(X_MID, 2.90, "food", KIND_TEAL)
    thermal = pill(X_MID, 1.80, "thermal", AFFORD_GOLD)
    manip = pill(X_MID, 1.04, "manipulation", AFFORD_GOLD)
    clean = pill(X_MID, 0.56, "cleaning", AFFORD_GOLD)

    # leaves: ALL ten most-specific synsets, one row each
    leaves = {}
    for cy, name, acc in ((3.02, "dairy", KIND_TEAL),
                          (2.76, "drink", KIND_TEAL),
                          (2.42, "spill", HAZARD_MAGENTA),
                          (2.08, "cookable", AFFORD_GOLD),
                          (1.80, "freezable", AFFORD_GOLD),
                          (1.52, "fridgable", AFFORD_GOLD),
                          (1.18, "graspable", AFFORD_GOLD),
                          (0.90, "pourable", AFFORD_GOLD),
                          (0.56, "washable", AFFORD_GOLD),
                          (0.22, "movable", ROLE_INDIGO)):
        leaves[name] = pill(X_LEAF, cy, name, acc, bold=True)

    # leaf -> hypernym / root edges (is-a arrows point RIGHT)
    edge(rgt(food), lft(roots["KIND"]), KIND_TEAL)
    edge(rgt(leaves["dairy"]), lft_at(food, 0.045), KIND_TEAL,
         rad=-0.10)
    edge(rgt(leaves["drink"]), lft_at(food, -0.045), KIND_TEAL,
         rad=0.10)
    edge(rgt(leaves["spill"]), lft(roots["HAZARD"]), HAZARD_MAGENTA)
    edge(rgt(thermal), lft_at(roots["AFFORDANCE"], 0.055),
         AFFORD_GOLD, rad=-0.14)
    edge(rgt(manip), lft(roots["AFFORDANCE"]), AFFORD_GOLD,
         rad=0.06)
    edge(rgt(clean), lft_at(roots["AFFORDANCE"], -0.055),
         AFFORD_GOLD, rad=0.14)
    edge(rgt(leaves["cookable"]), lft_at(thermal, 0.05),
         AFFORD_GOLD, rad=-0.10)
    edge(rgt(leaves["freezable"]), lft(thermal), AFFORD_GOLD)
    edge(rgt(leaves["fridgable"]), lft_at(thermal, -0.05),
         AFFORD_GOLD, rad=0.10)
    edge(rgt(leaves["graspable"]), lft_at(manip, 0.045),
         AFFORD_GOLD, rad=-0.08)
    edge(rgt(leaves["pourable"]), lft_at(manip, -0.045),
         AFFORD_GOLD, rad=0.08)
    edge(rgt(leaves["washable"]), lft(clean), AFFORD_GOLD)
    edge(rgt(leaves["movable"]), lft(roots["ROLE"]), ROLE_INDIGO)

    # milk on the LEFT, fanning to every most-specific synset
    icon_milk(fig, 0.28, 1.80, 0.36)
    ax.text(0.28, 1.48, "milk", ha="center", va="center", fontsize=13,
            color=TEXT_DARK, fontweight="bold", zorder=6)
    mp = (0.52, 1.64)
    for name, rad in (("dairy", -0.28), ("drink", -0.18),
                      ("spill", -0.10), ("cookable", -0.05),
                      ("freezable", -0.02), ("fridgable", 0.02),
                      ("graspable", 0.05), ("pourable", 0.09),
                      ("washable", 0.14), ("movable", 0.24)):
        lb = leaves[name]
        col = {"dairy": KIND_TEAL, "drink": KIND_TEAL,
               "spill": HAZARD_MAGENTA, "movable": ROLE_INDIGO}.get(
                   name, AFFORD_GOLD)
        edge(mp, lft(lb), col, rad=rad)

    for ext in ("png", "pdf"):
        out = os.path.join(OUT, f"manipverse_small.{ext}")
        fig.savefig(out, dpi=300 if ext == "png" else None)
        print(f"[fig] {out}")


if __name__ == "__main__":
    main()
