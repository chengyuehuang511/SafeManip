#!/usr/bin/env python3
"""ManipVerse showcase figure: one representative object (milk), four
faceted-hierarchy dimensions -- compact WordNet-style hypernym GRAPH
(user 2026-09-25: "make the figure a word net style"): synset nodes with
is-a edges (arrows point hyponym -> hypernym, WordNet convention), the
object at the bottom attaching to its most-specific synsets, and the
food -> dairy / food -> drink diamond making multiple hypernymy visible
as graph structure rather than a text note.

Deliberately styled apart from predicate_hierarchy.{png,pdf} (user
2026-09-25: "a little different from the hierarchy predicate ... very
compact"): flat borderless pastel tiles instead of white cards with
saturated borders + header strips, no chips, and a distinct accent family
(teal / olive-gold / magenta / indigo instead of blue / green / red /
purple). Palette re-validated with the ported dataviz checks: min pairwise
min(protan,deutan) OKLab dE = 12.6 (>= 8), normal-vision floor 17.3
(>= 15), all >= 3:1 on white, OKLCH L in [0.43, 0.77], C >= 0.10.

Object is milk (LIBERO): the one showcase entity with real senses in all
four dimensions AND WordNet-style multiple hypernymy in `kind`
(food -> dairy AND food -> drink). Data mirrors manipverse_wordnet.json's
libero:milk entry exactly. (Egg was dropped: robocasa metadata says
microwavable=True, but a real egg explodes in a microwave -- wrong thing
to showcase.)

Run with ~/miniconda3/bin/python (never /bin/python3).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                Polygon)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_W, FIG_H = 7.2, 2.55

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


def tile(fig, x, y, w, h, accent):
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.02",
        transform=fig.transFigure, facecolor=tint(accent, 0.12),
        edgecolor="none", linewidth=0, mutation_aspect=FIG_W / FIG_H,
        zorder=2))


def sense_line(fig, x, y, path, leaf, accent, fs=8.0):
    """Muted hypernym prefix + bold accent leaf, left-aligned at x.
    The prefix width is MEASURED off the Agg renderer, not estimated --
    estimates collided the two runs (first render, 2026-09-25)."""
    if path:
        t = fig.text(x, y, path, ha="left", va="center", fontsize=fs,
                     color=TEXT_MUTED, zorder=4)
        fig.canvas.draw()
        bb = t.get_window_extent(fig.canvas.get_renderer())
        x = fig.transFigure.inverted().transform((bb.x1, 0))[0] + 0.006
    fig.text(x, y, leaf, ha="left", va="center", fontsize=fs,
             color=shade(accent, 0.18), fontweight="bold", zorder=4)


def icon_milk(fig, cx, cy, w):
    h = w * FIG_W / FIG_H
    ax = fig.add_axes([cx - w / 2, cy - h / 2, w, h], frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_visible(False)
    ax.set_zorder(8)
    # carton body
    ax.add_patch(Polygon([(0.26, 0.08), (0.74, 0.08), (0.74, 0.62),
                          (0.26, 0.62)], closed=True, facecolor="white",
                         edgecolor=CARTON_EDGE, linewidth=1.6))
    # gabled top
    ax.add_patch(Polygon([(0.26, 0.62), (0.5, 0.86), (0.74, 0.62)],
                         closed=True, facecolor=tint(CARTON_EDGE, 0.22),
                         edgecolor=CARTON_EDGE, linewidth=1.6,
                         joinstyle="round"))
    ax.plot([0.5, 0.5], [0.66, 0.82], color=CARTON_EDGE, linewidth=1.2)
    # label band + drop
    ax.add_patch(Polygon([(0.26, 0.30), (0.74, 0.30), (0.74, 0.46),
                          (0.26, 0.46)], closed=True,
                         facecolor=tint(KIND_TEAL, 0.30), edgecolor="none"))
    ax.add_patch(Circle((0.5, 0.375), 0.055, facecolor="white",
                        edgecolor=CARTON_EDGE, linewidth=1.0))


def main():
    apply_style()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")
    fig.canvas.draw()

    def pill(cx, cy, text, accent, bold=False, fs=8.2, filled=True):
        """Synset node: measured text with a flat pastel pill behind it.
        Returns (x_left, x_right, y_bottom, y_top) in figure coords."""
        t = fig.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                     color=shade(accent, 0.22) if filled else TEXT_MUTED,
                     fontweight="bold" if bold else "normal", zorder=6)
        bb = t.get_window_extent(fig.canvas.get_renderer())
        inv = fig.transFigure.inverted()
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        px, py = 0.011, 0.030
        if filled:
            fig.patches.append(FancyBboxPatch(
                (x0 - px, y0 - py), (x1 - x0) + 2 * px, (y1 - y0) + 2 * py,
                boxstyle="round,pad=0.0,rounding_size=0.014",
                transform=fig.transFigure, facecolor=tint(accent, 0.15),
                edgecolor="none", linewidth=0,
                mutation_aspect=FIG_W / FIG_H, zorder=5))
        return (x0 - px, x1 + px, y0 - py, y1 + py)

    def edge(p_from, p_to, color, lw=1.1, rad=0.0):
        """is-a edge, arrow points at the hypernym (WordNet convention)."""
        fig.add_artist(FancyArrowPatch(
            p_from, p_to, transform=fig.transFigure,
            color=color, linewidth=lw, arrowstyle="-|>", mutation_scale=7,
            shrinkA=0, shrinkB=0, zorder=3,
            connectionstyle=f"arc3,rad={rad}"))

    def top(bb):
        return ((bb[0] + bb[1]) / 2, bb[3])

    def bot(bb):
        return ((bb[0] + bb[1]) / 2, bb[2])

    # --- layer y's: roots on top, milk at the bottom (WordNet reads up) ---
    Y_ROOT, Y_MID, Y_LEAF, Y_OBJ = 0.865, 0.635, 0.40, 0.135

    # lane headers (the four dimension roots)
    roots = {}
    for cx, name, a in ((0.175, "KIND", KIND_TEAL),
                        (0.425, "HAZARD", HAZARD_MAGENTA),
                        (0.645, "AFFORDANCE", AFFORD_GOLD),
                        (0.875, "ROLE", ROLE_INDIGO)):
        t = fig.text(cx, Y_ROOT, name, ha="center", va="center",
                     fontsize=7.8, color=shade(a, 0.15), fontweight="bold",
                     zorder=6)
        bb = t.get_window_extent(fig.canvas.get_renderer())
        inv = fig.transFigure.inverted()
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        roots[name] = (x0, x1, y0 - 0.01, y1)

    # --- kind lane: the multiple-hypernymy diamond ---
    food = pill(0.175, Y_MID, "food", KIND_TEAL)
    dairy = pill(0.10, Y_LEAF, "dairy", KIND_TEAL, bold=True)
    drink = pill(0.25, Y_LEAF, "drink", KIND_TEAL, bold=True)
    edge(top(food), bot(roots["KIND"]), KIND_TEAL)
    edge(top(dairy), (food[0] + 0.012, food[2]), KIND_TEAL, rad=0.12)
    edge(top(drink), (food[1] - 0.012, food[2]), KIND_TEAL, rad=-0.12)

    # --- hazard lane ---
    spill = pill(0.425, Y_LEAF, "spill", HAZARD_MAGENTA, bold=True)
    edge(top(spill), bot(roots["HAZARD"]), HAZARD_MAGENTA)
    fig.text(0.405, Y_LEAF - 0.095, "(liquid)", ha="center", va="center",
             fontsize=6.2, color=TEXT_MUTED, style="italic", zorder=6)

    # --- affordance lane ---
    manip = pill(0.645, Y_MID, "manipulation", AFFORD_GOLD)
    pour = pill(0.645, Y_LEAF, "pourable", AFFORD_GOLD, bold=True)
    edge(top(manip), bot(roots["AFFORDANCE"]), AFFORD_GOLD)
    edge(top(pour), bot(manip), AFFORD_GOLD)

    # --- role lane ---
    mov = pill(0.875, Y_LEAF, "movable object", ROLE_INDIGO, bold=True)
    edge(top(mov), bot(roots["ROLE"]), ROLE_INDIGO)

    # --- the object: milk, attaching to its most-specific synsets ---
    icon_milk(fig, 0.462, Y_OBJ + 0.015, 0.052)
    fig.text(0.505, Y_OBJ, "milk", ha="left", va="center",
             fontsize=11.5, color=TEXT_DARK, fontweight="bold", zorder=6)
    y_anchor = 0.245  # just above the carton roof, so no edge crosses it

    edge((0.44, y_anchor), bot(dairy), KIND_TEAL, lw=1.3, rad=-0.18)
    edge((0.455, y_anchor), bot(drink), KIND_TEAL, lw=1.3, rad=-0.10)
    edge((0.47, y_anchor), bot(spill), HAZARD_MAGENTA, lw=1.3, rad=0.06)
    edge((0.525, y_anchor), bot(pour), AFFORD_GOLD, lw=1.3, rad=0.14)
    edge((0.55, y_anchor), bot(mov), ROLE_INDIGO, lw=1.3, rad=0.22)

    # footers: milk's remaining affordances (left), ontology stats (right)
    fig.text(0.01, 0.035,
             "milk also:  graspable · cookable · fridgable · freezable "
             "· washable",
             ha="left", va="center", fontsize=6.6, color=TEXT_MUTED,
             style="italic", zorder=6)
    fig.text(0.99, 0.035,
             "ManipVerse · 262 entities · 60 properties · "
             "1,569 associations · 4 dimensions",
             ha="right", va="center", fontsize=6.6, color=TEXT_MUTED,
             zorder=6)

    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(os.path.join(HERE, f"manipverse_example.{ext}"),
                    facecolor="white", bbox_inches="tight", **kw)
    print("wrote manipverse_example.png / .pdf")


if __name__ == "__main__":
    main()
