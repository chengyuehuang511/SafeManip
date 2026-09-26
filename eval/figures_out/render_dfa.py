#!/usr/bin/env python3
"""Tiny cute-but-RIGOROUS DFA badge for the teaser's panel (2)
(SafeManip-teaser.png): the COMPLETE 3-state monitor DFA of the paper's
phi_2 template G(a -> (b U c)) with a=ObjGrasped, b=StableGrasp,
c=ObjReleased (legend at the bottom; full guards kept, user 2026-09-25:
"keep a ^ b ... don't be too abstract"). Styled per the
qualitative-figure-design skill: status colors + disc badges + glow
halos from predicate_hierarchy.py, and every guard label is a
WHITE-BACKED chip sitting ON its arc (the qualitative figure's
track-label style) -- also keeps text legible if the teaser composites
this onto a dark background. Tight margins, big fonts (user 2026-09-25).

    q_acc  (green double circle, check)  -- state 1: accepting, start state
        --[!a | c]--------> q_acc
        --[a & b & !c]----> q_pend
        --[a & !b & !c]---> q_viol       (unsafe the instant it opens)
    q_pend (amber, ellipsis)             -- strong-Until open; state 3 in the\n        compiled monitor DFA: NON-ACCEPTING but recoverable
        --[b & !c]--------> q_pend
        --[c]-------------> q_acc
        --[!b & !c]-------> q_viol
    q_viol (red, cross)                  -- state 2: the true violation trap

Transparent-background PNG (dpi 300) + PDF.
Run with ~/testnvme/miniconda3/bin/python (never /bin/python3).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))

GREEN = "#1FA054"
AMBER = "#E8A23D"
RED = "#E03131"
DARK = "#3A3A3A"


def tint(c, f):
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(c)
    return (1 - f * (1 - r), 1 - f * (1 - g), 1 - f * (1 - b))


def main():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "mathtext.fontset": "dejavusans",
    })
    W, H = 3.60, 2.00
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    R = 0.295
    yq = 1.02
    q0 = (0.56, yq)            # accepting
    q1 = (1.85, yq)            # pending (Until open)
    qv = (3.14, yq)            # violation trap

    def state(c, col, double=False):
        ax.add_patch(Circle(c, R, facecolor=tint(col, 0.14),
                            edgecolor=col, linewidth=2.0, zorder=3))
        if double:
            ax.add_patch(Circle(c, R - 0.048, facecolor="none",
                                edgecolor=col, linewidth=1.3, zorder=4))

    def badge(c, col, kind):
        ax.add_patch(Circle(c, 0.122, facecolor=col, edgecolor="white",
                            linewidth=0.9, zorder=5))
        if kind == "check":
            ax.plot([c[0] - 0.062, c[0] - 0.013, c[0] + 0.066],
                    [c[1], c[1] - 0.048, c[1] + 0.046], color="white",
                    lw=2.3, solid_capstyle="round",
                    solid_joinstyle="round", zorder=6)
        elif kind == "cross":
            for sx in (1, -1):
                ax.plot([c[0] - 0.050, c[0] + 0.050],
                        [c[1] - 0.050 * sx, c[1] + 0.050 * sx],
                        color="white", lw=2.3, solid_capstyle="round",
                        zorder=6)
        else:                          # pending ellipsis
            for dx in (-0.062, 0.0, 0.062):
                ax.add_patch(Circle((c[0] + dx, c[1]), 0.019,
                                    facecolor="white", zorder=6))

    def tarrow(p0, p1, rad, color=DARK, lw=1.7, shrink=23):
        ax.add_patch(FancyArrowPatch(
            p0, p1, connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=11, color=color,
            linewidth=lw, shrinkA=shrink, shrinkB=shrink, zorder=2))

    def self_loop(q, col):
        ax.add_patch(FancyArrowPatch(
            (q[0] - 0.095, q[1] + R - 0.030),
            (q[0] + 0.095, q[1] + R - 0.030),
            connectionstyle="arc3,rad=-2.1", arrowstyle="-|>",
            mutation_scale=9, color=col, linewidth=1.5, zorder=2))

    def chip(x, y, s, color=DARK, fs=9.5, pad=1.3):
        """White-backed guard chip beside its arc (track-label style)."""
        ax.text(x, y, s, fontsize=fs, ha="center", va="center",
                color=color, fontweight="bold", zorder=7,
                bbox=dict(facecolor="white", edgecolor="none", pad=pad))

    # states
    state(q0, GREEN, double=True)
    state(q1, AMBER)
    state(qv, RED)
    badge(q0, GREEN, "check")
    badge(q1, AMBER, "pending")
    badge(qv, RED, "cross")
    # state ids follow the COMPILED monitor DFA (user 2026-09-25):
    # 1 = accepting, 3 = non-accepting but RECOVERABLE, 2 = the true
    # violation trap. ALL state labels sit ABOVE the graph, each over
    # its own column (user 2026-09-25); state 3 wraps to two lines.
    for c, name, col in [(q0, "1 · accepting", GREEN),
                         (qv, "2 · violated", RED)]:
        ax.text(c[0], yq + R + 0.650, name, fontsize=9,
                ha="center", va="top", color=col, fontweight="bold",
                fontstyle="italic", zorder=7)
    ax.text(q1[0], yq + R + 0.650, "3 · non-accepting\n(recoverable)",
            fontsize=8.5, ha="center", va="top", color=AMBER,
            fontweight="bold", fontstyle="italic", linespacing=1.15,
            zorder=7)

    # initial-state marker (standard notation: arrow from nowhere), with
    # a tiny label so it doesn't read as a stray line at badge size
    tarrow((q0[0] - 0.53, q0[1]), q0, 0.0, shrink=3)
    ax.text(q0[0] - 0.425, q0[1] + 0.07, "start", fontsize=6.8,
            ha="center", va="center", color="#666666",
            fontstyle="italic", zorder=7)

    # self-loops, chips on the loop apex
    self_loop(q0, GREEN)
    chip(q0[0], yq + R + 0.30, r"$\neg a \vee c$", color="#177A41")
    self_loop(q1, AMBER)
    chip(q1[0], yq + R + 0.30, r"$b \wedge \neg c$", color="#B07A20")
    self_loop(qv, RED)
    chip(qv[0], yq + R + 0.30, r"$\mathsf{true}$", color=RED)

    # q0 <-> q1
    tarrow(q0, q1, -0.26)
    chip((q0[0] + q1[0]) / 2, yq + 0.325,
         r"$a \wedge b \wedge \neg c$")
    tarrow(q1, q0, -0.26)
    chip((q0[0] + q1[0]) / 2, yq - 0.27, r"$c$", pad=0.6)

    # q1 -> viol
    tarrow(q1, qv, -0.26, color=RED)
    chip((q1[0] + qv[0]) / 2, yq + 0.325,
         r"$\neg b \wedge \neg c$", color=RED)

    # q0 -> viol directly: unsafe the very step the obligation opens --
    # deep bottom arc, chip on its apex
    tarrow(q0, qv, 0.42, color=RED, shrink=24)
    chip((q0[0] + qv[0]) / 2, yq - 0.665,
         r"$a \wedge \neg b \wedge \neg c$", color=RED)

    # atom legend (rigor without abbreviation ambiguity)
    ax.text(W / 2, 0.045,
            r"$a=\mathsf{ObjGrasped}\;\;\; b=\mathsf{StableGrasp}"
            r"\;\;\; c=\mathsf{ObjReleased}$",
            fontsize=8.6, ha="center", va="bottom", color="#555555",
            zorder=7,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))

    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"dfa_grasp.{ext}")
        fig.savefig(out, dpi=300 if ext == "png" else None,
                    transparent=True)
        print(f"[fig] {out}")


if __name__ == "__main__":
    main()
