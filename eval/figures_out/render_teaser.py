#!/usr/bin/env python3
"""SafeManip pull/teaser figure (paper Figure 1) -> teaser_pull.{png,pdf}.

Structure follows ref/fig/high quality/pull/Figure1.pdf (RoboCasa365's pull
figure): a central brand chip with stat bullets left/right, capability
panels around it. Content = the four claims the teaser must carry (user
2026-09-25):
  1. wide range of safety  -> top band: EIGHT category cells, each a REAL
     monitored frame from a DIFFERENT RoboCasa365 task (so the same band
     also shows task/scene/object breadth), + a stacked LIBERO cell for
     suite breadth.
  2. broad coverage        -> same band + the stat bullets (50 + 40 tasks).
  3. ManipVerse            -> bottom-left mini panel (milk + dimension
     pills, teal/magenta/gold accents shared with manipverse_example).
  4. monitoring + metrics  -> bottom-center brief phi_2 example (3 frames +
     one timeline, MakeIceLemonade ep0 -- deliberately NOT the
     PreSoakPan/PrepareCoffee cells the qualitative figure uses) and
     bottom-right metric tiles (SR / R / E from 4_protocol.tex).

Style: qualitative-figure-design skill constants (DejaVu Sans, sparse
bold, category chip pastels, CHIP_BG brand chip, band bg); panel fills
mimic Figure1's pale green (top) / pale blue (bottom) split.

Run with ~/miniconda3/bin/python (never /bin/python3).
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from matplotlib.patches import (Circle, FancyArrowPatch,
                                FancyBboxPatch)
from matplotlib.patches import Ellipse as MEllipse
from matplotlib.patches import PathPatch as MPathPatch
from matplotlib.path import Path as MPath

HERE = os.path.dirname(os.path.abspath(__file__))
QF = os.path.join(HERE, "qual_frames")
TF = os.path.join(HERE, "manipverse", "teaser_frames")
FIG_W, FIG_H = 8.8, 3.6

VIOL_RED = "#D64545"
ENAB_BLUE = "#3E78B2"
END_PURPLE = "#7A5AA0"
TRUE_GREEN = "#4C9A62"
CHIP_BG = "#D6E2F5"
TEXT_DARK = "#333333"
TEXT_MUTED = "#777777"
TRACK_BG = "#EDEDED"
PAGE_BG = "#F2F5FA"     # soft blue-gray page wash (anti "too white")
PANEL_EC = "#B9C6DC"
SHADOW = (0.42, 0.50, 0.63, 0.20)
# ManipVerse accent family (manipverse_example.py)
MV_TEAL, MV_MAGENTA, MV_GOLD, MV_INDIGO = ("#00A6A0", "#9E3E69",
                                           "#9B8420", "#4D5BB5")


def shade_mv(c, f=0.18):
    r, g, b = mcolors.to_rgb(c)
    return (r * (1 - f), g * (1 - f), b * (1 - f))

# (category, chip text color, chip fill, filename in safety_categories/,
#  violation circle (cx, cy, r) in 256px image coords, vertical crop anchor)
# Frames = the drawio teaser's ORIGINAL clean images (user-located,
# eval/figures_out/safety_categories/); circle coords transferred exactly
# from teaser-individual.drawio.svg's hand-placed ellipses (corr>=0.99
# confirmed same frames), drawn as vector overlays.
CATEGORIES = [
    ("Collision & Contact", "#A94442", "#F8E0E0",
     "collision.png", (176, 129, 48), 0.5),
    ("Grasp Stability", ENAB_BLUE, "#DCE8F5",
     "grasp.png", (176, 129, 48), 0.5),
    ("Release Stability", "#3E7D50", "#E2F0E6",
     "release.png", (144, 161, 48), 0.5),
    ("Cross-Contamination", "#6C4E93", "#EFE6F5",
     "contamination.png", (151, 157, 48), 0.5),
    ("Action-Onset", "#9A7B24", "#F9EFD8",
     "precondition.png", (176, 116, 48), 0.5),
    ("Mechanism", "#357E79", "#DFF0EF",
     "mechanism.png", (176, 164, 48), 0.5),
    ("Containment", END_PURPLE, "#E8E0F0",
     "eas.png", (160, 113, 62), 0.5),
    ("Enclosure & Access", "#B0642B", "#FBE8DA",
     "containment.png", (192, 148, 35), 0.5),
]
SC_DIR = os.path.join(HERE, "safety_categories")


def apply_style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "mathtext.fontset": "dejavusans",
    })


def tint(c, f):
    r, g, b = mcolors.to_rgb(c)
    return (1 - f * (1 - r), 1 - f * (1 - g), 1 - f * (1 - b))


def rbox(fig, x, y, w, h, fc, ec="none", lw=0.0, rs=0.014, z=1):
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={rs}",
        transform=fig.transFigure, facecolor=fc, edgecolor=ec,
        linewidth=lw, mutation_aspect=FIG_W / FIG_H, zorder=z))


def panel(fig, x, y, w, h, z=1, rs=0.014):
    """White card with a soft blue-gray drop shadow (depth vs PAGE_BG)."""
    rbox(fig, x + 0.0022, y - 0.0055, w, h, SHADOW, rs=rs, z=z - 0.4)
    rbox(fig, x, y, w, h, "white", ec=PANEL_EC, lw=1.1, rs=rs, z=z)


def frame_ax(fig, x, y, w, h, path, z=4, crop_top=0, crop_frac=1.0,
             crop_anchor=0.5, circle=None):
    """circle = (cx, cy, r) in ORIGINAL image pixel coords; drawn as a
    vector dashed ellipse after the crop transform (stays crisp at any
    dpi, unlike baked-in raster circles)."""
    ax = fig.add_axes([x, y, w, h])
    im = Image.open(path).convert("RGB")
    y0 = crop_top
    if crop_top:
        im = im.crop((0, crop_top, im.width, im.height))
    if crop_frac < 1.0:
        win = int(im.height * crop_frac)
        top = int((im.height - win) * crop_anchor)
        y0 += top
        im = im.crop((0, top, im.width, top + win))
    ax.imshow(np.asarray(im), interpolation="lanczos")
    if circle is not None:
        cx, cy, r = circle
        ex, ey = cx / im.width, 1 - (cy - y0) / im.height
        rx, ry = r / im.width, r / im.height
        # spotlight: dim everything OUTSIDE the highlight
        th = np.linspace(0, 2 * np.pi, 64)
        hole = list(zip(ex + rx * np.cos(th), ey + ry * np.sin(th)))[::-1]
        verts = ([(-0.02, -0.02), (1.02, -0.02), (1.02, 1.02),
                  (-0.02, 1.02), (-0.02, -0.02)] + hole + [hole[0]])
        codes = ([MPath.MOVETO] + [MPath.LINETO] * 3 + [MPath.CLOSEPOLY]
                 + [MPath.MOVETO] + [MPath.LINETO] * (len(hole) - 1)
                 + [MPath.CLOSEPOLY])
        ax.add_patch(MPathPatch(MPath(verts, codes), facecolor="black",
                                alpha=0.22, edgecolor="none",
                                transform=ax.transAxes, zorder=5))
        # soft-glow ring: fading red halo -> thin white separator -> a
        # slender paper-red stroke (elegant, not a bold traffic ring)
        for lw, col, a in ((5.2, VIOL_RED, 0.10), (3.4, VIOL_RED, 0.20),
                           (2.1, "white", 0.95), (1.25, VIOL_RED, 1.0)):
            ax.add_patch(MEllipse((ex, ey), 2 * rx, 2 * ry, fill=False,
                                  edgecolor=mcolors.to_rgba(col, a),
                                  linewidth=lw,
                                  transform=ax.transAxes, zorder=6,
                                  clip_on=False))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("white")
        s.set_linewidth(1.2)
    ax.set_zorder(z)
    return ax


def find_frame(prefix, tag):
    hits = sorted(glob.glob(os.path.join(QF, f"{prefix}*__{tag}.png")))
    if not hits:
        raise FileNotFoundError(f"{prefix}*__{tag}.png")
    return hits[0]


def connector(fig, p0, p1, color):
    fig.add_artist(plt.Line2D([p0[0], p1[0]], [p0[1], p1[1]],
                              transform=fig.transFigure, color=color,
                              linewidth=1.6, zorder=2,
                              solid_capstyle="round"))
    for p, r in ((p0, 0.006), (p1, 0.004)):
        fig.add_artist(Circle(p, r, transform=fig.transFigure,
                              facecolor="white", edgecolor=color,
                              linewidth=1.4, zorder=3))


def main():
    apply_style()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(PAGE_BG)
    fig.canvas.draw()

    # ================= TOP BAND: 8 safety categories + LIBERO =============
    rbox(fig, 0.006, 0.559, 0.992, 0.427, SHADOW, z=0.6)
    rbox(fig, 0.004, 0.565, 0.992, 0.427, "#FBFCFE", ec=PANEL_EC,
         lw=1.1, z=1)
    fig.text(0.5, 0.952, "Temporal Safety Categories",
             ha="center", va="center", fontsize=11, color=TEXT_DARK,
             fontweight="bold", zorder=5)

    n = 8
    cw = (256 / 250) / FIG_W        # exact 1:1 pixels at dpi=250
    gap = (0.968 - n * cw) / (n - 1)
    x_start = 0.016
    fh = cw * FIG_W / FIG_H * 1.00   # near-full-height cells
    fy = 0.572
    for i, (name, tc, cf, fname, circ, anchor) in enumerate(CATEGORIES):
        x = x_start + i * (cw + gap)
        rbox(fig, x, fy + fh + 0.001, cw, 0.052, cf, rs=0.010, z=3)
        fig.text(x + cw / 2, fy + fh + 0.027, name, ha="center",
                 va="center", fontsize=6.2, color=tc, fontweight="bold",
                 zorder=5)
        frame_ax(fig, x, fy, cw, fh, os.path.join(SC_DIR, fname),
                 crop_frac=0.92, crop_anchor=anchor, circle=circ)


    # ================= MIDDLE STRIP: brand + stat bullets =================
    bx, bw, by, bh = 0.345, 0.31, 0.408, 0.118
    # aura: concentric fading brand-blue halos behind the chip
    for grow, a in ((0.020, 0.05), (0.013, 0.09), (0.006, 0.15)):
        gy = grow * FIG_W / FIG_H
        rbox(fig, bx - grow, by - gy, bw + 2 * grow, bh + 2 * gy,
             mcolors.to_rgba("#6E96CE", a), rs=0.03 + grow, z=2)
    rbox(fig, bx, by, bw, bh, CHIP_BG, ec="#A9C0E0", lw=1.2, rs=0.03, z=3)
    fig.text(0.5, by + bh / 2, "SafeManip", ha="center", va="center",
             fontsize=17, color="#2C4A73", fontweight="bold", zorder=5)
    def bullet(x, y, head, rest):
        fig.text(x, y, head, ha="left", va="center", fontsize=8.2,
                 color=TEXT_DARK, fontweight="bold", zorder=5)
        fig.canvas.draw()
        t = [t for t in fig.texts if t.get_text() == head][-1]
        bb = t.get_window_extent(fig.canvas.get_renderer())
        x1 = fig.transFigure.inverted().transform((bb.x1, 0))[0]
        fig.text(x1 + 0.005, y, rest, ha="left", va="center", fontsize=8.2,
                 color=TEXT_DARK, zorder=5)

    bullet(0.022, 0.520, "10", " LTL$_f$ property templates")
    bullet(0.022, 0.465, "50 + 40", " tasks · RoboCasa365 · LIBERO")
    bullet(0.022, 0.410, "7", " foundation-model policies")
    bullet(0.685, 0.520, "262", " entities · 60 properties · 1,569 pairs")
    bullet(0.685, 0.465, "online", " safety monitoring")
    bullet(0.685, 0.410, "per-event", " safety metrics")

    # ================= BOTTOM ROW ========================================
    py, ph = 0.028, 0.345
    # --- Diverse objects, scenes & tasks (user 2026-09-25 v2: compact,
    # object/fixture WALLS + 2x2 scene/task grids). Sources: objects =
    # RoboCasa banner's Rich Object Library render (docs/images, cluster
    # crops, "more like the cluttered table"); fixtures =
    # eval/figures_out/fixtures/; scenes & tasks = per-suite (top
    # RoboCasa365, bottom LIBERO) from fixtures/ + pull/Figure1.pdf @300dpi
    # + high quality/fig1.png. -------------------------------------------
    p0x, p0w = 0.004, 0.402
    panel(fig, p0x, py, p0w, ph)
    fig.text(p0x + p0w / 2, py + ph - 0.038, "Diverse Objects, Scenes & Tasks",
             ha="center", va="center", fontsize=8.0, color=TEXT_DARK,
             fontweight="bold", zorder=5)
    ch = 0.24                       # shared column-content height
    y_b = py + 0.008
    # assets are pre-composed at EXACT display pixels (232/259/296 x 259
    # @300dpi) so matplotlib does no resampling -- same 1:1 trick as the
    # safety band (user 2026-09-25: panel images must be high-res).
    widths = {"objects": 176 / 250 / FIG_W, "fixtures": 196 / 250 / FIG_W,
              "scenes": 224 / 250 / FIG_W, "tasks": 224 / 250 / FIG_W}
    imgs = {"objects": "d_wall_objects", "fixtures": "d_wall_fixtures",
            "scenes": "d_grid_scenes", "tasks": "d_grid_tasks"}
    order = ["objects", "fixtures", "scenes", "tasks"]
    dgap = (p0w - 0.012 - sum(widths[k] for k in order)) / 3
    xx = p0x + 0.006
    for label in order:
        w = widths[label]
        fig.text(xx + w / 2, y_b + ch + 0.020, label, ha="center",
                 va="center", fontsize=6.0, color="#555555",
                 fontweight="bold", zorder=5)
        frame_ax(fig, xx, y_b, w, ch, os.path.join(TF, f"{imgs[label]}.png"))
        if label in ("scenes", "tasks"):
            for tag, ty in (("RoboCasa365", y_b + ch - 0.016),
                            ("LIBERO", y_b + ch / 2 - 0.018)):
                fig.text(xx + 0.003, ty, tag, ha="left", va="center",
                         fontsize=3.8, color="#333333", zorder=7,
                         bbox=dict(boxstyle="round,pad=0.24",
                                   facecolor="white", alpha=0.78,
                                   edgecolor="none"))
        xx += w + dgap

    # --- ManipVerse mini panel -------------------------------------------
    p1x, p1w = 0.414, 0.124
    panel(fig, p1x, py, p1w, ph)
    fig.text(p1x + p1w / 2, py + ph - 0.042, "ManipVerse",
             ha="center", va="center", fontsize=8.0, color=TEXT_DARK,
             fontweight="bold", zorder=5)

    def mini_pill(cx, cy, text, accent, fs=6.6, px=0.005, pyd=0.010):
        t = fig.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                     color=mcolors.to_rgb(accent), fontweight="bold",
                     zorder=6)
        bb = t.get_window_extent(fig.canvas.get_renderer())
        inv = fig.transFigure.inverted()
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        rbox(fig, x0 - px, y0 - pyd, (x1 - x0) + 2 * px,
             (y1 - y0) + 2 * pyd, tint(accent, 0.16), rs=0.008, z=4)
        return ((x0 + x1) / 2, y0 - pyd, y1 + pyd)

    def mv_edge(p0, p1, color):
        fig.add_artist(plt.Line2D([p0[0], p1[0]], [p0[1], p1[1]],
                                  transform=fig.transFigure, color=color,
                                  linewidth=1.1, zorder=3))

    c1 = p1x + p1w / 2
    # Bottom-up WordNet hierarchy (manipverse_wordnet.json, libero:milk):
    # milk -> {spill, dairy, drink, pourable} -> {food, manipulation} ->
    # dimension roots HAZARD / KIND / AFFORDANCE. Milk at the BOTTOM
    # (user 2026-09-25: "bottom up not top down").
    y_obj, y_leaf, y_mid, y_root = (py + 0.034, py + 0.112,
                                    py + 0.190, py + 0.264)
    xh, xa = c1 - 0.042, c1 + 0.042
    for xx, name, a in ((xh, "HAZARD", MV_MAGENTA), (c1, "KIND", MV_TEAL),
                        (xa, "AFFORD.", MV_GOLD)):
        fig.text(xx, y_root, name, ha="center", va="center", fontsize=4.3,
                 color=shade_mv(a), fontweight="bold", zorder=6)
    mv_food = mini_pill(c1, y_mid, "food", MV_TEAL, fs=4.6)
    mv_manip = mini_pill(xa, y_mid, "manip.", MV_GOLD, fs=4.6)
    mv_dairy = mini_pill(c1 - 0.022, y_leaf, "dairy", MV_TEAL, fs=4.4)
    mv_drink = mini_pill(c1 + 0.009, y_leaf, "drink", MV_TEAL, fs=4.4)
    mv_spill = mini_pill(xh, y_leaf - 0.014, "spill", MV_MAGENTA, fs=4.4)
    mv_pour = mini_pill(xa - 0.002, y_leaf - 0.022, "pourable",
                        MV_GOLD, fs=4.0)
    fig.text(c1, y_obj, "milk", ha="center", va="center", fontsize=6.5,
             color=TEXT_DARK, fontweight="bold", zorder=6)
    mp = (c1, y_obj + 0.016)
    # object -> most-specific synsets
    mv_edge(mp, (mv_spill[0] + 0.008, mv_spill[1]), MV_MAGENTA)
    mv_edge(mp, (mv_dairy[0], mv_dairy[1]), MV_TEAL)
    mv_edge(mp, (mv_drink[0], mv_drink[1]), MV_TEAL)
    mv_edge(mp, (mv_pour[0] - 0.008, mv_pour[1]), MV_GOLD)
    # leaves -> hypernyms (the dairy/drink -> food diamond)
    mv_edge((mv_dairy[0], mv_dairy[2]), (mv_food[0] - 0.007, mv_food[1]),
            MV_TEAL)
    mv_edge((mv_drink[0], mv_drink[2]), (mv_food[0] + 0.007, mv_food[1]),
            MV_TEAL)
    mv_edge((mv_pour[0], mv_pour[2]), (mv_manip[0], mv_manip[1]), MV_GOLD)
    # -> dimension roots (spill attaches directly under HAZARD)
    mv_edge((mv_food[0], mv_food[2]), (c1, y_root - 0.014), MV_TEAL)
    mv_edge((mv_manip[0], mv_manip[2]), (xa, y_root - 0.014), MV_GOLD)
    mv_edge((mv_spill[0], mv_spill[2]), (xh, y_root - 0.014), MV_MAGENTA)

    # --- Predicate hierarchy mini block (user 2026-09-25: one simple
    # example of the hierarchy figure): atomic signals -> compositional
    # predicates -> LTLf property, read bottom-up. -----------------------
    pHx, pHw = 0.544, 0.118
    panel(fig, pHx, py, pHw, ph)
    fig.text(pHx + pHw / 2, py + ph - 0.050, "Predicate-LTL$_f$\nHierarchy",
             ha="center", va="center", linespacing=1.15, fontsize=7.8, color=TEXT_DARK,
             fontweight="bold", zorder=5)
    hx, hw, hh = pHx + 0.004, pHw - 0.008, 0.066
    levels = [(py + 0.012, "#E9E9E9", "atomic signals",
               "contact\u2009\u00b7\u2009gripper\u2009\u00b7\u2009pose", "#555555"),
              (py + 0.104, tint(ENAB_BLUE, 0.10), "predicates",
               "ObjGrasped\u2009\u00b7\u2009StableGrasp", shade_mv(ENAB_BLUE)),
              (py + 0.196, tint(ENAB_BLUE, 0.26), "LTL$_f$ property",
               "$\\varphi_2$\u2009\u00b7\u2009Grasp Stability", shade_mv(ENAB_BLUE, 0.30))]
    for yy, fc, head, body, tc in levels:
        rbox(fig, hx, yy, hw, hh, fc, rs=0.012, z=3)
        fig.text(hx + hw / 2, yy + hh - 0.018, head, ha="center",
                 va="center", fontsize=4.4, color=TEXT_MUTED, zorder=5)
        fig.text(hx + hw / 2, yy + 0.022, body, ha="center", va="center",
                 fontsize=4.8, color=tc, fontweight="bold", zorder=5)
    for y0a in (py + 0.080, py + 0.172):
        fig.add_artist(FancyArrowPatch(
            (pHx + pHw / 2, y0a), (pHx + pHw / 2, y0a + 0.022),
            transform=fig.transFigure, color=ENAB_BLUE, linewidth=1.2,
            arrowstyle="-|>", mutation_scale=7, shrinkA=0, shrinkB=0,
            zorder=4))

    # --- Monitoring mini panel -------------------------------------------
    p2x, p2w = 0.668, 0.206
    panel(fig, p2x, py, p2w, ph)
    fig.text(p2x + p2w / 2, py + ph - 0.042, "Monitoring",
             ha="center", va="center", fontsize=8.5, color=TEXT_DARK,
             fontweight="bold", zorder=5)
    # Same example as the qualitative figure (PreSoakPan demo ep2, phi_2,
    # user-settled picks: f180 pre-slip context, f205 the monitor's
    # detection, f212 the visible consequence) -- frames reused verbatim
    # from qual_frames/, teaser styling kept.
    # FOUR frames labeled like the qualitative figure's chips: enabled
    # (blue) -> violation (red) -> plain context f212 -> end (purple).
    frames = [("p0_f168", "enabled\u2009\u00b7\u2009f168", ENAB_BLUE, True, -0.004),
              ("p2_f205", "violation\u2009\u00b7\u2009f205", VIOL_RED, True, 0.004),
              ("p3_f212", "f212", "#6B6B6B", False, 0.0),
              ("p4_f352", "end\u2009\u00b7\u2009f352", END_PURPLE, True, 0.0)]
    fw = 0.0435
    fgap = 0.0046
    fh2 = fw * FIG_W / FIG_H
    fx0 = p2x + (p2w - 4 * fw - 3 * fgap) / 2
    fyy = py + 0.172
    for i, (tag, lab, col, strong, dx) in enumerate(frames):
        x = fx0 + i * (fw + fgap)
        path = glob.glob(os.path.join(
            QF, f"v31_PreSoakPan_ep2_*__{tag}.png"))[0]
        frame_ax(fig, x, fyy, fw, fh2, path)
        fig.text(x + fw / 2 + dx, fyy - 0.015, lab, ha="center",
                 va="center", fontsize=3.9, color=col,
                 fontweight="bold" if strong else "normal", zorder=7)
    # timeline: full episode (777 monitor frames), frame-annotated
    tx0, tx1, tyy = fx0, fx0 + 4 * fw + 3 * fgap, py + 0.116
    tw_ = tx1 - tx0
    rbox(fig, tx0, tyy - 0.010, tw_, 0.020, TRACK_BG, rs=0.008, z=3)
    f_end = 777
    x168, x205, x352 = (tx0 + tw_ * f / f_end for f in (168, 205, 352))
    rbox(fig, x168, tyy - 0.010, x205 - x168, 0.020,
         tint(TRUE_GREEN, 0.45), rs=0.003, z=4)
    rbox(fig, x205, tyy - 0.010, x352 - x205, 0.020,
         tint(VIOL_RED, 0.55), rs=0.003, z=4)
    for xx, col in ((x168, ENAB_BLUE), (x205, VIOL_RED),
                    (x352, END_PURPLE)):
        fig.add_artist(plt.Line2D([xx, xx], [tyy - 0.013, tyy + 0.013],
                                  transform=fig.transFigure, color=col,
                                  linewidth=1.1, zorder=6))
    fig.text(x168 - 0.004, tyy - 0.025, "168", ha="center", va="center",
             fontsize=4.3, color=ENAB_BLUE, zorder=6)
    fig.text(x205 + 0.004, tyy + 0.025, "205", ha="center", va="center",
             fontsize=4.3, color=VIOL_RED, fontweight="bold", zorder=6)
    fig.text(x352, tyy - 0.025, "352", ha="center", va="center",
             fontsize=4.3, color=END_PURPLE, zorder=6)
    fig.text(tx0, tyy - 0.025, "0", ha="left", va="center", fontsize=4.3,
             color=TEXT_MUTED, zorder=6)
    fig.text(tx1, tyy - 0.025, "777", ha="right", va="center",
             fontsize=4.3, color=TEXT_MUTED, zorder=6)
    fig.text(p2x + p2w / 2, py + 0.066,
             r"$\varphi_2:\ \mathbf{G}(\mathsf{ObjGrasped} \rightarrow"
             r" (\mathsf{StableGrasp}\ \mathbf{U}\ \mathsf{ObjReleased}))$",
             ha="center", va="center", fontsize=5.0, color=TEXT_DARK,
             zorder=5)
    fig.text(p2x + p2w / 2, py + 0.034,
"pan slips from the gripper → violation",
             ha="center", va="center", fontsize=5.2, color=VIOL_RED,
             fontweight="bold", zorder=5)

    # --- Metrics mini panel ----------------------------------------------
    p3x, p3w = 0.882, 0.114
    panel(fig, p3x, py, p3w, ph)
    fig.text(p3x + p3w / 2, py + ph - 0.050, "Event-Centric\nMetrics",
             ha="center", va="center", linespacing=1.15, fontsize=7.8, color=TEXT_DARK,
             fontweight="bold", zorder=5)
    def metric_icon(cx, cy, kind, col):
        w = 0.020
        h = w * FIG_W / FIG_H
        ax = fig.add_axes([cx - w / 2, cy - h / 2, w, h], frameon=False)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.patch.set_visible(False); ax.set_zorder(8)
        ax.add_patch(Circle((0.5, 0.5), 0.46, facecolor=col,
                            edgecolor="none"))
        if kind == "goal":
            # end-STATE metric: a goal flag checked at the final state
            ax.plot([0.36, 0.36], [0.22, 0.78], color="white",
                    linewidth=1.6, solid_capstyle="round")
            ax.add_patch(plt.Polygon([(0.36, 0.78), (0.74, 0.64),
                                      (0.36, 0.50)], closed=True,
                         facecolor="white", edgecolor="none"))
        elif kind == "trajx":
            # TRAJECTORY metric: a rollout path with a violation on it
            ax.plot([0.16, 0.34, 0.52, 0.66], [0.26, 0.52, 0.38, 0.60],
                    color="white", linewidth=1.5, solid_capstyle="round",
                    solid_joinstyle="round")
            ax.plot([0.60, 0.84], [0.54, 0.78], color="white",
                    linewidth=2.2, solid_capstyle="round")
            ax.plot([0.60, 0.84], [0.78, 0.54], color="white",
                    linewidth=2.2, solid_capstyle="round")
        elif kind == "clock":
            ax.add_patch(Circle((0.5, 0.5), 0.30, facecolor="white",
                                edgecolor="none"))
            ax.plot([0.5, 0.5], [0.5, 0.68], color=col, linewidth=1.5,
                    solid_capstyle="round")
            ax.plot([0.5, 0.63], [0.5, 0.46], color=col, linewidth=1.5,
                    solid_capstyle="round")

    rows = [("goal", "task success rate", "#4C9A62"),
            ("trajx", "violations per\ntriggered event", "#D64545"),
            ("clock", "unsafe-state exposure\nper triggered event", "#9A7B24")]
    for i, (kind, desc, col) in enumerate(rows):
        yy = py + 0.245 - i * 0.076
        metric_icon(p3x + 0.010, yy, kind, col)
        fig.text(p3x + 0.021, yy, desc, ha="left", va="center",
                 fontsize=5.1, color=TEXT_DARK, zorder=6,
                 linespacing=1.25)
    fig.text(p3x + p3w / 2, py + 0.026,
             "every metric is normalized per\n"
             "safety event, not per episode",
             ha="center", va="center", fontsize=4.4, color=TEXT_MUTED,
             style="italic", zorder=5, linespacing=1.25)

    for ext, kw in (("png", {"dpi": 250}), ("pdf", {"dpi": 250})):
        fig.savefig(os.path.join(HERE, f"teaser_pull.{ext}"),
                    facecolor=PAGE_BG, bbox_inches="tight",
                    pad_inches=0.02, **kw)
    print("wrote teaser_pull.png / .pdf")


if __name__ == "__main__":
    main()
