#!/usr/bin/env python3
"""Qualitative figure: two representative violations with viewer-style
predicate tracks, styled after ref/fig/high quality/figure-intro.pdf
(rounded light-blue italic header chips, white bg)
and the paper-wide regular-sans + sparse-bold convention.

Layout (user 2026-09-24, final): ONE column, two cells --
    Grasp stability   (PreSoakPan expert demo ep2, the pan slips)
    Containment       (PrepareCoffee GR00T-N1.5 target_only ep49, coffee
                       poured before the mug is placed)
Each cell:  five video frames -> LTL violation bar -> predicate tracks ->
paper-template formula -> one-sentence story with the violation REASON in
red. The tagged frames (enabled / violation / end) are tied to the LTL bar
by dotted connectors + ticks, and the bar labels their frame numbers.

The DISPLAYED formulas and atom names are the paper's property templates
from 3a_tab_formulas.tex (phi_2 grasp, phi_7 containment) -- the monitor's
real formulas are richer, but the figure speaks the paper's language
(user 2026-09-24). Atom mapping, grasp: ObjGrasped=object_grasped,
StableGrasp=object_sync, ObjReleased=!object_grasped_raw (masked before the
first grasp -- "released" before ever grasping would read as a green bar).
Containment: LiqTransfer=liquid_transfer_event, LiqContained=liquid_settled,
and the one sub-predicate that actually fails, support_type_matches_content,
shown as the leaf InIntendedReceiver; the passing leaves are elided.

Violation windows come from the monitor's recovery trace (the viewer's
OVERALL bar), never the absorbing primary trace.

ffmpeg/ffprobe required; frames cached in qual_frames/. Run with
~/testnvme/miniconda3/bin/python (never /bin/python3).
"""
import json
import os
import re
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/path/to/SafeManip"
DEMO_RC = os.path.join(ROOT, "SafeManip", "monitor", "output",
                       "v31_2026-09-20_claude_branch_place_precondition_"
                       "hygiene_removed")
DATASET_ROOT = os.path.expanduser("~/flash/datasets/robocasa/v1.0/target")
FRAME_CACHE = os.path.join(HERE, "qual_frames")
# liquid-transfer extraction (GR00T N1.5 target_only PrepareCoffee ep49 --
# replaced multitask ep47 whose spout was occluded, user 2026-09-24).
# GROUND TRUTH (privileged_information_49.json, verified 2026-09-24): the
# robot picks the mug from the cabinet and sets it down by ~f280 OFF the
# dispensing spot (front-right of the tray, mug pose then constant to the
# end); it presses start at f520 and the stream lands on the TRAY behind
# the mug rim -- support_type_matches_content false from f520, timeout
# violation f620. The earlier "mug placed too late at f750" reading was
# wrong: the arm merely occluded the always-there mug in the right view.
COFFEE_JSON = os.path.join(HERE, "qual_liquid_preparecoffee_to49.json")

VIOL_RED = "#D64545"
ENAB_BLUE = "#3E78B2"       # enabled chip/tick
END_PURPLE = "#7A5AA0"      # end chip/tick
TRUE_GREEN = "#4C9A62"
TRACK_BG = "#EDEDED"
CHIP_BG = "#D6E2F5"         # figure-intro's light blue header chip
TEXT_DARK = "#333333"


def apply_style():
    # NeurIPS / paper convention (paper-figure-polish): regular-weight sans
    # with SPARSE bold -- bold only on the chip, task name, and operators.
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "mathtext.fontset": "dejavusans",
        "axes.edgecolor": "black", "axes.linewidth": 1.1,
        "axes.axisbelow": True,
    })


def windows_of(flags):
    w, cur = [], None
    for i, x in enumerate(flags):
        if x and cur is None:
            cur = i
        if not x and cur is not None:
            w.append((cur, i - 1))
            cur = None
    if cur is not None:
        w.append((cur, len(flags) - 1))
    return w


def video_props(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,r_frame_rate",
         "-of", "json", path]).decode()
    st = json.loads(out)["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    return int(st["nb_read_frames"]), float(num) / float(den)


def extract_frame(video, t, out_png):
    if not os.path.exists(out_png):
        subprocess.check_call(
            ["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", video,
             "-frames:v", "1", out_png])


def frames_for(video, n_mon, picks, stem):
    nb_video, fps = video_props(video)
    pngs = []
    for i, f in enumerate(picks):
        vf = min(nb_video - 1, round(f / max(n_mon - 1, 1) * (nb_video - 1)))
        # frame number in the cache key, or a picks change silently reuses
        # stale frames
        png = os.path.join(FRAME_CACHE, f"{stem}__p{i}_f{f}.png")
        extract_frame(video, vf / fps, png)
        pngs.append(png)
    return pngs


def demo_video(task, ep, camera="robot0_agentview_left"):
    for cat in ("composite", "atomic"):
        base = os.path.join(DATASET_ROOT, cat, task)
        if not os.path.isdir(base):
            continue
        date = sorted(os.listdir(base))[-1]
        p = os.path.join(base, date, "lerobot", "videos", "chunk-000",
                         f"observation.images.{camera}",
                         f"episode_{ep:06d}.mp4")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no demo video for {task} ep{ep}")


def load_grasp():
    """PreSoakPan demo ep2: the pan (obj1 in the monitor's naming) slips.
    Middle frames per user 2026-09-24: f180 pre-slip context, f205 the
    monitor's detection, f212 the visible consequence -- the point is that
    the monitor fires before the failure is visually obvious."""
    task, ep, prop = "PreSoakPan", 2, "rc_grasp_remains_synced_until_dropped"
    mon = json.load(open(os.path.join(
        DEMO_RC, task, f"privileged_information_{ep}_monitor.json")))
    v = next(x for x in mon["violations"] if x["property_name"] == prop)
    tr = ((mon.get("recovery_accepting_by_property") or {}).get(prop)
          or (mon.get("accepting_by_property") or {}).get(prop))
    wins = windows_of([not x for x in tr])
    n = len(tr)
    trace = v["repeated"]["trace"]
    ser = lambda a: [bool(t["predicate_values"].get(a)) for t in trace]
    grasped, sync, raw = (ser("object_grasped"), ser("object_sync"),
                          ser("object_grasped_raw"))
    # obligation window: the antecedent (object_grasped) run containing the
    # first violation -- picture 1 = grasp onset, last picture = the drop
    runs = windows_of(grasped)
    a = wins[0][0]
    hit = next((r for r in runs if r[0] <= a <= r[1] + 1), (a, wins[-1][1]))
    enabled, endf = hit[0], min(n - 1, hit[1] + 1)
    middles = [(180, ""), (205, "violation"), (212, "")]
    picks = [enabled] + [f for f, _ in middles] + [endf]
    tags = ["enabled"] + [t for _, t in middles] + ["end"]
    first = raw.index(True) if True in raw else n
    released = [i > first and not x for i, x in enumerate(raw)]
    viol_f, cons_f = 205, 212
    return dict(
        display="Grasp stability", task=task, n_mon=n,
        obj="the pan slips", windows=wins, picks=picks, tags=tags,
        ltl=(r"$\varphi_2:\ \mathbf{G}(\mathsf{ObjGrasped} \rightarrow"
             r" (\mathsf{StableGrasp}\ \mathbf{U}\ \mathsf{ObjReleased}))$"),
        tracks=[("ObjGrasped", grasped, 0, None),
                ("StableGrasp", sync, 0, None),
                ("ObjReleased", released, 0, None)],
        caption=[
            [(f"The robot picks up the pan at f {enabled}, but ", TEXT_DARK),
             (f"the grasp becomes unstable at f {viol_f}", VIOL_RED)],
            [(f"— the pan slips, visibly only from f {cons_f} — ", VIOL_RED),
             (f"and the pan is dropped at f {endf}.", TEXT_DARK)]],
        pngs=frames_for(demo_video(task, ep, "robot0_agentview_right"),
                        n, picks, f"v31_{task}_ep{ep}_{prop}_"
                        "robot0_agentview_right"))


def load_coffee():
    g = json.load(open(COFFEE_JSON))
    n = g["num_frames"]
    P = g["predicates"]
    picks = [f for f, _ in g["picks"]]
    tags = [t for _, t in g["picks"]]
    enabled = picks[tags.index("enabled")]
    viol_f = picks[tags.index("violation")]
    endf = picks[tags.index("end")]
    place_f = picks[0]
    return dict(
        display="Containment", task=g["task"], n_mon=n,
        obj=g["obj"].replace(" - ", " — "),
        windows=[tuple(w) for w in g["windows"]], picks=picks, tags=tags,
        # bounded eventually (user 2026-09-24): the monitor's
        # SETTLE_TIMEOUT_FRAMES=100 makes the obligation a deadline --
        # violation f620 = enabled f520 + 100
        ltl=(r"$\varphi_7:\ \mathbf{G}(\mathsf{LiqTransfer} \rightarrow"
             r" \mathbf{F}_{\leq 100}\ \mathsf{LiqContained})$"),
        deadline=(enabled, viol_f),
        # paper-named atoms + the ONE failing leaf; the passing leaves
        # (content supported / stable) are elided behind the "..." row
        tracks=[("LiqTransfer", P["liquid_transfer_event"], 0, None),
                ("LiqContained", P["liquid_settled"], 0, None),
                ("InIntendedReceiver", P["support_type_matches_content"],
                 1, None),
                ("⋯", None, 1,
                 "remaining sub-predicates (supported, stable) are true — "
                 "omitted for space")],
        caption=[
            [(f"The robot places the mug off the dispensing spot "
              f"(f {place_f}) and presses the start button at f {enabled},",
              TEXT_DARK)],
            [("but the coffee misses the mug and pours onto the drip tray",
              VIOL_RED),
             (f" — timeout violation at f {viol_f}.", TEXT_DARK)]],
        # LEFT agentview (user 2026-09-24); the extraction JSON stores the
        # right-view path
        pngs=frames_for(g["video"].replace("agentview_right",
                                           "agentview_left"),
                        n, picks,
                        "policy_" + g["task"] + "_" + g["prop"] + "_left"))


def timeline(ax, n, spans, color, min_frac=0.0):
    """Frame f occupies exactly [f, f+1) -- NO padding on predicate tracks.
    min_frac widens a span SYMMETRICALLY to a minimum visible width; used
    only for the red violation windows."""
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_facecolor(TRACK_BG)
    for a, b in spans:
        lo, hi = a, b + 1
        half = max((hi - lo) / 2, min_frac * n / 2)
        c = (lo + hi) / 2
        ax.axvspan(c - half, c + half, color=color, lw=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#666666")
        s.set_linewidth(0.7)


def caption_segments(fig, xc, y, lines, fontsize, line_h):
    """Centered lines whose REASON span is a different color: place the
    colored segments left-to-right after measuring each width (matplotlib
    has no inline multicolor text)."""
    r = fig.canvas.get_renderer()
    for li, segments in enumerate(lines):
        widths = []
        for s, color in segments:
            wt = "bold" if color == VIOL_RED else "normal"
            t = fig.text(0, 0, s, fontsize=fontsize, fontweight=wt)
            widths.append(t.get_window_extent(renderer=r).width
                          / fig.bbox.width)
            t.remove()
        x = xc - sum(widths) / 2
        for (s, color), w in zip(segments, widths):
            fig.text(x, y - li * line_h, s, fontsize=fontsize,
                     color=color, ha="left", va="top",
                     fontweight="bold" if color == VIOL_RED else "normal")
            x += w


def draw_cell(fig, gs_cell, cell):
    n_tr = len(cell["tracks"])
    n_img = len(cell["pngs"])
    heights = [1.0] + [0.10] * (1 + n_tr)
    inner = gs_cell.subgridspec(2 + n_tr, n_img, height_ratios=heights,
                                hspace=0.16, wspace=0.03)
    # chip colors: red = the monitor's detection, blue/purple = the
    # obligation's enabled/end frames, black = context frames
    chip_fc = {"violation": VIOL_RED, "enabled": ENAB_BLUE,
               "end": END_PURPLE}
    img_axes = []
    for j, (png, mf) in enumerate(zip(cell["pngs"], cell["picks"])):
        ax = fig.add_subplot(inner[0, j])
        ax.imshow(mpimg.imread(png))
        # square frames letterbox inside the (taller) image row -- anchor
        # them to the BOTTOM so the slack sits under the header, not as a
        # gap between the pictures and the LTL tracks
        ax.set_anchor("S")
        ax.set_xticks([])
        ax.set_yticks([])
        tag = cell["tags"][j]
        lab = f"{tag} · f {mf}" if tag else f"f {mf}"
        ax.text(0.035, 0.955, lab, transform=ax.transAxes,
                ha="left", va="top", fontsize=13, color="white",
                fontweight="bold",
                bbox=dict(facecolor=chip_fc.get(tag, "black"), alpha=0.72,
                          pad=1.6, edgecolor="none"))
        for s in ax.spines.values():
            s.set_color("#444444")
            s.set_linewidth(0.8)
        img_axes.append(ax)
    # OVERALL row (the viewer's LTL bar): red where the monitor is in
    # violation; tagged frames get a tick + connector up to their picture
    ax = fig.add_subplot(inner[1, :])
    timeline(ax, cell["n_mon"], cell["windows"], VIOL_RED, min_frac=0.006)
    n = cell["n_mon"]
    # bounded-eventually deadline (F_{<=k}): shade the k-frame obligation
    # window on the LTL bar so the timeout is VISIBLE -- it runs from the
    # enabled tick to the violation, where the deadline expires
    if cell.get("deadline"):
        d0, d1 = cell["deadline"]
        ax.axvspan(d0, d1, color="#F2C94C", alpha=0.45, lw=0, zorder=2)
        ax.annotate("", xy=(d1, 0.24), xytext=(d0, 0.24),
                    arrowprops=dict(arrowstyle="<->", color="#9A6B15",
                                    lw=1.3, shrinkA=0, shrinkB=0), zorder=6)
        ax.text((d0 + d1) / 2, 0.67, f"≤{d1 - d0} f", ha="center",
                va="center", fontsize=9.5, fontweight="bold",
                color="#9A6B15", zorder=6)
    last_label_end = -1e9
    for a, b in cell["windows"]:
        # skip a label whose neighbour's label would overlap it (e.g. the
        # 177/213 grasp slips); the first window of a cluster names it
        if a < last_label_end + 0.02 * n:
            continue
        lab = f"f {a}–{b}" if b > a else f"f {a}"
        if b < 0.82 * n:
            ax.text(b + 0.022 * n, 0.5, lab, ha="left", va="center",
                    fontsize=11.5, color=VIOL_RED, fontweight="bold",
                    zorder=6)
            last_label_end = b + 0.022 * n + 0.016 * n * len(lab)
        else:
            ax.text(a - 0.022 * n, 0.5, lab, ha="right", va="center",
                    fontsize=11.5, color=VIOL_RED, fontweight="bold",
                    zorder=6)
            last_label_end = b
    # tagged-frame ticks + frame numbers (enabled/end -- the violation
    # window already labels itself in red) and picture->bar connectors
    for j, (mf, tag) in enumerate(zip(cell["picks"], cell["tags"])):
        if not tag:
            continue
        color = chip_fc.get(tag, "black")
        ax.plot([mf + 0.5] * 2, [0, 1], color=color, lw=2.2, zorder=5,
                solid_capstyle="butt")
        if tag == "enabled":       # number LEFT of the tick, clear of the
            ax.text(mf - 0.012 * n, 0.5, f"f {mf}", ha="right",  # red label
                    va="center", fontsize=11.5, color=ENAB_BLUE,
                    fontweight="bold", zorder=6)
        elif tag == "end":
            right = mf > 0.9 * n     # label inside the bar at the far edge
            ax.text(mf - 0.012 * n if right else mf + 0.012 * n, 0.5,
                    f"f {mf}", ha="right" if right else "left",
                    va="center", fontsize=11.5, color=END_PURPLE,
                    fontweight="bold", zorder=6)
    ax.text(0.004, 0.5, "LTL", transform=ax.transAxes, ha="left",
            va="center", fontsize=13, fontweight="bold", color=VIOL_RED,
            zorder=6, bbox=dict(facecolor="white", alpha=0.75, pad=0.6,
                                edgecolor="none"))
    # predicate activation tracks, viewer-style: green where the atom is
    # true; indent nests a sub-predicate; a None-series row renders its
    # note text (gray -- the red reasoning lives in the caption)
    for k, (atom, flags, indent, note) in enumerate(cell["tracks"]):
        ax = fig.add_subplot(inner[2 + k, :])
        timeline(ax, cell["n_mon"],
                 windows_of(flags) if flags is not None else [],
                 TRUE_GREEN, min_frac=0.003)
        if flags is None and note:
            ax.text(0.996, 0.5, note, transform=ax.transAxes, ha="right",
                    va="center", fontsize=10.5, fontstyle="italic",
                    color="#777777", zorder=6)
        if k == n_tr - 1:          # frame-number axis, once per cell
            nn = cell["n_mon"]
            ticks = [0, nn // 4, nn // 2, 3 * nn // 4, nn - 1]
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks], fontsize=11,
                               color="#555555")
            ax.tick_params(axis="x", length=2, pad=1.5)
            labs = ax.get_xticklabels()
            labs[0].set_ha("left")
            labs[-1].set_ha("right")
        # label INSIDE the track; hierarchy levels indent with an elbow
        label = " " * (2 * indent) + ("└ " if indent else "") + atom
        ax.text(0.004, 0.5, label, transform=ax.transAxes, ha="left",
                va="center", fontsize=12.5, fontweight="bold", color="#222222", zorder=6,
                bbox=dict(facecolor="white", alpha=0.75, pad=0.6,
                          edgecolor="none"))
    bb = gs_cell.get_position(fig)
    fh = fig.get_figheight()   # offsets in inches / fh -> height-independent
    # rounded pastel chip with the category (figure-intro's header style)
    chip_w = 0.026 + 0.0131 * len(cell["display"])
    fig.patches.append(FancyBboxPatch(
        (bb.x0, bb.y1 + 0.035 / fh), chip_w, 0.30 / fh,
        boxstyle="round,pad=0.0018,rounding_size=0.006",
        transform=fig.transFigure, facecolor=CHIP_BG, edgecolor="#7C8DB0",
        linewidth=0.9))
    yc = bb.y1 + 0.175 / fh
    fig.text(bb.x0 + chip_w / 2, yc, cell["display"],
             fontsize=19, fontweight="bold", fontstyle="italic",
             ha="center", va="center")
    fig.text(bb.x0 + chip_w + 0.012, yc, cell["task"],
             fontsize=18, fontweight="bold", ha="left", va="center",
             color="#444444")
    fig.text(bb.x1, yc, cell["obj"], fontsize=15.5, fontstyle="italic",
             ha="right", va="center", color="#8A4A45")
    # paper-template formula + one-line story under the tracks (dropped low
    # enough to clear the last track's frame-number ticks)
    fig.text((bb.x0 + bb.x1) / 2, bb.y0 - 0.16 / fh, cell["ltl"],
             fontsize=17, ha="center", va="top", color="#333333")
    caption_segments(fig, (bb.x0 + bb.x1) / 2, bb.y0 - 0.50 / fh,
                     cell["caption"], fontsize=14.5, line_h=0.30 / fh)
    return bb


def main():
    apply_style()
    grasp = load_grasp()
    coffee = load_coffee()

    fig = plt.figure(figsize=(11.0, 9.0))
    outer = fig.add_gridspec(
        2, 1, hspace=0.48,
        height_ratios=[1.0 + 0.10 * (1 + len(grasp["tracks"])),
                       1.0 + 0.10 * (1 + len(coffee["tracks"]))],
        left=0.012, right=0.985, top=0.955, bottom=0.117)
    bb0 = draw_cell(fig, outer[0, 0], grasp)
    bb1 = draw_cell(fig, outer[1, 0], coffee)
    # subtle divider between the two cells (user 2026-09-24: small, not
    # obvious): thin light-gray dashes halfway between the top caption and
    # the bottom header chip
    fh = fig.get_figheight()
    ysep = ((bb0.y0 - 1.05 / fh) + (bb1.y1 + 0.37 / fh)) / 2
    fig.add_artist(Line2D([0.06, 0.94], [ysep, ysep],
                          transform=fig.transFigure, color="#D5D5D5",
                          linewidth=0.9, linestyle=(0, (4, 4))))
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"qualitative_monitor.{ext}")
        fig.savefig(out, dpi=300 if ext == "png" else None)
        print(f"[fig] {out}")


if __name__ == "__main__":
    main()
