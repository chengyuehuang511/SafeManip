#!/usr/bin/env python3
"""Qualitative figure: representative violations with viewer-style predicate
tracks, plus the framework on expert demonstrations and the human audit.

Layout follows the reference family (ref/fig/high quality/figure-intro.pdf):
bold sans-serif, white background, rounded pastel header chips per example,
dashed separators between rows. Each example cell shows

    [chip: category]  TaskName
    three video frames (before / during / after the violation)
    LTL      |======red where in violation=====|     <- the viewer's OVERALL
    atom_1   |###green where true##            |     <- predicate activation,
    atom_2   |      ###                        |        exactly the viewer's
    ...                                                 breakdown tracks
    LTL template (mono, on the header line)

Violation windows and predicate traces come from the monitor's own per-frame
trace (violations[].repeated.trace), NOT from the curated filename ranges.

Sources (all per explicit user decision 2026-09-24):
  * Panel (a) cells come from the v31 RoboCasa EXPERT-DEMONSTRATION corpus
    (SafeManip/monitor/output/v31_*, the final RC demo corpus per the version
    ledger) -- the same episodes the viewer's Training Data tab shows -- and
    the OVERALL window is computed the way the VIEWER computes it:
    recovery_accepting_by_property when present, else accepting_by_property;
    violated = frames where that trace is False. NOT the repeated-trace
    in_violation flags (absorbing for several properties), and NOT the old
    examples/ set (old monitor version; its grasp property was retired).
    Only categories experts actually violate appear: collision/contact,
    cross-contamination, action-onset, grasp stability. Mechanism,
    containment, and enclosure/access never fire on demos -- that absence is
    part of the story, not a gap.
  * Panel (b): expert demo corpora (RoboCasa v31, LIBERO v40) vs the policy
    corpus (RQ3_conditional_episodes.csv).
  * Panel (c): all item-level verdicts in viewer/annotations/.

ffmpeg/ffprobe required; frames cached in qual_frames/, scans in
qual_stats.json (--recompute rescans).
"""
import glob
import json
import os
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import image as mpimg
from matplotlib.patches import FancyBboxPatch, Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/path/to/SafeManip"
EXAMPLES_DIR = os.path.join(ROOT, "examples")
ANNOT_DIR = os.path.join(ROOT, "viewer", "annotations")
DEMO_RC = os.path.join(ROOT, "SafeManip", "monitor", "output",
                       "v31_2026-09-20_claude_branch_place_precondition_"
                       "hygiene_removed")
DEMO_LB = os.path.join(ROOT, "SafeManip", "monitor", "output",
                       "v40_2026-09-21_claude_branch_libero_consolidated")
EP_CSV = os.path.join(HERE, "RQ3_conditional_episodes.csv")
DATASET_ROOT = os.path.expanduser("~/flash/datasets/robocasa/v1.0/target")
FRAME_CACHE = os.path.join(HERE, "qual_frames")
STATS_CACHE = os.path.join(HERE, "qual_stats.json")

# (category chip, Task, demo episode, registry property) -- all from the v31
# demo corpus. Episodes hand-picked for crisp recovery-trace windows.
# last field: the RELEVANT object(s) of the violation (from the monitor's own
# explanation -- the contact partner / blocker / unstable support, not merely
# the active object).
EXAMPLES = [
    ("Collision/contact", "GarnishPancake", 7, "rc_no_forbidden_contact",
     "gripper ↔ fridge (distractor)"),
    ("Grasp stability", "MakeIceLemonade", 0,
     "rc_grasp_remains_synced_until_dropped", "ice_cube1 slipping"),
    ("Cross-contamination", "PortionHotDogs", 5,
     "rc_raw_robot_contact_blocks_rte_grasp_until_sanitized",
     "raw sausage1/2 → stool"),
    ("Grasp stability", "ArrangeBreadBasket", 3,
     "rc_dropped_object_was_released", "bread dropped"),
    ("Action-onset", "WashFruitColander", 1, "rc_pick_preconditions_safe",
     "colander pick blocked by fruit1"),
    ("Action-onset", "StackBowlsCabinet", 6, "rc_place_preconditions_safe",
     "bowl2 onto unstable bowl1"),
]
MAX_TRACKS = 3          # at most this many predicate tracks per cell
VIOL_RED = "#D64545"
TRUE_GREEN = "#4C9A62"
TRACK_BG = "#EDEDED"
CHIP_BG = "#D6E2F5"     # figure-intro's light blue header chip
SEP_GRAY = "#9A9A9A"


def apply_style():
    # Reference style (ref/fig/high quality): bold sans-serif everywhere,
    # white background, light dashed grid, black axes box.
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.weight": "bold",
        "axes.labelweight": "bold", "axes.titleweight": "bold",
        "mathtext.fontset": "dejavusans", "mathtext.default": "bf",
        "axes.edgecolor": "black", "axes.linewidth": 1.1,
        "axes.axisbelow": True,
    })


def ltl_atoms(ltl):
    """Atoms in first-appearance order, operators excluded."""
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", ltl)
    out = []
    for t in toks:
        if t in ("G", "U", "F", "X") or t in out:
            continue
        out.append(t)
    return out


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
    for tag, f in zip(("pre", "mid", "end"), picks):
        vf = min(nb_video - 1, round(f / max(n_mon - 1, 1) * (nb_video - 1)))
        # frame number in the cache key, or a picks change silently reuses
        # stale frames
        png = os.path.join(FRAME_CACHE, f"{stem}__{tag}_f{f}.png")
        extract_frame(video, vf / fps, png)
        pngs.append(png)
    return pngs


def demo_video(task, ep):
    for cat in ("composite", "atomic"):
        base = os.path.join(DATASET_ROOT, cat, task)
        if not os.path.isdir(base):
            continue
        date = sorted(os.listdir(base))[-1]
        p = os.path.join(base, date, "lerobot", "videos", "chunk-000",
                         "observation.images.robot0_agentview_left",
                         f"episode_{ep:06d}.mp4")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no demo video for {task} ep{ep}")


def load_cells():
    os.makedirs(FRAME_CACHE, exist_ok=True)
    cells = []
    for display, task, ep, prop, obj in EXAMPLES:
        mon = json.load(open(os.path.join(
            DEMO_RC, task, f"privileged_information_{ep}_monitor.json")))
        v = next(x for x in mon["violations"] if x["property_name"] == prop)
        # THE VIEWER'S CALCULATION (server.py _real_trace_for): prefer the
        # recovery trace, fall back to the primary accepting trace; violated
        # frames are where the trace is False.
        tr = ((mon.get("recovery_accepting_by_property") or {}).get(prop)
              or (mon.get("accepting_by_property") or {}).get(prop))
        wins = windows_of([not x for x in tr])
        a, b = wins[0][0], wins[-1][1]
        n = len(tr)
        trace = v["repeated"]["trace"]
        atoms = [x for x in ltl_atoms(v["ltl"])
                 if x in trace[0]["predicate_values"]]
        extra = [x for x in trace[0]["predicate_values"] if x not in atoms]
        atoms = (atoms + extra)[:MAX_TRACKS]
        # The viewer's occurrence marks (user 2026-09-24): picture 1 is the
        # frame the obligation becomes ENABLED (start of the antecedent
        # atom's true-run containing the first violation), picture 2 the
        # violation itself, picture 3 the obligation's END (end of that
        # antecedent run). Properties with no antecedent (G(!x)) or whose
        # antecedent IS the violating onset fall back to window start/end.
        ante = None
        m = re.match(r"G\((\w+)\s*->", v["ltl"])
        if m and m.group(1) in trace[0]["predicate_values"]:
            ante = [bool(t["predicate_values"].get(m.group(1)))
                    for t in trace]
        enabled, endf = a, min(n - 1, b + 1)
        if ante:
            runs = windows_of(ante)
            hit = next((r for r in runs if r[0] <= a <= r[1] + 1), None)
            if hit and hit[1] > hit[0]:
                enabled, endf = hit[0], min(n - 1, hit[1] + 1)
        viol_f = wins[0][0]
        picks = [enabled, viol_f, endf]
        cells.append(dict(
            display=display, task=task, ltl=v["ltl"], n_mon=n, obj=obj,
            windows=wins, picks=picks, tags=("enabled", "violation", "end"),
            tracks=[(x, [bool(t["predicate_values"].get(x)) for t in trace])
                    for x in atoms],
            pngs=frames_for(demo_video(task, ep), n, picks,
                            f"v31_{task}_ep{ep}_{prop}")))
    return cells


def scan_stats():
    if os.path.exists(STATS_CACHE) and "--recompute" not in sys.argv:
        return json.load(open(STATS_CACHE))
    stats = {}
    # Cross-contamination excluded from panel (b) on BOTH sides (user
    # 2026-09-24) -- it dominates the demo count and would make the demo/policy
    # comparison about one property.
    XC = "rc_raw_robot_contact_blocks_rte_grasp_until_sanitized"
    for key, root in (("rc_demo", DEMO_RC), ("lb_demo", DEMO_LB)):
        n = viol = 0
        for f in glob.glob(os.path.join(root, "*", "*_monitor.json")):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            n += 1
            viol += any(v.get("property_name") != XC
                        for v in d.get("violations") or [])
        stats[key] = dict(episodes=n, violated=viol)
    # policy side, same exclusion, from the per-property metrics tables
    PD = os.path.join(ROOT, "eval", "saved_eval_rollouts", "monitor_files",
                      "0920")
    for key, pat in (("rc_policy", "plot_robocasa/processedData/"
                      "metrics_long_*.csv"),
                     ("lb_policy", "plot_libero/processedData/"
                      "metrics_long_*.csv")):
        tot = viol = 0
        for f in glob.glob(os.path.join(PD, pat)):
            d = pd.read_csv(f)
            d = d[d.property_name != XC]
            g = d.groupby(["model", "task", "episode"]).violated.max()
            tot += len(g)
            viol += int(g.sum())
        stats[key] = dict(episodes=tot, violated=viol)
    # Audit restricted to the FINAL corpora (v31 RC / v40 LIBERO demo
    # versions -- user 2026-09-24): every disputed verdict in the annotation
    # store belongs to earlier debugging iterations of the monitor.
    verdicts = {"confirmed": 0, "disputed": 0, "unsure": 0}
    sat = 0
    pats = ("*v31_2026-09-20*.json",
            "*v40_2026-09-21_claude_branch_libero_consolidated*.json")
    for pat in pats:
        for f in glob.glob(os.path.join(ANNOT_DIR, "*", pat)):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            for k, v in (d.get("violations") or {}).items():
                vd = (v or {}).get("verdict")
                if vd in verdicts:
                    verdicts[vd] += 1
            for k, v in (d.get("satisfied") or {}).items():
                if (v or {}).get("verdict") == "confirmed":
                    sat += 1
    stats["audit"] = verdicts
    stats["audit_satisfied_confirmed"] = sat
    json.dump(stats, open(STATS_CACHE, "w"), indent=1)
    return stats


def fmt_ltl(ltl):
    return (r"$\mathtt{" + ltl.replace("_", r"\_")
            .replace("->", r"\rightarrow ").replace("!", r"\neg ")
            .replace(" U ", r"\ \mathbf{U}\ ").replace("G(", r"\mathbf{G}(")
            + "}$")


def timeline(ax, n, spans, color, markers=None, min_frac=0.0):
    """Frame f occupies exactly [f, f+1) -- NO padding on predicate tracks
    (padding made adjacent true-spans swallow 1-frame gaps, so the gap no
    longer lined up with the violation window -- user-reported 2026-09-24).
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
    if markers:
        for f in markers:
            ax.plot([f], [0.5], marker="o", markersize=3.4,
                    markerfacecolor="white", markeredgecolor="#1A1A1A",
                    markeredgewidth=0.8, clip_on=False, zorder=5)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#666666")
        s.set_linewidth(0.7)


def draw_cell(fig, gs_cell, cell):
    n_tr = len(cell["tracks"])
    heights = [1.0] + [0.085] * (1 + n_tr)
    inner = gs_cell.subgridspec(2 + n_tr, 3, height_ratios=heights,
                                hspace=0.14, wspace=0.03)
    for j, (png, mf) in enumerate(zip(cell["pngs"], cell["picks"])):
        ax = fig.add_subplot(inner[0, j])
        ax.imshow(mpimg.imread(png))
        ax.set_xticks([])
        ax.set_yticks([])
        tag = cell["tags"][j]
        ax.text(0.035, 0.955, f"{tag} · f {mf}", transform=ax.transAxes,
                ha="left", va="top", fontsize=8.0, color="white",
                fontweight="bold",
                bbox=dict(facecolor=VIOL_RED if tag == "violation"
                          else "black", alpha=0.72, pad=1.6,
                          edgecolor="none"))
        for s in ax.spines.values():
            s.set_color("#444444")
            s.set_linewidth(0.8)
    # OVERALL row (the viewer's LTL bar): red where the monitor is in
    # violation, with the sampled frames marked.
    ax = fig.add_subplot(inner[1, :])
    # no frame markers: the pictures' "enabled/violation/end - f N" chips
    # already locate them, and the dots doubled up on short windows
    timeline(ax, cell["n_mon"], cell["windows"], VIOL_RED, min_frac=0.004)
    n = cell["n_mon"]
    last_label_end = -1e9
    for wi, (a, b) in enumerate(cell["windows"]):
        # skip a label whose neighbour's label would overlap it (e.g. the
        # 177/213 grasp slips); the first window of a cluster names it
        if a < last_label_end + 0.02 * n:
            continue
        lab = f"{a}\u2013{b}" if b > a else f"{a}"
        # label beside the window INSIDE the bar (above collided with images)
        if b < 0.88 * n:
            ax.text(b + 0.022 * n, 0.5, lab, ha="left", va="center",
                    fontsize=6.8, color=VIOL_RED, fontweight="bold", zorder=6)
            last_label_end = b + 0.022 * n + 0.018 * n * len(lab)
        else:
            ax.text(a - 0.022 * n, 0.5, lab, ha="right", va="center",
                    fontsize=6.8, color=VIOL_RED, fontweight="bold", zorder=6)
            last_label_end = b
    ax.text(0.004, 0.5, "LTL", transform=ax.transAxes, ha="left",
            va="center", fontsize=8.5, fontweight="bold", color=VIOL_RED,
            zorder=6, bbox=dict(facecolor="white", alpha=0.75, pad=0.6,
                                edgecolor="none"))
    # predicate activation tracks, viewer-style: green where the atom is true
    for k, (atom, flags) in enumerate(cell["tracks"]):
        ax = fig.add_subplot(inner[2 + k, :])
        timeline(ax, cell["n_mon"], windows_of(flags), TRUE_GREEN)
        if k == len(cell["tracks"]) - 1:   # frame-number axis, once per cell
            n = cell["n_mon"]
            ticks = [0, n // 4, n // 2, 3 * n // 4, n - 1]
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks], fontsize=6.8,
                               fontweight="normal", color="#555555")
            ax.tick_params(axis="x", length=2, pad=1.5)
        # label INSIDE the track (left-anchored labels collided across
        # columns and clipped at the canvas edge)
        ax.text(0.004, 0.5, atom, transform=ax.transAxes, ha="left",
                va="center", fontsize=7.5, family="monospace",
                fontweight="normal", color="#222222", zorder=6,
                bbox=dict(facecolor="white", alpha=0.75, pad=0.6,
                          edgecolor="none"))
    bb = gs_cell.get_position(fig)
    # rounded pastel chip with the category (figure-intro's header style)
    chip_w = 0.017 + 0.0068 * len(cell["display"])
    fig.patches.append(FancyBboxPatch(
        (bb.x0, bb.y1 + 0.0035), chip_w, 0.0135,
        boxstyle="round,pad=0.0018,rounding_size=0.006",
        transform=fig.transFigure, facecolor=CHIP_BG, edgecolor="#7C8DB0",
        linewidth=0.9))
    fig.text(bb.x0 + chip_w / 2, bb.y1 + 0.0103, cell["display"],
             fontsize=12.5, fontweight="bold", fontstyle="italic",
             ha="center", va="center")
    fig.text(bb.x0 + chip_w + 0.012, bb.y1 + 0.0103, cell["task"],
             fontsize=12, fontweight="bold", ha="left", va="center",
             color="#444444")
    fig.text(bb.x1, bb.y1 + 0.0103, cell["obj"], fontsize=10.5,
             fontstyle="italic", fontweight="normal", ha="right",
             va="center", color="#8A4A45")
    # the formula gets its own line under the tracks -- on the header line it
    # collided with the chip + task name in every long-formula cell
    # long templates (cross-contamination) shrink instead of overflowing;
    # dropped low enough to clear the frame-number ticks of the last track
    fs = 9 if len(cell["ltl"]) <= 75 else 7.2
    fig.text((bb.x0 + bb.x1) / 2, bb.y0 - 0.0125, fmt_ltl(cell["ltl"]),
             fontsize=fs, ha="center", va="top", color="#444444")


def demo_panel(ax, stats):
    def pct(k):
        return 100 * stats[k]["violated"] / stats[k]["episodes"]
    pol = {"RoboCasa": pct("rc_policy"), "LIBERO": pct("lb_policy")}
    demo = {"RoboCasa": pct("rc_demo"), "LIBERO": pct("lb_demo")}
    w = 0.36
    for i, (suite, dark, light) in enumerate(
            (("RoboCasa", "#D64545", "#F0A8A0"),
             ("LIBERO", "#4878A8", "#A8C4E0"))):
        ax.bar(i - w / 2, pol[suite], width=w, color=dark, zorder=3)
        ax.bar(i + w / 2, demo[suite], width=w, color=light, zorder=3)
        for dx, v in ((-w / 2, pol[suite]), (w / 2, demo[suite])):
            ax.text(i + dx, v + 0.9, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=12, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["RoboCasa365", "LIBERO"], fontsize=13,
                       fontweight="bold")
    # single line: the two-line version clipped at the canvas edge
    ax.set_ylabel("Episodes with ≥1 violation (%)", fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(0, 44)
    ax.grid(axis="y", color="#CCCCCC", linewidth=0.8, linestyle="--")
    ax.legend(handles=[Patch(facecolor="#555555", label="Policy rollouts"),
                       Patch(facecolor="#C9C9C9",
                             label="Expert demonstrations")],
              loc="upper right", frameon=False, fontsize=11,
              handlelength=1.3)
    ax.set_title("(b) Expert demonstrations", fontsize=14, pad=8)


def audit_panel(ax, stats):
    v = stats["audit"]
    total = sum(v.values())
    left = 0.0
    present = []
    for key, color in (("confirmed", TRUE_GREEN), ("disputed", VIOL_RED),
                       ("unsure", "#BBBBBB")):
        frac = 100 * v[key] / total
        if v[key] == 0:
            continue
        present.append((key, color))
        ax.barh(0.0, frac, left=left, height=0.5, color=color, zorder=3)
        if frac > 4:
            ax.text(left + frac / 2, 0.0, f"{v[key]}\n({frac:.0f}%)",
                    ha="center", va="center", fontsize=11.5, color="white",
                    fontweight="bold")
        left += frac
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 0.75)
    ax.set_yticks([])
    ax.set_xlabel("Share of audited violation flags (%)", fontsize=12.5)
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="x", color="#CCCCCC", linewidth=0.8, linestyle="--")
    labels = {"confirmed": "Confirmed", "disputed": "Disputed",
              "unsure": "Unsure"}
    ax.legend(handles=[Patch(facecolor=c, label=labels[k])
                       for k, c in present],
              loc="upper center", ncol=3, frameon=False, fontsize=11,
              handlelength=1.2, bbox_to_anchor=(0.5, 1.04))
    ax.set_title(f"(c) Human audit ($n$={total})", fontsize=14, pad=8)


def main():
    apply_style()
    cells = load_cells()
    stats = scan_stats()

    fig = plt.figure(figsize=(15.0, 13.8))
    outer = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.62],
                             hspace=0.44, wspace=0.14,
                             left=0.045, right=0.985, top=0.945, bottom=0.055)
    fig.text(0.045, 0.972, "(a) Representative violations (expert demonstrations, v31)", fontsize=16,
             fontweight="bold", ha="left")
    for i, cell in enumerate(cells):
        draw_cell(fig, outer[i // 2, i % 2], cell)
    # dashed separators between example rows (figure-intro style)
    for r in range(1, 3):
        top = outer[r, 0].get_position(fig).y1
        bot = outer[r - 1, 0].get_position(fig).y0
        y = (top + 0.0175 + bot) / 2
        fig.add_artist(plt.Line2D([0.03, 0.97], [y, y],
                                  transform=fig.transFigure, color=SEP_GRAY,
                                  linewidth=1.6, linestyle=(0, (6, 4))))
    axb = fig.add_subplot(outer[3, 0])
    demo_panel(axb, stats)
    axc = fig.add_subplot(outer[3, 1])
    audit_panel(axc, stats)
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"qualitative_monitor.{ext}")
        fig.savefig(out, dpi=200 if ext == "png" else None)
        print(f"[fig] {out}")


if __name__ == "__main__":
    main()
