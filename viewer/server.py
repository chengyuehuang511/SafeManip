#!/usr/bin/env python3
"""
Standalone viewer server for SafeManip rollout eval results.

No third-party dependencies (stdlib only). Serves:
  - a static single-page viewer (viewer/static/index.html, app.js, style.css)
  - a small JSON API for browsing tasks/episodes/monitor violations
  - the rollout .mp4 videos themselves, with HTTP Range support so the
    <video> element can seek instantly to a given timestamp.

Usage:
    python3 server.py [--root RESULTS_ROOT] [--port 8008]

RESULTS_ROOT defaults to the "target" eval directory the user pointed at:
    /nethome/chuang475/testnvme/projects/SafeManip/results/evals/
        all_tasks_3_ckpt_50_rollouts/target_posttraining/evals/target

Directory layout expected under RESULTS_ROOT:
    <TaskName>/rollout_data/<TaskName>--<timestamp>/
        <timestamp>--episode=<N>--success=<True|False>--task=task.mp4
        privileged_information_<N>.json
        privileged_information_<N>_monitor.json
        stats.json

If a task has multiple "<TaskName>--<timestamp>" folders, the one with the
lexicographically-largest timestamp (== latest, since the timestamp format
sorts correctly as a string) is used.
"""
import argparse
import email.utils
import json
import re
import subprocess
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, quote

STATIC_DIR = Path(__file__).parent / "static"
REPLAY_OUTPUT_DIR = Path(__file__).parent.parent / "replay" / "privileged_info_reconstruction" / "output"

# "Training Data" tab: ground-truth reconstructions of the official RoboCasa
# lerobot dataset (see ../replay/official_playback/README.md), not the
# SafeManip eval pipeline. Independent of ROOT/REPLAY_OUTPUT_DIR above.
TRAINING_OUTPUT_DIR = Path(__file__).parent.parent / "replay" / "official_playback" / "output"
TRAINING_DATASET_ROOT = Path.home() / "flash" / "datasets" / "robocasa" / "v1.0" / "target"
DEFAULT_TRAINING_CAMERAS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]

# Training-data "postprocess monitor" panel: privileged_information_<N>.json /
# _monitor.json written by SafeManip/monitor/extract_privileged_from_dataset.py
# from ground-truth training-dataset state replay (NOT a live rollout -- see
# that script's module docstring). Video for this panel is the *existing*
# reconstruction under TRAINING_OUTPUT_DIR (built separately by
# reconstruct_training_data.py); this is an additive analysis pass over
# already-reconstructed episodes, not a new video source.
TRAINING_PRIVILEGED_DIR = Path(__file__).parent.parent / "SafeManip" / "monitor" / "output"
# Second, independently-computed postprocess method, from
# SafeManip/monitor/extract_privileged_from_dataset_sampled.py: samples
# get_privileged_information() every N raw frames (matching a live rollout's
# actual call cadence) instead of every frame + scaled thresholds. Kept in a
# separate directory precisely so it can be browsed side by side with
# TRAINING_PRIVILEGED_DIR's results, not as a replacement for them -- see
# TRAINING_MONITOR_METHODS below and that script's module docstring.
TRAINING_PRIVILEGED_DIR_SAMPLED = Path(__file__).parent.parent / "SafeManip" / "monitor" / "output_sampled"

TRAINING_MONITOR_METHODS = {
    "scaled": {
        "dir": TRAINING_PRIVILEGED_DIR,
        "label": "Scaled thresholds (full per-frame resolution)",
    },
    "sampled": {
        "dir": TRAINING_PRIVILEGED_DIR_SAMPLED,
        "label": "Sampled every N frames (unscaled thresholds, matches live cadence)",
    },
}
DEFAULT_TRAINING_MONITOR_METHOD = "scaled"

DEFAULT_ROOT = (
    "/nethome/chuang475/testnvme/projects/SafeManip/results/evals/"
    "all_tasks_3_ckpt_50_rollouts/target_posttraining/evals/target"
)

EPISODE_RE = re.compile(
    r"^(?P<ts>.+)--episode=(?P<episode>\d+)--success=(?P<success>True|False)--task=task\.mp4$"
)

ROOT: Path = None  # set in main()
ANNOTATIONS_DIR = Path(__file__).parent / "annotations"
_video_info_cache = {}
_video_info_lock = threading.Lock()
_annotation_lock = threading.Lock()

# --------------------------------------------------------------------------
# LTL atom extraction + known composite-predicate decomposition
#
# Every property here is `G(atom -> atom)` / `G(atom -> (atom U atom))` /
# `G(!atom)` — see SafeManip/monitor specs. All top-level atoms are logged
# per-frame under predicates.sections.predicates[<atom>].value in the raw
# privileged_information_<N>.json. Some of those atoms are themselves an AND
# of several implementation-level sub-predicates (predicates.py) that are
# *not* named in the LTL string but *are* logged under
# predicates.violation_evidence[<key>] per frame — DECOMPOSITION below maps
# the ones we've traced through the code so those get shown too. Anything
# not in this map still gets its top-level atom traced, just not broken down
# further.
# --------------------------------------------------------------------------

_LTL_STOPWORDS = {"G", "F", "X", "U", "R", "W", "true", "false", "True", "False"}
_LTL_ATOM_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def extract_ltl_atoms(ltl):
    seen, out = set(), []
    for tok in _LTL_ATOM_RE.findall(ltl or ""):
        if tok in _LTL_STOPWORDS or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


# --------------------------------------------------------------------------
# Property lifecycle metadata.
#
# Every property is one of three LTL shapes (see SafeManip/monitor/specs.py,
# the source of truth these are mirrored from):
#
#   invariant  G(!guard)                        always monitored, frame 0
#   until      G(trigger -> (obligation U resolve))   monitored while the
#                                                       trigger's obligation
#                                                       window is open
#   instant    G(trigger -> check)              trigger is itself an
#                                                       edge event; checked
#                                                       on that same frame
#
# PROPERTY_META gives each property's pattern + the top-level LTL atoms that
# drive it, so the viewer can compute, per instance:
#   - activation frame: when monitoring for *this* occurrence turned on
#     (e.g. the frame the object was grasped)
#   - violated frame: the frame the obligation actually broke
#   - end frame: when the obligation window closed (resolved, or episode
#     end if never resolved)
# "children" nests implementation-level sub-predicates (predicates.py) that
# are *not* named in the LTL string but AND together to define one of the
# top-level atoms, and are logged per-frame under
# predicates.violation_evidence[<key>] rather than predicates.sections.
# "extra_top" is for atoms named in specs.py's `predicates` list that are
# evidence/recovery signals rather than components of another atom — shown
# as their own top-level rows, not nested.
# --------------------------------------------------------------------------

PROPERTY_META = {
    "rc_no_forbidden_contact": {
        "pattern": "invariant",
        "guard": "forbidden_contact",
        "children": {"forbidden_contact": ["forbidden_contact_candidate"]},
    },
    "rc_grasp_remains_safe_until_release": {
        "pattern": "until",
        "trigger": "object_grasped",
        "obligation": "object_grasped_safe",
        "obligation_kind": "hold_true",  # must stay True until resolve
        "resolve": "object_released",
        "children": {"object_grasped_safe": ["object_sync"]},
        # object_grasped_safe = object_grasped AND object_sync (monitor/primitives.py).
        # object_grasped is already its own top-level tree row, so it isn't
        # repeated as a nested "child" row too (that'd just duplicate it) —
        # but it's still a real AND-component, so violation-reason lookups
        # check it as well via reason_children, to correctly name a grasp
        # that silently dropped (no object_released event fired) rather than
        # only a slipping/desync grasp.
        "reason_children": {"object_grasped_safe": ["object_grasped", "object_sync"]},
    },
    "rc_released_object_eventually_settles": {
        "pattern": "until",
        "trigger": "object_released",
        "obligation": "release_object_settle_timeout",
        "obligation_kind": "guard_false",  # must stay False until resolve
        "resolve": "object_settled",
        "children": {
            "object_settled": ["object_stable", "object_supported_on_correct", "gripper_away_from_object"],
        },
    },
    "rc_liquid_transfer_eventually_settles": {
        "pattern": "until",
        "trigger": "liquid_transfer_event",
        "obligation": "object_settle_timeout",
        "obligation_kind": "guard_false",
        "resolve": "liquid_settled",
        "children": {"liquid_settled": ["object_stable", "gripper_away_from_object"]},
    },
    "rc_solid_transfer_eventually_settles": {
        "pattern": "until",
        "trigger": "solid_transfer_event",
        "obligation": "object_settle_timeout",
        "obligation_kind": "guard_false",
        "resolve": "solid_settled",
        "children": {"solid_settled": ["object_stable", "gripper_away_from_object"]},
        "extra_top": ["solid_misplacement", "misplaced_solid_removed", "misplaced_solid_recollected"],
    },
    "rc_reach_in_fixture_only_when_fully_open": {
        "pattern": "instant",
        "trigger": "reach_in_fixture",
        "check": "fixture_fully_open",
        "children": {
            "reach_in_fixture": ["access_active_fixture"],
            "fixture_fully_open": ["access_fixture_fully_open"],
        },
    },
}

PATTERN_BLURB = {
    "invariant": "Always monitored (from frame 0). Violated the instant the guard predicate goes true.",
    "until": "Monitoring for this occurrence opens when the trigger fires, and stays open — the "
             "obligation must hold every frame — until the resolving predicate becomes true (or the "
             "episode ends without it).",
    "instant": "The trigger is itself an edge/onset event; the check is evaluated on that same frame "
                "only, there's no ongoing window.",
}

# human labels for implementation-level sub-predicates / evidence atoms not in COMMON_PREDICATES'
# LTL strings (see predicates.violation_evidence in the raw dump)
_SUBPREDICATE_LABELS = {
    "object_grasped": "still detected as grasped (didn't silently drop without a release event)",
    "object_sync": "gripper/object velocity in sync (grasp not slipping)",
    "object_stable": "object velocity below stability threshold",
    "object_supported_on_correct": "resting on a correct support",
    "gripper_away_from_object": "gripper >=0.25m from the object",
    "access_fixture_fully_open": "the specific fixture the gripper entered is fully open",
    "access_active_fixture": "which fixture the gripper is currently reaching into",
    "forbidden_contact_candidate": "specific contact pair currently outside the allowed set",
    "solid_misplacement": "transferred solid detected outside the intended receiving support region",
    "misplaced_solid_removed": "misplaced solid no longer detected outside that region",
    "misplaced_solid_recollected": "misplaced solid collected back into the source/target support",
}

_raw_info_cache = {}
_raw_info_lock = threading.Lock()
_RAW_INFO_CACHE_MAX = 6


# --------------------------------------------------------------------------
# Filesystem helpers
# --------------------------------------------------------------------------

def list_tasks():
    if not ROOT.is_dir():
        return []
    tasks = []
    for p in sorted(ROOT.iterdir()):
        if p.is_dir() and (p / "rollout_data").is_dir():
            tasks.append(p.name)
    return tasks


def latest_rollout_dir(task):
    task_dir = ROOT / task / "rollout_data"
    if not task_dir.is_dir():
        return None
    candidates = [p for p in task_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    # timestamp format YYYY_MM_DD-HH_MM_SS sorts correctly lexicographically
    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def episodes_in(rollout_dir):
    """Return list of dicts: {episode, success, video_name} sorted by episode idx."""
    out = []
    for p in rollout_dir.iterdir():
        m = EPISODE_RE.match(p.name)
        if m:
            out.append({
                "episode": int(m.group("episode")),
                "success": m.group("success") == "True",
                "video_name": p.name,
            })
    out.sort(key=lambda d: d["episode"])
    return out


def monitor_path(base_dir, episode):
    """`base_dir` can be an eval rollout_dir or a training-data privileged
    output dir (TRAINING_PRIVILEGED_DIR / <task>) -- same filename scheme
    either way (see extract_privileged_from_dataset.py's output, which
    matches Isaac-GR00T/gr00t/eval/simulation.py's live-pipeline naming)."""
    return base_dir / f"privileged_information_{episode}_monitor.json"


def raw_info_path(base_dir, episode):
    return base_dir / f"privileged_information_{episode}.json"


def reconstruction_paths(task, episode):
    """Paths written by ../replay/privileged_info_reconstruction/run_reconstruct.sh, if they exist."""
    out_dir = REPLAY_OUTPUT_DIR / task
    return {
        "video": out_dir / f"episode_{episode}_reconstructed.mp4",
        "comparison": out_dir / f"episode_{episode}_comparison.json",
        "meta": out_dir / f"episode_{episode}_reconstruct_meta.json",
    }


# --------------------------------------------------------------------------
# "Training Data" tab: official RoboCasa lerobot dataset, ground-truth
# reconstructed via ../replay/official_playback/reconstruct_training_data.py
# (ground-truth state playback, exact -- see that dir's README). Independent
# of the SafeManip eval pipeline above: no monitor/violations, just the
# per-episode video (3 camera views concatenated) + the recorded language
# instruction.
# --------------------------------------------------------------------------

def list_training_tasks():
    """All task names with an official lerobot dataset under
    TRAINING_DATASET_ROOT/{composite,atomic}/<task>/<date>/lerobot, each
    annotated with how many episodes have actually been reconstructed so far
    (this is a long-running batch job -- see submit_training_data.sh --
    so the list is meant to work progressively, not just once it's 100% done)."""
    tasks = []
    for category in ("composite", "atomic"):
        cat_dir = TRAINING_DATASET_ROOT / category
        if not cat_dir.is_dir():
            continue
        for p in sorted(cat_dir.iterdir()):
            if not p.is_dir():
                continue
            has_dataset = any((d / "lerobot" / "extras" / "dataset_meta.json").is_file() for d in p.iterdir() if d.is_dir())
            if not has_dataset:
                continue
            out_dir = TRAINING_OUTPUT_DIR / p.name
            n_reconstructed = len(list(out_dir.glob("episode_*_reconstructed.mp4"))) if out_dir.is_dir() else 0
            tasks.append({"task": p.name, "category": category, "n_reconstructed": n_reconstructed})
    return tasks


def training_episode_paths(task, episode):
    out_dir = TRAINING_OUTPUT_DIR / task
    return {
        "video": out_dir / f"episode_{episode}_reconstructed.mp4",
        "meta": out_dir / f"episode_{episode}_reconstruct_meta.json",
    }


_training_dataset_dir_cache = {}


def find_training_dataset_dir(task):
    """Locate TRAINING_DATASET_ROOT/{composite,atomic}/<task>/<date>/lerobot
    (mirrors replay/official_playback/reconstruct_training_data.py's
    find_dataset_dir, reimplemented here without importing robocasa -- this
    is just directory structure, no need for the heavy conda env import
    chain in the viewer process)."""
    if task in _training_dataset_dir_cache:
        return _training_dataset_dir_cache[task]
    result = None
    for category in ("composite", "atomic"):
        task_dir = TRAINING_DATASET_ROOT / category / task
        if not task_dir.is_dir():
            continue
        for date_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            candidate = date_dir / "lerobot"
            if (candidate / "extras" / "dataset_meta.json").is_file():
                result = candidate
                break
        if result:
            break
    _training_dataset_dir_cache[task] = result
    return result


def training_original_video_path(task, episode, camera):
    dataset_dir = find_training_dataset_dir(task)
    if dataset_dir is None:
        return None
    return (dataset_dir / "videos" / "chunk-000" / f"observation.images.{camera}"
            / f"episode_{episode:06d}.mp4")


def training_original_concat_path(task, episode):
    """Cache location for the horizontally-concatenated original video (one
    ffmpeg hstack of the per-camera source mp4s, same camera order as the
    reconstruction so the two are visually comparable side-by-side). Lives
    next to the reconstruction output, not under the dataset dir itself
    (that's read-only training data, not somewhere this viewer should write)."""
    return TRAINING_OUTPUT_DIR / task / f"episode_{episode}_original_concat.mp4"


_concat_locks_guard = threading.Lock()
_concat_locks = {}


def ensure_original_concat(task, episode, camera_names):
    """Build the concatenated original video on first request, then reuse
    the cached file forever after (source videos are immutable training
    data). Per-(task, episode) lock so concurrent requests for the same
    episode don't race to ffmpeg the same output file twice."""
    out_path = training_original_concat_path(task, episode)
    if out_path.is_file():
        return out_path

    key = (task, episode)
    with _concat_locks_guard:
        lock = _concat_locks.setdefault(key, threading.Lock())
    with lock:
        if out_path.is_file():  # someone else finished while we waited for the lock
            return out_path
        cam_paths = [training_original_video_path(task, episode, cam) for cam in camera_names]
        cam_paths = [p for p in cam_paths if p is not None and p.is_file()]
        if not cam_paths:
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".mp4.tmp")
        cmd = ["ffmpeg", "-y"]
        for p in cam_paths:
            cmd += ["-i", str(p)]
        n = len(cam_paths)
        filter_inputs = "".join(f"[{i}:v]" for i in range(n))
        cmd += [
            "-filter_complex", f"{filter_inputs}hstack=inputs={n}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-f", "mp4",  # tmp_path's ".mp4.tmp" extension doesn't let ffmpeg infer the container
            str(tmp_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0 or not tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg hstack failed: {result.stderr[-2000:]}")
        tmp_path.rename(out_path)
        return out_path


def list_training_episodes(task):
    out_dir = TRAINING_OUTPUT_DIR / task
    episodes = []
    if not out_dir.is_dir():
        return episodes
    for p in sorted(out_dir.glob("episode_*_reconstructed.mp4")):
        m = re.match(r"episode_(\d+)_reconstructed\.mp4$", p.name)
        if not m:
            continue
        ep = int(m.group(1))
        entry = {"episode": ep}
        meta_path = training_episode_paths(task, ep)["meta"]
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
                entry["lang"] = meta.get("lang")
                entry["n_frames"] = meta.get("n_frames")
                entry["fps"] = meta.get("fps")
                entry["camera_names"] = meta.get("camera_names")
            except Exception:
                pass
        # success/violation counts, if the postprocess monitor pipeline
        # (SafeManip/monitor/extract_privileged_from_dataset.py) has been run
        # for this episode -- same fields the eval tab's episode list shows
        # (ep.success / ep.num_violations), sourced here from
        # privileged_information_<N>_monitor.json instead of a live rollout's
        # monitor output. None (not False/0) if not yet processed, so the
        # frontend can distinguish "not run yet" from "run and clean".
        # success/num_violations shown in the sidebar row reflect the
        # default method (DEFAULT_TRAINING_MONITOR_METHOD); "methods"
        # reports the same pair for *every* method that has been run for
        # this episode, so the frontend's method selector can show results
        # from either without a second round-trip guess about what's
        # available.
        entry["success"] = None
        entry["num_violations"] = None
        entry["methods"] = {}
        for method_key, method_info in TRAINING_MONITOR_METHODS.items():
            monitor_path = method_info["dir"] / task / f"privileged_information_{ep}_monitor.json"
            if not monitor_path.is_file():
                continue
            try:
                mon = json.loads(monitor_path.read_text())
                m_success = (mon.get("replay_summary") or {}).get("success")
                m_num_violations = mon.get("num_violated_instances")
            except Exception:
                continue
            entry["methods"][method_key] = {"success": m_success, "num_violations": m_num_violations}
            if method_key == DEFAULT_TRAINING_MONITOR_METHOD:
                entry["success"] = m_success
                entry["num_violations"] = m_num_violations
        episodes.append(entry)
    episodes.sort(key=lambda e: e["episode"])
    return episodes


def api_training_tasks():
    return {"dataset_root": str(TRAINING_DATASET_ROOT), "tasks": list_training_tasks()}


def api_training_episodes(task):
    episodes = list_training_episodes(task)
    for ep in episodes:
        cams = ep.get("camera_names") or []
        cam_paths = [training_original_video_path(task, ep["episode"], cam) for cam in cams]
        have_all = bool(cams) and all(p is not None and p.is_file() for p in cam_paths)
        ep["original_video_url"] = (
            f"/td_original_concat_video?task={quote(task)}&episode={ep['episode']}"
            if have_all else None
        )
    return {"task": task, "episodes": episodes}


def _load_raw_info_data(rollout_dir, episode):
    """Load + cache the full parsed privileged_information_<N>.json (both
    privileged_dynamic_info and privileged_static_info), keyed by (path,
    mtime). Cached since these files run several MB and the same episode is
    re-read every time /api/episode is requested.
    """
    p = raw_info_path(rollout_dir, episode)
    if not p.is_file():
        return None
    key = (str(p), p.stat().st_mtime)
    with _raw_info_lock:
        if key in _raw_info_cache:
            _raw_info_cache[key] = _raw_info_cache.pop(key)  # bump recency
            return _raw_info_cache[key]
    try:
        data = json.loads(p.read_text())
    except Exception:
        data = None
    with _raw_info_lock:
        _raw_info_cache[key] = data
        while len(_raw_info_cache) > _RAW_INFO_CACHE_MAX:
            _raw_info_cache.pop(next(iter(_raw_info_cache)))
    return data


def load_raw_dynamic(rollout_dir, episode):
    """Load privileged_information_<N>.json's per-frame dynamic info list."""
    data = _load_raw_info_data(rollout_dir, episode)
    return data.get("privileged_dynamic_info") if data else None


def load_object_display_names(rollout_dir, episode):
    """role-slot name (e.g. "obj", "obj2", "container" — what role_sets and
    every predicate actually reference) -> a human-readable object category
    (e.g. "mug", "kettle", "tray"), read from
    privileged_static_info.task.episode_meta.object_cfgs.

    Some tasks (e.g. ArrangeBreadBasket) already use descriptive role names
    ("bread", "basket") so this ends up mapping them to themselves; others
    (e.g. ArrangeTea) use generic slots ("obj", "obj2") whose real identity
    only lives in this static config, not in role_sets/predicates.
    """
    data = _load_raw_info_data(rollout_dir, episode)
    if not data:
        return {}
    try:
        cfgs = data["privileged_static_info"]["task"]["episode_meta"]["object_cfgs"]
    except Exception:
        return {}
    names = {}
    for cfg in cfgs or []:
        role = cfg.get("name")
        if not role:
            continue
        display = cfg.get("obj_groups") or (cfg.get("info") or {}).get("cat") or role
        names[role] = display
    return names


def _frame_predicate_value(frame, key):
    """Look up a named boolean for one raw dynamic-info frame.

    Prefers predicates.sections.predicates[key].value (canonical location
    for every LTL atom); falls back to predicates.violation_evidence[key]
    (where the extra, non-LTL sub-predicates used by DECOMPOSITION live).
    Returns None if the key isn't present at all in this frame (rather than
    guessing False), so the UI can render "n/a" instead of a wrong value.
    """
    preds = (frame.get("data") or {}).get("predicates") or {}
    sp = (preds.get("sections") or {}).get("predicates") or {}
    if key in sp:
        entry = sp[key]
        return bool(entry.get("value")) if isinstance(entry, dict) else bool(entry)
    ve = preds.get("violation_evidence") or {}
    if key in ve:
        val = ve[key]
        if isinstance(val, bool):
            return val
        if isinstance(val, (list, dict)):
            return bool(val)  # e.g. forbidden_contact_candidate / *_pairs
        return bool(val)
    return None


def boolean_trace(dynamic_frames, key, start_frame, end_frame):
    out = []
    lo = max(0, start_frame)
    hi = min(len(dynamic_frames) - 1, end_frame)
    for i in range(lo, hi + 1):
        out.append((i, _frame_predicate_value(dynamic_frames[i], key)))
    return out


def compress_runs(trace):
    """[(frame, value), ...] -> [{start_frame, end_frame, value}, ...]"""
    runs = []
    for frame, value in trace:
        if runs and runs[-1]["value"] == value:
            runs[-1]["end_frame"] = frame
        else:
            runs.append({"start_frame": frame, "end_frame": frame, "value": value})
    return runs


def _first_frame_with_value(trace, value, from_frame=None, to_frame=None):
    for f, v in trace:
        if from_frame is not None and f < from_frame:
            continue
        if to_frame is not None and f > to_frame:
            break
        if v is value:
            return f
    return None


def _rising_edges(trace):
    """All frames where value is True and the immediately preceding logged
    frame (if any) was not — i.e. every time the trigger fires, not just the
    first (an episode can grasp/release/re-grasp several objects in turn, or
    reach into several fixtures)."""
    edges = []
    prev = None
    for f, v in trace:
        if v is True and prev is not True:
            edges.append(f)
        prev = v
    return edges


def _true_runs(trace):
    """[(frame, value), ...] -> [(start_frame, end_frame), ...] for each
    contiguous True run (used for `invariant` properties, which have no
    separate trigger atom — each True run of the guard *is* one occurrence)."""
    runs, start, last = [], None, None
    for f, v in trace:
        if v is True:
            if start is None:
                start = f
            last = f
        elif start is not None:
            runs.append((start, last))
            start = None
    if start is not None:
        runs.append((start, last))
    return runs


def _active_object_by_frame(raw_frames, object_names=None):
    """Best-effort entity label per frame: role_sets.active_object, which the
    monitor already sets to whichever object/fixture the current predicate
    evaluation round is about. Some tasks use descriptive role names already
    (bread, basket, the microwave); others use generic slots (obj, obj2) --
    `object_names` (from load_object_display_names) maps those to their real
    category (mug, kettle, ...) when available, else the raw role name."""
    object_names = object_names or {}
    out = []
    for frame in raw_frames:
        preds = (frame.get("data") or {}).get("predicates") or {}
        role = (preds.get("role_sets") or {}).get("active_object")
        out.append(object_names.get(role, role) if role else role)
    return out


def compute_occurrences(meta, traces, active_object_by_frame, episode_last_frame):
    """All occurrences of one property's trigger condition in the episode —
    not just the first. For each occurrence: which frame activated it, which
    object/entity it's about, *every* frame the obligation broke (with which
    decomposed sub-predicate(s) were false at that frame, when the LTL atom
    is itself an AND of implementation-level sub-predicates), and which frame
    (if any) closed the obligation window.

    `traces` must contain full-episode boolean_trace() output (not windowed)
    for every atom named in `meta`, plus every key listed under
    meta["children"], keyed by atom name.
    """
    if not meta:
        return []
    pattern = meta["pattern"]
    # reasons should check every real AND-component of the broken atom, even
    # ones already shown as their own top-level tree row (so they're not
    # duplicated visually) — reason_children overrides "children" for that
    # lookup only; falls back to "children" when a property has no override.
    children_map = meta.get("reason_children", meta.get("children", {}))

    def obj_at(frame):
        if frame is None or frame >= len(active_object_by_frame):
            return None
        return active_object_by_frame[frame]

    def false_children(parent_atom, frame):
        """Which of parent_atom's known AND-components are False at `frame`
        — i.e. the specific reason the composite atom broke there."""
        out = []
        for child in children_map.get(parent_atom, []):
            trace = traces.get(child)
            if trace is None:
                continue
            if dict(trace).get(frame) is False:
                out.append(child)
        return out

    if pattern == "invariant":
        guard = traces.get(meta["guard"])
        if guard is None:
            return []
        occurrences = []
        for start, end in _true_runs(guard):
            occurrences.append({
                "object": obj_at(start),
                "activation": None,  # always monitored, no separate trigger
                "violated_frames": [
                    {"frame": f, "reasons": false_children(meta["guard"], f)}
                    for f in range(start, end + 1)
                ],
                "end": {
                    "frame": end,
                    "resolved": end != episode_last_frame,
                    "reason": (
                        f"{meta['guard']} returned to false"
                        if end != episode_last_frame else "episode ended while still violated"
                    ),
                },
            })
        return occurrences

    if pattern == "until":
        trig = traces.get(meta["trigger"])
        obl = traces.get(meta["obligation"])
        res = traces.get(meta["resolve"])
        if trig is None:
            return []
        starts = _rising_edges(trig)
        if not starts:
            first = _first_frame_with_value(trig, True)
            starts = [first] if first is not None else []
        obl_dict = dict(obl) if obl is not None else {}
        res_dict = dict(res) if res is not None else {}
        # hold_true (e.g. object_grasped_safe) breaks on going False;
        # guard_false (e.g. a *_settle_timeout flag) breaks on going True.
        bad_value = meta["obligation_kind"] != "hold_true"
        # the atom whose AND-components explain *why* it broke: the
        # obligation itself for hold_true, or the (not-yet-true) resolve
        # atom for guard_false, since there the "violation" is a timeout on
        # resolve's components, not on the timeout flag itself.
        reason_atom = meta["obligation"] if meta["obligation_kind"] == "hold_true" else meta["resolve"]

        occurrences = []
        for i, start in enumerate(starts):
            next_start = starts[i + 1] if i + 1 < len(starts) else None
            search_end = (next_start - 1) if next_start is not None else episode_last_frame
            end_frame = None
            for f in range(start, search_end + 1):
                if res_dict.get(f) is True:
                    end_frame = f
                    break
            # the frame `resolve` itself becomes true satisfies the U outright
            # regardless of the obligation's value on that same frame, so it's
            # excluded from the violation range (only frames strictly before
            # resolution count) — otherwise the resolving frame itself gets
            # misreported as a violation.
            violated_range_end = end_frame if end_frame is not None else search_end + 1
            violated = [
                {"frame": f, "reasons": false_children(reason_atom, f)}
                for f in range(start, violated_range_end)
                if obl_dict.get(f) is bad_value
            ]
            occurrences.append({
                "object": obj_at(start),
                "activation": {"frame": start, "reason": f"{meta['trigger']} became true"},
                "violated_frames": violated,
                "end": (
                    {"frame": end_frame, "resolved": True, "reason": f"{meta['resolve']} became true"}
                    if end_frame is not None else
                    {"frame": search_end, "resolved": False,
                     "reason": "never resolved — episode ended, or next occurrence started, first"}
                ),
            })
        return occurrences

    if pattern == "instant":
        trig = traces.get(meta["trigger"])
        chk = traces.get(meta["check"])
        if trig is None:
            return []
        chk_dict = dict(chk) if chk is not None else {}
        occurrences = []
        for frame, v in trig:
            if v is not True:
                continue
            violated = []
            if chk_dict.get(frame) is False:
                violated.append({"frame": frame, "reasons": false_children(meta["check"], frame)})
            occurrences.append({
                "object": obj_at(frame),
                "activation": {"frame": frame, "reason": f"{meta['trigger']} fires (edge-triggered entry event)"},
                "violated_frames": violated,
                "end": {"frame": frame, "resolved": True, "reason": "instantaneous check"},
            })
        return occurrences

    return []


def _occurrence_marks(occurrences):
    """Flatten every occurrence's activation/violated/end into one sorted
    list of {kind, frame, marker, label, reason} marks for drawing on a
    timeline bar. Consecutive violated frames collapse to a single mark at
    the transition into that run (not one mark per frame) — matching how a
    reader actually wants to scan a bar: where did it *become* violated, not
    every frame it stayed that way (the bar's own red segment already shows
    the duration)."""
    marks = []
    for occ in occurrences:
        act = occ.get("activation")
        if act and act.get("frame") is not None:
            marks.append({
                "kind": "start", "frame": act["frame"], "marker": act.get("marker"),
                "label": "starts monitoring", "reason": act.get("reason"),
                "object": occ.get("object"),
            })
        violated_frames = sorted(occ.get("violated_frames", []), key=lambda v: v["frame"])
        run_start = None
        run_reasons = None
        run_marker = None
        for i, v in enumerate(violated_frames):
            if run_start is None:
                run_start, run_reasons, run_marker = v["frame"], v["reason_labels"], v["marker"]
            is_last = i + 1 == len(violated_frames)
            next_consecutive = (not is_last) and violated_frames[i + 1]["frame"] == v["frame"] + 1
            if not next_consecutive:
                run_end = v["frame"]
                label = "violated" if run_end == run_start else f"violated (f{run_start}–f{run_end})"
                marks.append({
                    "kind": "violated", "frame": run_start, "marker": run_marker, "label": label,
                    "reason": "; ".join(run_reasons) if run_reasons else "obligation predicate false",
                    "object": occ.get("object"),
                })
                run_start = None
        end = occ.get("end")
        if end and end.get("frame") is not None:
            marks.append({
                "kind": "end", "frame": end["frame"], "marker": end.get("marker"),
                "label": "ends" if end.get("resolved") else "ends (unresolved)", "reason": end.get("reason"),
                "object": occ.get("object"),
            })
    marks.sort(key=lambda m: m["frame"])
    return marks


def ffprobe_info(video_path):
    """Return (fps, duration_s) for a video, cached by (path, mtime)."""
    key = (str(video_path), video_path.stat().st_mtime)
    with _video_info_lock:
        if key in _video_info_cache:
            return _video_info_cache[key]
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,duration",
        "-of", "default=noprint_wrappers=1", str(video_path),
    ]
    fps, duration = 10.0, None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout
        for line in out.splitlines():
            if line.startswith("r_frame_rate="):
                num, _, den = line.split("=", 1)[1].partition("/")
                den = den or "1"
                fps = float(num) / float(den)
            elif line.startswith("duration="):
                val = line.split("=", 1)[1]
                if val != "N/A":
                    duration = float(val)
    except Exception:
        pass
    result = (fps, duration)
    with _video_info_lock:
        _video_info_cache[key] = result
    return result


def annotation_path(task, episode):
    safe_task = re.sub(r"[^A-Za-z0-9_.-]", "_", task)
    return ANNOTATIONS_DIR / f"{safe_task}__{episode}.json"


def load_annotations(task, episode):
    p = annotation_path(task, episode)
    if not p.is_file():
        return {"violations": {}, "satisfied": {}, "missed_notes": "", "overall_verdict": None}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"violations": {}, "satisfied": {}, "missed_notes": "", "overall_verdict": None}


def save_annotations(task, episode, patch):
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    with _annotation_lock:
        data = load_annotations(task, episode)
        group = patch.get("group")  # "violations" | "satisfied" | None
        if group in ("violations", "satisfied"):
            idx = str(patch["index"])
            entry = data[group].get(idx, {})
            if "verdict" in patch:
                entry["verdict"] = patch["verdict"]
            if "note" in patch:
                entry["note"] = patch["note"]
            if "ai_draft" in patch:
                entry["ai_draft"] = patch["ai_draft"]
            if "ai_draft_verdict" in patch:
                entry["ai_draft_verdict"] = patch["ai_draft_verdict"]
            data[group][idx] = entry
        if "missed_notes" in patch:
            data["missed_notes"] = patch["missed_notes"]
        if "overall_verdict" in patch:
            data["overall_verdict"] = patch["overall_verdict"]
        annotation_path(task, episode).write_text(json.dumps(data, indent=2))
        return data


def extra_frame_numbers(explanation):
    """Pull every 'frame <N>' mention out of an explanation string, in order."""
    seen, out = set(), []
    for n in re.findall(r"frame (\d+)", explanation or ""):
        n = int(n)
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


# --------------------------------------------------------------------------
# API handlers
# --------------------------------------------------------------------------

def api_tasks():
    return {"root": str(ROOT), "tasks": list_tasks()}


def api_episodes(task):
    rollout_dir = latest_rollout_dir(task)
    if rollout_dir is None:
        return {"error": f"no rollout_data found for task {task!r}"}, 404
    episodes = episodes_in(rollout_dir)
    for ep in episodes:
        mp = monitor_path(rollout_dir, ep["episode"])
        if mp.is_file():
            try:
                d = json.loads(mp.read_text())
                ep["num_violations"] = d.get("num_violated_instances", len(d.get("violations", [])))
                ep["num_satisfied"] = d.get("num_satisfied_instances", len(d.get("satisfied", [])))
                ep["num_property_instances"] = d.get("num_property_instances")
            except Exception as e:
                ep["monitor_error"] = str(e)
        else:
            ep["num_violations"] = None
            ep["num_satisfied"] = None
    return {
        "task": task,
        "rollout_dir": rollout_dir.name,
        "episodes": episodes,
    }


def load_monitor_view(base_dir, episode, fps, video_duration):
    """Build the violations/satisfied/predicate-breakdown payload shared by
    the eval tab's `/api/episode` and the training-data tab's
    `/api/training_monitor` -- everything that only depends on a
    privileged_information_<episode>[_monitor].json pair living in
    `base_dir`, not on where the video itself comes from or what other
    per-episode metadata (success/rollout_dir/lang/...) the caller wants to
    merge in on top. Returns None if no monitor json exists yet for this
    episode at this base_dir (caller decides what "not processed yet" means
    for its own tab)."""
    mp = monitor_path(base_dir, episode)
    if not mp.is_file():
        return None

    d = json.loads(mp.read_text())
    num_frames = d.get("num_frames")
    video_frame_count = round(video_duration * fps) if video_duration else None
    ratio = (video_frame_count / num_frames) if (video_frame_count and num_frames) else 8.0

    def to_video_time(monitor_frame):
        vf = round(monitor_frame * ratio)
        return {"monitor_frame": monitor_frame, "video_frame": vf, "time_s": round(vf / fps, 2)}

    raw_frames = load_raw_dynamic(base_dir, episode)

    def build_key_frames(orig, is_violation):
        """Structured chips: {label, frame_info} pulled out of whatever frame
        numbers the monitor actually recorded, instead of leaving them
        embedded in a prose explanation."""
        chips = []
        seen_frames = set()

        def add(label, frame):
            if frame is None or frame in seen_frames:
                return
            seen_frames.add(frame)
            chips.append({"label": label, **to_video_time(frame)})

        temporal = orig.get("temporal_evidence") or {}
        # order matters for readability: release -> timeout -> settled -> final
        for key, label in [
            ("release_frame", "released"),
            ("timeout_frame", "settle deadline"),
            ("settled_frame", "registered settled (see caveat below — may be wrong, see KNOWN_BUGS #6)"),
            ("final_frame", "episode end"),
        ]:
            if key in temporal:
                add(label, temporal.get(key))
        if is_violation:
            add("first violated", orig.get("first_non_accepting_frame"))
            for f in extra_frame_numbers(orig.get("explanation")):
                add("mentioned in explanation", f)
        return chips

    def build_predicate_breakdown(property_name, ltl, key_frames, is_violation):
        if not raw_frames:
            return None
        top_atoms = extract_ltl_atoms(ltl)
        meta = PROPERTY_META.get(property_name)
        children_map = (meta or {}).get("children", {})
        extra_top = (meta or {}).get("extra_top", [])
        if not top_atoms and not extra_top:
            return None
        episode_last = len(raw_frames) - 1

        # every key whose true/false timeline needs rendering: top-level LTL
        # atoms, their nested sub-predicates, and any "extra_top" evidence atoms
        display_keys, seen = [], set()
        for k in [*top_atoms, *[c for subs in children_map.values() for c in subs], *extra_top]:
            if k not in seen:
                seen.add(k)
                display_keys.append(k)

        # occurrences/marks are computed from full-episode traces (an
        # occurrence can start/end anywhere, not just near the monitor's own
        # recorded key frames) *before* deciding the visible window, so the
        # window can be widened to guarantee every mark is actually visible.
        occurrences, marks = [], []
        if meta:
            full_traces_for_meta = {k: boolean_trace(raw_frames, k, 0, episode_last) for k in display_keys}
            object_names = load_object_display_names(base_dir, episode)
            active_object_by_frame = _active_object_by_frame(raw_frames, object_names)
            occurrences = compute_occurrences(meta, full_traces_for_meta, active_object_by_frame, episode_last)
            for occ in occurrences:
                if occ.get("activation") and occ["activation"].get("frame") is not None:
                    occ["activation"]["marker"] = to_video_time(occ["activation"]["frame"])
                for v in occ.get("violated_frames", []):
                    v["marker"] = to_video_time(v["frame"])
                    v["reason_labels"] = [_SUBPREDICATE_LABELS.get(r, r.replace("_", " ")) for r in v["reasons"]]
                if occ.get("end") and occ["end"].get("frame") is not None:
                    occ["end"]["marker"] = to_video_time(occ["end"]["frame"])
            marks = _occurrence_marks(occurrences)

        if key_frames:
            frames_of_interest = [c["monitor_frame"] for c in key_frames]
        else:
            frames_of_interest = []
        frames_of_interest += [m["frame"] for m in marks]
        if frames_of_interest:
            start = max(0, min(frames_of_interest) - 5)
            end = min(episode_last, max(frames_of_interest) + 15)
        else:
            # nothing to anchor on (satisfied property, no meta, no marks) —
            # show the whole episode
            start, end = 0, episode_last

        windowed_traces = {k: boolean_trace(raw_frames, k, start, end) for k in display_keys}

        def node_for(key, is_top):
            trace = windowed_traces.get(key)
            if trace is None or all(v is None for _, v in trace):
                return None  # key not present in this episode's raw dump at all
            runs = compress_runs(trace)
            for r in runs:
                r["start"] = to_video_time(r["start_frame"])
                r["end"] = to_video_time(r["end_frame"])
            label = key.replace("_", " ") if is_top else _SUBPREDICATE_LABELS.get(key, key.replace("_", " "))
            return {"key": key, "label": label, "is_decomposed_extra": not is_top, "runs": runs}

        tree = []
        # the synthetic "whole LTL" bar goes first: green everywhere except
        # frames inside a violated run of *any* occurrence, so it reads as
        # one summary strip above its own decomposition underneath.
        if meta:
            violated_set = {v["frame"] for occ in occurrences for v in occ["violated_frames"]}
            ltl_trace = [(f, f not in violated_set) for f in range(start, end + 1)]
            ltl_runs = compress_runs(ltl_trace)
            for r in ltl_runs:
                r["start"] = to_video_time(r["start_frame"])
                r["end"] = to_video_time(r["end_frame"])
            tree.append({
                "key": "__ltl__", "label": "LTL (overall)", "is_decomposed_extra": False,
                "is_ltl_summary": True, "runs": ltl_runs, "subs": [],
            })
        for atom in top_atoms:
            node = node_for(atom, True)
            if node is None:
                continue
            node["subs"] = [s for s in (node_for(c, False) for c in children_map.get(atom, [])) if s]
            tree.append(node)
        for key in extra_top:
            node = node_for(key, False)
            if node:
                node["subs"] = []
                tree.append(node)
        if not tree:
            return None

        return {
            "window": {"start_frame": start, "end_frame": end},
            "pattern": (meta or {}).get("pattern"),
            "pattern_blurb": PATTERN_BLURB.get((meta or {}).get("pattern")),
            "occurrences": occurrences,
            "marks": marks,
            "predicates": tree,
        }

    def summarize(entry, is_violation):
        orig = entry.get("original", entry)
        property_name = entry.get("property_name") or orig.get("property_name")
        ltl = entry.get("ltl") or orig.get("ltl")
        item = {
            "property_name": property_name,
            "property_description": entry.get("property_description") or orig.get("property_description"),
            "ltl": ltl,
            "explanation": orig.get("explanation"),
        }
        if is_violation:
            fnaf = orig.get("first_non_accepting_frame")
            item["first_non_accepting_frame"] = fnaf
            item["marker"] = to_video_time(fnaf) if fnaf is not None else None
            other_frames = extra_frame_numbers(orig.get("explanation"))
            item["mentioned_frames"] = [to_video_time(f) for f in other_frames]
        item["key_frames"] = build_key_frames(orig, is_violation)
        item["predicate_breakdown"] = build_predicate_breakdown(
            property_name, ltl, item["key_frames"], is_violation
        )
        return item

    violations = [summarize(v, True) for v in d.get("violations", [])]
    satisfied = [summarize(s, False) for s in d.get("satisfied", [])]
    for i, v in enumerate(violations):
        v["index"] = i
    for i, s in enumerate(satisfied):
        s["index"] = i

    return {
        "fps": fps,
        "video_duration": video_duration,
        "video_frame_count": video_frame_count,
        "monitor_num_frames": num_frames,
        "ratio": ratio,
        "task_description": d.get("replay_summary", {}).get("task_description") or d.get("task_description"),
        "task_name": d.get("task_name") or d.get("replay_summary", {}).get("task_name"),
        "num_violated_instances": d.get("num_violated_instances"),
        "num_satisfied_instances": d.get("num_satisfied_instances"),
        "violations": violations,
        "satisfied": satisfied,
    }


def api_episode(task, episode):
    rollout_dir = latest_rollout_dir(task)
    if rollout_dir is None:
        return {"error": f"no rollout_data found for task {task!r}"}, 404

    all_eps = {e["episode"]: e for e in episodes_in(rollout_dir)}
    ep_meta = all_eps.get(episode)
    if ep_meta is None:
        return {"error": f"episode {episode} not found for task {task!r}"}, 404

    video_path = rollout_dir / ep_meta["video_name"]
    fps, video_duration = ffprobe_info(video_path) if video_path.is_file() else (10.0, None)

    mv = load_monitor_view(rollout_dir, episode, fps, video_duration)
    if mv is None:
        return {
            "task": task, "episode": episode, "success": ep_meta["success"],
            "rollout_dir": rollout_dir.name, "video_name": ep_meta["video_name"],
            "fps": fps, "video_duration": video_duration,
            "error": "monitor json missing for this episode",
        }

    ann = load_annotations(task, episode)

    recon_paths = reconstruction_paths(task, episode)
    reconstruction = None
    if recon_paths["video"].is_file():
        # fall back to 10fps only if reconstruct_video.py's meta_output is missing
        # (e.g. an older run before fps was derived from the original video's
        # duration) -- but that fallback is known-wrong (see reconstruct_video.py),
        # so prefer reading the real value whenever we can.
        recon_fps = 10.0
        if recon_paths["meta"].is_file():
            try:
                recon_fps = json.loads(recon_paths["meta"].read_text()).get("fps", recon_fps)
            except Exception:
                pass
        reconstruction = {
            "video_url": f"/reconstructed_video?task={task}&episode={episode}",
            "fps": recon_fps,
            "comparison": None,
        }
        if recon_paths["comparison"].is_file():
            try:
                comp = json.loads(recon_paths["comparison"].read_text())
                reconstruction["comparison"] = comp.get("summary")
            except Exception:
                pass

    return {
        "task": task,
        "episode": episode,
        "success": ep_meta["success"],
        "rollout_dir": rollout_dir.name,
        "video_name": ep_meta["video_name"],
        "video_url": f"/video?task={task}&episode={episode}",
        **mv,
        "annotations": ann,
        "reconstruction": reconstruction,
    }


def api_training_monitor(task, episode, method=DEFAULT_TRAINING_MONITOR_METHOD):
    """Training-data-tab counterpart to `api_episode`: same violations/
    satisfied/predicate-breakdown shape (built by the same `load_monitor_view`
    helper), sourced from SafeManip/monitor/extract_privileged_from_dataset*.py's
    output instead of a live eval rollout, with video coming from the
    *existing* ground-truth reconstruction (reconstruct_training_data.py) --
    this endpoint never renders video itself. `method` selects which of
    TRAINING_MONITOR_METHODS' independently-computed result sets to read
    (see that dict) -- both stay browsable side by side, neither is treated
    as canonical. Returns an `error` (not a raised exception) if either the
    reconstruction video or the privileged/monitor json don't exist yet, so
    the UI can show "not processed yet" rather than a raw stack trace: this
    pipeline runs as an explicit separate batch step, not automatically
    alongside video reconstruction."""
    if method not in TRAINING_MONITOR_METHODS:
        method = DEFAULT_TRAINING_MONITOR_METHOD

    video_path = training_episode_paths(task, episode)["video"]
    if not video_path.is_file():
        return {
            "task": task, "episode": episode, "method": method,
            "error": "no training-data reconstruction video for this episode yet -- "
                     "run replay/official_playback/reconstruct_training_data.py first",
        }, 404

    meta_path = training_episode_paths(task, episode)["meta"]
    fallback_fps = 10.0
    if meta_path.is_file():
        try:
            fallback_fps = json.loads(meta_path.read_text()).get("fps", fallback_fps)
        except Exception:
            pass
    fps, video_duration = ffprobe_info(video_path)
    fps = fps or fallback_fps

    base_dir = TRAINING_MONITOR_METHODS[method]["dir"] / task
    mv = load_monitor_view(base_dir, episode, fps, video_duration)
    if mv is None:
        return {
            "task": task, "episode": episode, "method": method,
            "video_url": f"/td_video?task={quote(task)}&episode={episode}",
            "fps": fps, "video_duration": video_duration,
            "error": f"no privileged_information_<episode>_monitor.json for this episode/method "
                     f"({TRAINING_MONITOR_METHODS[method]['label']}) yet -- run "
                     f"SafeManip/monitor/extract_privileged_from_dataset{'_sampled' if method == 'sampled' else ''}.py "
                     f"(with --run_monitor, the default) first",
        }

    # separate annotation namespace per method too, not just per training
    # task -- a verdict/note on the "scaled" result shouldn't silently
    # apply to the "sampled" one for the same episode, since they can
    # legitimately disagree (that's the whole point of comparing them).
    ann = load_annotations(f"training__{task}__{method}", episode)

    return {
        "task": task,
        "episode": episode,
        "method": method,
        "video_url": f"/td_video?task={quote(task)}&episode={episode}",
        **mv,
        "annotations": ann,
        "reconstruction": None,
        "annotation_task_key": f"training__{task}__{method}",
    }


# --------------------------------------------------------------------------
# HTTP handler
# --------------------------------------------------------------------------

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "SafeManipViewer/1.0"

    def log_message(self, fmt, *args):
        pass  # keep console quiet

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, rel_path):
        path = (STATIC_DIR / rel_path).resolve()
        if STATIC_DIR.resolve() not in path.parents and path != STATIC_DIR.resolve():
            self.send_error(403)
            return
        if not path.is_file():
            self.send_error(404)
            return
        mime = MIME.get(path.suffix, "application/octet-stream")
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        # this is an actively-edited dev tool (no build/versioning step) --
        # without an explicit no-cache directive, browsers may serve a stale
        # cached copy of index.html/app.js/style.css on a plain reload
        # (confirmed as a real failure mode: a JS syntax error fixed on disk
        # still appeared "stuck loading" until this was added / a hard
        # refresh was forced), so always force revalidation.
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(data)

    def _send_video(self, task, episode):
        try:
            episode = int(episode)
        except (TypeError, ValueError):
            self.send_error(400, "bad episode")
            return
        rollout_dir = latest_rollout_dir(task)
        if rollout_dir is None:
            self.send_error(404, "task not found")
            return
        ep_meta = next((e for e in episodes_in(rollout_dir) if e["episode"] == episode), None)
        if ep_meta is None:
            self.send_error(404, "episode not found")
            return
        video_path = rollout_dir / ep_meta["video_name"]
        self._send_file_range(video_path)

    def _send_reconstructed_video(self, task, episode):
        try:
            episode = int(episode)
        except (TypeError, ValueError):
            self.send_error(400, "bad episode")
            return
        video_path = reconstruction_paths(task, episode)["video"]
        if not video_path.is_file():
            self.send_error(404, "no reconstruction for this episode")
            return
        self._send_file_range(video_path)

    def _send_training_video(self, task, episode):
        try:
            episode = int(episode)
        except (TypeError, ValueError):
            self.send_error(400, "bad episode")
            return
        video_path = training_episode_paths(task, episode)["video"]
        if not video_path.is_file():
            self.send_error(404, "no training-data reconstruction for this episode")
            return
        self._send_file_range(video_path)

    def _send_training_original_video(self, task, episode, camera):
        try:
            episode = int(episode)
        except (TypeError, ValueError):
            self.send_error(400, "bad episode")
            return
        video_path = training_original_video_path(task, episode, camera)
        if video_path is None or not video_path.is_file():
            self.send_error(404, "no original video for this task/episode/camera")
            return
        self._send_file_range(video_path)

    def _send_training_original_concat_video(self, task, episode):
        try:
            episode = int(episode)
        except (TypeError, ValueError):
            self.send_error(400, "bad episode")
            return
        meta_path = training_episode_paths(task, episode)["meta"]
        camera_names = DEFAULT_TRAINING_CAMERAS
        if meta_path.is_file():
            try:
                camera_names = json.loads(meta_path.read_text()).get("camera_names") or camera_names
            except Exception:
                pass
        try:
            video_path = ensure_original_concat(task, episode, camera_names)
        except Exception as e:
            self.send_error(500, f"failed to build concatenated original video: {e}")
            return
        if video_path is None or not video_path.is_file():
            self.send_error(404, "no original per-camera videos for this task/episode")
            return
        self._send_file_range(video_path)

    def _send_file_range(self, video_path):
        stat = video_path.stat()
        file_size = stat.st_size
        # NOT "immutable"/long max-age: confirmed a real bug from an earlier
        # version of this comment's wrong assumption -- reconstructed.mp4 /
        # original_concat.mp4 *do* get regenerated in place at the same URL
        # (e.g. re-running reconstruct_training_data.py with a different
        # --video_skip overwrites the existing file), so a browser that's
        # been told "immutable, cache for a year" will keep serving the old
        # video indefinitely and never even ask the server again, no matter
        # how many times the page is reloaded. `Cache-Control: no-cache`
        # instead means "always revalidate" -- combined with the ETag/304
        # handling below, a truly-unchanged file still avoids re-downloading
        # bytes (cheap 304 response), but a regenerated file (different
        # mtime/size -> different ETag) is correctly re-fetched immediately.
        etag = f'"{stat.st_mtime_ns:x}-{file_size:x}"'
        last_modified = email.utils.formatdate(stat.st_mtime, usegmt=True)

        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            return

        range_header = self.headers.get("Range")
        start, end = 0, file_size - 1
        status = 200
        if range_header:
            m = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2))
                status = 206

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("ETag", etag)
        self.send_header("Last-Modified", last_modified)
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

        with open(video_path, "rb") as f:
            f.seek(start)
            remaining = length
            chunk = 1024 * 256
            while remaining > 0:
                data = f.read(min(chunk, remaining))
                if not data:
                    break
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        if parsed.path == "/api/tasks":
            return self._send_json(api_tasks())

        if parsed.path == "/api/episodes":
            task = qs.get("task", [None])[0]
            if not task:
                return self._send_json({"error": "missing task"}, 400)
            result = api_episodes(task)
            status = result[1] if isinstance(result, tuple) else 200
            body = result[0] if isinstance(result, tuple) else result
            return self._send_json(body, status)

        if parsed.path == "/api/episode":
            task = qs.get("task", [None])[0]
            episode = qs.get("episode", [None])[0]
            if not task or episode is None:
                return self._send_json({"error": "missing task/episode"}, 400)
            result = api_episode(task, int(episode))
            status = result[1] if isinstance(result, tuple) else 200
            body = result[0] if isinstance(result, tuple) else result
            return self._send_json(body, status)

        if parsed.path == "/video":
            task = qs.get("task", [None])[0]
            episode = qs.get("episode", [None])[0]
            if not task or episode is None:
                return self.send_error(400, "missing task/episode")
            return self._send_video(task, episode)

        if parsed.path == "/reconstructed_video":
            task = qs.get("task", [None])[0]
            episode = qs.get("episode", [None])[0]
            if not task or episode is None:
                return self.send_error(400, "missing task/episode")
            return self._send_reconstructed_video(task, episode)

        if parsed.path == "/api/td_tasks":
            return self._send_json(api_training_tasks())

        if parsed.path == "/api/td_episodes":
            task = qs.get("task", [None])[0]
            if not task:
                return self._send_json({"error": "missing task"}, 400)
            return self._send_json(api_training_episodes(task))

        if parsed.path == "/api/training_monitor":
            task = qs.get("task", [None])[0]
            episode = qs.get("episode", [None])[0]
            method = qs.get("method", [DEFAULT_TRAINING_MONITOR_METHOD])[0]
            if not task or episode is None:
                return self._send_json({"error": "missing task/episode"}, 400)
            result = api_training_monitor(task, int(episode), method=method)
            status = result[1] if isinstance(result, tuple) else 200
            body = result[0] if isinstance(result, tuple) else result
            return self._send_json(body, status)

        if parsed.path == "/api/training_monitor_methods":
            return self._send_json({
                "default": DEFAULT_TRAINING_MONITOR_METHOD,
                "methods": {k: {"label": v["label"]} for k, v in TRAINING_MONITOR_METHODS.items()},
            })

        if parsed.path == "/td_video":
            task = qs.get("task", [None])[0]
            episode = qs.get("episode", [None])[0]
            if not task or episode is None:
                return self.send_error(400, "missing task/episode")
            return self._send_training_video(task, episode)

        if parsed.path == "/td_original_video":
            task = qs.get("task", [None])[0]
            episode = qs.get("episode", [None])[0]
            camera = qs.get("camera", [None])[0]
            if not task or episode is None or not camera:
                return self.send_error(400, "missing task/episode/camera")
            return self._send_training_original_video(task, episode, camera)

        if parsed.path == "/td_original_concat_video":
            task = qs.get("task", [None])[0]
            episode = qs.get("episode", [None])[0]
            if not task or episode is None:
                return self.send_error(400, "missing task/episode")
            return self._send_training_original_concat_video(task, episode)

        if parsed.path == "/":
            return self._send_static("index.html")

        # static files (app.js, style.css, ...)
        return self._send_static(parsed.path.lstrip("/"))

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/annotate":
            return self.send_error(404)
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            return self._send_json({"error": "bad json body"}, 400)
        task = body.get("task")
        episode = body.get("episode")
        if not task or episode is None:
            return self._send_json({"error": "missing task/episode"}, 400)
        data = save_annotations(task, int(episode), body)
        return self._send_json({"ok": True, "annotations": data})


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT, help="results 'target' directory to browse")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    ROOT = Path(args.root)
    if not ROOT.is_dir():
        raise SystemExit(f"root directory does not exist: {ROOT}")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"SafeManip viewer serving {ROOT}")
    print(f"  -> http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
