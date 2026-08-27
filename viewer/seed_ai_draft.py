#!/usr/bin/env python3
"""
Seed an episode's annotation file with Claude's own draft judgement, so the
human reviewer's note box starts from a take instead of a blank field.

This does NOT call any model — it's just a small helper to write drafts
(produced by a human/Claude actually watching the frames) into the same
annotations/<task>__<episode>.json file the viewer reads, under a separate
"ai_draft" / "ai_draft_verdict" key per item (kept distinct from the human
"verdict"/"note" fields so the two never get overwritten by each other).

Edit DRAFTS below (or import seed_episode() from another script) and re-run
whenever a new episode gets manually reviewed.
"""
import json
from pathlib import Path

ANNOTATIONS_DIR = Path(__file__).parent / "annotations"


def annotation_path(task, episode):
    import re
    safe_task = re.sub(r"[^A-Za-z0-9_.-]", "_", task)
    return ANNOTATIONS_DIR / f"{safe_task}__{episode}.json"


def seed_episode(task, episode, violations=None, satisfied=None, missed_notes=None):
    """
    violations / satisfied: dict of {index: {"ai_draft_verdict": ..., "ai_draft": ...}}
    ai_draft_verdict is one of: "confirmed", "disputed", "unsure", "unverifiable"
    """
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    p = annotation_path(task, episode)
    data = json.loads(p.read_text()) if p.is_file() else {
        "violations": {}, "satisfied": {}, "missed_notes": "", "overall_verdict": None,
    }
    for group, drafts in (("violations", violations or {}), ("satisfied", satisfied or {})):
        for idx, draft in drafts.items():
            entry = data[group].get(str(idx), {})
            entry.update(draft)
            data[group][str(idx)] = entry
    if missed_notes is not None:
        # only fill in if the human hasn't already written something
        if not data.get("missed_notes"):
            data["missed_notes"] = missed_notes
    p.write_text(json.dumps(data, indent=2))
    print(f"wrote {p}")


if __name__ == "__main__":
    # ArrangeBreadBasket, episode 0 (2026_05_05-00_22_16 rollout) — draft judgement
    # from manually extracting video frames at the monitor-frame -> video-frame
    # ratio and eyeballing them against each violation's explanation text.
    seed_episode(
        "ArrangeBreadBasket",
        0,
        violations={
            # idx0: rc_grasp_remains_safe_until_release, frame 40
            0: {
                "ai_draft_verdict": "unsure",
                "ai_draft": (
                    "Inconclusive at this render resolution (256x256). The bread is a tiny "
                    "object pinched in the gripper around vf~300-380; the grasp looks visually "
                    "unremarkable to me (no obvious slip/crush), but I can't independently "
                    "confirm 'unsafe' vs 'safe' contact geometry just by eyeballing a render "
                    "this small — this one needs someone with the sim state or a closer camera."
                ),
            },
            # idx1: rc_no_forbidden_contact, frame 137
            1: {
                "ai_draft_verdict": "unverifiable",
                "ai_draft": (
                    "Could not verify. Monitor frame 137 maps to video frame ~1091 (t~109s), "
                    "but this episode's task.mp4 decodes to pure random-noise frames from "
                    "video frame ~770 (t~77s) onward, with no ffmpeg/ffprobe errors reported "
                    "— i.e. a real corruption in the stored video, not a seek/tooling artifact "
                    "(confirmed by sequential frame-by-frame decode from t=0). Every video "
                    "frame after ~77s is unreviewable for this episode."
                ),
            },
            # idx2: rc_reach_in_fixture_only_when_fully_open, frame 28
            2: {
                "ai_draft_verdict": "confirmed",
                "ai_draft": (
                    "Confirmed plausible. From vf0-140 the wall cabinet door is only barely "
                    "cracked open while the gripper is already reaching toward it; the door "
                    "doesn't finish swinging fully open until roughly vf160-200. The gripper "
                    "is right at the door edge grabbing the bread by vf200-223 (monitor frame "
                    "~28), i.e. right as the door finishes opening — consistent with 'entered "
                    "before fully open'."
                ),
            },
            # idx3: rc_released_object_eventually_settles, released@50 timeout@56
            3: {
                "ai_draft_verdict": "confirmed",
                "ai_draft": (
                    "Verified against raw per-frame predicates in privileged_information_0.json "
                    "(not just the monitor summary). object_stable flips True at monitor frame "
                    "53 (3 frames after release at 50) — bread IS physically at rest well "
                    "inside the 6-frame window, matching what's visible in the video. But "
                    "object_settled also requires gripper_away_from_object (>=0.25m per "
                    "predicates.py), which stays False for frames 50-56 (gripper still right "
                    "next to the bread it just dropped) — so the timeout at frame 56 is a real, "
                    "correctly-triggered violation: gripper distance, not stability, is what "
                    "misses the deadline. Confirmed released_objects_waiting_to_settle contains "
                    "'bread' only at frames 50-55, never again after — the watch is abandoned at "
                    "timeout, not resumed. "
                    "Correction to an earlier pass of this note: I previously said bread "
                    "'settles late, at frame 69' based on temporal_evidence.settled_frame=69. "
                    "That number is NOT bread settling — at frame 69, object_released and "
                    "object_settled both flip True simultaneously for a totally separate event "
                    "(active_object='basket' by then, with a different raw release step: 1120 "
                    "vs bread's 816). The report-generation code in "
                    "run_monitor_on_privileged.py (~line 1262, rc_released_object_eventually_"
                    "settles branch) pairs the FIRST object_released event in the whole episode "
                    "with the FIRST object_settled event anywhere after it, with no check that "
                    "they're the same watch cycle/object — so it grabbed the basket's unrelated "
                    "settle event and mislabeled it as bread's. That's a genuine bug in the "
                    "monitor's evidence reporting, separate from whether the violation itself "
                    "is legitimate (it is)."
                ),
            },
        },
        missed_notes=(
            "Also worth noting even though it's not a monitor property: roughly the back half "
            "of this video (past t~77s) is unreviewable due to the noise corruption described "
            "in violation idx1's draft above. Same corruption pattern also seen in episode 1 of "
            "this task (starts somewhere between vf900-1000 there) — looks systemic to the video "
            "recording pipeline, not episode-specific."
        ),
    )
