#!/usr/bin/env python3
"""Data pass for appendix item G (app:training_variants): the three GR00T
Foundation-Model-Learning adaptation recipes on the RoboCasa365 TARGET split.

Separate from compute_appendix_cache.py on purpose. That cache is the
cross-policy corpus: five policies, one adaptation corpus, and the *pretrain*
evaluation split, loaded through plot_combined/load_both.load_robocasa(), whose
model roster (plot_robocasa/style.MODEL_ORDER) deliberately excludes the
variants. These runs are a different estimand -- one architecture, three
adaptation recipes, all rolled out in the ten held-out target kitchens -- and
mixing them into the same CSV would let a reader average across two evaluation
distributions.

Scope note: the Multitask-Learning checkpoint is deliberately NOT here. Its
target-split rollouts exist but were never monitored (the monitor replays every
episode's physics through MuJoCo at ~38 min/episode, and a 2,500-episode sweep
was stopped partway by request). ~1,381 partial episodes are on disk under
eval_monitor_staging/target_groot_multitask_learning; they are intentionally
unreachable from here because `VARIANTS` does not list the variant, so a rerun
cannot silently mix a partial variant into the figure. To add it later: finish
run_extract_privileged_per_episode_slim.sbatch for that staging tree (it skips
episodes that already have a monitor JSON), submit shards 60-79 of
extract_variants.sbatch, then re-add the variant to VARIANTS below.

Sources, all produced by the item-G sweep:
  plot_robocasa/processedData/variant_shards/metrics_<variant>_s*.csv
      per (variant, task, episode, property) violation windows/duration
      (extract_metrics.py, the same unified window convention as the main
      corpus).
  plot_combined/processedData/activations_rc/activations_<variant>_s*.csv
      plus .../activations_rc/_per_episode/<variant>/act_*.csv
      antecedent-atom rising edges = the per-triggered-event DENOMINATOR. The
      shard form comes from extract_activations_variants.sbatch (variants whose
      privileged dumps still existed); the per-episode form from
      run_extract_privileged_per_episode_slim.sbatch, which extracts
      activations inside the same array task that made the dump and then
      deletes it.

Writes next to this file:
  appendix_variant_prop_cells.csv  per (variant, category, task, property,
                                   success): V, A, D, W, n
  appendix_variant_episodes.csv    per (variant, task, episode): category,
                                   tier, n_subtasks, task_success, V, A, D,
                                   act_unique, n_applicable

Column semantics match compute_appendix_cache.py exactly so the render scripts
can share helpers: V = violation windows, A = triggered events (activations),
D = sum of violation_duration/num_frames, W = violated instances, n =
applicable instances.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MON = ("/path/to/SafeManip/eval/saved_eval_rollouts"
       "/monitor_files/0920")
PC = os.path.join(MON, "plot_combined")
PR = os.path.join(MON, "plot_robocasa")
VARIANT_SHARDS = os.path.join(PR, "processedData", "variant_shards")
ACT_DIR = os.path.join(PC, "processedData", "activations_rc")
TASKDIFF = ("/path/to/SafeManip/SafeManip"
            "/analysis/taskDiff.csv")
ROLLOUTS = ("/path/to/SafeManip/eval"
            "/saved_eval_rollouts/target/groot")
# variant -> the eval harness's own rollout dir, which carries a per-task
# stats.json. That file is the authoritative success label; see harness_success.
ROLLOUT_DIR = {
    "target_groot_multitask_learning": "multitask_learning",
    "target_groot_target_only": "target_only",
    "target_groot_pretraining": "pretraining",
    "target_groot_target_posttraining": "target_posttraining",
}

# Display order is the adaptation-data ladder, not alphabetical: pretraining
# data only, target data only, then both. (target_groot_multitask_learning is
# excluded on purpose -- see the scope note in the module docstring.)
VARIANTS = [
    "target_groot_pretraining",
    "target_groot_target_only",
    "target_groot_target_posttraining",
]
VARIANT_DISPLAY = {
    "target_groot_multitask_learning": "Multitask",
    "target_groot_target_only": "GR00T-to",
    "target_groot_pretraining": "GR00T-pt",
    "target_groot_target_posttraining": "GR00T-tpt",
}


def _first_line(p):
    with open(p) as fh:
        return fh.readline().rstrip("\n")


def read_csvs(paths, what):
    """Concatenate shard CSVs, tolerating the two states a sweep leaves behind.

    Empty: a shard CSV is created by its array task before the first row is
    written, and preemptible-queue tasks can be killed, so a sweep that is still running
    always has a few zero-byte files in it. Those are "not done yet", not
    corrupt.

    Headerless: the extractors append, and emit the header only when the output
    file does not already exist. A task that is cancelled or preempted after
    creating its (zero-byte) file and then re-runs therefore appends data rows
    to a file that never got a header, and pandas silently promotes the first
    data row to the column names -- which shows up downstream as one NaN-keyed
    property and an IntCastingNaNError, not as a parse error. So the header is
    identified here as the modal first line (every well-formed shard writes the
    identical header; a stray data row is unique to its file) and supplied
    explicitly to the files that lack it.
    """
    nonempty = [p for p in paths if os.path.getsize(p) > 0]
    skipped = len(paths) - len(nonempty)
    if skipped:
        print(f"[warn] {skipped}/{len(paths)} {what} CSVs are still empty "
              f"(shard in flight) -- excluded", flush=True)
    if not nonempty:
        raise SystemExit(f"every {what} CSV is empty")

    firsts = [_first_line(p) for p in nonempty]
    header = pd.Series(firsts).mode().iat[0]
    cols = header.split(",")
    headerless = [p for p, f in zip(nonempty, firsts) if f != header]
    if headerless:
        print(f"[warn] {len(headerless)}/{len(nonempty)} {what} CSVs have no "
              f"header row (requeued shard appended to a pre-created file) -- "
              f"reading them with the header taken from their siblings",
              flush=True)

    frames = []
    for p, f in zip(nonempty, firsts):
        kw = dict(dtype={"episode": str})
        if f != header:
            kw.update(header=None, names=cols)
        d = pd.read_csv(p, **kw)
        # A headerless file is the tail of an interrupted run, so it can also
        # carry a stray repeated header line as data.
        if f != header and len(cols):
            d = d[d[cols[0]] != cols[0]]
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def tier_rule(n):
    if n is None or not np.isfinite(n):
        return None
    n = int(n)
    return ("Atomic" if n <= 1 else "Short" if n == 2 else
            "Medium" if n <= 4 else "Long")


def harness_success(variants):
    """(variant, task) -> the eval harness's own success rate, from the
    per-task stats.json written next to the rollouts.

    Why not the replay flag or the length rule this corpus's loader would use:
    neither works here. `corrected_task_success` falls back to the monitor's
    replay flag whenever a harness never early-stops, and none of these four
    variants early-stops on the target split, so the whole corpus lands on the
    flag. That flag is evaluated one frame before the true terminal state --
    `extract_privileged_from_dataset.py` cannot reconstruct the dropped last
    state because the eval-staging lerobot datasets carry no .parquet actions --
    which undercounts success by ~5-6 points per variant here. stats.json is
    the harness's own label and has no such gap. It is per (variant, task)
    rather than per episode, which is enough: every task contributes exactly
    50 episodes, so the task mean IS the episode rate.
    """
    import json
    rows = []
    for v in variants:
        d = os.path.join(ROLLOUTS, ROLLOUT_DIR[v])
        for st in sorted(glob.glob(os.path.join(d, "*", "stats.json"))):
            j = json.load(open(st))
            rows.append(dict(variant=v, task=j["task"], split=j.get("split"),
                             n_episodes=j.get("num_episodes"),
                             success_harness=j["success_rate"]))
    out = pd.DataFrame(rows)
    bad = out[out["split"] != "target"]
    if len(bad):
        raise SystemExit(
            "stats.json reports a non-target split for "
            f"{bad['variant'].unique().tolist()} -- wrong rollout tree")
    return out


def load_variant_metrics(variants):
    """Concatenated metrics_long for `variants`, with the same corrections the
    main RoboCasa loader applies: canonical categories, the unified violation
    window convention, applicability, and length-derived task success."""
    sys.path.insert(0, PC)
    from load_both import import_isolated  # noqa: E402
    rc = import_isolated(PR, ["load_data"])["load_data"]

    frames = []
    for v in variants:
        fs = sorted(glob.glob(os.path.join(VARIANT_SHARDS, f"metrics_{v}_s*.csv")))
        if not fs:
            raise SystemExit(f"no metric shards for {v} in {VARIANT_SHARDS}")
        d = read_csvs(fs, f"{v} metric")
        # Shards are disjoint by construction, but a requeued preemptible task can
        # re-append rows it had already written before preemption.
        d = d.drop_duplicates(["model", "task", "episode", "property_name"])
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    # NOT pd.Categorical(MODEL_ORDER) as load_metrics_long does -- MODEL_ORDER
    # is the five-policy roster and would silently NaN every variant row.
    df["model"] = df["model"].astype(str)
    df = rc.canonical_categories(df)
    df = rc.apply_violation_window_convention(df, label="RoboCasa variants")
    _pid_to_cat, applicable_wide = rc.load_applicable_property()
    df = rc.add_applicability(df, applicable_wide)
    # Generic in the model column: per model, "did this harness ever early-stop
    # on this corpus" decides whether episode length or the replay flag carries
    # the success label. The `cap` it compares against is a per-task max over
    # the frame passed in, which is why the four variants are corrected together
    # -- they share the target split's horizon caps.
    df = rc.corrected_task_success(df)
    return df, rc


def load_variant_activations(variants):
    """model/task/episode/property_name -> activations, for the variants.

    Two on-disk forms are merged: per-shard CSVs and the one-episode CSVs the
    slim sweep writes. Both have the same header, so they concatenate directly.
    """
    sys.path.insert(0, PC)
    from extract_activations_rc import PROPERTY_ATOM  # noqa: E402

    fs = []
    for v in variants:
        fs += sorted(glob.glob(os.path.join(ACT_DIR, f"activations_{v}_s*.csv")))
        fs += sorted(glob.glob(os.path.join(ACT_DIR, "_per_episode", v,
                                            "act_*.csv")))
    if not fs:
        raise SystemExit(f"no activation CSVs for the variants under {ACT_DIR}")
    act = read_csvs(fs, "activation")
    act = act[act["model"].isin(variants)]
    act = act.drop_duplicates(["model", "task", "episode"])

    out = []
    for p, atom in PROPERTY_ATOM.items():
        col = f"{atom}_onsets"
        if col in act.columns:
            out.append(act[["model", "task", "episode"]].assign(
                property_name=p, activations=act[col].values))
    return pd.concat(out, ignore_index=True)


def main():
    present = [v for v in VARIANTS
               if glob.glob(os.path.join(VARIANT_SHARDS, f"metrics_{v}_s*.csv"))]
    missing = [v for v in VARIANTS if v not in present]
    if missing:
        print(f"[warn] no metrics yet for: {', '.join(missing)} -- caching the "
              f"{len(present)} variant(s) that are ready", flush=True)

    df, _rc = load_variant_metrics(present)
    df = df[df["applicable"] == 1].copy()
    df["violations"] = df["violation_count"].astype(int)
    df["duration_frac"] = (df["violation_duration"]
                           / df["num_frames"].replace(0, np.nan))

    act = load_variant_activations(present)
    n = len(df)
    df = df.merge(act, on=["model", "task", "episode", "property_name"],
                  how="inner")
    if len(df) != n:
        print(f"[warn] {n - len(df)} of {n} applicable instances lost in the "
              f"activation join -- extraction incomplete?", flush=True)
    for v in present:
        sub = df[df["model"] == v]
        print(f"[{v}] {sub.groupby(['task', 'episode']).ngroups} episodes, "
              f"{len(sub)} applicable instances", flush=True)

    td = pd.read_csv(TASKDIFF).set_index("taskID")["numberSubtask"]
    sys.path.insert(0, PC)
    from load_both import import_isolated  # noqa: E402
    rc = import_isolated(PR, ["load_data"])["load_data"]
    suite_map = rc.load_suites()

    cells = (df.groupby(["model", "category", "task", "property_name",
                         "task_success"], observed=True)
             .agg(V=("violations", "sum"), A=("activations", "sum"),
                  D=("duration_frac", "sum"), W=("violated", "sum"),
                  n=("episode", "size"))
             # NOT named `success` like compute_appendix_cache.py's cells: on
             # this corpus the per-episode flag is the replay flag, which is
             # biased low (harness_success). Success-conditioned splits of the
             # variant cells should use it only with that in mind.
             .reset_index().rename(columns={"model": "variant",
                                           "task_success": "success_replay"}))
    cells["variant_display"] = cells["variant"].map(VARIANT_DISPLAY)
    cells.to_csv(os.path.join(HERE, "appendix_variant_prop_cells.csv"),
                 index=False)

    ep = (df.groupby(["model", "task", "episode"], observed=True)
          .agg(task_success=("task_success", "max"),
               V=("violations", "sum"),
               A=("activations", "sum"),
               D=("duration_frac", "sum"),
               act_unique=("activations", lambda s: int((s > 0).sum())),
               n_applicable=("property_name", "nunique"))
          .reset_index().rename(columns={"model": "variant"}))
    ep = ep.rename(columns={"task_success": "task_success_replay"})
    ep["variant_display"] = ep["variant"].map(VARIANT_DISPLAY)
    ep["category"] = ep["task"].map(suite_map)
    ep["n_subtasks"] = ep["task"].map(td)
    ep["tier"] = ep["n_subtasks"].map(tier_rule)
    # Success comes from the harness, per (variant, task); the replay flag is
    # kept alongside it so the gap stays auditable rather than invisible.
    hs = harness_success(present)
    hs.to_csv(os.path.join(HERE, "appendix_variant_task_success.csv"),
              index=False)
    ep = ep.merge(hs[["variant", "task", "success_harness"]],
                  on=["variant", "task"], how="left")
    if ep["success_harness"].isna().any():
        miss = ep.loc[ep["success_harness"].isna(), "task"].unique()[:5]
        raise SystemExit(f"no stats.json success for e.g. {miss.tolist()}")
    ep.to_csv(os.path.join(HERE, "appendix_variant_episodes.csv"), index=False)
    gap = (ep.groupby("variant_display", observed=True)
           .agg(replay=("task_success_replay", "mean"),
                harness=("success_harness", "mean")))
    gap["gap_pts"] = 100 * (gap["harness"] - gap["replay"])
    print("[success] replay flag vs harness stats.json (see harness_success):")
    print(gap.round(4).to_string())

    print(f"[done] {len(cells)} cells, {len(ep)} episodes over "
          f"{ep['variant'].nunique()} variant(s)")
    print(ep.groupby("variant_display", observed=True)
          .agg(episodes=("episode", "size"),
               success=("success_harness", "mean"),
               V=("V", "sum"), A=("A", "sum")).round(4).to_string())


if __name__ == "__main__":
    main()
