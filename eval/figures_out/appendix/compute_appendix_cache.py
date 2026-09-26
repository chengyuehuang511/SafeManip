#!/usr/bin/env python3
"""One-shot data pass for ALL appendix figures (PLAN.md, execution step 1).

Writes two granular caches next to this file; every appendix render script
reads these and never touches the slow pipeline again:

  appendix_prop_cells.csv   per (suite, model, category, task, property,
                            success): V (violations), A (triggered events),
                            D (sum of violation_duration/num_frames),
                            W (violated instances), n (applicable instances)
      -> C1 crude-vs-conditional, D2 counts heatmap, E2 raw-metric heatmaps.

  appendix_episodes.csv     per episode: suite, model, task, category, tier,
                            n_subtasks, task_success, V, A, D, act_unique,
                            n_applicable
      -> A1 outcome figure, B1 model-level validation (coverage =
         act_unique/n_applicable), B2 complexity, E1 raw RQ1 scatter
         (violated episode = V > 0), E3 raw RQ3 (exposure = mean D).

Task metadata follows render_rq3_conditional.py exactly: RoboCasa tiers and
categories from load_suite's meta maps, n_subtasks from
SafeManip/analysis/taskDiff.csv; LIBERO tier/category/primitives from
libero_task_skill_sequences_categorized.csv re-binned by the RoboCasa
TIER_RULE.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIGOUT = os.path.dirname(HERE)
PC = ("/path/to/SafeManip/eval/saved_eval_rollouts"
      "/monitor_files/0920/plot_combined")
TASKDIFF = ("/path/to/SafeManip/SafeManip"
            "/analysis/taskDiff.csv")
LB_SKILLS = os.path.join(FIGOUT, "libero_task_skill_sequences_categorized.csv")
SUITES = ("RoboCasa", "LIBERO")

GT_CATEGORY_KEY = {
    "Atomic/Fixture": "AtomicFixture",
    "Beverage Preparation/Serving": "BeveragePreparationServing",
    "Cooking/Ingredient Preparation": "CookingIngredientPreparation",
    "Plating/Serving/Portioning": "PlatingServingPortioning",
    "Storage/Organization": "StorageOrganization",
}


def tier_rule(n):
    if n is None or not np.isfinite(n):
        return None
    n = int(n)
    return ("Atomic" if n <= 1 else "Short" if n == 2 else
            "Medium" if n <= 4 else "Long")


def main():
    sys.path.insert(0, PC)
    from activation_data import episode_table, load_suite  # noqa: E402

    lb = pd.read_csv(LB_SKILLS)
    lb_key = lb["task"].astype(str)
    lb_tier = dict(zip(lb_key, lb["total_primitives"].map(tier_rule)))
    lb_prim = dict(zip(lb_key, lb["total_primitives"]))
    lb_cat = dict(zip(lb_key, lb["category"].map(GT_CATEGORY_KEY)))
    td = pd.read_csv(TASKDIFF).set_index("taskID")["numberSubtask"]

    props, eps = [], []
    for suite in SUITES:
        df, meta = load_suite(suite)
        df = df[df["applicable"] == 1].copy()
        df["duration_frac"] = (df["violation_duration"]
                               / df["num_frames"].replace(0, np.nan))
        g = (df.groupby(["model", "category", "task", "property_name",
                         "task_success"], observed=True)
             .agg(V=("violations", "sum"), A=("activations", "sum"),
                  D=("duration_frac", "sum"), W=("violated", "sum"),
                  n=("episode", "size"))
             .reset_index().rename(columns={"task_success": "success"}))
        g.insert(0, "suite", suite)
        props.append(g)

        ep = episode_table(df)
        d = (df.groupby(["model", "task", "episode"], observed=True)
             ["duration_frac"].sum().rename("D").reset_index())
        ep = ep.merge(d, on=["model", "task", "episode"], how="left")
        ep.insert(0, "suite", suite)
        if suite == "RoboCasa":
            ep["category"] = ep["task"].map(meta["suite_map"])
            ep["tier"] = ep["task"].map(meta["horizon_map"])
            ep["n_subtasks"] = ep["task"].map(td)
        else:
            spaced = ep["task"].str.replace("_", " ", regex=False)
            ep["category"] = spaced.map(lb_cat)
            ep["tier"] = spaced.map(lb_tier)
            ep["n_subtasks"] = spaced.map(lb_prim)
        n_miss = int(ep["n_subtasks"].isna().sum())
        if n_miss:
            miss = sorted(set(ep.loc[ep.n_subtasks.isna(), "task"]))[:10]
            print(f"[warn] {suite}: {n_miss} episodes without n_subtasks, "
                  f"e.g. {miss}")
        eps.append(ep)

    pc = pd.concat(props, ignore_index=True)
    pc.to_csv(os.path.join(HERE, "appendix_prop_cells.csv"), index=False)
    print(f"[csv] appendix_prop_cells.csv ({len(pc):,} cells)")

    ec = pd.concat(eps, ignore_index=True).rename(
        columns={"violations": "V", "act_total": "A"})[
        ["suite", "model", "task", "episode", "category", "tier",
         "n_subtasks", "task_success", "V", "A", "D", "act_unique",
         "n_applicable"]]
    ec.to_csv(os.path.join(HERE, "appendix_episodes.csv"), index=False)
    print(f"[csv] appendix_episodes.csv ({len(ec):,} episodes)")


if __name__ == "__main__":
    main()
