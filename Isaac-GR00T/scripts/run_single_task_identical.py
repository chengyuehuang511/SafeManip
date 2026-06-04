#!/usr/bin/env python3
"""Fixed-scene variant of run_single_task.py.

Behaves exactly like ``scripts/run_single_task.py`` and takes the SAME command
line arguments, with one difference: every episode reset reuses the SAME seed,
so all episodes replay the IDENTICAL RoboCasa scene and initial state (same
layout, style, and object placement). Only the policy's own stochasticity varies
between episodes.

This file is fully self-contained and edits nothing shared. It works by
monkey-patching the env factory ``gr00t.eval.simulation._create_single_env`` so
each created env is wrapped such that ``reset()`` always forces the fixed seed.
RoboCasa's gym wrapper reseeds ``env.rng`` from that seed (see
``robocasa/robocasa/wrappers/gym_wrapper.py``), and that rng fully determines the
sampled scene. It then delegates to the unmodified run_single_task.py CLI.

Run it just like the original, e.g.:
    python scripts/run_single_task_identical.py \
        --model_path <ckpt> --task <Task> --split target \
        --seed 42 --n_episodes 50 --save_privileged_info
"""

import os
import runpy

import gymnasium as gym

import gr00t.eval.simulation as simulation


class FixedSceneResetWrapper(gym.Wrapper):
    """Force every reset to reuse one seed -> identical scene + initial state.

    Auto-resets between episodes call ``reset()`` with ``seed=None``; we ignore
    that and always pass our fixed seed down, which reseeds the RoboCasa rng to
    the same value each episode.
    """

    def __init__(self, env, seed):
        super().__init__(env)
        self._fixed_seed = int(seed)

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=self._fixed_seed, options=options)


_original_create_single_env = simulation._create_single_env


def _create_single_env_fixed(config, idx):
    # Reuse the original wrapper stack, then force a fixed reset seed on top.
    env = _original_create_single_env(config, idx)
    fixed_seed = (config.seed + idx) if config.seed is not None else 0
    return FixedSceneResetWrapper(env, fixed_seed)


simulation._create_single_env = _create_single_env_fixed


if __name__ == "__main__":
    print(
        "[fixed-scene] every episode will replay the IDENTICAL scene + initial "
        "state (seed-locked). Only policy stochasticity varies."
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    runpy.run_path(
        os.path.join(script_dir, "run_single_task.py"),
        run_name="__main__",
    )
