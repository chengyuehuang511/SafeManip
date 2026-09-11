#!/usr/bin/env python3
"""Runs the pristine eval/models/openpi/scripts/serve_policy.py, but first
sets robocasa.macros.DATASET_BASE_PATH from the outside -- without writing
anything into the pristine eval/simulators/robocasa submodule.

Why this is needed
-------------------
`openpi/training/config.py` does `from robocasa.macros import
DATASET_BASE_PATH` at module import time, to locate the RoboCasa training
dataset's normalization stats when loading a checkpoint. robocasa's only
built-in way to set this is a `macros_private.py` file inside the robocasa
package itself (created via `python -m robocasa.scripts.setup_macros`) --
but that means writing a file into eval/simulators/robocasa, which we're
keeping untouched (not even gitignored local config), so this script
monkeypatches `robocasa.macros.DATASET_BASE_PATH` from outside instead.

Why `runpy` instead of just importing and calling serve_policy's `main`
-------------------------------------------------------------------------
serve_policy.py's own top-level `from openpi.training import config as
_config` (which transitively does `from robocasa.macros import
DATASET_BASE_PATH`) runs as soon as that module is first imported/executed.
A plain `import` of serve_policy.py after already importing
`openpi.training.config` elsewhere in this process wouldn't help either --
the `from robocasa.macros import DATASET_BASE_PATH` binding is resolved (and
cached in sys.modules) the first time `openpi.training.config` is imported,
not re-read on every subsequent reference. So the patch below must happen
strictly before openpi.training.config is imported for the first time in
this process at all. `runpy.run_path(..., run_name="__main__")` re-executes
serve_policy.py's top-level code (imports included) fresh, in this same
process, after our patch -- exactly like `python scripts/serve_policy.py`
would, just with the patch already in place first.

Usage (drop-in replacement for `python scripts/serve_policy.py ...`):
    python eval/single_task/serve_policy_wrapper.py \\
        --port=8000 policy:checkpoint --policy.config=... --policy.dir=...

The dataset root defaults to $ROBOCASA_DATASET_BASE_PATH if set, else
~/flash/datasets/robocasa (this machine's actual RoboCasa v1.0 dataset
location, confirmed to contain v1.0/target/...).
"""
import os
import runpy
import sys
from pathlib import Path

DEFAULT_DATASET_BASE_PATH = os.path.expanduser("~/flash/datasets/robocasa")


def main() -> None:
    dataset_base_path = os.environ.get("ROBOCASA_DATASET_BASE_PATH", DEFAULT_DATASET_BASE_PATH)

    import robocasa.macros as _macros

    _macros.DATASET_BASE_PATH = dataset_base_path

    openpi_root = Path(
        os.environ.get(
            "OPENPI_ROOT", Path(__file__).resolve().parent.parent / "models" / "openpi"
        )
    )
    serve_policy_path = openpi_root / "scripts" / "serve_policy.py"
    if not serve_policy_path.is_file():
        raise FileNotFoundError(f"serve_policy.py not found under OPENPI_ROOT={openpi_root}")

    sys.argv = [str(serve_policy_path)] + sys.argv[1:]
    runpy.run_path(str(serve_policy_path), run_name="__main__")


if __name__ == "__main__":
    main()
