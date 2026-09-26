#!/usr/bin/bash
# Shared by every LIBERO run script (eval_rldx1_libero_single_task.sh,
# eval_openpi_libero_suite.sh, eval_groot_n1d6_libero_single_task.sh).
# `source` this AFTER LIBERO_ROOT and PROJECT_ROOT are set.
#
# Why this exists
# ----------------
# LIBERO's own `libero.libero.__init__` resolves `get_libero_path("bddl_files"
# /"init_states"/"datasets"/"assets")` from a MACHINE-WIDE config file
# (~/.libero/config.yaml by default, or $LIBERO_CONFIG_PATH/config.yaml if
# set) -- completely independent of PYTHONPATH or which `libero` package is
# actually on it. Confirmed by reading eval/simulators/libero/libero/libero/
# __init__.py directly: `get_libero_path` just does
# `yaml.load(open(config_file))`, ignoring the caller's own import path
# entirely, and never does any relative-path or env-var expansion itself.
#
# This machine already had a STALE ~/.libero/config.yaml from an unrelated,
# earlier project (pointing at /path/to/LIBERO/...,
# a symlink to .../Inspire/LIBERO/, a modified checkout) -- confirmed by a
# real diff across all 4 suites: every one of libero_object's 10 bddl files
# differs there (a genuinely different scene -- table vs. floor placement,
# same task names), while libero_spatial/libero_goal/libero_10 are all
# byte-identical. Every LIBERO run on this machine was silently loading
# bddl/init_states/assets from that unrelated project instead of whichever
# LIBERO submodule was actually intended.
#
# Fix, in two parts:
#   1. eval/.libero_configs/{shared,openpi}/config.yaml -- committed,
#      human-reviewable templates with paths RELATIVE TO $PROJECT_ROOT
#      (e.g. `eval/simulators/libero/libero/libero/bddl_files`), one per
#      actual submodule pin in use across this repo (there are only two:
#      the shared eval/simulators/libero pin used by RLDX-1/GR00T-N1.6, and
#      openpi's own dedicated eval/simulators/libero_openpi pin). Portable
#      by construction -- nothing machine-specific is stored here, so this
#      is safe to commit and never needs to be ignored or regenerated.
#   2. This script reads the matching template and resolves each relative
#      path against $PROJECT_ROOT into a REAL config.yaml that LIBERO's own
#      get_libero_path() can actually use (it has no relative-path
#      resolution of its own) -- written to per-job scratch space
#      (/tmp/${USER}/libero_config-*), the same convention already used
#      elsewhere in these scripts for TRITON_CACHE_DIR, not under
#      eval/.libero_configs/ itself. This keeps the committed templates
#      permanently clean while still producing a correct absolute path at
#      run time on whatever machine/checkout this actually runs on.
#
# Concurrency: multiple jobs sharing the same LIBERO_ROOT (e.g. many RLDX-1
# array tasks, or a simultaneous GR00T-N1.6 run) could in principle collide
# on the same scratch path -- avoided by keying the scratch directory on
# SLURM_JOB_ID/SLURM_ARRAY_TASK_ID (falling back to $$ for non-SLURM runs),
# so concurrent jobs never share a config.yaml file at all, not even
# momentarily.

set -euo pipefail

if [[ -z "${LIBERO_ROOT:-}" ]]; then
  echo "_setup_libero_config.sh: LIBERO_ROOT must be set before sourcing this." >&2
  exit 1
fi
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  echo "_setup_libero_config.sh: PROJECT_ROOT must be set before sourcing this." >&2
  exit 1
fi

case "${LIBERO_ROOT}" in
  */libero_openpi)
    config_name="openpi"
    ;;
  */libero)
    config_name="shared"
    ;;
  *)
    echo "_setup_libero_config.sh: unrecognized LIBERO_ROOT=${LIBERO_ROOT} -- expected it to end in 'libero' or 'libero_openpi'. Add a new template under eval/.libero_configs/ and a case here if this is a genuinely new LIBERO pin." >&2
    exit 1
    ;;
esac

_template="${PROJECT_ROOT}/eval/.libero_configs/${config_name}/config.yaml"
if [[ ! -f "${_template}" ]]; then
  echo "_setup_libero_config.sh: template ${_template} not found." >&2
  exit 1
fi

_job_key="${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+_${SLURM_ARRAY_TASK_ID}}_$$"
export LIBERO_CONFIG_PATH="/tmp/${USER}/libero_config-${_job_key}"
mkdir -p "${LIBERO_CONFIG_PATH}"

# Resolve each `key: relative/path` line against PROJECT_ROOT. Values here
# are always plain relative paths (no leading slash, no special yaml
# syntax) by construction of the committed templates, so a literal
# string-prefix join is sufficient -- not general yaml/path parsing.
awk -v root="${PROJECT_ROOT}" '{print $1" "root"/"$2}' "${_template}" \
  > "${LIBERO_CONFIG_PATH}/config.yaml"

echo "LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH} (resolved from ${_template})"
