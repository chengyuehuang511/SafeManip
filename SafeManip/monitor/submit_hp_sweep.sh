#!/usr/bin/bash
# Submit one SLURM array job with one array index per hyperparameter sweep cell.
#
# The cell list comes from hp_sweep_grid.cells() rather than being written here,
# so the grid has exactly one definition; the generated CELLS_FILE is what the
# array indexes into. It is written under monitor/hp_sweep (NFS-shared), never
# /tmp: compute nodes get a private /tmp and would not see it.
#
# The default cell is submitted as array index 0 and is the reference every other
# cell is scored against, so if the array is cancelled partway there is still
# something to score against.
#
# Usage:
#   bash submit_hp_sweep.sh                 # every cell not already complete
#   bash submit_hp_sweep.sh --force         # every cell, ignoring existing output
set -uo pipefail

SCRIPT_DIR="/nethome/chuang475/testnvme/projects/SafeManip/SafeManip/monitor"
SWEEP_DIR="${SCRIPT_DIR}/hp_sweep"
CELLS_FILE="${SWEEP_DIR}/cells.txt"
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

mkdir -p "${SWEEP_DIR}" "${SCRIPT_DIR}/logs"

if [[ ! -f "${SWEEP_DIR}/subset.json" ]]; then
  echo "building episode subset"
  python3 "${SCRIPT_DIR}/hp_sweep_grid.py" || exit 1
fi

FORCE="${FORCE}" python3 - "${CELLS_FILE}" <<'PY'
import json, os, sys
sys.path.insert(0, "/nethome/chuang475/testnvme/projects/SafeManip/SafeManip")
from monitor.hp_sweep_grid import cells, load_subset, SWEEP_DIR

force = os.environ.get("FORCE") == "1"
subset = load_subset()
out = []
for c in cells():
    want = sum(len(subset[s][t]) for s in c["suites"] for t in subset[s])
    have = len(list((SWEEP_DIR / "cells" / c["name"]).rglob(
        "privileged_information_*_monitor.json")))
    if force or have < want:
        out.append(c["name"])
    else:
        print(f"[skip] {c['name']}: {have}/{want} already done")
with open(sys.argv[1], "w") as f:
    f.write("\n".join(out) + ("\n" if out else ""))
print(f"[cells] {len(out)} to run -> {sys.argv[1]}")
PY

N=$(grep -c . "${CELLS_FILE}" 2>/dev/null || echo 0)
if [[ "${N}" -eq 0 ]]; then
  echo "nothing to submit"
  exit 0
fi

job_id=$(sbatch --parsable --array="0-$((N - 1))" \
  --export="ALL,CELLS_FILE=${CELLS_FILE}" \
  "${SCRIPT_DIR}/run_hp_sweep_cell.sbatch")
echo "job ${job_id} (array 0-$((N - 1)), ${N} cells) submitted"
echo "  watch:  squeue -j ${job_id}"
echo "  logs:   ${SCRIPT_DIR}/logs/hp_sweep_cell-${job_id}_*.out"
echo "  score:  python3 ${SCRIPT_DIR}/hp_sweep_score.py"
