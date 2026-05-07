#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

INSTALL_ROBOCASA=1
INSTALL_ROBOSUITE=1
INSTALL_GROOT=0
INSTALL_OPENPI=0
INSTALL_MONITOR=1
INSTALL_MONA=1
CLONE_ROBOSUITE=1
ROBOSUITE_URL="${ROBOSUITE_URL:-https://github.com/ARISE-Initiative/robosuite.git}"

usage() {
  cat <<'EOF'
Usage:
  bash install.sh [options]

Options:
  --monitor-only  Install only monitor Python deps and MONA.
  --with-groot    Install GR00T in editable mode.
  --with-openpi   Install OpenPI in editable mode.
  --no-clone-robosuite
                  Do not clone RoboSuite if ./robosuite is missing.
  --no-robosuite  Skip editable RoboSuite install.
  --no-robocasa   Skip editable RoboCasa install.
  --no-monitor    Skip monitor Python deps and MONA.
  --no-mona       Skip MONA install.
  -h, --help      Show this help.

Environment variables:
  PIP              Python package installer command. Default: python -m pip
  ROBOSUITE_URL    RoboSuite git URL. Default: https://github.com/ARISE-Initiative/robosuite.git
  MONA_PREFIX      Prefix passed to SafeManip/install_mona.sh.
  MONA_BUILD_ROOT  Build directory passed to SafeManip/install_mona.sh.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --monitor-only)
      INSTALL_ROBOSUITE=0
      INSTALL_ROBOCASA=0
      INSTALL_GROOT=0
      INSTALL_OPENPI=0
      INSTALL_MONITOR=1
      INSTALL_MONA=1
      ;;
    --with-groot)
      INSTALL_GROOT=1
      ;;
    --with-openpi)
      INSTALL_OPENPI=1
      ;;
    --no-clone-robosuite)
      CLONE_ROBOSUITE=0
      ;;
    --no-robosuite)
      INSTALL_ROBOSUITE=0
      ;;
    --no-robocasa)
      INSTALL_ROBOCASA=0
      ;;
    --no-monitor)
      INSTALL_MONITOR=0
      INSTALL_MONA=0
      ;;
    --no-mona)
      INSTALL_MONA=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

PIP_CMD="${PIP:-python -m pip}"

run_pip() {
  echo "+ ${PIP_CMD} $*"
  # shellcheck disable=SC2086
  ${PIP_CMD} "$@"
}

install_editable() {
  local name="$1"
  local path="$2"
  local marker="$3"

  if [[ ! -e "${path}/${marker}" ]]; then
    echo "Skipping ${name}: missing ${path}/${marker}"
    return
  fi

  echo "Installing ${name} from ${path}"
  run_pip install -e "${path}"
}

ensure_robosuite_checkout() {
  local path="${PROJECT_ROOT}/robosuite"
  if [[ -e "${path}/setup.py" ]]; then
    return
  fi
  if [[ "${CLONE_ROBOSUITE}" != "1" ]]; then
    return
  fi
  if [[ -e "${path}" ]]; then
    echo "Skipping RoboSuite clone: ${path} exists but setup.py was not found." >&2
    return
  fi
  if ! command -v git >/dev/null 2>&1; then
    echo "Skipping RoboSuite clone: git is not available." >&2
    return
  fi
  echo "Cloning RoboSuite from ${ROBOSUITE_URL}"
  git clone "${ROBOSUITE_URL}" "${path}"
}

install_monitor_deps() {
  echo "Installing SafeManip monitor dependencies"
  run_pip install pydot networkx tqdm numpy pandas
}

install_mona() {
  if command -v mona >/dev/null 2>&1; then
    echo "MONA already available at $(command -v mona)"
    return
  fi

  echo "Installing MONA"
  bash "${PROJECT_ROOT}/SafeManip/install_mona.sh"
}

main() {
  cd "${PROJECT_ROOT}"

  if [[ "${INSTALL_ROBOSUITE}" == "1" ]]; then
    ensure_robosuite_checkout
    install_editable "RoboSuite" "${PROJECT_ROOT}/robosuite" "setup.py"
  fi

  if [[ "${INSTALL_ROBOCASA}" == "1" ]]; then
    install_editable "RoboCasa" "${PROJECT_ROOT}/robocasa" "setup.py"
  fi

  if [[ "${INSTALL_GROOT}" == "1" ]]; then
    install_editable "GR00T" "${PROJECT_ROOT}/Isaac-GR00T" "pyproject.toml"
  fi

  if [[ "${INSTALL_OPENPI}" == "1" ]]; then
    install_editable "OpenPI" "${PROJECT_ROOT}/openpi" "pyproject.toml"
    install_editable "OpenPI client" "${PROJECT_ROOT}/openpi/packages/openpi-client" "pyproject.toml"
  fi

  if [[ "${INSTALL_MONITOR}" == "1" ]]; then
    install_monitor_deps
  fi

  if [[ "${INSTALL_MONA}" == "1" ]]; then
    install_mona
  fi

  echo "Install complete."
}

main
