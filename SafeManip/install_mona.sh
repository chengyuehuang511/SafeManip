#!/usr/bin/env bash

set -euo pipefail

MONA_VERSION="${MONA_VERSION:-1.4-18}"
MONA_ARCHIVE="mona-${MONA_VERSION}.tar.gz"
MONA_URL="${MONA_URL:-https://www.brics.dk/mona/download/${MONA_ARCHIVE}}"

# Default to the active conda env when available, otherwise install into ~/.local.
PREFIX="${MONA_PREFIX:-${CONDA_PREFIX:-$HOME/.local}}"
BUILD_ROOT="${MONA_BUILD_ROOT:-$PWD}"

usage() {
  cat <<EOF
Usage:
  bash install_mona.sh
  bash install_mona.sh --uninstall

Optional environment variables:
  MONA_PREFIX      Install prefix. Default: \$CONDA_PREFIX or ~/.local
  MONA_BUILD_ROOT  Directory where sources are downloaded and built. Default: current directory
  MONA_VERSION     MONA version suffix. Default: ${MONA_VERSION}
  MONA_URL         Full archive URL override. Default: ${MONA_URL}

Examples:
  MONA_PREFIX="\$CONDA_PREFIX" bash install_mona.sh
  MONA_PREFIX="$HOME/.local" bash install_mona.sh
EOF
}

need_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd"
    return 1
  fi
}

check_dependencies() {
  local missing=0
  for cmd in wget tar make gcc g++ flex bison; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      echo "Missing required build dependency: $cmd"
      missing=1
    fi
  done
  if [[ "$missing" -ne 0 ]]; then
    cat <<EOF

This script no longer uses sudo. Please install the missing tools in user space first.

If you are using conda, a typical fix is:
  conda install -c conda-forge flex bison make gcc gxx wget

Then rerun:
  MONA_PREFIX="${PREFIX}" bash install_mona.sh
EOF
    exit 1
  fi
}

installed_mona() {
  if command -v mona >/dev/null 2>&1; then
    local mona_path
    mona_path="$(command -v mona)"
    case "$mona_path" in
      "$PREFIX"/*) return 0 ;;
    esac
  fi
  [[ -x "$PREFIX/bin/mona" ]]
}

do_uninstall() {
  local extracted_dir=""
  local archive_listing=""

  if [[ -f "$BUILD_ROOT/$MONA_ARCHIVE" ]]; then
    archive_listing="$(mktemp)"
    tar -tzf "$BUILD_ROOT/$MONA_ARCHIVE" > "$archive_listing"
    extracted_dir="$(head -1 "$archive_listing" | cut -d/ -f1)"
    rm -f "$archive_listing"
  fi

  echo "Removing MONA from $PREFIX"
  rm -f "$PREFIX/bin/mona"
  rm -f "$PREFIX/share/man/man1/mona.1" "$PREFIX/man/man1/mona.1"
  if [[ -n "$extracted_dir" ]]; then
    rm -rf "$BUILD_ROOT/$extracted_dir"
  fi
  echo "Uninstall complete."
}

do_install() {
  local extracted_dir
  local archive_listing

  mkdir -p "$BUILD_ROOT"
  cd "$BUILD_ROOT"

  if [[ ! -f "$MONA_ARCHIVE" ]]; then
    echo "Downloading $MONA_URL"
    wget -O "$MONA_ARCHIVE" "$MONA_URL"
  fi

  archive_listing="$(mktemp)"
  tar -tzf "$MONA_ARCHIVE" > "$archive_listing"
  extracted_dir="$(head -1 "$archive_listing" | cut -d/ -f1)"
  rm -f "$archive_listing"
  if [[ -z "$extracted_dir" ]]; then
    echo "Could not determine extracted directory from $MONA_ARCHIVE"
    exit 1
  fi

  rm -rf "$extracted_dir"
  tar -xzf "$MONA_ARCHIVE"
  cd "$extracted_dir"

  echo "Configuring MONA with prefix $PREFIX"
  ./configure --prefix="$PREFIX" --enable-static --enable-shared=no

  echo "Building MONA"
  make -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"

  echo "Installing MONA into $PREFIX"
  make install-strip

  if [[ -x "$PREFIX/bin/mona" ]]; then
    echo "MONA installed successfully at $PREFIX/bin/mona"
    echo
    echo "If it is not already on PATH, add:"
    echo "  export PATH=\"$PREFIX/bin:\$PATH\""
  else
    echo "Install finished, but $PREFIX/bin/mona was not found."
    exit 1
  fi
}

main() {
  if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
  fi

  if [[ "${1:-}" == "--uninstall" ]]; then
    do_uninstall
    exit 0
  fi

  check_dependencies

  if installed_mona; then
    echo "MONA already appears installed under $PREFIX"
    echo "Run with --uninstall if you want to remove it first."
    exit 0
  fi

  do_install
}

main "${1:-}"
