#!/usr/bin/env bash
set -euo pipefail
# Install spm-python's MATLAB Runtime bindings into the active virtualenv.
# Usage: ./utils/install_spm_python.sh [/path/to/MCR]

VENV_DIR="$(cd "$(dirname "$0")/.." && pwd)/.venv"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "No virtualenv found at $VENV_DIR. Activate or create .venv first." >&2
  exit 1
fi

MCR_ROOT="${1:-/Applications/MATLAB/MATLAB_Runtime/R2024b}"

echo "Activating venv: $VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "Installing spm-python MATLAB runtime bindings using MCR root: $MCR_ROOT"
if python -m spm.install_matlab_runtime --help >/dev/null 2>&1; then
  python -m spm.install_matlab_runtime --mcr-root "$MCR_ROOT"
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "spm-python runtime installer returned exit code $RC" >&2
    exit $RC
  fi
else
  echo "spm.install_matlab_runtime entrypoint not found. Ensure spm-python is installed in the venv." >&2
  exit 2
fi

echo "spm-python MATLAB runtime bindings installation complete."
