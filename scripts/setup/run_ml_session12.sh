#!/usr/bin/env bash
set -euo pipefail

# run_ml_session12.sh
# Lightweight wrapper to run the VBM ML interaction for session 1 -> session 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv_ml"

if [ ! -x "$REPO_ROOT/install_ml.sh" ]; then
  echo "install_ml.sh not found or not executable. Run ./install_ml.sh first to create the venv." >&2
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Virtualenv $VENV_DIR not found. Run ./install_ml.sh to create it." >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"

python utils/vbm_ml_interaction.py \
  --design-json results/vbm/s6_vbm_final/design.json \
  --session-a 1 --session-b 2 \
  --output results/vbm/s6_vbm_final/ml_results_ses1_vs_2 \
  --mask templates/brainmask_GMtight.nii \
  --n-permutations 500 \
  --classifier svc \
  --group-col 3 \
  --cv-folds 5

deactivate

echo "Done. Results under results/vbm/s6_vbm_final/ml_results_ses1_vs_2"
