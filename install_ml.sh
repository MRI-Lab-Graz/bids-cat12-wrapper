#!/usr/bin/env bash
set -euo pipefail

# install_ml.sh
# Create a reproducible Python virtual environment for VBM ML analysis
# - creates a venv at .venv_ml
# - installs pinned package versions from requirements_ml.txt
# - writes an exact freeze to requirements_installed_ml.txt for provenance

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv_ml"
REQ_FILE="$REPO_ROOT/requirements_ml.txt"
FREEZE_FILE="$REPO_ROOT/requirements_installed_ml.txt"
PYTHON="$VENV_DIR/bin/python"

echo "[install_ml] repository root: $REPO_ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found in PATH. Please install Python 3 and retry." >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "[install_ml] creating venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
else
  echo "[install_ml] using existing venv at $VENV_DIR"
fi

echo "[install_ml] upgrading pip, setuptools, wheel"
"$PYTHON" -m pip install --upgrade pip setuptools wheel

if [ ! -f "$REQ_FILE" ]; then
  echo "[install_ml] writing default pinned requirements to $REQ_FILE"
  cat > "$REQ_FILE" <<'REQ'
nilearn==0.12.1
scikit-learn==1.7.2
nibabel==5.3.2
numpy==2.2.6
pandas==2.3.3
matplotlib==3.10.7
scipy==1.15.3
joblib==1.5.2
REQ
fi

echo "[install_ml] installing packages from $REQ_FILE"
"$PYTHON" -m pip install -r "$REQ_FILE"

echo "[install_ml] writing exact installed package list to $FREEZE_FILE"
"$PYTHON" -m pip freeze > "$FREEZE_FILE"

cat <<EOF
[install_ml] Done.

To activate the environment and run the ML script:

  source "$VENV_DIR/bin/activate"
  python utils/vbm_ml_interaction.py --participants-tsv participants.tsv --data-root data/cat12 \
    --output results/vbm/s6_vbm_final/ml_results --mask templates/brainmask_GMtight.nii \
    --n-permutations 500 --classifier svc --group-col 3

Notes:
- The pinned versions live in `requirements_ml.txt` for reproducibility.
- The exact installed freeze is saved to `requirements_installed_ml.txt`.
EOF

exit 0
