#!/usr/bin/env bash
set -euo pipefail

# install.sh
# Install Python dependencies for CAT12 longitudinal analysis pipeline
# Uses UV for fast package installation

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
REQ_FILE="$REPO_ROOT/requirements.txt"

echo "════════════════════════════════════════════════════════════════"
echo "CAT12 Longitudinal Analysis Pipeline - Installation"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check for uv
if ! command -v uv >/dev/null 2>&1; then
    echo "❌ UV not found. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "✓ UV found: $(command -v uv)"
echo ""

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    uv venv "$VENV_DIR"
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists at $VENV_DIR"
fi

# Create requirements.txt if it doesn't exist
if [ ! -f "$REQ_FILE" ]; then
    echo "Creating requirements.txt..."
    cat > "$REQ_FILE" <<'EOF'
# Core scientific computing
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0

# Neuroimaging
nibabel>=5.0.0

# Machine learning & visualization
matplotlib>=3.7.0
scikit-learn>=1.3.0
nilearn>=0.12.0
joblib>=1.3.0
EOF
    echo "✓ Created $REQ_FILE"
fi

echo ""
echo "Installing packages..."
uv pip install -r "$REQ_FILE"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✓ Installation complete!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run the analysis pipeline:"
echo "  ./cat12_longitudinal_analysis.sh --cat12-dir <path> --participants <tsv>"
echo ""
