#!/usr/bin/env bash
# Launch the CAT12 Web UI.
# Usage: bash scripts/webui/start_webui.sh [--host HOST] [--port PORT] [--no-browser]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate the project virtual environment if present.
VENV="$WORKSPACE_ROOT/.venv"
if [[ -f "$VENV/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$VENV/bin/activate"
fi

exec python "$SCRIPT_DIR/app.py" "$@"
