#!/bin/bash
#
# CAT12 Multi-Modality Pipeline Wrapper
# ======================================
#
# Orchestrates running the main pipeline for multiple modalities defined in config.json
#
# USAGE:
#   ./cat12_multi_modality.sh --config <json> --cat12-dir <path> [options]
#
# This script reads the "analysis.modalities" array from config.json and
# executes the main pipeline for each modality with its specific settings.
#

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PIPELINE_SCRIPT="$SCRIPT_DIR/cat12_longitudinal_analysis.sh"

# Parse arguments (pass most through to main script)
CONFIG_JSON=""
CAT12_DIR=""
PARTICIPANTS_FILE=""
SPECIFIC_MODALITY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_JSON="$2"
            shift 2
            ;;
        --cat12-dir)
            CAT12_DIR="$2"
            shift 2
            ;;
        --participants)
            PARTICIPANTS_FILE="$2"
            shift 2
            ;;
        --modality)
            # Allow filtering to single modality
            SPECIFIC_MODALITY="$2"
            shift 2
            ;;
        *)
            # Pass through unknown args
            shift
            ;;
    esac
done

if [[ -z "$CONFIG_JSON" ]] || [[ -z "$CAT12_DIR" ]]; then
    echo "Usage: $0 --config <json> --cat12-dir <path> [--modality <name>] [other options]"
    echo ""
    echo "This wrapper orchestrates multi-modality analysis."
    echo "Define modalities in config.json under 'analysis.modalities'."
    exit 1
fi

# Extract modalities from config
echo "Extracting modalities from config.json..."

if [[ -n "$SPECIFIC_MODALITY" ]]; then
    # Get single modality
    MODALITY_JSON=$(python3 "$ROOT_DIR/scripts/utils/extract_modalities.py" "$CONFIG_JSON" --modality "$SPECIFIC_MODALITY" --json)
    MODALITIES="[$MODALITY_JSON]"
else
    # Get all modalities
    MODALITIES=$(python3 "$ROOT_DIR/scripts/utils/extract_modalities.py" "$CONFIG_JSON" --json)
fi

echo "Modalities: $MODALITIES"
echo ""

# Parse JSON array (simple approach using jq if available, else fallback)
if command -v jq &> /dev/null; then
    N_MODALITIES=$(echo "$MODALITIES" | jq 'length')
else
    # Fallback: count comma-separated objects (crude but works for simple cases)
    N_MODALITIES=$(echo "$MODALITIES" | grep -o '"name"' | wc -l)
fi

echo "Found $N_MODALITIES modality configuration(s)"
echo ""

# Loop through modalities
FAILED_MODALITIES=()
SUCCESSFUL_MODALITIES=()

for ((i=0; i<N_MODALITIES; i++)); do
    if command -v jq &> /dev/null; then
        MOD_CONFIG=$(echo "$MODALITIES" | jq ".[$i]")
        MODALITY=$(echo "$MOD_CONFIG" | jq -r '.name')
        FOLDER_NAME=$(echo "$MOD_CONFIG" | jq -r '.folder_name // empty')
        SMOOTHING=$(echo "$MOD_CONFIG" | jq -r '.smoothing_kernel // empty')
        COVARIATES=$(echo "$MOD_CONFIG" | jq -r '.covariates | join(",") // empty')
    else
        # Crude Python fallback for extracting config
        MOD_CONFIG=$(python3 - "$MODALITIES" "$i" <<'PYEOF'
import json, sys
mods = json.loads(sys.argv[1])
print(json.dumps(mods[int(sys.argv[2])]))
PYEOF
)
        MODALITY=$(echo "$MOD_CONFIG" | python3 -c "import sys, json; print(json.load(sys.stdin).get('name', 'unknown'))")
        FOLDER_NAME=$(echo "$MOD_CONFIG" | python3 -c "import sys, json; print(json.load(sys.stdin).get('folder_name', ''))")
        SMOOTHING=$(echo "$MOD_CONFIG" | python3 -c "import sys, json; val = json.load(sys.stdin).get('smoothing_kernel'); print(val if val else '')")
        COVARIATES=$(echo "$MOD_CONFIG" | python3 -c "import sys, json; vals = json.load(sys.stdin).get('covariates', []); print(','.join(vals) if vals else '')")
    fi
    
    echo "════════════════════════════════════════════════════════════════════════"
    echo "Running: Modality $((i+1))/$N_MODALITIES - $MODALITY"
    echo "════════════════════════════════════════════════════════════════════════"
    echo "  Folder name: ${FOLDER_NAME:-default}"
    echo "  Smoothing kernel: ${SMOOTHING:-auto}"
    echo "  Covariates: ${COVARIATES:-none}"
    echo ""
    
    # Build command for this modality
    CMD="$PIPELINE_SCRIPT --config $CONFIG_JSON --cat12-dir $CAT12_DIR"
    
    if [[ -n "$PARTICIPANTS_FILE" ]]; then
        CMD="$CMD --participants $PARTICIPANTS_FILE"
    fi
    
    CMD="$CMD --modality $MODALITY"
    
    if [[ -n "$FOLDER_NAME" ]]; then
        CMD="$CMD --analysis-name $FOLDER_NAME"
    fi
    
    if [[ -n "$SMOOTHING" ]] && [[ "$SMOOTHING" != "null" ]]; then
        CMD="$CMD --smoothing $SMOOTHING"
    fi
    
    if [[ -n "$COVARIATES" ]]; then
        CMD="$CMD --covariates $COVARIATES"
    fi
    
    # Run the pipeline for this modality
    if bash $CMD; then
        echo ""
        echo "✓ Modality '$MODALITY' completed successfully"
        SUCCESSFUL_MODALITIES+=("$MODALITY")
    else
        echo ""
        echo "✗ Modality '$MODALITY' failed"
        FAILED_MODALITIES+=("$MODALITY")
    fi
    
    echo ""
done

# Summary
echo "════════════════════════════════════════════════════════════════════════"
echo "Multi-Modality Pipeline Summary"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

if [[ ${#SUCCESSFUL_MODALITIES[@]} -gt 0 ]]; then
    echo "✓ Successful modalities (${#SUCCESSFUL_MODALITIES[@]}):"
    for mod in "${SUCCESSFUL_MODALITIES[@]}"; do
        echo "    - $mod"
    done
    echo ""
fi

if [[ ${#FAILED_MODALITIES[@]} -gt 0 ]]; then
    echo "✗ Failed modalities (${#FAILED_MODALITIES[@]}):"
    for mod in "${FAILED_MODALITIES[@]}"; do
        echo "    - $mod"
    done
    echo ""
    exit 1
else
    echo "All modalities completed successfully!"
fi
