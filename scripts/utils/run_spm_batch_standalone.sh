#!/bin/bash
#
# run_spm_batch_standalone.sh
#
# Wrapper to run SPM batch files through cat_standalone.sh 
# without requiring custom MATLAB functions or addpath() calls.
#
# For standalone mode, this focuses on running pre-generated batch files
# that use only standard SPM modules.
#
# Usage:
#   ./run_spm_batch_standalone.sh <batch_file> <mcrroot> <spmroot> [log_file]
#

set -e

BATCH_FILE="$1"
MCR_ROOT="$2"
SPM_ROOT="$3"
LOG_FILE="${4:-/tmp/spm_batch.log}"

if [[ -z "$BATCH_FILE" ]] || [[ -z "$MCR_ROOT" ]] || [[ -z "$SPM_ROOT" ]]; then
    echo "Usage: $0 <batch_file> <mcrroot> <spmroot> [log_file]"
    echo ""
    echo "Example:"
    echo "  $0 stats/spm_batch.m /usr/local/lib/mcr/v923 /data/cat12/external/cat12 output.log"
    exit 1
fi

if [[ ! -f "$BATCH_FILE" ]]; then
    echo "Error: Batch file not found: $BATCH_FILE" >&2
    exit 1
fi

if [[ ! -d "$MCR_ROOT" ]]; then
    echo "Error: MCR directory not found: $MCR_ROOT" >&2
    exit 1
fi

if [[ ! -f "$SPM_ROOT/run_spm25.sh" ]] && [[ ! -f "$SPM_ROOT/run_spm12.sh" ]]; then
    echo "Error: SPM executable not found in $SPM_ROOT" >&2
    exit 1
fi

# Resolve absolute paths
BATCH_FILE="$(cd "$(dirname "$BATCH_FILE")" && pwd)/$(basename "$BATCH_FILE")"
MCR_ROOT="$(cd "$MCR_ROOT" && pwd)"
SPM_ROOT="$(cd "$SPM_ROOT" && pwd)"

echo "Running SPM batch via standalone mode:"
echo "  Batch file:  $BATCH_FILE"
echo "  MCR root:    $MCR_ROOT"
echo "  SPM root:    $SPM_ROOT"
echo "  Log file:    $LOG_FILE"
echo ""

# Choose SPM executable (prefer spm25 over spm12)
if [[ -f "$SPM_ROOT/run_spm25.sh" ]]; then
    SPM_EXE="$SPM_ROOT/run_spm25.sh"
else
    SPM_EXE="$SPM_ROOT/run_spm12.sh"
fi

echo "Using SPM executable: $SPM_EXE"
echo ""

# Run through cat_standalone.sh wrapper
# This calls: spm_exe $MCR_ROOT batch <batch_file>
exec "$SPM_EXE" "$MCR_ROOT" "batch" "$BATCH_FILE" 2>&1 | tee "$LOG_FILE"
