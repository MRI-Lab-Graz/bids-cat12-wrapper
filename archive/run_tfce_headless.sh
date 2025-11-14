#!/bin/bash
# Run TFCE screening in completely headless mode
# Usage: ./run_tfce_headless.sh <stats_folder_path> [--n-perm N] [--n-jobs N] [--p-thresh P] [--cluster-size N] [--force] [--no-background] [--pilot] [--multi-stage] [--n-perm-stage1 N] [--n-perm-stage2 N]
# Examples:
#   ./run_tfce_headless.sh depth
#   ./run_tfce_headless.sh depth --cluster-size 100 --n-perm 2000
#   ./run_tfce_headless.sh depth --pilot                              # Test with just 1 random contrast
#   ./run_tfce_headless.sh depth --no-background --n-perm 5000        # Wait for all TFCE to complete
#   ./run_tfce_headless.sh depth --force

if [ $# -lt 1 ]; then
    echo "Usage: $0 <stats_folder_path> [options]"
    echo ""
    echo "Options:"
    echo "  --n-perm N          Number of permutations (default: 1500)"
    echo "  --n-jobs N          Number of CPU cores (default: 4)"
    echo "  --p-thresh P        P-value threshold (default: 0.001)"
    echo "  --cluster-size N    Minimum cluster size (default: 50)"
    echo "  --force             Re-run even if TFCE files already exist"
    echo "  --no-background     Wait for all TFCE processing to complete (blocks terminal)"
    echo "  --pilot             Test mode: process only 1 random significant contrast"
    echo "  --multi-stage       Enable multi-stage TFCE (Stage 1: 500, Stage 2: 5000 perms)"
    echo "  --n-perm-stage1 N   Permutations for Stage 1 (default: 500)"
    echo "  --n-perm-stage2 N   Permutations for Stage 2 (default: 5000)"
    echo ""
    echo "Multi-stage workflow:"
    echo "  1. Check uncorrected results (p < 0.001 + cluster size)"
    echo "  2. Stage 1: TFCE with low permutations (quick check)"
    echo "  3. Stage 2: TFCE with high permutations (only if Stage 1 finds significance)"
    echo ""
    echo "Examples:"
    echo "  $0 depth"
    echo "  $0 depth --cluster-size 100 --n-perm 2000"
    echo "  $0 depth --pilot"
    echo "  $0 depth --no-background --force"
    exit 1
fi

STATS_FOLDER="$1"
shift  # Remove stats_folder from arguments, remaining will be processed below

# Convert to absolute path
# If relative path, resolve from current directory
if [[ ! "$STATS_FOLDER" = /* ]]; then
    CURRENT_DIR="$(pwd)"
    STATS_FOLDER="$CURRENT_DIR/$STATS_FOLDER"
fi

# Resolve any symlinks and normalize the path
STATS_FOLDER="$(cd "$STATS_FOLDER" 2>/dev/null && pwd)"
if [ $? -ne 0 ]; then
    echo "Error: Cannot access stats folder: $1"
    exit 1
fi

# Verify the folder exists
if [ ! -d "$STATS_FOLDER" ]; then
    echo "Error: Stats folder does not exist: $STATS_FOLDER"
    exit 1
fi

# Verify SPM.mat exists in the folder
if [ ! -f "$STATS_FOLDER/SPM.mat" ]; then
    echo "Error: SPM.mat not found in: $STATS_FOLDER/SPM.mat"
    exit 1
fi

# Get the directory where this script is located (where run_screen_and_tfce.m is)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Build the MATLAB command with all arguments
# Convert shell arguments to MATLAB name-value pairs
MATLAB_CMD="cd '$SCRIPT_DIR'; "

# Initialize with just the stats folder
MATLAB_CMD="$MATLAB_CMD run_screen_and_tfce('$STATS_FOLDER'"

# Process remaining arguments and convert to MATLAB syntax
while [ $# -gt 0 ]; do
    case "$1" in
        --n-perm)
            MATLAB_CMD="$MATLAB_CMD, 'n_perm', $2"
            shift 2
            ;;
        --n-jobs)
            MATLAB_CMD="$MATLAB_CMD, 'n_jobs', $2"
            shift 2
            ;;
        --p-thresh)
            MATLAB_CMD="$MATLAB_CMD, 'p_thresh', $2"
            shift 2
            ;;
        --cluster-size)
            MATLAB_CMD="$MATLAB_CMD, 'cluster_size', $2"
            shift 2
            ;;
        --mask-file)
            MATLAB_CMD="$MATLAB_CMD, 'mask_file', '$2'"
            shift 2
            ;;
        --force)
            MATLAB_CMD="$MATLAB_CMD, 'force', true"
            shift
            ;;
        --no-background)
            MATLAB_CMD="$MATLAB_CMD, 'no_background', true"
            shift
            ;;
        --pilot)
            MATLAB_CMD="$MATLAB_CMD, 'pilot', true"
            shift
            ;;
        --multi-stage)
            MATLAB_CMD="$MATLAB_CMD, 'multi_stage', true"
            shift
            ;;
        --n-perm-stage1)
            MATLAB_CMD="$MATLAB_CMD, 'n_perm_stage1', $2"
            shift 2
            ;;
        --n-perm-stage2)
            MATLAB_CMD="$MATLAB_CMD, 'n_perm_stage2', $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

MATLAB_CMD="$MATLAB_CMD)"

# Run MATLAB in headless mode on macOS
# Standard flags: -nodesktop -nodisplay -nosplash
# Modern MATLAB supports -batch which runs and exits cleanly

export DISPLAY=
export JAVA_TOOL_OPTIONS="-Djava.awt.headless=true"

# Prefer specific version if available, else first match
if [ -x "/Applications/MATLAB_R2025b.app/bin/matlab" ]; then
    MATLAB_EXE="/Applications/MATLAB_R2025b.app/bin/matlab"
else
    MATLAB_EXE=$(find /Applications -path "*/bin/matlab" -type f 2>/dev/null | head -1)
fi

if [ -z "$MATLAB_EXE" ]; then
    echo "Error: MATLAB not found"
    exit 1
fi

"$MATLAB_EXE" \
    -nodesktop \
    -nodisplay \
    -nosplash \
    -batch "$MATLAB_CMD" 2>&1

exit $?
