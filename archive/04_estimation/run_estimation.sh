#!/bin/bash
#
# RUN_ESTIMATION.sh - Run SPM model estimation from batch file
#
# This script runs SPM model estimation using a batch template and
# adds contrasts automatically.
#
# Usage:
#   ./run_estimation.sh <batch_file> <output_dir> [--screen]
#
# Arguments:
#   batch_file  - MATLAB batch file (e.g., batch_3x3_job.m)
#   output_dir  - Output directory for results
#   --screen    - Run screening after adding contrasts (optional)
#
# Example:
#   ./run_estimation.sh ../01_design/templates/batch_3x3_job.m ../results/vbm/s9_int_control --screen

set -euo pipefail

BATCH_FILE="${1:-}"
OUTPUT_DIR="${2:-}"
RUN_SCREENING=false

# Check for --screen flag
if [[ "${3:-}" == "--screen" ]]; then
    RUN_SCREENING=true
fi

if [[ -z "$BATCH_FILE" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: $0 <batch_file> <output_dir> [--screen]"
    exit 1
fi

if [[ ! -f "$BATCH_FILE" ]]; then
    echo "Error: Batch file not found: $BATCH_FILE"
    exit 1
fi

# Get absolute paths
BATCH_FILE=$(cd "$(dirname "$BATCH_FILE")" && pwd)/$(basename "$BATCH_FILE")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATS_DIR="$(dirname "$SCRIPT_DIR")"

# Find MATLAB
MATLAB_EXE=$(find /Applications -name "MATLAB_R*.app" -maxdepth 1 2>/dev/null | \
             sort -r | head -1)/bin/matlab

if [[ -z "$MATLAB_EXE" || ! -f "$MATLAB_EXE" ]]; then
    echo "Error: MATLAB not found in /Applications"
    exit 1
fi

echo "Running SPM estimation..."
echo "  Batch file:   $BATCH_FILE"
echo "  Output dir:   $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run estimation with SPM batch
"$MATLAB_EXE" -nodesktop -nodisplay -nosplash -batch "\
    addpath('$STATS_DIR/utils'); \
    addpath('$STATS_DIR/05_screening'); \
    configure_spm_path; \
    spm('defaults', 'FMRI'); \
    spm_jobman('initcfg'); \
    spm_get_defaults('cmdline', true); \
    global defaults; \
    defaults.cmdline = true; \
    fprintf('Loading batch file...\n'); \
    run('$BATCH_FILE'); \
    fprintf('Running factorial design specification...\n'); \
    spm_jobman('run', matlabbatch); \
    clear matlabbatch; \
    fprintf('Running model estimation...\n'); \
    matlabbatch{1}.spm.stats.fmri_est.spmmat = {'$OUTPUT_DIR/SPM.mat'}; \
    matlabbatch{1}.spm.stats.fmri_est.write_residuals = 0; \
    matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1; \
    spm_jobman('run', matlabbatch); \
    fprintf('Adding contrasts...\n'); \
    cd('$STATS_DIR/03_contrasts'); \
    add_contrasts_longitudinal('$OUTPUT_DIR'); \
    if $RUN_SCREENING; then \
        fprintf('\nRunning screening...\n'); \
        sig_cons = screen_contrasts('$OUTPUT_DIR'); \
        fprintf('Significant contrasts: %s\n', mat2str(sig_cons)); \
        save('$OUTPUT_DIR/screening_results.mat', 'sig_cons'); \
    end; \
    fprintf('\\n✓ Estimation complete\\n'); \
    exit;" || exit 1

echo ""
echo "✓ Estimation successful"
echo "  SPM.mat created in: $OUTPUT_DIR"
echo ""

exit 0
