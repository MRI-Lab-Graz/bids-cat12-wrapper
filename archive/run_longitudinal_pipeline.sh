#!/bin/bash
#
# Complete Pipeline for Longitudinal VBM/SBM Analysis
# ====================================================
#
# This script orchestrates the complete workflow:
#   1. Design specification (flexible factorial)
#   2. Add covariates (TIV, IQR)
#   3. Define contrasts (main effects, interactions, pairwise)
#   4. Estimate model (SPM GLM)
#   5. Screen contrasts (uncorrected p<0.001)
#   6. TFCE correction (on significant contrasts only)
#
# Usage:
#   ./run_longitudinal_pipeline.sh <analysis_type> <output_dir> [options]
#
# Arguments:
#   analysis_type: 'vbm' or 'sbm' (volume or surface-based)
#   output_dir:    Name for results directory (e.g., 's9_int_control')
#
# Options:
#   --with-covariates     Include TIV and IQR covariates
#   --skip-estimation     Skip model estimation (if already done)
#   --skip-screening      Skip screening (run TFCE on all contrasts)
#   --pilot               TFCE pilot mode (fewer permutations)
#   --n-perm N            Number of TFCE permutations (default: 5000)
#   --n-jobs N            Parallel TFCE jobs (default: 4)
#
# Example:
#   ./run_longitudinal_pipeline.sh vbm s9_int_control --with-covariates
#

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_TYPE="${1:-}"
OUTPUT_DIR="${2:-}"

# Default parameters
WITH_COVARIATES=false
SKIP_ESTIMATION=false
SKIP_SCREENING=false
PILOT_MODE=false
N_PERM=5000
N_JOBS=4

# Parse options
shift 2 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --with-covariates)
            WITH_COVARIATES=true
            shift
            ;;
        --skip-estimation)
            SKIP_ESTIMATION=true
            shift
            ;;
        --skip-screening)
            SKIP_SCREENING=true
            shift
            ;;
        --pilot)
            PILOT_MODE=true
            shift
            ;;
        --n-perm)
            N_PERM="$2"
            shift 2
            ;;
        --n-jobs)
            N_JOBS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ -z "$ANALYSIS_TYPE" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: $0 <analysis_type> <output_dir> [options]"
    echo ""
    echo "Example: $0 vbm s9_int_control --with-covariates"
    exit 1
fi

if [[ "$ANALYSIS_TYPE" != "vbm" && "$ANALYSIS_TYPE" != "sbm" ]]; then
    echo "Error: analysis_type must be 'vbm' or 'sbm'"
    exit 1
fi

# Set paths
if [[ "$ANALYSIS_TYPE" == "vbm" ]]; then
    RESULTS_DIR="$SCRIPT_DIR/results/vbm/$OUTPUT_DIR"
else
    RESULTS_DIR="$SCRIPT_DIR/results/sbm/$OUTPUT_DIR"
fi

# ============================================================================
# Helper Functions
# ============================================================================

log_step() {
    echo ""
    echo "========================================"
    echo "STEP $1: $2"
    echo "========================================"
    echo ""
}

log_info() {
    echo "[INFO] $1"
}

log_error() {
    echo "[ERROR] $1" >&2
    exit 1
}

# ============================================================================
# Pipeline Steps
# ============================================================================

log_info "Longitudinal Analysis Pipeline"
log_info "Analysis type: $ANALYSIS_TYPE"
log_info "Output directory: $OUTPUT_DIR"
log_info "With covariates: $WITH_COVARIATES"
log_info "Results path: $RESULTS_DIR"

# ----------------------------------------------------------------------------
# STEP 1: Design Specification
# ----------------------------------------------------------------------------

log_step "1" "Design Specification"

if [[ "$ANALYSIS_TYPE" == "vbm" ]]; then
    if [[ "$WITH_COVARIATES" == true ]]; then
        DESIGN_TEMPLATE="$SCRIPT_DIR/01_design/templates/batch_3x3_job_with_covariates.m"
    else
        DESIGN_TEMPLATE="$SCRIPT_DIR/01_design/templates/batch_3x3_job.m"
    fi
else
    if [[ "$WITH_COVARIATES" == true ]]; then
        log_error "SBM with covariates not yet implemented"
    else
        DESIGN_TEMPLATE="$SCRIPT_DIR/01_design/templates/batch_3x3_surface_job.m"
    fi
fi

if [[ ! -f "$DESIGN_TEMPLATE" ]]; then
    log_error "Design template not found: $DESIGN_TEMPLATE"
fi

log_info "Using design template: $(basename "$DESIGN_TEMPLATE")"
log_info "✓ Design specification ready"

# ----------------------------------------------------------------------------
# STEP 2: Covariates (if requested)
# ----------------------------------------------------------------------------

if [[ "$WITH_COVARIATES" == true ]]; then
    log_step "2" "Covariate Management"
    
    if [[ ! -f "$SCRIPT_DIR/02_covariates/data/TIV.txt" ]]; then
        log_error "TIV.txt not found in 02_covariates/data/"
    fi
    
    if [[ ! -f "$SCRIPT_DIR/02_covariates/data/IQR.txt" ]]; then
        log_error "IQR.txt not found in 02_covariates/data/"
    fi
    
    log_info "Verifying covariate alignment..."
    cd "$SCRIPT_DIR/02_covariates"
    python3 verify_covariates.py || log_error "Covariate verification failed"
    
    log_info "✓ Covariates verified and ready"
else
    log_info "Skipping covariates (not requested)"
fi

# ----------------------------------------------------------------------------
# STEP 3: Contrast Definition
# ----------------------------------------------------------------------------

log_step "3" "Contrast Definition"

log_info "Adding longitudinal contrasts..."
# This will be called within MATLAB during estimation

log_info "✓ Contrast definition ready"

# ----------------------------------------------------------------------------
# STEP 4: Model Estimation
# ----------------------------------------------------------------------------

if [[ "$SKIP_ESTIMATION" == false ]]; then
    log_step "4" "Model Estimation"
    
    log_info "Running SPM estimation..."
    log_info "This may take several minutes..."
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    
    # Run estimation in MATLAB
    cd "$SCRIPT_DIR/04_estimation"
    ./run_estimation.sh "$DESIGN_TEMPLATE" "$RESULTS_DIR" || log_error "Estimation failed"
    
    log_info "✓ Model estimation complete"
else
    log_info "Skipping estimation (already done)"
    
    if [[ ! -f "$RESULTS_DIR/SPM.mat" ]]; then
        log_error "SPM.mat not found in $RESULTS_DIR - cannot skip estimation"
    fi
fi

# ----------------------------------------------------------------------------
# STEP 5: Contrast Screening (Uncorrected)
# ----------------------------------------------------------------------------

if [[ "$SKIP_SCREENING" == false ]]; then
    log_step "5" "Uncorrected Screening"
    
    log_info "Screening contrasts at p<0.001 uncorrected, k≥50 voxels..."
    
    cd "$SCRIPT_DIR/05_screening"
    
    # Run screening in MATLAB
    matlab -nodesktop -nodisplay -nosplash -batch "\
        addpath('$SCRIPT_DIR/utils'); \
        configure_spm_path; \
        cd('$RESULTS_DIR'); \
        significant_contrasts = screen_contrasts('$RESULTS_DIR'); \
        save('significant_contrasts.mat', 'significant_contrasts'); \
        fprintf('Found %d significant contrasts\n', length(significant_contrasts)); \
        exit;" || log_error "Screening failed"
    
    # Count significant contrasts
    N_SIG=$(matlab -nodesktop -nodisplay -nosplash -batch "\
        load('$RESULTS_DIR/significant_contrasts.mat'); \
        fprintf('%d\n', length(significant_contrasts)); \
        exit;" 2>/dev/null | tail -1)
    
    log_info "✓ Screening complete: $N_SIG significant contrasts found"
else
    log_info "Skipping screening - will run TFCE on ALL contrasts"
fi

# ----------------------------------------------------------------------------
# STEP 6: TFCE Correction
# ----------------------------------------------------------------------------

log_step "6" "TFCE Multiple Comparison Correction"

cd "$SCRIPT_DIR/06_tfce"

TFCE_OPTS=""
[[ "$PILOT_MODE" == true ]] && TFCE_OPTS="$TFCE_OPTS --pilot"
TFCE_OPTS="$TFCE_OPTS --n-perm $N_PERM --n-jobs $N_JOBS"

if [[ "$SKIP_SCREENING" == false ]]; then
    TFCE_OPTS="$TFCE_OPTS --use-screening"
fi

log_info "Running TFCE with $N_PERM permutations, $N_JOBS parallel jobs..."
log_info "This will take several hours..."

./run_tfce_headless.sh "$RESULTS_DIR" $TFCE_OPTS || log_error "TFCE failed"

log_info "✓ TFCE correction complete"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "========================================"
echo "PIPELINE COMPLETE"
echo "========================================"
echo ""
echo "Results location: $RESULTS_DIR"
echo ""
echo "Next steps:"
echo "  1. Review TFCE results in: $RESULTS_DIR/TFCE/"
echo "  2. Check cluster tables: $RESULTS_DIR/TFCE/*/cluster_table.txt"
echo "  3. Visualize in SPM: Open SPM > Results > Select con_*.nii"
echo ""

exit 0
