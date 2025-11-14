#!/bin/bash
#
# CAT12 Longitudinal Analysis Pipeline
# =====================================
#
# Complete automated workflow from CAT12 preprocessing to TFCE-corrected results.
#
# USAGE:
#   ./cat12_longitudinal_analysis.sh --cat12-dir <path> --participants <tsv> [options]
#
# REQUIRED ARGUMENTS:
#   --cat12-dir <path>      Path to CAT12 preprocessing output directory
#   --participants <tsv>    Path to BIDS participants.tsv file
#
# ANALYSIS OPTIONS:
#   --modality <name>       Analysis type: vbm, thickness, depth, gyrification, fractal
#                           (default: vbm)
#   --smoothing <mm>        Smoothing kernel in mm (default: auto-detect)
#   --analysis-name <name>  Custom name for analysis (default: auto-generated)
#   --output-dir <path>     Custom output directory (overrides default location)
#
# DESIGN OPTIONS:
#   --group-col <name>      Column name for group variable in participants.tsv
#   --session-col <name>    Column name for session variable (default: session)
#   --sessions <list>       Sessions to include: "all" or "1,2,3" (default: all)
#   --covariates <list>     Comma-separated covariate columns (e.g., "age,sex,tiv")
#
# TFCE OPTIONS:
#   --n-perm <N> / --nperms <N>
#                         Number of TFCE permutations (default: 5000)
#   --pilot                Run pilot mode (100 permutations, 1 contrast)
#   --skip-screening       Run TFCE on all contrasts (not recommended)
#
# SCREENING OPTIONS:
#   --cluster-size <k>     Minimum cluster size for screening (default: 50)
#   --uncorrected-p <p>    Uncorrected p-value threshold for screening (default: 0.001)
#
# OTHER OPTIONS:
#   --force                Delete existing results directory before starting
#
# EXAMPLES:
#
#   # Basic VBM analysis with 6mm smoothing and sessions 1,3
#   ./cat12_longitudinal_analysis.sh \
#       --cat12-dir /data/cat12 \
#       --participants /data/participants.tsv \
#       --smoothing 6 \
#       --sessions "1,3"
#
#   # Cortical thickness with covariates
#   ./cat12_longitudinal_analysis.sh \
#       --cat12-dir /data/cat12 \
#       --participants /data/participants.tsv \
#       --modality thickness \
#       --smoothing 20 \
#       --covariates "age,sex,tiv"

set -euo pipefail

# Capture original arguments early so we can safely reconstruct the exact
# command line later (avoids issues with unmatched quotes when we pass the
# command line into reports). Store as array to preserve spacing and quoting.
ORIGINAL_ARGS=("$@")

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
STATS_DIR="$SCRIPT_DIR"

# ============================================================================
# Load Configuration from config.ini
# ============================================================================

# Function to read INI values
get_ini_value() {
    local section="$1"
    local key="$2"
    local default="$3"
    
    if [[ ! -f "$STATS_DIR/config.ini" ]]; then
        echo "$default"
        return
    fi
    
    local value=$(awk -F '=' -v section="[$section]" -v key="$key" '
        /^\[/ { current_section = $0 }
        current_section == section && $1 ~ /^[[:space:]]*'"$key"'[[:space:]]*$/ {
            val = $2
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", val)
            print val
            exit
        }
    ' "$STATS_DIR/config.ini")
    
    if [[ -z "$value" ]]; then
        echo "$default"
    else
        echo "$value"
    fi
}

# Load configuration defaults from config.ini
MATLAB_EXE=$(get_ini_value "MATLAB" "exe" "/Applications/MATLAB_R2025b.app/bin/matlab")
SPM_PATH=$(get_ini_value "SPM" "path" "")
PYTHON_EXE=$(get_ini_value "PYTHON" "exe" "python3")

# Allow graphics windows in MATLAB? If true we omit -nodisplay. If false we add -nodisplay
MATLAB_ALLOW_GRAPHICS=$(get_ini_value "MATLAB" "allow_graphics" "true")


MODALITY=$(get_ini_value "ANALYSIS" "modality" "vbm")
SMOOTHING=$(get_ini_value "ANALYSIS" "smoothing" "")
GROUP_COL=$(get_ini_value "ANALYSIS" "group_col" "")
SESSION_COL=$(get_ini_value "ANALYSIS" "session_col" "session")
SESSIONS=$(get_ini_value "ANALYSIS" "sessions" "all")
COVARIATES=$(get_ini_value "ANALYSIS" "covariates" "")

UNCORRECTED_P=$(get_ini_value "SCREENING" "uncorrected_p" "0.001")
CLUSTER_SIZE=$(get_ini_value "SCREENING" "cluster_size" "50")
SKIP_SCREENING=$(get_ini_value "SCREENING" "skip_screening" "false")

N_PERM=$(get_ini_value "TFCE" "n_perm" "5000")
PILOT_MODE=$(get_ini_value "TFCE" "pilot_mode" "false")

N_JOBS=$(get_ini_value "PERFORMANCE" "parallel_jobs" "4")

OUTPUT_DIR=$(get_ini_value "OUTPUT" "output_dir" "")
ANALYSIS_NAME=$(get_ini_value "OUTPUT" "analysis_name" "")
FORCE=$(get_ini_value "OUTPUT" "force_clean" "false")

# Auto-detect MATLAB if empty in config
if [[ -z "$MATLAB_EXE" ]] || [[ "$MATLAB_EXE" == "false" ]]; then
    FOUND_MATLAB=$(find /Applications -name "MATLAB_R*.app" -maxdepth 1 2>/dev/null | sort -r | head -1)
    if [[ -n "$FOUND_MATLAB" ]]; then
        MATLAB_EXE="$FOUND_MATLAB/bin/matlab"
    else
        MATLAB_EXE="matlab"
    fi
fi

# Build MATLAB flags depending on whether graphics are allowed
if [[ "$MATLAB_ALLOW_GRAPHICS" == "false" ]] || [[ "$MATLAB_ALLOW_GRAPHICS" == "0" ]]; then
    MATLAB_FLAGS="-nodesktop -nodisplay -nosplash -batch"
else
    # Allow graphics (still run non-interactively via -batch). Omitting -nodisplay
    # allows figure creation on systems with a display (or XQuartz on macOS).
    MATLAB_FLAGS="-nodesktop -nosplash -batch"
fi

# ============================================================================
# Default Parameters (for values not in config)
# ============================================================================

CAT12_DIR=""
PARTICIPANTS_FILE=""

# Show help if no arguments provided
if [[ $# -eq 0 ]]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║              CAT12 Longitudinal Analysis Pipeline                      ║"
    echo "╚════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "USAGE:"
    echo "  $0 --cat12-dir <path> --participants <tsv> [options]"
    echo ""
    echo "REQUIRED ARGUMENTS:"
    echo "  --cat12-dir <path>      Path to CAT12 preprocessing output"
    echo "  --participants <tsv>    Path to BIDS participants.tsv file"
    echo ""
    echo "ANALYSIS OPTIONS:"
    echo "  --modality <name>       vbm (default), thickness, depth, gyrification, fractal"
    echo "  --smoothing <mm>        Smoothing kernel (default: auto-detect)"
    echo "  --analysis-name <name>  Custom analysis name (default: auto-generated)"
    echo "  --group-col <name>      Group column in participants.tsv (auto-detect if omitted)"
    echo "  --covariates <list>     Covariates: age,sex,tiv (optional)"
    echo ""
    echo "TFCE OPTIONS:"
    echo "  --n-perm <N>            TFCE permutations (default: 5000)"
    echo "  --pilot                 Quick test mode (100 permutations)"
    echo "  --skip-screening        Run TFCE on all contrasts (not recommended)"
    echo ""
    echo "SCREENING OPTIONS:"
    echo "  --uncorrected-p <p>     P-value threshold (default: 0.001)"
    echo "  --cluster-size <k>      Minimum cluster size (default: 50 voxels)"
    echo ""
    echo "OTHER OPTIONS:"
    echo "  --force                 Delete existing results before starting"
    echo "  --n-jobs <N>            Parallel jobs for TFCE (default: 4)"
    echo "  --help, -h              Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo ""
    echo "  # Basic VBM analysis"
    echo "  $0 --cat12-dir /data/cat12 --participants participants.tsv"
    echo ""
    echo "  # Quick test"
    echo "  $0 --cat12-dir /data/cat12 --participants participants.tsv --pilot"
    echo ""
    echo "  # With covariates"
    echo "  $0 --cat12-dir /data/cat12 --participants participants.tsv \\"
    echo "     --covariates \"age,sex,tiv\""
    echo ""
    echo "  # Cortical thickness"
    echo "  $0 --cat12-dir /data/cat12 --participants participants.tsv \\"
    echo "     --modality thickness"
    echo ""
    echo "CONFIGURATION:"
    echo "  Edit config.ini to customize defaults for:"
    echo "    - MATLAB and SPM paths"
    echo "    - Analysis parameters (n_perm, uncorrected_p, cluster_size)"
    echo "    - Performance settings (parallel_jobs)"
    echo ""
    echo "  Command-line arguments override config.ini values."
    echo ""
    echo "RESULTS:"
    echo "  Saved to: results/<modality>/<analysis_name>/"
    echo "  - report.html          Interactive analysis report"
    echo "  - spm_batch.m          SPM batch file (for reproducibility)"
    echo "  - SPM.mat              Statistical model"
    echo "  - TFCE_*_fwe.nii       FWE-corrected results"
    echo ""
    echo "════════════════════════════════════════════════════════════════════════"
    echo ""
    exit 0
fi

# ============================================================================
# Parse Arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --cat12-dir)
            CAT12_DIR="$2"
            shift 2
            ;;
        --participants)
            PARTICIPANTS_FILE="$2"
            shift 2
            ;;
        --modality)
            MODALITY="$2"
            shift 2
            ;;
        --smoothing)
            SMOOTHING="$2"
            shift 2
            ;;
        --analysis-name)
            ANALYSIS_NAME="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --group-col)
            GROUP_COL="$2"
            shift 2
            ;;
        --session-col)
            SESSION_COL="$2"
            shift 2
            ;;
        --sessions)
            SESSIONS="$2"
            shift 2
            ;;
        --covariates)
            COVARIATES="$2"
            shift 2
            ;;
        --n-perm|--nperms)
            N_PERM="$2"
            shift 2
            ;;
        --pilot)
            PILOT_MODE=true
            N_PERM=100
            shift
            ;;
        --skip-screening)
            SKIP_SCREENING=true
            shift
            ;;
        --cluster-size)
            CLUSTER_SIZE="$2"
            shift 2
            ;;
        --uncorrected-p)
            UNCORRECTED_P="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --n-jobs)
            N_JOBS="$2"
            shift 2
            ;;
        --help|-h)
            # Print the top-of-file help header
            grep "^#" "$0" | tail -n +3

            # Also extract CLI flags declared in utility scripts under utils/
            echo ""
            echo "────────────────────────────────────────────────────────────────────────" 
            echo "Additional flags exposed by helper scripts in ./utils/ (extracted):"
            echo "(Showing raw add_argument(...) entries from each utils/*.py file)"
            echo ""
            for f in "$STATS_DIR"/utils/*.py; do
                if [[ -f "$f" ]]; then
                    echo "== $(basename "$f") =="
                    # print add_argument contents, one-per-line (safe text extraction)
                    grep -E "add_argument\(" "$f" 2>/dev/null | sed -E "s/.*add_argument\(([^)]*)\).*/  \1/" | sed -E "s/^[[:space:]]*//;s/[[:space:]]+/ /g" || true
                    echo ""
                fi
            done
            exit 0
            ;;
        *)
            echo "Error: Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Validation
# ============================================================================

if [[ -z "$CAT12_DIR" ]] || [[ -z "$PARTICIPANTS_FILE" ]]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 --cat12-dir <path> --participants <tsv>"
    echo "Run: $0 --help   for full help"
    exit 1
fi

if [[ ! -d "$CAT12_DIR" ]]; then
    echo "Error: CAT12 directory not found: $CAT12_DIR"
    exit 1
fi

if [[ ! -f "$PARTICIPANTS_FILE" ]]; then
    echo "Error: Participants file not found: $PARTICIPANTS_FILE"
    exit 1
fi

# Make paths absolute
CAT12_DIR="$(cd "$CAT12_DIR" && pwd)"
PARTICIPANTS_FILE="$(cd "$(dirname "$PARTICIPANTS_FILE")" && pwd)/$(basename "$PARTICIPANTS_FILE")"

# Auto-detect smoothing if not specified
if [[ -z "$SMOOTHING" ]]; then
    # Find one representative mwp1r file under the supplied CAT12 dir and
    # try to extract the smoothing kernel (e.g. 's6mwp1r' -> 6). If no
    # smoothing prefix is present, fall back to default 6 mm.
    FOUND_FILE=$(find "$CAT12_DIR" -type f -iname "*mwp1r*.nii*" 2>/dev/null | head -n 1 || true)
    if [[ -n "$FOUND_FILE" ]]; then
        basefn=$(basename "$FOUND_FILE")
        if [[ "$basefn" =~ s([0-9]+)mwp1r ]]; then
            SMOOTHING="${BASH_REMATCH[1]}"
        else
            # No explicit smoothing prefix found in filename; default to 6
            SMOOTHING="6"
        fi
    else
        SMOOTHING="6"
    fi
fi

# Set default analysis name if not provided
if [[ -z "$ANALYSIS_NAME" ]]; then
    ANALYSIS_NAME="${MODALITY}_smooth_auto"
fi

# Set results directory
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$STATS_DIR/results/${MODALITY}/${ANALYSIS_NAME}"
else
    OUTPUT_DIR="$(cd "$(dirname "$OUTPUT_DIR")" && pwd)/$(basename "$OUTPUT_DIR")"
fi

# Ensure output dir exists early so we can capture logs there
mkdir -p "$OUTPUT_DIR"

# Start capturing terminal output to a pipeline log inside the results folder
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline.log"
# tee all stdout/stderr to the log file (append)
exec > >(tee -a "$LOG_FILE") 2>&1

# Create temp directory
TEMP_DIR=$(mktemp -d "$STATS_DIR/.tmp_${ANALYSIS_NAME}_XXXXXX")
trap "rm -rf '$TEMP_DIR'" EXIT

# ============================================================================
# Banner
# ============================================================================

cat << 'EOF'

╔════════════════════════════════════════════════════════════════════════╗
║              CAT12 Longitudinal Analysis Pipeline                      ║
╚════════════════════════════════════════════════════════════════════════╝

EOF

echo "Configuration:"
echo "  CAT12 directory:    $CAT12_DIR"
echo "  Participants file:  $PARTICIPANTS_FILE"
echo "  Modality:           $MODALITY"
echo "  Smoothing:          ${SMOOTHING}mm"
echo "  Analysis name:      $ANALYSIS_NAME"
echo "  Results directory:  $OUTPUT_DIR"
echo ""
echo "Design:"
echo "  Group column:       ${GROUP_COL:-auto-detect}"
echo "  Session column:     $SESSION_COL"
echo "  Sessions:           $SESSIONS"
echo "  Covariates:         ${COVARIATES:-none}"
echo ""
echo "TFCE:"
echo "  Permutations:       $N_PERM"
echo "  Pilot mode:         $PILOT_MODE"
echo "  Skip screening:     $SKIP_SCREENING"
echo "  Parallel jobs:      $N_JOBS"
echo ""
echo "Options:"
echo "  Force clean:        $FORCE"
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# One-time SPM configuration step
# Run configure_spm_path once early so subsequent MATLAB calls don't re-run the
# interactive/config detection tool and clutter the logs.
# ============================================================================
echo "Checking SPM configuration (one-time)..."
MATLAB_SPM_LOG="$LOG_DIR/matlab_configure_spm.log"
"$MATLAB_EXE" $MATLAB_FLAGS "addpath('$STATS_DIR/utils'); try, configure_spm_path; catch e, fprintf('Warning: configure_spm_path failed: %s\n', e.message); end; exit;" 2>&1 | tee -a "$MATLAB_SPM_LOG" || {
    echo "Warning: one-time SPM configuration step failed (see $MATLAB_SPM_LOG). Continuing, but later MATLAB calls may need SPM path set."
}
echo ""

# ============================================================================
# Step 0: Clean existing results if --force
#
# When --force is provided we remove the full results directory and any
# temporary directories left from previous runs. For safety we only allow a
# full recursive removal when the target is under "$STATS_DIR/results".
# If the output directory is outside that location we remove only its
# contents to avoid accidental deletion of unrelated paths.
# ============================================================================

if [[ "$FORCE" == true ]]; then
    if [[ -d "$OUTPUT_DIR" ]]; then
        # Safety: allow full rm -rf only for expected results locations
        case "$OUTPUT_DIR" in
            "$STATS_DIR"/results/*)
                echo "Removing entire results directory: $OUTPUT_DIR"
                rm -rf "$OUTPUT_DIR"
                echo "✓ Removed $OUTPUT_DIR"
                ;;
            *)
                echo "Warning: OUTPUT_DIR ($OUTPUT_DIR) is outside expected results path."
                echo "Removing contents of $OUTPUT_DIR instead of the whole directory."
                rm -rf "$OUTPUT_DIR"/*
                echo "✓ Cleaned contents of $OUTPUT_DIR"
                ;;
        esac
    else
        echo "No existing results directory to remove: $OUTPUT_DIR"
    fi

    # Remove any stale temporary directories for this analysis
    TMP_PATTERN="$STATS_DIR/.tmp_${ANALYSIS_NAME}_*"
    shopt -s nullglob
    tmpdirs=( $TMP_PATTERN )
    for d in "${tmpdirs[@]:-}"; do
        echo "Removing temp directory: $d"
        rm -rf "$d"
    done
    shopt -u nullglob

    echo ""
fi

# ============================================================================
# Step 1: Parse Participants & Design
# ============================================================================

echo "┌────────────────────────────────────────────────────────────────────────┐"
echo "│ STEP 0: PREFLIGHT CHECKS (Python packages, CAT12 files, participants)  │"
echo "└────────────────────────────────────────────────────────────────────────┘"
echo ""

python3 "$STATS_DIR/utils/preflight_check.py" --cat12-dir "$CAT12_DIR" --participants "$PARTICIPANTS_FILE" --smoothing "$SMOOTHING" || {
    echo "Error: Preflight checks failed. Fix issues above and re-run."
    exit 1
}


echo "┌────────────────────────────────────────────────────────────────────────┐"
echo "│ STEP 1: Parsing Participants File                                     │"
echo "└────────────────────────────────────────────────────────────────────────┘"
echo ""

python3 "$STATS_DIR/utils/parse_participants.py" \
    --cat12-dir "$CAT12_DIR" \
    --participants "$PARTICIPANTS_FILE" \
    --modality "$MODALITY" \
    --smoothing "$SMOOTHING" \
    --output "$TEMP_DIR" \
    ${GROUP_COL:+--group-col "$GROUP_COL"} \
    --session-col "$SESSION_COL" \
    --sessions "$SESSIONS" \
    ${COVARIATES:+--covariates "$COVARIATES"} || {
        echo "Error: Failed to parse participants file"
        exit 1
    }

echo ""

# Persist design.json into the results folder so reports can be generated even
# if temporary directories are cleaned up (helps when --force is used).
if [[ -f "$TEMP_DIR/design.json" ]]; then
    mkdir -p "$OUTPUT_DIR"
    cp "$TEMP_DIR/design.json" "$OUTPUT_DIR/design.json"
    echo "Design JSON copied to: $OUTPUT_DIR/design.json"
fi

# ============================================================================
# Step 2a: Explicit mask handling (use canonical repo template mask)
# ============================================================================

# We no longer generate per-results VBM masks (mask_vbm.nii). Instead the
# pipeline prefers the repo-level canonical CAT12 tight brainmask located at
# $STATS_DIR/templates/brainmask_GMtight.nii when present. This keeps masking
# consistent across analyses.

MASK_FILE=""
# Determine the template/GM mask to use. Allow override from config.ini via
# MASKING.gm_mask (can be an absolute path or relative to the repo root).
GM_MASK_CONFIG=$(get_ini_value "MASKING" "gm_mask" "")
if [[ -n "$GM_MASK_CONFIG" ]]; then
    if [[ "$GM_MASK_CONFIG" = /* ]]; then
        TEMPLATE_MASK="$GM_MASK_CONFIG"
    else
        TEMPLATE_MASK="$STATS_DIR/$GM_MASK_CONFIG"
    fi
else
    TEMPLATE_MASK="$STATS_DIR/templates/brainmask_GMtight.nii"
fi

if [[ -f "$TEMPLATE_MASK" ]]; then
    echo "Using GM mask: $TEMPLATE_MASK"
    MASK_FILE="$TEMPLATE_MASK"
else
    echo "No GM mask found at $TEMPLATE_MASK — running without an explicit mask"
    MASK_FILE=""
fi


# ============================================================================
# Step 2b: Generate SPM Batch File
# ============================================================================

echo "┌────────────────────────────────────────────────────────────────────────┐"
echo "│ STEP 2b: Generating SPM Factorial Design                              │"
echo "└────────────────────────────────────────────────────────────────────────┘"
echo ""

MASK_ARG=""
if [[ -n "$MASK_FILE" ]]; then
    MASK_ARG="--mask-file $MASK_FILE"
fi

python3 "$STATS_DIR/utils/generate_spm_batch.py" \
    --design-file "$TEMP_DIR/design.json" \
    --output-dir "$OUTPUT_DIR" \
    --modality "$MODALITY" \
    --output "$TEMP_DIR/spm_batch.m" \
    $MASK_ARG || {
        echo "Error: Failed to generate SPM batch"
        exit 1
    }

# Copy batch file to output directory for reproducibility
cp "$TEMP_DIR/spm_batch.m" "$OUTPUT_DIR/spm_batch.m"
echo "✓ SPM batch file generated and saved to: $OUTPUT_DIR/spm_batch.m"
echo ""

# ============================================================================
# Step 3: Run Model Estimation
# ============================================================================

echo "┌────────────────────────────────────────────────────────────────────────┐"
echo "│ STEP 3: SPM Model Estimation                                          │"
echo "└────────────────────────────────────────────────────────────────────────┘"
echo ""

mkdir -p "$OUTPUT_DIR"

# Delete any existing SPM.mat to avoid "Overwrite?" dialog in headless mode
if [[ -f "$OUTPUT_DIR/SPM.mat" ]]; then
    rm -f "$OUTPUT_DIR/SPM.mat"
    echo "Removed existing SPM.mat to avoid overwrite dialog"
fi

# Ensure logs directory exists
mkdir -p "$LOG_DIR"

MATLAB_MODEL_LOG="$LOG_DIR/matlab_model_estimation.log"
"$MATLAB_EXE" $MATLAB_FLAGS "warning('off','MATLAB:dispatcher:nameConflict'); warning('off','all'); set(0,'DefaultFigureVisible','off'); set(0,'DefaultFigureCreateFcn',@(h,ev)[]); addpath('$STATS_DIR/utils'); spm('defaults', 'FMRI'); spm_jobman('initcfg'); fprintf('═══════════════════════════════════════════════════════\n'); fprintf('Running Factorial Design Specification\n'); fprintf('═══════════════════════════════════════════════════════\n\n'); run('$TEMP_DIR/spm_batch.m'); try, spm_jobman('run', matlabbatch); catch e, fprintf('Warning: Design reporting failed (expected in headless mode):\n%s\n', e.message); end; clear matlabbatch; fprintf('\n═══════════════════════════════════════════════════════\n'); fprintf('Running Model Estimation\n'); fprintf('═══════════════════════════════════════════════════════\n\n'); matlabbatch{1}.spm.stats.fmri_est.spmmat = {'$OUTPUT_DIR/SPM.mat'}; matlabbatch{1}.spm.stats.fmri_est.write_residuals = 0; matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1; spm_jobman('run', matlabbatch); fprintf('\n✓ Model estimation complete\n\n'); exit;" 2>&1 | tee -a "$MATLAB_MODEL_LOG" || {
        echo "Error: Model estimation failed"
        echo "Check MATLAB log: $LOG_DIR/matlab_model_estimation.log"
        exit 1
    }

echo "✓ Model estimation complete"
echo ""

# ============================================================================
# Step 3b: Check for missing voxels across images (optional diagnostic)
# This helps detect voxels with many NaNs or missing data that can break
# permutation schemes. Writes summary JSON and an exclusion mask PNG/NIfTI.
# ============================================================================

echo "Running missing-voxel diagnostics (this is fast)"
# Read optional failure threshold from config.ini (empty disables failure)
MISSING_FAIL_PCT=$(get_ini_value "TFCE" "missing_fail_pct" "")
GM_MASK_ARG=""
if [[ -n "$MASK_FILE" ]]; then
    GM_MASK_ARG="--gm-mask $MASK_FILE"
fi
if [[ -n "$MISSING_FAIL_PCT" && "$MISSING_FAIL_PCT" != "false" ]]; then
    python3 "$STATS_DIR/utils/check_missing_voxels.py" --spm "$OUTPUT_DIR/SPM.mat" --output-dir "$OUTPUT_DIR" --threshold 0.05 --fail-if-pct-excluded "$MISSING_FAIL_PCT" || {
        echo "❌ Missing-voxel fraction exceeded ${MISSING_FAIL_PCT}% — aborting pipeline."
        exit 1
    }
else
    python3 "$STATS_DIR/utils/check_missing_voxels.py" --spm "$OUTPUT_DIR/SPM.mat" --output-dir "$OUTPUT_DIR" --threshold 0.05 $GM_MASK_ARG || {
        echo "⚠️  Warning: missing-voxel diagnostic failed (see script output above). Continuing analysis."
    }
fi


# ============================================================================
# Step 4: Add Contrasts
# ============================================================================

echo "┌────────────────────────────────────────────────────────────────────────┐"
echo "│ STEP 4: Adding Contrasts                                              │"
echo "└────────────────────────────────────────────────────────────────────────┘"
echo ""

# Ensure logs directory exists for this step
mkdir -p "$LOG_DIR"

MATLAB_CONTRAST_LOG="$LOG_DIR/matlab_contrasts.log"
"$MATLAB_EXE" $MATLAB_FLAGS "warning('off','MATLAB:dispatcher:nameConflict'); warning('off','all'); addpath('$STATS_DIR/utils'); spm('defaults', 'FMRI'); spm_jobman('initcfg'); try, add_contrasts_longitudinal('$OUTPUT_DIR'); catch e, fprintf('ERROR in add_contrasts_longitudinal:\n%s\n', e.message); end; exit;" 2>&1 | tee -a "$MATLAB_CONTRAST_LOG" || {
        echo "Error: Adding contrasts failed"
        echo "Check MATLAB log: $LOG_DIR/matlab_contrasts.log"
        if [[ -f "$LOG_DIR/matlab_contrasts.log" ]]; then
            echo ""
            echo "Last lines of MATLAB log:"
            tail -20 "$LOG_DIR/matlab_contrasts.log"
        fi
        exit 1
    }

echo "✓ Contrasts added"
echo ""

# Verify contrasts were written to disk. If none found, fail early with diagnostics.
echo "Verifying contrast files written to: $OUTPUT_DIR"
shopt -s nullglob
cons=( "$OUTPUT_DIR"/con_*.nii )
spmTs=( "$OUTPUT_DIR"/spmT_*.nii )
spmFs=( "$OUTPUT_DIR"/spmF_*.nii )
if [[ ${#cons[@]} -eq 0 && ${#spmTs[@]} -eq 0 && ${#spmFs[@]} -eq 0 ]]; then
    echo "ERROR: No contrast or statistic files found in $OUTPUT_DIR after adding contrasts."
    echo "Contents of results folder:";
    ls -al "$OUTPUT_DIR" || true
    echo "Check MATLAB console output above for errors during contrast creation."
    exit 1
else
    echo "Found ${#cons[@]} contrast files and ${#spmTs[@]} spmT files and ${#spmFs[@]} spmF files"
fi
shopt -u nullglob

# Generate design matrix visualization
echo "Generating design matrix image..."
"$MATLAB_EXE" $MATLAB_FLAGS "\
    % Suppress warnings and GUI creation\n    warning('off','all'); set(0,'DefaultFigureVisible','off'); set(0,'DefaultFigureCreateFcn',@(h,ev)[]); beep off; \
    addpath('$STATS_DIR/utils'); \
    generate_design_matrix_image('$OUTPUT_DIR/SPM.mat', '$OUTPUT_DIR/design_matrix.png'); \
    exit;" || {
        echo "⚠️  Warning: Design matrix image generation failed"
    }

echo ""

# ============================================================================
# Step 5: Screen Contrasts
# ============================================================================

if [[ "$SKIP_SCREENING" == false ]]; then
    echo "┌────────────────────────────────────────────────────────────────────────┐"
    echo "│ STEP 5: Screening Contrasts (p<$UNCORRECTED_P uncorrected)                     │"
    echo "└────────────────────────────────────────────────────────────────────────┘"
    echo ""
    
    "$MATLAB_EXE" $MATLAB_FLAGS "warning('off','MATLAB:dispatcher:nameConflict'); warning('off','all'); set(0,'DefaultFigureVisible','off'); set(0,'DefaultFigureCreateFcn',@(h,ev)[]); addpath('$STATS_DIR/utils'); spm('defaults', 'FMRI'); spm_jobman('initcfg'); significant_contrasts = screen_contrasts('$OUTPUT_DIR', 'p_thresh', $UNCORRECTED_P, 'cluster_size', $CLUSTER_SIZE); fprintf('\n✓ Screening complete with %d significant contrasts\n\n', length(significant_contrasts)); exit;" || {
            echo "Error: Screening failed"
            exit 1
        }
    
    echo "✓ Screening complete"
    echo ""
else
    echo "┌────────────────────────────────────────────────────────────────────────┐"
    echo "│ STEP 5: Skipped (running TFCE on all contrasts)                       │"
    echo "└────────────────────────────────────────────────────────────────────────┘"
    echo ""
fi

# ============================================================================
# Step 6: TFCE Correction
# ============================================================================

mkdir -p "$LOG_DIR"

TFCE_LOG="$LOG_DIR/matlab_tfce.log"

echo "┌────────────────────────────────────────────────────────────────────────┐"
echo "│ STEP 6: TFCE Permutation Testing                                      │"
echo "└────────────────────────────────────────────────────────────────────────┘"
echo ""

"$MATLAB_EXE" $MATLAB_FLAGS "warning('off','MATLAB:dispatcher:nameConflict'); warning('off','all'); set(0,'DefaultFigureVisible','off'); set(0,'DefaultFigureCreateFcn',@(h,ev)[]); addpath('$STATS_DIR/utils'); spm('defaults', 'FMRI'); spm_jobman('initcfg'); fprintf('Starting TFCE with %d permutations\n', $N_PERM); run_tfce_correction('$OUTPUT_DIR', 'n_perm', $N_PERM, 'n_jobs', $N_JOBS); exit;" 2>&1 | tee -a "$TFCE_LOG" || {
        echo "Error: TFCE correction failed"
        exit 1
    }

echo "✓ TFCE correction complete"
echo ""

# ============================================================================
# Step 6b: Generate TFCE Summary
# ============================================================================

echo "Generating TFCE results summary..."
python3 "$STATS_DIR/utils/generate_tfce_images.py" \
    --output-dir "$OUTPUT_DIR" \
    --fwe-threshold 0.05 || {
        echo "⚠️  Warning: TFCE summary generation failed"
    }

echo ""

# ============================================================================
# Step 7: Generate HTML Report
# ============================================================================

echo "┌────────────────────────────────────────────────────────────────────────┐"
echo "│ STEP 7: Generating HTML Report                                        │"
echo "└────────────────────────────────────────────────────────────────────────┘"
echo ""

# Provide number of contrasts to the report generator (count con_*.nii)
N_CONTRASTS=$(ls -1 "$OUTPUT_DIR"/con_*.nii 2>/dev/null | wc -l)

# Build a safely-quoted command-line string from the original args. Use
# printf '%q' so special characters and quotes are escaped and the result
# is safe to pass as a single argument to Python.
SAFE_CMDLINE="$(printf '%q ' "$0" "${ORIGINAL_ARGS[@]}")"

python3 "$STATS_DIR/utils/generate_html_report.py" \
    --design-json "$TEMP_DIR/design.json" \
    --output "$OUTPUT_DIR/report.html" \
    --analysis-name "$ANALYSIS_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --command-line "$SAFE_CMDLINE" \
    --n-contrasts "$N_CONTRASTS" || {
        echo "⚠️  Warning: HTML report generation failed"
    }

# Create symlink in script directory for quick access
if [[ -f "$OUTPUT_DIR/report.html" ]]; then
    LINK_NAME="$STATS_DIR/report_latest.html"
    rm -f "$LINK_NAME"
    ln -s "$OUTPUT_DIR/report.html" "$LINK_NAME"
    echo "✓ Quick link created: $LINK_NAME -> $OUTPUT_DIR/report.html"
fi

echo ""

# ============================================================================
# Completion
# ============================================================================

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                     ✓ ANALYSIS COMPLETE                                ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

# Run verification checks on the output
"$STATS_DIR/utils/verify_analysis_output.sh" "$OUTPUT_DIR"
VERIFY_RESULT=$?

echo ""
echo "Results saved to:"
echo "  $OUTPUT_DIR"
echo ""
echo "Key output files:"
echo "  - report.html                    Analysis report (open in browser)"
echo "  - spm_batch.m                    SPM batch file (for reproducibility)"
echo "  - SPM.mat                        Statistical model"
echo "  - beta_*.nii                     Parameter estimates"
echo "  - con_*.nii, spmT_*.nii          Contrast maps"
echo "  - design_matrix.png              Design visualization"
echo "  - screening_results.mat          Contrast screening results"
echo "  - tfce_*_fwe.nii                 TFCE-corrected maps"
echo ""
echo "Quick access:"
echo "  report_latest.html → $OUTPUT_DIR/report.html"
echo ""
