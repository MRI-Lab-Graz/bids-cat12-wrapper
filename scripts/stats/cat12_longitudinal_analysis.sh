#!/bin/bash
#
# CAT12 Longitudinal Analysis Pipeline
# =====================================
#
# Complete automated workflow from CAT12 preprocessing to TFCE-corrected results.
# Production Ready - Reviewed 2025-11-19
#
# USAGE:
#   ./cat12_longitudinal_analysis.sh --config <json> --cat12-dir <path> --participants <tsv> [options]
#
# REQUIRED ARGUMENTS:
#   --config <json>         Path to config.json file (required)
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
# Behavior note: the pipeline now runs TFCE in an automatic two-stage
# probe-then-full strategy by default (a short probe run is performed to
# inspect the permutation diagnostic `cc` and the full run will switch to
# Freedman–Lane nuisance handling if the probe indicates instability). No
# additional CLI flag is required to enable this behavior.
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

# ============================================================================
# Logging Functions
# ============================================================================

# ANSI color codes
COLOR_RESET='\033[0m'
COLOR_INFO='\033[0;36m'      # Cyan
COLOR_WARNING='\033[0;33m'   # Yellow
COLOR_ERROR='\033[0;31m'     # Red
COLOR_SUCCESS='\033[0;32m'   # Green
COLOR_DEBUG='\033[0;90m'     # Gray

# Indent external tool output (e.g., SPM/MATLAB) for readability
external_prefix() {
    sed 's/^/    /'
}

# Logging functions with timestamp and color
log_info() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} - ${COLOR_INFO}INFO${COLOR_RESET} - $*"
}

log_warning() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} - ${COLOR_WARNING}WARNING${COLOR_RESET} - $*"
}

log_error() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} - ${COLOR_ERROR}ERROR${COLOR_RESET} - $*" >&2
}

log_success() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} - ${COLOR_SUCCESS}SUCCESS${COLOR_RESET} - $*"
}

log_debug() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} - ${COLOR_DEBUG}DEBUG${COLOR_RESET} - $*"
}

# Capture original arguments early so we can safely reconstruct the exact
# command line later (avoids issues with unmatched quotes when we pass the
# command line into reports). Store as array to preserve spacing and quoting.
ORIGINAL_ARGS=("$@")

# Capture pipeline start time for filtering old results
PIPELINE_START_TIME=$(date +%s)

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
STATS_DIR="$ROOT_DIR"
UTILS_DIR="$ROOT_DIR/scripts/utils"
CONFIG_JSON=""

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

get_json_value() {
    local key="$1"
    local default="$2"

    if [[ ! -f "$CONFIG_JSON" ]]; then
        echo "$default"
        return
    fi

    "$PYTHON_EXE" - <<PY 2>/dev/null || echo "$default"
import json
import sys

path = "$CONFIG_JSON"
key = "$key"
default = "$default"

try:
    with open(path, "r") as f:
        data = json.load(f)
    value = data
    for part in key.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            value = default
            break
    if value is None:
        value = default
    if isinstance(value, (dict, list)):
        print(default)
    elif isinstance(value, bool):
        # Convert Python bool to lowercase string for bash
        print(str(value).lower())
    else:
        print(value)
except Exception:
    print(default)
PY
}

# Load configuration defaults from config.ini
MATLAB_EXE=$(get_ini_value "MATLAB" "exe" "/Applications/MATLAB_R2025b.app/bin/matlab")
SPM_PATH=$(get_ini_value "SPM" "path" "")
PYTHON_EXE=$(get_ini_value "PYTHON" "exe" "python3")

# Allow graphics windows in MATLAB? If true we omit -nodisplay. If false we add -nodisplay
MATLAB_ALLOW_GRAPHICS=$(get_ini_value "MATLAB" "allow_graphics" "true")

# Initialize variables that will be populated from config.json AFTER argument parsing
# (see "Read configuration from config.json" section after CONFIG_JSON validation)
MODALITY=""
SMOOTHING=""
GROUP_COL=""
SESSION_COL=""
SESSIONS=""
COVARIATES=""
STANDARDIZE_CONTINUOUS=""
UNCORRECTED_P=""
CLUSTER_SIZE=""
SKIP_SCREENING=""
N_PERM=""
PILOT_MODE=""
INITIAL_PERM=""
CC_THRESHOLD=""
N_JOBS=""
OUTPUT_DIR=""
ANALYSIS_NAME=""
FORCE=""
PARTICIPANTS_FILE=""
MATLAB_EXE=""
NO_TFCE=false

# Auto-detect MATLAB if empty in config
if [[ -z "$MATLAB_EXE" ]] || [[ "$MATLAB_EXE" == "false" ]]; then
    FOUND_MATLAB=$(find /Applications -name "MATLAB_R*.app" -maxdepth 1 2>/dev/null | sort -r | head -1)
    if [[ -n "$FOUND_MATLAB" ]]; then
        MATLAB_EXE="$FOUND_MATLAB/bin/matlab"
    else
        MATLAB_EXE="matlab"
    fi
fi

# Check for Python 3
if ! command -v "$PYTHON_EXE" &> /dev/null; then
    log_error "Python executable '$PYTHON_EXE' not found."
    echo "Please install Python 3 or update [PYTHON] exe in config.ini."
    exit 1
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
DESIGN_FILE=""

# Show help if no arguments provided
if [[ $# -eq 0 ]]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║              CAT12 Longitudinal Analysis Pipeline                      ║"
    echo "╚════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "USAGE:"
    echo "  $0 --config <json> --cat12-dir <path> --participants <tsv> [options]"
    echo ""
    echo "REQUIRED ARGUMENTS:"
    echo "  --config <json>         Path to config.json file"
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
    echo "  --no-tfce               Stop after screening (skip TFCE correction)"
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
    echo "  $0 --config config/config.json --cat12-dir /data/cat12 --participants participants.tsv"
    echo ""
    echo "  # Quick test"
    echo "  $0 --config config/config.json --cat12-dir /data/cat12 --participants participants.tsv --pilot"
    echo ""
    echo "  # With covariates"
    echo "  $0 --config config/config.json --cat12-dir /data/cat12 --participants participants.tsv \\"
    echo "     --covariates \"age,sex,tiv\""
    echo ""
    echo "  # Cortical thickness"
    echo "  $0 --config config/config.json --cat12-dir /data/cat12 --participants participants.tsv \\"
    echo "     --modality thickness"
    echo ""
    echo "CONFIGURATION:"
    echo "  Edit config.json to customize defaults for:"
    echo "    - MATLAB and SPM paths"
    echo "    - Analysis parameters (n_perm, uncorrected_p, cluster_size)"
    echo "    - Performance settings (parallel_jobs)"
    echo ""
    echo "  Command-line arguments override config.json values."
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
        --design)
            DESIGN_FILE="$2"
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
        --no-tfce)
            NO_TFCE=true
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
            echo "Additional flags exposed by helper scripts in ./scripts/utils/ (extracted):"
            echo "(Showing raw add_argument(...) entries from each utils/*.py file)"
            echo ""
            for f in "$UTILS_DIR"/*.py; do
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
            log_error "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$CONFIG_JSON" ]]; then
    echo "Error: --config <json> is required."
    echo "Usage: $0 --config <json> --cat12-dir <path> --participants <tsv> [options]"
    exit 1
fi

if [[ ! -f "$CONFIG_JSON" ]]; then
    log_error "config file not found: $CONFIG_JSON"
    exit 1
fi

# ============================================================================
# Read configuration from config.json (now that it's validated)
# ============================================================================

# Modality and analysis settings
if [[ -z "$MODALITY" ]]; then
    MODALITY=$(get_json_value "analysis.modalities[0].name" "vbm")
fi
if [[ -z "$SMOOTHING" ]]; then
    SMOOTHING=$(get_json_value "analysis.modalities[0].smoothing_kernel" "")
fi
if [[ -z "$GROUP_COL" ]]; then
    GROUP_COL=$(get_json_value "analysis.group_column" "")
fi
if [[ -z "$SESSION_COL" ]]; then
    SESSION_COL=$(get_json_value "analysis.session_column" "session")
fi
if [[ -z "$SESSIONS" ]]; then
    # Read sessions from config - converts JSON array ["1", "2"] to "1,2"
    SESSIONS=$(python3 << PYEOF
import json
try:
    with open("$CONFIG_JSON") as f:
        config = json.load(f)
    sessions = config.get("analysis", {}).get("sessions", ["all"])
    if sessions == ["all"]:
        print("all")
    else:
        print(",".join(str(s) for s in sessions))
except Exception:
    print("all")
PYEOF
)
    echo "DEBUG: Sessions from config = '$SESSIONS'"
fi

# Extract covariates from the modality config (if not already set by --covariates)
if [[ -z "$COVARIATES" ]]; then
    # Get covariates array from the matching modality in config.json
    COVARIATES=$(python3 << PYEOF
import json
try:
    with open("$CONFIG_JSON") as f:
        config = json.load(f)
    # Find the matching modality
    modality_name = "$MODALITY"
    for mod in config.get("analysis", {}).get("modalities", []):
        if mod.get("name") == modality_name:
            covs = mod.get("covariates", [])
            if covs:
                # Convert list to comma-separated string
                print(",".join(covs))
            break
except Exception:
    pass
PYEOF
)
    echo "DEBUG: Covariates from modality config = '$COVARIATES'"
fi
if [[ -z "$STANDARDIZE_CONTINUOUS" ]]; then
    STANDARDIZE_CONTINUOUS=$(get_json_value "analysis.standardize_continuous" "true")
fi

# Screening settings
if [[ -z "$UNCORRECTED_P" ]]; then
    UNCORRECTED_P=$(get_json_value "screening.uncorrected_p" "0.001")
fi
if [[ -z "$CLUSTER_SIZE" ]]; then
    CLUSTER_SIZE=$(get_json_value "screening.cluster_size_voxels" "10")
fi
if [[ -z "$SKIP_SCREENING" ]]; then
    SKIP_SCREENING=$(get_json_value "screening.skip_screening" "false")
fi

# TFCE settings
N_PERM=$(get_json_value "tfce.n_permutations" "5000")
PILOT_MODE=$(get_json_value "tfce.pilot_mode" "false")

# Debug: show what was read
echo "DEBUG: PILOT_MODE from config = '$PILOT_MODE'"

# If pilot mode is enabled, override N_PERM to 100
if [[ "$PILOT_MODE" == "true" ]]; then
    N_PERM=100
    echo "DEBUG: Pilot mode enabled, N_PERM set to 100"
else
    echo "DEBUG: Pilot mode disabled, N_PERM = $N_PERM"
fi

INITIAL_PERM=100
CC_THRESHOLD=0.98

# Performance settings
if [[ -z "$N_JOBS" ]]; then
    N_JOBS=$(get_json_value "performance.parallel_jobs" "1")
fi

# Output settings
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR=$(get_json_value "output.output_dir" "")
fi
if [[ -z "$ANALYSIS_NAME" ]]; then
    ANALYSIS_NAME=$(get_json_value "output.analysis_name" "")
fi
if [[ -z "$FORCE" ]]; then
    FORCE=$(get_json_value "output.force_clean" "false")
fi

if [[ -z "$PARTICIPANTS_FILE" ]]; then
    PARTICIPANTS_FILE=$(get_json_value "analysis.participants_file" "")
fi

# ============================================================================
# Validation
# ============================================================================

if [[ -n "$DESIGN_FILE" ]]; then
    if [[ ! -f "$DESIGN_FILE" ]]; then
        log_error "Design file not found: $DESIGN_FILE"
        exit 1
    fi
else
    if [[ -z "$CAT12_DIR" ]] || [[ -z "$PARTICIPANTS_FILE" ]]; then
        echo "Error: Missing required arguments"
        echo "Usage: $0 --config <json> --cat12-dir <path> --participants <tsv>"
        echo "   OR: $0 --design <json_file>"
        echo "Run: $0 --help   for full help"
        exit 1
    fi
fi

if [[ -n "$CAT12_DIR" ]] && [[ ! -d "$CAT12_DIR" ]]; then
    echo "Error: CAT12 directory not found: $CAT12_DIR"
    exit 1
fi

if [[ -n "$PARTICIPANTS_FILE" ]] && [[ ! -f "$PARTICIPANTS_FILE" ]]; then
    echo "Error: Participants file not found: $PARTICIPANTS_FILE"
    exit 1
fi

# Make paths absolute
if [[ -n "$CAT12_DIR" ]]; then
    CAT12_DIR="$(cd "$CAT12_DIR" && pwd)"
fi
if [[ -n "$PARTICIPANTS_FILE" ]]; then
    PARTICIPANTS_FILE="$(cd "$(dirname "$PARTICIPANTS_FILE")" && pwd)/$(basename "$PARTICIPANTS_FILE")"
fi
if [[ -n "$DESIGN_FILE" ]]; then
    DESIGN_FILE="$(cd "$(dirname "$DESIGN_FILE")" && pwd)/$(basename "$DESIGN_FILE")"
fi

# Auto-detect smoothing if not specified
if [[ -z "$SMOOTHING" ]]; then
    # Find one representative mwp1r file under the supplied CAT12 dir and
    # try to extract the smoothing kernel (e.g. 's6mwp1r' -> 6). If no
    # smoothing prefix is present, fall back to default 6 mm.
    if [[ -n "$CAT12_DIR" ]]; then
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
    else
        SMOOTHING="6"
    fi
fi

# Set analysis name from config's folder_name field or use provided ANALYSIS_NAME
if [[ -z "$ANALYSIS_NAME" ]]; then
    # Get folder_name from the matching modality in config.json
    ANALYSIS_NAME=$(python3 << PYEOF
import json
try:
    with open("$CONFIG_JSON") as f:
        config = json.load(f)
    modality_name = "$MODALITY"
    for mod in config.get("analysis", {}).get("modalities", []):
        if mod.get("name") == modality_name:
            folder_name = mod.get("folder_name", "")
            if folder_name:
                print(folder_name)
            else:
                # Fallback: construct from modality and smoothing
                print(f"{modality_name}_smooth_auto")
            break
except Exception:
    print("${MODALITY}_smooth_auto")
PYEOF
)
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
"$MATLAB_EXE" $MATLAB_FLAGS "addpath('$UTILS_DIR'); try, configure_spm_path; catch e, fprintf('Warning: configure_spm_path failed: %s\n', e.message); end; exit;" 2>&1 | tee -a "$MATLAB_SPM_LOG" | external_prefix || {
    log_warning "one-time SPM configuration step failed (see $MATLAB_SPM_LOG). Continuing, but later MATLAB calls may need SPM path set."
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
                log_info "Removing entire results directory: $OUTPUT_DIR"
                rm -rf "$OUTPUT_DIR"
                log_success "Removed $OUTPUT_DIR"
                ;;
            *)
                log_warning "OUTPUT_DIR ($OUTPUT_DIR) is outside expected results path."
                log_warning "Removing contents of $OUTPUT_DIR instead of the whole directory."
                rm -rf "$OUTPUT_DIR"/*
                log_success "Cleaned contents of $OUTPUT_DIR"
                ;;
        esac
    else
        log_info "No existing results directory to remove: $OUTPUT_DIR"
    fi

    # Remove any stale temporary directories for this analysis
    TMP_PATTERN="$STATS_DIR/.tmp_${ANALYSIS_NAME}_*"
    shopt -s nullglob
    tmpdirs=( $TMP_PATTERN )
    for d in "${tmpdirs[@]:-}"; do
        log_info "Removing temp directory: $d"
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

if [[ -n "$CAT12_DIR" ]] && [[ -n "$PARTICIPANTS_FILE" ]]; then
    PRECHECK_MASK=$(get_json_value "analysis.mask" "")
    PRECHECK_MASK_ARG=""
    if [[ -n "$PRECHECK_MASK" ]]; then
        PRECHECK_MASK_ARG="--mask $PRECHECK_MASK"
    fi
    python3 "$UTILS_DIR/preflight_check.py" --cat12-dir "$CAT12_DIR" --participants "$PARTICIPANTS_FILE" --smoothing "$SMOOTHING" --modality "$MODALITY" $PRECHECK_MASK_ARG || {
        log_error "Preflight checks failed. Fix issues above and re-run."
        exit 1
    }
else
    log_info "Skipping preflight checks (CAT12_DIR or PARTICIPANTS_FILE not provided)."
fi


echo "┌────────────────────────────────────────────────────────────────────────┐"
echo "│ STEP 1: Parsing Participants File                                     │"
echo "└────────────────────────────────────────────────────────────────────────┘"
echo ""

if [[ -n "$DESIGN_FILE" ]]; then
    log_info "Using provided design file: $DESIGN_FILE"
    cp "$DESIGN_FILE" "$TEMP_DIR/design.json"
else
    # If modality is thickness, drop TIV as a covariate even if the user
    # requested it. TIV is not an appropriate covariate for cortical
    # thickness and would remove the effect of interest.
    if [[ "$MODALITY" == "thickness" && -n "$COVARIATES" ]]; then
        COVARIATES=$(python3 - <<PY
cov_str = "${COVARIATES}"
parts = [c.strip() for c in cov_str.split(',') if c.strip().lower() != 'tiv']
print(','.join(parts))
PY
)
        if [[ -z "$COVARIATES" ]]; then
            log_info "Thickness modality: removed TIV from covariates; no covariates remain."
        else
            log_info "Thickness modality: removed TIV from covariates. Using: $COVARIATES"
        fi
    fi

    python3 "$UTILS_DIR/parse_participants.py" \
        --cat12-dir "$CAT12_DIR" \
        --participants "$PARTICIPANTS_FILE" \
        --modality "$MODALITY" \
        --smoothing "$SMOOTHING" \
        --output "$TEMP_DIR" \
        ${GROUP_COL:+--group-col "$GROUP_COL"} \
        --session-col "$SESSION_COL" \
        --sessions "$SESSIONS" \
        ${COVARIATES:+--covariates "$COVARIATES"} \
        ${STANDARDIZE_CONTINUOUS:+--standardize-continuous} || {
            log_error "Failed to parse participants file"
            exit 1
        }
fi

echo ""

# Persist design.json into the results folder so reports can be generated even
# if temporary directories are cleaned up (helps when --force is used).
if [[ -f "$TEMP_DIR/design.json" ]]; then
    mkdir -p "$OUTPUT_DIR"
    cp "$TEMP_DIR/design.json" "$OUTPUT_DIR/design.json"
    echo "Design JSON copied to: $OUTPUT_DIR/design.json"
fi

# If covariates were resolved in the design, append them to the analysis name
# so output folders reflect covariate usage (e.g., vbm_smooth_auto_tiv)
# UNLESS the analysis name from config (folder_name) already includes them
if [[ -f "$TEMP_DIR/design.json" ]]; then
    COV_LIST=$(python3 - <<PY
import json
d=json.load(open('$TEMP_DIR/design.json'))
covs=list(d.get('covariates',{}).keys())
print(','.join(covs))
PY
)
    if [[ -n "$COV_LIST" ]]; then
        # convert comma-separated to underscore-separated suffix
        COV_SUFFIX=$(echo "$COV_LIST" | sed 's/,/_/g')
        # Check if the suffix is already in the ANALYSIS_NAME (from config folder_name)
        if [[ "$ANALYSIS_NAME" != *"$COV_SUFFIX"* ]]; then
            # Only append if not already present
            NEW_ANALYSIS_NAME="${ANALYSIS_NAME}_${COV_SUFFIX}"
            NEW_OUTPUT_DIR="$STATS_DIR/results/${MODALITY}/${NEW_ANALYSIS_NAME}"
            if [[ "$NEW_OUTPUT_DIR" != "$OUTPUT_DIR" ]]; then
                # Ensure parent exists
                mkdir -p "$(dirname "$NEW_OUTPUT_DIR")"
                # Move current output dir to new name (keep existing logs/files)
                mv "$OUTPUT_DIR" "$NEW_OUTPUT_DIR" 2>/dev/null || true
                OUTPUT_DIR="$NEW_OUTPUT_DIR"
                ANALYSIS_NAME="$NEW_ANALYSIS_NAME"
                LOG_DIR="$OUTPUT_DIR/logs"
                LOG_FILE="$LOG_DIR/pipeline.log"
                log_info "Renamed results folder to include covariates: $OUTPUT_DIR"
            fi
        else
            log_info "Analysis name already includes covariates: $ANALYSIS_NAME"
        fi
    fi
fi

    # Generate an ASCII preview of the design matrix and save it to results
    if [[ -f "$OUTPUT_DIR/design.json" ]]; then
        log_info "Generating ASCII design-matrix preview (text)..."
        python3 "$UTILS_DIR/print_design_ascii.py" "$OUTPUT_DIR/design.json" --output "$OUTPUT_DIR/design_ascii.txt" --rows 20 || {
            log_warning "ASCII design preview generation failed"
        }
        if [[ -f "$OUTPUT_DIR/design_ascii.txt" ]]; then
            echo "--- Design ASCII preview (first lines) ---"
            head -n 30 "$OUTPUT_DIR/design_ascii.txt" || true
            echo "-----------------------------------------"
        fi
    fi

# ============================================================================
# Step 2a: Explicit mask handling
# ============================================================================

# For cortical thickness (surface-based GIfTI analysis) we do not use an
# explicit volumetric GM mask. The design/batch utilities will instead work
# directly with surface files.
MASK_FILE=""

if [[ "$MODALITY" != "thickness" ]]; then
    # For non-thickness modalities we prefer the repo-level canonical CAT12
    # tight brainmask located at templates/brainmask_GMtight.nii (or an
    # override from config.ini) to keep masking consistent across analyses.
    GM_MASK_CONFIG=$(get_json_value "analysis.mask" "")
    if [[ -z "$GM_MASK_CONFIG" ]]; then
        GM_MASK_CONFIG=$(get_ini_value "MASKING" "gm_mask" "")
    fi
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
else
    log_info "Thickness modality detected – running without an explicit GM mask"
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

python3 "$UTILS_DIR/generate_spm_batch.py" \
    --design-file "$TEMP_DIR/design.json" \
    --output-dir "$OUTPUT_DIR" \
    --modality "$MODALITY" \
    --output "$TEMP_DIR/spm_batch.m" \
    $MASK_ARG || {
        log_error "Failed to generate SPM batch"
        exit 1
    }

# Copy batch file to output directory for reproducibility
cp "$TEMP_DIR/spm_batch.m" "$OUTPUT_DIR/spm_batch.m"
log_success "SPM batch file generated and saved to: $OUTPUT_DIR/spm_batch.m"
echo ""

# ============================================================================
# Step 3: Run Model Estimation
# ============================================================================

echo "┌────────────────────────────────────────────────────────────────────────┐"
echo "│ STEP 3: SPM Model Estimation                                          │"
echo "└────────────────────────────────────────────────────────────────────────┘"
echo ""

mkdir -p "$OUTPUT_DIR"

# Delete any existing SPM.mat and derived files to ensure a clean estimation
if [[ -f "$OUTPUT_DIR/SPM.mat" ]]; then
    log_info "Removing existing model and derived files to ensure consistency..."
    rm -f "$OUTPUT_DIR"/SPM.mat
    rm -f "$OUTPUT_DIR"/beta_*.nii
    rm -f "$OUTPUT_DIR"/con_*.nii
    rm -f "$OUTPUT_DIR"/spmT_*.nii
    rm -f "$OUTPUT_DIR"/spmF_*.nii
    rm -f "$OUTPUT_DIR"/ResMS.nii
    rm -f "$OUTPUT_DIR"/mask.nii
    rm -f "$OUTPUT_DIR"/RPV.nii
    # Also clean old TFCE results and reports to avoid confusion
    rm -f "$OUTPUT_DIR"/tfce_*.nii
    rm -f "$OUTPUT_DIR"/*_log_pfwe*.nii
    rm -f "$OUTPUT_DIR"/report.html
    rm -rf "$OUTPUT_DIR"/report
    log_success "Cleaned old results"
fi

# Ensure logs directory exists
mkdir -p "$LOG_DIR"

# Determine estimation method
EST_METHOD="matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;"
echo "Using Classical Estimation (ReML)"

MATLAB_MODEL_LOG="$LOG_DIR/matlab_model_estimation.log"
"$MATLAB_EXE" $MATLAB_FLAGS "warning('off','MATLAB:dispatcher:nameConflict'); warning('off','all'); set(0,'DefaultFigureVisible','off'); set(0,'DefaultFigureCreateFcn',@(h,ev)[]); addpath('$STATS_DIR/scripts/utils'); spm('defaults', 'FMRI'); spm_jobman('initcfg'); fprintf('═══════════════════════════════════════════════════════\n'); fprintf('Running Factorial Design Specification\n'); fprintf('═══════════════════════════════════════════════════════\n\n'); run('$TEMP_DIR/spm_batch.m'); try, spm_jobman('run', matlabbatch); catch e, fprintf('Warning: Design reporting failed (expected in headless mode):\n%s\n', e.message); end; clear matlabbatch; fprintf('\n═══════════════════════════════════════════════════════\n'); fprintf('Running Model Estimation\n'); fprintf('═══════════════════════════════════════════════════════\n\n'); matlabbatch{1}.spm.stats.fmri_est.spmmat = {'$OUTPUT_DIR/SPM.mat'}; matlabbatch{1}.spm.stats.fmri_est.write_residuals = 0; $EST_METHOD spm_jobman('run', matlabbatch); fprintf('\n✓ Model estimation complete\n\n'); exit;" 2>&1 | tee -a "$MATLAB_MODEL_LOG" | external_prefix || {
        echo "Error: Model estimation failed"
        echo "Check MATLAB log: $LOG_DIR/matlab_model_estimation.log"
        exit 1
    }

echo "✓ Model estimation complete"
echo ""

# Export design matrix to CSV for inspection (Priority Request)
echo "Exporting design matrix to CSV..."
"$MATLAB_EXE" $MATLAB_FLAGS "warning('off','all'); load('$OUTPUT_DIR/SPM.mat'); X = SPM.xX.X; writematrix(X, '$OUTPUT_DIR/design_matrix.csv'); exit;" 2>&1 | external_prefix || {
    echo "Warning: Failed to export design matrix to CSV"
}
if [[ -f "$OUTPUT_DIR/design_matrix.csv" ]]; then
    echo "✓ Design matrix exported to: $OUTPUT_DIR/design_matrix.csv"
fi
echo ""

# ============================================================================
# Step 3b: Check for missing voxels across images (optional diagnostic)
# This helps detect voxels with many NaNs or missing data that can break
# permutation schemes. Writes summary JSON and an exclusion mask PNG/NIfTI.
# ============================================================================

echo "Running missing-voxel diagnostics (this is fast)"
# Read optional failure threshold from config.ini (empty disables failure)
MISSING_FAIL_PCT=$(get_ini_value "TFCE" "missing_fail_pct" "")

# Surface modalities (thickness, depth, gyrification) are surface-based; skip volumetric missing-voxel diagnostics.
if [[ "$MODALITY" == "thickness" || "$MODALITY" == "depth" || "$MODALITY" == "gyrification" || "$MODALITY" == "fractal" ]]; then
    echo "Skipping volumetric missing-voxel diagnostic for surface modality ($MODALITY)"
else
    GM_MASK_ARG=""
    if [[ -n "$MASK_FILE" ]]; then
        GM_MASK_ARG="--gm-mask $MASK_FILE"
    fi
    if [[ -n "$MISSING_FAIL_PCT" && "$MISSING_FAIL_PCT" != "false" ]]; then
        python3 "$UTILS_DIR/check_missing_voxels.py" --spm "$OUTPUT_DIR/SPM.mat" --output-dir "$OUTPUT_DIR" --threshold 0.05 --fail-if-pct-excluded "$MISSING_FAIL_PCT" || {
            log_error "Missing-voxel fraction exceeded ${MISSING_FAIL_PCT}% — aborting pipeline."
            exit 1
        }
    else
        python3 "$UTILS_DIR/check_missing_voxels.py" --spm "$OUTPUT_DIR/SPM.mat" --output-dir "$OUTPUT_DIR" --threshold 0.05 $GM_MASK_ARG || {
            log_warning "missing-voxel diagnostic failed (see script output above). Continuing analysis."
        }
    fi
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
"$MATLAB_EXE" $MATLAB_FLAGS "warning('off','MATLAB:dispatcher:nameConflict'); warning('off','all'); addpath('$STATS_DIR/scripts/utils'); spm('defaults', 'FMRI'); spm_jobman('initcfg'); try, add_contrasts_longitudinal('$OUTPUT_DIR'); catch e, fprintf('ERROR in add_contrasts_longitudinal:\n%s\n', e.message); end; exit;" 2>&1 | tee -a "$MATLAB_CONTRAST_LOG" | external_prefix || {
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
if [[ "$MODALITY" == "vbm" ]]; then
    cons=( "$OUTPUT_DIR"/con_*.nii )
    spmTs=( "$OUTPUT_DIR"/spmT_*.nii )
    spmFs=( "$OUTPUT_DIR"/spmF_*.nii )
else
    # Surface-based modalities (e.g. thickness) write GIfTI outputs
    cons=( "$OUTPUT_DIR"/con_*.gii )
    spmTs=( "$OUTPUT_DIR"/spmT_*.gii )
    spmFs=( "$OUTPUT_DIR"/spmF_*.gii )
fi
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
    addpath('$STATS_DIR/scripts/utils'); \
    generate_design_matrix_image('$OUTPUT_DIR/SPM.mat', '$OUTPUT_DIR/design_matrix.png'); \
    exit;" 2>&1 | external_prefix || {
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
    
    "$MATLAB_EXE" $MATLAB_FLAGS "warning('off','MATLAB:dispatcher:nameConflict'); warning('off','all'); set(0,'DefaultFigureVisible','off'); set(0,'DefaultFigureCreateFcn',@(h,ev)[]); addpath('$STATS_DIR/scripts/utils'); spm('defaults','FMRI'); spm_jobman('initcfg'); try, significant_contrasts = screen_contrasts('$OUTPUT_DIR','p_thresh',$UNCORRECTED_P,'cluster_size',$CLUSTER_SIZE); fprintf('\\n✓ Screening complete with %d significant contrasts\\n\\n', length(significant_contrasts)); fid=fopen(fullfile('$OUTPUT_DIR','logs','significant_contrasts.txt'),'w'); if fid>0, for ii=1:numel(significant_contrasts), fprintf(fid,'%d\\n',significant_contrasts(ii)); end; fclose(fid); end; catch e, fprintf('MATLAB_ERROR:%s\\n', e.message); end; exit;" 2>&1 | external_prefix || {
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

if [[ "$NO_TFCE" == true ]]; then
    echo "┌────────────────────────────────────────────────────────────────────────┐"
    echo "│ STEP 6: Skipped (TFCE disabled by --no-tfce)                          │"
    echo "└────────────────────────────────────────────────────────────────────────┘"
    echo ""
    echo "Pipeline stopping early as requested."
    echo "Results saved to: $OUTPUT_DIR"
    exit 0
fi

# ============================================================================
# Step 6: TFCE Correction
# ============================================================================

mkdir -p "$LOG_DIR"

TFCE_LOG="$LOG_DIR/matlab_tfce.log"
TFCE_SUMMARY="$LOG_DIR/tfce_cc_summary.json"

print_tfce_summary_table() {
    local summary_path="$1"
    local threshold="$2"
    python3 - "$summary_path" "$threshold" <<'PY'
import json
import sys

summary_path = sys.argv[1]
threshold = float(sys.argv[2])
try:
    with open(summary_path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
except FileNotFoundError:
    print(f"  Summary file not found: {summary_path}")
    sys.exit(1)
except json.JSONDecodeError as exc:
    print(f"  Could not parse summary JSON ({exc})")
    sys.exit(1)

if not data:
    print("  (no contrasts recorded)")
    sys.exit(0)

print("  Contrast  Probe_cc  Recommended_full_method  Logged_full_method")
for entry in data:
    con = entry.get('contrast')
    con_str = str(con) if con is not None else '--'
    cc = entry.get('probe_cc')
    try:
        cc_val = float(cc) if cc is not None else None
    except (TypeError, ValueError):
        cc_val = None
    cc_str = f"{cc_val:.4f}" if cc_val is not None else "--"
    recommended = 'freedman-lane' if (cc_val is not None and cc_val < threshold) else 'smith'
    logged = entry.get('chosen_full_method') or '--'
    print(f"    {con_str:>4}     {cc_str:>8}  {recommended:<22} {logged:<18}")
PY
}

echo "┌────────────────────────────────────────────────────────────────────────┐"
echo "│ STEP 6: TFCE Permutation Testing                                      │"
echo "└────────────────────────────────────────────────────────────────────────┘"
echo ""

# If screening was run and produced an (empty) significant list, skip TFCE.
SKIP_TFCE=false

SIGNIF_FILE="$OUTPUT_DIR/logs/significant_contrasts.txt"
if [[ "$PILOT_MODE" != true && "$SKIP_SCREENING" == false && -f "$SIGNIF_FILE" ]]; then
    if [[ ! -s "$SIGNIF_FILE" ]]; then
        echo "No significant contrasts found by screening (file: $SIGNIF_FILE). Skipping TFCE step."
        SKIP_TFCE=true
    fi
fi

if [[ "$SKIP_TFCE" == true ]]; then
    echo "Skipping TFCE step because no screened contrasts were significant."
else
if [[ "$PILOT_MODE" == true ]]; then
    # In pilot mode run the quick TFCE directly (keep behavior simple)
    echo "Pilot mode: running single short TFCE run (${N_PERM} perms)"
    "$MATLAB_EXE" $MATLAB_FLAGS "warning('off','MATLAB:dispatcher:nameConflict'); warning('off','all'); set(0,'DefaultFigureVisible','off'); set(0,'DefaultFigureCreateFcn',@(h,ev)[]); addpath('$STATS_DIR/scripts/utils'); spm('defaults', 'FMRI'); spm_jobman('initcfg'); fprintf('Starting pilot TFCE with %d permutations\n', $N_PERM); run_tfce_correction('$OUTPUT_DIR', 'n_perm', $N_PERM, 'n_jobs', $N_JOBS, 'pilot', true); exit;" 2>&1 | tee -a "$TFCE_LOG" | external_prefix || {
        log_error "TFCE correction (pilot) failed"
        exit 1
    }
else
    # Standard TFCE run (single stage, no probe)
    echo "Running TFCE correction (${N_PERM} permutations)"
    
    # Convert SKIP_SCREENING (bash string) to MATLAB boolean
    if [[ "$SKIP_SCREENING" == "true" ]]; then
        USE_SCREENING="false"
    else
        USE_SCREENING="true"
    fi

    "$MATLAB_EXE" $MATLAB_FLAGS "warning('off','MATLAB:dispatcher:nameConflict'); warning('off','all'); set(0,'DefaultFigureVisible','off'); set(0,'DefaultFigureCreateFcn',@(h,ev)[]); addpath('$STATS_DIR/scripts/utils'); spm('defaults', 'FMRI'); spm_jobman('initcfg'); fprintf('Starting TFCE with %d permutations\n', $N_PERM); run_tfce_correction('$OUTPUT_DIR', 'n_perm', $N_PERM, 'n_jobs', $N_JOBS, 'use_screening', $USE_SCREENING); exit;" 2>&1 | tee -a "$TFCE_LOG" | external_prefix || {
        log_error "TFCE correction failed"
        exit 1
    }
fi

log_success "TFCE correction complete"
echo ""

# ============================================================================
# Step 6b: Generate TFCE Summary
# ============================================================================

echo "Generating TFCE results summary..."
python3 "$UTILS_DIR/generate_tfce_images.py" \
    --output-dir "$OUTPUT_DIR" \
    --fwe-threshold 0.05 \
    --start-time "$PIPELINE_START_TIME" || {
        log_warning "TFCE summary generation failed"
    }

echo ""
fi

echo "┌────────────────────────────────────────────────────────────────────────┐"
echo "│ STEP 7: Generating HTML Report                                        │"
echo "└────────────────────────────────────────────────────────────────────────┘"
echo ""

# Provide number of contrasts to the report generator (count con_*.nii)
# Provide number of contrasts to the report generator, using extension
# appropriate for modality (.nii for VBM, .gii for surface modalities).
if [[ "$MODALITY" == "vbm" ]]; then
    N_CONTRASTS=$(ls -1 "$OUTPUT_DIR"/con_*.nii 2>/dev/null | wc -l)
else
    N_CONTRASTS=$(ls -1 "$OUTPUT_DIR"/con_*.gii 2>/dev/null | wc -l)
fi

# Build a safely-quoted command-line string from the original args. Use
# printf '%q' so special characters and quotes are escaped and the result
# is safe to pass as a single argument to Python.
SAFE_CMDLINE="$(printf '%q ' "$0" "${ORIGINAL_ARGS[@]}")"

python3 "$UTILS_DIR/generate_html_report.py" \
    --design-json "$TEMP_DIR/design.json" \
    --output "$OUTPUT_DIR/report.html" \
    --analysis-name "$ANALYSIS_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --command-line "$SAFE_CMDLINE" \
    --n-contrasts "$N_CONTRASTS" \
    --n-perm "$N_PERM" \
    --cluster-size "$CLUSTER_SIZE" \
    --uncorrected-p "$UNCORRECTED_P" \
    --start-time "$PIPELINE_START_TIME" || {
        log_warning "HTML report generation failed"
    }

echo ""

# ============================================================================
# Cleanup
# ============================================================================

echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo "Pipeline complete! Results saved to: $OUTPUT_DIR"
echo ""
