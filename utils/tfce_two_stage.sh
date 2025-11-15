#!/usr/bin/env bash
# tfce_two_stage.sh
# Run TFCE in two stages per-contrast: probe (low perms) then full run.
# Usage: ./utils/tfce_two_stage.sh /path/to/stats_folder [initial_perm] [full_perm] [cc_threshold]

set -euo pipefail

STATS_FOLDER="${1:-}"
INITIAL_PERM=${2:-100}
FULL_PERM=${3:-5000}
CC_THRESH=${4:-0.98}

if [[ -z "$STATS_FOLDER" ]]; then
  echo "Usage: $0 /path/to/stats_folder [initial_perm] [full_perm] [cc_threshold]"
  exit 2
fi

LOG_DIR="$STATS_FOLDER/logs"
mkdir -p "$LOG_DIR"

# Detect utils dir and matlab binary
UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATLAB_BIN="/Applications/MATLAB_R2025b.app/bin/matlab"

if [[ ! -x "$MATLAB_BIN" ]]; then
  # Fallback to matlab in PATH
  MATLAB_BIN="$(command -v matlab || true)"
fi

if [[ -z "$MATLAB_BIN" ]]; then
  echo "Error: MATLAB executable not found. Set path in config.ini or ensure 'matlab' is on PATH." >&2
  exit 1
fi

run_matlab_batch() {
  local cmd="$1"
  echo "Running MATLAB batch: $cmd"
  # Build a single-line MATLAB -batch command that adds utils to the path
  local batchcmd
  batchcmd="addpath('${UTILS_DIR}'); try, spm('defaults','FMRI'); spm_jobman('initcfg'); ${cmd}; catch e, fprintf('MATLAB_ERROR:%s\\n', e.message); end;"
  PATH="$(dirname "$MATLAB_BIN"):$PATH" "$MATLAB_BIN" -batch "$batchcmd"
}

# List contrasts (indices) available in SPM.mat
contrast_list_raw=$(PATH="$(dirname "$MATLAB_BIN"):$PATH" "$MATLAB_BIN" -batch "addpath('${UTILS_DIR}'); try, S=load(fullfile('${STATS_FOLDER}','SPM.mat')); for i=1:numel(S.SPM.xCon), fprintf('%d\\n',i); end; catch e, fprintf('MATLAB_ERROR\\n'); end;" 2>/dev/null || true)

# Filter to only numeric lines (MATLAB may print warnings that we should ignore)
contrast_list=$(printf "%s\n" "$contrast_list_raw" | grep -E '^[0-9]+$' || true)

# If screening produced a significants list, prefer that (one item per line)
SIGNIF_FILE="$STATS_FOLDER/logs/significant_contrasts.txt"
if [[ -f "$SIGNIF_FILE" ]]; then
  echo "Using screened significant contrasts from: $SIGNIF_FILE"
  contrast_list=$(grep -E '^[0-9]+' "$SIGNIF_FILE" || true)
fi

if [[ -z "$contrast_list" ]]; then
  echo "Could not read contrasts from SPM.mat (or no significant contrasts listed). Ensure MATLAB can run and SPM.mat is present." 
  exit 1
fi

# iterate through contrasts
for con in $contrast_list; do
  printf "\n=== Contrast %s ===\n" "$con"

  # 1) Probe run (low permutations)
  printf "Probe run: %s permutations\n" "$INITIAL_PERM"
  run_matlab_batch "run_tfce_correction('${STATS_FOLDER}','n_perm',${INITIAL_PERM},'contrast_list',${con},'force',true)" || {
    echo "MATLAB probe call failed for contrast ${con}" >&2
    continue
  }

  # 2) Parse the latest cc value for this contrast from the TFCE log
  tfce_log="$LOG_DIR/matlab_tfce.log"
  if [[ ! -f "$tfce_log" ]]; then
    echo "TFCE log not found: $tfce_log" >&2
    break
  fi

  # Extract last cc= value that appears after the line indicating this contrast
  # Use Perl for robust multiline regex extraction (more portable than complex awk on macOS)
  if command -v perl >/dev/null 2>&1; then
    cc_val=$(perl -0777 -ne "while (/(Use contrast #${con}[\s\S]*?cc=([0-9]*\.?[0-9]+))/g) { \$last = \$1 } print \$last if defined \$last;" "$tfce_log" ) || true
  else
    cc_val=$(grep -o 'cc=[0-9]*\.?[0-9]*' "$tfce_log" | tail -1 | sed 's/cc=//' ) || true
  fi

  printf "Probe cc value: %s\n" "${cc_val:-not found}"

  # 3) Decide which nuisance method to use for full run
  if [[ -n "$cc_val" ]]; then
    awk -v cc="$cc_val" -v thr="$CC_THRESH" 'BEGIN{ if(cc+0 >= thr+0) exit 0; else exit 1 }'
    if [[ $? -eq 0 ]]; then
      echo "cc >= $CC_THRESH: continue with default nuisance method (smith) for full run"
      run_matlab_batch "run_tfce_correction('${STATS_FOLDER}','n_perm',${FULL_PERM},'contrast_list',${con},'force',true)" || echo "Full run failed (smith) for contrast ${con}" >&2
    else
      echo "cc < $CC_THRESH: switching to Freedman-Lane nuisance handling for full run"
      run_matlab_batch "run_tfce_correction('${STATS_FOLDER}','n_perm',${FULL_PERM},'contrast_list',${con},'force',true,'nuisance_method','freedman-lane')" || echo "Full run failed (freedman-lane) for contrast ${con}" >&2
    fi
  else
    echo "Could not determine cc; running full TFCE with default settings"
    run_matlab_batch "run_tfce_correction('${STATS_FOLDER}','n_perm',${FULL_PERM},'contrast_list',${con},'force',true)" || echo "Full run failed (default) for contrast ${con}" >&2
  fi
done

echo "All contrasts processed. Check $LOG_DIR/matlab_tfce.log for details."
echo "All contrasts processed. Check $LOG_DIR/matlab_tfce.log for details."
