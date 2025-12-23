#!/usr/bin/env bash
# tfce_two_stage.sh
# Run TFCE in two stages per-contrast: probe (low perms) then full run.
# Usage: ./utils/tfce_two_stage.sh /path/to/stats_folder [initial_perm] [full_perm] [cc_threshold]

set -euo pipefail

# Support optional mode flags: --probe-only and --full-only
PROBE_ONLY=false
FULL_ONLY=false
if [[ "$1" == "--probe-only" ]]; then
  PROBE_ONLY=true; shift
elif [[ "$1" == "--full-only" ]]; then
  FULL_ONLY=true; shift
fi

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
TFCE_LOG="$LOG_DIR/matlab_tfce.log"
SAFE_LOG="$LOG_DIR/matlab_tfce.log.clean"
META_FILE="$LOG_DIR/tfce_contrast_metadata.json"
PLAN_TABLE="$LOG_DIR/tfce_full_plan.tsv"

# Detect utils dir and matlab binary
UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATLAB_BIN="/Applications/MATLAB_R2025b.app/bin/matlab"
PLAN_SCRIPT="$UTILS_DIR/tfce_full_run_plan.py"

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

ensure_contrast_metadata() {
  local spm_mat="$STATS_FOLDER/SPM.mat"
  if [[ ! -f "$spm_mat" ]]; then
    echo "Error: missing SPM.mat at $spm_mat" >&2
    exit 1
  fi
  local regenerate=false
  if [[ ! -f "$META_FILE" ]]; then
    regenerate=true
  elif [[ "$META_FILE" -ot "$spm_mat" ]]; then
    regenerate=true
  fi
  if [[ "$regenerate" == true ]]; then
    echo "Refreshing TFCE contrast metadata: $META_FILE"
    run_matlab_batch "export_tfce_probe_metadata('${STATS_FOLDER}','${META_FILE}')" || {
      echo "Warning: could not export TFCE contrast metadata (heuristics limited)" >&2
    }
  fi
}

refresh_clean_log() {
  if [[ ! -f "$TFCE_LOG" ]]; then
    return 0
  fi
  tr -d '\r' < "$TFCE_LOG" | sed 's/[[:cntrl:]]\[[0-9;]*[A-Za-z]//g' > "$SAFE_LOG" 2>/dev/null || cp "$TFCE_LOG" "$SAFE_LOG" 2>/dev/null || true
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

ensure_contrast_metadata

# iterate through contrasts
if [[ "$PROBE_ONLY" == true ]]; then
  echo "Running probe-only mode: will run probe=${INITIAL_PERM} for each contrast and write summary JSON"
  for con in $contrast_list; do
    printf "\n=== Contrast %s ===\n" "$con"
    printf "Probe run: %s permutations\n" "$INITIAL_PERM"
    echo "Probe nuisance method: smith (default)"
    run_matlab_batch "run_tfce_correction('${STATS_FOLDER}','n_perm',${INITIAL_PERM},'contrast_list',${con},'force',true,'nuisance_method','smith')" || {
      echo "MATLAB probe call failed for contrast ${con}" >&2
      continue
    }

    refresh_clean_log
    # small sleep to let MATLAB flush logs
    sleep 0.5
  done

  refresh_clean_log
  if [[ ! -f "$SAFE_LOG" ]]; then
    echo "Warning: cleaned TFCE log not found at $SAFE_LOG; summary JSON will be skipped" >&2
    exit 0
  fi
  # After probes, generate JSON summary from cleaned log
  python3 "${UTILS_DIR}/tfce_summary_from_log.py" "$SAFE_LOG" "$LOG_DIR/tfce_cc_summary.json" || {
    echo "Warning: could not generate tfce summary JSON" >&2
  }
  echo "Probe-only phase complete. Summary: $LOG_DIR/tfce_cc_summary.json"
  exit 0
fi

if [[ "$FULL_ONLY" == true ]]; then
  summary_file="$LOG_DIR/tfce_cc_summary.json"
  if [[ ! -f "$summary_file" ]]; then
    echo "Error: summary JSON not found: $summary_file" >&2
    exit 2
  fi
  echo "Running full-only mode: reading $summary_file and applying DF-aware heuristics (max perms=${FULL_PERM})"
  refresh_clean_log

  plan_output=""
  if [[ -f "$PLAN_SCRIPT" ]]; then
    plan_output=$(python3 "$PLAN_SCRIPT" "$summary_file" "$META_FILE" "$CC_THRESH" "$FULL_PERM" "$INITIAL_PERM" 2>/dev/null || true)
  else
    echo "Warning: plan script missing at $PLAN_SCRIPT; falling back to cc-only decisions" >&2
  fi

  if [[ -z "$plan_output" ]]; then
    echo "Heuristic planner produced no output; falling back to cc-only strategy"
    plan_output=$(python3 - "$summary_file" "$CC_THRESH" "$FULL_PERM" <<'PY'
import json, sys
summary_path, cc_threshold, full_perm = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
with open(summary_path, 'r', encoding='utf-8') as fh:
    data = json.load(fh)
for entry in data:
    con = entry.get('contrast')
    if con is None:
        continue
    method = entry.get('chosen_full_method')
    if not method:
        cc = entry.get('probe_cc')
        if cc is None:
            method = 'smith'
        else:
            method = 'freedman-lane' if float(cc) < cc_threshold else 'smith'
    print(f"{con}\t{method}\t{full_perm}\tcc_only")
PY
)
  fi

  if [[ -z "$plan_output" ]]; then
    echo "Error: no contrasts found in $summary_file" >&2
    exit 2
  fi

  {
    echo -e "contrast\tmethod\tn_perm\treasons"
    echo "$plan_output"
  } > "$PLAN_TABLE" 2>/dev/null || true

  echo "$plan_output" | while IFS=$'\t' read -r con method perms reasons; do
    if [[ -z "$con" || -z "$method" || -z "$perms" ]]; then
      continue
    fi
    printf "\n=== Contrast %s ===\n" "$con"
    echo "Full TFCE permutations: $perms"
    echo "Using nuisance method: $method"
    if [[ -n "$reasons" && "$reasons" != "-" ]]; then
      echo "Heuristic reasons: $reasons"
    fi
    run_matlab_batch "run_tfce_correction('${STATS_FOLDER}','n_perm',${perms},'contrast_list',${con},'force',true,'nuisance_method','$method')" || echo "Full run failed for contrast ${con}" >&2
  done

  refresh_clean_log
  if [[ -f "$SAFE_LOG" ]]; then
    python3 "${UTILS_DIR}/tfce_summary_from_log.py" "$SAFE_LOG" "$summary_file" || {
      echo "Warning: could not refresh summary JSON after full run" >&2
    }
  fi
  exit 0
fi

for con in $contrast_list; do
  printf "\n=== Contrast %s ===\n" "$con"

  # 1) Probe run (low permutations)
  printf "Probe run: %s permutations\n" "$INITIAL_PERM"
  # Run probe explicitly with default nuisance method (smith) so the probe
  # behavior is deterministic and logged. We still allow the MATLAB side to
  # override via its own config if needed.
  echo "Probe nuisance method: smith (default)"
  run_matlab_batch "run_tfce_correction('${STATS_FOLDER}','n_perm',${INITIAL_PERM},'contrast_list',${con},'force',true,'nuisance_method','smith')" || {
    echo "MATLAB probe call failed for contrast ${con}" >&2
    continue
  }

  # 2) Parse the latest cc value for this contrast from the TFCE log
  if [[ ! -f "$TFCE_LOG" ]]; then
    echo "TFCE log not found: $TFCE_LOG" >&2
    break
  fi

  # 2) Parse the latest cc value for this contrast from the TFCE log.
  # Prefer a robust Perl multiline search, but fall back to a simple grep
  # if Perl extraction fails. Sanitize the extracted value to ensure we
  # only pass a numeric token to awk (avoid "newline in string" errors).
  cc_val=""
  if command -v perl >/dev/null 2>&1; then
    cc_val=$(perl -0777 -ne "while (/(Use contrast #${con}[\s\S]*?cc=([0-9]*\.?[0-9]+))/g) { \$last = \$1 } print \$last if defined \$last;" "$TFCE_LOG" ) || true
  fi

  # Fallback: grep the last cc= occurrence in the logfile
  if [[ -z "$cc_val" ]]; then
    refresh_clean_log
    # Try Perl multiline match first (handles wrapped/newline-separated numbers)
    if command -v perl >/dev/null 2>&1; then
      # Capture the last cc= occurrence robustly even if the numeric token
      # is wrapped across lines: capture up to 40 chars after 'cc=' then
      # strip non-digit/dot chars to reconstruct the numeric token.
      cc_val=$(perl -0777 -ne 'while (/cc=(.{0,40})/gs) { $t=$1; $t=~s/[^0-9.]//g; $last=$t } print $last if defined $last' "$SAFE_LOG" 2>/dev/null || true)
    fi
    # Fallback to grep if perl didn't find anything
    if [[ -z "$cc_val" ]]; then
      cc_val=$(grep -o 'cc=[0-9]*\.?[0-9]*' "$SAFE_LOG" 2>/dev/null | tail -1 | sed 's/cc=//' ) || true
    fi
    # Also write a short excerpt around the last occurrence for debugging
    if [[ -n "$cc_val" ]]; then
      # write last ~200 lines for debugging (portable fallback)
      tail -n 200 "$SAFE_LOG" > "$LOG_DIR/tfce_cc_excerpt.txt" 2>/dev/null || true
    fi
  fi

  # Sanitize: strip whitespace and any surrounding text, keep only first numeric token
  if [[ -n "$cc_val" ]]; then
    # remove all whitespace
    cc_val=$(printf "%s" "$cc_val" | tr -d '[:space:]')
    # extract the leading numeric part (e.g. 0.00326233)
    cc_val=$(printf "%s" "$cc_val" | sed -n 's/[^0-9.]*\([0-9]\+\(\.[0-9]\+\)\?\).*/\1/p') || true
  fi

  # Also try to specifically capture the WARNING line that contains the cc
  # (this is the canonical line produced by the TFCE estimator).
  warn_line=""
  # Extract the canonical WARNING line robustly (allow wrapped/newline tokens)
  warn_line=""
  if command -v perl >/dev/null 2>&1; then
    warn_line=$(perl -0777 -ne 'if (/(WARNING:\s*Large discrepancy[\s\S]*?cc=\s*[0-9\s\.]+)/i) { my $w = $1; $w =~ s/[\r\n]+/ /g; print $w; }' "$SAFE_LOG" 2>/dev/null || true)
  fi
  if [[ -z "$warn_line" ]]; then
    # Last-resort grep (may fail if number is wrapped)
    warn_line=$(grep -E "Large discrepancy between parametric and non-parametric" "$SAFE_LOG" 2>/dev/null | tail -1 || true)
  fi
  if [[ -n "$warn_line" ]]; then
    printf "Probe warning line: %s\n" "$warn_line"
    # extract numeric cc from the warning line if not already extracted
    if [[ -z "$cc_val" ]]; then
      # remove whitespace/newlines, then extract number
      compact_warn=$(printf "%s" "$warn_line" | tr -d '[:space:]')
      cc_val=$(printf "%s" "$compact_warn" | sed -n 's/.*cc=\([0-9]*\.?[0-9]*\).*/\1/p' ) || true
    fi
    # save the warning excerpt for diagnostics
    echo "$warn_line" > "$LOG_DIR/tfce_probe_warning_contrast_$(printf "%04d" "$con").txt" 2>/dev/null || true
  fi

  printf "Probe cc value: %s\n" "${cc_val:-not found}"

  # 3) Decide which nuisance method to use for full run. If cc is not a
  # numeric value, treat it as 'not found' and run default behavior.
  if [[ -n "$cc_val" ]]; then
    # Ensure cc_val is numeric before calling awk
    if [[ "$cc_val" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
      awk -v cc="$cc_val" -v thr="$CC_THRESH" 'BEGIN{ if(cc+0 >= thr+0) exit 0; else exit 1 }'
      if [[ $? -eq 0 ]]; then
        echo "cc >= $CC_THRESH: continue with default nuisance method (smith) for full run"
        echo "Using nuisance method: smith"
        run_matlab_batch "run_tfce_correction('${STATS_FOLDER}','n_perm',${FULL_PERM},'contrast_list',${con},'force',true,'nuisance_method','smith')" || echo "Full run failed (smith) for contrast ${con}" >&2
      else
        echo "cc < $CC_THRESH: switching to Freedman-Lane nuisance handling for full run"
        echo "Using nuisance method: freedman-lane"
        run_matlab_batch "run_tfce_correction('${STATS_FOLDER}','n_perm',${FULL_PERM},'contrast_list',${con},'force',true,'nuisance_method','freedman-lane')" || echo "Full run failed (freedman-lane) for contrast ${con}" >&2
      fi
    else
      echo "Extracted cc value is not numeric ('$cc_val'); running full TFCE with default settings"
      run_matlab_batch "run_tfce_correction('${STATS_FOLDER}','n_perm',${FULL_PERM},'contrast_list',${con},'force',true)" || echo "Full run failed (default) for contrast ${con}" >&2
    fi
  else
    echo "Could not determine cc; running full TFCE with default settings"
    run_matlab_batch "run_tfce_correction('${STATS_FOLDER}','n_perm',${FULL_PERM},'contrast_list',${con},'force',true)" || echo "Full run failed (default) for contrast ${con}" >&2
  fi
done

refresh_clean_log
if [[ -f "$SAFE_LOG" ]]; then
  python3 "${UTILS_DIR}/tfce_summary_from_log.py" "$SAFE_LOG" "$LOG_DIR/tfce_cc_summary.json" || {
    echo "Warning: could not generate tfce summary JSON" >&2
  }
fi

echo "All contrasts processed. Check $TFCE_LOG for details."
echo "All contrasts processed. Check $TFCE_LOG for details."
