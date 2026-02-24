#!/bin/bash
# Full CAT12 Pipeline Runner
# Runs preprocessing -> stats -> report generation

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Activate environment
source .venv/bin/activate
source .env

# Disable progress bars for non-interactive mode to prevent BrokenPipeError
export TQDM_DISABLE=1
export TQDM_MININTERVAL=9999999

echo "=========================================="
echo "🚀 CAT12 Full Pipeline Started"
echo "=========================================="
echo "Time: $(date)"
echo ""

# Phase 0: Cleanup existing demo data
echo "📊 PHASE 0: Cleaning up existing demo data"
echo "---"

if [ -d "projects/demo/derivatives" ]; then
    echo "Removing existing derivatives folder..."
    rm -rf projects/demo/derivatives
fi

if [ -d "projects/demo/work" ]; then
    echo "Removing existing work folder..."
    rm -rf projects/demo/work
fi

if [ -d "projects/demo/results" ]; then
    echo "Removing existing results folder..."
    rm -rf projects/demo/results
fi

if [ -d "openneuro/ds004937" ]; then
    echo "Removing existing OpenNeuro download..."
    rm -rf openneuro/ds004937
fi

echo "✅ Cleanup complete!"
echo ""

EXPECTED_OUTPUTS=$(python3 - <<'PY'
import json
from pathlib import Path

cfg = Path("projects/demo/run_demo.json")
if not cfg.exists():
  print(16)
else:
  data = json.loads(cfg.read_text())
  bids = data.get("preprocessing", {}).get("bids", {})
  n_sub = len(bids.get("participant_label", []) or [])
  n_ses = len(bids.get("session_label", []) or [])
  print(n_sub * n_ses if n_sub and n_ses else 16)
PY
)

# Phase 1: Full Preprocessing
echo "📊 PHASE 1: Preprocessing all subjects (expected ${EXPECTED_OUTPUTS} images)"
echo "Expected duration: 2-4 hours"
echo "---"

./cat12_prepro --config projects/demo/run_demo.json

echo ""
echo "✅ Preprocessing complete!"
echo "Completion time: $(date)"
echo ""

# Phase 2: Verify outputs
echo "📊 PHASE 2: Verifying preprocessing outputs"
echo "---"

OUTPUT_COUNT=$(find projects/demo/derivatives/cat12 -name "mwp1*.nii*" 2>/dev/null | wc -l)
echo "Found $OUTPUT_COUNT modulated segmentations (expect ${EXPECTED_OUTPUTS})"

if [ "$OUTPUT_COUNT" -lt "$EXPECTED_OUTPUTS" ]; then
  echo "⚠️  Warning: Expected ${EXPECTED_OUTPUTS} outputs, found $OUTPUT_COUNT"
fi

echo ""

# Phase 3: Extract Covariates
echo "📊 PHASE 3: Extracting covariates from CAT12 outputs"
echo "---"

python scripts/utils/extract_covariates_from_xml.py \
  --cat12 projects/demo/derivatives/cat12 \
  --participants openneuro/ds004937/participants.tsv \
  --out projects/demo/participants_ds004937.tsv

echo "✅ Covariates extracted!"
echo ""

# Phase 4: Statistics Pipeline
echo "📊 PHASE 4: Running statistics pipeline (VBM, thickness, depth, gyrification)"
echo "Expected duration: 30-60 minutes"
echo "---"

bash scripts/analysis/cat12_multi_modality.sh \
  --config projects/demo/run_demo.json \
  --cat12-dir projects/demo/derivatives/cat12 \
  --participants projects/demo/participants_ds004937.tsv

echo "✅ Statistics complete!"
echo "Completion time: $(date)"
echo ""

# Phase 5: Generate Report
echo "📊 PHASE 5: Generating HTML report"
echo "---"

python scripts/reporting/post_stats_report.py \
  projects/demo/results/vbm_smooth6 \
  projects/demo/report_vbm.html

echo "✅ Report generated: projects/demo/report_vbm.html"
echo ""

echo "=========================================="
echo "✅ Full pipeline complete!"
echo "=========================================="
echo "Completion time: $(date)"
echo ""
echo "Results location:"
echo "  - Preprocessing: projects/demo/derivatives/cat12/"
echo "  - Statistics: projects/demo/results/"
echo "  - Report: projects/demo/report_vbm.html"
echo ""
