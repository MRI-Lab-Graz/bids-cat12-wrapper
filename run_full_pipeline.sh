#!/bin/bash
# Full CAT12 Pipeline Runner
# Runs preprocessing -> stats -> report generation

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Activate environment
source .venv/bin/activate
source .env

echo "=========================================="
echo "🚀 CAT12 Full Pipeline Started"
echo "=========================================="
echo "Time: $(date)"
echo ""

# Phase 1: Full Preprocessing
echo "📊 PHASE 1: Preprocessing all subjects (4 subjects × 2 sessions = 8 images)"
echo "Expected duration: 2-4 hours"
echo "---"

./cat12_prepro \
  openneuro/ds000114 \
  projects/demo/derivatives/cat12 \
  participant \
  --preproc \
  --participant-label 01 --participant-label 02 --participant-label 03 --participant-label 04 \
  --session-label test --session-label retest \
  --smooth-volume 6 --smooth-surface 12 \
  --qa --tiv --no-validate

echo ""
echo "✅ Preprocessing complete!"
echo "Completion time: $(date)"
echo ""

# Phase 2: Verify outputs
echo "📊 PHASE 2: Verifying preprocessing outputs"
echo "---"

OUTPUT_COUNT=$(find projects/demo/derivatives/cat12 -name "mwp1*.nii*" 2>/dev/null | wc -l)
echo "Found $OUTPUT_COUNT modulated segmentations (expect 8)"

if [ "$OUTPUT_COUNT" -lt 8 ]; then
    echo "⚠️  Warning: Expected 8 outputs, found $OUTPUT_COUNT"
fi

echo ""

# Phase 3: Extract Covariates
echo "📊 PHASE 3: Extracting covariates from CAT12 outputs"
echo "---"

python scripts/utils/extract_covariates_from_xml.py \
  projects/demo/derivatives/cat12 \
  projects/demo/participants_demo.tsv

echo "✅ Covariates extracted!"
echo ""

# Phase 4: Statistics Pipeline
echo "📊 PHASE 4: Running statistics pipeline (VBM, thickness, depth, gyrification)"
echo "Expected duration: 30-60 minutes"
echo "---"

bash scripts/analysis/cat12_multi_modality.sh \
  --config projects/demo/project_config.json \
  --cat12-dir projects/demo/derivatives/cat12 \
  --participants projects/demo/participants_demo.tsv

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
