#!/bin/bash
# Monitor CAT12 preprocessing and automatically run stats when complete

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

LOG_FILE="$REPO_ROOT/full_preprocessing_sequential.log"
OUTPUT_DIR="$REPO_ROOT/projects/demo/derivatives/cat12"
EXPECTED_OUTPUTS=8  # 4 subjects × 2 sessions

echo "=========================================="
echo "🔍 CAT12 Preprocessing Monitor"
echo "=========================================="
echo "Watching: $LOG_FILE"
echo "Expected outputs: $EXPECTED_OUTPUTS modulated segmentations"
echo "Starting monitor..."
echo ""

# Wait for preprocessing to complete
max_wait=14400  # 4 hours
interval=60  # Check every minute
elapsed=0

while [ $elapsed -lt $max_wait ]; do
    if [ -f "$LOG_FILE" ]; then
        if tail -20 "$LOG_FILE" 2>/dev/null | grep -q "❌ No subjects were successfully processed"; then
            echo "❌ Preprocessing failed - no subjects completed"
            exit 1
        fi
        
        if tail -20 "$LOG_FILE" 2>/dev/null | grep -q "boilerplate.md"; then
            # Preprocessing appears complete
            break
        fi
    fi
    
    # Count actual outputs
    output_count=$(find "$OUTPUT_DIR" -name "mwp1*.nii*" 2>/dev/null | wc -l)
    echo "[$(date '+%H:%M:%S')] Outputs: $output_count/$EXPECTED_OUTPUTS (elapsed: ${elapsed}s)"
    
    if [ "$output_count" -ge "$EXPECTED_OUTPUTS" ]; then
        echo "✅ All preprocessing outputs found!"
        break
    fi
    
    sleep "$interval"
    ((elapsed += interval))
done

if [ $elapsed -ge $max_wait ]; then
    echo "⚠️  Timeout waiting for preprocessing (4 hours elapsed)"
fi

echo ""
echo "=========================================="
echo "📊 STARTING STATISTICS PIPELINE"
echo "=========================================="
echo ""

# Activate environment
source .venv/bin/activate
source .env

# Extract covariates
echo "📊 Extracting covariates from CAT12 outputs..."
python scripts/utils/extract_covariates_from_xml.py \
  projects/demo/derivatives/cat12 \
  projects/demo/participants_demo.tsv || echo "⚠️  Covariate extraction had issues, continuing anyway..."

echo ""

# Run statistics
echo "📊 Running statistics pipeline..."
bash scripts/analysis/cat12_multi_modality.sh \
  --config projects/demo/project_config.json \
  --cat12-dir projects/demo/derivatives/cat12 \
  --participants projects/demo/participants_demo.tsv

echo ""
echo "=========================================="
echo "✅ Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results location: projects/demo/results/"
echo ""
