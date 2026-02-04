#!/bin/bash
# Monitor the OpenNeuro download progress

PROJECT_DIR="/data/local/software/cat-12/projects/openneuro_ds003138"
LOG_FILE="/data/local/software/cat-12/project_run.log"

echo "=== OpenNeuro Download Monitor ==="
echo "Project: $PROJECT_DIR"
echo "Log: $LOG_FILE"
echo ""

# Check process
PID=$(ps aux | grep "python.*project_runner" | grep -v grep | awk '{print $2}')
if [ -n "$PID" ]; then
    echo "✓ Process running (PID: $PID)"
    ps aux | grep "$PID" | grep -v grep | awk '{print "  CPU: "$3"% | MEM: "$4"%"}'
else
    echo "✗ Process not running"
fi

echo ""
echo "=== Files Downloaded ===="

# Count files
NUM_NII=$(find "$PROJECT_DIR/bids_data" -name "*.nii.gz" 2>/dev/null | wc -l)
NUM_JSON=$(find "$PROJECT_DIR/bids_data" -name "*.json" 2>/dev/null | wc -l)
NUM_TSV=$(find "$PROJECT_DIR/bids_data" -name "*.tsv" 2>/dev/null | wc -l)

echo "NIfTI files (.nii.gz): $NUM_NII"
echo "JSON files: $NUM_JSON"
echo "TSV files: $NUM_TSV"

# Show total size
TOTAL_SIZE=$(du -sh "$PROJECT_DIR/bids_data" 2>/dev/null | awk '{print $1}')
echo "Total size: $TOTAL_SIZE"

echo ""
echo "=== Last 15 Lines of Log ==="
tail -15 "$LOG_FILE"
