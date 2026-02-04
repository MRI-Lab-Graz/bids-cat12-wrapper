#!/bin/bash
# Monitor the automated CAT12 pipeline progress

LOG_FILE="/data/local/software/cat-12/project_complete_run.log"

echo "=================================="
echo "CAT12 Pipeline Status Monitor"
echo "=================================="
echo ""

# Check if process is running
if ps aux | grep -q "[p]roject_runner.py"; then
    echo "✓ Process is RUNNING"
    PID=$(ps aux | grep "[p]roject_runner.py" | awk '{print $2}')
    echo "  PID: $PID"
    echo "  CPU: $(ps aux | grep "[p]roject_runner.py" | awk '{print $3}')%"
    echo "  MEM: $(ps aux | grep "[p]roject_runner.py" | awk '{print $4}')%"
else
    echo "✗ Process is NOT running"
fi

echo ""
echo "=================================="
echo "Latest Log Entries:"
echo "=================================="

if [ -f "$LOG_FILE" ]; then
    tail -30 "$LOG_FILE"
    echo ""
    echo "=================================="
    echo "Progress Summary:"
    echo "=================================="
    
    # Count completed phases
    grep -c "✓ All preflight checks PASSED" "$LOG_FILE" && echo "  ✓ Preflight checks" || echo "  ⏳ Preflight checks"
    grep -c "✓ OpenNeuro download complete" "$LOG_FILE" && echo "  ✓ Data download" || echo "  ⏳ Data download"
    grep -c "✓ Preprocessing complete" "$LOG_FILE" && echo "  ✓ Preprocessing" || echo "  ⏳ Preprocessing"
    grep -c "✓ Statistics phase complete" "$LOG_FILE" && echo "  ✓ Statistics" || echo "  ⏳ Statistics"
    grep -c "✓ PROJECT COMPLETED SUCCESSFULLY" "$LOG_FILE" && echo "  ✓ COMPLETE" || echo "  ⏳ In progress"
    
else
    echo "Log file not found: $LOG_FILE"
fi

echo ""
echo "To monitor in real-time: tail -f $LOG_FILE"
