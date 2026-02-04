#!/bin/bash
# Simple progress checker for CAT12 preprocessing

echo "=== CAT12 Preprocessing Progress ==="
echo ""
echo "Completed output files (mwp1*.nii):"
find /Users/karl/work/github/bids-cat12-wrapper/projects/demo/derivatives/cat12 -name "mwp1*.nii" 2>/dev/null | wc -l

echo ""
echo "Session folders created:"
ls -d /Users/karl/work/github/bids-cat12-wrapper/projects/demo/derivatives/cat12/sub-*/ses-* 2>/dev/null | wc -l

echo ""
echo "Latest processing (from CAT12 logs):"
find /Users/karl/work/github/bids-cat12-wrapper/projects/demo/derivatives/cat12 -name "catlog*.txt" -type f 2>/dev/null | tail -1 | xargs tail -3 2>/dev/null || echo "No logs yet"

echo ""
echo "Expected: 8 output files total (4 subjects × 2 sessions)"
