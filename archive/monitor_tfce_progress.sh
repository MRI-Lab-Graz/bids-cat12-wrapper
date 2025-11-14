#!/bin/bash
# Monitor TFCE progress by watching output file creation
# Usage: ./monitor_tfce_progress.sh <stats_folder> [interval_seconds]

if [ $# -lt 1 ]; then
    echo "Usage: $0 <stats_folder> [check_interval_seconds]"
    echo "Example: $0 s9_int_control 10"
    exit 1
fi

STATS_FOLDER="$1"
INTERVAL="${2:-5}"  # Default 5 seconds

if [ ! -d "$STATS_FOLDER" ]; then
    echo "Error: Folder not found: $STATS_FOLDER"
    exit 1
fi

echo "🔍 Monitoring TFCE progress in: $STATS_FOLDER"
echo "   Checking every $INTERVAL seconds... (Press Ctrl+C to stop)"
echo ""

last_count=0

while true; do
    # Count main TFCE files (TFCE_0001, TFCE_0002, etc.)
    count=$(ls "$STATS_FOLDER"/TFCE_[0-9]*.gii 2>/dev/null | wc -l)
    count=$((count + $(ls "$STATS_FOLDER"/TFCE_[0-9]*.nii 2>/dev/null | wc -l)))
    
    # Count log_p files
    log_p_count=$(ls "$STATS_FOLDER"/TFCE_log_p_[0-9]*.gii 2>/dev/null | wc -l)
    log_p_count=$((log_p_count + $(ls "$STATS_FOLDER"/TFCE_log_p_[0-9]*.nii 2>/dev/null | wc -l)))
    
    # Count FWE files
    fwe_count=$(ls "$STATS_FOLDER"/TFCE_log_pFWE_[0-9]*.gii 2>/dev/null | wc -l)
    fwe_count=$((fwe_count + $(ls "$STATS_FOLDER"/TFCE_log_pFWE_[0-9]*.nii 2>/dev/null | wc -l)))
    
    # Count FDR files
    fdr_count=$(ls "$STATS_FOLDER"/TFCE_log_pFDR_[0-9]*.gii 2>/dev/null | wc -l)
    fdr_count=$((fdr_count + $(ls "$STATS_FOLDER"/TFCE_log_pFDR_[0-9]*.nii 2>/dev/null | wc -l)))
    
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [ $count -ne $last_count ] || [ $log_p_count -gt 0 ]; then
        printf "\r[$timestamp] TFCE: %3d | log_p: %3d | FWE: %3d | FDR: %3d" $count $log_p_count $fwe_count $fdr_count
        last_count=$count
    fi
    
    # Check if all 4 file types are present (indicates completion)
    if [ $count -gt 0 ] && [ $log_p_count -eq $count ] && [ $fwe_count -eq $count ] && [ $fdr_count -eq $count ]; then
        printf "\n✅ TFCE processing complete!\n"
        printf "   Total contrasts processed: %d\n" $count
        printf "   Files generated: %d\n" $((count * 4))
        break
    fi
    
    sleep $INTERVAL
done
