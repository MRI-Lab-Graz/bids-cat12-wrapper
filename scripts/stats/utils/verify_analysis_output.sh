#!/bin/bash
# Verification script for CAT12 longitudinal analysis pipeline output
# Checks that all expected files were created at each step with reasonable sizes

OUTPUT_DIR="${1:-.}"
START_TIME="${2:-}"
REF_FILE=""

if [[ -n "$START_TIME" ]]; then
    # Create a reference file with the start timestamp
    REF_FILE=$(mktemp /tmp/pipeline_ref_XXXXXX)
    # Convert unix timestamp to format YYYYMMDDhhmm.ss for touch
    # macOS date -r takes seconds
    TOUCH_TIME=$(date -r "$START_TIME" +%Y%m%d%H%M.%S)
    touch -t "$TOUCH_TIME" "$REF_FILE"
    echo "Filtering results newer than: $(date -r "$START_TIME")"
    # Ensure cleanup on exit
    trap "rm -f '$REF_FILE'" EXIT
fi

if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "Error: Output directory not found: $OUTPUT_DIR"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║           Pipeline Output Verification & Quality Checks                ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

check_failed=0

# ============================================================================
# CHECK 1: SPM.mat file
# ============================================================================
echo "┌────────────────────────────────────────────────────────────────────────┐"
echo "│ CHECK 1: Statistical Model (SPM.mat)                                   │"
echo "└────────────────────────────────────────────────────────────────────────┘"

if [[ -f "$OUTPUT_DIR/SPM.mat" ]]; then
    # Check timestamp if reference file exists
    is_new=1
    if [[ -n "$REF_FILE" && "$OUTPUT_DIR/SPM.mat" -ot "$REF_FILE" ]]; then
        is_new=0
    fi

    if [[ $is_new -eq 1 ]]; then
        size=$(stat -f%z "$OUTPUT_DIR/SPM.mat" 2>/dev/null || stat -c%s "$OUTPUT_DIR/SPM.mat" 2>/dev/null)
        size=${size:-0}
        
        if command -v bc &> /dev/null; then
            size_mb=$(echo "scale=2; $size / 1048576" | bc)
        else
            size_mb="?"
        fi
        
        if (( size > 100000 )); then  # At least 100KB
            echo -e "${GREEN}✓ SPM.mat exists${NC}"
            echo "  Size: $size_mb MB"
            echo ""
            spm_status="✓"
        else
            echo -e "${RED}✗ SPM.mat exists but is suspiciously small${NC}"
            echo "  Size: $size_mb MB (expected > 0.1 MB)"
            check_failed=1
            spm_status="✗ (small)"
            echo ""
        fi
    else
        echo -e "${YELLOW}⚠ SPM.mat exists but is from a PREVIOUS run (older than start time)${NC}"
        check_failed=1
        spm_status="✗ (old)"
        echo ""
    fi
else
    echo -e "${RED}✗ SPM.mat NOT FOUND${NC}"
    check_failed=1
    spm_status="✗"
    echo ""
fi

# ============================================================================
# CHECK 2: Contrast files
# ============================================================================
echo "┌────────────────────────────────────────────────────────────────────────┐"
echo "│ CHECK 2: Contrast Maps (con_*.nii, spmT_*.nii, spmF_*.nii)             │"
echo "└────────────────────────────────────────────────────────────────────────┘"

FIND_OPTS=""
if [[ -n "$REF_FILE" ]]; then
    FIND_OPTS="-newer $REF_FILE"
fi

con_count=$(find "$OUTPUT_DIR" -maxdepth 1 -name "con_*.nii" $FIND_OPTS 2>/dev/null | wc -l)
spmt_count=$(find "$OUTPUT_DIR" -maxdepth 1 -name "spmT_*.nii" $FIND_OPTS 2>/dev/null | wc -l)
spmf_count=$(find "$OUTPUT_DIR" -maxdepth 1 -name "spmF_*.nii" $FIND_OPTS 2>/dev/null | wc -l)

echo "Contrast files found:"
echo "  - con_*.nii (raw contrasts):     $con_count files"
echo "  - spmT_*.nii (t-statistics):     $spmt_count files"
echo "  - spmF_*.nii (F-statistics):     $spmf_count files"
echo ""

total_con=$((con_count + spmt_count + spmf_count))
if (( total_con > 0 )); then
    # Check file sizes (Total size)
    # Note: du might pick up old files if we don't filter, but for simplicity we just check existence here
    # or we can try to be more precise.
    
    echo -e "${GREEN}✓ Contrast files present${NC}"
    echo "  Total: $total_con files"
    
    # List all contrasts
    echo ""
    echo "Contrasts (in order):"
    i=1
    # Use consistent glob pattern but filter
    find "$OUTPUT_DIR" -maxdepth 1 -name "spmT_*.nii" $FIND_OPTS 2>/dev/null | sort | while read -r f; do
        # Extract just the number from spmT_XXXX
        num=$(basename "$f" | sed 's/spmT_//' | sed 's/.nii//')
        # Try to read from screening file for descriptive names
        if [[ -f "$OUTPUT_DIR/screening_header_debug.txt" ]]; then
            # Look for the contrast description in the screening file
            description=$(grep -i "^contrast.*$num" "$OUTPUT_DIR/screening_header_debug.txt" 2>/dev/null | head -1 | sed 's/^[^:]*: *//')
            if [[ -z "$description" ]]; then
                description="Contrast $num"
            fi
        else
            description="Contrast $num"
        fi
        printf "  %2d. %s\n" "$i" "$description"
        ((i++))
    done
    echo ""
else
    echo -e "${RED}✗ NO CONTRAST FILES FOUND${NC}"
    echo "  This suggests the contrast step failed"
    check_failed=1
    echo ""
fi

# ============================================================================
# CHECK 3: Beta/parameter estimates
# ============================================================================
echo "┌────────────────────────────────────────────────────────────────────────┐"
echo "│ CHECK 3: Parameter Estimates (beta_*.nii)                              │"
echo "└────────────────────────────────────────────────────────────────────────┘"

beta_count=$(find "$OUTPUT_DIR" -maxdepth 2 -type f -name 'beta_*.nii' $FIND_OPTS 2>/dev/null | wc -l)

if (( beta_count > 0 )); then
    echo -e "${GREEN}✓ Beta parameter estimates exist${NC}"
    echo "  Count: $beta_count files"
    echo ""
else
    echo -e "${YELLOW}⚠ No beta files found${NC}"
    echo "  (This may be OK if model estimation succeeded)"
    echo ""
fi

# ============================================================================
# CHECK 4: Design matrix visualization
# ============================================================================
echo "┌────────────────────────────────────────────────────────────────────────┐"
echo "│ CHECK 4: Design Matrix Visualization                                   │"
echo "└────────────────────────────────────────────────────────────────────────┘"

DESIGN_PNG=""
if [[ -f "$OUTPUT_DIR/design_matrix.png" ]]; then
    if [[ -z "$REF_FILE" || "$OUTPUT_DIR/design_matrix.png" -nt "$REF_FILE" ]]; then
        DESIGN_PNG="$OUTPUT_DIR/design_matrix.png"
    fi
elif [[ -f "$OUTPUT_DIR/report/design_matrix.png" ]]; then
    if [[ -z "$REF_FILE" || "$OUTPUT_DIR/report/design_matrix.png" -nt "$REF_FILE" ]]; then
        DESIGN_PNG="$OUTPUT_DIR/report/design_matrix.png"
    fi
fi

if [[ -n "$DESIGN_PNG" && -f "$DESIGN_PNG" ]]; then
    echo -e "${GREEN}✓ Design matrix image generated${NC}"
    echo "  File: $(basename "$DESIGN_PNG") (location: ${DESIGN_PNG#$OUTPUT_DIR/})"
    echo ""
else
    echo -e "${YELLOW}⚠ Design matrix image not found${NC}"
    echo "  Checked: $OUTPUT_DIR/design_matrix.png and $OUTPUT_DIR/report/design_matrix.png"
    echo ""
fi

# ============================================================================
# CHECK 5: Screening results
# ============================================================================
echo "┌────────────────────────────────────────────────────────────────────────┐"
echo "│ CHECK 5: Screening Results (initial p<0.001 thresholding)              │"
echo "└────────────────────────────────────────────────────────────────────────┘"

if [[ -f "$OUTPUT_DIR/screening_results.mat" ]]; then
    # Check timestamp
    is_new=1
    if [[ -n "$REF_FILE" && "$OUTPUT_DIR/screening_results.mat" -ot "$REF_FILE" ]]; then
        is_new=0
    fi

    if [[ $is_new -eq 1 ]]; then
        echo -e "${GREEN}✓ Screening results file exists${NC}"
        echo "  File: screening_results.mat"
        echo ""
        
        # Count screened contrasts (those that passed p<0.001 uncorrected)
        screened_count=0
        if [[ -f "$OUTPUT_DIR/screening_header_debug.txt" ]]; then
            echo "  Contrasts passing p<0.001 uncorrected threshold:"
            # Extract lines that show which contrasts passed screening
            grep -i "passed\|significant\|voxels" "$OUTPUT_DIR/screening_header_debug.txt" 2>/dev/null | head -20 | while read line; do
                echo "    $line"
            done
            
            # Count unique contrasts in screening file
            screened_count=$(grep -i "contrast" "$OUTPUT_DIR/screening_header_debug.txt" 2>/dev/null | wc -l)
            if (( screened_count > 0 )); then
                echo ""
                echo "  Found $screened_count contrasts with voxels in screening data"
                echo ""
            fi
        fi
        
        if (( screened_count == 0 )); then
            echo "  (No screening details available or no voxels passed threshold)"
            echo ""
        fi
    else
        echo -e "${YELLOW}⚠ Screening results exist but are OLDER than pipeline start time${NC}"
        echo ""
    fi
else
    echo -e "${YELLOW}⚠ Screening results not found${NC}"
    echo "  This may indicate no voxels passed the p<0.001 uncorrected threshold"
    echo ""
fi

# ============================================================================
# CHECK 6: TFCE results
# ============================================================================
echo "┌────────────────────────────────────────────────────────────────────────┐"
echo "│ CHECK 6: TFCE Permutation Testing (FWE correction)                     │"
echo "└────────────────────────────────────────────────────────────────────────┘"

# Search TFCE outputs in either the output root or the report subfolder (some report generators place assets there)
# Use case-insensitive search (-iname) to handle TFCE vs tfce naming variations
# Updated to match TFCE_log_pFWE_*.nii pattern as well
tfce_count=$(find "$OUTPUT_DIR" -maxdepth 2 -type f \( -iname 'tfce_*_fwe.nii' -o -iname '*_log_pfwe*.nii' \) $FIND_OPTS 2>/dev/null | wc -l)
tfce_uncorr=$(find "$OUTPUT_DIR" -maxdepth 2 -type f -iname 'tfce_*.nii' $FIND_OPTS 2>/dev/null | grep -vi '_fwe.nii' | grep -vi '_log_pfwe' | wc -l)

if (( tfce_count > 0 )); then
    echo -e "${GREEN}✓ TFCE results found (FWE-corrected)${NC}"
    echo "  FWE-corrected maps found: $tfce_count"
    echo ""
    echo "  Contrasts with FWE-corrected maps:"
    find "$OUTPUT_DIR" -maxdepth 2 -type f \( -iname 'tfce_*_fwe.nii' -o -iname '*_log_pfwe*.nii' \) $FIND_OPTS 2>/dev/null | while read -r f; do
        base=$(basename "$f")
        # Clean up label for display
        label=$(echo "$base" | sed 's/tfce_//I; s/_fwe.nii//I; s/_log_pfwe//I; s/.nii//I; s/^_//')
        echo "    - $label    (path: ${f#$OUTPUT_DIR/})"
    done
    echo ""
elif (( tfce_uncorr > 0 )); then
    echo -e "${YELLOW}⚠ TFCE files exist but NO SIGNIFICANT RESULTS (p<0.05 FWE)${NC}"
    echo "  Total TFCE outputs (unthresholded): $tfce_uncorr"
    echo "  Note: TFCE outputs were produced but no voxels survived FWE correction."
    echo "  This is expected for pilot analyses or when effects are subtle."
    echo "  TFCE files were searched under: $OUTPUT_DIR and $OUTPUT_DIR/report/"
    echo ""
else
    echo -e "${YELLOW}⚠ No TFCE results found${NC}"
    echo "  Possible reasons:" 
    echo "    - No voxels passed initial screening (p<0.001 uncorrected) so TFCE was not executed" 
    echo "    - TFCE permutation test did not produce any outputs (error or misconfiguration)" 
    echo "    - Report assets may have been placed elsewhere; check $OUTPUT_DIR/report/ and logs/" 
    echo ""
fi

# ============================================================================
# CHECK 7: Residual variance map
# ============================================================================
echo "┌────────────────────────────────────────────────────────────────────────┐"
echo "│ CHECK 7: Residual Variance Map (ResMS.nii)                             │"
echo "└────────────────────────────────────────────────────────────────────────┘"

if [[ -f "$OUTPUT_DIR/ResMS.nii" ]]; then
    if [[ -n "$REF_FILE" && "$OUTPUT_DIR/ResMS.nii" -ot "$REF_FILE" ]]; then
        echo -e "${YELLOW}⚠ ResMS.nii exists but is OLDER than pipeline start time${NC}"
        echo ""
    else
        echo -e "${GREEN}✓ Residual variance map exists${NC}"
        echo "  File: ResMS.nii (indicates model fit quality)"
        echo ""
    fi
else
    echo -e "${YELLOW}⚠ ResMS.nii not found${NC}"
    echo ""
fi

# ============================================================================
# CHECK 8: HTML Report
# ============================================================================
echo "┌────────────────────────────────────────────────────────────────────────┐"
echo "│ CHECK 8: HTML Report                                                   │"
echo "└────────────────────────────────────────────────────────────────────────┘"

if [[ -f "$OUTPUT_DIR/report.html" ]]; then
    if [[ -n "$REF_FILE" && "$OUTPUT_DIR/report.html" -ot "$REF_FILE" ]]; then
        echo -e "${YELLOW}⚠ HTML report exists but is OLDER than pipeline start time${NC}"
        echo ""
    else
        echo -e "${GREEN}✓ HTML report generated${NC}"
        echo "  File: report.html"
        echo ""
    fi
else
    echo -e "${YELLOW}⚠ HTML report not found${NC}"
    echo ""
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                          VERIFICATION SUMMARY                          ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

echo "Output directory: $OUTPUT_DIR"
echo ""
echo "File counts:"
echo "  - SPM.mat:           ${spm_status:-✗}"
echo "  - Contrasts:         $total_con files"
echo "  - Beta estimates:    $beta_count files"
echo "  - Design matrix:     $(test -n "$DESIGN_PNG" && echo "✓" || echo "✗")"

SCREENING_OK="✗"
if [[ -f "$OUTPUT_DIR/screening_results.mat" ]]; then
    if [[ -z "$REF_FILE" || "$OUTPUT_DIR/screening_results.mat" -nt "$REF_FILE" ]]; then
        SCREENING_OK="✓"
    else
        SCREENING_OK="✗ (old)"
    fi
fi
echo "  - Screening results: $SCREENING_OK"

echo "  - TFCE significant:  $tfce_count contrasts"
# Report file may live at top-level or in the `report/` subfolder
REPORT_HTML=""
if [[ -f "$OUTPUT_DIR/report.html" ]]; then
    if [[ -z "$REF_FILE" || "$OUTPUT_DIR/report.html" -nt "$REF_FILE" ]]; then
        REPORT_HTML="$OUTPUT_DIR/report.html"
    fi
elif [[ -f "$OUTPUT_DIR/report/report.html" ]]; then
    if [[ -z "$REF_FILE" || "$OUTPUT_DIR/report/report.html" -nt "$REF_FILE" ]]; then
        REPORT_HTML="$OUTPUT_DIR/report/report.html"
    fi
fi
echo "  - HTML report:       $(test -n "$REPORT_HTML" && echo "✓" || echo "✗")"
echo ""

# Overall status
if (( check_failed == 0 )); then
    echo -e "${GREEN}✓ Pipeline output verification PASSED${NC}"
    echo "  All critical files present and reasonable sizes"
    exit 0
else
    echo -e "${RED}✗ Pipeline output verification FAILED${NC}"
    echo "  Some critical files missing or too small"
    exit 1
fi
