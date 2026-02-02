#!/bin/bash
#
# Re-run CAT12 with Surface Extraction for Missing Subjects
# Uses existing T1w files in cat12/data/cat12/sub-*/
#

set -euo pipefail

CAT12_DATA_DIR="/Volumes/Thunder/129_PK01/cat12/data/cat12"
STATS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATLAB_EXE="/Applications/MATLAB_R2025b.app/bin/matlab"
SPM_DIR="/Volumes/Evo/software/spm25"

# Read missing subjects
if [[ ! -f "$STATS_DIR/missing_surface_subjects.txt" ]]; then
    echo "Error: missing_surface_subjects.txt not found!"
    echo "Run the Python script first to generate this file."
    exit 1
fi

MISSING_SUBJECTS=($(cat "$STATS_DIR/missing_surface_subjects.txt"))
echo "Found ${#MISSING_SUBJECTS[@]} subjects missing surface data"
echo ""

# Process each subject
for SUBJECT_ID in "${MISSING_SUBJECTS[@]}"; do
    echo "════════════════════════════════════════════════════════════════"
    echo "Processing: sub-$SUBJECT_ID"
    echo "════════════════════════════════════════════════════════════════"
    
    SUBJECT_DIR="$CAT12_DATA_DIR/sub-$SUBJECT_ID"
    
    if [[ ! -d "$SUBJECT_DIR" ]]; then
        echo "⚠️  Subject directory not found: $SUBJECT_DIR"
        continue
    fi
    
    # Find T1w files for this subject
    T1W_FILES=($(find "$SUBJECT_DIR" -maxdepth 1 -name "sub-${SUBJECT_ID}_ses-*_acq-mprage_T1w.nii" -o -name "sub-${SUBJECT_ID}_ses-*_T1w.nii"))
    
    if [[ ${#T1W_FILES[@]} -eq 0 ]]; then
        echo "⚠️  No T1w files found for sub-$SUBJECT_ID"
        continue
    fi
    
    echo "Found ${#T1W_FILES[@]} T1w file(s):"
    for f in "${T1W_FILES[@]}"; do
        echo "  - $(basename $f)"
    done
    
    # Create MATLAB batch script
    BATCH_SCRIPT="$SUBJECT_DIR/cat12_surface_batch_${SUBJECT_ID}.m"
    
    cat > "$BATCH_SCRIPT" << 'EOFMATLAB'
% CAT12 Surface Re-processing Batch
% Auto-generated script

% Add SPM/CAT12 to path
addpath(genpath('SPM_DIR_PLACEHOLDER'));
spm('defaults', 'FMRI');
spm_jobman('initcfg');

% Input files
matlabbatch{1}.spm.tools.cat.estwrite.data = {
FILES_PLACEHOLDER
};

% Standard CAT12 settings
matlabbatch{1}.spm.tools.cat.estwrite.opts.tpm = {fullfile(spm('dir'),'tpm','TPM.nii')};
matlabbatch{1}.spm.tools.cat.estwrite.opts.affreg = 'mni';
matlabbatch{1}.spm.tools.cat.estwrite.opts.biasstr = 0.5;

% ENABLE SURFACE EXTRACTION
matlabbatch{1}.spm.tools.cat.estwrite.output.surface = 1;

% Output options
matlabbatch{1}.spm.tools.cat.estwrite.output.GM.native = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.GM.mod = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.GM.dartel = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.WM.native = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.WM.mod = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.WM.dartel = 0;

% Run
fprintf('Starting CAT12 surface processing...\n');
spm_jobman('run', matlabbatch);
fprintf('CAT12 processing completed\n');
exit;
EOFMATLAB
    
    # Replace placeholders
    sed -i '' "s|SPM_DIR_PLACEHOLDER|$SPM_DIR|g" "$BATCH_SCRIPT"
    
    # Build file list for MATLAB and replace placeholder via Python (safe multiline)
    python3 - "$BATCH_SCRIPT" "${T1W_FILES[@]}" << 'PYEOF'
import sys

batch_path = sys.argv[1]
with open(batch_path, "r", encoding="utf-8") as f:
    content = f.read()

file_list = "\n".join([f"    '{t1w},1'" for t1w in sys.argv[2:]])
if not file_list:
    file_list = "    ''"

content = content.replace("FILES_PLACEHOLDER", file_list)

with open(batch_path, "w", encoding="utf-8") as f:
    f.write(content)
PYEOF
    
    echo ""
    echo "Running CAT12 with surface extraction..."
    
    # Run MATLAB batch
    "$MATLAB_EXE" -nodisplay -nosplash -nodesktop -r "run('$BATCH_SCRIPT')" 2>&1 | grep -E "CAT12|surface|thickness|Error|Warning|^[0-9]" || true
    
    # Check if surface files were created
    if ls "$SUBJECT_DIR/surf/"*.gii 1> /dev/null 2>&1; then
        echo "✓ Surface files created for sub-$SUBJECT_ID"
    else
        echo "✗ No surface files found - processing may have failed"
    fi
    
    echo ""
done

echo "════════════════════════════════════════════════════════════════"
echo "Surface re-processing complete!"
echo "════════════════════════════════════════════════════════════════"
