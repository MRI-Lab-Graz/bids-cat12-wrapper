#!/bin/bash
# QUICK REFERENCE CARD - CAT12 Longitudinal Analysis Pipeline

cat << 'EOF'

╔════════════════════════════════════════════════════════════════════════╗
║              CAT12 PIPELINE - QUICK REFERENCE                         ║
╚════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BASIC USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Standard analysis (full TFCE, 5000 permutations):
   
   ./cat12_longitudinal_analysis.sh \
     --cat12-dir /path/to/cat12/data \
     --participants participants.tsv \
     --smoothing 6 \
     --force \
     --analysis-name "my_analysis"

2. Quick test (pilot mode, 100 permutations):
   
   ./cat12_longitudinal_analysis.sh \
     --cat12-dir /path/to/cat12/data \
     --participants participants.tsv \
     --smoothing 6 \
     --force \
     --pilot \
     --analysis-name "test_analysis"

3. With covariates:
   
   ./cat12_longitudinal_analysis.sh \
     --cat12-dir /path/to/cat12/data \
     --participants participants.tsv \
     --smoothing 6 \
     --covariates age,sex,tiv \
     --analysis-name "with_covariates"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMON OPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  --cat12-dir PATH           Path to CAT12 segmentation directory (required)
  --participants FILE        Participants TSV file (required)
  --smoothing N              Smoothing kernel in mm (6, 8, 12, etc.)
  --force                    Delete existing results before starting
  --pilot                    Quick test with 100 TFCE permutations
  --analysis-name NAME       Output folder name (required)
  --cluster-size N           Minimum cluster size for screening (default: 50)
  --uncorrected-p P          Threshold for initial screening (default: 0.001)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VERIFY RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After analysis completes, verify output quality:

   ./utils/verify_analysis_output.sh results/vbm/my_analysis

This checks:
  ✓ SPM.mat exists and has reasonable size
  ✓ All contrast files present
  ✓ Beta estimates generated
  ✓ Screening results available
  ✓ TFCE results (if any significant voxels)
  ✓ HTML report generated

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FILES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

results/vbm/my_analysis/
├── report.html               ← Open in browser for results summary
├── SPM.mat                   ← Statistical model
├── design_matrix.png         ← Design visualization
├── con_*.nii                 ← Raw contrast maps (37 files)
├── spmT_*.nii                ← T-statistics (37 files)
├── spmF_*.nii                ← F-statistics (4 files)
├── beta_*.nii                ← Parameter estimates
├── ResMS.nii                 ← Residual variance map
├── RPV.nii                   ← Estimated variance ratio
├── screening_results.mat     ← Initial screening (p<0.001 uncorrected)
├── tfce_*_fwe.nii            ← FWE-corrected results (if significant)
└── spm_batch.m               ← Reproducible batch file

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTERPRETING RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"No TFCE results found"
→ Not an error. Means no voxels survived FWE correction.
→ Try removing --pilot for more power
→ Or inspect con_*.nii files for uncorrected effects

"Screening results not found"
→ No voxels passed p<0.001 uncorrected threshold
→ Try lowering --uncorrected-p (e.g., 0.01 or 0.05)

"SPM.mat created successfully"
✓ Model estimation worked. Check contrast files next.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Problem: Pipeline exits early
Solution: Check that paths exist:
  • CAT12 files in --cat12-dir
  • participants.tsv exists and is valid format

Problem: "MATLAB not found"
Solution: Set MATLAB path in config.ini:
  [MATLAB]
  exe = /Applications/MATLAB_R2025b.app/bin/matlab

Problem: "SPM not found"
Solution: Pipeline auto-detects SPM. Check:
  ./utils/configure_spm_path.m

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HEADLESS NOTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This pipeline runs in fully headless mode (-nodisplay):
  ✓ No MATLAB GUI required
  ✓ No user interaction needed
  ✓ Suitable for batch/cluster execution
  ✓ All dialogs automatically answered
  ✓ All figures suppressed

Shadow functions in utils/ handle:
  • Interactive input prompts
  • Confirmation dialogs
  • Progress bars
  • Figure creation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📖 For full documentation, see: PIPELINE_SETUP_COMPLETE.md

EOF
