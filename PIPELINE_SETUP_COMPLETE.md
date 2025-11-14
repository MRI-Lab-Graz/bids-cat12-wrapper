# CAT12 Longitudinal Analysis Pipeline - Setup Complete ✓

## Summary

The CAT12 longitudinal analysis pipeline is now **fully functional** in headless mode. The pipeline successfully:

1. ✅ Parses participants and CAT12 data files
2. ✅ Generates SPM factorial design specifications
3. ✅ Runs model estimation (creates SPM.mat)
4. ✅ Generates contrasts (t-tests and F-tests)
5. ✅ Performs initial uncorrected screening (p<0.001)
6. ✅ Runs TFCE permutation testing for FWE correction
7. ✅ Generates HTML reports with visualizations
8. ✅ Verifies output quality

## Key Fixes Applied

### 1. Headless MATLAB Shadow Functions

Created shadow functions in `utils/` to intercept interactive UI calls that would fail in headless mode:

- **`spm_figure.m`** - Prevents figure window creation
- **`spm_progress_bar.m`** - Suppresses progress bar display
- **`spm_input.m`** - Returns defaults instead of showing dialogs
- **`questdlg.m`** - Confirms dialogs (especially "Overwrite SPM.mat?")
- **`uiconfirm.m`** - Handles confirmation dialogs
- **`input.m`** - Intercepts MATLAB's `input()` prompts

### 2. SPM.mat Overwrite Prevention

Added logic to delete any existing `SPM.mat` before model estimation to prevent the "Overwrite?" dialog that was blocking headless execution.

### 3. Figure Suppression

Set MATLAB global figure defaults at startup:
```matlab
set(0,'DefaultFigureVisible','off');
set(0,'DefaultFigureCreateFcn',@(h,ev)[]);
```

### 4. Warning Suppression

Suppressed the "Function input has the same name as a MATLAB built-in" warning:
```matlab
warning('off','MATLAB:dispatcher:nameConflict');
```

## Usage

### Run a full analysis:

```bash
./cat12_longitudinal_analysis.sh \
  --cat12-dir /Volumes/Thunder/129_PK01/cat12 \
  --participants participants.tsv \
  --smoothing 6 \
  --force \
  --cluster-size 60 \
  --uncorrected-p 0.001 \
  --analysis-name "my_analysis_name"
```

### Verify results:

```bash
./utils/verify_analysis_output.sh results/vbm/my_analysis_name
```

## Output Verification

The pipeline now includes a comprehensive verification script (`verify_analysis_output.sh`) that checks:

✓ **SPM.mat** - Statistical model exists and has reasonable size (>100KB)
✓ **Contrasts** - All contrast maps present (con_*.nii, spmT_*.nii, spmF_*.nii)
✓ **Parameters** - Beta parameter estimates (beta_*.nii)
✓ **Design** - Design matrix visualization
✓ **Screening** - Initial uncorrected screening results
✓ **TFCE** - Permutation-corrected results (if any voxels pass FWE)
✓ **Residuals** - Residual variance map (ResMS.nii)
✓ **Report** - HTML analysis report

### Verification Output Example:

```
┌────────────────────────────────────────────────────────────────────────┐
│ CHECK 2: Contrast Maps                                                 │
└────────────────────────────────────────────────────────────────────────┘
✓ Contrast files present
  Total: 78 files (37 con + 37 spmT + 4 spmF)

Contrasts (in order):
   1. Contrast 0005
   2. Contrast 0006
   ...
  37. Contrast 0041
```

## Test Results

**Latest successful run**: `s6_vbm_verification_test2`

```
- SPM.mat:               ✓ (1.27 MB)
- Contrasts:             ✓ (78 files)
- Beta estimates:        ✓ (9 files)
- Residual variance:     ✓ (ResMS.nii)
- HTML report:           ✓
- TFCE results:          No significant voxels (expected with pilot mode)
```

## File Structure

All shadow functions are in `utils/`:
```
utils/
├── spm_figure.m              ← Prevents figure creation
├── spm_progress_bar.m        ← Suppresses progress display
├── spm_input.m               ← Handles SPM input dialogs
├── questdlg.m                ← Handles question dialogs
├── uiconfirm.m               ← Handles confirmation dialogs
├── input.m                   ← Intercepts input() calls
├── verify_analysis_output.sh ← Verification script (NEW)
├── configure_spm_path.m
├── add_contrasts_longitudinal.m
└── ... (other utilities)
```

## Troubleshooting

### "No voxels passed screening" / "No TFCE results"

This is **not an error**. It means:
- The effect sizes are subtle
- The analysis needs more power (more subjects, more permutations, or lower p-threshold)
- Try removing `--pilot` flag to use full permutation testing (5000 instead of 100)

### "spm_input has the same name as a MATLAB built-in"

This is expected and harmless. The warning is suppressed with `warning('off','MATLAB:dispatcher:nameConflict')`.

### Pipeline takes a long time

This is normal:
- Model estimation: ~2-5 minutes
- TFCE with 5000 permutations: ~10-30 minutes depending on cluster size
- Use `--pilot` flag for quick testing (100 permutations instead of 5000)

## Configuration

See `config.ini` for settings:
- MATLAB executable path
- SPM installation path
- Python executable
- TFCE permutation count
- Default analysis parameters

## Next Steps

1. **Run full analysis** (without `--pilot` for publication-ready results):
   ```bash
   ./cat12_longitudinal_analysis.sh ... --analysis-name "final_analysis"
   ```

2. **Inspect results**:
   - Open `results/vbm/final_analysis/report.html` in browser
   - Check beta maps for anatomical plausibility
   - Review contrast maps (spmT_*.nii) in SPM viewer

3. **Export for presentation**:
   - Use design_matrix.png for methods section
   - Use report.html for supplementary materials
   - Reference spm_batch.m for reproducibility

## Document History

- **2025-11-04**: Pipeline fully functional, all headless issues resolved
  - Shadow functions created and integrated
  - SPM.mat overwrite dialog prevented
  - Output verification script added
  - Comprehensive documentation provided

---

**Status**: ✅ **PRODUCTION READY**

The pipeline is now suitable for running large-scale longitudinal analyses without user interaction in batch/cluster environments.
