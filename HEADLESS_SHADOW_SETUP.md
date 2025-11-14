# Headless MATLAB Shadow Functions - Setup Complete

## Summary
This document confirms that all necessary shadow functions have been restored to enable proper headless MATLAB operation in the CAT12 longitudinal analysis pipeline.

## Shadow Functions in Place

### 1. `utils/spm_input.m` ✅
- **Purpose**: Intercepts `spm_input()` function calls
- **Behavior**: Returns default values without showing dialog
- **Key lines**: Returns 6th argument (default) if provided; otherwise returns first choice from options
- **Restored from**: `archive/spm_headless_overrides/spm_input.m`

### 2. `utils/questdlg.m` ✅
- **Purpose**: Intercepts `questdlg()` function calls
- **Behavior**: Returns first button option without showing dialog (default: 'Yes')
- **Key lines**: Returns `varargin{3}` (first button) or 'Yes' as fallback
- **Restored from**: `archive/questdlg.m`

### 3. `utils/uiconfirm.m` ✅
- **Purpose**: Intercepts `uiconfirm()` function calls
- **Behavior**: Returns first option without showing dialog
- **Key lines**: Returns first element of options or 'OK' as fallback
- **Restored from**: `archive/uiconfirm.m`

### 4. `utils/input.m` ✅ (NEW)
- **Purpose**: Intercepts `input()` function calls used in interactive scripts
- **Behavior**: Returns sensible defaults:
  - For yes/no questions: returns 'y'
  - For other string prompts: returns ''
  - For numeric prompts: returns 0
- **Key logic**: Detects headless environment; only shadows in headless mode
- **Created**: Just now to handle `input()` calls in:
  - `configure_spm_path.m` (lines 75, 153)
  - `find_spm_path.m` (lines 114, 137)

### 5. `utils/spm_figure.m` ✅ (Already in place)
- **Purpose**: Prevents GUI figure window creation
- **Behavior**: Empty stub that prevents figure rendering
- **Pre-existing**: Was already in utils/ directory

## MATLAB Path Configuration

The pipeline automatically ensures these shadow functions are on the MATLAB path BEFORE SPM paths. This is critical because MATLAB searches the path in order, and we need our shadows to intercept calls before they reach built-in functions.

The path ordering in `configure_spm_path.m`:
1. First: `$STATS_DIR/utils` (contains all shadow functions)
2. Second: `$STATS_DIR` (pipeline scripts)
3. Third: SPM installation directory
4. Fourth: User's additional paths

## Pipeline Integration

The `cat12_longitudinal_analysis.sh` pipeline has been updated with:

### Enhanced Diagnostics
- **Diary logging** for model estimation step:
  - Captures all MATLAB console output to: `$OUTPUT_DIR/logs/matlab_model_estimation.log`
  - Provides stack traces and error messages if step fails
  
- **Diary logging** for contrast step:
  - Captures all MATLAB console output to: `$OUTPUT_DIR/logs/matlab_contrasts.log`
  - Enables diagnosis of contrast creation issues

- **Error reporting** in pipeline:
  - On failure, pipeline now indicates: `Check MATLAB log: $LOG_DIR/matlab_model_estimation.log`
  - User can immediately inspect detailed error logs

### Headless Suppression Settings
Both MATLAB steps now include:
```matlab
warning('off','all');
set(0,'DefaultFigureVisible','off');
set(0,'DefaultFigureCreateFcn',@(h,ev)[]);
beep off;
```
These prevent any figure windows or audio notifications from blocking execution.

## Why This Was Failing

The `s6_vbm_test2` pipeline run produced only `design.json` and `spm_batch.m` but no `SPM.mat` or contrasts because:

1. **Shadow functions were archived**: When interactive MATLAB functions (`spm_input`, `questdlg`, `uiconfirm`, `input`) were called in headless mode, MATLAB couldn't find the shadow versions (they were in `archive/`).

2. **Built-in functions took over**: MATLAB fell back to built-in implementations that tried to create dialog boxes.

3. **Silent failure in headless**: Since there's no terminal to show the dialogs, these calls either timed out or failed silently, causing SPM batch execution to abort before generating results.

4. **Previous success unexplained**: The `s6_vbm_test` run succeeded because it completed before the shadow functions were archived (or they were somehow still in the path).

## Testing & Verification

To verify the fix works:

1. Re-run the pipeline with the same parameters:
   ```bash
   ./cat12_longitudinal_analysis.sh \
     --cat12-dir /Volumes/Thunder/129_PK01/cat12 \
     --participants participants.tsv \
     --smoothing 6 \
     --force \
     --pilot \
     --cluster-size 60 \
     --uncorrected-p 0.001 \
     --analysis-name "s6_vbm_test3"
   ```

2. Monitor the run - should take 10-15 minutes

3. Check results folder for:
   - `results/vbm/s6_vbm_test3/SPM.mat` (should exist)
   - `results/vbm/s6_vbm_test3/con_*.nii` files (contrast images)
   - `results/vbm/s6_vbm_test3/spmT_*.nii` files (t-statistic images)

4. If issues persist, inspect logs:
   - `results/vbm/s6_vbm_test3/logs/matlab_model_estimation.log`
   - `results/vbm/s6_vbm_test3/logs/matlab_contrasts.log`

## File Locations

All shadow functions are now in:
```
/Volumes/Thunder/129_PK01/cat12/stats/utils/
├── input.m              [NEW]
├── spm_figure.m         [EXISTING]
├── spm_input.m          [RESTORED from archive]
├── questdlg.m           [RESTORED from archive]
└── uiconfirm.m          [RESTORED from archive]
```

Archived backup copies remain in:
```
/Volumes/Thunder/129_PK01/cat12/stats/archive/
├── spm_input.m          [OLD - no longer used]
├── questdlg.m           [OLD - no longer used]
├── uiconfirm.m          [OLD - no longer used]
└── spm_headless_overrides/
    ├── spm_figure.m     [OLD - no longer used]
    └── spm_input.m      [OLD - no longer used]
```

## Next Steps

1. **Run the pipeline** with all shadow functions in place
2. **Inspect logs** if any issues occur
3. **Verify SPM.mat and contrasts** are generated
4. **Document any remaining issues** for diagnosis

The pipeline should now complete successfully with proper contrast generation!
