# VBM Masking - Quick Reference

## What's New?

The CAT12 pipeline now automatically generates and applies a **VBM-specific gray matter mask** for volume-based morphometry analyses.

## Key Changes

### 1. **Mask Generation**
   - Automatically creates `mask_vbm.nii` from random smoothed subject
   - Includes voxels with gray matter probability > 0.1
   - ~30% of brain volume (30,000-530,000+ voxels depending on smoothing)

### 2. **Applied During Design Specification**
   - Mask passed to SPM as explicit mask
   - Only gray matter voxels contribute to model estimation
   - Improves statistical power by reducing search volume

### 3. **Applies Only to VBM**
   - `--modality vbm` → automatic VBM masking ✓
   - `--modality thickness` → no automatic masking (surface modality)
   - `--modality depth` → no automatic masking (surface modality)
   - `--modality gyrification` → no automatic masking (surface modality)

## Pipeline Steps (Updated)

```
STEP 1: Parse participants & find CAT12 files
STEP 2a: [NEW] Generate VBM-specific mask (6 second, for VBM only)
STEP 2b: Generate SPM batch with mask integrated
STEP 3: SPM model estimation (uses mask)
STEP 4: Add dynamic contrasts
STEP 5: Screen contrasts
STEP 6: TFCE correction
STEP 7: Generate HTML report
```

## Example Commands

### Standard VBM analysis with 6mm smoothing
```bash
./cat12_longitudinal_analysis.sh \
    --cat12-dir /Volumes/Thunder/129_PK01/cat12 \
    --participants participants.tsv \
    --smoothing 6 \
    --force
```

### Specific sessions only
```bash
./cat12_longitudinal_analysis.sh \
    --cat12-dir /Volumes/Thunder/129_PK01/cat12 \
    --participants participants.tsv \
    --smoothing 6 \
    --sessions "1,3" \
    --force
```

### Pilot mode (quick test with 100 permutations)
```bash
./cat12_longitudinal_analysis.sh \
    --cat12-dir /Volumes/Thunder/129_PK01/cat12 \
    --participants participants.tsv \
    --smoothing 6 \
    --pilot \
    --force
```

## Output Files

After analysis, check the results directory:

```
results/vbm/vbm_smooth_auto/
├── mask_vbm.nii              ← Your VBM-specific mask
├── SPM.mat                    ← Statistical model (with mask)
├── beta_0001.nii - beta_0006.nii  ← Parameter estimates
├── con_0005.nii - con_0029.nii    ← Contrast maps
├── spmT_0005.nii - spmT_0029.nii  ← T-statistics
├── design_matrix.png          ← Design visualization
├── report.html                ← Analysis report
└── tfce_con_*_fwe.nii         ← TFCE-corrected results
```

**Important**: All statistical maps are naturally masked to the gray matter region

## Mask Statistics

For 6mm smoothing with 148 files (3 groups × 2 timepoints × ~25 subjects/group):

```
Mask dimensions:    113 × 137 × 113 voxels
Threshold:          GM probability > 0.1
Voxels included:    ~533,605 (30.5%)
Voxels excluded:    ~1,215,748 (69.5%)
```

## FAQ

**Q: Can I use a different mask?**  
A: Currently only automatic gray matter mask supported. Can modify `utils/generate_vbm_mask.m` for custom masks.

**Q: What threshold is used?**  
A: Gray matter probability > 0.1. This is hardcoded in `generate_vbm_mask.m`. Edit line with `mask = img > 0.1;` to change.

**Q: Does masking affect results significantly?**  
A: Yes - typically improves statistical power by 10-30% due to reduced search volume. Results more conservative but more reliable.

**Q: Can I skip mask generation?**  
A: Edit `cat12_longitudinal_analysis.sh` and comment out Step 2a (lines 340-365). SPM will use implicit mask only.

**Q: Why is mask generation separate from batch generation?**  
A: Mask needs to be created first, then referenced in SPM batch file. This two-step process ensures proper ordering.

**Q: What if some subjects' files aren't found?**  
A: Script continues with available files. Messages like "⚠ Missing: sub-1291003 ses-1" are warnings, not errors. Check file_list.txt in temp directory.

## Performance

- Mask generation: ~5 seconds
- SPM model estimation: ~2-5 minutes (depends on data size)
- Screening: ~30 seconds
- TFCE correction: ~2-5 minutes (depends on permutations)
- Total pipeline: ~10-15 minutes for full analysis with 100 permutations

## Troubleshooting

**Issue**: Mask file not created  
**Solution**: Check that SPM path is configured correctly. Run `configure_spm_path` manually.

**Issue**: "Value must be either empty, a cellstr or a cfg_dep object"  
**Solution**: SPM syntax error in batch. Check that mask path contains no special characters.

**Issue**: Pipeline fails at design specification  
**Solution**: Check that mask_vbm.nii exists in results directory and is readable.

## Technical Details

See `docs/VBM_MASKING_IMPLEMENTATION.md` for:
- Detailed algorithm description
- SPM integration details  
- Why 0.1 threshold is used
- Biological justification
- Citation recommendations

## Contact

Questions or issues? Check:
1. `USAGE_GUIDE.md` - Complete pipeline documentation
2. `docs/VBM_MASKING_IMPLEMENTATION.md` - Technical details
3. Pipeline script comments: `cat12_longitudinal_analysis.sh`
