# VBM-Specific Masking Implementation

## Overview

For VBM (Voxel-Based Morphometry) analyses, the standard whole-brain mask is not optimal because it includes many voxels with low probability gray matter values. This implementation creates a **VBM-specific mask** derived from an actual smoothed subject image, using a gray matter probability threshold of > 0.1.

## Why This Approach?

**Problem with standard masking:**
- Standard whole-brain masks include the entire head volume
- Many voxels contain air, bone, CSF with minimal signal
- These low-probability voxels introduce noise into the analysis
- Results in larger search volumes and reduced statistical power

**Solution: Gray Matter Probability Masking:**
- Uses actual smoothed gray matter images from the study
- Selects one random subject image
- Creates binary mask for voxels with GM probability > 0.1
- Excludes ~70% of voxels as "non-brain" or very low probability
- More conservative, biologically meaningful search volume

## Implementation Details

### Files Created

1. **`utils/generate_vbm_mask.m`** - MATLAB function that:
   - Finds all smoothed sNmwp1r*.nii files in CAT12 folder
   - Randomly selects one image for mask generation
   - Loads the image and creates binary mask for values > 0.1
   - Saves as `mask_vbm.nii` in results directory
   - Outputs mask statistics (voxels included/excluded, percentage)

2. **`utils/generate_spm_batch.py`** - Updated to:
   - Accept `--mask-file` argument
   - Pass mask to SPM as explicit mask in factorial design
   - Proper MATLAB syntax: `{'/path/to/mask.nii,1'}`

3. **`cat12_longitudinal_analysis.sh`** - Main pipeline updated to:
   - Call `generate_vbm_mask.m` for VBM modality (Step 2a)
   - Pass mask to SPM batch generation (Step 2b)
   - Works with any smoothing kernel and design structure

### Mask Characteristics (6mm smoothing example)

```
Input: Random smoothed subject from CAT12 data
  - Dimensions: 113 × 137 × 113
  - Value range: [0.0000, 0.9533] (GM probability)
  
Output: mask_vbm.nii
  - Binary mask (0 or 1)
  - Threshold: > 0.1
  - Voxels included: 533,605 (30.5%)
  - Voxels excluded: 1,215,748 (69.5%)
  - Biologically meaningful gray matter region
```

### Integration with Design Matrix

The mask is used **at design specification stage**, not as post-hoc thresholding:

```matlab
% In SPM batch (auto-generated):
matlabbatch{1}.spm.stats.factorial_design.masking.em = {'/path/to/mask_vbm.nii,1'};

% This ensures:
% - Only voxels in mask contribute to model estimation
% - Design matrix computed only for gray matter voxels
% - Contrast maps naturally respect the mask
```

## Usage

### Basic VBM Analysis (automatic masking)

```bash
./cat12_longitudinal_analysis.sh \
    --cat12-dir /path/to/cat12 \
    --participants participants.tsv \
    --modality vbm \
    --smoothing 6
```

### With Session Filtering

```bash
./cat12_longitudinal_analysis.sh \
    --cat12-dir /path/to/cat12 \
    --participants participants.tsv \
    --modality vbm \
    --smoothing 6 \
    --sessions "1,3"
```

### Surface-based Modalities (NO automatic masking)

```bash
./cat12_longitudinal_analysis.sh \
    --cat12-dir /path/to/cat12 \
    --participants participants.tsv \
    --modality thickness \
    --smoothing 20
```

Note: VBM-specific masking is only applied for `--modality vbm`. Surface modalities (thickness, depth, gyrification, fractal) use implicit masking only.

## Key Features

✅ **Automatic generation** - No manual mask creation needed  
✅ **Random subject selection** - Avoids bias toward specific subject  
✅ **Gray matter optimized** - Uses biologically meaningful threshold (0.1)  
✅ **Integrated into pipeline** - No extra steps required  
✅ **Flexible smoothing** - Works with any kernel size (6, 8, 9mm, etc.)  
✅ **Modality-specific** - Only applied when appropriate  
✅ **Reproducible** - Same mask used for all analyses in output directory  

## Output Files

After analysis completion, check:

```
results/vbm/vbm_smooth_auto/
  ├── mask_vbm.nii           ← Generated VBM-specific mask
  ├── SPM.mat                 ← Model with mask applied
  ├── beta_*.nii              ← Parameter estimates (in mask region only)
  ├── con_*.nii               ← Contrast maps (in mask region only)
  ├── design_matrix.png       ← Visual design representation
  ├── report.html             ← Analysis report
  └── tfce_*_fwe.nii          ← TFCE-corrected results (in mask region)
```

## Technical Notes

### Mask Generation Algorithm

1. **Find images**: `find $CAT12_DIR -name 's<N>mwp1r*.nii'`
2. **Select random**: `rand_idx = randi(num_images)`
3. **Load volume**: `img = spm_read_vols(V)`
4. **Create mask**: `mask = img > 0.1`
5. **Save**: `spm_write_vol(mask, 'mask_vbm.nii')`

### Why 0.1 Threshold?

- CAT12 smoothed images are GM probability maps [0, 1]
- Threshold 0.1 captures:
  - High-probability GM (>0.5)
  - Moderate GM probability regions (0.1-0.5)
  - Excludes: CSF, bone, air, very low probability voxels
- Produces ~30% of total brain volume (typical for GM)
- Biologically motivated for VBM studies

### SPM Integration

- Mask applied through `masking.em` (explicit mask)
- Works alongside implicit mask (zeros excluded automatically)
- Design matrix computed only for mask voxels
- Statistical power improved due to reduced search volume

## Performance Impact

**Positive impacts:**
- ✅ Increased statistical power (smaller search volume)
- ✅ More stringent multiple comparison correction
- ✅ Reduced computational burden
- ✅ Fewer spurious results in non-brain regions

**Neutral/Tradeoff:**
- ⚠ Some true findings outside 0.1 threshold might be missed (rare)
- ⚠ Results not directly comparable to non-masked analyses

## Compatibility

Works with:
- ✅ Any group/timepoint combination (flexible factorial design)
- ✅ Session filtering (`--sessions "1,3"`)
- ✅ Covariates (TIV, age, sex, etc.)
- ✅ Any smoothing kernel
- ✅ Pilot and full TFCE modes

Not applicable to:
- ❌ Surface-based morphometry (thickness, depth, gyrification)
- ❌ Other modalities requiring whole-brain coverage

## Citation Notes

When reporting results:

> *VBM analysis was performed using a binary gray matter mask derived from a randomly selected smoothed subject image (threshold > 0.1 GM probability), encompassing 30.5% of the total intracranial volume. This mask was applied during model specification to optimize sensitivity for gray matter changes while excluding non-brain and very low-probability regions.*

