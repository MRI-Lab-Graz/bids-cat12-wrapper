# Multi-Modality Analysis Setup

The pipeline now supports running multiple modalities (VBM, thickness, depth, etc.) in a single configuration.

## Configuration

Edit `config/config.json` and define your modalities under `analysis.modalities`:

```json
{
  "analysis": {
    "participants_file": "...",
    "group_column": "group",
    "session_column": "session",
    "sessions": ["1", "2"],
    
    "modalities": [
      {
        "name": "vbm",
        "smoothing_kernel": 8,
        "covariates": ["tiv", "sex", "age"],
        "mask": "/path/to/vbm_mask.nii"
      },
      {
        "name": "thickness",
        "smoothing_kernel": 15,
        "covariates": ["sex", "age"],
        "mask": null
      },
      {
        "name": "depth",
        "smoothing_kernel": 20,
        "covariates": ["sex", "age"],
        "mask": null
      }
    ]
  },
  
  "tfce": {
    "n_permutations": 5000,
    "pilot_mode": true
  },
  
  "reporting": { ... }
}
```

## Key Points

### Smoothing Kernel
- Set to a number (e.g., `8`, `15`, `20`) to use that specific kernel
- Set to `null` to auto-detect (requires **exactly one** smoothing kernel available)
- If multiple kernels are found, you must specify explicitly

### Covariates
- Each modality can have different covariates
- **Volumetric (VBM)**: Can include TIV (total intracranial volume)
- **Surface-based (thickness, depth, gyrification)**: Cannot use TIV; typically use only `sex` and `age`
- Standardization is applied per-modality

### Mask
- **VBM**: Should point to explicit GM mask (e.g., `brainmask_GMtight.nii`)
- **Surface modalities**: Set to `null` (masks not applicable to surface data)

## Running Multi-Modality Analysis

### Option 1: Run all modalities
```bash
./scripts/analysis/cat12_multi_modality.sh \
  --config config/config.json \
  --cat12-dir /path/to/cat12/data
```

### Option 2: Run specific modality only
```bash
./scripts/analysis/cat12_multi_modality.sh \
  --config config/config.json \
  --cat12-dir /path/to/cat12/data \
  --modality thickness
```

### Option 3: Run single modality via main script (legacy)
```bash
./scripts/analysis/cat12_longitudinal_analysis.sh \
  --config config/config.json \
  --cat12-dir /path/to/cat12/data \
  --modality vbm \
  --smoothing 8
```

## Output Organization

Results are organized by modality:

```
results/
  vbm/
    vbm_smooth8_default_analysis/
      report.html
      SPM.mat
      TFCE_*_fwe.nii
      logs/
  thickness/
    thickness_smooth15_default_analysis/
      report.html
      SPM.mat
      TFCE_*_fwe.nii
      logs/
  depth/
    depth_smooth20_default_analysis/
      ...
```

## Troubleshooting

### "Multiple smoothing kernels detected"
This happens when you set `smoothing_kernel: null` but the CAT12 directory has multiple smoothing kernels (e.g., s8, s15, s20 files).

**Solution**: Specify the smoothing kernel explicitly in config.json

```json
"smoothing_kernel": 15  // Choose one
```

### "Modality not found in config"
Make sure the modality name matches exactly (case-sensitive):
- `vbm`
- `thickness`
- `depth`
- `gyrification`
- `fractal`

### "TIV not found for surface modality"
Surface modalities (thickness, depth, etc.) cannot use TIV. Remove it from covariates for those modalities.

```json
{
  "name": "thickness",
  "covariates": ["sex", "age"]  // No TIV
}
```
