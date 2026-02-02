# Integration Summary: Preprocessing + Statistics

**Date:** February 2, 2026  
**Status:** ✅ Complete

## What Was Added

### 1. Preprocessing Infrastructure (from bids-cat12-wrapper)

**Files copied:**
- `cat12_prepro` - Main entry point script
- `scripts/preprocessing/` - Python preprocessing modules
  - `bids_cat12_processor.py` - Main BIDS processor
  - `subject_processor.py` - Individual subject handler
- `scripts/utils/cat12_utils.py` - CAT12 execution utilities
- `scripts/utils/bids_utils.py` - BIDS validation helpers
- `scripts/utils/generate_boilerplate.py` - HTML report generation
- `templates/preprocessing/` - MATLAB batch templates

**Dependencies added to requirements.txt:**
- `pybids>=0.16.0` - BIDS dataset handling
- `click>=8.1.0` - CLI framework
- `colorama>=0.4.6` - Colored terminal output
- `tqdm>=4.65.0` - Progress bars
- `pyyaml>=6.0` - YAML config parsing
- `defusedxml>=0.7.1` - Safe XML parsing

### 2. Statistics Pipeline (Already Existed)

**No changes needed** - kept as-is:
- `scripts/analysis/cat12_longitudinal_analysis.sh` - Main stats pipeline
- `scripts/analysis/cat12_multi_modality.sh` - Multi-modality wrapper
- All statistics utilities, parsers, and reporting tools

## Complete Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    1. PREPROCESSING                         │
│                    (cat12_prepro)                          │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ Input: BIDS dataset (T1w images)
                 │ Output: CAT12 derivatives (mwp1*, surf/*.gii)
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              2. STATISTICAL ANALYSIS                         │
│         (cat12_multi_modality.sh)                           │
└─────────────────────────────────────────────────────────────┘
                 │
                 │ Input: CAT12 derivatives + participants.tsv
                 │ Output: SPM results, TFCE maps, reports
                 │
                 ▼
           Results in results/ directory
```

## Quick Start

### Run Preprocessing

```bash
# Preprocess all subjects with surface extraction
./cat12_prepro /path/to/bids /path/to/derivatives/cat12 participant --preproc

# Volume-only (faster)
./cat12_prepro /path/to/bids /path/to/derivatives/cat12 participant \
    --preproc --no-surface

# With smoothing
./cat12_prepro /path/to/bids /path/to/derivatives/cat12 participant \
    --preproc \
    --smooth-volume "6 8" \
    --smooth-surface "12 15"
```

### Run Statistics

```bash
# All modalities (auto-detects from config.json)
./scripts/analysis/cat12_multi_modality.sh \
    --config config/config.json \
    --cat12-dir /path/to/derivatives/cat12

# Single modality
./scripts/analysis/cat12_longitudinal_analysis.sh \
    --config config/config.json \
    --cat12-dir /path/to/derivatives/cat12 \
    --modality vbm
```

## Key Improvements

### 1. **Complete Pipeline**
   - Now handles entire workflow from raw BIDS → statistical results
   - No need for external preprocessing steps

### 2. **Preprocessing Capabilities**
   - ✅ Longitudinal & cross-sectional processing
   - ✅ Volume + surface modalities
   - ✅ Multiple smoothing kernels
   - ✅ Quality assessment
   - ✅ TIV extraction
   - ✅ Parallel processing support

### 3. **Statistics Enhancements** (Recently Added)
   - ✅ Color-coded logging with timestamps
   - ✅ External tool output indentation
   - ✅ Multi-modality support (VBM + surfaces)
   - ✅ Skip existing results (--skip-existing, --force-modality, --force-all)
   - ✅ Surface file pattern fixes (rsub- prefix)
   - ✅ Surface-aware diagnostics (skips voxel checks for surfaces)
   - ✅ Proper sex encoding ('F'→0, 'M'→1)

## Configuration

### Environment Variables (Optional)

Create `.env` file:
```bash
export SPMROOT="/Volumes/Evo/software/spm25"
export CAT12_ROOT="/Volumes/Evo/software/spm25/toolbox/cat12"
# export MCR_ROOT="/path/to/matlab_runtime"  # Only for standalone
```

### Statistics Config

Edit `config/config.json`:
```json
{
  "analysis": {
    "participants_file": "results/data/participants.tsv",
    "sessions": ["1", "2"],
    "modalities": [
      {"name": "vbm", "smoothing_kernel": null, "covariates": ["tiv", "sex", "age"]},
      {"name": "thickness", "smoothing_kernel": 15, "covariates": ["sex", "age"]},
      {"name": "depth", "smoothing_kernel": 12, "covariates": ["sex", "age"]},
      {"name": "gyrification", "smoothing_kernel": 12, "covariates": ["sex", "age"]}
    ]
  }
}
```

## Directory Structure

```
cat12/stats/                        # This repository
├── cat12_prepro                    # ← NEW: Preprocessing entry point
├── scripts/
│   ├── preprocessing/              # ← NEW: Preprocessing modules
│   │   ├── bids_cat12_processor.py
│   │   └── subject_processor.py
│   ├── analysis/                   # Existing: Statistics pipeline
│   │   ├── cat12_longitudinal_analysis.sh
│   │   └── cat12_multi_modality.sh
│   └── utils/
│       ├── cat12_utils.py          # ← NEW: Preprocessing utilities
│       ├── bids_utils.py           # ← NEW: BIDS helpers
│       ├── generate_boilerplate.py # ← NEW: HTML reports
│       └── parse_participants.py   # Existing: Stats utilities
├── templates/
│   ├── preprocessing/              # ← NEW: MATLAB templates
│   │   └── longitudinal_template.m
│   ├── aal3.csv                    # Existing: Atlas
│   └── brainmask_GMtight.nii       # Existing: VBM mask
├── config/
│   └── config.json                 # Existing: Stats config
├── results/                        # Output directory
│   ├── vbm/
│   ├── thickness/
│   ├── depth/
│   └── gyrification/
├── requirements.txt                # ← UPDATED: Added prepro deps
├── README.md                       # Existing
├── PREPROCESSING.md                # ← NEW: Prepro documentation
└── INTEGRATION.md                  # ← NEW: This file
```

## What Was NOT Copied

From bids-cat12-wrapper, we **excluded**:
- `scripts/stats/` - Duplicate stats pipeline (kept our existing one)
- `docs/` - Documentation (created new PREPROCESSING.md instead)
- `activate_cat12.sh` - Not needed (using .venv directly)
- Test files and demos

## Testing

### 1. Test Preprocessing

```bash
# Dry run (plan without execution)
./cat12_prepro /data/bids /data/derivatives/cat12 participant \
    --preproc --dry-run

# Process single subject
./cat12_prepro /data/bids /data/derivatives/cat12 participant \
    --preproc --participant-label 01

# Pilot mode (1-2 subjects)
./cat12_prepro /data/bids /data/derivatives/cat12 participant \
    --preproc --pilot
```

### 2. Test Statistics

```bash
# Your current working setup - now with preprocessing support!
./scripts/analysis/cat12_multi_modality.sh \
    --config config/config.json \
    --cat12-dir /Volumes/Thunder/129_PK01/cat12/data
```

## Next Steps

1. **Run preprocessing** on subjects missing surface data (64 subjects identified)
2. **Update participants.tsv** if needed after preprocessing
3. **Re-run statistics** with complete dataset using `--force-modality depth`
4. **Document any project-specific preprocessing settings** in config/

## Troubleshooting

### ImportError: No module named 'pybids'
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### cat12_prepro: command not found
```bash
chmod +x cat12_prepro
```

### SPMROOT not set
```bash
# Add to .env file:
echo 'export SPMROOT="/Volumes/Evo/software/spm25"' > .env
source .env
```

## References

- **Source Repo**: https://github.com/MRI-Lab-Graz/bids-cat12-wrapper
- **Integration Date**: February 2, 2026
- **Maintained By**: MRI-Lab Graz
