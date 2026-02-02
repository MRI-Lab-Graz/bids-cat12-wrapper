# CAT12 Preprocessing Integration

This repository now includes CAT12 preprocessing capabilities integrated from [bids-cat12-wrapper](https://github.com/MRI-Lab-Graz/bids-cat12-wrapper).

## Workflow

### 1. Preprocessing (New!)

Use `cat12_prepro` to preprocess BIDS T1w images with CAT12:

```bash
# Full preprocessing with surface extraction
./cat12_prepro /path/to/bids_data /path/to/output participant --preproc

# Volume-only (faster, no surface)
./cat12_prepro /path/to/bids_data /path/to/output participant --preproc --no-surface

# With smoothing
./cat12_prepro /path/to/bids_data /path/to/output participant \
    --preproc \
    --smooth-volume "6 8 10" \
    --smooth-surface "12 15"

# Specific subjects
./cat12_prepro /path/to/bids_data /path/to/output participant \
    --preproc \
    --participant-label 01 02 03
```

**Output structure:**
```
output/
├── sub-01/
│   ├── mri/          # VBM outputs (mwp1*, mwp2*)
│   └── surf/         # Surface outputs (thickness, depth, gyrification)
├── sub-02/
└── ...
```

### 2. Statistical Analysis (Existing)

Once preprocessing is complete, run statistical analysis:

```bash
# Single modality
./scripts/analysis/cat12_longitudinal_analysis.sh \
    --config config/config.json \
    --cat12-dir /path/to/output \
    --modality vbm

# All modalities (auto-detects and runs each)
./scripts/analysis/cat12_multi_modality.sh \
    --config config/config.json \
    --cat12-dir /path/to/output
```

## Installation

Install additional preprocessing dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

### Preprocessing Config (Optional)

Set environment variables in `.env`:
```bash
export SPMROOT="/path/to/spm12"
export CAT12_ROOT="/path/to/spm12/toolbox/cat12"
export MCR_ROOT="/path/to/matlab_runtime"  # For standalone
```

### Statistics Config

Edit `config/config.json` to set:
- Participants file path
- Sessions to analyze
- Modalities (vbm, thickness, depth, gyrification)
- Covariates
- TFCE parameters

## Key Features

### Preprocessing (`cat12_prepro`)
- ✅ BIDS-compliant input/output
- ✅ Longitudinal & cross-sectional
- ✅ Volume (VBM) processing
- ✅ Surface extraction (thickness, depth, gyrification)
- ✅ Multiple smoothing kernels
- ✅ Quality assessment
- ✅ TIV extraction
- ✅ Parallel processing

### Statistics (`cat12_multi_modality.sh`)
- ✅ Factorial design (Group × Session)
- ✅ Multiple modalities
- ✅ Covariates (TIV, age, sex)
- ✅ Screening (uncorrected p < 0.001)
- ✅ TFCE permutation testing
- ✅ Automated reporting with brain visualizations
- ✅ Skip existing results (re-run control)

## Directory Structure

```
.
├── cat12_prepro                    # Preprocessing entry point (NEW)
├── scripts/
│   ├── preprocessing/              # Preprocessing modules (NEW)
│   │   ├── bids_cat12_processor.py
│   │   └── subject_processor.py
│   ├── analysis/                   # Statistics pipeline
│   │   ├── cat12_longitudinal_analysis.sh
│   │   └── cat12_multi_modality.sh
│   └── utils/
│       ├── cat12_utils.py          # Preprocessing utilities (NEW)
│       ├── bids_utils.py           # BIDS helpers (NEW)
│       ├── generate_boilerplate.py # HTML reports (NEW)
│       └── parse_participants.py   # Statistics utilities
├── templates/
│   ├── preprocessing/              # MATLAB templates (NEW)
│   │   └── longitudinal_template.m
│   ├── aal3.csv                    # Atlas labels
│   └── brainmask_GMtight.nii       # VBM mask
└── config/
    └── config.json                 # Statistics configuration
```

## Examples

### Complete Workflow

```bash
# 1. Preprocess BIDS data
./cat12_prepro /data/bids /data/derivatives/cat12 participant \
    --preproc \
    --smooth-volume "6 8" \
    --smooth-surface "12 15"

# 2. Run multi-modality statistics
./scripts/analysis/cat12_multi_modality.sh \
    --config config/config.json \
    --cat12-dir /data/derivatives/cat12

# Results in: results/vbm/, results/thickness/, results/depth/, results/gyrification/
```

### Preprocessing Only Missing Subjects

The preprocessing pipeline can skip already-processed subjects:

```bash
./cat12_prepro /data/bids /data/derivatives/cat12 participant \
    --preproc \
    --skip-existing
```

### Force Reanalysis in Statistics

```bash
# Force rerun of VBM only
./scripts/analysis/cat12_multi_modality.sh \
    --config config/config.json \
    --cat12-dir /data/derivatives/cat12 \
    --force-modality vbm

# Force rerun all modalities
./scripts/analysis/cat12_multi_modality.sh \
    --config config/config.json \
    --cat12-dir /data/derivatives/cat12 \
    --force-all
```

## References

- **Preprocessing**: Based on [MRI-Lab-Graz/bids-cat12-wrapper](https://github.com/MRI-Lab-Graz/bids-cat12-wrapper)
- **CAT12**: [Structural Brain Mapping Group](http://www.neuro.uni-jena.de/cat/)
- **SPM**: [Statistical Parametric Mapping](https://www.fil.ion.ucl.ac.uk/spm/)
