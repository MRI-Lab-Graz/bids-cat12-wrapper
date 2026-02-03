# Demo Run - Complete Workflow

Step-by-step execution from preprocessing to statistics using OpenNeuro ds000114.

**Prerequisites:** Complete `INSTALL_GUIDE.md` first.

---

## Setup

### 1. Activate Virtual Environment

```bash
cd /Users/karl/work/github/bids-cat12-wrapper
source .venv/bin/activate
```

### 2. Configure Project

The demo config is at `projects/demo/project_config.json`.

**Key settings to verify/update:**

```json
{
  "study": {
    "project_folder": "/Users/karl/work/github/bids-cat12-wrapper/projects/demo"
  },
  "software": {
    "matlab": {
      "executable": "/Applications/MATLAB_R2025b.app/bin/matlab"
    },
    "spm": {
      "path": "/Users/karl/work/github/bids-cat12-wrapper/external/spm12"
    }
  }
}
```

**For Standalone CAT12** (preferred on macOS):
- Set environment variable before running preprocessing
- See preprocessing section below

---

## Phase 1: Preprocessing

### Option A: Using Standalone CAT12 (Recommended)

```bash
# Set environment variables
export SPMROOT="/Users/karl/work/github/bids-cat12-wrapper/external/cat12"
export MCR_ROOT="/Applications/MATLAB/MATLAB_Runtime/R2023b"

# Run preprocessing for all subjects
./cat12_prepro \
  openneuro/ds000114 \
  projects/demo/derivatives/cat12 \
  participant \
  --preproc \
  --participant-label 01 02 03 04 \
  --session-label test retest \
  --smooth-volume "6" \
  --smooth-surface "12" \
  --qa \
  --tiv
```

### Option B: Using MATLAB + SPM12/CAT12

```bash
# Environment variables (if not in project_config.json)
export SPMROOT="/Users/karl/work/github/bids-cat12-wrapper/external/spm12"
export CAT12_ROOT="/Users/karl/work/github/bids-cat12-wrapper/external/spm12/toolbox/cat12"

# Run preprocessing
./cat12_prepro \
  openneuro/ds000114 \
  projects/demo/derivatives/cat12 \
  participant \
  --preproc \
  --participant-label 01 02 03 04 \
  --session-label test retest \
  --smooth-volume "6" \
  --smooth-surface "12" \
  --qa \
  --tiv
```

### What This Does:

1. **Validates** BIDS dataset structure
2. **Segments** T1w images into GM/WM/CSF
3. **Normalizes** to MNI space (VBM: `mwp1*` files)
4. **Extracts surfaces** (thickness, depth, gyrification)
5. **Smooths** data (6mm for volume, 12mm for surface)
6. **Computes QA metrics** (image quality assessment)
7. **Extracts TIV** (total intracranial volume)

### Expected Output:

```
projects/demo/derivatives/cat12/
├── sub-01/
│   ├── ses-test/
│   │   ├── mri/
│   │   │   ├── mwp1sub-01_ses-test_T1w.nii          # GM (VBM)
│   │   │   ├── mwp2sub-01_ses-test_T1w.nii          # WM (VBM)
│   │   │   ├── s6.mwp1sub-01_ses-test_T1w.nii       # Smoothed 6mm
│   │   │   └── ...
│   │   ├── surf/
│   │   │   ├── s12.mesh.thickness.resampled.sub-01_ses-test_T1w.gii
│   │   │   ├── s12.mesh.depth.resampled.sub-01_ses-test_T1w.gii
│   │   │   ├── s12.mesh.gyrification.resampled.sub-01_ses-test_T1w.gii
│   │   │   └── ...
│   │   └── report/
│   │       └── cat_sub-01_ses-test_T1w.xml          # QA + TIV
│   └── ses-retest/
│       └── ... (same structure)
├── sub-02/
└── ...
```

### Processing Time:

- **Per subject/session**: ~15-30 minutes
- **Total (4 subjects × 2 sessions)**: ~2-4 hours

---

## Phase 2: Generate Participants File

After preprocessing, extract TIV and merge with participant metadata:

```bash
# Extract TIV from CAT12 XML reports
python scripts/utils/extract_covariates_from_xml.py \
  projects/demo/derivatives/cat12 \
  projects/demo/tiv_extracted.tsv

# Generate full participants file for statistics
python scripts/utils/generate_participants_from_bids.py \
  openneuro/ds000114 \
  projects/demo/derivatives/cat12 \
  --output projects/demo/participants_demo.tsv \
  --merge-tiv projects/demo/tiv_extracted.tsv
```

**Result:** `projects/demo/participants_demo.tsv` with columns:
- `participant_id`, `session`, `group`, `age`, `sex`, `tiv`, etc.

---

## Phase 3: Statistics

### Option 1: Single Modality

Run VBM analysis:

```bash
./scripts/analysis/cat12_longitudinal_analysis.sh \
  --config projects/demo/project_config.json \
  --cat12-dir projects/demo/derivatives/cat12 \
  --participants projects/demo/participants_demo.tsv \
  --modality vbm \
  --smoothing 6 \
  --sessions test,retest \
  --group-col group \
  --session-col session
```

### Option 2: All Modalities (Recommended)

Run VBM, thickness, depth, and gyrification:

```bash
./scripts/analysis/cat12_multi_modality.sh \
  --config projects/demo/project_config.json \
  --cat12-dir projects/demo/derivatives/cat12 \
  --participants projects/demo/participants_demo.tsv
```

### What This Does:

1. **Creates factorial design** (Group × Session)
2. **Builds design matrix** with covariates
3. **Estimates SPM model**
4. **Runs contrasts** (main effects, interactions)
5. **Applies TFCE correction** (permutation testing)
6. **Generates thresholded maps**

### Expected Output:

```
projects/demo/results/
├── vbm_smooth6/
│   ├── SPM.mat
│   ├── design_matrix.png
│   ├── beta_*.nii
│   ├── con_*.nii
│   ├── spmT_*.nii
│   └── tfce/
│       ├── tfce_corrp_tstat_*.nii
│       └── ...
├── thickness_smooth12/
├── depth_smooth12/
└── gyrification_smooth12/
```

---

## Phase 4: Interactive Report

Generate HTML visualization:

```bash
python scripts/reporting/post_stats_report.py \
  projects/demo/results/vbm_smooth6 \
  projects/demo/demo_vbm_report.html
```

Open in browser:
```bash
open projects/demo/demo_vbm_report.html
```

### Report Features:

- ✅ Multi-atlas labeling (AAL3, COBRA)
- ✅ Interactive thresholding
- ✅ Cluster table with coordinates
- ✅ Effect size visualization
- ✅ Filter by p-value, correction type

---

## Quick Reference Commands

### Full Pipeline (One Command):

```bash
# After preprocessing is complete
./scripts/analysis/cat12_multi_modality.sh \
  --config projects/demo/project_config.json \
  --cat12-dir projects/demo/derivatives/cat12 \
  --participants projects/demo/participants_demo.tsv \
  --force-all
```

### Check Progress:

```bash
# Count preprocessed files
find projects/demo/derivatives/cat12 -name "mwp1*.nii" | wc -l
find projects/demo/derivatives/cat12 -name "*thickness*.gii" | wc -l

# Check logs
tail -f projects/demo/logs/preproc/*.log
tail -f projects/demo/logs/*.log
```

### Cleanup:

```bash
# Remove intermediate files
rm -rf projects/demo/work

# Rerun specific modality
./scripts/analysis/cat12_multi_modality.sh \
  --config projects/demo/project_config.json \
  --cat12-dir projects/demo/derivatives/cat12 \
  --force-modality thickness
```

---

## Troubleshooting

### Preprocessing Fails:

1. **Check CAT12 installation:**
   ```bash
   ls -la $SPMROOT/standalone/
   ls -la $MCR_ROOT/
   ```

2. **Check BIDS structure:**
   ```bash
   python -c "from bids import BIDSLayout; print(BIDSLayout('openneuro/ds000114', validate=False))"
   ```

3. **Run single subject test:**
   ```bash
   ./cat12_prepro openneuro/ds000114 projects/demo/test participant --preproc --participant-label 01 --session-label test
   ```

### Statistics Fails:

1. **Verify SPM/MATLAB:**
   ```bash
   $MATLAB_BIN -batch "addpath('$SPMROOT'); spm('ver')"
   ```

2. **Check participants file:**
   ```bash
   head projects/demo/participants_demo.tsv
   ```

3. **Check preprocessed files exist:**
   ```bash
   ls projects/demo/derivatives/cat12/sub-*/ses-*/mri/s6.mwp1*.nii
   ```

---

## Expected Timeline

| Phase | Time |
|-------|------|
| Installation | 30-60 min |
| Data download | 10-30 min |
| Preprocessing | 2-4 hours |
| Statistics | 30-60 min |
| Report generation | 5-10 min |
| **Total** | **~4-6 hours** |

---

## Next Steps

After completing the demo:

1. **Try your own data**: Copy `projects/demo/` structure
2. **Customize design**: Edit contrasts in analysis scripts
3. **Explore results**: Interactive HTML reports
4. **Scale up**: Add more subjects, sessions, covariates
