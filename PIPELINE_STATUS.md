# CAT12 Pipeline - Status Report

**Date**: February 3, 2026  
**Time**: 15:45 UTC  
**Status**: ✅ RUNNING

---

## 🚀 Pipeline Overview

This is a complete end-to-end demonstration of the CAT12 preprocessing and statistics pipeline for longitudinal neuroimaging data.

**Workflow:**
```
1. Preprocessing (CAT12 standalone) → 8 images (4 subjects × 2 sessions)
2. Covariate extraction
3. Statistics analysis (VBM, thickness, depth, etc.)
4. HTML report generation
```

---

## 📊 Current Status

### ✅ Completed
- [x] CAT12.9 R2023b standalone downloaded (705MB)
- [x] MATLAB Runtime R2023b installed
- [x] Python environment setup (62 packages)
- [x] OpenNeuro ds000114 dataset downloaded (8 T1w images)
- [x] Single-subject test preprocessing (**SUCCESS**)
- [x] CAT12 batch script generation fixed for standalone

### 🔄 In Progress
- [ ] Full preprocessing (4 subjects × 2 sessions = 8 images)
  - **Started**: 15:45 UTC
  - **Expected completion**: ~18:45 UTC (3 hours)
  - **Current**: Processing sub-01 session test
  - **Log**: `full_preprocessing_sequential.log`

- [ ] Monitor script watching for completion
  - **Will auto-trigger stats** when preprocessing finishes
  - **Log**: `monitor_stats.log`

### ⏳ Pending
- [ ] Covariate extraction from XML outputs
- [ ] Statistics pipeline (VBM, thickness, depth, gyrification, fractal)
- [ ] HTML report generation

---

## 📁 Output Locations

| Item | Path |
|------|------|
| Raw data | `openneuro/ds000114/` |
| CAT12 outputs | `projects/demo/derivatives/cat12/` |
| Stats results | `projects/demo/results/` |
| Logs | `projects/demo/derivatives/cat12/logs/` |

---

## 🛠️ Key Fixes Applied

1. **OS-aware installer** (`scripts/setup/install_cat12_standalone.sh`)
   - Detects Linux/macOS and downloads correct CAT12/MCR versions
   - Windows support provided with manual instructions

2. **CAT12 standalone path fixes** (`scripts/utils/cat12_utils.py`)
   - Added `standalone/` subdirectory lookup
   - Fixed SPM batch script generation for standalone format

3. **Standalone batch templates**
   - Single-session: Uses `cat_standalone_segment.m` format
   - Multi-session: Adapted from CAT12's official `cat_standalone_segment_long.m`
   - Both properly formatted for SPM batch execution

4. **Sequential processing**
   - Process each subject individually to avoid complex multi-session batch issues
   - Each command: `./cat12_prepro --participant-label XX --session-label YY`

---

## 💻 System Info

| Item | Value |
|------|-------|
| OS | macOS |
| Architecture | ARM64 (Apple Silicon) |
| MATLAB | R2025b (available, not using) |
| MCR | R2023b (/Applications/MATLAB/MATLAB_Runtime/R2023b) |
| CAT12 | 9.0 r2566 Standalone |
| SPM12 | Version 7771 |
| Python | 3.11+, venv at `.venv/` |

---

## 📝 Automation Scripts

### Main Pipeline
```bash
./cat12_prepro <bids_dir> <output_dir> participant \
  --preproc \
  --participant-label XX \
  --session-label YY \
  --smooth-volume 6 --smooth-surface 12 \
  --qa --tiv --no-validate
```

### Monitor & Auto-Stats
```bash
bash monitor_and_run_stats.sh
# Monitors preprocessing logs and automatically runs stats when complete
```

### Full Pipeline (manual)
```bash
bash run_demo.sh
# Comprehensive pipeline: preproc → covariate extraction → stats → report
```

---

## 🔧 Troubleshooting

### If preprocessing fails:
```bash
# Check latest log
tail -100 projects/demo/derivatives/cat12/logs/*.log

# Check CAT12 report
tail -50 projects/demo/derivatives/cat12/sub-XX/report/catlog_*.txt
```

### If stats don't run:
```bash
# Check covariates extracted
ls -la projects/demo/participants_*.tsv

# Extract manually
python scripts/utils/extract_covariates_from_xml.py \
  projects/demo/derivatives/cat12 \
  projects/demo/participants_demo.tsv

# Run stats manually
bash scripts/analysis/cat12_multi_modality.sh \
  --config projects/demo/project_config.json \
  --cat12-dir projects/demo/derivatives/cat12
```

---

## 📚 Documentation Files

- `INSTALLATION_GUIDE_STANDALONE.md` - OS-specific installation
- `INSTALL_GUIDE.md` - Original installation guide
- `DEMO_RUN.md` - Demo workflow
- `README.md` - Project overview

---

## ⏱️ Expected Timeline

| Phase | Duration | ETA |
|-------|----------|-----|
| Preprocessing | 2-4 hours | ~18:45 UTC |
| Covariates | 5 min | ~18:50 UTC |
| Statistics | 30-60 min | ~19:50 UTC |
| Report | 5 min | ~19:55 UTC |
| **TOTAL** | **~3-4 hours** | **~19:55 UTC** |

---

## 📊 Next Steps When Complete

1. ✅ Check output files in `projects/demo/results/vbm_smooth6/`
2. ✅ View report: `projects/demo/report_vbm.html` (if generated)
3. ✅ Review statistics: T-maps, cluster extents, thresholded results
4. ✅ Validate results against expected group differences

---

**Last Updated**: 2026-02-03 15:45:04 UTC
