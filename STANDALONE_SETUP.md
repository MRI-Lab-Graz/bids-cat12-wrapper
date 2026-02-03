# CAT12 Standalone Installation - Completion Guide

## Status: Ready for Final Step ✓

Everything is downloaded and prepared. Just need to complete **one manual installation** of MATLAB Runtime.

---

## Current Setup

### Downloaded Files:
- ✅ **CAT12.9 R2023b Standalone (ARM64)**: `external/CAT12.9_R2023b_MCR_MAC_arm64/` (705MB)
- ✅ **MATLAB Runtime R2023b DMG**: `external/MATLAB_Runtime_R2023b_maca64.dmg` (2GB)
- ✅ **Python Environment**: `.venv/` with all dependencies
- ✅ **Test Dataset**: OpenNeuro ds000114 (4 subjects, 2 sessions, 8 T1w scans)

---

## Next Step: Install MATLAB Runtime

### Method 1: GUI (Recommended)

1. Open DMG file:
   ```bash
   open external/MATLAB_Runtime_R2023b_maca64.dmg
   ```

2. Double-click `InstallForMacOSAppleSilicon.app`

3. Follow installation prompts:
   - Accept license
   - **Install location: `/Applications/MATLAB/MATLAB_Runtime/R2023b`** (important!)
   - Complete installation (~5 min)

4. Unmount DMG when done:
   ```bash
   hdiutil detach /Volumes/MATLAB_Runtime_R2023b_Update_10_maca64
   ```

### Method 2: Command Line (Advanced)

```bash
cd /Volumes/MATLAB_Runtime_R2023b_Update_10_maca64
InstallForMacOSAppleSilicon.app/Contents/MacOS/installForMacOSAppleSilicon \
  -destinationFolder /Applications/MATLAB/MATLAB_Runtime/R2023b \
  -agreeToLicense yes
```

---

## Verify Installation

After installing MCR:

```bash
ls -la /Applications/MATLAB/MATLAB_Runtime/R2023b/
```

Should show MCR files. Then test CAT12 standalone:

```bash
/Users/karl/work/github/bids-cat12-wrapper/external/CAT12.9_R2023b_MCR_MAC_arm64/standalone/cat_standalone.sh --help
```

---

## After MCR Installation: Run Preprocessing

Once MCR is installed, run:

```bash
cd /Users/karl/work/github/bids-cat12-wrapper
source .venv/bin/activate

# Run CAT12 standalone preprocessing (test with subject 01)
./cat12_prepro \
  openneuro/ds000114 \
  projects/demo/derivatives/cat12 \
  participant \
  --preproc \
  --participant-label 01 \
  --session-label test \
  --smooth-volume "6" \
  --smooth-surface "12" \
  --qa \
  --tiv \
  --no-validate
```

**Expected output:**
- Processing with CAT12 standalone executable
- No MATLAB licensing messages
- Processing 1 subject × 1 session (~15-30 min)
- Output in `projects/demo/derivatives/cat12/sub-01/`

---

## Environment Variables (Auto-loaded from .env)

```bash
export CAT12_STANDALONE=/Users/karl/work/github/bids-cat12-wrapper/external/CAT12.9_R2023b_MCR_MAC_arm64
export MCR_ROOT=/Applications/MATLAB/MATLAB_Runtime/R2023b
export USE_STANDALONE=true
```

---

## Troubleshooting

**MCR installation fails/not recognized?**
- Check installation path: `ls /Applications/MATLAB/MATLAB_Runtime/R2023b`
- Ensure you used ARM64 installer (not Intel x86_64)
- May need admin password during installation

**CAT12 standalone not found?**
```bash
ls /Users/karl/work/github/bids-cat12-wrapper/external/CAT12.9_R2023b_MCR_MAC_arm64/standalone/
```

**Preprocessing still using old MATLAB?**
- Verify `.env` file has `USE_STANDALONE=true`
- Check Python sees environment: `python -c "import os; print(os.getenv('MCR_ROOT'))"`

---

## Demo Workflow Summary

Once MCR is installed, you can run the complete demo:

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Preprocess all 4 subjects
./cat12_prepro openneuro/ds000114 projects/demo/derivatives/cat12 participant \
  --preproc --participant-label 01 02 03 04 --session-label test retest \
  --smooth-volume "6" --smooth-surface "12" --qa --tiv --no-validate

# 3. Extract covariates and generate participants file
python scripts/utils/extract_covariates_from_xml.py \
  projects/demo/derivatives/cat12 projects/demo/tiv.tsv

# 4. Run statistics
./scripts/analysis/cat12_multi_modality.sh \
  --config projects/demo/project_config.json \
  --cat12-dir projects/demo/derivatives/cat12 \
  --participants projects/demo/participants_demo.tsv

# 5. Generate report
python scripts/reporting/post_stats_report.py \
  projects/demo/results/vbm_smooth6 \
  projects/demo/report.html
```

---

**Status**: Waiting for MCR installation → Ready to run preprocessing!
