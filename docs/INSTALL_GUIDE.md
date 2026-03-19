# CAT12 Installation Guide

Complete setup for running CAT12 preprocessing and statistics pipeline.

---

## Prerequisites

- **macOS** (ARM64/Apple Silicon or Intel)
- **Python 3.8+**
- **Git**

---

## Option 1: Standalone CAT12 (Recommended - No MATLAB License Needed)

### macOS ARM64 (Apple Silicon) - CAT12.9 STANDALONE WORKING ✓

**No MATLAB license needed! Uses free MATLAB Runtime R2023b**

1. **CAT12 Standalone** - Already downloaded and extracted:
   ```bash
   /Users/karl/work/github/bids-cat12-wrapper/external/CAT12.9_R2023b_MCR_MAC_arm64/
   ```

2. **Install MATLAB Runtime R2023b** (one-time setup):
   - DMG already downloaded: `external/MATLAB_Runtime_R2023b_maca64.dmg`
   - Double-click to mount and run the installer
   - Or from terminal (wait for GUI):
   ```bash
   open external/MATLAB_Runtime_R2023b_maca64.dmg
   ```
   - **Installation path:** Accept default → `/Applications/MATLAB/MATLAB_Runtime/R2023b`

3. **Verify Installation**:
   ```bash
   ls -la /Applications/MATLAB/MATLAB_Runtime/R2023b
   /Users/karl/work/github/bids-cat12-wrapper/external/CAT12.9_R2023b_MCR_MAC_arm64/standalone/cat_standalone.sh --help
   ```

2. **Download MATLAB Runtime R2023b** (Update 10):
   ```bash
   curl -L -o MATLAB_Runtime_R2023b.dmg https://ssd.mathworks.com/supportfiles/downloads/R2023b/Release/10/deployment_files/installer/complete/maca64/MATLAB_Runtime_R2023b_Update_10_maca64.dmg
   ```
   
   Then:
   - Open the DMG file
   - Run the installer
   - Follow installation prompts (installs to `/Applications/MATLAB/MATLAB_Runtime/R2023b`)

3. **Verify Installation**:
   ```bash
   ls -la /Users/karl/work/github/bids-cat12-wrapper/external/cat12/
   ls -la /Applications/MATLAB/MATLAB_Runtime/R2023b/
   ```

### macOS Intel (x86_64)

Use the Intel version instead:
```bash
cd /Users/karl/work/github/bids-cat12-wrapper/external
curl -L -o cat12_macos_intel.zip https://dbm.neuro.uni-jena.de/cat12/cat12_latest_R2023b_MCR_Mac.zip
unzip cat12_macos_intel.zip
mv cat12_latest_R2023b_MCR_Mac cat12
```

---

## Option 2: MATLAB + SPM12/CAT12 (Current Working Method)

**Recommended for macOS until standalone is accessible:**

1. **Clone SPM12** (~91MB):
   ```bash
   cd /Users/karl/work/github/bids-cat12-wrapper/external
   git clone --depth=1 https://github.com/spm/spm.git spm12
   ```

2. **Clone CAT12** (~205MB):
   ```bash
   cd /Users/karl/work/github/bids-cat12-wrapper/external/spm12/toolbox
   git clone --depth=1 https://github.com/ChristianGaser/cat12.git
   ```

3. **Update project config** at `projects/demo/project_config.json`:
   ```json
   "software": {
     "matlab": {
       "executable": "/Applications/MATLAB_R2025b.app/bin/matlab"
     },
     "spm": {
       "path": "/Users/karl/work/github/bids-cat12-wrapper/external/spm12"
     }
   }
   ```

---

## Python Dependencies

Install Python packages (works for both options):

```bash
cd /Users/karl/work/github/bids-cat12-wrapper
./scripts/setup/install.sh
```

This creates a virtual environment at `.venv/` and installs:
- pybids (BIDS data handling)
- nibabel (neuroimaging I/O)
- numpy, pandas, scipy (data processing)
- matplotlib, seaborn (visualization)
- click, colorama, tqdm (CLI tools)
- openneuro-py (dataset download)

---

## Download Demo Dataset

OpenNeuro ds000114 (test-retest, 4 subjects):

```bash
# Using openneuro-py (Python API)
source .venv/bin/activate
python -c "
from openneuro import download
download(
    dataset='ds000114',
    target_dir='openneuro/ds000114',
    include=['sub-01/ses-*/*T1w.*', 'sub-02/ses-*/*T1w.*', 
             'sub-03/ses-*/*T1w.*', 'sub-04/ses-*/*T1w.*']
)
print('Download complete!')
"
```

Or download manually from: https://openneuro.org/datasets/ds000114/

---

## Verification Checklist

Before running the pipeline, verify:

### Standalone Setup:
- [ ] `/Users/karl/work/github/bids-cat12-wrapper/external/cat12/` exists
- [ ] `/Applications/MATLAB/MATLAB_Runtime/R2023b/` exists
- [ ] `.venv/bin/activate` works
- [ ] `openneuro/ds000114/sub-*/ses-*/*T1w.nii.gz` files exist

### MATLAB Setup:
- [ ] `/Users/karl/work/github/bids-cat12-wrapper/external/spm12/` exists
- [ ] `/Users/karl/work/github/bids-cat12-wrapper/external/spm12/toolbox/cat12/` exists
- [ ] MATLAB license valid
- [ ] `.venv/bin/activate` works
- [ ] `openneuro/ds000114/sub-*/ses-*/*T1w.nii.gz` files exist

---

## Next Steps

Once installation is complete, proceed to `DEMO_RUN.md` for execution steps.

---

## Reproducibility Notes

**Software Versions (Demo Setup):**
- CAT12: Latest R2023b (from GitHub: https://github.com/ChristianGaser/cat12.git)
- SPM12: Latest (from GitHub: https://github.com/spm/spm.git)
- MATLAB Runtime: R2023b Update 10 (for standalone)
- Python: 3.8+ with pybids 0.16.0+, nibabel 5.0.0+

**Dataset:**
- OpenNeuro ds000114 (https://openneuro.org/datasets/ds000114/)
- 4 subjects, 2 sessions each (test-retest design)
