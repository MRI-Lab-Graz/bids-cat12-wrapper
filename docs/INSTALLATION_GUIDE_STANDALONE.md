# CAT12 Standalone Installation Guide

This guide covers OS-specific CAT12 standalone installation using the automated installer script.

## Supported Operating Systems

| OS | Status | Method | Notes |
|---|---|---|---|
| **Linux** | ✅ Supported | Standalone | x86_64 (glnxa64) |
| **macOS** | ✅ Supported | Standalone | ARM64 (Apple Silicon) & Intel (x86_64) |
| **Windows** | ❌ Not Yet Supported | Manual | See Windows section below |

---

## Quick Start

### One-Command Installation

For **Linux** or **macOS**:

```bash
cd /path/to/bids-cat12-wrapper
bash ./scripts/setup/install_cat12_standalone.sh
```

The script will:
1. ✅ Detect your OS and architecture
2. ✅ Download CAT12.9 standalone (~700MB)
3. ✅ Download MATLAB Runtime R2023b (~2GB)
4. ✅ Create Python environment
5. ✅ Configure environment variables
6. ✅ Provide next-step instructions

---

## Step-by-Step Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_REPO/bids-cat12-wrapper.git
cd bids-cat12-wrapper
```

### Step 2: Run Installer

```bash
bash ./scripts/setup/install_cat12_standalone.sh
```

The installer will detect your system and download appropriate files.

### Step 3: Install MATLAB Runtime

#### macOS (ARM64 - Apple Silicon)

```bash
open external/MATLAB_Runtime_R2023b_Update_10_maca64.dmg
```

1. Open the DMG file
2. Run `InstallForMacOSAppleSilicon.app`
3. Follow the installer prompts
4. Install to `/Applications/MATLAB/MATLAB_Runtime/R2023b`
5. Complete installation

#### macOS (Intel - x86_64)

```bash
open external/MATLAB_Runtime_R2023b_Update_10_maci64.dmg
```

Same as above, but look for `InstallForMacOSX.app`

#### Linux (x86_64)

```bash
cd external
unzip MATLAB_Runtime_R2023b_Update_10_glnxa64.zip
./install -destinationFolder /opt/MATLAB/MATLAB_Runtime/R2023b -agreeToLicense yes
```

### Step 4: Activate Python Environment

```bash
source .venv/bin/activate
```

### Step 5: Install Python Dependencies

```bash
bash ./scripts/setup/install.sh
```

Or manually:

```bash
pip install -r requirements.txt
```

### Step 6: Verify Installation

```bash
# Check CAT12 standalone
external/CAT12.9_R2023b_MCR_MAC_arm64/standalone/cat_standalone.sh --help

# Check Python environment
python -c "import pybids; print('✓ pybids installed')"
```

---

## Usage After Installation

### Run Preprocessing

**Single subject (one session):**

```bash
# Activate environment
source .venv/bin/activate

# Run CAT12 preprocessing on one subject, one session
./cat12_prepro openneuro/ds000114 projects/demo/derivatives/cat12 participant \
  --preproc \
  --participant-label 01 \
  --session-label test \
  --smooth-volume 6 \
  --smooth-surface 12 \
  --qa \
  --tiv \
  --no-validate
```

**Multiple subjects and sessions:**

```bash
# Process all 4 subjects, both sessions
# Note: Repeat --participant-label and --session-label for each value
./cat12_prepro openneuro/ds000114 projects/demo/derivatives/cat12 participant \
  --preproc \
  --participant-label 01 --participant-label 02 --participant-label 03 --participant-label 04 \
  --session-label test --session-label retest \
  --smooth-volume 6 \
  --smooth-surface 12 \
  --qa \
  --tiv \
  --no-validate
```

This will process all 8 images (4 subjects × 2 sessions) - approximately 2-4 hours total on a modern CPU.

### Run Statistics

```bash
bash ./scripts/analysis/cat12_multi_modality.sh \
  --config projects/demo/project_config.json \
  --cat12-dir projects/demo/derivatives/cat12 \
  --participants projects/demo/participants_demo.tsv
```

---

## Troubleshooting

### CAT12 Standalone Not Found

**Error:** `cat_standalone.sh: command not found`

**Solution:**
```bash
# Check if CAT12 was extracted
ls external/ | grep -i cat12

# If missing, manually extract
cd external
unzip CAT12*.zip
```

### MATLAB Runtime Not Found

**Error:** `Cannot locate MCR`

**Solution:**
- Verify MCR installation path matches your OS
- macOS: `/Applications/MATLAB/MATLAB_Runtime/R2023b`
- Linux: `/opt/MATLAB/MATLAB_Runtime/R2023b`

### Python Environment Issues

**Error:** `ModuleNotFoundError: No module named 'pybids'`

**Solution:**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Windows Support (Coming Soon)

Windows support requires:
1. **CAT12.9_R2023b_MCR_Win.zip** (available at neuro.uni-jena.de)
2. **MATLAB Runtime R2023b for Windows**
3. WSL2 or native Windows preprocessing support

For now, Windows users can:
- Use WSL2 with Linux instructions
- Use MATLAB directly (with license)
- Use Docker container (if available)

---

## Advanced Configuration

### Environment Variables

Edit `.env` file to customize paths:

```bash
# CAT12 Standalone
export CAT12_STANDALONE=/path/to/CAT12.9_R2023b_MCR_MAC_arm64
export MCR_ROOT=/Applications/MATLAB/MATLAB_Runtime/R2023b

# Use standalone
export USE_STANDALONE=true
```

### Custom Installation Paths

If you install MCR in a non-standard location, update `.env`:

```bash
export MCR_ROOT=/your/custom/mcr/path
export CAT12_STANDALONE=/your/custom/cat12/path
```

---

## Reference

- **CAT12 Downloads:** https://www.neuro.uni-jena.de/cat12/
- **CAT12 Documentation:** https://neuro-jena.github.io/cat12/
- **MATLAB Runtime:** https://mathworks.com/products/compiler/matlab-runtime.html

---

## More Information

- [INSTALL_GUIDE.md](INSTALL_GUIDE.md) - Original installation guide
- [STANDALONE_SETUP.md](STANDALONE_SETUP.md) - macOS-specific standalone setup
- [DEMO_RUN.md](DEMO_RUN.md) - Full demo workflow
- [README.md](README.md) - Project overview
