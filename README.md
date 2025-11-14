# CAT12 Longitudinal Analysis Pipeline

Automated workflow for longitudinal VBM and surface-based morphometry analysis using CAT12 preprocessed data.

## Quick Start

```bash
./cat12_longitudinal_analysis.sh \
    --cat12-dir /path/to/cat12 \
    --participants /path/to/participants.tsv
```

## Common Usage

```bash
# Basic VBM analysis
./cat12_longitudinal_analysis.sh \
    --cat12-dir /Volumes/Thunder/129_PK01/cat12 \
    --participants participants.tsv

# Cortical thickness
./cat12_longitudinal_analysis.sh \
    --cat12-dir /Volumes/Thunder/129_PK01/cat12 \
    --participants participants.tsv \
    --modality thickness

# Quick test (pilot mode)
./cat12_longitudinal_analysis.sh \
    --cat12-dir /Volumes/Thunder/129_PK01/cat12 \
    --participants participants.tsv \
    --pilot

# With covariates
./cat12_longitudinal_analysis.sh \
    --cat12-dir /Volumes/Thunder/129_PK01/cat12 \
    --participants participants.tsv \
    --covariates "age,sex,tiv"
```

## Key Options

| Option | Default | Description |
|--------|---------|-------------|
| `--cat12-dir` | Required | CAT12 preprocessing directory |
| `--participants` | Required | BIDS participants.tsv file |
| `--modality` | `vbm` | vbm, thickness, depth, gyrification, fractal |
| `--smoothing` | auto | Kernel size in mm |
| `--group-col` | auto-detect | Group column name |
| `--covariates` | none | Comma-separated (e.g., "age,sex,tiv") |
| `--n-perm` | 5000 | TFCE permutations |
| `--pilot` | off | Quick test (100 perms) |
| `--force` | off | Clean results before starting |

## Configuration

Edit `config.ini` to customize defaults:

**System Paths:**
- `[MATLAB]` - MATLAB executable path (auto-detected if empty)
- `[SPM]` - SPM installation path (optional)
- `[PYTHON]` - Python 3 executable

**Analysis Defaults:**
- `[ANALYSIS]` - modality, smoothing, group_col, sessions, covariates
- `[SCREENING]` - uncorrected_p, cluster_size, skip_screening
- `[TFCE]` - n_perm, pilot_mode
- `[PERFORMANCE]` - parallel_jobs, memory_limit_gb
- `[OUTPUT]` - output_dir, analysis_name, force_clean

**Command-line arguments override config.ini values:**

```bash
# Config says n_perm=5000, but this will override it:
./cat12_longitudinal_analysis.sh \
    --cat12-dir /path/to/cat12 \
    --participants participants.tsv \
    --n-perm 10000              # Overrides config.ini setting
```

## Pipeline Steps

1. Parse participants.tsv and find CAT12 files
2. Generate SPM factorial design
3. Estimate GLM model
4. Add contrasts
5. Screen significant effects (p<0.001 uncorrected)
6. TFCE correction (family-wise error)
7. Generate HTML report

## Results

Results are saved to:
```
results/<modality>/<analysis_name>/
├── report.html                    # Analysis report (open in browser)
├── spm_batch.m                    # SPM batch file (reproducibility)
├── SPM.mat                        # Statistical model
├── design_matrix.png              # Design visualization
├── spmT_*.nii                     # T-statistic maps
├── con_*.nii                      # Contrast maps
└── TFCE_*/logP_max.nii           # FWE-corrected results
```

**Quick Access:**
A symbolic link `report_latest.html` in the script directory points to the most recent HTML report.

```bash
# Open latest results directly
open report_latest.html     # macOS
xdg-open report_latest.html # Linux
```

## Help

```bash
./cat12_longitudinal_analysis.sh --help
```

---

**Pipeline Structure:**
- `cat12_longitudinal_analysis.sh` - Main entry point
- `utils/` - MATLAB and Python helper functions
- `templates/` - Analysis templates (brainmask)
- `archive/` - Old/unused scripts (reference only)
