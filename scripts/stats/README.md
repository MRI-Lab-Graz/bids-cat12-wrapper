# CAT12 Longitudinal Analysis Pipeline

Automated workflow for longitudinal VBM and surface-based morphometry analysis using CAT12 preprocessed data.

## Quick Start

Preferred: use the root `cat12_stats` wrapper which sets up the virtualenv and
optional background logging (`--nohup`). It invokes the pipeline script
`scripts/stats/cat12_longitudinal_analysis.sh` under the hood.

```bash
cat12_stats \
    --cat12-dir /path/to/cat12 \
    --participants /path/to/participants.tsv
```

## Common Usage

```bash
# Basic VBM analysis
cat12_stats \
    --cat12-dir /Volumes/Thunder/129_PK01/cat12 \
    --participants participants.tsv

# Cortical thickness
cat12_stats \
    --cat12-dir /Volumes/Thunder/129_PK01/cat12 \
    --participants participants.tsv \
    --modality thickness

# Quick test (pilot mode)
cat12_stats \
    --cat12-dir /Volumes/Thunder/129_PK01/cat12 \
    --participants participants.tsv \
    --pilot

# With covariates
cat12_stats \
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

Edit `config.ini` to customize defaults.

**Command-line arguments override `config.ini` values.**

```bash
# Config says n_perm=5000, but this will override it:
cat12_stats \
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
xdg-open report_latest.html # Linux
```

## Help

```bash
cat12_stats --help
```

---

**Pipeline Structure:**
- `cat12_stats` - Root wrapper (recommended entry point)
- `scripts/stats/cat12_longitudinal_analysis.sh` - Pipeline implementation
- `utils/` - MATLAB and Python helper functions
- `templates/` - Analysis templates (brainmask)
- `archive/` - Old/unused scripts (reference only)
