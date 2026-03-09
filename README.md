# CAT12 Statistics Analysis Pipeline

**Clean workflow for statistical analysis of CAT12-preprocessed neuroimaging data**

---

## What This Pipeline Does

1. **Automated Statistical Workflows** - Design matrices, SPM estimation, TFCE correction
2. **Interactive Result Reports** - Multi-atlas visualization, filtering, cluster exploration
3. **Threshold Sweeping** - Automated double-threshold + effect size map generation

---

## Quick Start

Practical examples for the most common workflow in this repo:

### 1. Activate environment
```bash
cd /Volumes/Thunder/129_PK01/cat12/stats
source .venv/bin/activate
```

### 2. Run stats + sweep + report from one config
```bash
python run_pipeline.py \
  --config config/config_14_2_26.json \
  --cat12-dir /Volumes/Thunder/129_PK01/cat12/data/cat12 \
  --only stats,sweep,report
```

Notes:
- `stats` runs design/estimate.
- `sweep` runs threshold/effect-size/TFCE sweep.
- `report` generates interactive HTML.

### 3. Regenerate report from existing results (report-only)
Use this when stats already finished and you only want a new HTML.

```bash
python scripts/reporting/post_stats_report.py \
  results/vbm/vbm_9mm_social_2TP_tiv_sex_age \
  results/vbm/vbm_9mm_social_2TP_tiv_sex_age/report_tfce.html \
  --filter all \
  --quality low \
  --config config/config_14_2_26.json
```

### 4. Open report
```bash
open results/vbm/vbm_9mm_social_2TP_tiv_sex_age/report_tfce.html
```

### 5. Quick interpretation examples

- If TFCE files exist but you do not see them in HTML, regenerate with `--filter all` (or `--filter tfce`).
- Contrast labels are not always pairwise in the way the name suggests. Check the embedded vectors in report JS:
  - `Positive effect of Group_1` (contrast `6`) = `[1,1,-1,-1,0,0,...]` = Group1 vs Group2.
  - `Overall: alone - control` (contrast `31`) = `[1,1,0,0,-1,-1,...]` = Group1 vs Group3.

---

## Directory Structure

```
scripts/
├── analysis/           Main workflows
│   ├── cat12_longitudinal_analysis.sh
│   ├── run_stats_sweep.py          ← Interactive sweep + report
│   └── rebuild_participants.py
├── reporting/          Report generation
│   └── post_stats_report.py        ← Interactive HTML reports
└── utils/              Shared utilities (40+ files)

results/
├── vbm/                VBM analysis results
└── data/               Participant data files

config/
├── config.ini          Main configuration
└── spm_config.txt      SPM-specific settings

logs/                   Execution logs
tmp/                    Temporary files
```

---

## Main Scripts

### `run_stats_sweep.py`
**Purpose**: Automated statistical analysis with report generation  
**Input**: Results directory with SPM.mat  
**Output**: Interactive HTML report + threshold maps

```bash
python scripts/analysis/run_stats_sweep.py <results_dir> [--use-matlab]
```

### `post_stats_report.py`
**Purpose**: Generate interactive HTML report from existing maps  
**Input**: Results directory with threshold maps  
**Output**: HTML report with filtering, clustering, multi-atlas

```bash
python scripts/reporting/post_stats_report.py <results_dir> <output_file>
```

### `cat12_longitudinal_analysis.sh`
**Purpose**: CAT12 statistical design and estimation  
**Input**: CAT12 preprocessed data + participant file  
**Output**: SPM results directory

```bash
./scripts/analysis/cat12_longitudinal_analysis.sh --cat12-dir <path> --participants <file>
```

---

## Interactive Report Features

✨ **Multi-level Filtering**
- By p-value threshold
- By correction type (FWE, FDR, TFCE, double-threshold)
- By contrast
- By atlas

✨ **Visualizations**
- Glass-brain (3D overlay)
- 4-view surface mesh (frontal, parietal, etc.)
- Cluster galleries (top 5 peaks)

✨ **Multi-Atlas Support**
- 5 volume atlases (AAL3, Harvard-Oxford, COBRA, etc.)
- 4 surface atlases (Desikan-Killiany, Destrieux, etc.)

---

## Configuration

All settings in `config/config.json`:

```json
{
  "matlab": {
    "executable": "/Applications/MATLAB_R2025b.app/bin/matlab",
    "allow_graphics": false
  },
  "spm": {
    "path": "/Volumes/Evo/software/spm25/"
  },
  "analysis": {
    "modality": "vbm",
    "smoothing_kernel": 8,
    "group_column": "group_beh",
    "covariates": ["tiv"]
  }
}
```

### Study-Specific Configs

Create variants for different studies:

```bash
# Base config (all studies)
config/config.json

# Study-specific overrides
config/config.study_intervention.json
config/config.study_controls.json
```

Then use:
```bash
python scripts/analysis/run_stats_sweep.py ./results/vbm/analysis \
  --config config/config.study_intervention.json
```

---

## Participant Data

Participant TSV files stored in `results/data/`:
- `participants.tsv` - Main participant file
- `*_females_only.tsv` - Filtered subsets
- `*_intervention_control.tsv` - Group assignments

---

## Output Files

- **Interactive Reports**: `post_stats_sweep_report.html`
- **Threshold Maps**: `double_threshold_*.nii` / `effectsize_*.nii`
- **Log Files**: `logs/analysis_*.log`
- **SPM Results**: `results/vbm/analysis/SPM.mat` (main results file)

---

## Common Workflows

### Full Analysis (Design → Estimate → Report)
```bash
# Step 1: Statistical analysis
./scripts/analysis/cat12_longitudinal_analysis.sh \
  --cat12-dir /path/to/cat12 \
  --participants results/data/participants.tsv

# Step 2: Generate report with sweep
python scripts/analysis/run_stats_sweep.py ./results/vbm/analysis
```

### Report Only (from existing results)
```bash
python scripts/reporting/post_stats_report.py \
  ./results/vbm/analysis \
  report.html
```

### Interactive Exploration
```bash
open results/vbm/analysis/post_stats_sweep_report.html
# Click rows to update plots
# Use dropdowns to change atlas/correction
```

---

## Supported Analysis Types

- **Correction Methods**: FWE (voxel), FDR (voxel), TFCE (permutation), Double-threshold, Effect size
- **Modalities**: VBM (gray matter), Surface (thickness, depth, gyrification), Custom maps
- **Designs**: 2-group, multi-group, continuous covariates, longitudinal

---

## Troubleshooting

**MATLAB not found?**  
Update `config/config.json` with correct MATLAB path

**Report not generating?**  
Check `logs/analysis_*.log` for errors

**Missing atlases?**  
Script searches multiple locations automatically

**Which config to use?**  
- Use `config/config.json` for default/generic studies
- Use `config/config.study_*.json` for specific studies
- Overrides from command-line flags take precedence

---

## Need More Details?

All scripts have built-in help:

```bash
python scripts/analysis/run_stats_sweep.py --help
python scripts/reporting/post_stats_report.py --help
./scripts/analysis/cat12_longitudinal_analysis.sh --help
```

---

**Architecture**: Modular Python + MATLAB | **Updated**: Feb 1, 2026
