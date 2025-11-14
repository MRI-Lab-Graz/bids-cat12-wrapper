# CAT12 Longitudinal Analysis Pipeline - Usage Guide

## Quick Start

```bash
./cat12_longitudinal_analysis.sh \
    --cat12-dir /Volumes/Thunder/129_PK01/cat12 \
    --participants participants.tsv \
    --smoothing 6 \
    --force \
    --pilot
```

## All Available Flags

### Required Arguments
- `--cat12-dir <path>` - Path to CAT12 preprocessing output
- `--participants <tsv>` - Path to BIDS participants.tsv file

### Analysis Options
- `--modality <name>` - Analysis type: `vbm`, `thickness`, `depth`, `gyrification`, `fractal` (default: `vbm`)
- `--smoothing <mm>` - Smoothing kernel in mm or `auto` (default: auto-detect)
- `--analysis-name <name>` - Custom name for analysis (default: auto-generated)
- `--output-dir <path>` - Custom output directory (overrides default location)

### Design Options
- `--group-col <name>` - Column name for group variable (default: auto-detect)
- `--session-col <name>` - Column name for session variable (default: `session`)
- `--covariates <list>` - Comma-separated covariates: `"age,sex,tiv"`

### TFCE Options
- `--n-perm <N>` - Number of TFCE permutations (default: 5000)
- `--pilot` - Run pilot mode (100 permutations, faster testing)
- `--skip-screening` - Run TFCE on all contrasts (not recommended)
- `--n-jobs <N>` - Number of parallel jobs (default: 4)

### Other Options
- `--force` - Delete existing results directory before starting
- `--help` - Show help message

## Examples

### 1. Basic VBM Analysis (Default)
```bash
./cat12_longitudinal_analysis.sh \
    --cat12-dir /data/cat12 \
    --participants participants.tsv \
    --force
```

### 2. Pilot Test (Fast, for Testing)
```bash
./cat12_longitudinal_analysis.sh \
    --cat12-dir /data/cat12 \
    --participants participants.tsv \
    --pilot \
    --force
```

### 3. VBM with Specific Smoothing
```bash
./cat12_longitudinal_analysis.sh \
    --cat12-dir /data/cat12 \
    --participants participants.tsv \
    --smoothing 6 \
    --analysis-name "vbm_s6_final" \
    --force
```

### 4. Surface Analysis (Thickness)
```bash
./cat12_longitudinal_analysis.sh \
    --cat12-dir /data/cat12 \
    --participants participants.tsv \
    --modality thickness \
    --smoothing 15 \
    --force
```

### 5. With Covariates
```bash
./cat12_longitudinal_analysis.sh \
    --cat12-dir /data/cat12 \
    --participants participants.tsv \
    --covariates "age,sex,tiv" \
    --force
```

### 6. Custom Output Directory
```bash
./cat12_longitudinal_analysis.sh \
    --cat12-dir /data/cat12 \
    --participants participants.tsv \
    --output-dir /custom/path/my_analysis \
    --force
```

## Pipeline Steps

The pipeline automatically runs:

1. **Participant Parsing** - Match CAT12 files with participants.tsv
2. **SPM Design Generation** - Create well-conditioned factorial design
3. **GLM Estimation** - Estimate statistical model
4. **Contrast Specification** - Add all longitudinal contrasts
5. **Screening** - Find significant clusters (p<0.001 uncorrected)
6. **TFCE Correction** - FWE-corrected statistics (on significant contrasts)
7. **HTML Report** - Generate interactive analysis report

## Output Files

All results are saved in the output directory (default: `results/vbm/<analysis_name>/`):

### Main Files
- 📊 **report.html** - Interactive analysis report (open in browser)
- 📁 **SPM.mat** - Statistical model structure
- 🗺️ **spmT_*.nii** - T-statistic maps (uncorrected)
- 🔍 **screening_results.mat** - Screening results
- ✨ **TFCE_*/** - TFCE-corrected results directories
- 📈 **TFCE_*/logP_max.nii** - FWE-corrected p-value maps

### Additional Files
- `beta_*.nii` - Parameter estimate images
- `ResMS.nii` - Residual mean squares
- `mask.nii` - Analysis mask
- `spmF_*.nii` - F-statistic maps

## Design Matrix

The pipeline uses a **well-conditioned design** to avoid ill-conditioned matrices:

- ✅ **Factor 1: Group** (dept=0, between-subject, independent)
  - Levels: control, intervention_2w, intervention_4w

- ✅ **Factor 2: Time** (dept=1, within-subject, repeated measures)
  - Levels: Session 1, Session 2, Session 3

- ❌ **NO subject factors** - SPM handles within-subject dependencies implicitly

## BIDS participants.tsv Format

The pipeline expects BIDS-compliant participants.tsv:

```tsv
participant_id  nr_sessions  group
sub-1291003     3            control
sub-1291005     3            control
sub-1291043     3            intervention_2w
```

**Required columns:**
- `participant_id` - Subject identifier (e.g., `sub-1291003`)
- `nr_sessions` - Number of sessions per subject (e.g., `3`)
- `group` - Group assignment (column name can be customized with `--group-col`)

**Optional columns:**
- `age`, `sex`, `tiv`, etc. - Can be used as covariates

## Troubleshooting

### Issue: SPM not found
**Solution:** Create `spm_config.txt` with SPM path:
```bash
echo "/Volumes/Evo/software/spm25" > spm_config.txt
```

### Issue: Old SPM.mat causing conflicts
**Solution:** Use `--force` flag to clean output directory:
```bash
./cat12_longitudinal_analysis.sh --force ...
```

### Issue: Missing files
**Solution:** Check smoothing kernel:
```bash
# Count available files
find /data/cat12 -name "s6mwp1r*.nii" | wc -l   # 6mm smoothing
find /data/cat12 -name "s8mwp1r*.nii" | wc -l   # 8mm smoothing

# Use explicit smoothing
./cat12_longitudinal_analysis.sh --smoothing 6 ...
```

### Issue: Wrong design matrix
**Solution:** The pipeline automatically creates well-conditioned designs. Check:
- No subject factors in design (correct: Group × Time only)
- All subjects have same number of sessions in participants.tsv
- Group labels match across all rows

## Tips

1. **Always use `--force` during development** to avoid conflicts with old results
2. **Use `--pilot` for quick testing** (100 permutations vs 5000)
3. **Check report.html first** before diving into SPM
4. **Specify `--smoothing` explicitly** if auto-detection picks wrong kernel
5. **Use `--output-dir` for multiple analyses** to keep results organized

## Performance

- **Pilot mode**: ~5-10 minutes (100 permutations)
- **Full analysis**: ~1-2 hours (5000 permutations)
- **Parallel jobs**: Use `--n-jobs` to adjust (default: 4)

## Next Steps After Analysis

1. **Open report.html** in browser to review analysis summary
2. **Load SPM.mat** in SPM GUI to explore results
3. **Check TFCE results** in `TFCE_*/` directories
4. **Extract clusters** from significant maps
5. **Create publication figures** using SPM or external tools
