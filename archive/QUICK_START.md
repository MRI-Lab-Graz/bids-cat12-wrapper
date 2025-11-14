# CAT12 Longitudinal Analysis - Quick Reference

## 🚀 Basic Command

```bash
./cat12_longitudinal_analysis.sh \
    --cat12-dir /path/to/cat12 \
    --participants /path/to/participants.tsv \
    --group-col <column_name>
```

## 📋 Common Usage Patterns

### VBM Analysis (Default)
```bash
./cat12_longitudinal_analysis.sh \
    --cat12-dir /Volumes/Thunder/129_PK01/cat12 \
    --participants /Volumes/Thunder/129_PK01/participants.tsv \
    --group-col intervention_group
```

### Cortical Thickness
```bash
./cat12_longitudinal_analysis.sh \
    --cat12-dir /Volumes/Thunder/129_PK01/cat12 \
    --participants /Volumes/Thunder/129_PK01/participants.tsv \
    --modality thickness \
    --group-col group
```

### With Covariates
```bash
./cat12_longitudinal_analysis.sh \
    --cat12-dir /Volumes/Thunder/129_PK01/cat12 \
    --participants /Volumes/Thunder/129_PK01/participants.tsv \
    --covariates "age,sex,tiv" \
    --group-col group
```

### Quick Test (Pilot Mode)
```bash
./cat12_longitudinal_analysis.sh \
    --cat12-dir /Volumes/Thunder/129_PK01/cat12 \
    --participants /Volumes/Thunder/129_PK01/participants.tsv \
    --pilot
```

### Production Run (High Permutations)
```bash
./cat12_longitudinal_analysis.sh \
    --cat12-dir /Volumes/Thunder/129_PK01/cat12 \
    --participants /Volumes/Thunder/129_PK01/participants.tsv \
    --n-perm 10000 \
    --group-col group
```

## 📊 Modalities

| Modality | Description | Default Smoothing |
|----------|-------------|-------------------|
| `vbm` | Gray matter volume | 8mm |
| `thickness` | Cortical thickness | 15mm |
| `depth` | White matter depth | 15mm |
| `gyrification` | Gyrification index | 15mm |
| `fractal` | Fractal dimension | 15mm |

## 🔧 All Options

| Option | Default | Description |
|--------|---------|-------------|
| `--cat12-dir <path>` | **Required** | CAT12 preprocessing directory |
| `--participants <tsv>` | **Required** | BIDS participants.tsv file |
| `--modality <name>` | `vbm` | Analysis type |
| `--smoothing <mm>` | 8 or 15 | Kernel size |
| `--analysis-name <name>` | Auto | Custom analysis name |
| `--group-col <name>` | Auto-detect | Group column |
| `--session-col <name>` | `session` | Session column |
| `--covariates <list>` | None | Comma-separated covariates |
| `--n-perm <N>` | 5000 | TFCE permutations |
| `--pilot` | Off | Quick test mode |
| `--skip-screening` | Off | Run TFCE on all contrasts |
| `--n-jobs <N>` | 4 | Parallel jobs |

## 📁 Participants.tsv Format

Required columns:
- `participant_id`: Subject IDs (e.g., sub-001)
- `session`: Session numbers (e.g., 1, 2, 3)
- Group column (name specified via `--group-col`)

Optional columns (for `--covariates`):
- `age`, `sex`, `tiv`, `education`, etc.

Example: `docs/participants_example.tsv`

## 📂 Results Location

```
results/<modality>/<analysis_name>/
├── SPM.mat                    # Statistical model
├── spmT_*.nii                 # Uncorrected T-maps
├── screening_results.mat      # Screening results
└── TFCE_*/                    # TFCE-corrected results
    ├── logP_max.nii          # FWE-corrected p-values
    ├── TFCE_max.nii          # TFCE statistic
    └── cluster_table.txt     # Cluster report
```

## ⚡ Workflow Steps

1. Parse participants.tsv → Find CAT12 files
2. Generate SPM batch → Factorial design (Group x Time, **well-conditioned**)
3. Estimate model → GLM
4. Add contrasts → Automatic
5. Screen contrasts → p<0.001 uncorrected
6. TFCE correction → FWE-corrected

**Important**: Uses proper repeated measures design (NO subject factors) to avoid ill-conditioned matrices. See `docs/DESIGN_MATRIX_TECHNICAL_NOTE.md` for details.

## 🆘 Troubleshooting

### No files found
```bash
# Check your paths
ls /path/to/cat12/mri/*.nii         # VBM
ls /path/to/cat12/surf/*.gii        # Surface
```

### Can't auto-detect group column
```bash
# Specify explicitly
--group-col "intervention_group"
```

### Out of memory during TFCE
```bash
# Reduce parallel jobs
--n-jobs 2
```

### Test before full run
```bash
# Always use pilot mode first!
--pilot
```

## 💡 Pro Tips

✅ **Always run `--pilot` first** to verify setup  
✅ **Check screening results** before full TFCE  
✅ **Use covariates** for confound control  
✅ **Monitor disk space** (TFCE creates large files)  
✅ **Backup participants.tsv** before starting  

## 📖 Help

```bash
./cat12_longitudinal_analysis.sh --help
```

---

**Last Updated**: November 2025
