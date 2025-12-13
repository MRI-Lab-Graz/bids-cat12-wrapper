# Workflows (Preprocessing → Longitudinal Statistics)

This repo is built around two workflows:

1. **Preprocessing**: run CAT12 standalone on a BIDS dataset and produce BIDS-derivatives outputs.
2. **Statistics**: run a longitudinal group model (VBM or surface modalities) on CAT12 outputs, screen contrasts, and optionally run TFCE correction.

## Workflow A — Preprocessing (BIDS App style)

Entry point: `./cat12_prepro`

Inputs:
- BIDS dataset (`bids_dir`)
- Output derivatives directory (`output_dir`)

Outputs (typical):
- `output_dir/dataset_description.json`
- `output_dir/sub-*/ses-*/mri/…` (VBM volumes like `mwp1*`, optionally smoothed)
- `output_dir/sub-*/ses-*/surf/…` (thickness etc., optionally resampled/smoothed)
- `output_dir/quality_measures_*.csv`, `TIV.txt` (if enabled)

Longitudinal behavior:
- If a subject has **2+ sessions**, the processor selects the CAT12 longitudinal batch template.
- If a subject has **1 session**, it runs the cross-sectional template.

## Workflow B — Longitudinal statistics

Entry point: `./cat12_stats`

Inputs (standard mode):
- `--cat12-dir`: the preprocessing output directory
- `--participants`: a BIDS `participants.tsv`

Outputs (typical):
- `results/<modality>/<analysis_name>/`
  - `SPM.mat`, contrast images, screening summary
  - TFCE outputs (when enabled)
  - `report.html`
  - `logs/pipeline.log`

TFCE behavior:
- The pipeline supports a **two-stage (probe → full)** strategy by default to detect permutation instability and adjust nuisance handling.

## Recommended end-to-end pattern

```bash
# 1) Preprocess into derivatives
./cat12_prepro /data/bids /data/derivatives/cat12 participant --preproc --qa --tiv

# 2) Run longitudinal group analysis
./cat12_stats --cat12-dir /data/derivatives/cat12 --participants /data/bids/participants.tsv --modality vbm
```
