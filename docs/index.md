# CAT12 Standalone Wrapper (Preprocessing + Longitudinal Statistics)

This repository provides a **standalone, reproducible wrapper** around CAT12 (SPM12) for:

- **BIDS-compatible preprocessing** (cross-sectional and longitudinal, auto-detected)
- **Longitudinal group statistics** (VBM + surface modalities) with screening and TFCE correction

It is designed for **headless Linux servers** and does **not require a MATLAB license** when using the bundled CAT12 standalone + MATLAB Runtime.

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
workflows
usage/preprocessing
usage/statistics
troubleshooting
```

```{toctree}
:maxdepth: 1
:caption: Reference

QUICK_START
BIDS_APP_USAGE
CAT12_COMMANDS
PROJECT_SUMMARY
```

## What you get

- **Two entrypoints**
  - `./cat12_prepro`: preprocessing (BIDS App style)
  - `./cat12_stats`: longitudinal group analysis
- **Contained install** under the repo directory:
  - `external/` for CAT12 + MATLAB Runtime (+ local Deno)
  - `.venv/` for Python dependencies

## When to use which workflow

- Run **preprocessing** first to produce CAT12 derivatives.
- Then run **statistics** on the derivatives (typically `vbm` or `thickness`) and your `participants.tsv`.
