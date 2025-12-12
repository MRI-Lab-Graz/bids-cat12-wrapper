# Scripts Directory

This directory contains the main processing scripts for the CAT12 pipeline.

## Structure

- **preprocessing/**: Scripts for BIDS validation and CAT12 preprocessing (segmentation).
  - `bids_cat12_processor.py`: Main entry point for preprocessing.
  - `subject_processor.py`: Helper for individual subject processing.

- **stats/**: Scripts for longitudinal statistical analysis.
  - `cat12_stats`: Wrapper (recommended) that runs the stats pipeline.
  - `cat12_longitudinal_analysis.sh`: Pipeline implementation (invoked by wrapper).
  - `run_matlab_standalone.py`: Wrapper for running MATLAB/Standalone commands.
  - `utils/`: Helper scripts for the stats pipeline.
  - `templates/`: Templates for the stats pipeline.

## Usage

Prefer using the wrapper scripts in the root directory:
- `../run_cat12.sh` -> Runs preprocessing.
- `../run_stats.sh` -> Runs stats analysis.
