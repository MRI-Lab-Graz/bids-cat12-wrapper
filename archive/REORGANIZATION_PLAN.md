# Workspace Reorganization Plan

## Current Problem
The stats directory has become cluttered with:
- 20+ documentation files (many redundant)
- Mixed scripts (MATLAB, Python, Bash)
- Template files scattered everywhere
- Old/deprecated files
- Analysis outputs mixed with scripts

## Proposed Structure

```
stats/
├── README.md                          # Main entry point
├── run_longitudinal_pipeline.sh       # MAIN WORKFLOW SCRIPT
│
├── 01_design/                         # Design & Setup Phase
│   ├── make_ff_design_random_effects.m
│   ├── make_ff_design_surface_from_csv.m
│   ├── check_design_structure.m
│   ├── validate_design_stability.m
│   └── templates/
│       ├── template_3x3_vbm.m
│       ├── template_3x3_sbm.m
│       └── template_3x3_sbm_with_covariates.m
│
├── 02_covariates/                     # Covariate Management
│   ├── manage_covariates.py
│   ├── add_covariates.py
│   ├── verify_covariates.py
│   ├── data/
│   │   ├── TIV.txt
│   │   ├── IQR.txt
│   │   └── design_mri_s6.csv
│   └── docs/
│       ├── COVARIATE_VERIFICATION_GUIDE.md
│       └── MANAGE_COVARIATES_README.md
│
├── 03_contrasts/                      # Contrast Specification
│   ├── add_contrasts_longitudinal.m
│   ├── list_contrasts.m
│   ├── simple_contrasts.m
│   └── outputs/
│       ├── contrast_all.mat
│       └── full_contrast_list.mat
│
├── 04_estimation/                     # Model Estimation
│   ├── run_estimation.sh
│   └── check_estimation_output.m
│
├── 05_screening/                      # Uncorrected Screening
│   ├── screen_contrasts.m            # Extract from run_screen_and_tfce.m
│   └── thresholding.m
│
├── 06_tfce/                           # TFCE Correction
│   ├── run_tfce_headless.sh
│   ├── run_tfce_correction.m         # Extract from run_screen_and_tfce.m
│   ├── monitor_tfce_progress.sh
│   ├── check_tfce_output_data.m
│   ├── spm_figure.m                  # Shadow function (CRITICAL)
│   └── logs/
│
├── utils/                             # Utility Scripts
│   ├── configure_spm_path.m
│   ├── find_spm_path.m
│   ├── diagnose_design_dependencies.m
│   ├── inspect_spm_mat.m
│   └── convert_to_surface.py
│
├── docs/                              # Documentation
│   ├── README.md                      # Main documentation
│   ├── QUICK_REFERENCE.md
│   ├── WORKFLOW_GUIDE.md              # New: step-by-step guide
│   └── TFCE_PROCESSING_EXPLAINED.md
│
├── archive/                           # Old/Deprecated Files
│   ├── batch_3x3.m
│   ├── batch_3x3.mat
│   ├── Ohne Titel.mat
│   ├── startup.m
│   └── old_reports/
│
└── results/                           # Analysis Outputs (gitignored)
    ├── vbm/
    │   ├── s9_int_control/
    │   ├── s9_int_control_cov/
    │   └── vol_mri_s6_*/
    └── sbm/
        ├── surf_int_control/
        ├── thickness/
        ├── depth/
        ├── gyrification/
        └── fractal/
```

## Main Workflow Script: `run_longitudinal_pipeline.sh`

This script orchestrates the entire analysis:

```bash
#!/bin/bash
# Complete pipeline for longitudinal VBM/SBM analysis

# Step 1: Design specification
# Step 2: Add covariates
# Step 3: Add contrasts
# Step 4: Estimate model
# Step 5: Screen contrasts (uncorrected)
# Step 6: TFCE correction on significant contrasts
```

## Implementation Steps

1. Create new directory structure
2. Move files to appropriate locations
3. Create unified pipeline script
4. Update documentation
5. Test workflow
6. Clean up old files

## Benefits

- **Clear workflow**: 01 → 02 → 03 → 04 → 05 → 06
- **Single entry point**: run_longitudinal_pipeline.sh
- **Organized outputs**: results/ separated from scripts
- **Easy maintenance**: Related files grouped together
- **Better documentation**: Consolidated in docs/
