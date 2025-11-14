# CAT12 Flexible Factorial Design - Complete File Summary

## Overview
This directory contains batch files for CAT12 flexible factorial design analysis with both volume-based and surface-based approaches, with and without covariates.

---

## Batch Files for SPM/CAT12

### Volume-Based Analysis (VBM)
| File | Description | Output Directory | Covariates |
|------|-------------|------------------|------------|
| `batch_3x3_job.m` | Volume-based factorial design | `s9_int_control/` | No |
| `batch_3x3_job_with_covariates.m` | Volume-based with TIV & IQR | `s9_int_control_cov/` | Yes (TIV, IQR) |

### Surface-Based Analysis (SBM)
| File | Description | Output Directory | Covariates | Contrasts |
|------|-------------|------------------|------------|-----------|
| `template_surface_job.m` | **RECOMMENDED** - Complete surface analysis | `surf_int_control/` | No | 35 |
| `batch_3x3_surface_job.m` | Surface-based factorial design | `surf_int_control/` | No | 0 |
| `batch_3x3_surface_job_with_covariates.m` | Surface-based with TIV & IQR | `surf_int_control_cov/` | Yes (TIV, IQR) | 0 |

---

## Covariate Data Files

| File | Description | Values | Source |
|------|-------------|--------|--------|
| `TIV.txt` | Total Intracranial Volume | 411 | CAT12 segmentation |
| `IQR.txt` | Image Quality Rating | 411 | CAT12 quality control |
| `IQR_job.m` | Reference for file order | - | Used for ordering |

---

## Python Scripts

### Main Processing Scripts
| File | Purpose |
|------|---------|
| `add_covariates.py` | Adds TIV and IQR covariates to batch files in correct order |
| `convert_to_surface.py` | Converts volume-based paths to surface-based paths |
| `convert_template_to_surface.py` | Converts template_job.m to surface (no covariates, no threshold) |
| `verify_covariates.py` | Verifies covariate order matches file order |

### Usage Examples
```bash
# Add covariates to batch file
python3 add_covariates.py

# Convert to surface-based analysis
python3 convert_to_surface.py

# Verify covariate order
python3 verify_covariates.py
```

---

## Documentation

| File | Description |
|------|-------------|
| `README.md` | This file - complete summary |
| `COVARIATE_REPORT.md` | Detailed report on covariate addition process |
| `SURFACE_ANALYSIS_REPORT.md` | Detailed report on surface-based conversion |
| `TEMPLATE_SURFACE_REPORT.md` | Template surface conversion (main effects, interactions, 35 contrasts) |

---

## Analysis Design

### Factorial Design: 3 × 3
- **Factor 1 - Group** (between-subject, 3 levels):
  - 2w intervention group (53 subjects)
  - 4w intervention group (31 subjects)
  - Control group (40 subjects)

- **Factor 2 - Time** (within-subject, 3 levels):
  - Session 1 (baseline)
  - Session 2 (follow-up 1)
  - Session 3 (follow-up 2)

- **Total scans**: 369 (124 subjects, but not all have 3 timepoints)

### Covariates (when included)
1. **TIV** - Total Intracranial Volume
   - Purpose: Control for head size
   - Interaction: Enabled (iCFI = 1)
   - Centering: Enabled (iCC = 1)

2. **IQR** - Image Quality Rating
   - Purpose: Control for image quality
   - Interaction: Enabled (iCFI = 1)
   - Centering: Enabled (iCC = 1)

---

## File Path Conventions

### Volume-Based (VBM)
```
/Volumes/Thunder/129_PK01/cat12/s9/<group>/<subject>_<session>.nii,1

Groups: 2w_group, 2w_single, 4w_group, 4w_single, control
Example: s9/2w_group/sub-1291145_ses-1.nii,1
```

### Surface-Based (SBM)
```
/Volumes/Thunder/129_PK01/cat12/data/cat12/<subject>/surf/s15.mesh.thickness.resampled_32k.r<subject>_<session>_acq-mprage_T1w.gii

Example: data/cat12/sub-1291145/surf/s15.mesh.thickness.resampled_32k.rsub-1291145_ses-1_acq-mprage_T1w.gii
```

---

## Workflow

### 1. Volume-Based Analysis (VBM)

#### Without Covariates:
1. Open SPM/CAT12 in MATLAB
2. Load `batch_3x3_job.m`
3. Run the batch
4. Results in `s9_int_control/`

#### With Covariates:
1. Open SPM/CAT12 in MATLAB
2. Load `batch_3x3_job_with_covariates.m`
3. Verify covariates in GUI
4. Run the batch
5. Results in `s9_int_control_cov/`

### 2. Surface-Based Analysis (SBM)

#### Without Covariates:
1. Open SPM/CAT12 in MATLAB
2. Load `batch_3x3_surface_job.m`
3. Run the batch
4. Results in `surf_int_control/`

#### With Covariates:
1. Open SPM/CAT12 in MATLAB
2. Load `batch_3x3_surface_job_with_covariates.m`
3. Verify covariates in GUI
4. Run the batch
5. Results in `surf_int_control_cov/`

---

## Quality Assurance

### ✓ Verification Steps Completed
- [x] All file paths extracted correctly
- [x] Subject/session matching verified
- [x] Covariate order matches file order (100% match)
- [x] 369 files in each batch
- [x] Volume to surface conversion successful
- [x] Output directories set appropriately
- [x] No missing data in covariates

### ⚠ Important Checks Before Running
- [ ] Ensure all .nii files exist (for VBM)
- [ ] Ensure all .gii surface files exist (for SBM)
- [ ] Verify file permissions
- [ ] Check MATLAB/SPM version compatibility
- [ ] Ensure sufficient disk space for results

---

## File Statistics

### Volume-Based Batch Files
- Files per batch: 369
- File format: NIfTI (.nii)
- Smoothing: s9 (9mm kernel)
- Space: MNI152

### Surface-Based Batch Files
- Files per batch: 369
- File format: GIFTI (.gii)
- Smoothing: s15 (15mm kernel)
- Vertices: 32k (resampled)
- Measure: Cortical thickness

---

## Contact & Support

For issues or questions about:
- **File ordering**: Check `COVARIATE_REPORT.md`
- **Surface conversion**: Check `SURFACE_ANALYSIS_REPORT.md`
- **Script usage**: See Python script headers for documentation

---

*Last updated: 2025-10-16*
*Location: /Volumes/Thunder/129_PK01/cat12/stats/*
