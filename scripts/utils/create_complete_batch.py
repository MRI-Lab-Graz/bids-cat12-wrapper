#!/usr/bin/env python3
"""
Create a complete SPM batch file that runs design + model estimation
(suitable for standalone mode without custom functions).

This combines the factorial design with model estimation in a single
batch, avoiding the need for addpath() or custom CAT12 functions.
"""

import json
import sys
from pathlib import Path


def create_complete_batch(design_file, output_dir, modality, output_file):
    """Generate batch file with design + estimation."""
    
    with open(design_file) as f:
        design = json.load(f)
    
    safe_output = str(output_dir).replace("'", "''")
    
    # Build cells
    cell_code = []
    cell_idx = 1
    for finfo in design["files"]:
        group = finfo.get("group")
        session = finfo.get("session")
        filepath = finfo.get("filepath", "")
        
        # Add volume index for volumetric data
        if modality in ("vbm", "vbm_dartel") and not filepath.endswith(",1"):
            filepath = f"{filepath},1"
        
        filepath_safe = filepath.replace("'", "''")
        
        if cell_idx == 1:
            cell_code.append(f"% Cell {cell_idx}: {group} × Session {session}")
            cell_code.append(f"matlabbatch{{1}}.spm.stats.factorial_design.des.fd.icell({cell_idx}).scans = {{")
        elif cell_idx > 1 and (cell_idx - 1) % (len(design["sessions"])) == 0:
            cell_code.append(f"\n% Next group")
        
        if cell_idx  == 1 or cell_code[-1].startswith("matlabbatch"):
            pass
        
        cell_code.append(f"    '{filepath_safe}'")
        cell_idx += 1
    
    if cell_code:
        cell_code.append("    };")
    
    # Build covariates
    cov_code = []
    for cov_idx, (cov_name, cov_values) in enumerate(design.get("covariates", {}).items(), 1):
        cov_name_safe = cov_name.replace("'", "''")
        cov_code.append(f"matlabbatch{{1}}.spm.stats.factorial_design.cov({cov_idx}).c = [")
        for val in cov_values:
            cov_code.append(f"    {val};")
        cov_code.append("];")
        cov_code.append(f"matlabbatch{{1}}.spm.stats.factorial_design.cov({cov_idx}).cname = '{cov_name_safe}';")
        cov_code.append(f"matlabbatch{{1}}.spm.stats.factorial_design.cov({cov_idx}).iCFI = 1;")
        cov_code.append(f"matlabbatch{{1}}.spm.stats.factorial_design.cov({cov_idx}).iCC = 1;")
    
    # Build complete batch
    batch_content = f"""% Complete SPM Batch: Design + Model Estimation
% Auto-generated for standalone mode
% {modality}, {design['smoothing']}mm smoothing
% {len(design['groups'])} groups × {len(design['sessions'])} sessions

% ================================================================
% JOB 1: Factorial Design Specification
% ================================================================
matlabbatch{{1}}.spm.stats.factorial_design.dir = {{'{safe_output}'}};

% Factors
matlabbatch{{1}}.spm.stats.factorial_design.des.fd.fact(1).name = 'Group';
matlabbatch{{1}}.spm.stats.factorial_design.des.fd.fact(1).levels = {len(design['groups'])};
matlabbatch{{1}}.spm.stats.factorial_design.des.fd.fact(1).dept = 0;
matlabbatch{{1}}.spm.stats.factorial_design.des.fd.fact(1).variance = 1;
matlabbatch{{1}}.spm.stats.factorial_design.des.fd.fact(1).gmsca = 0;
matlabbatch{{1}}.spm.stats.factorial_design.des.fd.fact(1).ancova = 0;

matlabbatch{{1}}.spm.stats.factorial_design.des.fd.fact(2).name = 'Time';
matlabbatch{{1}}.spm.stats.factorial_design.des.fd.fact(2).levels = {len(design['sessions'])};
matlabbatch{{1}}.spm.stats.factorial_design.des.fd.fact(2).dept = 1;
matlabbatch{{1}}.spm.stats.factorial_design.des.fd.fact(2).variance = 1;
matlabbatch{{1}}.spm.stats.factorial_design.des.fd.fact(2).gmsca = 0;
matlabbatch{{1}}.spm.stats.factorial_design.des.fd.fact(2).ancova = 0;

% Input cells
{chr(10).join(cell_code) if cell_code else "% [cells code]"}

% Covariates
{chr(10).join(cov_code) if cov_code else "% [no covariates]"}

% Masking
matlabbatch{{1}}.spm.stats.factorial_design.masking.tm.tm_none = 1;
matlabbatch{{1}}.spm.stats.factorial_design.masking.im = 1;
matlabbatch{{1}}.spm.stats.factorial_design.globalc.g_omit = 1;
matlabbatch{{1}}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
matlabbatch{{1}}.spm.stats.factorial_design.globalm.glonorm = 1;

% ================================================================
% JOB 2: fMRI Model Estimation  
% ================================================================
matlabbatch{{2}}.spm.stats.fmri_est.spmmat(1) = {{'[OUTPUT_DIR]/SPM.mat'}};
matlabbatch{{2}}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{{2}}.spm.stats.fmri_est.method.Classical = 1;
"""
    
    # Write to file
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(batch_content)
    
    print(f"✓ Complete batch file generated: {output_file}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: create_complete_batch.py <design.json> <output_dir> <modality> <output.m>")
        sys.exit(1)
    
    sys.exit(create_complete_batch(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]))
