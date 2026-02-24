#!/usr/bin/env python3
"""
Generate complete SPM batch file for factorial design + model estimation + contrasts.

This extends generate_spm_batch.py by including model estimation and standard
SPM contrast definitions (no custom CAT12 functions needed). The output batch
can run entirely in standalone mode via cat_standalone.sh.

Usage:
  python3 generate_spm_batch_full.py --design-file design.json \\
      --output-dir results/ --modality vbm --output spm_batch_full.m
"""

import argparse
import json
import sys
from pathlib import Path
import unicodedata
import re


def generate_design_section(design, subject_map):
    """Generate the flexible factorial design section."""
    groups = list(design["groups"].keys())
    sessions = design["sessions"]
    
    # Build cells code
    cells_code = []
    cell_idx = 1
    
    for g_idx, group in enumerate(groups, 1):
        for s_idx, session in enumerate(sessions, 1):
            cell_key = (group, session)
            cell_files = []
            
            # Find all files for this cell
            for finfo in design["files"]:
                if finfo.get("group") == group and finfo.get("session") == session:
                    filepath = finfo.get("filepath", "")
                    # Add volume index for VBM (NIfTI); surface data uses GIfTI paths as-is
                    if design["modality"] in ("vbm", "vbm_dartel"):
                        if not filepath.endswith(",1"):
                            filepath = f"{filepath},1"
                    cell_files.append(filepath)
            
            if cell_files:
                cells_code.append(f"% Cell {cell_idx}: {group} × Session {session}")
                cells_code.append(f"matlabbatch{{1}}.spm.stats.factorial_design.des.fd.icell({cell_idx}).scans = {{...\n")
                for fpath in cell_files:
                    safe_fpath = fpath.replace("'", "''")
                    cells_code.append(f"    '{safe_fpath}'\n")
                cells_code.append("    };\n")
                cell_idx += 1
    
    return "\n".join(cells_code)


def generate_covariates_section(design):
    """Generate covariate section."""
    if not design.get("covariates"):
        return "% No covariates"
    
    cov_code = []
    for cov_idx, (cov_name, cov_values) in enumerate(design["covariates"].items(), 1):
        safe_cname = re.sub(r"[^A-Za-z0-9_]+", "_", str(cov_name))[:31]  # MATLAB identifier limit
        cov_code.append(f"% Covariate {cov_idx}: {cov_name}")
        cov_code.append(f"matlabbatch{{1}}.spm.stats.factorial_design.cov({cov_idx}).c = [...")
        for val in cov_values:
            cov_code.append(f"    {val}")
        cov_code.append("];")
        cov_code.append(f"matlabbatch{{1}}.spm.stats.factorial_design.cov({cov_idx}).cname = '{safe_cname}';")
        cov_code.append(f"matlabbatch{{1}}.spm.stats.factorial_design.cov({cov_idx}).iCFI = 1;")
        cov_code.append(f"matlabbatch{{1}}.spm.stats.factorial_design.cov({cov_idx}).iCC = 1;")
    
    return "\n".join(cov_code)


def generate_contrasts_section(design):
    """Generate standard contrasts for flexible factorial design.
    
    Creates contrasts for:
    - Main effects (if multiple groups/sessions)
    - Linear time effects
    - Simple interactions
    """
    groups = list(design["groups"].keys())
    sessions = design["sessions"]
    n_params = len(design["files"])  # Approximation
    
    contrasts = []
    con_idx = 1
    
    # ***** Basic strategy: Create simple contrasts *****
    # For flexibility, we'll create:
    # 1. Overall average (all cells equally weighted)
    # 2. Main effects of group (if >1 group)
    # 3. Main effects of time (if >1 session)
    # 4. Simple interaction effects
    
    # Contrast 1: Overall average
    contrasts.append("% Contrast 1: Overall average (all cells positive)")
    contrasts.append(f"matlabbatch{{2}}.spm.stats.con.spmmat = {{'[OUTPUT_DIR]/SPM.mat'}};")
    contrasts.append(f"matlabbatch{{2}}.spm.stats.con.consess{{{con_idx}}}.tcon.name = 'Overall_mean';")
    contrasts.append(f"matlabbatch{{2}}.spm.stats.con.consess{{{con_idx}}}.tcon.convec = ones(1, {n_params});")
    contrasts.append(f"matlabbatch{{2}}.spm.stats.con.consess{{{con_idx}}}.tcon.convec = matlabbatch{{2}}.spm.stats.con.consess{{{con_idx}}}.tcon.convec / {n_params};  % Normalize")
    contrasts.append(f"matlabbatch{{2}}.spm.stats.con.consess{{{con_idx}}}.tcon.sessreg = '';")
    
    con_idx += 1
    
    if len(groups) > 1:
        # Contrast: Group 1 vs others
        contrasts.append(f"\n% Contrast {con_idx}: Group '{groups[0]}' vs others")
        contrasts.append(f"matlabbatch{{2}}.spm.stats.con.consess{{{con_idx}}}.tcon.name = 'Group_{groups[0]}_vs_others';")
        contrasts.append(f"matlabbatch{{2}}.spm.stats.con.consess{{{con_idx}}}.tcon.convec = [... % Placeholder")
        contrasts.append("    1, 1, 1, -0.5, -0.5, -0.5];  % Example for 2 groups × 3 sessions")
        contrasts.append(f"matlabbatch{{2}}.spm.stats.con.consess{{{con_idx}}}.tcon.sessreg = '';")
        con_idx += 1
    
    if len(sessions) > 1:
        # Contrast: Linear time trend
        contrasts.append(f"\n% Contrast {con_idx}: Linear time trend")
        contrasts.append(f"matlabbatch{{2}}.spm.stats.con.consess{{{con_idx}}}.tcon.name = 'Time_linear_trend';")
        linear_trend = list(range(-.len(sessions)//2, len(sessions)//2 + 1))
        contrasts.append(f"matlabbatch{{2}}.spm.stats.con.consess{{{con_idx}}}.tcon.convec = repmat({linear_trend}, 1, {len(groups)});  % Linear contrast replicated per group")
        contrasts.append(f"matlabbatch{{2}}.spm.stats.con.consess{{{con_idx}}}.tcon.sessreg = '';")
        con_idx += 1
    
    contrasts.append(f"\nmatlabbatch{{2}}.spm.stats.con.delete = 0;  % Don't delete previous contrasts")
    
    return "\n".join(contrasts)


def sanitize_path(path_str):
    """Escape single quotes in path for MATLAB string."""
    return str(path_str).replace("'", "''")


def generate_full_batch(args):
    """Generate complete batch file with design + estimation + contrasts."""
    
    print("Loading design structure...")
    with open(args.design_file, "r") as f:
        design = json.load(f)
    
    subject_map = {str(f.get("subject", "")): i+1 for i, f in enumerate(design["files"])}
    
    safe_output_dir = sanitize_path(args.output_dir)
    
    # Build the complete batch
    batch_lines = [
        "% SPM Batch: Complete Factorial Design Analysis",
        "% Auto-generated by generate_spm_batch_full.py",
        "% Includes: Design + Model Estimation + Contrasts (no custom functions)",
        "%",
        f"% Modality: {design['modality']}",
        f"% Smoothing: {design['smoothing']}mm",
        f"% Groups: {len(design['groups'])}",
        f"% Sessions: {len(design['sessions'])}",
        "",
        "% ================================================================",
        "% JOB 1: Factorial Design Specification",
        "% ================================================================",
        "",
        f"matlabbatch{{1}}.spm.stats.factorial_design.dir = {{'{safe_output_dir}'}};",
        "",
        "% Factors",
        "matlabbatch{1}.spm.stats.factorial_design.des.fd.fact(1).name = 'Group';",
        f"matlabbatch{{1}}.spm.stats.factorial_design.des.fd.fact(1).levels = {len(design['groups'])};",
        "matlabbatch{1}.spm.stats.factorial_design.des.fd.fact(1).dept = 0;  % Independent",
        "matlabbatch{1}.spm.stats.factorial_design.des.fd.fact(1).variance = 1;",
        "matlabbatch{1}.spm.stats.factorial_design.des.fd.fact(1).gmsca = 0;",
        "matlabbatch{1}.spm.stats.factorial_design.des.fd.fact(1).ancova = 0;",
        "",
        "matlabbatch{1}.spm.stats.factorial_design.des.fd.fact(2).name = 'Time';",
        f"matlabbatch{{1}}.spm.stats.factorial_design.des.fd.fact(2).levels = {len(design['sessions'])};",
        "matlabbatch{1}.spm.stats.factorial_design.des.fd.fact(2).dept = 1;  % Dependent (Repeated Measures)",
        "matlabbatch{1}.spm.stats.factorial_design.des.fd.fact(2).variance = 1;",
        "matlabbatch{1}.spm.stats.factorial_design.des.fd.fact(2).gmsca = 0;",
        "matlabbatch{1}.spm.stats.factorial_design.des.fd.fact(2).ancova = 0;",
        "",
        "% Input cells",
        generate_design_section(design, subject_map),
        "",
        "% Covariates",
        generate_covariates_section(design),
        "",
        "% Masking",
        "matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;",
        "matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;",
        "matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;",
        "matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;",
        "matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;",
        "",
        "",
        "% ================================================================",
        "% JOB 2: fMRI Model Estimation",
        "% ================================================================",
        "",
        f"matlabbatch{{2}}.spm.stats.fmri_est.spmmat(1) = {{'[OUTPUT_DIR]/SPM.mat'}};",
        "matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;",
        "matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;",
        "",
        "",
        "% ================================================================",
        "% JOB 3: Contrast Manager (Add Contrasts)",
        "% ================================================================",
        "",
        "% NOTE: These are placeholder contrasts. For custom contrasts,",
        "% run standalone after inspection of the design, or modify this batch.",
        "",
        generate_contrasts_section(design),
        "",
    ]
    
    batch_content = "\n".join(batch_lines)
    
    # Save to output file
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        f.write(batch_content)
    
    print(f"✓ Complete SPM batch file generated: {output_file}")
    print(f"  - Includes factorial design specification")
    print(f"  - Includes fMRI model estimation (Classical/ReML)")
    print(f"  - Includes placeholder contrasts (editable)")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate complete SPM batch (design + estimation + contrasts)"
    )
    parser.add_argument("--design-file", required=True, help="Path to design.json")
    parser.add_argument("--output-dir", required=True, help="SPM output directory")
    parser.add_argument("--modality", required=True, help="Analysis modality (vbm, thickness, etc.)")
    parser.add_argument("--output", required=True, help="Output batch file path")
    parser.add_argument("--mask-file", default=None, help="Optional explicit mask")
    
    args = parser.parse_args()
    sys.exit(generate_full_batch(args))
