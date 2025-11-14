#!/usr/bin/env python3
"""
Script to add TIV and IQR covariates to the batch_3x3_job.m file
in the correct order matching the file list.
"""

import re
from pathlib import Path

def extract_files_from_batch(batch_file):
    """Extract all .nii files from batch_3x3_job.m in order"""
    with open(batch_file, 'r') as f:
        content = f.read()
    
    # Find all .nii file paths
    pattern = r"'(/Volumes/Thunder/129_PK01/cat12/s9/.*?\.nii),1'"
    files = re.findall(pattern, content)
    
    return files

def extract_files_from_iqr(iqr_file):
    """Extract all XML files from IQR_job.m in order"""
    with open(iqr_file, 'r') as f:
        content = f.read()
    
    # Find all XML file paths
    pattern = r"'(/Volumes/Thunder/129_PK01/cat12/data/cat12/.*?\.xml)'"
    files = re.findall(pattern, content)
    
    return files

def parse_filename(filepath):
    """Extract subject ID and session from filepath"""
    # For .nii files: extract sub-XXXXXX and ses-X
    # For .xml files: extract sub-XXXXXX and ses-X
    sub_match = re.search(r'sub-(\d+)', filepath)
    ses_match = re.search(r'ses-(\d+)', filepath)
    
    if sub_match and ses_match:
        return (sub_match.group(1), ses_match.group(1))
    return None

def read_values(file_path):
    """Read numeric values from text file"""
    values = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    values.append(float(line))
                except ValueError:
                    continue
    return values

def create_mapping(iqr_files, batch_files):
    """Create mapping from IQR file order to batch file order"""
    # Create lookup dictionary: (subject, session) -> index in IQR list
    iqr_lookup = {}
    for idx, xml_file in enumerate(iqr_files):
        key = parse_filename(xml_file)
        if key:
            iqr_lookup[key] = idx
    
    # Create ordered list of indices for batch files
    batch_indices = []
    missing_files = []
    
    for nii_file in batch_files:
        key = parse_filename(nii_file)
        if key and key in iqr_lookup:
            batch_indices.append(iqr_lookup[key])
        else:
            print(f"Warning: No IQR/TIV data found for {nii_file}")
            missing_files.append(nii_file)
            batch_indices.append(None)
    
    return batch_indices, missing_files

def reorder_values(values, indices):
    """Reorder values according to indices mapping"""
    reordered = []
    for idx in indices:
        if idx is not None:
            reordered.append(values[idx])
        else:
            reordered.append(float('nan'))  # Use NaN for missing values
    return reordered

def format_covariate_line(name, values):
    """Format covariate data for MATLAB batch file"""
    lines = []
    lines.append(f"matlabbatch{{1}}.spm.tools.cat.factorial_design.cov(1).c = [")
    for val in values:
        lines.append(f"                                                              {val:.2f}")
    lines.append("                                                              ];")
    lines.append(f"matlabbatch{{1}}.spm.tools.cat.factorial_design.cov(1).cname = '{name}';")
    lines.append("matlabbatch{1}.spm.tools.cat.factorial_design.cov(1).iCFI = 1;")
    lines.append("matlabbatch{1}.spm.tools.cat.factorial_design.cov(1).iCC = 1;")
    return '\n'.join(lines)

def add_covariates_to_batch(batch_file, tiv_values, iqr_values, output_file):
    """Add TIV and IQR covariates to the batch file"""
    with open(batch_file, 'r') as f:
        content = f.read()
    
    # Find the line with empty cov struct
    old_cov_line = "matlabbatch{1}.spm.tools.cat.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});"
    
    # Create new covariate lines for both TIV and IQR
    new_cov_lines = []
    
    # TIV covariate
    new_cov_lines.append("matlabbatch{1}.spm.tools.cat.factorial_design.cov(1).c = [")
    for val in tiv_values:
        new_cov_lines.append(f"                                                              {val:.2f}")
    new_cov_lines.append("                                                              ];")
    new_cov_lines.append("matlabbatch{1}.spm.tools.cat.factorial_design.cov(1).cname = 'TIV';")
    new_cov_lines.append("matlabbatch{1}.spm.tools.cat.factorial_design.cov(1).iCFI = 1;")
    new_cov_lines.append("matlabbatch{1}.spm.tools.cat.factorial_design.cov(1).iCC = 1;")
    
    # IQR covariate
    new_cov_lines.append("matlabbatch{1}.spm.tools.cat.factorial_design.cov(2).c = [")
    for val in iqr_values:
        new_cov_lines.append(f"                                                              {val:.6f}")
    new_cov_lines.append("                                                              ];")
    new_cov_lines.append("matlabbatch{1}.spm.tools.cat.factorial_design.cov(2).cname = 'IQR';")
    new_cov_lines.append("matlabbatch{1}.spm.tools.cat.factorial_design.cov(2).iCFI = 1;")
    new_cov_lines.append("matlabbatch{1}.spm.tools.cat.factorial_design.cov(2).iCC = 1;")
    
    new_cov_section = '\n'.join(new_cov_lines)
    
    # Replace the old line with new covariate section
    new_content = content.replace(old_cov_line, new_cov_section)
    
    # Write to output file
    with open(output_file, 'w') as f:
        f.write(new_content)
    
    return new_content

def main():
    # File paths
    stats_dir = Path('/Volumes/Thunder/129_PK01/cat12/stats')
    batch_file = stats_dir / 'batch_3x3_job.m'
    iqr_job_file = stats_dir / 'IQR_job.m'
    tiv_file = stats_dir / 'TIV.txt'
    iqr_file = stats_dir / 'IQR.txt'
    output_file = stats_dir / 'batch_3x3_job_with_covariates.m'
    
    print("Extracting file lists...")
    batch_files = extract_files_from_batch(batch_file)
    iqr_files = extract_files_from_iqr(iqr_job_file)
    
    print(f"Found {len(batch_files)} files in batch_3x3_job.m")
    print(f"Found {len(iqr_files)} files in IQR_job.m")
    
    print("\nReading TIV and IQR values...")
    tiv_values = read_values(tiv_file)
    iqr_values = read_values(iqr_file)
    
    print(f"Found {len(tiv_values)} TIV values")
    print(f"Found {len(iqr_values)} IQR values")
    
    print("\nCreating file mapping...")
    batch_indices, missing_files = create_mapping(iqr_files, batch_files)
    
    if missing_files:
        print(f"\nWarning: {len(missing_files)} files in batch without IQR/TIV data")
        for f in missing_files[:5]:  # Show first 5
            print(f"  - {f}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")
    
    print("\nReordering values to match batch file order...")
    reordered_tiv = reorder_values(tiv_values, batch_indices)
    reordered_iqr = reorder_values(iqr_values, batch_indices)
    
    print(f"\nReordered {len(reordered_tiv)} TIV values")
    print(f"Reordered {len(reordered_iqr)} IQR values")
    
    print("\nAdding covariates to batch file...")
    add_covariates_to_batch(batch_file, reordered_tiv, reordered_iqr, output_file)
    
    print(f"\n✓ Successfully created {output_file}")
    print("\nSummary:")
    print(f"  - Total files: {len(batch_files)}")
    print(f"  - Files with TIV/IQR: {len([i for i in batch_indices if i is not None])}")
    print(f"  - Files missing TIV/IQR: {len([i for i in batch_indices if i is None])}")
    
    # Show first few matches as verification
    print("\nFirst 5 file matches (verification):")
    for i in range(min(5, len(batch_files))):
        batch_key = parse_filename(batch_files[i])
        if batch_indices[i] is not None:
            iqr_key = parse_filename(iqr_files[batch_indices[i]])
            tiv = reordered_tiv[i]
            iqr = reordered_iqr[i]
            print(f"  {i+1}. sub-{batch_key[0]} ses-{batch_key[1]}: TIV={tiv:.2f}, IQR={iqr:.6f}")

if __name__ == '__main__':
    main()
