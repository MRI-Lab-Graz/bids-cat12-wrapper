#!/usr/bin/env python3
"""
Script to convert template_job.m (volume-based) to surface-based
WITHOUT covariates and WITHOUT threshold.

Modifications:
1. Convert .nii paths to .gii paths
2. Remove TIV and IQR covariates
3. Remove threshold (tm.tma.athresh) and use tm.tm_none
4. Keep main effects, interactions, and all contrasts
"""

import re
from pathlib import Path

def convert_volume_to_surface_path(volume_path):
    """Convert a volume-based .nii path to a surface-based .gii path"""
    match = re.search(r'(sub-\d+)_(ses-\d+)\.nii', volume_path)
    
    if not match:
        print(f"Warning: Could not parse {volume_path}")
        return volume_path
    
    subject = match.group(1)  # e.g., sub-1291145
    session = match.group(2)  # e.g., ses-1
    
    # Build the new surface path
    base_path = '/Volumes/Thunder/129_PK01/cat12/data/cat12'
    surface_filename = f's15.mesh.thickness.resampled_32k.r{subject}_{session}_acq-mprage_T1w.gii'
    
    new_path = f"{base_path}/{subject}/surf/{surface_filename}"
    
    return new_path

def convert_template_to_surface(input_file, output_file):
    """Convert template batch file to surface-based without covariates"""
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    print("Converting template_job.m to surface-based...")
    
    # 1. Convert all .nii file paths to .gii
    pattern = r"'(/Volumes/Thunder/129_PK01/cat12/s9/[^']+\.nii),1'"
    
    def replace_path(match):
        volume_path = match.group(1)
        surface_path = convert_volume_to_surface_path(volume_path)
        return f"'{surface_path}'"
    
    matches = re.findall(pattern, content)
    print(f"Found {len(matches)} volume-based file paths to convert")
    
    content = re.sub(pattern, replace_path, content)
    
    # 2. Update output directory
    content = content.replace(
        "'/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control_cov'",
        "'/Volumes/Thunder/129_PK01/cat12/stats/surf_int_control'"
    )
    
    # 3. Remove covariates section (TIV and IQR)
    # Find the entire cov section and replace with empty struct
    cov_pattern = r'matlabbatch\{1\}\.spm\.tools\.cat\.factorial_design\.cov\(1\)\.c = \[.*?\];.*?'
    cov_pattern += r'matlabbatch\{1\}\.spm\.tools\.cat\.factorial_design\.cov\(1\)\.cname = \'TIV\';.*?'
    cov_pattern += r'matlabbatch\{1\}\.spm\.tools\.cat\.factorial_design\.cov\(1\)\.iCFI = \d+;.*?'
    cov_pattern += r'matlabbatch\{1\}\.spm\.tools\.cat\.factorial_design\.cov\(1\)\.iCC = \d+;.*?'
    cov_pattern += r'%%.*?'
    cov_pattern += r'matlabbatch\{1\}\.spm\.tools\.cat\.factorial_design\.cov\(2\)\.c = \[.*?\];.*?'
    cov_pattern += r'matlabbatch\{1\}\.spm\.tools\.cat\.factorial_design\.cov\(2\)\.cname = \'IQR\';.*?'
    cov_pattern += r'matlabbatch\{1\}\.spm\.tools\.cat\.factorial_design\.cov\(2\)\.iCFI = \d+;.*?'
    cov_pattern += r'matlabbatch\{1\}\.spm\.tools\.cat\.factorial_design\.cov\(2\)\.iCC = \d+;'
    
    content = re.sub(cov_pattern, 
                    "matlabbatch{1}.spm.tools.cat.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});",
                    content, flags=re.DOTALL)
    
    # 4. Remove threshold - replace tm.tma.athresh with tm.tm_none
    content = re.sub(
        r'matlabbatch\{1\}\.spm\.tools\.cat\.factorial_design\.masking\.tm\.tma\.athresh = [0-9.]+;',
        'matlabbatch{1}.spm.tools.cat.factorial_design.masking.tm.tm_none = 1;',
        content
    )
    
    # Write the new file
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"\n✓ Successfully created: {output_file}")
    
    return len(matches)

def verify_conversion(batch_file):
    """Verify the conversion"""
    
    with open(batch_file, 'r') as f:
        content = f.read()
    
    # Check for surface files
    gii_files = re.findall(r"'(/Volumes/Thunder/129_PK01/cat12/data/cat12/[^']+\.gii)'", content)
    
    # Check for remaining .nii files
    nii_files = re.findall(r'\.nii', content)
    
    # Check for covariates
    has_tiv = 'TIV' in content
    has_iqr_cov = re.search(r"cov\(\d+\)\.cname = 'IQR'", content)
    
    # Check for threshold
    has_threshold = 'tm.tma.athresh' in content
    has_tm_none = 'tm.tm_none' in content
    
    # Check for main effects and interactions
    has_maininters = 'maininters' in content
    
    # Count contrasts
    contrast_count = len(re.findall(r'consess\{\d+\}', content))
    
    print("\n" + "=" * 80)
    print("VERIFICATION REPORT")
    print("=" * 80)
    print(f"\n✓ Surface files (.gii): {len(gii_files)}")
    print(f"{'✓' if not nii_files else '✗'} Remaining .nii references: {len(nii_files)}")
    print(f"{'✓' if not has_tiv else '✗'} TIV covariate removed: {not has_tiv}")
    print(f"{'✓' if not has_iqr_cov else '✗'} IQR covariate removed: {not has_iqr_cov}")
    print(f"{'✓' if not has_threshold else '✗'} Threshold removed: {not has_threshold}")
    print(f"{'✓' if has_tm_none else '✗'} No threshold (tm_none): {has_tm_none}")
    print(f"{'✓' if has_maininters else '✗'} Main effects/interactions: {has_maininters}")
    print(f"{'✓' if contrast_count > 0 else '✗'} Contrasts defined: {contrast_count}")
    
    if gii_files:
        print("\nFirst 3 surface files:")
        for i, path in enumerate(gii_files[:3], 1):
            filename = path.split('/')[-1]
            subject = path.split('/')[-3]
            print(f"  {i}. {subject}/surf/{filename}")
    
    print("=" * 80)

def main():
    stats_dir = Path('/Volumes/Thunder/129_PK01/cat12/stats')
    input_file = stats_dir / 'template_job.m'
    output_file = stats_dir / 'template_surface_job.m'
    
    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        return
    
    print("=" * 80)
    print("Converting template_job.m to Surface-Based Analysis")
    print("=" * 80)
    print("\nModifications:")
    print("  • Volume (.nii) → Surface (.gii) file paths")
    print("  • Remove TIV and IQR covariates")
    print("  • Remove absolute threshold masking")
    print("  • Keep main effects and interactions")
    print("  • Keep all contrasts")
    print("\n" + "=" * 80)
    
    num_converted = convert_template_to_surface(input_file, output_file)
    print(f"Converted {num_converted} file paths")
    
    verify_conversion(output_file)
    
    print(f"\n✓ Surface-based batch file ready: {output_file.name}")
    print("\nTo use:")
    print("  1. Load template_surface_job.m in SPM/CAT12")
    print("  2. Review the design (main effects, interactions, contrasts)")
    print("  3. Run the batch")
    print("  4. Results will be saved to: surf_int_control/")

if __name__ == '__main__':
    main()
