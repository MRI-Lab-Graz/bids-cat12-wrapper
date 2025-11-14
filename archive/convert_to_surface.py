#!/usr/bin/env python3
"""
Script to convert volume-based batch file to surface-based batch file
for CAT12 flexible factorial design.

Converts:
'/Volumes/Thunder/129_PK01/cat12/s9/2w_group/sub-1291145_ses-1.nii,1'

To:
'/Volumes/Thunder/129_PK01/cat12/data/cat12/sub-1291145/surf/s15.mesh.thickness.resampled_32k.rsub-1291145_ses-1_acq-mprage_T1w.gii'
"""

import re
from pathlib import Path

def convert_volume_to_surface_path(volume_path):
    """
    Convert a volume-based .nii path to a surface-based .gii path
    
    Args:
        volume_path: Path like '/Volumes/Thunder/129_PK01/cat12/s9/2w_group/sub-1291145_ses-1.nii,1'
    
    Returns:
        Surface path like '/Volumes/Thunder/129_PK01/cat12/data/cat12/sub-1291145/surf/s15.mesh.thickness.resampled_32k.rsub-1291145_ses-1_acq-mprage_T1w.gii'
    """
    # Extract subject ID and session from the filename
    # Pattern: sub-XXXXXX_ses-X.nii
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

def convert_batch_file(input_file, output_file):
    """Convert the entire batch file from volume to surface paths"""
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Find all .nii file references
    # Pattern: '/Volumes/Thunder/129_PK01/cat12/s9/.../*.nii,1'
    pattern = r"'(/Volumes/Thunder/129_PK01/cat12/s9/[^']+\.nii),1'"
    
    def replace_path(match):
        volume_path = match.group(1)
        surface_path = convert_volume_to_surface_path(volume_path)
        return f"'{surface_path}'"
    
    # Count replacements
    matches = re.findall(pattern, content)
    print(f"Found {len(matches)} volume-based file paths to convert")
    
    # Replace all volume paths with surface paths
    new_content = re.sub(pattern, replace_path, content)
    
    # Update the output directory name
    # Change from s9_int_control to surf_int_control
    new_content = new_content.replace(
        "'/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control'",
        "'/Volumes/Thunder/129_PK01/cat12/stats/surf_int_control'"
    )
    
    # Write the new file
    with open(output_file, 'w') as f:
        f.write(new_content)
    
    print(f"\n✓ Successfully created surface-based batch file: {output_file}")
    
    return len(matches)

def verify_conversion(batch_file):
    """Verify the conversion by showing sample paths"""
    
    with open(batch_file, 'r') as f:
        content = f.read()
    
    # Find all .gii file references
    pattern = r"'(/Volumes/Thunder/129_PK01/cat12/data/cat12/[^']+\.gii)'"
    gii_files = re.findall(pattern, content)
    
    print("\n" + "=" * 80)
    print("VERIFICATION: Surface-based file paths")
    print("=" * 80)
    print(f"\nTotal surface files: {len(gii_files)}")
    
    if gii_files:
        print("\nFirst 5 surface file paths:")
        for i, path in enumerate(gii_files[:5], 1):
            # Extract just the filename for display
            filename = path.split('/')[-1]
            subject = path.split('/')[-3]
            print(f"  {i}. {subject}/surf/{filename}")
        
        print("\nLast 5 surface file paths:")
        for i, path in enumerate(gii_files[-5:], len(gii_files)-4):
            filename = path.split('/')[-1]
            subject = path.split('/')[-3]
            print(f"  {i}. {subject}/surf/{filename}")
    
    # Check if any .nii files remain
    nii_pattern = r"\.nii"
    remaining_nii = re.findall(nii_pattern, content)
    
    if remaining_nii:
        print(f"\n⚠ Warning: Found {len(remaining_nii)} remaining .nii references")
    else:
        print("\n✓ All .nii files successfully converted to .gii")
    
    print("=" * 80)

def main():
    stats_dir = Path('/Volumes/Thunder/129_PK01/cat12/stats')
    
    # Convert both the original and the one with covariates
    input_files = [
        (stats_dir / 'batch_3x3_job.m', stats_dir / 'batch_3x3_surface_job.m'),
        (stats_dir / 'batch_3x3_job_with_covariates.m', stats_dir / 'batch_3x3_surface_job_with_covariates.m')
    ]
    
    for input_file, output_file in input_files:
        if input_file.exists():
            print("\n" + "=" * 80)
            print(f"Converting: {input_file.name}")
            print("=" * 80)
            
            num_converted = convert_batch_file(input_file, output_file)
            print(f"Converted {num_converted} file paths")
            
            # Verify the conversion
            verify_conversion(output_file)
        else:
            print(f"\nSkipping {input_file.name} (file not found)")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nCreated surface-based batch files:")
    for _, output_file in input_files:
        if output_file.exists():
            print(f"  ✓ {output_file.name}")
    
    print("\nNote: The surface files follow CAT12 naming convention:")
    print("  s15.mesh.thickness.resampled_32k.r<subject>_<session>_acq-mprage_T1w.gii")

if __name__ == '__main__':
    main()
