#!/usr/bin/env python3
"""
Script to create separate surface-based batch files for different CAT12 surface parameters.

Creates 4 batch files:
- template_depth_surface_job.m
- template_fractaldimension_surface_job.m
- template_gyrification_surface_job.m
- template_thickness_surface_job.m

Each uses s12.mesh.<parameter>.resampled_32k format.
"""

import re
from pathlib import Path

# Define surface parameters
SURFACE_PARAMS = {
    'depth': {
        'pattern': 'depth',
        'output_dir': 'surf_depth_int_control',
        'description': 'Sulcal depth analysis'
    },
    'fractaldimension': {
        'pattern': 'fractaldimension',
        'output_dir': 'surf_fractal_int_control',
        'description': 'Fractal dimension analysis'
    },
    'gyrification': {
        'pattern': 'gyrification',
        'output_dir': 'surf_gyrification_int_control',
        'description': 'Gyrification index analysis'
    },
    'thickness': {
        'pattern': 'thickness',
        'output_dir': 'surf_thickness_int_control',
        'description': 'Cortical thickness analysis'
    }
}

def create_surface_batch(template_file, param_name, param_info, output_file):
    """Create a surface batch file for a specific parameter"""
    
    with open(template_file, 'r') as f:
        content = f.read()
    
    print(f"\nCreating batch for {param_name}...")
    
    # Replace the surface parameter in file paths
    # From: s15.mesh.thickness.resampled_32k
    # To: s12.mesh.{param}.resampled_32k
    
    pattern = r's15\.mesh\.thickness\.resampled_32k'
    replacement = f"s12.mesh.{param_info['pattern']}.resampled_32k"
    
    content = re.sub(pattern, replacement, content)
    
    # Update output directory
    content = re.sub(
        r"'/Volumes/Thunder/129_PK01/cat12/stats/surf_int_control'",
        f"'/Volumes/Thunder/129_PK01/cat12/stats/{param_info['output_dir']}'",
        content
    )
    
    # Write the new file
    with open(output_file, 'w') as f:
        f.write(content)
    
    # Count files
    gii_files = re.findall(r's12\.mesh\.' + param_info['pattern'] + r'\.resampled_32k', content)
    
    print(f"  ✓ Created: {output_file.name}")
    print(f"  ✓ Parameter: {param_name}")
    print(f"  ✓ Files: {len(gii_files)}")
    print(f"  ✓ Output dir: {param_info['output_dir']}/")
    
    return len(gii_files)

def verify_batch(batch_file, param_name, param_info):
    """Verify the created batch file"""
    
    with open(batch_file, 'r') as f:
        content = f.read()
    
    # Check for correct parameter
    param_pattern = f"s12.mesh.{param_info['pattern']}.resampled_32k"
    param_files = re.findall(param_pattern, content)
    
    # Check for wrong parameters (should be 0)
    wrong_params = []
    for other_param in SURFACE_PARAMS.keys():
        if other_param != param_name:
            wrong = re.findall(f"mesh.{SURFACE_PARAMS[other_param]['pattern']}", content)
            if wrong:
                wrong_params.extend(wrong)
    
    # Check output directory
    has_correct_dir = param_info['output_dir'] in content
    
    # Check for contrasts
    contrast_count = len(re.findall(r'consess\{\d+\}', content))
    
    return {
        'correct_param_count': len(param_files),
        'wrong_param_count': len(wrong_params),
        'correct_dir': has_correct_dir,
        'contrast_count': contrast_count
    }

def main():
    stats_dir = Path('/Volumes/Thunder/129_PK01/cat12/stats')
    template_file = stats_dir / 'template_surface_job.m'
    
    if not template_file.exists():
        print(f"Error: {template_file} not found!")
        return
    
    print("=" * 80)
    print("Creating Surface Parameter Batch Files")
    print("=" * 80)
    print(f"\nSource: {template_file.name}")
    print(f"Parameters: {len(SURFACE_PARAMS)}")
    
    created_files = []
    
    # Create batch file for each parameter
    for param_name, param_info in SURFACE_PARAMS.items():
        output_file = stats_dir / f'template_{param_name}_surface_job.m'
        
        num_files = create_surface_batch(
            template_file, 
            param_name, 
            param_info, 
            output_file
        )
        
        created_files.append({
            'name': param_name,
            'file': output_file,
            'info': param_info,
            'num_files': num_files
        })
    
    # Verify all files
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    all_good = True
    for file_info in created_files:
        verification = verify_batch(file_info['file'], file_info['name'], file_info['info'])
        
        status = "✓" if (verification['correct_param_count'] == 369 and 
                        verification['wrong_param_count'] == 0 and
                        verification['correct_dir'] and
                        verification['contrast_count'] == 35) else "✗"
        
        print(f"\n{status} {file_info['name'].upper()}")
        print(f"  File: {file_info['file'].name}")
        print(f"  Correct parameters: {verification['correct_param_count']}/369")
        print(f"  Wrong parameters: {verification['wrong_param_count']}")
        print(f"  Correct output dir: {verification['correct_dir']}")
        print(f"  Contrasts: {verification['contrast_count']}/35")
        print(f"  Output: {file_info['info']['output_dir']}/")
        
        if verification['wrong_param_count'] > 0 or verification['correct_param_count'] != 369:
            all_good = False
    
    # Summary table
    print("\n" + "=" * 80)
    print("CREATED BATCH FILES")
    print("=" * 80)
    print(f"\n{'Parameter':<20} {'File':<40} {'Output Directory':<30}")
    print("-" * 90)
    
    for file_info in created_files:
        print(f"{file_info['name']:<20} {file_info['file'].name:<40} {file_info['info']['output_dir']:<30}")
    
    print("\n" + "=" * 80)
    print("SURFACE PARAMETERS")
    print("=" * 80)
    print("\n1. DEPTH (Sulcal Depth)")
    print("   - Measures depth of cortical sulci")
    print("   - Pattern: s12.mesh.depth.resampled_32k")
    
    print("\n2. FRACTAL DIMENSION")
    print("   - Measures cortical complexity/folding patterns")
    print("   - Pattern: s12.mesh.fractaldimension.resampled_32k")
    
    print("\n3. GYRIFICATION")
    print("   - Measures degree of cortical folding")
    print("   - Pattern: s12.mesh.gyrification.resampled_32k")
    
    print("\n4. THICKNESS")
    print("   - Measures cortical thickness")
    print("   - Pattern: s12.mesh.thickness.resampled_32k")
    
    print("\n" + "=" * 80)
    print("USAGE")
    print("=" * 80)
    print("\nIn MATLAB/SPM, load any of the batch files:")
    print("  load('template_depth_surface_job.m')")
    print("  load('template_fractaldimension_surface_job.m')")
    print("  load('template_gyrification_surface_job.m')")
    print("  load('template_thickness_surface_job.m')")
    print("\nThen run:")
    print("  spm_jobman('run', matlabbatch);")
    
    print("\n" + "=" * 80)
    if all_good:
        print("✓ ALL BATCH FILES CREATED SUCCESSFULLY")
    else:
        print("⚠ WARNING: Some batch files may have issues - check verification above")
    print("=" * 80)

if __name__ == '__main__':
    main()
