#!/usr/bin/env python3
"""
Generate double-threshold maps using CAT12's T2x thresholding tool.

This script creates MATLAB batch jobs to apply CAT12's thresholding
function to SPM T-maps, generating double-thresholded results with
uncorrected voxel thresholding and cluster-level FWE correction.

The approach:
1. Collect spmT_*.nii maps
2. Build a CAT12 T2x batch (uncorrected p, cluster FWE)
3. Run via SPM batch system
4. Output: CAT12 thresholded maps (pkFWE files)

Usage:
    python generate_double_threshold.py <stats_folder>
    python generate_double_threshold.py /path/to/vbm_9mm_3G_2TP_tiv_sex_age
"""

import os
import sys
import glob
import subprocess
import json
from pathlib import Path
import argparse


def find_spm_path():
    """Auto-detect SPM installation from config or environment."""
    # Check config first
    script_dir = Path(__file__).parent.parent.parent
    config_paths = [
        script_dir / "config" / "config.json",
        script_dir / "config" / "config_14_2_26.json",
    ]
    
    for cfg_path in config_paths:
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    config = json.load(f)
                    if "spm" in config and "path" in config["spm"]:
                        spm_path = config["spm"]["path"]
                        if os.path.exists(spm_path):
                            return spm_path
            except:
                pass
    
    # Check environment variable
    if "SPM_PATH" in os.environ:
        return os.environ["SPM_PATH"]
    
    # Common locations on macOS/Linux
    common_paths = [
        "/Volumes/Evo/software/spm25",
        "/Volumes/Evo/software/spm12",
        "/usr/local/spm12",
        "/opt/spm12",
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return None


def generate_thresholding_batch(stats_folder, spm_path, config=None):
    """
    Generate MATLAB batch file for CAT12 thresholding.
    
    Uses CAT12's Threshold module to create double-threshold maps from spmT files.
    This applies both an intensity threshold and cluster threshold for multiple
    comparison correction.
    """
    stats_path = os.path.abspath(stats_folder)
    
    # Get double threshold parameters from config or use defaults
    p_intensity_threshold = 0.001
    p_fwe_level = 0.05
    spm_defaults = "FMRI"
    
    if config and "double_threshold" in config:
        dt_config = config["double_threshold"]
        p_intensity_threshold = dt_config.get("intensity_p_uncorrected", p_intensity_threshold)
        p_fwe_level = dt_config.get("fwe_level", p_fwe_level)
        spm_defaults = dt_config.get("spm_defaults", spm_defaults)
    
    # Find spmT files
    spmt_files = sorted(glob.glob(os.path.join(stats_path, "spmT_*.nii")))
    
    if not spmt_files:
        print(f"Warning: No spmT_*.nii files found in {stats_path}")
        return None
    
    print(f"Found {len(spmt_files)} spmT maps to threshold")
    
    if not spm_path:
        spm_path = find_spm_path()
    
    if not spm_path:
        print("Error: Could not find SPM installation")
        return None
    
    print(f"Using SPM at: {spm_path}")
    
    # MATLAB batch template using CAT12 T2x (matches saved job)
    # For double-threshold approach:
    # - Uncorrected voxel threshold (p < 0.001)
    # - Cluster-level FWE (p < 0.05)
    spmt_list = "\n".join([f"                                                   '{p},1'" for p in spmt_files])
    uncorrected_key = "thresh001" if abs(p_intensity_threshold - 0.001) < 1e-9 else "thresh001"
    fwe_key = "thresh05" if abs(p_fwe_level - 0.05) < 1e-9 else "thresh05"

    if abs(p_intensity_threshold - 0.001) >= 1e-9:
        print("Warning: CAT12 batch expects uncorrected 0.001 (thresh001). Using thresh001 field with configured value.")
    if abs(p_fwe_level - 0.05) >= 1e-9:
        print("Warning: CAT12 batch expects FWE 0.05 (thresh05). Using thresh05 field with configured value.")

    batch_content = f'''% CAT12 Double Threshold Batch
% Generated for: {stats_path}
% Purpose: Apply uncorrected voxel threshold and cluster-level FWE
% Config: intensity_p_uncorrected={p_intensity_threshold}, fwe_level={p_fwe_level}, spm_defaults={spm_defaults}

spm('defaults', '{spm_defaults}');
spm_jobman('initcfg');

% Add SPM path
addpath('{spm_path}');
addpath(fullfile('{spm_path}', 'toolbox', 'cat12'));

matlabbatch{{1}}.spm.tools.cat.tools.T2x.data_T2x = {{
{spmt_list}
                                                   }};
matlabbatch{{1}}.spm.tools.cat.tools.T2x.conversion.sel = 2;
matlabbatch{{1}}.spm.tools.cat.tools.T2x.conversion.threshdesc.uncorr.{uncorrected_key} = {p_intensity_threshold};
matlabbatch{{1}}.spm.tools.cat.tools.T2x.conversion.inverse = 0;
matlabbatch{{1}}.spm.tools.cat.tools.T2x.conversion.cluster.fwe2.{fwe_key} = {p_fwe_level};
matlabbatch{{1}}.spm.tools.cat.tools.T2x.conversion.cluster.fwe2.noniso = 1;
matlabbatch{{1}}.spm.tools.cat.tools.T2x.atlas = 'None';

spm_jobman('run', matlabbatch);
'''
    
    # Write batch file
    batch_file = os.path.join(stats_path, "cat12_double_threshold_batch.m")
    with open(batch_file, "w") as f:
        f.write(batch_content)
    
    print(f"Generated batch file: {batch_file}")
    return batch_file


def run_matlab_batch(batch_file, matlab_exe=None):
    """Run MATLAB batch file."""
    if not matlab_exe:
        # Try to find MATLAB
        result = subprocess.run(["which", "matlab"], capture_output=True, text=True)
        if result.returncode == 0:
            matlab_exe = result.stdout.strip()
        else:
            # Common macOS path
            matlab_exe = "/Applications/MATLAB_R2025b.app/bin/matlab"
    
    if not os.path.exists(matlab_exe):
        print(f"Error: MATLAB not found at {matlab_exe}")
        return False
    
    print(f"\nRunning MATLAB with batch file...")
    print(f"MATLAB: {matlab_exe}")
    print(f"Batch: {batch_file}")
    
    # Run MATLAB in batch mode
    cmd = [
        matlab_exe,
        "-nodesktop",
        "-nosplash",
        "-r",
        f"run('{batch_file}'); quit",
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running MATLAB: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate CAT12 double-threshold maps from spmT files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_double_threshold.py ./results/vbm/vbm_9mm_3G_2TP_tiv_sex_age
    python generate_double_threshold.py ./results/vbm/vbm_9mm_3G_2TP_tiv_sex_age --run
    python generate_double_threshold.py ./results/vbm/vbm_9mm_3G_2TP_tiv_sex_age --matlab /Applications/MATLAB_R2025b.app/bin/matlab --run
        """,
    )
    
    parser.add_argument(
        "stats_folder",
        help="Path to SPM stats folder containing SPM.mat and spmT_*.nii files",
    )
    parser.add_argument(
        "--spm-path",
        default=None,
        help="Path to SPM installation (auto-detected if not specified)",
    )
    parser.add_argument(
        "--matlab",
        default=None,
        help="Path to MATLAB executable (auto-detected if not specified)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run MATLAB job immediately (default: just generate batch file)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.json file (default: config/config_14_2_26.json from workspace root)",
    )
    
    args = parser.parse_args()
    
    # Load config if available
    config = None
    config_path = args.config
    if not config_path:
        # Try to find config in default locations
        script_dir = Path(__file__).parent.parent.parent
        default_paths = [
            script_dir / "config" / "config.json",
            script_dir / "config" / "config_14_2_26.json",
        ]
        for cfg_path in default_paths:
            if cfg_path.exists():
                config_path = str(cfg_path)
                break
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path) as f:
                config = json.load(f)
            print(f"Loaded config from: {config_path}")
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
    
    batch_file = generate_thresholding_batch(args.stats_folder, args.spm_path, config)
    
    if batch_file and args.run:
        success = run_matlab_batch(batch_file, args.matlab)
        if success:
            print("\n✓ Double thresholding completed successfully!")
            print("You can now regenerate the HTML report to include the thresholded maps.")
        else:
            print("\n✗ MATLAB batch execution failed")
            sys.exit(1)
    elif batch_file:
        print(f"\n✓ Batch file generated: {batch_file}")
        print("To run it, execute:")
        print(f"  matlab -nodesktop -nosplash -r \"run('{batch_file}'); quit\"")
        print("\nOr use the --run flag to execute automatically:")
        print(f"  python {__file__} {args.stats_folder} --run")


if __name__ == "__main__":
    main()
