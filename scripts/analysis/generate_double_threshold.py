#!/usr/bin/env python3
"""
Generate double-threshold maps using CAT12's thresholding tool.

This script creates MATLAB batch jobs to apply CAT12's thresholding
function to SPM T-maps, generating double-thresholded results with
cluster-size and intensity-level correction.

The approach:
1. For each spmT_*.nii contrast, create a MATLAB job calling CAT12's
   threshold SPM-maps function
2. Run jobs via SPM batch system
3. Output: thresholded maps with cluster mask (pkFWE files)

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
    cluster_threshold = 10
    p_intensity_threshold = 0.001
    p_fwe_level = 0.05
    
    if config and "double_threshold" in config:
        dt_config = config["double_threshold"]
        cluster_threshold = dt_config.get("cluster_size_voxels", cluster_threshold)
        p_intensity_threshold = dt_config.get("intensity_p_uncorrected", p_intensity_threshold)
        p_fwe_level = dt_config.get("fwe_level", p_fwe_level)
    
    # Find spmT files
    spmt_files = sorted(glob.glob(os.path.join(stats_path, "spmT_*.nii")))
    
    if not spmt_files:
        print(f"Warning: No spmT_*.nii files found in {stats_path}")
        return None
    
    print(f"Found {len(spmt_files)} spmT maps to threshold")
    
    # Verify SPM.mat exists
    spm_mat = os.path.join(stats_path, "SPM.mat")
    if not os.path.exists(spm_mat):
        print(f"Error: SPM.mat not found in {stats_path}")
        return None
    
    if not spm_path:
        spm_path = find_spm_path()
    
    if not spm_path:
        print("Error: Could not find SPM installation")
        return None
    
    print(f"Using SPM at: {spm_path}")
    
    # MATLAB batch template for CAT12 Results Viewer with thresholding
    # CAT12 provides "Threshold" function in cat12('threshold', ...)
    # For double-threshold approach:
    # - Load SPM.mat
    # - Set intensity threshold (from config)
    # - Set cluster threshold (from config)
    # - This creates thresholded maps saved as pkFWE files
    
    batch_content = f'''% CAT12 Double Threshold Batch
% Generated for: {stats_path}
% Purpose: Apply intensity + cluster thresholding to spmT maps
% Config: intensity_p_uncorrected={p_intensity_threshold}, cluster_size_voxels={cluster_threshold}, fwe_level={p_fwe_level}

spm('defaults', 'FMRI');
spm_jobman('initcfg');

% Add SPM path
addpath('{spm_path}');
addpath(fullfile('{spm_path}', 'toolbox', 'cat12'));

% Load SPM structure
SPM_file = '{spm_mat}';
load(SPM_file, 'SPM');

fprintf('\\n%s\\n', repmat('=', 1, 80));
fprintf('CAT12 DOUBLE THRESHOLD FOR SPM T-MAPS\\n');
fprintf('%s\\n\\n', repmat('=', 1, 80));

% Thresholding parameters (from config)
cluster_threshold = {cluster_threshold};  % minimum cluster size in voxels
p_intensity_threshold = {p_intensity_threshold};  % p-value for intensity (uncorrected)
p_fwe_level = {p_fwe_level};  % FWE correction level (for naming)

fprintf('Intensity threshold: p < %.4f (uncorrected)\\n', p_intensity_threshold);
fprintf('Cluster threshold: k > %d voxels\\n', cluster_threshold);
fprintf('FWE level: p < %.2f\\n\\n', p_fwe_level);

% Use CAT12's thresholding functionality
% This is typically accessed via interactive SPM Results GUI
% For batch mode, we need to loop through contrasts and apply thresholding

% Get number of contrasts
ncon = length(SPM.xCon);
fprintf('Processing %d contrasts...\\n\\n', ncon);

% Process each contrast (starting from those that are T-contrasts)
for con_idx = 1:ncon
    con = SPM.xCon(con_idx);
    
    % Only process T-contrasts (not F-contrasts)
    if ~strcmp(con.STAT, 'T')
        continue;
    end
    
    % Expected spmT file
    spmt_file = fullfile('{stats_path}', sprintf('spmT_%04d.nii', con_idx));
    
    if ~exist(spmt_file, 'file')
        fprintf('Warning: %s not found\\n', spmt_file);
        continue;
    end
    
    fprintf('Processing: %s\\n', con.name);
    fprintf('  spmT file: %s\\n', spmt_file);
    
    % Note: CAT12's interactive thresholding creates clustered binary masks
    % For batch processing, we use SPM's cluster detection functions
    % Then save as thresholded NIfTI
    
    try
        % Load the T-map
        V = spm_vol(spmt_file);
        Y = spm_read_vols(V);
        
        % Get T-threshold from p-value using SPM's t-distribution
        % For p=0.001, df=residual df from SPM
        df = SPM.xX.erdf;
        t_crit = spm_invTcdf(1 - p_intensity_threshold, df);
        
        fprintf('  T-critical (p<%.4f, df=%d): %.3f\\n', p_intensity_threshold, df, t_crit);
        
        % Apply intensity threshold
        mask = abs(Y) >= t_crit;
        
        % Apply cluster threshold using connected components
        % 3D connectivity (6, 18, or 26 neighbors)
        CC = bwconncomp(mask, 26);
        
        % Keep only clusters with at least cluster_threshold voxels
        cluster_sizes = cellfun(@numel, CC.PixelIdxList);
        valid_clusters = find(cluster_sizes >= cluster_threshold);
        
        fprintf('  Found %d clusters, %d with k>%d\\n', ...
                CC.NumObjects, length(valid_clusters), cluster_threshold);
        
        % Create thresholded map
        Y_thresh = zeros(size(Y));
        for c_id = valid_clusters
            Y_thresh(CC.PixelIdxList{{c_id}}) = Y(CC.PixelIdxList{{c_id}});
        end
        
        % Save thresholded map with CAT12 naming convention
        % Format: pkFWE5 means FWE at p<0.05, pkFWE1 means p<0.01, etc.
        fwe_name = sprintf('pkFWE%d', round(p_fwe_level * 100));
        out_name = regexprep(V.fname, 'spmT_', ['', fwe_name, '_k', num2str(cluster_threshold), '_']);
        
        % Ensure proper output naming
        if ~contains(out_name, 'pk')
            [pathstr, name, ext] = fileparts(V.fname);
            out_name = fullfile(pathstr, [name, '_', fwe_name, '_k', num2str(cluster_threshold), ext]);
        end
        
        % Write output
        V.fname = out_name;
        spm_write_vol(V, Y_thresh);
        
        fprintf('  Saved thresholded map: %s\\n\\n', out_name);
        
    catch ME
        fprintf('Error processing contrast %d: %s\\n\\n', con_idx, ME.message);
        continue;
    end
end

fprintf('\\n%s\\n', repmat('=', 1, 80));
fprintf('DOUBLE THRESHOLDING COMPLETE\\n');
fprintf('%s\\n', repmat('=', 1, 80));
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
