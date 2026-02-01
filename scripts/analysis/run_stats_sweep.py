#!/usr/bin/env python3
"""
Automated statistical sweep for CAT12 results.

Generates multiple thresholded maps and effect size maps, then creates
an interactive HTML report.

Features:
- Double-threshold sweep (voxel + cluster level)
- Effect size (Cohen's d) map generation
- Automatic HTML report generation
- Config-based MATLAB/SPM path management

Usage:
    python run_stats_sweep.py ./results/vbm/analysis
    python run_stats_sweep.py ./results/vbm/analysis --use-matlab --matlab-exe /path/to/matlab
    python run_stats_sweep.py ./results/vbm/analysis --force  # Recompute even if exists
"""

import os
import argparse
import subprocess
import sys
import json
from pathlib import Path


def get_config_defaults(config_file=None):
    """Read defaults from config/config.json (or custom study config)"""
    defaults = {
        'matlab_exe': '/Applications/MATLAB_R2025b.app/bin/matlab',
        'spm_path': ''
    }
    
    # Find config file
    workspace_root = Path(__file__).resolve().parents[3]  # Go up to workspace root
    if config_file is None:
        config_file = workspace_root / "config" / "config.json"
    else:
        config_file = Path(config_file)
    
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
            
            if 'matlab' in config and config['matlab'].get('executable'):
                defaults['matlab_exe'] = config['matlab']['executable']
            if 'spm' in config and config['spm'].get('path'):
                defaults['spm_path'] = config['spm']['path']
        except Exception as e:
            print(f"Warning: Could not read config file {config_file}: {e}")
    
    return defaults


def run_matlab_cmd(cmd, args):
    """Run a MATLAB command using the standalone runner or direct MATLAB."""
    workspace_root = Path(__file__).resolve().parents[3]
    utils_dir = workspace_root / "scripts" / "utils"
    
    # Try to use runner if available
    runner = Path(__file__).parent.parent / "utils" / "run_matlab_standalone.py"
    
    if runner.exists() and not args.use_matlab:
        full_cmd = [
            sys.executable, str(runner),
            cmd,
            "--utils", str(utils_dir)
        ]
        
        if args.mcr:
            full_cmd.extend(["--mcr", args.mcr])
        if args.standalone:
            full_cmd.extend(["--standalone", args.standalone])
    else:
        # Use direct MATLAB
        matlab_exe = args.matlab_exe if args.matlab_exe else get_config_defaults()['matlab_exe']
        full_cmd = [
            matlab_exe,
            "-nodisplay", "-nosplash", "-r",
            f"addpath('{utils_dir}'); {cmd}; exit;"
        ]
    
    print(f"Running: {cmd}")
    try:
        subprocess.check_call(full_cmd)
    except subprocess.CalledProcessError as e:
        print(f"Error running MATLAB command: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run a sweep of statistical thresholds for CAT12 results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_stats_sweep.py ./results/vbm/analysis
  python run_stats_sweep.py ./results/vbm/analysis --use-matlab
  python run_stats_sweep.py ./results/vbm/analysis --force
  python run_stats_sweep.py ./results/vbm/analysis --config config/config.study_intervention.json
        """
    )
    parser.add_argument("results_dir",
                        help="Path to the directory containing SPM.mat")
    
    # Config option
    parser.add_argument("--config", help="Path to config.json (default: config/config.json)")
    
    # MATLAB Options
    parser.add_argument("--use-matlab", action="store_true",
                        help="Use local MATLAB instead of standalone MCR")
    parser.add_argument("--matlab-exe", help="Path to MATLAB executable")
    parser.add_argument("--spm-path", help="Path to SPM installation")
    
    # MCR Options
    parser.add_argument("--mcr", help="Path to MCR")
    parser.add_argument("--standalone", help="Path to run_spm12.sh")
    
    # Control Options
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing statistic maps")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip plot generation (faster)")
    
    args = parser.parse_args()
    
    # Load config
    defaults = get_config_defaults(args.config)
    if args.matlab_exe:
        defaults['matlab_exe'] = args.matlab_exe
    if args.spm_path:
        defaults['spm_path'] = args.spm_path
    
    results_dir = os.path.abspath(args.results_dir)
    
    if not os.path.exists(os.path.join(results_dir, "SPM.mat")):
        print(f"Error: SPM.mat not found in {results_dir}")
        sys.exit(1)
    
    print(f"Running statistical sweep for: {results_dir}")
    print("=" * 80)
    
    # 1. Double-Threshold Maps
    print("\n[1/3] Generating Double-Threshold maps...")
    print("-" * 80)
    
    sweeps = [
        (0.001, 0.05),   # p_unc < 0.001, p_FWE < 0.05
        (0.005, 0.05),   # p_unc < 0.005, p_FWE < 0.05
    ]
    
    for p_unc, p_fwe in sweeps:
        # Convert p_unc to CAT12 percentage format
        # 0.001 -> 0.1%, 0.005 -> 0.5%
        p_str = f"p{p_unc*100:g}"
        pattern = f"*_{p_str}_pkFWE{int(p_fwe*100)}*"
        
        existing = list(Path(results_dir).glob(pattern))
        if existing and not args.force:
            print(f"--- Skipping Double Threshold (p_unc={p_unc}, p_FWE={p_fwe}): Results already exist. ---")
            continue
        
        print(f"--- Running Double Threshold: p_unc < {p_unc}, p_FWE < {p_fwe} ---")
        cmd = f"cat12_threshold_maps('{results_dir}', 'p_unc', {p_unc}, 'p_fwe', {p_fwe}, 'both', true, 'log', true)"
        try:
            run_matlab_cmd(cmd, args)
        except Exception as e:
            print(f"Warning: Double-threshold generation failed: {e}")
    
    # 2. Effect Size Maps (Cohen's d)
    print("\n[2/3] Generating Effect Size (Cohen's d) maps...")
    print("-" * 80)
    
    if list(Path(results_dir).glob("Cohen_d_*.nii")) and not args.force:
        print("--- Skipping Cohen's d Maps: Results already exist. ---")
    else:
        print("--- Generating Cohen's d Effect Size Maps ---")
        cmd = f"generate_effect_size('{results_dir}')"
        try:
            run_matlab_cmd(cmd, args)
        except Exception as e:
            print(f"Warning: Effect size generation failed: {e}")
    
    # 3. Generate Interactive HTML Report
    print("\n[3/3] Generating interactive HTML report...")
    print("-" * 80)
    
    report_script = Path(__file__).parent / "post_stats_report.py"
    report_html = os.path.join(results_dir, "post_stats_sweep_report.html")
    
    if report_script.exists():
        print(f"--- Generating HTML Report: {report_html} ---")
        
        report_cmd = [sys.executable, str(report_script), results_dir, report_html]
        if args.spm_path:
            report_cmd.extend(["--spm-path", args.spm_path])
        
        try:
            subprocess.check_call(report_cmd)
            print(f"✓ Report generated successfully: {report_html}")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Report generation failed: {e}")
    else:
        print(f"Warning: post_stats_report.py not found at {report_script}")
    
    print("\n" + "=" * 80)
    print("Sweep complete!")
    print(f"Results directory: {results_dir}")
    print(f"Report: {report_html}")
    print("=" * 80)


if __name__ == "__main__":
    main()
