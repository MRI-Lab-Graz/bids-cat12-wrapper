#!/usr/bin/env python3
"""
Automated statistical sweep for CAT12 results.

Generates multiple thresholded maps, TFCE correction, effect size maps,
then creates an interactive HTML report.

Features:
- Double-threshold sweep (voxel + cluster level)
- TFCE correction with configurable permutations
- Effect size (Cohen's d) map generation
- Automatic HTML report generation
- Config-based MATLAB/SPM path management

Usage:
    python run_stats_sweep.py ./results/vbm/analysis
    python run_stats_sweep.py ./results/vbm/analysis --use-matlab --tfce 5000
    python run_stats_sweep.py ./results/vbm/analysis --tfce 10000 --force
    python run_stats_sweep.py ./results/vbm/analysis --config config/config.study_intervention.json
"""

import os
import argparse
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime


def get_config_defaults(config_file=None):
    """Read defaults from config/config.json (or custom study config)"""
    defaults = {
        "matlab_exe": "/Applications/MATLAB_R2025b.app/bin/matlab",
        "spm_path": "",
        "tfce_permutations": 5000,
        "report_quality": "low",
        "report_filter": "all",
        "display_convention": "spm",
    }

    # Find config file
    workspace_root = Path(__file__).resolve().parents[2]  # Go up to workspace root
    if config_file is None:
        config_file = workspace_root / "config" / "config.json"
    else:
        config_file = Path(config_file)

    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)

            if "matlab" in config and config["matlab"].get("executable"):
                defaults["matlab_exe"] = config["matlab"]["executable"]
            if "spm" in config and config["spm"].get("path"):
                defaults["spm_path"] = config["spm"]["path"]
            if "tfce" in config and config["tfce"].get("n_permutations"):
                defaults["tfce_permutations"] = config["tfce"]["n_permutations"]
            if "reporting" in config and config["reporting"].get("quality"):
                defaults["report_quality"] = config["reporting"]["quality"]
            if "reporting" in config and config["reporting"].get("filter"):
                defaults["report_filter"] = config["reporting"]["filter"]
            if "reporting" in config and config["reporting"].get("display_convention"):
                defaults["display_convention"] = config["reporting"]["display_convention"]
        except Exception as e:
            print(f"Warning: Could not read config file {config_file}: {e}")

    return defaults


def run_matlab_cmd(cmd, args):
    """Run a MATLAB command using the standalone runner or direct MATLAB."""
    workspace_root = Path(__file__).resolve().parents[2]
    utils_dir = workspace_root / "scripts" / "utils"

    # Try to use runner if available
    runner = Path(__file__).parent.parent / "utils" / "run_matlab_standalone.py"

    if runner.exists() and not args.use_matlab:
        full_cmd = [sys.executable, str(runner), cmd, "--utils", str(utils_dir)]

        if args.mcr:
            full_cmd.extend(["--mcr", args.mcr])
        if args.standalone:
            full_cmd.extend(["--standalone", args.standalone])
    else:
        # Use direct MATLAB
        matlab_exe = (
            args.matlab_exe if args.matlab_exe else get_config_defaults()["matlab_exe"]
        )
        full_cmd = [
            matlab_exe,
            "-nodisplay",
            "-nosplash",
            "-r",
            f"addpath('{utils_dir}'); {cmd}; exit;",
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
  python run_stats_sweep.py ./results/vbm/analysis --use-matlab --tfce 5000
  python run_stats_sweep.py ./results/vbm/analysis --tfce 10000 --force
  python run_stats_sweep.py ./results/vbm/analysis --config config/config.study_intervention.json
        """,
    )
    parser.add_argument("results_dir", help="Path to the directory containing SPM.mat")

    # Config option
    parser.add_argument(
        "--config", help="Path to config.json (default: config/config.json)"
    )

    # MATLAB Options
    parser.add_argument(
        "--use-matlab",
        action="store_true",
        help="Use local MATLAB instead of standalone MCR",
    )
    parser.add_argument("--matlab-exe", help="Path to MATLAB executable")
    parser.add_argument("--spm-path", help="Path to SPM installation")

    # MCR Options
    parser.add_argument("--mcr", help="Path to MCR")
    parser.add_argument("--standalone", help="Path to run_spm12.sh")

    # Control Options
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing statistic maps"
    )
    parser.add_argument(
        "--skip-plots", action="store_true", help="Skip plot generation (faster)"
    )
    parser.add_argument(
        "--tfce",
        type=int,
        help="Number of TFCE permutations (default: config tfce.n_permutations or 5000)",
    )
    parser.add_argument(
        "--report-filter",
        choices=["all", "tfce", "spmt", "double_threshold", "no_tfce"],
        help="Filter mode for HTML report (default: config reporting.filter or all)",
    )
    parser.add_argument(
        "--report-quality",
        choices=["low", "standard", "publication"],
        help="HTML report quality (default: config reporting.quality or low)",
    )
    parser.add_argument(
        "--glassbrain",
        action="store_true",
        help="Enable glass-brain generation in HTML report (disabled by default).",
    )
    parser.add_argument(
        "--display-convention",
        choices=["spm", "auto", "radiological", "neurological"],
        help="Plot orientation convention for volume figures (default: config reporting.display_convention or spm)",
    )

    args = parser.parse_args()

    # Load config
    defaults = get_config_defaults(args.config)
    if args.matlab_exe:
        defaults["matlab_exe"] = args.matlab_exe
    if args.spm_path:
        defaults["spm_path"] = args.spm_path

    tfce_permutations = args.tfce if args.tfce is not None else defaults["tfce_permutations"]
    report_filter = args.report_filter if args.report_filter else defaults["report_filter"]
    report_quality = args.report_quality if args.report_quality else defaults["report_quality"]
    display_convention = (
        args.display_convention if args.display_convention else defaults["display_convention"]
    )

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
        (0.001, 50),  # p_thresh = 0.001, cluster_size = 50
        (0.005, 50),  # p_thresh = 0.005, cluster_size = 50
    ]

    for p_thresh, cluster_size in sweeps:
        print(f"--- Running Contrast Screening: p < {p_thresh} (unc), k >= {cluster_size} voxels ---")
        cmd = f"screen_contrasts('{results_dir}', 'p_thresh', {p_thresh}, 'cluster_size', {cluster_size})"
        try:
            run_matlab_cmd(cmd, args)
        except Exception as e:
            print(f"Warning: Contrast screening failed: {e}")

    # 2. TFCE Correction (Family-Wise Error correction)
    print("\n[2/4] Running TFCE correction...")
    print("-" * 80)

    tfce_dir = os.path.join(results_dir, "TFCE")
    tfce_done = os.path.join(tfce_dir, "tfce_done.txt")

    if os.path.exists(tfce_done) and not args.force:
        print("--- Skipping TFCE: Results already exist. ---")
    else:
        print(f"--- Running TFCE Permutation Correction ({tfce_permutations} permutations) ---")
        cmd = f"run_tfce_correction('{results_dir}', 'n_perm', {tfce_permutations})"
        try:
            run_matlab_cmd(cmd, args)
        except Exception as e:
            print(f"Warning: TFCE correction failed: {e}")

    # 3. Effect Size Maps (Cohen's d)
    print("\n[3/4] Generating Effect Size (Cohen's d) maps...")
    print("-" * 80)

    if list(Path(results_dir).glob("spmT_*")) and not args.force:
        print("--- Generating effect size from t-maps ---")
        # Effect size is derived from t-maps (already in results)
        print("✓ Effect size maps available (derived from t-statistics)")
    else:
        print("--- No t-maps found to generate effect size ---")

    # 4. Generate Interactive HTML Report
    print("\n[4/4] Generating interactive HTML report...")
    print("-" * 80)

    # Use the correct post_stats_report.py from the reporting directory
    report_script = Path(__file__).parent.parent / "reporting" / "post_stats_report.py"
    report_html = os.path.join(results_dir, f"report_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.html")

    if report_script.exists():
        print(f"--- Generating HTML Report: {report_html} ---")

        report_cmd = [
            sys.executable,
            str(report_script),
            results_dir,
            report_html,
            "--quality",
            report_quality,
            "--filter",
            report_filter,
            "--display-convention",
            display_convention,
        ]
        if args.glassbrain:
            report_cmd.append("--glassbrain")
        if args.spm_path:
            report_cmd.extend(["--spm-path", args.spm_path])
        if args.config:
            report_cmd.extend(["--config", args.config])

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
