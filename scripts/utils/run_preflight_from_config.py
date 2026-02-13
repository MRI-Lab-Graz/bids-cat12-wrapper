#!/usr/bin/env python3
"""Run preflight checks from project config."""

import argparse
import json
import os
import sys
import glob
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
import preflight_check as pc


def main():
    parser = argparse.ArgumentParser(
        description="Run preflight checks from project config"
    )
    parser.add_argument(
        "--project-config",
        required=True,
        help="Path to project_config.json",
    )
    args = parser.parse_args()

    config_path = Path(args.project_config)
    if not config_path.exists():
        print(f"ERROR: Project config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        project_config = json.load(f)

    # Check if preflight checks are configured
    preflight_config = project_config.get("preprocessing", {}).get("preflight_checks", {})
    check_environment = preflight_config.get("check_environment", True)
    check_dependencies = preflight_config.get("check_dependencies", True)

    # Run checks
    print("=" * 70)
    print("PREFLIGHT CHECKS")
    print("=" * 70)

    all_ok = True

    # Check MATLAB + SPM (only if check_environment is enabled)
    if check_environment:
        print("\n1. Checking MATLAB and SPM...")
        if not pc.check_matlab_and_spm():
            all_ok = False
    else:
        print("\n1. Checking MATLAB and SPM... SKIPPED (check_environment: false)")
        print("   ℹ️  Using standalone mode - MATLAB/SPM not required")

    # Check CAT12 (if preprocessing and check_dependencies enabled)
    if "preproc" in project_config and check_dependencies:
        print("\n2. Checking CAT12 mode (preprocessing)...")
        # This is handled in cat12_utils.py, but we can at least verify paths exist
        matlab_exe = project_config.get("matlab", {}).get("executable")
        spm_path = project_config.get("spm", {}).get("path")
        
        if matlab_exe and Path(matlab_exe).exists():
            print(f"   ✓ MATLAB executable: {matlab_exe}")
        else:
            print(f"   ✗ MATLAB not found: {matlab_exe}")
            all_ok = False

        if spm_path and Path(spm_path).exists():
            print(f"   ✓ SPM path: {spm_path}")
        else:
            print(f"   ✗ SPM not found: {spm_path}")
            all_ok = False
    elif "preproc" in project_config and not check_dependencies:
        print("\n2. Checking CAT12 mode... SKIPPED (check_dependencies: false)")
        print("   ℹ️  Standalone mode configured")

    # Check CAT12 surface tools if surface processing is requested
    preproc_cfg = project_config.get("preprocessing", {})
    proc_cfg = preproc_cfg.get("processing", {})
    smooth_cfg = preproc_cfg.get("smoothing", {})
    surface_requested = (not proc_cfg.get("no_surface", False)) or bool(
        smooth_cfg.get("smooth_surface", [])
    )

    if "preproc" in project_config and surface_requested:
        print("\n2b. Checking CAT12 surface tools...")
        software_cfg = project_config.get("software", {})
        mode = software_cfg.get("mode", "matlab")
        cat12_root = None
        if mode == "standalone":
            cat12_root = software_cfg.get("cat12_standalone", {}).get("path")
        else:
            cat12_root = software_cfg.get("spm", {}).get("path")

        warn_only = not check_dependencies
        if not cat12_root:
            print("   ✗ CAT12 root not configured; cannot verify surface tools")
            if not warn_only:
                all_ok = False
        else:
            search_root = Path(cat12_root)
            candidates = glob.glob(
                str(search_root / "**" / "CAT_RefineMesh"), recursive=True
            )
            if not candidates:
                print("   ✗ CAT_RefineMesh not found; surface processing likely unavailable")
                if not warn_only:
                    all_ok = False
            else:
                refinemesh = candidates[0]
                if os.access(refinemesh, os.X_OK):
                    print(f"   ✓ CAT_RefineMesh found: {refinemesh}")
                else:
                    print(
                        "   ✗ CAT_RefineMesh is not executable (fix with: chmod a+x)"
                    )
                    if not warn_only:
                        all_ok = False
        if warn_only:
            print("   ℹ️  Dependency checks disabled; surface warnings only")
    elif "preproc" in project_config and not surface_requested:
        print("\n2b. Checking CAT12 surface tools... SKIPPED (no_surface: true)")

    # Check stats (if stats config exists and check_dependencies enabled)
    if "stats" in project_config and check_dependencies:
        print("\n3. Checking statistics setup...")
        stats_cfg = project_config["stats"]
        cat12_dir = Path(stats_cfg.get("cat12_dir", ""))
        participants = Path(stats_cfg.get("participants", ""))
        
        if cat12_dir.exists():
            print(f"   ✓ CAT12 directory: {cat12_dir}")
        else:
            print(f"   ℹ️  CAT12 directory will be created: {cat12_dir}")

        if participants.exists():
            print(f"   ✓ Participants file: {participants}")
        else:
            print(f"   ℹ️  Participants file will be created during preprocessing: {participants}")
    elif "stats" in project_config and not check_dependencies:
        print("\n3. Checking statistics setup... SKIPPED (check_dependencies: false)")

    print("\n" + "=" * 70)
    if all_ok:
        print("✓ All preflight checks PASSED")
        sys.exit(0)
    else:
        print("✗ Some preflight checks FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
