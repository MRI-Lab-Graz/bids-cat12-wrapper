#!/usr/bin/env python3
"""Run preflight checks from project config."""

import argparse
import json
import sys
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

    # Run checks
    print("=" * 70)
    print("PREFLIGHT CHECKS")
    print("=" * 70)

    all_ok = True

    # Check MATLAB + SPM
    print("\n1. Checking MATLAB and SPM...")
    if not pc.check_matlab_and_spm():
        all_ok = False

    # Check CAT12 (if preprocessing)
    if "preproc" in project_config:
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

    # Check stats (if stats config exists)
    if "stats" in project_config:
        print("\n3. Checking statistics setup...")
        stats_cfg = project_config["stats"]
        cat12_dir = Path(stats_cfg.get("cat12_dir", ""))
        participants = Path(stats_cfg.get("participants", ""))
        
        if cat12_dir.exists():
            print(f"   ✓ CAT12 directory: {cat12_dir}")
        else:
            print(f"   ✗ CAT12 directory not found (will be created): {cat12_dir}")

        if participants.exists():
            print(f"   ✓ Participants file: {participants}")
        else:
            print(f"   ⚠ Participants file not found (create before stats): {participants}")

    print("\n" + "=" * 70)
    if all_ok:
        print("✓ All preflight checks PASSED")
        sys.exit(0)
    else:
        print("✗ Some preflight checks FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
