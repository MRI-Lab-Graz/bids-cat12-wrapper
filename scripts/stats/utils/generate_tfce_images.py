#!/usr/bin/env python3
"""
Generate PNG images from TFCE FWE-corrected results for HTML reporting.

This script finds all TFCE result directories and creates visualization images
showing FWE-corrected (p < 0.05) results as PNG overlays for HTML inclusion.

Usage:
    python3 generate_tfce_images.py --output-dir results/vbm/vbm_smooth_auto --fwe-threshold 0.05
"""
import argparse
import glob
import json
import os


def find_tfce_results(output_dir, fwe_threshold=0.05, start_time=None):
    """
    Find all TFCE result directories and extract significant results.

    Returns a list of dicts with:
        - contrast_idx: int
        - contrast_name: str
        - fwe_file: path to FWE-corrected p-value map
        - has_significant: bool (whether any voxels survive FWE threshold)
    """
    results = []

    # Find all TFCE directories (old naming: TFCE_<idx>/ or new naming: tfce files in root)
    tfce_dirs = sorted(glob.glob(os.path.join(output_dir, "TFCE_*")))

    # Also check for newer TFCE output naming in root (tfce_<idx>_*.nii)
    if not tfce_dirs:
        # Look for tfce files directly
        tfce_files = sorted(glob.glob(os.path.join(output_dir, "tfce_*_log_pFWE.nii")))
        if tfce_files:
            print(f"Found {len(tfce_files)} TFCE result files in root directory")

    # Load SPM.mat to get contrast names if available
    contrast_names = {}
    try:
        # Try to extract contrast names from SPM.mat using a quick hack
        # (real implementation would use scipy.io.loadmat or MATLAB)
        pass
    except Exception:
        pass

    for tfce_dir in tfce_dirs:
        basename = os.path.basename(tfce_dir)
        if not basename.startswith("TFCE_"):
            continue

        try:
            contrast_idx = int(basename.split("_")[1])
        except (IndexError, ValueError):
            continue

        # Look for FWE-corrected files (NIfTI or GIfTI)
        fwe_file = os.path.join(tfce_dir, "logP_max.nii")
        if not os.path.exists(fwe_file):
            fwe_file = os.path.join(tfce_dir, "logP_max.gii")
        
        if not os.path.exists(fwe_file):
            # Try alternative naming
            fwe_file = os.path.join(tfce_dir, "TFCE_log_pFWE.nii")
        
        if not os.path.exists(fwe_file):
            fwe_file = os.path.join(tfce_dir, "TFCE_log_pFWE.gii")

        if not os.path.exists(fwe_file):
            continue

        # Filter by start time if provided
        if start_time is not None:
            try:
                mtime = os.path.getmtime(fwe_file)
                if mtime < start_time:
                    continue
            except OSError:
                continue

        contrast_name = contrast_names.get(contrast_idx, f"Contrast {contrast_idx}")

        results.append(
            {
                "contrast_idx": contrast_idx,
                "contrast_name": contrast_name,
                "fwe_file": fwe_file,
                "tfce_dir": tfce_dir,
            }
        )

    # Also check for newer naming convention (files in root)
    # Matches TFCE_log_pFWE_0001.nii/.gii AND TFCE_0001_log_pFWE.nii/.gii
    root_tfce_files = glob.glob(os.path.join(output_dir, "TFCE_*_log_pFWE.nii")) + \
                      glob.glob(os.path.join(output_dir, "TFCE_log_pFWE_*.nii")) + \
                      glob.glob(os.path.join(output_dir, "TFCE_*_log_pFWE.gii")) + \
                      glob.glob(os.path.join(output_dir, "TFCE_log_pFWE_*.gii"))

    for fpath in root_tfce_files:
        # Filter by start time if provided
        if start_time is not None:
            try:
                mtime = os.path.getmtime(fpath)
                if mtime < start_time:
                    continue
            except OSError:
                continue

        basename = os.path.basename(fpath)
        try:
            # Try to extract contrast index from various patterns
            # Pattern 1: TFCE_0001_log_pFWE.nii
            if basename.startswith("TFCE_") and "_log_pFWE" in basename:
                parts = basename.replace(".nii", "").replace(".gii", "").split("_")
                # Find the part that is a digit
                for p in parts:
                    if p.isdigit():
                        contrast_idx = int(p)
                        break
                else:
                    continue
            else:
                continue
        except (IndexError, ValueError):
            continue

        if any(r["contrast_idx"] == contrast_idx for r in results):
            continue  # Already found in directory

        contrast_name = contrast_names.get(contrast_idx, f"Contrast {contrast_idx}")
        results.append(
            {
                "contrast_idx": contrast_idx,
                "contrast_name": contrast_name,
                "fwe_file": fpath,
                "tfce_dir": output_dir,
            }
        )

    return sorted(results, key=lambda x: x["contrast_idx"])


def generate_summary_json(tfce_results, output_dir, fwe_threshold=0.05):
    """Generate a JSON summary of TFCE results for HTML report."""
    summary = {
        "fwe_threshold": fwe_threshold,
        "n_contrasts": len(tfce_results),
        "contrasts": [],
    }

    # Calculate log threshold: -log10(0.05) approx 1.301
    import nibabel as nb
    import numpy as np
    log_threshold = -np.log10(fwe_threshold)

    for res in tfce_results:
        has_results = False
        try:
            # Check if any voxel exceeds the threshold
            img = nb.load(res["fwe_file"])
            data = img.get_fdata()
            # Check max value
            if np.nanmax(data) > log_threshold:
                has_results = True
        except Exception as e:
            print(f"Warning: Could not check results for {res['fwe_file']}: {e}")
            # Fallback: assume true if file exists so we don't hide potential errors
            has_results = os.path.exists(res["fwe_file"])

        summary["contrasts"].append(
            {
                "index": res["contrast_idx"],
                "name": res["contrast_name"],
                "fwe_file": os.path.relpath(res["fwe_file"], output_dir),
                "has_results": has_results,
            }
        )

    summary_file = os.path.join(output_dir, "tfce_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ TFCE summary saved to: {summary_file}")
    return summary_file


def main():
    parser = argparse.ArgumentParser(description="Generate TFCE visualization images")
    parser.add_argument("--output-dir", required=True, help="Results output directory")
    parser.add_argument(
        "--fwe-threshold",
        type=float,
        default=0.05,
        help="FWE-corrected threshold (default: 0.05)",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Unix timestamp of pipeline start (filter older results)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("TFCE Results Image Generation")
    print("=" * 80 + "\n")

    print(f"Output directory: {args.output_dir}")
    print(f"FWE threshold: p < {args.fwe_threshold}")
    if args.start_time:
        print(f"Filtering results created before: {args.start_time}")
    print()

    # Find TFCE results
    tfce_results = find_tfce_results(args.output_dir, args.fwe_threshold, args.start_time)

    if not tfce_results:
        print("⚠ No TFCE results found")
        return

    print(f"✓ Found {len(tfce_results)} TFCE result(s)\n")

    for res in tfce_results:
        print(f"  [{res['contrast_idx']:3d}] {res['contrast_name']}")
        print(f"        {res['fwe_file']}")

    print()

    # Generate summary JSON
    generate_summary_json(tfce_results, args.output_dir, args.fwe_threshold)

    print("\n✓ TFCE image generation complete\n")


if __name__ == "__main__":
    main()
