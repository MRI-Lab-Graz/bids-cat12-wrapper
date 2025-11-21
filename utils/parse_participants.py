#!/usr/bin/env python3
"""
Parse BIDS participants.tsv and build file lists for CAT12 analysis.

This script reads a BIDS participants.tsv file, matches subjects/sessions
with CAT12 preprocessed files, and generates the design structure for SPM.
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import re
import xml.etree.ElementTree as ET


def find_xml_for_subject(cat12_dir, subject, session=None):
    """Find CAT12 XML report for a given subject/session."""
    subj_norm = str(subject).replace("sub-", "").replace("SUB-", "")
    # Prefer CAT12 "report" XMLs (cat_r*.xml) which contain the subject
    # measures (e.g. <vol_TIV>). Fall back to long/parameter XMLs
    # (catlong_*.xml) or any XML found in the report folder.
    session_token = f"ses-{session}" if session is not None else None

    for subject_dir in _subject_data_dirs(cat12_dir, subject):
        report_dir = subject_dir / "report"
        if not report_dir.exists():
            continue

        # Build candidate patterns - include both subject/session orders so we
        # robustly match filenames like `cat_rsub-<id>_ses-<n>_...` and
        # `cat_r_ses-<n>_sub-<id>_...`.
        patterns = []
        if session_token:
            # Prefer report XMLs with either order
            patterns.extend(
                [
                    f"cat_r*{session_token}*{subj_norm}*.xml",
                    f"cat_r*{subj_norm}*{session_token}*.xml",
                    f"*{session_token}*{subj_norm}*.xml",
                    f"*{subj_norm}*{session_token}*.xml",
                ]
            )
            # Fallback to long parameter XMLs in both orders
            patterns.extend(
                [
                    f"catlong*{session_token}*{subj_norm}*.xml",
                    f"catlong*{subj_norm}*{session_token}*.xml",
                ]
            )
        else:
            patterns.extend(
                [
                    f"cat_r*{subj_norm}*.xml",
                    f"catlong*{subj_norm}*.xml",
                    f"*{subj_norm}*.xml",
                ]
            )

        # Finally accept any XML
        patterns.append("*.xml")

        # Try each pattern in order and return the first sensible match.
        for pat in patterns:
            matches = sorted(report_dir.glob(pat))
            if matches:
                # Prefer files whose name contains 'cat_r' (standard report)
                for m in matches:
                    if "cat_r" in m.name:
                        return str(m)
                # Otherwise return the first match
                return str(matches[0])

    return None


def extract_measure_from_xml(xml_path, measure_name):
    """Extract a single measure from CAT12 XML report."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        target = measure_name.lower()

        for node in root.iter():
            if node.tag.lower() != target:
                continue
            if node.text is None:
                continue
            val = node.text.strip()
            val = re.sub(r"^[\[\s]+|[\]\s]+$", "", val)
            try:
                return float(val)
            except ValueError:
                continue
        return None
    except Exception:
        return None


XML_COVARIATE_ALIASES = {
    "tiv": "vol_TIV",
}


def _resolve_xml_covariate(covariate_name):
    return XML_COVARIATE_ALIASES.get(covariate_name.lower(), covariate_name)


def extract_smoothing_from_filename(filepath):
    """
    Extract smoothing kernel size from CAT12 filename.

    Parameters:
    -----------
    filepath : str
        Path to CAT12 file

    Returns:
    --------
    int or None
        Smoothing kernel size in mm, or None if not found
    """
    filename = Path(filepath).name

    # VBM: s<N>mwp1*.nii (e.g., s6mwp1, s8mwp1)
    vbm_match = re.search(r"s(\d+)mwp1", filename)
    if vbm_match:
        return int(vbm_match.group(1))

    # Surface: s<N>.mesh.* (e.g., s15.mesh.thickness)
    surf_match = re.search(r"s(\d+)\.mesh\.", filename)
    if surf_match:
        return int(surf_match.group(1))

    # Default CAT12 (no number means default smoothing)
    # smwp1* without number = 8mm for VBM
    if "smwp1" in filename and not vbm_match:
        return 8  # CAT12 default for VBM

    return None


def detect_available_smoothing(cat12_dir, modality):
    """
    Detect available smoothing kernels in CAT12 directory.

    Parameters:
    -----------
    cat12_dir : str
        Path to CAT12 directory
    modality : str
        Analysis modality

    Returns:
    --------
    list
        List of available smoothing kernel sizes
    """
    smoothing_kernels = set()
    bases = _candidate_cat12_bases(cat12_dir)

    if modality == "vbm":
        for base in bases:
            for nii_file in base.rglob("mri/s*mwp1*.nii"):
                smooth = extract_smoothing_from_filename(nii_file.name)
                if smooth:
                    smoothing_kernels.add(smooth)
    else:
        # Check surf directory for smoothed surface files
        for base in bases:
            for gii_file in base.rglob("surf/s*.mesh.*.gii"):
                smooth = extract_smoothing_from_filename(gii_file.name)
                if smooth:
                    smoothing_kernels.add(smooth)

    return sorted(list(smoothing_kernels))


def _candidate_cat12_bases(cat12_dir):
    base = Path(cat12_dir)
    candidates = [
        base,
        base / "data",
        base / "cat12",
        base / "data" / "cat12",
        base / "cat12" / "data",
    ]
    seen = []
    for path in candidates:
        if path.exists() and path not in seen:
            seen.append(path)
    return seen


def _subject_data_dirs(cat12_dir, subject):
    if not subject.startswith("sub-"):
        subject = f"sub-{subject}"
    for base in _candidate_cat12_bases(cat12_dir):
        candidate = base / subject
        if candidate.exists():
            yield candidate


def find_cat12_files(cat12_dir, subject, session, modality, smoothing):
    """
    Find CAT12 preprocessed files for a given subject and session.

    Parameters:
    -----------
    cat12_dir : str
        Path to CAT12 directory
    subject : str
        Subject ID (with or without 'sub-' prefix)
    session : str
        Session ID (with or without 'ses-' prefix)
    modality : str
        Analysis modality (vbm, thickness, depth, gyrification, fractal)
    smoothing : int or str
        Smoothing kernel size in mm, or 'auto' to auto-detect

    Returns:
    --------
    str or None
        Path to the file, or None if not found
    """
    # Ensure proper BIDS formatting
    if not subject.startswith("sub-"):
        subject = f"sub-{subject}"

    # Handle session as string or int
    session = str(session)
    if not session.startswith("ses-"):
        session = f"ses-{session}"

    session = str(session)
    bases = list(_subject_data_dirs(cat12_dir, subject))
    if not bases:
        return None

    # Normalize smoothing flag
    smoothing_kw = None if smoothing in (None, "auto", "") else str(smoothing)

    for subject_dir in bases:
        if modality == "vbm":
            mri_dir = subject_dir / "mri"
            if not mri_dir.exists():
                continue

            matches = list(mri_dir.glob(f"*{subject}_{session}_*.nii"))
            if not matches:
                continue

            if smoothing_kw:
                # STRICT FILTERING: Only accept files that match the requested smoothing
                # This prevents mixing s6, s9, and unsmoothed files in the same analysis.
                filtered_matches = []
                for m in matches:
                    # Check if filename starts with s<N> or contains s<N>.mesh
                    # We use the robust extractor to be sure
                    s_val = extract_smoothing_from_filename(m.name)
                    
                    # Case 1: Requested smoothing is explicit (e.g. 6, 8, 9)
                    if isinstance(args.smoothing, int):
                        if s_val == args.smoothing:
                            filtered_matches.append(m)
                    # Case 2: Requested smoothing is string/auto (should be resolved by now, but just in case)
                    elif str(s_val) == str(args.smoothing):
                        filtered_matches.append(m)
                        
                matches = filtered_matches

            if not matches:
                return None

            # If multiple matches remain (unlikely with strict smoothing), pick the first one
            # that looks like a VBM file
            for match in matches:
                if "mwp1" in match.name:
                    return str(match)
        else:
            measure_map = {
                "thickness": "thickness",
                "depth": "depthWM",
                "gyrification": "gyrification",
                "fractal": "fractaldimension",
            }
            measure = measure_map.get(modality)
            if not measure:
                continue

            surf_dir = subject_dir / "surf"
            if not surf_dir.exists():
                continue

            matches = list(surf_dir.glob(f"*{measure}*.r{subject}_{session}_*.gii"))
            if not matches:
                continue

            if smoothing_kw:
                matches = sorted(matches, key=lambda p: smoothing_kw not in p.name)
            if matches:
                return str(matches[0])

    return None


def parse_participants(args):
    """Main parsing function."""

    print("Reading participants file...")
    df = pd.read_csv(args.participants, sep="\t")

    print(f"Found {len(df)} rows in participants file")
    print(f"Columns: {', '.join(df.columns)}")

    # Auto-detect group column if not specified
    if not args.group_col:
        group_candidates = [
            col
            for col in df.columns
            if "group" in col.lower() or "condition" in col.lower()
        ]
        if group_candidates:
            args.group_col = group_candidates[0]
            print(f"Auto-detected group column: {args.group_col}")
        else:
            print(
                "Error: Could not auto-detect group column. Please specify --group-col"
            )
            sys.exit(1)

    # Check required columns exist
    # BIDS-compliant format: one row per subject with nr_sessions column
    if "nr_sessions" in df.columns:
        # BIDS format: one row per subject
        print("Detected BIDS-compliant format (one row per subject)")
        required_cols = ["participant_id", "nr_sessions", args.group_col]
        is_bids_format = True
    else:
        # Old format: one row per scan
        print("Detected scan-level format (one row per scan)")
        required_cols = ["participant_id", args.group_col]
        is_bids_format = False
        if args.session_col not in df.columns:
            print(f"Error: session column '{args.session_col}' not found")
            sys.exit(1)
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Error: Missing required columns: {', '.join(missing)}")
        print(f"Available columns: {', '.join(df.columns)}")
        sys.exit(1)

    # Auto-detect smoothing if not specified
    if args.smoothing == "auto":
        print(f"\nAuto-detecting smoothing kernels for {args.modality}...")
        available_smoothing = detect_available_smoothing(args.cat12_dir, args.modality)

        if not available_smoothing:
            print(
                f"Error: Could not find any smoothed {args.modality} files in {args.cat12_dir}"
            )
            sys.exit(1)

        print(f"Found smoothing kernels: {', '.join(map(str, available_smoothing))}mm")

        # Use the most common smoothing for the modality
        if args.modality == "vbm":
            # Prefer 8mm for VBM
            args.smoothing = 8 if 8 in available_smoothing else available_smoothing[0]
        else:
            # Prefer 15mm for surface
            args.smoothing = (
                15 if 15 in available_smoothing else available_smoothing[-1]
            )

        print(f"Selected smoothing: {args.smoothing}mm")
    else:
        args.smoothing = int(args.smoothing)

    # Parse covariate columns
    covariate_cols = []
    categorical_covariates = set()
    
    if args.covariates:
        covariate_cols = [c.strip() for c in args.covariates.split(",")]
        
        # --------------------------------------------------------------------
        # Auto-encode categorical covariates (BIDS-aware)
        # --------------------------------------------------------------------
        json_path = Path(args.participants).with_suffix('.json')
        sidecar = {}
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    sidecar = json.load(f)
                print(f"Loaded BIDS sidecar: {json_path.name}")
            except Exception as e:
                print(f"Warning: Could not read {json_path.name}: {e}")

        for cov in covariate_cols:
            if cov in df.columns:
                # Check if encoding is needed:
                # 1. Explicit "Levels" in JSON sidecar
                # 2. Non-numeric data type in DataFrame
                is_categorical = False
                if cov in sidecar and "Levels" in sidecar[cov]:
                    is_categorical = True
                elif not pd.api.types.is_numeric_dtype(df[cov]):
                    is_categorical = True

                if is_categorical:
                    categorical_covariates.add(cov)
                    # Get unique values, sorted for deterministic mapping
                    unique_vals = sorted(df[cov].dropna().unique())
                    mapping = {val: i for i, val in enumerate(unique_vals)}
                    
                    print(f"  → Auto-encoding categorical covariate '{cov}':")
                    for val, code in mapping.items():
                        print(f"      '{val}' -> {code}")
                    
                    # Apply mapping to DataFrame
                    df[cov] = df[cov].map(mapping)
        # --------------------------------------------------------------------

        # Check which covariates are already in participants.tsv
        missing_cov = [col for col in covariate_cols if col not in df.columns]
        if missing_cov:
            print(
                f"Note: Covariates {', '.join(missing_cov)} not in participants.tsv → will extract from CAT12 XML reports"
            )
        present_cov = [col for col in covariate_cols if col in df.columns]
        if present_cov:
            print(
                f"Note: Using covariates {', '.join(present_cov)} from participants.tsv"
            )

    # Build file list and design structure
    print(f"\nSearching for CAT12 files ({args.modality}, {args.smoothing}mm)...")

    design = {
        "modality": args.modality,
        "smoothing": args.smoothing,
        "groups": {},
        "sessions": [],
        "covariates": {},
        "files": [],
    }

    # Get unique groups and determine sessions
    groups = df[args.group_col].dropna().unique()

    if is_bids_format:
        # Determine sessions from nr_sessions column
        max_sessions = int(df["nr_sessions"].max())
        all_sessions = list(range(1, max_sessions + 1))
        print(f"BIDS format: {len(df)} subjects, up to {max_sessions} sessions each")
    else:
        # Determine sessions from session column
        all_sessions = sorted(df[args.session_col].dropna().unique())
        print(f"Scan format: {len(df)} scans")

    # Filter sessions based on --sessions argument
    if args.sessions == "all":
        sessions = all_sessions
    else:
        # Parse comma-separated session list
        requested_sessions = [int(s.strip()) for s in args.sessions.split(",")]
        sessions = [s for s in all_sessions if s in requested_sessions]
        if len(sessions) != len(requested_sessions):
            print("⚠️  Warning: Some requested sessions not found")
            print(f"   Requested: {requested_sessions}")
            print(f"   Available: {all_sessions}")
            print(f"   Using: {sessions}")

    print(f"Groups: {', '.join(map(str, groups))}")
    print(f"Sessions: {', '.join(map(str, sessions))}")

    design["sessions"] = [str(s) for s in sessions]

    # Initialize group structure
    for group in groups:
        design["groups"][str(group)] = {"sessions": {str(s): [] for s in sessions}}

    covariate_values = {cov: [] for cov in covariate_cols}
    covariate_missing = {cov: [] for cov in covariate_cols}

    # Track detected smoothing values
    detected_smoothing = set()

    # Process each participant
    files_found = 0
    files_missing = 0

    for _, row in df.iterrows():
        subject = row["participant_id"]
        group = row[args.group_col]

        # Skip if required fields are missing
        if pd.isna(subject) or pd.isna(group):
            continue

        # Determine which sessions to process
        if is_bids_format:
            # BIDS format: enumerate sessions based on nr_sessions
            nr_sessions = int(row["nr_sessions"])
            all_subj_sessions = list(range(1, nr_sessions + 1))
            # Filter to only requested sessions
            sessions_to_process = [s for s in all_subj_sessions if s in sessions]
        else:
            # Scan format: single session from this row
            session = row[args.session_col]
            if pd.isna(session):
                continue
            # Only process if this session is in the requested list
            if session not in sessions:
                continue
            sessions_to_process = [session]

        # Process each session for this subject
        for session in sessions_to_process:
            # Find CAT12 file
            filepath = find_cat12_files(
                args.cat12_dir, subject, session, args.modality, args.smoothing
            )

            if filepath:
                # Verify smoothing matches
                file_smooth = extract_smoothing_from_filename(filepath)
                if file_smooth:
                    detected_smoothing.add(file_smooth)
                    if file_smooth != args.smoothing:
                        print(
                            f"  ⚠ Warning: {subject} ses-{session} has {file_smooth}mm smoothing, expected {args.smoothing}mm"
                        )

                # Add to design structure
                design["groups"][str(group)]["sessions"][str(session)].append(filepath)
                design["files"].append(
                    {
                        "subject": subject,
                        "session": str(session),
                        "group": str(group),
                        "path": filepath,
                    }
                )

                # Add covariate values. Support both scan-level and
                # subject-level (BIDS) participants.tsv: if the TSV is
                # subject-level (has `nr_sessions`) the same covariate
                # value will be appended for each session of that subject.
                if covariate_cols:
                    for cov in covariate_cols:
                        # Try to get covariate from participants.tsv first
                        val = None
                        cov_lower = cov.lower()
                        if cov in row.index and not pd.isna(row[cov]):
                            val = row[cov]
                        elif cov_lower in ("ses", "session"):
                            val = session
                        else:
                            xml_path = find_xml_for_subject(
                                args.cat12_dir, subject, session
                            )
                            if xml_path:
                                measure = _resolve_xml_covariate(cov)
                                val = extract_measure_from_xml(xml_path, measure)
                                if val is not None:
                                    print(
                                        f"  → Extracted {cov}={val} from XML for {subject} ses-{session}"
                                    )

                        if val is None or pd.isna(val):
                            covariate_missing[cov].append((subject, session))
                            continue

                        # Convert to float for SPM regressors; provide clear error
                        try:
                            covariate_values[cov].append(float(val))
                        except Exception as e:
                            print(
                                f"Error: Covariate '{cov}' for subject {subject} session {session} is not numeric: {e}"
                            )
                            sys.exit(1)

                files_found += 1
            else:
                print(f"  ⚠ Missing: {subject} ses-{session}")
                files_missing += 1

    print(f"\n✓ Found {files_found} files")
    if files_missing > 0:
        print(f"⚠ Missing {files_missing} files")

    if detected_smoothing:
        print(
            f"Detected smoothing kernels in files: {', '.join(map(str, sorted(detected_smoothing)))}mm"
        )

    resolved_covariates = {}
    dropped_covariates = []
    for cov in covariate_cols:
        missing_entries = covariate_missing[cov]
        if missing_entries:
            sample = ", ".join(f"{sub} ses-{ses}" for sub, ses in missing_entries[:5])
            print(
                f"⚠ Missing {len(missing_entries)} value(s) for '{cov}' (examples: {sample})"
            )
            print(
                "  Provide the missing values in participants.tsv or CAT12 XML or rerun without that covariate."
            )
            if not args.allow_missing_covariates:
                print("  Pass --allow-missing-covariates to continue without it.")
                sys.exit(1)
            dropped_covariates.append(cov)
            continue
        resolved_covariates[cov] = covariate_values[cov]

    if dropped_covariates:
        print(f"⚠ Continuing without covariates: {', '.join(dropped_covariates)}")

    # Standardize continuous covariates if requested
    if args.standardize_continuous:
        import numpy as np
        print("\nStandardizing continuous covariates (z-score)...")
        for cov, values in resolved_covariates.items():
            if cov in categorical_covariates:
                print(f"  Skipping categorical covariate: {cov}")
                continue
            
            try:
                vals = np.array(values)
                mean_val = np.nanmean(vals)
                std_val = np.nanstd(vals)
                
                if std_val == 0:
                    print(f"  ⚠ Warning: Covariate '{cov}' has zero variance, skipping standardization")
                    continue
                    
                z_vals = (vals - mean_val) / std_val
                resolved_covariates[cov] = z_vals.tolist()
                print(f"  ✓ Standardized '{cov}' (mean={mean_val:.2f}, sd={std_val:.2f})")
            except Exception as e:
                print(f"  ⚠ Failed to standardize '{cov}': {e}")

    design["covariates"] = resolved_covariates

    # Validate design
    print("\nValidating design structure...")

    total_scans = 0
    for group, group_data in design["groups"].items():
        for session, files in group_data["sessions"].items():
            n = len(files)
            total_scans += n
            # Use '_by_' in printed cell names to avoid non-ASCII characters
            print(f"  {group}_by_{session}: {n} scans")

    print(f"\nTotal scans: {total_scans}")

    if total_scans == 0:
        print("Error: No files found!")
        sys.exit(1)

    # Validate covariates match
    for cov, values in design["covariates"].items():
        if len(values) != total_scans:
            print(
                f"Error: Covariate '{cov}' has {len(values)} values but {total_scans} scans"
            )
            sys.exit(1)

    # Save design structure
    output_file = Path(args.output) / "design.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(design, f, indent=2)

    print(f"\n✓ Design structure saved to: {output_file}")

    # Save file list
    files_file = Path(args.output) / "file_list.txt"
    with open(files_file, "w") as f:
        for file_info in design["files"]:
            f.write(f"{file_info['path']}\n")

    print(f"✓ File list saved to: {files_file}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse BIDS participants.tsv for CAT12 analysis"
    )

    parser.add_argument(
        "--cat12-dir", required=True, help="Path to CAT12 preprocessing directory"
    )
    parser.add_argument(
        "--participants", required=True, help="Path to BIDS participants.tsv file"
    )
    parser.add_argument(
        "--modality",
        default="vbm",
        choices=["vbm", "thickness", "depth", "gyrification", "fractal"],
        help="Analysis modality (vbm uses mri/, others use surf/)",
    )
    parser.add_argument(
        "--smoothing",
        default="auto",
        help='Smoothing kernel size in mm, or "auto" to auto-detect (default: auto)',
    )
    parser.add_argument(
        "--group-col",
        default="",
        help="Column name for group variable (auto-detect if not specified)",
    )
    parser.add_argument(
        "--session-col", default="session", help="Column name for session variable"
    )
    parser.add_argument(
        "--sessions",
        default="all",
        help='Sessions to include: "all" or comma-separated list like "1,3"',
    )
    parser.add_argument(
        "--covariates", default="", help="Comma-separated covariate column names"
    )
    parser.add_argument(
        "--allow-missing-covariates",
        action="store_true",
        help="Continue with available covariates even if some values are missing",
    )
    parser.add_argument(
        "--standardize-continuous",
        action="store_true",
        help="Standardize (z-score) continuous covariates",
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for design files"
    )

    args = parser.parse_args()
    sys.exit(parse_participants(args))
