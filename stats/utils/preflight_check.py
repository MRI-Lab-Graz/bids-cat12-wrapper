#!/usr/bin/env python3
"""Preflight checks for the CAT12 longitudinal pipeline."""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
from typing import Dict, List, Tuple

import pandas as pd

# Try to import local helpers
try:
    import parse_participants as pp
except ImportError:
    # If running from a different directory, ensure utils is in path
    script_dir = os.path.dirname(__file__)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    try:
        import parse_participants as pp
    except ImportError:
        pp = None  # Handle gracefully in functions


def check_matlab_and_spm() -> bool:
    """Ensure MATLAB executable and SPM installation path are available."""
    script_dir = os.path.dirname(__file__)
    repo_root = os.path.dirname(script_dir)

    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    try:
        from load_config import load_config, get_matlab_exe, get_spm_path  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive import guard
        print(f"ERROR: Unable to import load_config helper: {exc}")
        return False

    config = load_config()
    matlab_exe = (get_matlab_exe(config) or "").strip()
    overall_ok = True

    def _resolve_matlab_path(candidate: str) -> Tuple[bool, str | None]:
        if not candidate:
            return False, None
        expanded = os.path.expanduser(candidate)
        if os.path.sep in expanded or expanded.startswith("."):
            if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
                return True, os.path.realpath(expanded)
            return False, expanded
        resolved = shutil.which(expanded)
        if resolved:
            return True, resolved
        return False, expanded

    matlab_ok, matlab_path = _resolve_matlab_path(matlab_exe)
    if matlab_ok and matlab_path:
        print(f"✓ MATLAB executable found: {matlab_path}")
    else:
        print(
            "ERROR: MATLAB executable not found. Update [MATLAB] exe in config.ini or ensure 'matlab' is on PATH."
        )
        if matlab_path:
            print(f"       Checked: {matlab_path}")
        overall_ok = False

    spm_candidates: List[str] = []
    cfg_spm = get_spm_path(config)
    if cfg_spm:
        spm_candidates.append(cfg_spm)
    env_spm = os.environ.get("SPM_PATH")
    if env_spm:
        spm_candidates.append(env_spm)
    spm_config_file = os.path.join(repo_root, "spm_config.txt")
    if os.path.isfile(spm_config_file):
        try:
            with open(spm_config_file, "r") as fh:
                first_line = fh.readline().strip()
                if first_line:
                    spm_candidates.append(first_line)
        except OSError:
            pass

    seen: set[str] = set()
    resolved_spm: str | None = None
    missing_details: Tuple[str, List[str]] | None = None
    for candidate in spm_candidates:
        normalized = os.path.realpath(os.path.expanduser(candidate))
        if normalized in seen:
            continue
        seen.add(normalized)
        if not os.path.isdir(normalized):
            continue
        required_files = ["spm.m", "spm_get_defaults.m", "spm_vol.m", "spm_read_vols.m"]
        missing = [
            req
            for req in required_files
            if not os.path.isfile(os.path.join(normalized, req))
        ]
        if missing:
            missing_details = (normalized, missing)
            continue
        resolved_spm = normalized
        break

    if resolved_spm:
        print(f"✓ SPM installation found: {resolved_spm}")
    else:
        print("ERROR: Unable to locate a valid SPM installation.")
        if missing_details:
            path, missing = missing_details
            print(f"       Checked {path} but missing files: {', '.join(missing)}")
        print(
            "       Set [SPM] path in config.ini, export SPM_PATH, or create spm_config.txt with the correct path."
        )
        overall_ok = False

    return overall_ok


def check_python_packages() -> bool:
    ok = True
    try:
        import numpy as np  # noqa: F401
    except Exception as exc:
        print(f"ERROR: numpy not importable: {exc}")
        ok = False
    else:
        print(f"✓ numpy {np.__version__} available")

    try:
        import nibabel as nb  # noqa: F401
    except Exception as exc:
        print(f"ERROR: nibabel not importable: {exc}")
        ok = False
    else:
        print(f"✓ nibabel {nb.__version__} available")

    try:
        import pandas as pd_check  # noqa: F401
    except Exception as exc:
        print(f"ERROR: pandas not importable: {exc}")
        ok = False
    else:
        print(f"✓ pandas {pd_check.__version__} available")

    try:
        import matplotlib  # noqa: F401
    except Exception:
        print("⚠️  matplotlib not available - thumbnails will be skipped")
    else:
        print(f"✓ matplotlib {matplotlib.__version__} available")

    return ok


def gather_expected_sessions(
    participants_file: str, session_col: str
) -> Dict[str, List[str]]:
    df = pd.read_csv(participants_file, sep="\t")
    if "participant_id" not in df.columns:
        raise ValueError("participants.tsv must contain 'participant_id' column")

    expected: Dict[str, List[str]] = {}
    if "nr_sessions" in df.columns:
        for _, row in df.iterrows():
            subject = row["participant_id"]
            if pd.isna(subject):
                continue
            nr_sessions = row["nr_sessions"]
            try:
                nr_sessions = int(nr_sessions)
            except Exception:
                continue
            if nr_sessions < 1:
                continue
            expected.setdefault(subject, []).extend(
                [str(i) for i in range(1, nr_sessions + 1)]
            )
    else:
        if session_col not in df.columns:
            raise ValueError(
                f"participants.tsv is scan-level but lacks session column '{session_col}'"
            )
        for _, row in df.iterrows():
            subject = row["participant_id"]
            session = row[session_col]
            if pd.isna(subject) or pd.isna(session):
                continue
            expected.setdefault(subject, []).append(str(session))

    normalized: Dict[str, List[str]] = {}
    for subject, sessions in expected.items():
        unique_sessions: List[str] = []
        seen: set[str] = set()
        for session in sessions:
            normalized_session = session
            if normalized_session in seen:
                continue
            seen.add(normalized_session)
            unique_sessions.append(normalized_session)
        if unique_sessions:
            normalized[subject] = unique_sessions
    return normalized


def check_xml_reports(cat12_dir: str, participants_file: str, session_col: str) -> bool:
    if pp is None:
        print("ERROR: Unable to import parse_participants helper.")
        return False

    try:
        expected = gather_expected_sessions(participants_file, session_col)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return False

    get_subject_dirs = getattr(pp, "_subject_data_dirs", None)
    if get_subject_dirs is None:
        print("ERROR: parse_participants is missing _subject_data_dirs helper")
        return False

    missing: List[Tuple[str, str]] = []
    subjects_with_data = {}
    for subject in expected:
        data_dirs = list(get_subject_dirs(cat12_dir, subject))
        subjects_with_data[subject] = bool(data_dirs)

    for subject, sessions in expected.items():
        if not subjects_with_data.get(subject):
            continue
        for session in sessions:
            xml_path = pp.find_xml_for_subject(cat12_dir, subject, session)
            if not xml_path:
                missing.append((subject, session))

    if missing:
        print("ERROR: Missing CAT12 XML reports for the following subject/sessions:")
        for subject, session in missing[:10]:
            print(f"  - {subject} ses-{session}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        print(
            "Please rerun CAT12 preprocessing or add the XML files before running the pipeline."
        )
        return False

    print("✓ CAT12 XML reports present for all expected sessions")
    return True


def check_tiv_presence(
    cat12_dir: str, participants_file: str, session_col: str
) -> bool:
    """Ensure TIV is available either in participants.tsv or extractable from CAT12 XMLs.

    The function looks for a `tiv` (or `vol_TIV`) column in the participants file.
    If absent, it will attempt to extract `vol_TIV` from each subject/session XML.
    Returns True if TIV is present for all expected subject/sessions, False otherwise.
    """
    if pp is None:
        print("ERROR: Unable to import parse_participants helper for TIV check.")
        return False

    # read participants to check for a tiv column
    try:
        df = pd.read_csv(participants_file, sep="\t")
    except Exception as exc:
        print(f"ERROR: Could not read participants.tsv for TIV check: {exc}")
        return False

    # detect if tiv column present (case-insensitive)
    tiv_col = None
    for col in df.columns:
        if col.lower() in ("tiv", "vol_tiv"):
            tiv_col = col
            break

    if tiv_col is not None:
        nonnull = df[tiv_col].notna().sum()
        if nonnull > 0:
            print(
                f"✓ TIV column '{tiv_col}' found in participants.tsv ({nonnull} non-empty entries)"
            )
            return True
        else:
            print(
                f"Note: TIV column '{tiv_col}' exists but contains only missing values; will try XML extraction"
            )

    # no usable tiv in TSV -> attempt XML extraction per expected session
    try:
        expected = gather_expected_sessions(participants_file, session_col)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return False

    # Determine which subjects actually have CAT12 data directories
    subjects_with_data = {}
    get_subject_dirs = getattr(pp, "_subject_data_dirs", None)
    if get_subject_dirs is None:
        # fallback: assume all expected have data (conservative)
        for subj in expected:
            subjects_with_data[subj] = True
    else:
        for subj in expected:
            subjects_with_data[subj] = bool(list(get_subject_dirs(cat12_dir, subj)))

    missing_tiv: List[Tuple[str, str]] = []
    resolve_alias = getattr(pp, "_resolve_xml_covariate", None)
    if resolve_alias is None:
        alias_map = getattr(pp, "XML_COVARIATE_ALIASES", {"tiv": "vol_TIV"})

        def _resolve(x: str) -> str:
            key = x.lower() if isinstance(x, str) else x
            return alias_map.get(key, x)

    else:
        _resolve = resolve_alias

    for subject, sessions in expected.items():
        # skip participants with no CAT12 folder (user requested behavior)
        if not subjects_with_data.get(subject, False):
            continue
        for session in sessions:
            xml_path = pp.find_xml_for_subject(cat12_dir, subject, session)
            if not xml_path:
                # subject has data dir but no xml -> treat as missing
                missing_tiv.append((subject, session))
                continue
            measure = _resolve("tiv")
            try:
                val = pp.extract_measure_from_xml(xml_path, measure)
            except Exception:
                val = None
            if val is None:
                missing_tiv.append((subject, session))

    if missing_tiv:
        print(
            "ERROR: TIV (vol_TIV) not found for the following subject/sessions (only subjects with CAT12 data are checked):"
        )
        for subj, ses in missing_tiv[:20]:
            print(f"  - {subj} ses-{ses}")
        if len(missing_tiv) > 20:
            print(f"  ... and {len(missing_tiv)-20} more")
        print(
            "Please add TIV to your participants.tsv or ensure CAT12 XMLs contain 'vol_TIV' entries for these subjects."
        )
        return False

    print("✓ TIV available for all expected subject/sessions via CAT12 XML extraction")
    return True


def check_covariates_presence(
    cat12_dir: str,
    participants_file: str,
    session_col: str,
    covariates: List[str],
    allow_missing: bool = False,
) -> bool:
    """Check that requested covariates are available in participants.tsv or extractable from CAT12 XMLs.

    - Ignores participants that do not have a CAT12 data folder.
    - For covariates not present in the TSV, attempts XML extraction for subjects with data.
    - If missing entries remain and allow_missing is False, returns False.
    """
    if pp is None:
        print("ERROR: Unable to import parse_participants helper for covariate check.")
        return False

    try:
        df = pd.read_csv(participants_file, sep="\t")
    except Exception as exc:
        print(f"ERROR: Could not read participants.tsv for covariate check: {exc}")
        return False

    try:
        expected = gather_expected_sessions(participants_file, session_col)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return False

    get_subject_dirs = getattr(pp, "_subject_data_dirs", None)
    subjects_with_data = {}
    if get_subject_dirs is None:
        for subj in expected:
            subjects_with_data[subj] = True
    else:
        for subj in expected:
            subjects_with_data[subj] = bool(list(get_subject_dirs(cat12_dir, subj)))

    overall_ok = True
    for cov in covariates:
        # present in TSV?
        if cov in df.columns and df[cov].notna().sum() > 0:
            print(
                f"✓ Covariate '{cov}' present in participants.tsv ({df[cov].notna().sum()} non-empty entries)"
            )
            continue

        # attempt XML extraction for subjects that have CAT12 data
        missing_entries: List[Tuple[str, str]] = []
        resolve_alias = getattr(pp, "_resolve_xml_covariate", None)
        if resolve_alias is None:
            alias_map = getattr(pp, "XML_COVARIATE_ALIASES", {})

            def _resolve_local(x):
                key = x.lower() if isinstance(x, str) else x
                return alias_map.get(key, x)

        else:
            _resolve_local = resolve_alias

        measure_name = _resolve_local(cov)
        for subj, sessions in expected.items():
            if not subjects_with_data.get(subj, False):
                # ignore participants with no CAT12 folder
                continue
            for ses in sessions:
                xml_path = pp.find_xml_for_subject(cat12_dir, subj, ses)
                if not xml_path:
                    missing_entries.append((subj, ses))
                    continue
                try:
                    val = pp.extract_measure_from_xml(xml_path, measure_name)
                except Exception:
                    val = None
                if val is None:
                    missing_entries.append((subj, ses))

        if missing_entries:
            if not allow_missing:
                print(
                    f"ERROR: Covariate '{cov}' missing for the following subject/sessions (only subjects with CAT12 data are checked):"
                )
                for subj, ses in missing_entries[:10]:
                    print(f"  - {subj} ses-{ses}")
                if len(missing_entries) > 10:
                    print(f"  ... and {len(missing_entries)-10} more")
                overall_ok = False
            else:
                print(
                    f"⚠ Covariate '{cov}' missing for {len(missing_entries)} subject/sessions; continuing because --allow-missing-covariates is set"
                )
        else:
            print(
                f"✓ Covariate '{cov}' available via CAT12 XML for all checked subjects"
            )

    return overall_ok


def check_cat12_dir(cat12_dir: str, smoothing: str, modality: str = "vbm") -> bool:
    candidates = [
        cat12_dir,
        os.path.join(cat12_dir, "data", "cat12"),
        os.path.join(cat12_dir, "cat12"),
    ]
    data_dir = None
    for candidate in candidates:
        if os.path.isdir(candidate):
            data_dir = candidate
            break
    if data_dir is None:
        print(f"ERROR: Expected CAT12 data folder not found. Tried: {candidates}")
        return False

    # Determine pattern based on modality
    if modality == "vbm":
        if smoothing:
            pattern = f"*s{smoothing}mwp1r*.nii*"
        else:
            pattern = "*mwp1r*.nii*"
        # VBM files are typically in 'mri' subfolder
        search_path = os.path.join(data_dir, "**", "mri", pattern)
    else:
        # Surface modalities (thickness, gyrification, etc.)
        # Pattern: s{smoothing}.mesh.{modality}.resampled*.gii
        # Location: 'surf' subfolder
        if smoothing:
            pattern = f"s{smoothing}.mesh.{modality}.resampled*.gii"
        else:
            pattern = f"*.mesh.{modality}.resampled*.gii"
        search_path = os.path.join(data_dir, "**", "surf", pattern)

    samples = glob.glob(search_path, recursive=True)
    
    if not samples:
        # Fallback search without specific subfolder if strict structure not found
        fallback_path = os.path.join(data_dir, "**", pattern)
        samples = glob.glob(fallback_path, recursive=True)

    if not samples:
        print(
            f"WARNING: No files found with pattern {pattern} under {data_dir} (modality: {modality})"
        )
        return False

    sample = samples[0]
    if len(samples) > 1:
        print(f"   Found {len(samples)} candidate files, showing first 5:")
        for s in samples[:5]:
            print(f"     - {s}")
    try:
        import nibabel as nb  # noqa: F401

        img = nb.load(sample)
        # For GIfTI (surface), header handling is different than NIfTI
        if sample.endswith('.gii'):
             print(f"✓ Sample GIfTI OK: {os.path.basename(sample)} (surface data)")
        else:
            hdr = img.header
            data_shape = hdr.get_data_shape()
            print(
                f"✓ Sample NIfTI OK: {os.path.basename(sample)} shape={data_shape} bytes={os.path.getsize(sample)}"
            )
    except Exception as exc:
        print(f"ERROR: Failed to read sample file {sample}: {exc}")
        return False

    return True


def find_and_copy_cat12_brainmask(cat12_dir: str) -> str | None:
    candidates: List[str] = []
    candidates.append(
        os.path.join(
            cat12_dir, "templates_MNI152NLin2009cAsym", "brainmask_GMtight.nii"
        )
    )
    for root, _, files in os.walk(cat12_dir):
        for fn in files:
            if fn.startswith("brainmask_GMtight") and fn.endswith(".nii"):
                candidates.append(os.path.join(root, fn))

    found: str | None = None
    for path in candidates:
        if os.path.exists(path):
            found = path
            break

    if found is None:
        matches = glob.glob(
            os.path.join(cat12_dir, "**", "brainmask_GMtight*.nii*"), recursive=True
        )
        if matches:
            found = matches[0]

    if not found:
        print("⚠️  brainmask_GMtight.nii not found under CAT12 dir; skipping auto-copy")
        return None

    if "," in found:
        found = found.split(",")[0]

    repo_utils = os.path.dirname(__file__)
    repo_root = os.path.dirname(repo_utils)
    templates_dir = os.path.join(repo_root, "templates")
    os.makedirs(templates_dir, exist_ok=True)
    dest = os.path.join(templates_dir, "brainmask_GMtight.nii")

    try:
        import shutil

        shutil.copy2(found, dest)
        print(f"✓ Copied CAT12 brainmask to repo templates: {dest}")
        return dest
    except Exception as exc:
        print(f"ERROR: Failed to copy brainmask from {found} -> {dest}: {exc}")
        return None


def check_participants_file(participants_file: str) -> bool:
    if not os.path.isfile(participants_file):
        print(f"ERROR: Participants file not found: {participants_file}")
        return False
    try:
        with open(participants_file, "r") as fh:
            lines = [line for line in fh if line.strip()]
        if len(lines) < 2:
            print(
                f"ERROR: Participants file appears empty or only header: {participants_file}"
            )
            return False
        print(f"✓ Participants file OK: {participants_file} (rows={len(lines)})")
        return True
    except Exception as exc:
        print(f"ERROR: Could not read participants file: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight checks for CAT12 pipeline")
    parser.add_argument(
        "--cat12-dir",
        required=True,
        help="Path to CAT12 data folder or repository root",
    )
    parser.add_argument("--participants", required=True, help="participants.tsv path")
    parser.add_argument(
        "--covariates",
        default="",
        help='Comma-separated covariates to check (e.g., "age,sex,tiv")',
    )
    parser.add_argument(
        "--allow-missing-covariates",
        action="store_true",
        help="Allow missing covariates (will continue and drop missing ones)",
    )
    parser.add_argument(
        "--mask",
        default=None,
        help="Optional path to a custom brain mask to use (will be copied to stats/templates/brainmask_GMtight.nii)",
    )
    parser.add_argument("--smoothing", default="", help="Smoothing mm (optional)")
    parser.add_argument(
        "--session-col",
        default="session",
        help="Column to read session IDs from when participants.tsv is scan-level",
    )
    parser.add_argument(
        "--modality",
        default="vbm",
        help="Analysis modality (vbm, thickness, gyrification, etc.)",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Skip MATLAB/SPM checks for standalone mode",
    )
    args = parser.parse_args()

    overall_ok = True

    print("\n=== Preflight: Python package checks ===")
    if not check_python_packages():
        overall_ok = False

    print("\n=== Preflight: MATLAB/SPM availability ===")
    if args.standalone:
        print("✓ Standalone mode enabled: Skipping MATLAB/SPM checks.")
    elif not check_matlab_and_spm():
        overall_ok = False

    print("\n=== Preflight: CAT12 files ===")
    if not check_cat12_dir(args.cat12_dir, args.smoothing, args.modality):
        overall_ok = False

    print("\n=== Preflight: Participants file ===")
    if not check_participants_file(args.participants):
        overall_ok = False

    print("\n=== Preflight: CAT12 XML reports ===")
    if not check_xml_reports(args.cat12_dir, args.participants, args.session_col):
        overall_ok = False

    # Check requested covariates (if any); default behavior retains TIV check when no covariates requested
    covs = [c.strip() for c in args.covariates.split(",") if c.strip()]
    if covs:
        print("\n=== Preflight: Requested covariates presence/availability ===")
        if not check_covariates_presence(
            args.cat12_dir,
            args.participants,
            args.session_col,
            covs,
            allow_missing=args.allow_missing_covariates,
        ):
            overall_ok = False
    else:
        print("\n=== Preflight: TIV presence (vol_TIV) ===")
        if not check_tiv_presence(args.cat12_dir, args.participants, args.session_col):
            overall_ok = False

    if not overall_ok:
        print("\n❌ Preflight checks failed. Fix above issues before continuing.")
        sys.exit(2)

    if args.mask:
        if find_and_copy_cat12_brainmask(args.mask):
            print("✓ Custom mask installed")

    repo_utils = os.path.dirname(__file__)
    repo_root = os.path.dirname(repo_utils)
    templates_dir = os.path.join(repo_root, "templates")
    dest = os.path.join(templates_dir, "brainmask_GMtight.nii")
    if not os.path.exists(dest):
        print("\nNotice: repo template mask not found at %s" % dest)
        copied = find_and_copy_cat12_brainmask(args.cat12_dir)
        if copied is None or not os.path.exists(dest):
            print("\nERROR: Canonical CAT12 brainmask not available.")
            print("To ensure future SPM jobs use the correct GM mask please either:")
            print(
                "  - Run this script again with --mask /path/to/brainmask_GMtight.nii"
            )
            print("  - Or copy your CAT12 brainmask into the repository at: %s" % dest)
            print(
                "\nThis pipeline enforces the use of a canonical GM-tight mask to avoid non-brain clusters in SPM Results."
            )
            sys.exit(3)

    print(
        "\n✅ Preflight successful: Python packages, CAT12 files, participants file, and XML reports appear OK"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
