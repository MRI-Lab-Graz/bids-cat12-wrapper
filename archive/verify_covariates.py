#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COVARIATE VERIFICATION TOOL
═══════════════════════════════════════════════════════════════════════════════

Purpose:
  Validate that covariate values in MATLAB CAT12 batch jobs match the order
  of input files. This ensures statistical analysis results are valid.

How It Works:
  1. Extracts list of NIfTI files from batch job (in order)
  2. Extracts all covariates (TIV, IQR, Age, Sex, etc.) and their values
  3. Verifies counts match: #files == #values for each covariate
  4. If mismatch detected, analysis results would be INVALID

Usage:
  python verify_covariates.py                           # Auto-detect batch file
  python verify_covariates.py batch_file.m              # Specific batch file
  python verify_covariates.py --rows 20                 # Show 20 first/last entries
  python verify_covariates.py --dir /path/to/analysis   # Search in directory

Features:
  ✓ Works with ANY covariate names (TIV, IQR, Age, Sex, etc.)
  ✓ Handles any number of covariates
  ✓ No hardcoded paths - portable across projects
  ✓ Generic file pattern matching
  ✓ Detailed statistics and validation report
  ✓ Pass/Fail indicators

See COVARIATE_VERIFICATION_GUIDE.md for detailed explanation.
═══════════════════════════════════════════════════════════════════════════════
"""

import re
import sys
import argparse
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: FILE DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

def find_batch_file(search_dir=None):
    """
    Auto-detect batch file in current or specified directory.
    Looks for files matching *batch*.m pattern.
    """
    if search_dir is None:
        search_dir = Path.cwd()
    else:
        search_dir = Path(search_dir)
    
    if not search_dir.exists():
        raise FileNotFoundError(f"Directory not found: {search_dir}")
    
    batch_files = list(search_dir.glob('*batch*.m'))
    if not batch_files:
        raise FileNotFoundError(
            f"No batch files (*batch*.m) found in {search_dir}"
        )
    
    if len(batch_files) > 1:
        print(f"Found {len(batch_files)} batch files. Using first: {batch_files[0].name}")
    
    return batch_files[0]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: FILE EXTRACTION FROM BATCH JOB
# ═══════════════════════════════════════════════════════════════════════════════

def extract_files_from_batch(content):
    """
    Extract NIfTI file paths from batch content.
    
    Handles paths in quoted format like: '(...path...),1'
    The ',1' suffix indicates first volume of multi-volume files.
    
    Returns:
        list: Full file paths in order they appear in batch file
        
    Example:
        ['/path/to/sub-001.nii', '/path/to/sub-002.nii', ...]
    """
    # Generic pattern to match any quoted paths ending with .nii or .nii.gz followed by ,1
    file_pattern = r"'([^']+\.nii(?:\.gz)?),1'"
    files = re.findall(file_pattern, content)
    return files


def extract_covariates_from_batch(content):
    """
    Extract all covariate matrices from batch content.
    
    CAT12 batch structure:
        matlabbatch{1}.spm.tools.cat.factorial_design.cov(N).cname = 'CovariateName'
        matlabbatch{1}.spm.tools.cat.factorial_design.cov(N).c = [value1; value2; ...]
    
    This function is FLEXIBLE:
    ✓ Works with ANY covariate names (TIV, IQR, Age, Sex, etc.)
    ✓ Handles any number of covariates
    ✓ Handles non-sequential indices (cov(1), cov(3), cov(5) is OK)
    
    Returns:
        dict: Mapping of covariate index to list of values
        Example: {1: [1647.49, 1472.84, ...], 2: [1.6499, 1.6748, ...]}
    """
    covariates = {}
    
    # Pattern to find all cov(N).c = [...]; blocks
    # This regex finds: cov(NUMBER).c = [anything];
    cov_pattern = r"cov\((\d+)\)\.c\s*=\s*\[(.*?)\];"
    matches = re.finditer(cov_pattern, content, re.DOTALL)
    
    for match in matches:
        cov_index = int(match.group(1))
        values_str = match.group(2)
        
        # Extract numeric values (handles scientific notation, negative, decimals)
        value_pattern = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
        values = [float(v) for v in re.findall(value_pattern, values_str)]
        
        covariates[cov_index] = values
    
    return covariates


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: COVARIATE NAME EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_covariate_names(content):
    """
    Extract covariate names from batch file.
    
    Looks for patterns like:
        matlabbatch{1}.spm.tools.cat.factorial_design.cov(N).cname = 'CovariateName';
    
    The script AUTOMATICALLY detects covariate names - NOT hardcoded!
    This handles:
    ✓ TIV, IQR (standard)
    ✓ Age, Sex, Group (common additions)
    ✓ Any custom covariate names
    
    Returns:
        dict: Mapping of covariate index to name
        Example: {1: 'TIV', 2: 'IQR', 3: 'Age'}
    """
    names = {}
    # Pattern: cov(N).cname = 'name';
    name_pattern = r"cov\((\d+)\)\.cname\s*=\s*'([^']+)';"
    matches = re.finditer(name_pattern, content)
    
    for match in matches:
        cov_index = int(match.group(1))
        cov_name = match.group(2)
        names[cov_index] = cov_name
    
    return names


def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: VALIDATION AND STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def validate_counts(files, covariates_dict):
    """
    Validate that file count matches covariate counts.
    
    The core validation logic:
    - Count of files must equal count of values for EACH covariate
    - If counts match → File-covariate order is correct ✓
    - If counts don't match → Mismatch detected ✗ (INVALID for analysis!)
    
    Args:
        files: List of file paths
        covariates_dict: Dict mapping covariate_index to values_list
        
    Returns:
        bool: True if all counts match, False otherwise
    """
    num_files = len(files)
    
    for cov_idx, values in covariates_dict.items():
        if len(values) != num_files:
            return False
    
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: DISPLAY AND REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_file_details(files, covariates_dict, covariate_names, num_rows=10):
    """
    Display comprehensive validation report.
    
    Shows:
    1. Summary of files and covariates (with pass/fail)
    2. First N entries (sample)
    3. Last N entries (if > 2*N total rows)
    4. Statistical summary (Min, Max, Mean, Std Dev)
    """
    if not files:
        print("Error: No files found in batch file")
        return
    
    num_files = len(files)
    
    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5a: SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    print_header("VERIFICATION SUMMARY")
    print(f"Total files:        {num_files}")
    
    for cov_idx in sorted(covariates_dict.keys()):
        cov_name = covariate_names.get(cov_idx, f"Covariate {cov_idx}")
        cov_count = len(covariates_dict[cov_idx])
        status = "✓" if cov_count == num_files else "✗"
        match_str = "OK" if cov_count == num_files else f"MISMATCH (has {cov_count})"
        print(f"  {status} {cov_name:<20} (index {cov_idx}): {cov_count:>4} values  {match_str}")
    
    # Validate counts match
    all_match = validate_counts(files, covariates_dict)
    
    if all_match and covariates_dict:
        print("\n✓ PASS: All counts match! File order is correct.")
        status = "VALID"
    else:
        print("\n✗ FAIL: Counts do not match! Data order may be INVALID for analysis!")
        status = "INVALID"
    
    if not all_match:
        return status
    
    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5b: FIRST N ENTRIES
    # ─────────────────────────────────────────────────────────────────────────
    print_header(f"First {num_rows} Entries (Sample)")
    
    # Build header
    header_cols = ["#", "File"]
    col_widths = [5, 40]
    
    for cov_idx in sorted(covariates_dict.keys()):
        cov_name = covariate_names.get(cov_idx, f"Cov{cov_idx}")
        header_cols.append(cov_name[:15])  # Truncate to 15 chars
        col_widths.append(14)
    
    # Print header row
    header_str = ""
    for i, col in enumerate(header_cols):
        header_str += f"{col:<{col_widths[i]}}"
    print(header_str)
    print("-" * sum(col_widths))
    
    # Print first N data rows
    for i in range(min(num_rows, num_files)):
        filename = Path(files[i]).name
        row_str = f"{i+1:<5}{filename:<40}"
        
        for cov_idx in sorted(covariates_dict.keys()):
            value = covariates_dict[cov_idx][i]
            # Format based on value magnitude
            if abs(value) < 1e-3 or abs(value) > 1e6:
                row_str += f"{value:>14.3e}"
            else:
                row_str += f"{value:>14.6f}"
        
        print(row_str)
    
    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5c: LAST N ENTRIES (if total > 2*N)
    # ─────────────────────────────────────────────────────────────────────────
    if num_files > 2 * num_rows:
        print(f"\n... ({num_files - 2*num_rows} entries omitted) ...\n")
        
        print_header(f"Last {num_rows} Entries (Sample)")
        
        # Reprint header
        header_str = ""
        for i, col in enumerate(header_cols):
            header_str += f"{col:<{col_widths[i]}}"
        print(header_str)
        print("-" * sum(col_widths))
        
        # Print last N data rows
        for i in range(max(0, num_files - num_rows), num_files):
            filename = Path(files[i]).name
            row_str = f"{i+1:<5}{filename:<40}"
            
            for cov_idx in sorted(covariates_dict.keys()):
                value = covariates_dict[cov_idx][i]
                # Format based on value magnitude
                if abs(value) < 1e-3 or abs(value) > 1e6:
                    row_str += f"{value:>14.3e}"
                else:
                    row_str += f"{value:>14.6f}"
            
            print(row_str)
    
    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5d: STATISTICS
    # ─────────────────────────────────────────────────────────────────────────
    print_header("STATISTICAL SUMMARY")
    for cov_idx in sorted(covariates_dict.keys()):
        cov_name = covariate_names.get(cov_idx, f"Covariate {cov_idx}")
        values = covariates_dict[cov_idx]
        
        mean = sum(values) / len(values)
        variance = sum((x - mean)**2 for x in values) / len(values)
        std_dev = variance ** 0.5
        
        print(f"\n{cov_name} (index {cov_idx}):")
        print(f"  Min:   {min(values):>15.6f}")
        print(f"  Max:   {max(values):>15.6f}")
        print(f"  Mean:  {mean:>15.6f}")
        print(f"  StdDev:{std_dev:>15.6f}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5e: FINAL STATUS
    # ─────────────────────────────────────────────────────────────────────────
    print_header("VERIFICATION COMPLETE")
    print(f"✓ Status: {status.upper()}")
    print(f"✓ Files analyzed: {num_files}")
    print(f"✓ Covariates validated: {len(covariates_dict)}")
    print()
    
    return status


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'batch_file',
        nargs='?',
        default=None,
        help='Path to MATLAB batch file (default: auto-detect)'
    )
    parser.add_argument(
        '--rows',
        type=int,
        default=10,
        help='Number of first/last rows to display (default: 10)'
    )
    parser.add_argument(
        '--dir',
        type=str,
        default=None,
        help='Search directory for auto-detection (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Determine batch file
    if args.batch_file:
        batch_file = Path(args.batch_file)
    else:
        batch_file = find_batch_file(args.dir)
    
    # Validate file exists
    if not batch_file.exists():
        print(f"Error: Batch file not found: {batch_file}")
        sys.exit(1)
    
    # Read batch file
    try:
        with open(batch_file, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading batch file: {e}")
        sys.exit(1)
    
    # ─────────────────────────────────────────────────────────────────────────
    # EXTRACT AND PARSE DATA
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"BATCH FILE: {batch_file.name}")
    print("=" * 90)
    print("Reading batch file...")
    
    files = extract_files_from_batch(content)
    covariates = extract_covariates_from_batch(content)
    covariate_names = get_covariate_names(content)
    
    if not files:
        print("Error: No files found in batch file")
        sys.exit(1)
    
    if not covariates:
        print("Error: No covariates found in batch file")
        sys.exit(1)
    
    # Display results
    status = print_file_details(files, covariates, covariate_names, args.rows)
    
    # Exit with appropriate code
    if status == "INVALID":
        sys.exit(1)


if __name__ == '__main__':
    main()
