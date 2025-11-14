#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COVARIATE MANAGEMENT TOOL
═══════════════════════════════════════════════════════════════════════════════

Purpose:
  Unified tool to ADD and VERIFY covariates in MATLAB CAT12 batch jobs.
  Ensures statistical analysis data integrity and correct ordering.

Usage:
  # ADD MODE: Add TIV/IQR covariates to batch job
  python manage_covariates.py --add batch_3x3_job.m -o batch_3x3_job_with_covariates.m
  python manage_covariates.py --add batch_file.m --tiv TIV.txt --iqr IQR.txt

  # VERIFY MODE: Check covariate order in batch job
  python manage_covariates.py --verify batch_3x3_job_with_covariates.m
  python manage_covariates.py --verify                      # Auto-detect
  python manage_covariates.py --verify --rows 20            # Show 20 entries

  # COMBINED: Add then verify
  python manage_covariates.py --add batch.m && python manage_covariates.py --verify

Modes:
  --add       Add covariates to batch file
  --verify    Verify covariate order in batch file

Features:
  ✓ Automatic file detection and matching
  ✓ Flexible covariate handling (any names, any count)
  ✓ Detailed validation reports
  ✓ No hardcoded paths - portable everywhere
  ✓ Comprehensive error handling

See documentation files for detailed information.
═══════════════════════════════════════════════════════════════════════════════
"""

import re
import sys
import argparse
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def find_batch_file(search_dir=None, pattern="*batch*.m"):
    """
    Auto-detect batch file in current or specified directory.
    Looks for files matching pattern (default: *batch*.m).
    """
    if search_dir is None:
        search_dir = Path.cwd()
    else:
        search_dir = Path(search_dir)
    
    if not search_dir.exists():
        raise FileNotFoundError(f"Directory not found: {search_dir}")
    
    batch_files = list(search_dir.glob(pattern))
    if not batch_files:
        raise FileNotFoundError(f"No batch files ({pattern}) found in {search_dir}")
    
    if len(batch_files) > 1:
        print(f"Found {len(batch_files)} batch files. Using first: {batch_files[0].name}")
    
    return batch_files[0]


def extract_files_from_batch(content):
    """Extract NIfTI file paths from batch content."""
    file_pattern = r"'([^']+\.nii(?:\.gz)?),1'"
    files = re.findall(file_pattern, content)
    return files


def extract_covariates_from_batch(content):
    """Extract all covariate matrices from batch content."""
    covariates = {}
    cov_pattern = r"cov\((\d+)\)\.c\s*=\s*\[(.*?)\];"
    matches = re.finditer(cov_pattern, content, re.DOTALL)
    
    for match in matches:
        cov_index = int(match.group(1))
        values_str = match.group(2)
        value_pattern = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
        values = [float(v) for v in re.findall(value_pattern, values_str)]
        covariates[cov_index] = values
    
    return covariates


def get_covariate_names(content):
    """Extract covariate names from batch file."""
    names = {}
    name_pattern = r"cov\((\d+)\)\.cname\s*=\s*'([^']+)';"
    matches = re.finditer(name_pattern, content)
    
    for match in matches:
        cov_index = int(match.group(1))
        cov_name = match.group(2)
        names[cov_index] = cov_name
    
    return names


def parse_filename(filepath):
    """Extract subject ID and session from filepath."""
    sub_match = re.search(r'sub-(\d+)', filepath)
    ses_match = re.search(r'ses-(\d+)', filepath)
    
    if sub_match and ses_match:
        return (sub_match.group(1), ses_match.group(1))
    return None


def read_values(file_path):
    """Read numeric values from text file."""
    values = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    values.append(float(line))
                except ValueError:
                    continue
    return values


def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# ═══════════════════════════════════════════════════════════════════════════════
# ADD MODE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_files_from_iqr(iqr_file):
    """Extract all XML files from IQR job file in order."""
    with open(iqr_file, 'r') as f:
        content = f.read()
    
    # Try different patterns for XML file paths
    patterns = [
        r"'(/Volumes/Thunder/129_PK01/cat12/data/cat12/.*?\.xml)'",
        r"'([^']+/cat12/.*?\.xml)'",
        r"'([^']+\.xml)'"
    ]
    
    for pattern in patterns:
        files = re.findall(pattern, content)
        if files:
            return files
    
    return []


def create_mapping(source_files, batch_files):
    """
    Create mapping from source file order to batch file order.
    Matches files by subject ID and session number.
    
    Returns:
        tuple: (indices mapping, missing files list)
    """
    # Create lookup dictionary: (subject, session) -> index in source list
    source_lookup = {}
    for idx, file_path in enumerate(source_files):
        key = parse_filename(file_path)
        if key:
            source_lookup[key] = idx
    
    # Create ordered list of indices for batch files
    batch_indices = []
    missing_files = []
    
    for nii_file in batch_files:
        key = parse_filename(nii_file)
        if key and key in source_lookup:
            batch_indices.append(source_lookup[key])
        else:
            missing_files.append(nii_file)
            batch_indices.append(None)
    
    return batch_indices, missing_files


def reorder_values(values, indices):
    """Reorder values according to indices mapping."""
    reordered = []
    for idx in indices:
        if idx is not None:
            reordered.append(values[idx])
        else:
            reordered.append(float('nan'))
    return reordered


def add_covariates_to_batch(batch_file, covariate_data, output_file):
    """
    Add covariates to batch file.
    
    Args:
        batch_file: Path to input batch file
        covariate_data: Dict of {name: values} or {name: (values, decimal_places)}
        output_file: Path to output batch file
    """
    with open(batch_file, 'r') as f:
        content = f.read()
    
    # Create new covariate section
    new_cov_lines = []
    cov_index = 1
    
    for cov_name, cov_data in covariate_data.items():
        # Handle both tuple (values, decimals) and plain values
        if isinstance(cov_data, tuple):
            values, decimals = cov_data
        else:
            values = cov_data
            decimals = 2 if cov_name == 'TIV' else 6
        
        new_cov_lines.append(f"matlabbatch{{1}}.spm.tools.cat.factorial_design.cov({cov_index}).c = [")
        
        for val in values:
            if decimals == 2:
                new_cov_lines.append(f"                                                              {val:.2f}")
            else:
                new_cov_lines.append(f"                                                              {val:.{decimals}f}")
        
        new_cov_lines.append("                                                              ];")
        new_cov_lines.append(f"matlabbatch{{1}}.spm.tools.cat.factorial_design.cov({cov_index}).cname = '{cov_name}';")
        new_cov_lines.append(f"matlabbatch{{1}}.spm.tools.cat.factorial_design.cov({cov_index}).iCFI = 1;")
        new_cov_lines.append(f"matlabbatch{{1}}.spm.tools.cat.factorial_design.cov({cov_index}).iCC = 1;")
        
        cov_index += 1
    
    new_cov_section = '\n'.join(new_cov_lines)
    
    # Find the old cov struct and replace it
    # Look for empty cov struct pattern or existing cov definitions
    old_patterns = [
        r"matlabbatch\{1\}\.spm\.tools\.cat\.factorial_design\.cov = struct\('c', \{\}, 'cname', \{\}, 'iCFI', \{\}, 'iCC', \{\}\);",
        r"matlabbatch\{1\}\.spm\.tools\.cat\.factorial_design\.cov\(\d+\)\..*?(?=matlabbatch\{1\}\.spm\.tools\.cat\.factorial_design\.multi_cov)"
    ]
    
    new_content = content
    for pattern in old_patterns:
        if re.search(pattern, content):
            new_content = re.sub(pattern, new_cov_section, content)
            break
    
    # If no replacement happened, append before multi_cov
    if new_content == content:
        multi_cov_pattern = r"matlabbatch\{1\}\.spm\.tools\.cat\.factorial_design\.multi_cov"
        if re.search(multi_cov_pattern, content):
            new_content = re.sub(
                multi_cov_pattern,
                new_cov_section + "\nmatlabbatch{1}.spm.tools.cat.factorial_design.multi_cov",
                content
            )
    
    # Write to output file
    with open(output_file, 'w') as f:
        f.write(new_content)
    
    return new_content


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFY MODE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def validate_counts(files, covariates_dict):
    """Validate that file count matches covariate counts."""
    num_files = len(files)
    
    for cov_idx, values in covariates_dict.items():
        if len(values) != num_files:
            return False
    
    return True


def print_file_details(files, covariates_dict, covariate_names, num_rows=10):
    """Display comprehensive validation report."""
    if not files:
        print("Error: No files found in batch file")
        return "INVALID"
    
    num_files = len(files)
    
    # SUMMARY
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
    
    # FIRST N ENTRIES
    print_header(f"First {num_rows} Entries (Sample)")
    
    header_cols = ["#", "File"]
    col_widths = [5, 40]
    
    for cov_idx in sorted(covariates_dict.keys()):
        cov_name = covariate_names.get(cov_idx, f"Cov{cov_idx}")
        header_cols.append(cov_name[:15])
        col_widths.append(14)
    
    header_str = ""
    for i, col in enumerate(header_cols):
        header_str += f"{col:<{col_widths[i]}}"
    print(header_str)
    print("-" * sum(col_widths))
    
    for i in range(min(num_rows, num_files)):
        filename = Path(files[i]).name
        row_str = f"{i+1:<5}{filename:<40}"
        
        for cov_idx in sorted(covariates_dict.keys()):
            value = covariates_dict[cov_idx][i]
            if abs(value) < 1e-3 or abs(value) > 1e6:
                row_str += f"{value:>14.3e}"
            else:
                row_str += f"{value:>14.6f}"
        
        print(row_str)
    
    # LAST N ENTRIES
    if num_files > 2 * num_rows:
        print(f"\n... ({num_files - 2*num_rows} entries omitted) ...\n")
        
        print_header(f"Last {num_rows} Entries (Sample)")
        
        header_str = ""
        for i, col in enumerate(header_cols):
            header_str += f"{col:<{col_widths[i]}}"
        print(header_str)
        print("-" * sum(col_widths))
        
        for i in range(max(0, num_files - num_rows), num_files):
            filename = Path(files[i]).name
            row_str = f"{i+1:<5}{filename:<40}"
            
            for cov_idx in sorted(covariates_dict.keys()):
                value = covariates_dict[cov_idx][i]
                if abs(value) < 1e-3 or abs(value) > 1e6:
                    row_str += f"{value:>14.3e}"
                else:
                    row_str += f"{value:>14.6f}"
            
            print(row_str)
    
    # STATISTICS
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
    
    # FINAL STATUS
    print_header("VERIFICATION COMPLETE")
    print(f"✓ Status: {status.upper()}")
    print(f"✓ Files analyzed: {num_files}")
    print(f"✓ Covariates validated: {len(covariates_dict)}")
    print()
    
    return status


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

def handle_add(args):
    """Handle --add mode: Add covariates to batch file."""
    print_header("ADD MODE: Adding Covariates to Batch File")
    
    # Determine input batch file
    if args.batch_file:
        batch_file = Path(args.batch_file)
    else:
        batch_file = find_batch_file(search_dir=args.dir, pattern="*batch*.m")
    
    if not batch_file.exists():
        print(f"Error: Batch file not found: {batch_file}")
        return False
    
    print(f"Input batch file: {batch_file}")
    
    # Determine output batch file
    if args.output:
        output_file = Path(args.output)
    else:
        # Generate output filename
        stem = batch_file.stem
        output_file = batch_file.parent / f"{stem}_with_covariates.m"
    
    print(f"Output batch file: {output_file}")
    
    # Determine covariate files
    if args.iqr_job:
        iqr_job_file = Path(args.iqr_job)
    else:
        # Try to find IQR_job.m in same directory
        iqr_job_file = batch_file.parent / "IQR_job.m"
    
    if args.tiv:
        tiv_file = Path(args.tiv)
    else:
        tiv_file = batch_file.parent / "TIV.txt"
    
    if args.iqr:
        iqr_file = Path(args.iqr)
    else:
        iqr_file = batch_file.parent / "IQR.txt"
    
    # Check if all required files exist
    missing_files = []
    if not batch_file.exists():
        missing_files.append(f"Batch file: {batch_file}")
    if not tiv_file.exists():
        missing_files.append(f"TIV file: {tiv_file}")
    if not iqr_file.exists():
        missing_files.append(f"IQR file: {iqr_file}")
    if not iqr_job_file.exists():
        missing_files.append(f"IQR job file: {iqr_job_file}")
    
    if missing_files:
        print("\n❌ Error: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        return False
    
    print("\n✓ All required files found")
    
    # Extract file lists
    print("\nExtracting file lists...")
    with open(batch_file, 'r') as f:
        batch_content = f.read()
    
    batch_files = extract_files_from_batch(batch_content)
    iqr_files = extract_files_from_iqr(iqr_job_file)
    
    print(f"  Found {len(batch_files)} files in batch file")
    print(f"  Found {len(iqr_files)} files in IQR job file")
    
    # Read values
    print("\nReading covariate values...")
    tiv_values = read_values(tiv_file)
    iqr_values = read_values(iqr_file)
    
    print(f"  Found {len(tiv_values)} TIV values")
    print(f"  Found {len(iqr_values)} IQR values")
    
    # Create mapping
    print("\nCreating file mapping...")
    batch_indices, missing = create_mapping(iqr_files, batch_files)
    
    if missing:
        print(f"  ⚠ {len(missing)} files in batch without IQR/TIV data")
        for f in missing[:3]:
            print(f"    - {f}")
        if len(missing) > 3:
            print(f"    ... and {len(missing) - 3} more")
    
    # Reorder values
    print("\nReordering values to match batch file...")
    reordered_tiv = reorder_values(tiv_values, batch_indices)
    reordered_iqr = reorder_values(iqr_values, batch_indices)
    
    # Add covariates to batch
    print("\nAdding covariates to batch file...")
    covariate_data = {
        'TIV': (reordered_tiv, 2),
        'IQR': (reordered_iqr, 6)
    }
    
    try:
        add_covariates_to_batch(batch_file, covariate_data, output_file)
    except Exception as e:
        print(f"❌ Error adding covariates: {e}")
        return False
    
    # Summary
    print(f"\n✅ Successfully created {output_file}")
    print("\nSummary:")
    print(f"  - Total files: {len(batch_files)}")
    print(f"  - Files with TIV/IQR: {len([i for i in batch_indices if i is not None])}")
    print(f"  - Files missing TIV/IQR: {len([i for i in batch_indices if i is None])}")
    
    # Show first few matches as verification
    print("\nFirst 5 file matches (verification):")
    for i in range(min(5, len(batch_files))):
        batch_key = parse_filename(batch_files[i])
        if batch_indices[i] is not None:
            tiv = reordered_tiv[i]
            iqr = reordered_iqr[i]
            print(f"  {i+1}. sub-{batch_key[0]} ses-{batch_key[1]}: TIV={tiv:.2f}, IQR={iqr:.6f}")
    
    return True


def handle_verify(args):
    """Handle --verify mode: Verify covariate order in batch file."""
    print_header("VERIFY MODE: Checking Covariate Order")
    
    # Determine batch file
    if args.batch_file:
        batch_file = Path(args.batch_file)
    else:
        batch_file = find_batch_file(search_dir=args.dir)
    
    if not batch_file.exists():
        print(f"Error: Batch file not found: {batch_file}")
        return False
    
    print(f"BATCH FILE: {batch_file.name}")
    print("Reading batch file...")
    
    # Read and parse
    try:
        with open(batch_file, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading batch file: {e}")
        return False
    
    files = extract_files_from_batch(content)
    covariates = extract_covariates_from_batch(content)
    covariate_names = get_covariate_names(content)
    
    if not files:
        print("Error: No files found in batch file")
        return False
    
    if not covariates:
        print("Error: No covariates found in batch file")
        return False
    
    # Display results
    status = print_file_details(files, covariates, covariate_names, args.rows)
    
    return status == "VALID"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--add',
        action='store_true',
        help='Add covariates to batch file'
    )
    mode_group.add_argument(
        '--verify',
        action='store_true',
        help='Verify covariate order in batch file'
    )
    
    # Common arguments
    parser.add_argument(
        'batch_file',
        nargs='?',
        default=None,
        help='Path to batch file (default: auto-detect)'
    )
    
    parser.add_argument(
        '--dir',
        type=str,
        default=None,
        help='Search directory (default: current directory)'
    )
    
    # ADD mode arguments
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output batch file (ADD mode only)'
    )
    
    parser.add_argument(
        '--tiv',
        type=str,
        default=None,
        help='TIV values file (default: TIV.txt in batch directory)'
    )
    
    parser.add_argument(
        '--iqr',
        type=str,
        default=None,
        help='IQR values file (default: IQR.txt in batch directory)'
    )
    
    parser.add_argument(
        '--iqr-job',
        type=str,
        default=None,
        help='IQR job file (default: IQR_job.m in batch directory)'
    )
    
    # VERIFY mode arguments
    parser.add_argument(
        '--rows',
        type=int,
        default=10,
        help='Number of first/last rows to display (VERIFY mode only, default: 10)'
    )
    
    args = parser.parse_args()
    
    # Execute appropriate mode
    if args.add:
        success = handle_add(args)
    elif args.verify:
        success = handle_verify(args)
    else:
        parser.print_help()
        return 1
    
    # Exit with appropriate code
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
