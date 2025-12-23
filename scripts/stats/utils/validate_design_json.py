#!/usr/bin/env python3
import json
import os
import re
import sys
from collections import Counter


def validate_design(json_file):
    print(f"Validating design file: {json_file}")
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return False

    files = data.get('files', [])
    covariates = data.get('covariates', {})
    
    print(f"Total files: {len(files)}")
    
    # 1. Check Covariate Lengths
    print("\n--- Covariate Consistency ---")
    issues_found = False
    for cov_name, cov_values in covariates.items():
        if len(cov_values) != len(files):
            print(f"❌ ERROR: Covariate '{cov_name}' has {len(cov_values)} values, but there are {len(files)} files.")
            issues_found = True
        else:
            print(f"✓ Covariate '{cov_name}' length matches file count.")
            
    # 2. Check File Existence and Naming Consistency
    print("\n--- File Consistency ---")
    prefixes = []
    missing_files = []
    
    for i, file_entry in enumerate(files):
        path = file_entry.get('path', '')
        if not os.path.exists(path):
            missing_files.append(path)
        
        filename = os.path.basename(path)
        # Extract prefix (everything before 'sub-')
        match = re.match(r'^(.*?)sub-', filename)
        if match:
            prefix = match.group(1)
            prefixes.append(prefix)
        else:
            prefixes.append("unknown")

    if missing_files:
        print(f"❌ ERROR: {len(missing_files)} files are missing from disk.")
        for f in missing_files[:5]:
            print(f"  - {f}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files)-5} more")
        issues_found = True
    else:
        print("✓ All files exist on disk.")

    # Check prefixes
    prefix_counts = Counter(prefixes)
    print(f"Smoothing prefixes found: {dict(prefix_counts)}")
    
    if len(prefix_counts) > 1:
        print("❌ ERROR: Inconsistent smoothing prefixes detected!")
        print("  This means you are mixing files with different smoothing kernels (e.g., s6, s9, or none).")
        print("  All files must have the same prefix for valid statistical analysis.")
        issues_found = True
    elif len(prefix_counts) == 1:
        print(f"✓ Consistent smoothing prefix: '{list(prefix_counts.keys())[0]}'")
    else:
        print("⚠️ Warning: No prefixes detected (filenames might not start with standard CAT12 prefixes).")

    # 3. Check Factor Consistency (Basic)
    print("\n--- Factor Consistency ---")
    # Check if subjects have consistent sessions
    subjects = {}
    for f in files:
        sub = f.get('subject')
        ses = f.get('session')
        if sub not in subjects:
            subjects[sub] = []
        subjects[sub].append(ses)
    
    # Check if all subjects have the same number of sessions (balanced design check - optional but good)
    session_counts = [len(s) for s in subjects.values()]
    if len(set(session_counts)) > 1:
        print("⚠️ Warning: Unbalanced design detected.")
        print(f"  Subjects have varying numbers of sessions: {set(session_counts)}")
        # This might be intended (missing data), but worth noting.
    else:
        print(f"✓ Balanced design: All {len(subjects)} subjects have {session_counts[0]} sessions.")

    if issues_found:
        print("\n❌ VALIDATION FAILED: Critical issues found.")
        return False
    else:
        print("\n✓ VALIDATION PASSED: No critical issues found.")
        return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 validate_design_json.py <design.json>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    if not validate_design(json_file):
        sys.exit(1)
