# Script Structure - Visual Guide

## 📐 File Organization

```
verify_covariates.py
│
├─ HEADER & DOCUMENTATION (lines 1-50)
│  ├─ Purpose description
│  ├─ Usage examples
│  ├─ Features list
│  └─ Import statements
│
├─ SECTION 1: FILE DISCOVERY (lines ~55-80)
│  └─ find_batch_file()
│     ├─ Auto-detect batch files
│     ├─ Error handling
│     └─ Return: Path object
│
├─ SECTION 2: FILE EXTRACTION (lines ~85-105)
│  └─ extract_files_from_batch(content)
│     ├─ Regex pattern matching
│     ├─ Extract NIfTI paths
│     └─ Return: List of file paths
│
├─ SECTION 3: COVARIATE EXTRACTION (lines ~110-145)
│  ├─ extract_covariates_from_batch(content)
│  │  ├─ Regex for cov(N).c = [...]
│  │  ├─ Parse numeric values
│  │  └─ Return: Dict {index: [values]}
│  │
│  └─ get_covariate_names(content)
│     ├─ Regex for cov(N).cname = '...'
│     ├─ Parse covariate names
│     └─ Return: Dict {index: name}
│
├─ SECTION 4: VALIDATION (lines ~150-170)
│  └─ validate_counts(files, covariates_dict)
│     ├─ Count files
│     ├─ Count covariate values
│     ├─ Compare counts
│     └─ Return: bool (all_match)
│
├─ SECTION 5: DISPLAY & REPORTING (lines ~175-300)
│  ├─ print_header(title)
│  │  └─ Format section separators
│  │
│  └─ print_file_details(files, covariates_dict, names, num_rows)
│     ├─ 5a: Summary section
│     │    ├─ Total files
│     │    ├─ Each covariate status
│     │    └─ Pass/Fail verdict
│     │
│     ├─ 5b: First N entries
│     │    ├─ Build table header
│     │    ├─ Print first N rows
│     │    └─ Format numbers
│     │
│     ├─ 5c: Last N entries (if > 2N)
│     │    └─ Same as 5b for tail
│     │
│     ├─ 5d: Statistics
│     │    ├─ For each covariate:
│     │    │  ├─ Min
│     │    │  ├─ Max
│     │    │  ├─ Mean
│     │    │  └─ StdDev
│     │    └─ Calculate on-the-fly
│     │
│     └─ 5e: Final status
│            ├─ VALID/INVALID
│            ├─ File count
│            └─ Covariate count
│
└─ MAIN EXECUTION (lines ~305-end)
   └─ main()
      ├─ Parse arguments
      ├─ Find/load batch file
      ├─ Extract data (call sections 1-3)
      ├─ Validate data (call section 4)
      ├─ Display report (call section 5)
      └─ Exit with status code
```

---

## 🔄 Data Flow

```
INPUT
  ↓
  ├─ Command: python3 verify_covariates.py [batch_file]
  ↓
main()
  ├─ Parse arguments (--rows, --dir, batch_file)
  ├─ Find batch file (auto-detect or explicit)
  ├─ Read file content
  └─→ process data
       ↓
   ┌───────────────────────────────────────┐
   │   SECTION 2: EXTRACT FILES            │
   │   Input: raw MATLAB text              │
   │   Regex: '(...).nii,1'                │
   │   Output: [file1, file2, ...]         │
   └───────────────────────────────────────┘
       ↓
   ┌───────────────────────────────────────┐
   │   SECTION 3: EXTRACT COVARIATES       │
   │   Input: raw MATLAB text              │
   │   Parse:                              │
   │   - cov(N).cname → names dict         │
   │   - cov(N).c → values dict            │
   │   Output: {1: [v,v,...], 2: [...]}    │
   └───────────────────────────────────────┘
       ↓
   ┌───────────────────────────────────────┐
   │   SECTION 4: VALIDATE                 │
   │   Compare: len(files) vs len(covs)    │
   │   Result: all_match (bool)            │
   └───────────────────────────────────────┘
       ↓
   ┌───────────────────────────────────────┐
   │   SECTION 5: DISPLAY REPORT           │
   │   - Summary section                   │
   │   - First N entries                   │
   │   - Last N entries                    │
   │   - Statistics                        │
   │   - Final status                      │
   └───────────────────────────────────────┘
       ↓
OUTPUT
  ├─ Console report (validation results)
  ├─ Exit code: 0 (VALID) or 1 (INVALID)
  └─ Status message
```

---

## 🎯 Function Signatures

```python
# SECTION 1
find_batch_file(search_dir: str | None) → Path

# SECTION 2
extract_files_from_batch(content: str) → List[str]

# SECTION 3
extract_covariates_from_batch(content: str) → Dict[int, List[float]]
get_covariate_names(content: str) → Dict[int, str]

# SECTION 4
validate_counts(files: List[str], covariates_dict: Dict) → bool

# SECTION 5
print_header(title: str) → None
print_file_details(
    files: List[str],
    covariates_dict: Dict[int, List[float]],
    covariate_names: Dict[int, str],
    num_rows: int = 10
) → str (status: "VALID" or "INVALID")

# MAIN
main() → None
```

---

## 🔗 Dependencies Between Functions

```
main()
  ├─ calls: find_batch_file()
  ├─ calls: extract_files_from_batch()
  ├─ calls: extract_covariates_from_batch()
  ├─ calls: get_covariate_names()
  ├─ calls: validate_counts()  (indirectly via print_file_details)
  └─ calls: print_file_details()
      ├─ calls: print_header()
      ├─ calls: validate_counts()
      └─ (performs calculations internally)
```

---

## 📊 Processing Steps in main()

```python
def main():
    # Step 1: Parse command-line arguments
    parser = argparse.ArgumentParser(...)
    args = parser.parse_args()
    
    # Step 2: Determine batch file (auto-detect or explicit)
    if args.batch_file:
        batch_file = Path(args.batch_file)
    else:
        batch_file = find_batch_file(args.dir)
    
    # Step 3: Validate file exists
    if not batch_file.exists():
        sys.exit(1)
    
    # Step 4: Read file content
    with open(batch_file, 'r') as f:
        content = f.read()
    
    # Step 5: Extract data
    files = extract_files_from_batch(content)
    covariates = extract_covariates_from_batch(content)
    covariate_names = get_covariate_names(content)
    
    # Step 6: Validate extraction
    if not files or not covariates:
        sys.exit(1)
    
    # Step 7: Display report
    status = print_file_details(
        files,
        covariates,
        covariate_names,
        args.rows
    )
    
    # Step 8: Exit with appropriate code
    if status == "INVALID":
        sys.exit(1)
```

---

## 🧩 Key Regex Patterns

```python
# SECTION 2: Extract file paths
pattern = r"'([^']+\.nii(?:\.gz)?),1'"
example: '/path/to/file.nii,1'
         '/path/to/file.nii.gz,1'

# SECTION 3a: Extract covariate values
pattern = r"cov\((\d+)\)\.c\s*=\s*\[(.*?)\];"
example: matlabbatch{1}.spm.tools.cat.factorial_design.cov(1).c = [
             1647.49
             1472.84
         ];

# SECTION 3b: Extract covariate names
pattern = r"cov\((\d+)\)\.cname\s*=\s*'([^']+)';"
example: matlabbatch{1}.spm.tools.cat.factorial_design.cov(1).cname = 'TIV';

# Value extraction (within covariate array)
pattern = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
matches: 1647.49, -123.45, 1.5e-6, .999
```

---

## 💾 Data Structures

### After Section 2:
```python
files = [
    '/path/to/sub-001.nii',
    '/path/to/sub-002.nii',
    ...
]
# Length: 369
```

### After Section 3a:
```python
covariates_dict = {
    1: [1647.49, 1472.84, 1551.73, ...],     # TIV
    2: [1.6499, 1.6748, 1.6488, ...]        # IQR
}
```

### After Section 3b:
```python
covariate_names = {
    1: 'TIV',
    2: 'IQR'
}
```

### Section 4 produces:
```python
valid = True  # or False if counts don't match
```

### Section 5 produces:
```python
status = "VALID"  # or "INVALID"
```

---

## ⚡ Complexity Analysis

| Component | Time | Space |
|-----------|------|-------|
| File I/O | O(n) | O(n) |
| Regex extraction | O(n) | O(m) |
| Validation | O(m) | O(1) |
| Statistics | O(m) | O(1) |
| Display | O(m) | O(m) |
| **Total** | **O(n)** | **O(max(n,m))** |

Where:
- n = file size (batch file)
- m = number of files + covariates

Typical: < 1 second for batches with 1000s of files

---

## 🔐 Error Handling

```
main()
  ├─ find_batch_file()
  │  └─ raises: FileNotFoundError
  │
  ├─ Path.exists()
  │  └─ sys.exit(1) if not found
  │
  ├─ open(batch_file)
  │  └─ except: print error, sys.exit(1)
  │
  ├─ extract_*() functions
  │  └─ return empty if pattern not found
  │
  ├─ validate extraction
  │  └─ sys.exit(1) if empty
  │
  └─ exit code
     └─ 0 = VALID / 1 = INVALID or error
```

---

## 🎨 Output Structure

```
═══════════════════════════════════
BATCH FILE: name.m
═══════════════════════════════════

═══════════════════════════════════
VERIFICATION SUMMARY
═══════════════════════════════════
Total files:        N
  ✓ CovName (idx):  M values  OK/MISMATCH

✓/✗ PASS/FAIL message

═══════════════════════════════════
First N Entries (Sample)
═══════════════════════════════════
#    File    Cov1    Cov2
--  ----------  ------  ------
1    file.nii  val1    val2
...

[... omitted entries ...]

═══════════════════════════════════
Last N Entries (Sample)
═══════════════════════════════════
[... last entries ...]

═══════════════════════════════════
STATISTICAL SUMMARY
═══════════════════════════════════
CovName (idx):
  Min:    value
  Max:    value
  Mean:   value
  StdDev: value

═══════════════════════════════════
VERIFICATION COMPLETE
═══════════════════════════════════
✓ Status: VALID/INVALID
✓ Files analyzed: N
✓ Covariates validated: M
```

---

## 📝 Summary

The script is organized into 5 clear, independent sections:
1. **Discovery** - Find batch file
2. **Extraction** - Parse files from batch
3. **Extraction** - Parse covariates from batch
4. **Validation** - Compare counts
5. **Reporting** - Display results

Each section is independently testable and modular.
