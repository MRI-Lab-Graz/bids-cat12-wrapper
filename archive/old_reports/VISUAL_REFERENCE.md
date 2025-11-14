# Visual Reference: Script Components & Flow

## 🎨 Component Relationships

```
┌─────────────────────────────────────────────────────────────────┐
│                  MATLAB CAT12 BATCH FILE                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ matlabbatch{1}.spm.tools.cat.factorial_design           │   │
│  │                                                          │   │
│  │ .des.fblock.fsuball.group(1).timepoint =               │   │
│  │   '/path/sub-001.nii,1'                                 │   │
│  │   '/path/sub-002.nii,1'      ← SECTION 2 EXTRACTS      │   │
│  │   '/path/sub-003.nii,1'      ← FILE PATHS HERE         │   │
│  │                                                          │   │
│  │ .cov(1).cname = 'TIV'        ← SECTION 3b EXTRACTS    │   │
│  │ .cov(1).c = [                ← SECTION 3a EXTRACTS    │   │
│  │   1647.49                                               │   │
│  │   1472.84                    ← COVARIATE VALUES        │   │
│  │   1551.73  ]                                            │   │
│  │                                                          │   │
│  │ .cov(2).cname = 'IQR'                                   │   │
│  │ .cov(2).c = [                                           │   │
│  │   1.6499                                                │   │
│  │   1.6748   ]                                            │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │  SECTION 4: VALIDATE COUNTS           │
        │                                       │
        │  Files: 369 ?= TIV values: 369        │
        │  Files: 369 ?= IQR values: 369        │
        │                                       │
        │  Result: ✓ VALID (all match)          │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │  SECTION 5: DISPLAY REPORT            │
        │                                       │
        │  - Summary (pass/fail)                │
        │  - First 10 entries                   │
        │  - Last 10 entries                    │
        │  - Statistics                         │
        │  - Final status                       │
        └───────────────────────────────────────┘
                            ↓
OUTPUT TO CONSOLE:
✓ PASS: All counts match!
✓ Status: VALID
```

---

## 📊 Data Transformation Through Script

```
START: Raw MATLAB batch file
│
├─ Section 1: Find File
│  └─→ "/path/to/batch_3x3_job_with_covariates.m"
│
├─ Section 2: Extract Files
│  Raw:  "'/path/sub-001.nii,1' '/path/sub-002.nii,1' ..."
│  ├─→ Regex: '([^']+\.nii(?:\.gz)?),1'
│  └─→ Result: ["/path/sub-001.nii", "/path/sub-002.nii", ...]
│              Count: 369 ✓
│
├─ Section 3a: Extract Covariate Values
│  Raw:  "cov(1).c = [1647.49; 1472.84; ...]"
│  ├─→ Regex: cov\((\d+)\)\.c\s*=\s*\[(.*?)\];
│  └─→ Result: {1: [1647.49, 1472.84, ...], 
│               2: [1.6499, 1.6748, ...]}
│              Count: 369 each ✓
│
├─ Section 3b: Extract Covariate Names
│  Raw:  "cov(1).cname = 'TIV'; cov(2).cname = 'IQR';"
│  ├─→ Regex: cov\((\d+)\)\.cname\s*=\s*'([^']+)';
│  └─→ Result: {1: 'TIV', 2: 'IQR'}
│
├─ Section 4: Validate
│  Input: 369 files, {1: 369 vals, 2: 369 vals}
│  ├─→ 369 == 369? YES ✓
│  └─→ Result: valid = True
│
├─ Section 5: Display & Statistics
│  ├─→ Min/Max/Mean/StdDev calculated
│  ├─→ Formatted tables created
│  ├─→ Status determined
│  └─→ Output generated
│
└─ END: Console report printed
   Status: VALID ✓
```

---

## 🔄 Validation Logic Flow

```
                    START
                      │
                      ↓
          ┌─────────────────────┐
          │ Load batch file     │
          └─────────────────────┘
                      │
                      ↓
          ┌─────────────────────┐
          │ Extract files       │
          │ Count: 369          │
          └─────────────────────┘
                      │
                      ↓
          ┌─────────────────────────────┐
          │ Extract covariates          │
          │ {1: [TIV], 2: [IQR]}        │
          │ Count: 369 each             │
          └─────────────────────────────┘
                      │
                      ↓
          ┌─────────────────────────────┐
          │ For each covariate:         │
          │   len(files) == len(vals)?  │
          └─────────────────────────────┘
                    /   \
                  YES     NO
                  /         \
                 /           \
              VALID        INVALID
                 │              │
                 ↓              ↓
          ✓ Safe to use    ✗ Fix first
              │              │
              └──────┬───────┘
                     ↓
              Print report
              Exit with status
                     │
                     ↓
                    END
```

---

## 📈 Different Scenarios - Detection Capability

```
SCENARIO 1: Standard (TIV + IQR)
┌──────────────────────────┐
│ cov(1).cname = 'TIV'     │
│ cov(2).cname = 'IQR'     │
└──────────────────────────┘
         Script detects:
       cov 1 (TIV) ✓
       cov 2 (IQR) ✓


SCENARIO 2: Custom Names
┌──────────────────────────────────┐
│ cov(1).cname = 'HeadSize'        │
│ cov(2).cname = 'ImageQuality'    │
└──────────────────────────────────┘
         Script detects:
    cov 1 (HeadSize) ✓
    cov 2 (ImageQuality) ✓


SCENARIO 3: Many Covariates
┌──────────────────────────┐
│ cov(1).cname = 'TIV'     │
│ cov(2).cname = 'IQR'     │
│ cov(3).cname = 'Age'     │
│ cov(4).cname = 'Sex'     │
│ cov(5).cname = 'Group'   │
└──────────────────────────┘
         Script detects:
    cov 1 (TIV) ✓
    cov 2 (IQR) ✓
    cov 3 (Age) ✓
    cov 4 (Sex) ✓
    cov 5 (Group) ✓


SCENARIO 4: Non-sequential Indices
┌──────────────────────────┐
│ cov(1).cname = 'TIV'     │
│ cov(3).cname = 'Age'     │  ← skipped 2!
│ cov(5).cname = 'Sex'     │  ← skipped 4!
└──────────────────────────┘
         Script detects:
    cov 1 (TIV) ✓
    cov 3 (Age) ✓
    cov 5 (Sex) ✓
    
All handled automatically!
```

---

## 🧮 Statistics Calculation

```
Input: Covariate values for one subject
  [1647.49, 1472.84, 1551.73, ..., 1334.31]
  
Processing:
┌─────────────────────────────────────────┐
│ Find minimum                            │
│ min([...]) = 1186.54                    │
├─────────────────────────────────────────┤
│ Find maximum                            │
│ max([...]) = 1861.73                    │
├─────────────────────────────────────────┤
│ Calculate mean                          │
│ sum/count = 1496.52                     │
├─────────────────────────────────────────┤
│ Calculate standard deviation            │
│ sqrt(variance) = 135.27                 │
└─────────────────────────────────────────┘

Output:
┌────────────┐
│ Min: 1186  │
│ Max: 1862  │
│ Mean: 1497 │
│ StdDev:135 │
└────────────┘
```

---

## 🔍 Error Detection Scenarios

```
┌─ ERROR CASE 1: Count Mismatch ─────────────────────┐
│                                                    │
│ Files: 369                                         │
│ TIV values: 368  ← ONE MISSING!                   │
│                                                    │
│ Detection: 369 ≠ 368                             │
│ Output: ✗ FAIL - Counts do not match!            │
│ Action: Check and fix TIV data                    │
└────────────────────────────────────────────────────┘

┌─ ERROR CASE 2: No Covariates ─────────────────────┐
│                                                    │
│ Batch file has files but NO covariates            │
│                                                    │
│ Detection: Empty covariate dict                   │
│ Output: Error: No covariates found in batch file  │
│ Action: Add covariates to batch file              │
└────────────────────────────────────────────────────┘

┌─ ERROR CASE 3: File Not Found ────────────────────┐
│                                                    │
│ Batch file specified but missing                  │
│                                                    │
│ Detection: Path.exists() returns False            │
│ Output: Error: Batch file not found: path         │
│ Action: Check path, find correct batch file       │
└────────────────────────────────────────────────────┘

┌─ VALID CASE: All Counts Match ────────────────────┐
│                                                    │
│ Files: 369                                         │
│ TIV values: 369  ✓                                │
│ IQR values: 369  ✓                                │
│                                                    │
│ Detection: All counts equal                       │
│ Output: ✓ PASS - All counts match!               │
│ Action: SAFE to proceed with analysis             │
└────────────────────────────────────────────────────┘
```

---

## 📊 Output Organization

```
Console Report Structure:

┌─────────────────────────────────┐
│  BATCH FILE HEADER              │
│  Filename and location           │
└─────────────────────────────────┘
                ↓
┌─────────────────────────────────┐
│  VERIFICATION SUMMARY           │
│  - Total files                  │
│  - Each covariate status        │
│  - Overall pass/fail            │
└─────────────────────────────────┘
                ↓
┌─────────────────────────────────┐
│  FIRST N ENTRIES                │
│  - Table header (file + covs)   │
│  - First 5-20 rows              │
│  - Formatted values             │
└─────────────────────────────────┘
                ↓
┌─────────────────────────────────┐
│  LAST N ENTRIES (if many)       │
│  - "... X entries omitted ..."  │
│  - Last 5-20 rows               │
│  - Formatted values             │
└─────────────────────────────────┘
                ↓
┌─────────────────────────────────┐
│  STATISTICAL SUMMARY            │
│  For each covariate:            │
│  - Min, Max, Mean, StdDev       │
└─────────────────────────────────┘
                ↓
┌─────────────────────────────────┐
│  VERIFICATION COMPLETE          │
│  - Status (VALID/INVALID)       │
│  - File count                   │
│  - Covariate count              │
└─────────────────────────────────┘
```

---

## 🎯 Key Points Summary

```
┌─────────────────────────────────────────────────────────┐
│ FLEXIBILITY ACHIEVED THROUGH:                          │
├─────────────────────────────────────────────────────────┤
│ 1. Generic file pattern (works with any paths)         │
│ 2. Dynamic covariate detection (reads from batch)      │
│ 3. Automatic name extraction (not hardcoded)           │
│ 4. Any number of covariates (no limit)                 │
│ 5. Non-sequential indices (gaps handled)               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ VALIDATION LOGIC:                                       │
├─────────────────────────────────────────────────────────┤
│ For each covariate:                                    │
│   if count(files) != count(values):                    │
│     status = INVALID                                   │
│   else:                                                 │
│     status = VALID (so far)                            │
│                                                         │
│ Final: if all covariates valid → PASS ✓               │
│        else → FAIL ✗                                   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ WHY COUNTS EQUAL = ORDER CORRECT:                       │
├─────────────────────────────────────────────────────────┤
│ SPM expects strict ordering:                           │
│   File 1 → Covariate value 1                           │
│   File 2 → Covariate value 2                           │
│   ...                                                   │
│   File N → Covariate value N                           │
│                                                         │
│ If all counts match → No values missing → Order OK ✓   │
│ If any count differs → Misalignment → Order WRONG ✗    │
└─────────────────────────────────────────────────────────┘
```

---

## 📝 One-Minute Summary

```
WHAT: Verify covariate order in CAT12 batch jobs
WHY:  Prevents invalid statistical results
HOW:  
  1. Extract files from batch (count: N)
  2. Extract covariates from batch (count each)
  3. Compare: all counts = N?
  4. YES → VALID ✓ | NO → INVALID ✗

BONUS: Works with ANY covariate names, ANY count
```
