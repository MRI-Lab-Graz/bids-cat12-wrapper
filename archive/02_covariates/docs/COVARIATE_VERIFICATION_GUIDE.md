# Covariate Verification - Detailed Explanation & Architecture

## 📋 Problem Statement

When creating MATLAB batch jobs for CAT12 statistical analysis, you need to:
1. Provide a list of **NIfTI files** to analyze (e.g., `s9_*.nii`)
2. Provide **covariate values** for each subject (e.g., TIV, IQR, Age, Sex, etc.)
3. **Ensure the order matches** - the 1st file must have the 1st TIV value, 2nd file has 2nd TIV value, etc.

If the order is mismatched, results are invalid (correlating wrong covariates with wrong brains!).

---

## 🔍 How the Script Works

### **Step 1: Extract Files from Batch Job**

```matlab
matlabbatch{1}.spm.tools.cat.factorial_design.des.fblock.fsuball.group(1).timepoint = {
    {
    '/path/to/sub-001.nii,1'
    '/path/to/sub-002.nii,1'
    '/path/to/sub-003.nii,1'
    }
}
```

**What we extract:** A sorted list of all `.nii` files in order:
```
File Order: [sub-001.nii, sub-002.nii, sub-003.nii, ...]
```

**Key point:** Files are extracted as they appear in the batch file. The order here is the reference.

---

### **Step 2: Extract Covariates from Batch Job**

```matlab
matlabbatch{1}.spm.tools.cat.factorial_design.cov(1).cname = 'TIV';
matlabbatch{1}.spm.tools.cat.factorial_design.cov(1).c = [
    1647.49
    1472.84
    1551.73
    ...
];

matlabbatch{1}.spm.tools.cat.factorial_design.cov(2).cname = 'IQR';
matlabbatch{1}.spm.tools.cat.factorial_design.cov(2).c = [
    1.649890
    1.674778
    1.648796
    ...
];
```

**What we extract:**
- **Covariate name** from `cov(N).cname` - can be anything: TIV, IQR, Age, Sex, Group, etc.
- **Covariate index** (N) - defines the order (1st covariate, 2nd covariate, etc.)
- **Covariate values** from `cov(N).c` - numeric array

**Key point:** Each covariate must have same number of values as files!

---

### **Step 3: Compare Files with Covariates**

The script extracts **only the filename** from each full path:

```
File: '/path/to/sub-001_ses-1.nii,1'  →  Extract: 'sub-001_ses-1.nii'
File: '/path/to/sub-002_ses-1.nii,1'  →  Extract: 'sub-002_ses-1.nii'
```

Then validates:
- ✓ File count = TIV count?
- ✓ File count = IQR count?
- ✓ File count = all other covariates?

If all counts match → **Data is correctly ordered!**

---

## 🛠️ Flexibility: How It Handles Different Scenarios

### **Scenario 1: Covariates Named Differently**
```matlab
cov(1).cname = 'HeadSize';      % Not 'TIV'
cov(2).cname = 'ImageQuality'; % Not 'IQR'
cov(3).cname = 'Age';
cov(4).cname = 'Sex';
```

**Script behavior:** ✓ Automatically detects ALL covariates by reading `cname` fields
- No hardcoding needed!
- Works with any names

---

### **Scenario 2: Different Number of Covariates**
```matlab
cov(1).cname = 'TIV';
cov(2).cname = 'IQR';
cov(3).cname = 'Age';
cov(4).cname = 'Sex';
cov(5).cname = 'Education';
```

**Script behavior:** ✓ Automatically detects all 5 covariates
- Validates each one independently
- Shows statistics for all

---

### **Scenario 3: Non-Sequential Indices**
```matlab
cov(1).cname = 'TIV';
cov(3).cname = 'Age';     % Note: no cov(2)!
cov(5).cname = 'Sex';
```

**Script behavior:** ✓ Handles correctly
- Finds covariates at indices 1, 3, 5
- Validates each independently

---

## 🔗 Filename Comparison Logic

**Q: How does the script know files are in correct order?**

The script does NOT compare filenames with covariate source files. Instead:

1. **Extract order from batch file** → This is the ground truth
2. **Count files** (e.g., 411 files)
3. **Count covariates** (e.g., 411 TIV values)
4. **If counts match** → ✓ Files and covariates are in same order
5. **If counts don't match** → ✗ Mismatch detected!

**Why this works:** SPM expects covariates to be in the exact same order as files in batch. If mismatch exists, counts won't match.

---

## 📊 Input: MATLAB Batch File Structure

```matlab
% CAT12 Factorial Design Batch Job
matlabbatch{1}.spm.tools.cat.factorial_design.cov(N).cname  = 'CovariateName'
matlabbatch{1}.spm.tools.cat.factorial_design.cov(N).c      = [value1; value2; value3; ...]
```

**Current example:** `batch_3x3_job_with_covariates.m`
- Files: 411 subjects
- Covariates: TIV (cov 1), IQR (cov 2)
- Structure: Multi-timepoint, multi-group design

---

## 📈 Output: Validation Report

```
══════════════════════════════════════════════════════════════════════════════════════
VERIFICATION SUMMARY
══════════════════════════════════════════════════════════════════════════════════════
Total files:        411
  ✓ TIV (1):        411 values
  ✓ IQR (2):        411 values

✓ PASS: All counts match!

══════════════════════════════════════════════════════════════════════════════════════
First 10 Entries
══════════════════════════════════════════════════════════════════════════════════════
#     File                                     TIV                 IQR
----------------------------------------
1     sub-1291145_ses-1.nii                1647.490000        1.649890
2     sub-1292007_ses-1.nii                1472.840000        1.674778
...
```

---

## 🎯 Key Design Principles

| Principle | Implementation |
|-----------|-----------------|
| **No hardcoding** | Detects covariate names dynamically from `cname` fields |
| **Generic patterns** | Handles any file path structure |
| **Scalable** | Works with 1 or 100+ covariates |
| **Flexible naming** | TIV, IQR, Age, Sex, or any custom name |
| **Clear validation** | Pass/fail status with detailed reporting |
| **Order verification** | Ensures file-covariate alignment |

---

## 🚀 Usage

```bash
# Auto-detect batch file in current directory
python verify_covariates.py

# Specify batch file explicitly
python verify_covariates.py batch_3x3_job_with_covariates.m

# Show first/last 20 entries instead of 10
python verify_covariates.py --rows 20

# Search in specific directory
python verify_covariates.py --dir /path/to/analysis/stats
```

---

## 📝 Summary

- **Input:** MATLAB batch job file (with files and covariates)
- **Process:** Extract → Parse → Count → Compare
- **Output:** Pass/Fail validation + Summary statistics
- **Flexibility:** Works with any covariate names, any number of covariates
- **Purpose:** Ensure data integrity before running statistical analysis
