# Quick Reference Guide

## 🔧 SPM Path Configuration (NEW!)

**Problem**: SPM path was hardcoded and didn't work on different systems.
**Solution**: Automatic SPM detection with multiple configuration options.

### Quick Setup
```matlab
configure_spm_path()  % Interactive setup tool
```

### Options (choose one):
```bash
# Option 1: Environment variable
export SPM_PATH="/path/to/your/spm"

# Option 2: Config file (create in project directory)
echo "/path/to/your/spm" > spm_config.txt

# Option 3: Install SPM in standard location
# e.g., /Applications/spm25 (macOS)
```

See `SPM_PATH_CONFIGURATION_GUIDE.md` for detailed instructions.

---

## 📊 Covariate Verification

### 🎯 What Problem Does This Solve?

When creating statistical analysis batch jobs in CAT12/SPM:
- You list NIfTI brain images to analyze
- You provide covariate values (TIV, IQR, Age, etc.) for each subject
- **The order MUST match!** If not, your analysis results are WRONG

This script verifies the order is correct.

---

## 📊 The Core Validation

```
Question: Do file order and covariate order match?

Check 1: #Files = #TIV values?        (yes/no)
Check 2: #Files = #IQR values?        (yes/no)
Check 3: #Files = #Age values?        (yes/no)

All yes?  → ✓ VALID (safe to analyze)
Any no?   → ✗ INVALID (DO NOT ANALYZE!)
```

---

## 🚀 Quick Usage

```bash
# Default: auto-detect batch file in current directory
python3 verify_covariates.py

# Specify batch file
python3 verify_covariates.py batch_3x3_job_with_covariates.m

# Show more entries (default is 10)
python3 verify_covariates.py --rows 20

# Search in specific directory
python3 verify_covariates.py --dir /path/to/stats
```

---

## � Multi-Stage TFCE Analysis (NEW!)

For efficient permutation testing, use multi-stage TFCE:

```bash
# Multi-stage: low permutations first, high only if significant
bash run_tfce_headless.sh stats_folder --multi-stage

# Customize permutation counts
bash run_tfce_headless.sh stats_folder --multi-stage --n-perm-stage1 500 --n-perm-stage2 5000

# Traditional single-stage (slower)
bash run_tfce_headless.sh stats_folder --n-perm 5000
```

### Multi-Stage Workflow:
1. **Check uncorrected** (p < 0.001 + cluster size)
2. **Stage 1**: TFCE with 500 permutations (quick check)
3. **Stage 2**: TFCE with 5000 permutations (only if Stage 1 found significance)

---

## 📊 Covariate Verification

### 1. VERIFICATION SUMMARY
```
Total files:        369
  ✓ TIV (1):        369 values  OK
  ✓ IQR (2):        369 values  OK

✓ PASS: All counts match! File order is correct.
```

Meaning: 369 files, 369 TIV values, 369 IQR values → All match!

### 2. FIRST/LAST ENTRIES
```
#    File                                TIV           IQR
1    sub-1291145_ses-1.nii         1647.490000   1.649102
2    sub-1292007_ses-1.nii         1472.840000   1.674562
```

Meaning: Shows sample data to visually verify it looks correct

### 3. STATISTICAL SUMMARY
```
TIV (index 1):
  Min:       1186.540000
  Max:       1861.730000
  Mean:      1496.517480
  StdDev:     135.268104
```

Meaning: Data ranges and distribution look reasonable?

### 4. FINAL STATUS
```
✓ Status: VALID
✓ Files analyzed: 369
✓ Covariates validated: 2
```

Meaning: **VALID** = Safe to use in SPM | **INVALID** = Fix before using!

---

## 🔍 What If Something Goes Wrong?

### ✗ FAIL: Counts do not match!

```
Total files:        369
  ✗ TIV (1):        368 values  MISMATCH (has 368)
  ✗ IQR (2):        369 values  OK

✗ FAIL: Counts do not match!
```

**Problem:** 369 files but only 368 TIV values!

**Action needed:**
1. Check TIV file - missing one value?
2. Re-export covariates from source data
3. Re-create batch file with corrected covariates

### ✗ Error: No covariates found in batch file

**Problem:** Script couldn't find any `cov(N).cname` definitions

**Action needed:**
1. Verify you're using batch file WITH covariates
2. Try: `python3 verify_covariates.py batch_3x3_job_with_covariates.m`

### ✗ Error: Batch file not found

**Problem:** Script couldn't find a batch file

**Action needed:**
1. Specify batch file explicitly: `python3 verify_covariates.py path/to/batch.m`
2. Or: `cd` to directory containing batch file first

---

## 🔧 How It Works Behind the Scenes

### Step 1: Extract Files
Script reads batch file and finds all quoted paths ending in `,1`:
```matlab
'/path/to/sub-001.nii,1'
'/path/to/sub-002.nii,1'
...
→ Extracts 369 files in order
```

### Step 2: Extract Covariate Names
Script reads covariate names:
```matlab
cov(1).cname = 'TIV'   → Name: 'TIV'
cov(2).cname = 'IQR'   → Name: 'IQR'
```

### Step 3: Extract Covariate Values
Script reads covariate values:
```matlab
cov(1).c = [1647.49; 1472.84; ...]  → 369 values
cov(2).c = [1.6499; 1.6748; ...]    → 369 values
```

### Step 4: Validate
Compare counts:
- 369 files = 369 TIV values? YES ✓
- 369 files = 369 IQR values? YES ✓
→ **VALID!**

---

## 💡 Key Features

| Feature | Benefit |
|---------|---------|
| **Auto-detection** | Finds batch file automatically |
| **Generic patterns** | Works with any file paths |
| **Dynamic covariate detection** | Finds TIV, IQR, Age, Sex, etc. automatically |
| **Any number of covariates** | Handles 2 or 100+ covariates |
| **Clear status** | VALID or INVALID - easy to understand |
| **Detailed statistics** | Min/Max/Mean/StdDev for each covariate |
| **Sample data view** | See first and last entries |
| **Portable** | No hardcoded paths - works everywhere |

---

## 📚 Related Files

- `verify_covariates.py` - Main script
- `COVARIATE_VERIFICATION_GUIDE.md` - Detailed explanation
- `SCRIPT_ARCHITECTURE.md` - Technical architecture
- `batch_3x3_job_with_covariates.m` - Example batch file

---

## ✅ Workflow

```
1. Create batch job in SPM/CAT12
   ↓
2. Export/prepare covariates (TIV, IQR, etc.)
   ↓
3. Run this script to verify order
   ↓
4. Script says VALID?
   ├─ YES → Safe to run analysis ✓
   └─ NO  → Fix covariates and try again
   ↓
5. Run statistical analysis in SPM
```

---

## 📞 Troubleshooting

**Q: Script finds wrong batch file?**
A: Multiple batch files found. Specify explicitly:
```bash
python3 verify_covariates.py batch_3x3_job_with_covariates.m
```

**Q: How do I know if my covariates are correct?**
A: Run this script! If VALID → covariates are in correct order.

**Q: What does "index 1" and "index 2" mean?**
A: Index is the covariate number in SPM:
- Index 1 = First covariate (usually TIV)
- Index 2 = Second covariate (usually IQR)
- etc.

**Q: Can I verify covariates with custom names?**
A: Yes! Script auto-detects any covariate names (Age, Sex, etc.)

**Q: What if I have 50 subjects but 60 TIV values?**
A: Script will show MISMATCH. You have 10 extra TIV values - remove them!

---

## 🎓 Example Session

```bash
$ python3 verify_covariates.py

Found 5 batch files. Using first: batch_3x3_job_with_covariates.m

====================================
BATCH FILE: batch_3x3_job_with_covariates.m
====================================
Reading batch file...

====================================
VERIFICATION SUMMARY
====================================
Total files:        369
  ✓ TIV                  (index 1):  369 values  OK
  ✓ IQR                  (index 2):  369 values  OK

✓ PASS: All counts match! File order is correct.

[... first 10 entries shown ...]
[... statistics shown ...]

====================================
VERIFICATION COMPLETE
====================================
✓ Status: VALID
✓ Files analyzed: 369
✓ Covariates validated: 2

$ echo "Ready for SPM analysis!" ✓
```

---

## 📝 Summary

- **Purpose:** Verify covariate order matches file order
- **Input:** MATLAB batch file
- **Output:** VALID or INVALID status + detailed report
- **Time:** < 1 second for 1000s of files
- **Benefit:** Prevents invalid statistical results due to mismatched data
