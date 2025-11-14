# Manage Covariates - Quick Usage Guide

## What It Does

`manage_covariates.py` is your one-stop tool for covariate management:

- **ADD**: Take a batch file without covariates + covariate values files → Create batch file with covariates properly ordered
- **VERIFY**: Check that covariates in a batch file are correctly ordered and match file count

## Typical Workflow

```bash
# Step 1: Add covariates to your batch file
python3 manage_covariates.py --add batch_3x3_job.m \
  -o batch_3x3_job_with_covariates.m

# Step 2: Verify it worked correctly
python3 manage_covariates.py --verify batch_3x3_job_with_covariates.m

# Step 3: If VALID ✓, use it in SPM analysis!
```

## Mode: ADD

### What It Needs

```
Input:
  ├─ batch_3x3_job.m                    (your MATLAB batch job)
  ├─ TIV.txt                            (one TIV value per line)
  ├─ IQR.txt                            (one IQR value per line)
  └─ IQR_job.m                          (reference for file order)

Output:
  └─ batch_3x3_job_with_covariates.m    (new batch file with covariates)
```

### Basic Command

```bash
python3 manage_covariates.py --add batch_3x3_job.m \
  -o batch_3x3_job_with_covariates.m
```

### Auto-Detect (Easiest)

If you're in the directory with all files:

```bash
# Auto-detects: batch_*.m, TIV.txt, IQR.txt, IQR_job.m
python3 manage_covariates.py --add batch_3x3_job.m
```

### Custom Paths

```bash
python3 manage_covariates.py --add batch.m \
  -o output.m \
  --tiv /path/to/tiv_values.txt \
  --iqr /path/to/iqr_values.txt \
  --iqr-job /path/to/iqr_job.m
```

### Output

```
✓ All required files found
✓ Found 369 files in batch file
✓ Found 411 files in IQR job file
✓ Found 411 TIV values
✓ Found 411 IQR values

✅ Successfully created batch_3x3_job_with_covariates.m

Summary:
  - Total files: 369
  - Files with TIV/IQR: 369
  - Files missing TIV/IQR: 0
```

## Mode: VERIFY

### What It Checks

1. Do we have the same number of files and covariate values?
2. Are the covariates correctly ordered?
3. What are the statistics for each covariate?

### Basic Command

```bash
python3 manage_covariates.py --verify batch_3x3_job_with_covariates.m
```

### Auto-Detect

```bash
# Finds first *batch*.m file in current directory
python3 manage_covariates.py --verify
```

### Show More Entries

```bash
# Default: shows 10 first + 10 last entries
# Show 20 instead:
python3 manage_covariates.py --verify --rows 20
```

### Output Examples

**✅ VALID (Good to proceed with SPM)**

```
==========================================================================================
VERIFICATION SUMMARY
==========================================================================================
Total files:        369
  ✓ TIV                  (index 1):  369 values  OK
  ✓ IQR                  (index 2):  369 values  OK

✓ PASS: All counts match! File order is correct.
...
✓ Status: VALID
✓ Files analyzed: 369
✓ Covariates validated: 2
```

**❌ INVALID (DO NOT proceed with SPM!)**

```
==========================================================================================
VERIFICATION SUMMARY
==========================================================================================
Total files:        369
  ✗ TIV                  (index 1):  361 values  MISMATCH (has 361)
  ✗ IQR                  (index 2):  361 values  MISMATCH (has 361)

✗ FAIL: Counts do not match! Data order may be INVALID for analysis!
...
✓ Status: INVALID
```

## Combined Workflow (Recommended)

Add AND verify in one command:

```bash
python3 manage_covariates.py --add batch.m -o batch_new.m && \
  python3 manage_covariates.py --verify batch_new.m
```

If it ends with ✓ Status: VALID, you're ready for SPM analysis!

## Real-World Examples

### Scenario 1: First Time Setup

```bash
# You just got a new batch file without covariates
# You have TIV.txt and IQR.txt files
# You have the IQR_job.m that was used to generate TIV/IQR values

# Step 1: Add covariates
python3 manage_covariates.py --add batch_3x3_job.m \
  -o batch_3x3_job_with_covariates.m

# Step 2: Verify
python3 manage_covariates.py --verify batch_3x3_job_with_covariates.m

# If VALID: Ready for SPM!
```

### Scenario 2: Verify Existing File

```bash
# Someone gave you a batch file with covariates
# You want to make sure it's not corrupted

python3 manage_covariates.py --verify their_batch_file.m

# If VALID: Use it
# If INVALID: Ask for it to be regenerated
```

### Scenario 3: Different Covariates

```bash
# Your batch file has Age, Sex, Group instead of TIV, IQR
# The script AUTOMATICALLY detects these!

python3 manage_covariates.py --verify batch_with_age_sex_group.m

# Output will show:
#   ✓ Age (index 1): 369 values OK
#   ✓ Sex (index 2): 369 values OK
#   ✓ Group (index 3): 369 values OK
```

### Scenario 4: Large Batch

```bash
# Your batch has 2000 files
# Default shows 10 first + 10 last entries
# That might not be enough to spot patterns

python3 manage_covariates.py --verify large_batch.m --rows 50

# Now shows 50 first + 50 last entries
# Much easier to spot any issues
```

## Common Tasks

### Task: Check file count

```bash
python3 manage_covariates.py --verify batch.m | head -10
```

### Task: Get statistics only

```bash
python3 manage_covariates.py --verify batch.m | grep -A 20 "STATISTICAL SUMMARY"
```

### Task: Save verification report

```bash
python3 manage_covariates.py --verify batch.m > verification_report.txt
```

### Task: Quick validation (exit code only)

```bash
python3 manage_covariates.py --verify batch.m > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Batch file is valid"
else
    echo "✗ Batch file is invalid"
fi
```

## Troubleshooting

### "No batch files found"

```bash
# Make sure you're in right directory
pwd

# Or provide explicit path
python3 manage_covariates.py --verify /full/path/batch.m
```

### "Missing required files" (ADD mode)

```bash
# ADD needs: batch file + TIV.txt + IQR.txt + IQR_job.m
# Check they exist:
ls TIV.txt IQR.txt IQR_job.m batch_3x3_job.m

# If missing, provide paths:
python3 manage_covariates.py --add batch.m \
  --tiv path/to/tiv.txt \
  --iqr path/to/iqr.txt \
  --iqr-job path/to/iqr_job.m
```

### "INVALID" status

```bash
# File count doesn't match covariate count
# This is a critical issue!
# DO NOT use this batch file with SPM

# Investigate:
python3 manage_covariates.py --verify batch.m --rows 30

# Look at the summary - which covariate has wrong count?
# Then investigate that covariate file
```

## Key Points to Remember

✅ **Always verify before SPM analysis** - It takes 2 seconds and saves hours of wasted computation

✅ **"VALID" status is required** - If not VALID, investigate before proceeding

✅ **Script automatically detects covariate names** - Works with TIV/IQR, Age/Sex, or custom names

✅ **File matching is intelligent** - Matches by subject ID and session, not just line order

✅ **Exit code 0 = success, 1 = failure** - Good for scripting and automation

## Next Steps

- Read `MANAGE_COVARIATES_README.md` for complete documentation
- Use the tool!
- Done.

That's it!
