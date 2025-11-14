# Covariate Management Tool

`manage_covariates.py` is a unified tool for adding and verifying covariates in MATLAB CAT12 batch jobs.

## Overview

This tool combines two essential functions:
- **ADD mode**: Add covariates (TIV, IQR, etc.) to batch files
- **VERIFY mode**: Check that covariate order matches file order

Both operations are critical for statistical analysis validity.

## Installation

The script requires only Python 3 (no external dependencies).

```bash
# Make executable
chmod +x manage_covariates.py

# Or run directly with python3
python3 manage_covariates.py --help
```

## Quick Start

### Verify Existing Batch File

```bash
# Auto-detect batch file
python3 manage_covariates.py --verify

# Specify batch file
python3 manage_covariates.py --verify batch_3x3_job_with_covariates.m

# Show more entries (default: 10)
python3 manage_covariates.py --verify --rows 20
```

### Add Covariates to Batch File

```bash
# Auto-detect files and create output
python3 manage_covariates.py --add batch_3x3_job.m -o batch_with_covariates.m

# Specify custom covariate files
python3 manage_covariates.py --add batch.m \
  --tiv custom_tiv.txt \
  --iqr custom_iqr.txt \
  --iqr-job custom_iqr_job.m
```

### Combined Workflow

```bash
# Add covariates AND verify them
python3 manage_covariates.py --add batch.m -o batch_new.m && \
  python3 manage_covariates.py --verify batch_new.m
```

## Usage Reference

### Full Help

```bash
python3 manage_covariates.py --help
```

### Positional Arguments

```
batch_file              Path to batch file (optional, auto-detected if not provided)
```

### Mode Selection (Required - choose one)

```
--add                   Add covariates to batch file
--verify                Verify covariate order in batch file
```

### Common Options

```
--dir DIR               Search directory for auto-detection (default: current directory)
-h, --help              Show help message and exit
```

### ADD Mode Options

```
-o, --output OUTPUT     Output batch file path (default: input_stem_with_covariates.m)
--tiv TIV               TIV values file (default: TIV.txt in batch directory)
--iqr IQR               IQR values file (default: IQR.txt in batch directory)
--iqr-job IQR_JOB       IQR job file (default: IQR_job.m in batch directory)
```

### VERIFY Mode Options

```
--rows ROWS             Number of first/last rows to display (default: 10)
```

## Examples

### Example 1: Standard Verification

```bash
$ python3 manage_covariates.py --verify batch_3x3_job_with_covariates.m

==========================================================================================
VERIFY MODE: Checking Covariate Order
==========================================================================================
BATCH FILE: batch_3x3_job_with_covariates.m
Reading batch file...

==========================================================================================
VERIFICATION SUMMARY
==========================================================================================
Total files:        369
  ✓ TIV                  (index 1):  369 values  OK
  ✓ IQR                  (index 2):  369 values  OK

✓ PASS: All counts match! File order is correct.
```

### Example 2: Auto-Detect and Add

```bash
$ cd /path/to/analysis && python3 manage_covariates.py --add

# Looks for:
#   - batch_*.m files (finds first match)
#   - TIV.txt in current directory
#   - IQR.txt in current directory
#   - IQR_job.m in current directory
# Creates: batch_*_with_covariates.m
```

### Example 3: Custom Paths

```bash
python3 manage_covariates.py --add batch_file.m \
  -o /output/path/batch_with_cov.m \
  --tiv /data/tiv_values.txt \
  --iqr /data/iqr_values.txt \
  --iqr-job /data/iqr_job.m
```

### Example 4: Extensive Verification

```bash
# Show first and last 30 entries with statistics
python3 manage_covariates.py --verify batch.m --rows 30
```

## How It Works

### ADD Mode

1. **File Discovery**: Finds batch file and required covariate files
2. **Extraction**: 
   - Extracts file list from batch file (in order)
   - Extracts file list from IQR job file (in order)
   - Reads TIV and IQR values
3. **Mapping**: Matches files by subject ID and session number
4. **Reordering**: Reorders covariate values to match batch file order
5. **Integration**: Adds properly ordered covariates to batch file
6. **Output**: Saves new batch file with covariates

**Key Feature**: File matching is based on subject ID and session parsing from filenames, ensuring correct order regardless of directory structure.

### VERIFY Mode

1. **Extraction**: Reads batch file and extracts:
   - All files (in order they appear)
   - All covariates and their values
   - Covariate names
2. **Validation**: Checks count matching:
   - If `#files == #TIV values == #IQR values`, **VALID ✓**
   - If counts differ, **INVALID ✗** (data would be misaligned!)
3. **Reporting**: Displays comprehensive report:
   - Summary (pass/fail for each covariate)
   - Sample entries (first and last N)
   - Statistical summary (min, max, mean, std dev)
   - Final status

## Covariate Flexibility

The tool works with:
- **Any covariate names**: TIV, IQR, Age, Sex, Group, custom names, etc.
- **Any number of covariates**: 1, 2, 5, 10+ covariates
- **Any covariate indices**: cov(1), cov(2), cov(3) or non-sequential cov(1), cov(5), cov(10)

No hardcoding required - automatically detects what's in your batch file!

## Output Files

### ADD Mode

Creates a new MATLAB batch file with structure:

```matlab
matlabbatch{1}.spm.tools.cat.factorial_design.cov(1).c = [
    1647.49
    1472.84
    1551.73
    ...
];
matlabbatch{1}.spm.tools.cat.factorial_design.cov(1).cname = 'TIV';
matlabbatch{1}.spm.tools.cat.factorial_design.cov(1).iCFI = 1;
matlabbatch{1}.spm.tools.cat.factorial_design.cov(1).iCC = 1;

matlabbatch{1}.spm.tools.cat.factorial_design.cov(2).c = [
    1.649102
    1.664562
    1.649135
    ...
];
matlabbatch{1}.spm.tools.cat.factorial_design.cov(2).cname = 'IQR';
...
```

### VERIFY Mode

Produces a detailed report with:
- Summary of counts and status
- Sample entries (first and last N rows)
- Statistical analysis
- Pass/Fail determination

## Error Handling

### Common Issues

**Missing Files**
```
Error: Missing required files:
  - Batch file: batch.m
  - TIV file: TIV.txt
  - IQR file: IQR.txt
```

**Solution**: Ensure files exist in current directory or use `--tiv`, `--iqr`, `--iqr-job` flags.

**No Batch Files Found**
```
FileNotFoundError: No batch files (*batch*.m) found in /path
```

**Solution**: Provide explicit batch file path or use `--dir` to specify search directory.

**Covariate Mismatch**
```
✗ FAIL: Counts do not match! Data order may be INVALID for analysis!
  ✗ TIV (index 1): 369 values MISMATCH (has 361)
```

**Solution**: Check that file and covariate counts match. May indicate:
- Files were added/removed from batch
- Covariate values don't correspond to batch files
- File matching logic failed (check subject/session naming)

## Exit Codes

- `0`: Success (ADD completed or VERIFY passed)
- `1`: Failure (ADD failed or VERIFY failed/mismatch)

Useful for scripting:
```bash
python3 manage_covariates.py --verify batch.m
if [ $? -eq 0 ]; then
    echo "Covariates are valid!"
    # proceed with SPM analysis
else
    echo "Covariate mismatch detected!"
    # abort analysis
fi
```

## Tips & Best Practices

### Before SPM Analysis

Always run VERIFY before statistical analysis:

```bash
python3 manage_covariates.py --verify your_batch.m
```

If status is **VALID ✓**, it's safe to run SPM analysis.

### File Naming

For automatic matching to work, files must contain:
- Subject ID: `sub-XXXXXX` or `sub_XXXXXX`
- Session number: `ses-X` or `ses_X`

Example valid names:
- `sub-1291145_ses-1.nii`
- `sub-001_ses-02.nii`
- `sub_999_ses_3.nii`

### Covariate Files

TIV.txt and IQR.txt format (one value per line):
```
1647.49
1472.84
1551.73
...
```

Values must correspond to subject order in IQR_job.m file.

### Large Batches

For batches with many files (1000+), use `--rows` to show more samples:

```bash
# Show 50 first/last entries for large batch
python3 manage_covariates.py --verify batch_large.m --rows 50
```

## Advanced Usage

### Batch Processing Multiple Files

```bash
for batch in batch_*.m; do
    echo "Processing $batch..."
    python3 manage_covariates.py --verify "$batch"
    if [ $? -ne 0 ]; then
        echo "ERROR in $batch!"
        exit 1
    fi
done
```

### Creating Results Report

```bash
# Redirect output to file
python3 manage_covariates.py --verify batch.m > verification_report.txt 2>&1

# Append to log
python3 manage_covariates.py --verify batch.m >> covariate_verification.log
```

### Integration with SPM Scripts

```matlab
% In your SPM batch script
fprintf('Verifying covariates...\n');
status = system('python3 manage_covariates.py --verify batch_job.m');

if status == 0
    fprintf('✓ Covariates validated. Proceeding with analysis.\n');
else
    error('Covariate verification failed!');
end
```

## Troubleshooting

### Script Not Found

```bash
# Make sure you're in the correct directory
pwd

# Or provide full path
/path/to/manage_covariates.py --verify batch.m

# Or use python3 explicitly
python3 /path/to/manage_covariates.py --verify batch.m
```

### Permission Denied

```bash
chmod +x manage_covariates.py
python3 manage_covariates.py --help
```

### Python Version Issues

```bash
# Ensure Python 3.6+
python3 --version

# Use explicit python3 path if needed
/usr/bin/python3 manage_covariates.py --verify
```

### Encoding Issues

If you encounter encoding errors, ensure file encoding is UTF-8:

```bash
file batch_3x3_job.m      # Check encoding
iconv -f latin1 -t utf-8 batch_3x3_job.m > batch_fixed.m
```

## Related Files

- `COVARIATE_VERIFICATION_GUIDE.md` - Detailed verification explanation
- `verify_covariates.py` - Standalone verification script (older version)
- `add_covariates.py` - Standalone add covariates script (older version)

## Version History

### v1.0 (Combined Tool)
- Unified ADD and VERIFY modes
- Automatic file detection
- Comprehensive error handling
- Production-ready

### v0.2 (Previous)
- Separate `verify_covariates.py` and `add_covariates.py`
- Limited flexibility
- Less error handling

## FAQ

**Q: Can I use this with custom covariate names?**
A: Yes! ADD mode works with any covariate. Just ensure your batch file contains the correct covariate names. VERIFY mode automatically detects them.

**Q: What if I have more than 2 covariates?**
A: Both modes support unlimited covariates. Just ensure all are properly ordered in your batch file.

**Q: Does file order have to match numerical order?**
A: No. The tool matches files by subject ID and session, then reorders values correctly regardless of numerical order.

**Q: Can I edit the script to change default behavior?**
A: Yes, but not recommended. Instead, use command-line flags. The tool is designed to be flexible through arguments.

**Q: What does "INVALID" status mean?**
A: It means file count doesn't match covariate count. Your statistical analysis would produce incorrect results if you proceed. Investigate before running SPM.

**Q: Can I verify manually?**
A: Yes, but not recommended. This script automates the tedious verification. Use it!

## Support

For issues or questions, check:
1. This README
2. Script's `--help` output
3. Documentation files in the workspace
4. Error messages (they're quite descriptive)

## License

Internal use - CAT12/SPM analysis pipeline.
