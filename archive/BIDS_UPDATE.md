# BIDS Compliance Update

**Date**: 2025
**Status**: ✅ COMPLETE

## Summary

Updated the CAT12 longitudinal analysis pipeline to support **BIDS-compliant participants.tsv** format (one row per subject) while maintaining backward compatibility with scan-level format (one row per scan).

## Changes Made

### 1. participants.tsv (Regenerated)

**Old format** (scan-level, NOT BIDS-compliant):
```tsv
participant_id  session  group
sub-1291003     1        control
sub-1291003     2        control
sub-1291003     3        control
...
```
❌ Problem: Multiple rows for same subject, violates BIDS specification

**New format** (BIDS-compliant):
```tsv
participant_id  nr_sessions  group
sub-1291003     3            control
sub-1291005     3            control
...
```
✅ Solution: One row per subject with `nr_sessions` column

### 2. parse_participants.py (Updated)

**Added BIDS format detection:**
- Detects BIDS format by checking for `nr_sessions` column
- Falls back to scan-level format if `session` column present
- Maintains backward compatibility

**Key code changes:**

```python
# Format detection (lines 220-236)
if 'nr_sessions' in df.columns:
    print("Detected BIDS-compliant format (one row per subject)")
    is_bids_format = True
else:
    print("Detected scan-level format (one row per scan)")
    is_bids_format = False

# Session enumeration (lines 277-290)
if is_bids_format:
    max_sessions = int(df['nr_sessions'].max())
    sessions = list(range(1, max_sessions + 1))
else:
    sessions = sorted(df[session_col].unique())

# Processing loop (lines ~295-355)
if is_bids_format:
    # Enumerate sessions for each subject
    nr_sessions = int(row['nr_sessions'])
    sessions_to_process = list(range(1, nr_sessions + 1))
else:
    # Single session from row
    sessions_to_process = [row[session_col]]
```

## Validation Results

✅ **Parser test successful:**
```
BIDS format: 123 subjects, up to 3 sessions each
✓ Found 369 files
Groups: control (52), intervention_2w (31), intervention_4w (40)
Total scans: 369
Design matrix: 3 groups × 3 timepoints = 9 cells
```

✅ **Design structure validated:**
- `control × 1`: 52 scans
- `control × 2`: 52 scans  
- `control × 3`: 52 scans
- `intervention_2w × 1`: 31 scans
- `intervention_2w × 2`: 31 scans
- `intervention_2w × 3`: 31 scans
- `intervention_4w × 1`: 40 scans
- `intervention_4w × 2`: 40 scans
- `intervention_4w × 3`: 40 scans

**Total**: 369 scans from 123 subjects ✓

## Pipeline Usage

The main pipeline script now works with BIDS-compliant participants.tsv:

```bash
./cat12_longitudinal_analysis.sh \
    --cat12-dir /Volumes/Thunder/129_PK01/cat12 \
    --participants participants.tsv \
    --modality vbm \
    --smoothing 6 \
    --pilot
```

## Design Matrix

The analysis uses a **well-conditioned design** (no subject factors):

- **Factor 1**: Group (dept=0, between-subject)
  - 3 levels: control, intervention_2w, intervention_4w
  
- **Factor 2**: Time (dept=1, within-subject)  
  - 3 levels: session 1, 2, 3

**Critical**: SPM handles within-subject dependencies implicitly through `dept=1`, so NO explicit subject regressors are needed. This avoids ill-conditioned/rank-deficient design matrices.

## Backward Compatibility

The parser still supports scan-level format:

```tsv
participant_id  session  group
sub-1291003     1        control
sub-1291003     2        control
```

Format is auto-detected based on column names.

## Files Modified

1. `/Volumes/Thunder/129_PK01/cat12/stats/participants.tsv` - Regenerated
2. `/Volumes/Thunder/129_PK01/cat12/stats/utils/parse_participants.py` - Updated
   - Backup saved as `parse_participants.py.backup`

## Next Steps

- ✅ BIDS compliance achieved
- ✅ Parser updated and tested
- ⏳ End-to-end pipeline testing (pilot mode)
- ⏳ Full analysis run

## References

- BIDS specification: https://bids.neuroimaging.io/
- participants.tsv format: One row per subject, required columns: `participant_id`
- Optional columns: `age`, `sex`, `group`, `nr_sessions` (custom), etc.
