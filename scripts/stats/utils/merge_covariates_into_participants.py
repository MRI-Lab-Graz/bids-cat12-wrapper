#!/usr/bin/env python3
"""
Merge a covariates TSV (extracted from CAT12 XMLs) into an existing
participants.tsv file.

The covariates TSV is expected to have columns: `file_path`, `subject`,
`session` (optional) and one or more measure columns (e.g. `vol_TIV`, `SQR`).

This script will detect whether `participants.tsv` is
 - subject-level (has `nr_sessions`) -> it will merge subject-level covariates
   (aggregated if multiple sessions exist) into the subject rows, or
 - scan-level (has one row per scan with a `session` column) -> it will
   merge covariates by `(participant, session)`.

Output is a TSV with the same structure as the input `participants.tsv`
but with the covariate columns appended.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def find_participant_column(df):
    for col in df.columns:
        if 'participant' in col.lower() or 'participant_id' in col.lower() or col.lower() == 'id':
            return col
    return None


def main():
    p = argparse.ArgumentParser(description='Merge covariates TSV into participants.tsv')
    p.add_argument('--participants', required=True, help='Path to participants.tsv')
    p.add_argument('--covariates', required=True, help='Path to covariates TSV (from extract_covariates_from_xml.py)')
    p.add_argument('--out', required=True, help='Output merged participants TSV')
    p.add_argument('--session-col', default='session', help='Session column name in participants file')
    args = p.parse_args()

    ppath = Path(args.participants)
    cpath = Path(args.covariates)

    if not ppath.exists():
        print(f"ERROR: participants file not found: {ppath}")
        sys.exit(2)
    if not cpath.exists():
        print(f"ERROR: covariates file not found: {cpath}")
        sys.exit(2)

    dfp = pd.read_csv(ppath, sep='\t')
    dfc = pd.read_csv(cpath, sep='\t')

    part_col = find_participant_column(dfp)
    if part_col is None:
        print('ERROR: Could not detect participant id column in participants.tsv')
        print('Expected column like participant_id or participant')
        sys.exit(2)

    # Normalize subject ids in covariates -> remove leading 'sub-'
    if 'subject' in dfc.columns:
        dfc['subject_norm'] = dfc['subject'].astype(str).str.replace('^sub[-_]', '', regex=True)
    else:
        dfc['subject_norm'] = dfc['file_path'].astype(str).apply(lambda x: '')

    # Normalize participant ids in participants
    dfp['__part_norm'] = dfp[part_col].astype(str).str.replace('^sub[-_]', '', regex=True)

    measures = [c for c in dfc.columns if c not in ('file_path', 'subject', 'subject_norm', 'session')]

    # Decide subject-level or scan-level
    if 'nr_sessions' in dfp.columns:
        # Subject-level participants.tsv
        # If the covariates file contains session-level rows (multiple rows per subject),
        # expand the participants dataframe to one row per (subject,session) and merge
        # session-specific covariates. Otherwise aggregate subject-level covariates.

        # Detect whether covariates are session-level
        session_level = False
        if 'session' in dfc.columns and dfc['session'].notna().any():
            # More than one unique session value for at least one subject
            try:
                ses_counts = dfc.groupby('subject_norm')['session'].nunique()
                if ses_counts.max() > 1:
                    session_level = True
            except Exception:
                session_level = True

        if session_level:
            # Expand participants into one row per (subject, session)
            rows = []
            for _, prow in dfp.iterrows():
                subj = prow['__part_norm']
                matching = dfc[dfc['subject_norm'] == subj]
                if matching.empty:
                    # No session info available for this subject; keep original subject row
                    rows.append(prow.to_dict())
                else:
                    # Use the session values found in the covariates file (preserve order)
                    ses_vals = matching['session'].astype(str).fillna('').tolist()
                    # dedupe while preserving order
                    seen = set()
                    sess_unique = []
                    for s in ses_vals:
                        if s not in seen:
                            seen.add(s)
                            sess_unique.append(s)
                    for s in sess_unique:
                        newrow = prow.to_dict()
                        # Ensure session column exists in the participants row
                        newrow[args.session_col] = s
                        rows.append(newrow)

            if len(rows) == 0:
                print('Warning: attempted expansion to session-level but found no session rows; falling back to subject-level aggregation')
                session_level = False
            else:
                dfp_exp = pd.DataFrame(rows)
                # normalize helper columns
                dfp_exp['__part_norm'] = dfp_exp['__part_norm'].astype(str)
                dfp_exp['__ses_str'] = dfp_exp[args.session_col].astype(str)
                dfc['__ses_str'] = dfc['session'].astype(str) if 'session' in dfc.columns else ''

                merged = pd.merge(dfp_exp, dfc[['subject_norm', '__ses_str'] + measures], left_on=['__part_norm', '__ses_str'], right_on=['subject_norm', '__ses_str'], how='left')
                # If merged produced no matched covariates, fall back to subject-level aggregation
                if merged[measures].isna().all().all():
                    print('Warning: No session-level covariates matched after expansion; falling back to subject-level aggregation')
                    session_level = False
                else:
                    dfp = merged

        if not session_level:
            # Subject-level: aggregate covariates per subject
            # Build a dict subject -> values (take first non-null if multiple sessions exist)
            sub_vals = {}
            for subj, row in dfc.groupby('subject_norm'):
                vals = {}
                for m in measures:
                    colvals = row[m].dropna().unique()
                    if len(colvals) == 0:
                        vals[m] = None
                    else:
                        if len(colvals) > 1:
                            print(f"Warning: multiple values for covariate {m} for subject {subj}; taking the first")
                        vals[m] = colvals[0]
                sub_vals[subj] = vals

            # Append columns to dfp
            for m in measures:
                dfp[m] = dfp['__part_norm'].map(lambda s: sub_vals.get(s, {}).get(m, None))

    else:
        # Scan-level: need session column
        if args.session_col not in dfp.columns:
            print(f"ERROR: participants.tsv appears scan-level but session column '{args.session_col}' not found")
            sys.exit(2)

        # Normalize session types to string for matching
        dfp['__ses_str'] = dfp[args.session_col].astype(str)
        dfc['__ses_str'] = dfc['session'].astype(str) if 'session' in dfc.columns else ''

        # Merge on normalized participant and session
        merged = pd.merge(dfp, dfc[['subject_norm', '__ses_str'] + measures], left_on=['__part_norm', '__ses_str'], right_on=['subject_norm', '__ses_str'], how='left')

        # If merge failed to attach (all NaNs), try merging by participant only
        if merged[measures].isna().all().all():
            print('Warning: No covariates matched by (participant,session). Trying participant-only merge...')
            merged = pd.merge(dfp, dfc[['subject_norm'] + measures].drop_duplicates('subject_norm'), left_on='__part_norm', right_on='subject_norm', how='left')

        dfp = merged

    # Clean helper columns
    dfp = dfp.drop(columns=[c for c in dfp.columns if c.startswith('__')])

    # Write out
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dfp.to_csv(out, sep='\t', index=False)
    print(f"Wrote merged participants file to: {out}")


if __name__ == '__main__':
    main()
