#!/usr/bin/env python3
"""
Extract covariates (TIV, quality measures) from CAT12 per-subject XML
reports and write a TSV in the exact order of images stored in an SPM.mat.

Usage:
  utils/extract_covariates_from_xml.py --spm /path/to/SPM.mat --cat12 /path/to/cat12 --out covariates.tsv

The script will try to read the file order from `SPM.mat` and then for each
image find the corresponding XML report in `CAT12_DIR/sub-*/report/` and
extract the requested measures (defaults: vol_TIV, SQR, ICR, SurfaceEulerNumber).

The output TSV contains one row per SPM image in the same order and columns:
  file_path<TAB>subject<TAB>session<TAB>vol_TIV<TAB>SQR<TAB>ICR<TAB>SurfaceEulerNumber

This TSV can be used as `--participants` (scan-level) or you can point the
pipeline to it and pass `--covariates "vol_TIV,SQR"` to include those regressors.
"""
import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


def load_spm_file_list(spm_mat):
    """Attempt to extract SPM.xY.P (ordered file paths) from SPM.mat.
    Returns list of file paths (strings) in SPM order.
    """
    try:
        from scipy.io import loadmat
    except Exception:
        print("ERROR: scipy is required to read SPM.mat (pip install scipy)")
        sys.exit(2)

    try:
        mat = loadmat(spm_mat, struct_as_record=False, squeeze_me=True)
    except Exception as e:
        print(f"ERROR: Could not load {spm_mat}: {e}")
        sys.exit(2)

    SPM = mat.get('SPM', None)
    if SPM is None:
        print(f"ERROR: 'SPM' variable not found inside {spm_mat}")
        sys.exit(2)

    # Try multiple access patterns (scipy loads MATLAB structs differently)
    P = None
    # pattern 1: object with attribute xY
    try:
        xY = getattr(SPM, 'xY')
        P = getattr(xY, 'P')
    except Exception:
        pass

    # pattern 2: dict-like
    if P is None:
        try:
            xY = SPM.get('xY')
            if xY is not None:
                P = xY.get('P')
        except Exception:
            pass

    # pattern 3: nested numpy/object arrays
    if P is None:
        try:
            xY = SPM['xY']
            if isinstance(xY, (list, tuple)):
                xY = xY[0]
            P = xY['P']
        except Exception:
            P = None

    if P is None:
        print("ERROR: Could not find xY.P inside SPM.mat. The file may be saved in an unusual format.")
        print("Please provide a participants.tsv or a plain file-list instead.")
        sys.exit(2)

    # Convert MATLAB cell arrays / numpy arrays to Python list of strings
    files = []
    if isinstance(P, str):
        files = [P]
    else:
        try:
            for p in P:
                # p may be bytes, numpy.str_, or object
                if isinstance(p, bytes):
                    files.append(p.decode('utf-8'))
                else:
                    files.append(str(p))
        except Exception:
            files = list(map(str, P))

    # Normalize paths (MATLAB often uses full paths with commas like ',1')
    cleaned = []
    for f in files:
        if isinstance(f, str) and ',1' in f:
            f = f.split(',')[0]
        cleaned.append(os.path.normpath(f))

    return cleaned


def find_xml_for_subject(cat12_dir, subject, session=None):
    """Return path to the XML report for subject (and session if provided).
    Searches under cat12_dir/sub-<subject>/report/ for matching xml file.
    """
    base = Path(cat12_dir)
    subdir = base / f"sub-{subject}" / 'report'
    if not subdir.exists():
        # try without 'sub-' prefix
        subdir = base / subject / 'report'
        if not subdir.exists():
            return None

    # Prefer a file matching session if provided
    if session is not None:
        pattern = f"*ses-{session}*{subject}*.xml"
        gl = list(subdir.glob(pattern))
        if gl:
            return str(gl[0])

    # fallback to any xml in report dir
    gl = list(subdir.glob('*.xml'))
    if not gl:
        return None
    return str(gl[0])


def extract_measures_from_xml(xml_path, measures):
    """Parse XML and extract requested measures (tags).
    measures: list of tag names (e.g. 'vol_TIV', 'SQR')
    Returns dict {measure: value or ''}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    out = {}

    for m in measures:
        # Collect all occurrences (case-insensitive) of the tag
        vals = []
        for node in root.iter():
            if node.tag.lower() == m.lower():
                if node.text is None:
                    continue
                v = node.text.strip()
                v = re.sub(r'^[\[\s]+|[\]\s]+$', '', v)
                if v != '':
                    vals.append(v)

        # Prefer the first value that parses as a number. If none parse,
        # fall back to the first available occurrence (usually a label).
        chosen = ''
        for v in vals:
            try:
                float(v)
                chosen = v
                break
            except Exception:
                # try to extract a numeric substring
                mnum = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', v)
                if mnum:
                    chosen = mnum.group(0)
                    break

        if chosen == '' and vals:
            chosen = vals[0]

        out[m] = chosen
    return out


def parse_subject_and_session_from_path(path):
    # Try to extract subject (sub-123) and session (ses-1) from filename or path
    m_sub = re.search(r'(sub-[A-Za-z0-9-_]+)', path)
    subject = None
    if m_sub:
        subject = m_sub.group(1).replace('sub-', '')
    else:
        m2 = re.search(r'sub[_-]?([0-9]+)', path)
        if m2:
            subject = m2.group(1)

    m_ses = re.search(r'(ses-[A-Za-z0-9-_]+)', path)
    session = None
    if m_ses:
        session = m_ses.group(1).replace('ses-', '')
    else:
        m3 = re.search(r'ses[_-]?([0-9]+)', path)
        if m3:
            session = m3.group(1)

    return subject, session


def main():
    p = argparse.ArgumentParser(description='Extract covariates from CAT12 XMLs in SPM.mat or participants order')
    p.add_argument('--spm', required=False, help='Path to SPM.mat (preferred for exact SPM ordering)')
    p.add_argument('--participants', required=False, help='Path to participants.tsv to determine ordering when SPM.mat is not available')
    p.add_argument('--cat12', required=True, help='Path to CAT12 directory (contains sub-*/report/*.xml)')
    p.add_argument('--out', required=True, help='Output TSV file to write (ordered)')
    p.add_argument('--measures', default='vol_TIV,SQR,ICR,SurfaceEulerNumber',
                   help='Comma-separated XML tag names to extract')
    args = p.parse_args()

    spm_mat = args.spm
    cat12_dir = args.cat12
    out_file = args.out
    measures = [m.strip() for m in args.measures.split(',') if m.strip()]
    participants = args.participants

    rows = []
    if spm_mat:
        files = load_spm_file_list(spm_mat)
        if not files:
            print('No files found in SPM.mat')
            sys.exit(1)

        for f in files:
            subj, ses = parse_subject_and_session_from_path(f)
            if subj is None:
                print(f"Warning: could not parse subject from path: {f}; skipping")
                rows.append((f, '', '') + tuple('' for _ in measures))
                continue

            xml = find_xml_for_subject(cat12_dir, subj, ses)
            if xml is None:
                print(f"Warning: XML report not found for subject {subj} (session {ses}) under {cat12_dir}")
                vals = {m: '' for m in measures}
            else:
                vals = extract_measures_from_xml(xml, measures)

            row = [f, subj, ses if ses is not None else ''] + [vals.get(m, '') for m in measures]
            rows.append(row)
    elif participants:
        # Use participants.tsv order. If participants has per-scan rows (with a session column)
        # iterate rows and extract xml for each (subject,session). If participants is subject-level
        # (contains nr_sessions) we will extract one row per subject (session left empty).
        if not Path(participants).exists():
            print(f"ERROR: participants file not found: {participants}")
            sys.exit(2)
        dfp = pd.read_csv(participants, sep='\t')

        # detect participant id column
        pid_col = None
        for col in dfp.columns:
            if 'participant' in col.lower() or 'participant_id' in col.lower() or col.lower() == 'id':
                pid_col = col
                break
        if pid_col is None:
            print('ERROR: Could not find participant id column in participants.tsv')
            sys.exit(2)

        if 'session' in dfp.columns and dfp['session'].notna().any():
            # Per-scan participants: follow rows
            for _, prow in dfp.iterrows():
                subj = str(prow[pid_col]).replace('sub-', '').replace('SUB-', '')
                ses = str(prow['session']) if not pd.isna(prow['session']) else ''
                fpath = ''
                xml = find_xml_for_subject(cat12_dir, subj, ses if ses != '' else None)
                if xml is None:
                    print(f"Warning: XML report not found for subject {subj} session {ses}")
                    vals = {m: '' for m in measures}
                else:
                    vals = extract_measures_from_xml(xml, measures)
                rows.append([fpath, subj, ses] + [vals.get(m, '') for m in measures])
        else:
            # Subject-level participants: one row per subject
            for _, prow in dfp.iterrows():
                subj = str(prow[pid_col]).replace('sub-', '').replace('SUB-', '')
                fpath = ''
                xml = find_xml_for_subject(cat12_dir, subj, None)
                if xml is None:
                    print(f"Warning: XML report not found for subject {subj}")
                    vals = {m: '' for m in measures}
                else:
                    vals = extract_measures_from_xml(xml, measures)
                rows.append([fpath, subj, ''] + [vals.get(m, '') for m in measures])
    else:
        print('ERROR: Either --spm or --participants must be provided')
        sys.exit(2)

    # Write TSV
    header = ['file_path', 'subject', 'session'] + measures
    with open(out_file, 'w') as fo:
        fo.write('\t'.join(header) + '\n')
        for r in rows:
            safe = [str(x) if x is not None else '' for x in r]
            fo.write('\t'.join(safe) + '\n')

    print(f"Wrote {len(rows)} rows to {out_file}")
    print("Note: order matches file order in SPM.mat. Use this TSV as scan-level participants.tsv or pass --covariates accordingly.")


if __name__ == '__main__':
    main()
