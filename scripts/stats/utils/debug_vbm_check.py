#!/usr/bin/env python3
"""
Debug helper for VBM ML pipeline: check mask and delta image validity.

Usage (from repo root):
  python utils/debug_vbm_check.py \
    --participants-tsv participants.tsv \
    --data-root /Volumes/Thunder/129_PK01/cat12/data/cat12 \
    --mask templates/brainmask_GMtight.nii \
    --session-a 1 --session-b 2 --use-unsmoothed

This script will:
 - read participants.tsv to extract subject IDs
 - try to find image files for the requested sessions under data_root
 - optionally try common substitutions to prefer unsmoothed images
 - compute per-subject delta arrays (T2 - T1) and print shapes/stats
 - load the mask and report number of masked voxels and shape
 - compute NaN/inf/variance checks similar to the main script and report

It does NOT require scikit-learn or nilearn; it only uses nibabel and numpy.
"""

import argparse
import json
import os
import re
import sys
from glob import iglob

import numpy as np

try:
    import nibabel as nib
except Exception as e:
    print('ERROR: nibabel not available in this Python. Activate the project venv or install nibabel.')
    raise

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def load_participants_simple(tsv_path, group_col=None):
    import csv
    participants = []
    with open(tsv_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if not row:
                continue
            if row[0].startswith('#'):
                continue
            # simple heuristics
            if row[0].lower().startswith('sub-') or row[0].lower().startswith('sub'):
                sub = row[0]
                group = None
                if len(row) >= 3:
                    group = row[2]
                elif len(row) == 2:
                    group = row[1]
                participants.append({'subject': sub, 'group': group})
            else:
                # header present -> use first column as participant id
                header = row
                data_rows = list(reader)
                for r in data_rows:
                    if not r:
                        continue
                    sub = r[0]
                    grp = None
                    if len(r) >= 3:
                        grp = r[2]
                    elif len(r) == 2:
                        grp = r[1]
                    participants.append({'subject': sub, 'group': grp})
                break
    return participants


def try_map_to_unsmoothed(path):
    if not isinstance(path, str):
        return None
    candidates = []
    try:
        candidates.append(re.sub(r'(?i)s6mwp1', 'mwp1', path))
        candidates.append(re.sub(r'(?i)s6', 'mwp1', path))
    except Exception:
        pass
    candidates.append(path.replace('_s6', '_mwp1'))
    candidates.append(path.replace('-s6', '-mwp1'))
    candidates.append(re.sub(r'(?i)smw?c', 'mwp', path))
    seen = set()
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        if os.path.exists(c):
            return c
    return None


def find_images_for_subjects(participants, data_root, session_ids=('1','2')):
    results = []
    data_root = os.path.abspath(data_root)
    sess_tokens_map = {}
    for sid in session_ids:
        ival = sid
        tokens = [f'ses-{sid}', f'ses{sid}']
        try:
            ival_i = int(sid)
            if ival_i < 10:
                tokens.append(f'ses-0{ival_i}')
                tokens.append(f'ses0{ival_i}')
        except Exception:
            pass
        sess_tokens_map[str(sid)] = [t.lower() for t in tokens]

    for p in participants:
        subj = p['subject']
        group = p.get('group')
        matches_by_session = {}
        pattern = os.path.join(data_root, '**', f'*{subj}*')
        for fp in iglob(pattern, recursive=True):
            if not os.path.isfile(fp):
                continue
            if not (fp.lower().endswith('.nii') or fp.lower().endswith('.nii.gz')):
                continue
            low = fp.lower()
            for sid, toks in sess_tokens_map.items():
                if any(tok in low for tok in toks):
                    matches_by_session.setdefault(sid, []).append(fp)
                    break
        for sid in session_ids:
            sid = str(sid)
            cand = None
            if sid in matches_by_session and matches_by_session[sid]:
                for m in matches_by_session[sid]:
                    lowm = m.lower()
                    if 's6' in lowm or 'mwp' in lowm:
                        cand = m
                        break
                if cand is None:
                    cand = matches_by_session[sid][0]
            if cand:
                results.append({'subject': subj, 'session': sid, 'group': group, 'path': cand})
    return {'files': results}


def build_entries_for_d21(design_files, session_a='1', session_b='2', use_unsmoothed=False):
    # build subj_map
    subj_map = {}
    for e in design_files:
        subj = e['subject']
        sess = e['session']
        subj_map.setdefault(subj, {})[str(sess)] = e['path']
    subj_ids = []
    subj_groups = []
    entries = []
    for subj, sessdict in subj_map.items():
        if str(session_a) in sessdict and str(session_b) in sessdict:
            p1 = sessdict[str(session_a)]
            p2 = sessdict[str(session_b)]
            if use_unsmoothed:
                p1 = try_map_to_unsmoothed(p1) or p1
                p2 = try_map_to_unsmoothed(p2) or p2
            entries.append((p1, p2))
            subj_ids.append(subj)
            # pick group from first matching file entry
            group = None
            for e in design_files:
                if e['subject'] == subj:
                    group = e.get('group')
                    break
            subj_groups.append(group)
    return subj_ids, subj_groups, entries


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--participants-tsv', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--mask', required=True)
    p.add_argument('--session-a', default='1')
    p.add_argument('--session-b', default='2')
    p.add_argument('--use-unsmoothed', action='store_true')
    p.add_argument('--max-subjects', type=int, default=20)
    p.add_argument('--plot-nans', action='store_true', help='Save a barplot (subject x, number of NaNs in delta y) as PNG')
    p.add_argument('--plot-out', default='nan_counts_barplot.png', help='Output path for the NaN counts barplot PNG')
    p.add_argument('--nan-csv', default=None, help='Optional CSV path to write per-subject NaN/mask stats')
    args = p.parse_args()

    parts = load_participants_simple(args.participants_tsv)
    print(f'Participants loaded: {len(parts)}')
    design = find_images_for_subjects(parts, args.data_root, session_ids=[str(args.session_a), str(args.session_b)])
    n_files = len(design.get('files', []))
    print(f'Design files found: {n_files}')

    # write small sample design to disk for inspection
    sample_design_path = 'debug_design_sample.json'
    json.dump(design, open(sample_design_path, 'w'), indent=2)
    print('Wrote sample design to', sample_design_path)

    subj_ids, subj_groups, entries = build_entries_for_d21(design.get('files', []), session_a=args.session_a, session_b=args.session_b, use_unsmoothed=args.use_unsmoothed)
    print(f'Number of subjects with both sessions: {len(subj_ids)}')
    if len(subj_ids) == 0:
        print('No subjects with required sessions found; aborting')
        sys.exit(2)

    # load mask
    if not os.path.exists(args.mask):
        print('Mask file not found:', args.mask)
    else:
        mimg = nib.load(args.mask)
        mdata = mimg.get_fdata(dtype=np.float32)
        print('Mask shape:', mdata.shape, 'affine:', mimg.affine)
        n_mask_vox = int(np.sum(mdata != 0))
        print('Non-zero mask voxels:', n_mask_vox)

    # compute deltas for up to max_subjects
    arrays = []
    imgs = []
    per_subject_stats = []
    for idx, ent in enumerate(entries[:args.max_subjects]):
        p1, p2 = ent
        print('\nSubject', subj_ids[idx])
        print('  p1 exists?', os.path.exists(p1), 'p2 exists?', os.path.exists(p2))
        try:
            img1 = nib.load(p1)
            img2 = nib.load(p2)
        except Exception as e:
            print('  Failed to load images:', e)
            continue
        a1 = img1.get_fdata(dtype=np.float32)
        a2 = img2.get_fdata(dtype=np.float32)
        print('  img shape:', a2.shape, 'affines equal?', np.allclose(img1.affine, img2.affine))
        # report NaN/inf
        print('  img1: min/max/mean/finite/allfinite:', np.nanmin(a1), np.nanmax(a1), np.nanmean(a1), np.isfinite(a1).all())
        print('  img2: min/max/mean/finite/allfinite:', np.nanmin(a2), np.nanmax(a2), np.nanmean(a2), np.isfinite(a2).all())
        d = a2 - a1
        arrays.append(d)
        imgs.append(img2)
        # collect per-subject stats (overall NaNs and NaNs inside mask if mask matches)
        n_nan_total = int(np.sum(~np.isfinite(d)))
        n_vox_total = int(d.size)
        n_nan_in_mask = None
        pct_mask_nan = None
        if 'mdata' in locals() and mdata.shape == d.shape:
            mask_bool = mdata != 0
            n_nan_in_mask = int(np.sum(~np.isfinite(d[mask_bool])))
            n_mask_vox = int(np.sum(mask_bool))
            pct_mask_nan = float(n_nan_in_mask) / float(n_mask_vox) if n_mask_vox > 0 else 0.0
        per_subject_stats.append({
            'subject': subj_ids[idx],
            'n_nan_total': n_nan_total,
            'n_vox_total': n_vox_total,
            'n_nan_in_mask': n_nan_in_mask,
            'pct_mask_nan': pct_mask_nan,
            'img_shape': tuple(d.shape),
            'affine_match': bool(np.allclose(img1.affine, img2.affine)),
        })

    if not arrays:
        print('\nNo arrays computed (no readable images). Aborting.')
        sys.exit(3)

    # Optionally compute per-subject NaN counts and plot
    if args.plot_nans:
        if plt is None:
            print('\nPlotting requested but matplotlib not available. Install matplotlib to enable plotting.')
        else:
            nan_counts = [int(np.sum(~np.isfinite(a))) for a in arrays]
            subj_plot_ids = [s.replace('sub-', '') for s in subj_ids[:len(nan_counts)]]
            # create a wide figure for many subjects
            w = max(6, min(0.2 * len(nan_counts), 40))
            h = 6
            fig, ax = plt.subplots(figsize=(w, h))
            ax.bar(range(len(nan_counts)), nan_counts, color='C0')
            ax.set_xlabel('Subject')
            ax.set_ylabel('Number of NaN voxels in delta')
            ax.set_title('Per-subject NaN counts (delta image)')
            # label every Nth tick to avoid crowding
            if len(subj_plot_ids) <= 40:
                ax.set_xticks(range(len(subj_plot_ids)))
                ax.set_xticklabels(subj_plot_ids, rotation=90)
            else:
                step = int(np.ceil(len(subj_plot_ids) / 40.0))
                ticks = list(range(0, len(subj_plot_ids), step))
                ax.set_xticks(ticks)
                ax.set_xticklabels([subj_plot_ids[i] for i in ticks], rotation=90)
            plt.tight_layout()
            outpath = args.plot_out
            try:
                fig.savefig(outpath, dpi=150)
                print('\nWrote NaN counts barplot to', outpath)
            except Exception as e:
                print('\nFailed to save plot:', e)
    # Optionally write CSV with per-subject stats
    if args.nan_csv:
        try:
            import csv as _csv
            with open(args.nan_csv, 'w', newline='') as _f:
                writer = _csv.DictWriter(_f, fieldnames=['subject','n_nan_total','n_vox_total','n_nan_in_mask','pct_mask_nan','img_shape','affine_match'])
                writer.writeheader()
                for r in per_subject_stats:
                    writer.writerow(r)
            print('Wrote per-subject NaN stats CSV to', args.nan_csv)
        except Exception as e:
            print('Failed to write CSV:', e)

    stacked = np.stack(arrays, axis=-1)
    print('\nStacked shape (x,y,z,n_subjects):', stacked.shape)
    # examine NaNs across voxels
    finite_mask = np.isfinite(stacked).all(axis=-1)
    n_finite_vox = int(np.sum(finite_mask))
    print('Finite voxels across all subjects:', n_finite_vox)
    var = np.nanvar(stacked, axis=-1)
    valid_var_mask = var > 1e-8
    n_valid_var = int(np.sum(valid_var_mask))
    print('Voxels with var > 1e-8:', n_valid_var)

    # If mask present, check overlap
    if os.path.exists(args.mask):
        mask_small = mdata != 0
        # try to broadcast shapes if necessary
        if mask_small.shape == stacked.shape[:3]:
            overlap = np.sum(mask_small & valid_var_mask)
            print('Overlap between mask and valid-variance voxels:', int(overlap))
        else:
            print('Mask shape does not match image shape; mask shape:', mdata.shape, 'image shape:', stacked.shape[:3])

    print('\nDiagnostics complete. If n_valid_var == 0 then there are no usable voxels after filtering.')

if __name__ == '__main__':
    main()
