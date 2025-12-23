#!/usr/bin/env python3
"""
vbm_ml_interaction.py

Lightweight Nilearn + scikit-learn prototype to analyse Group x Time interaction
using a pattern (machine-learning) approach on VBM delta maps (time2 - time1).

Usage example:
  python utils/vbm_ml_interaction.py \
    --design-json results/vbm/s6_vbm_final/design.json \
    --mask templates/brainmask_GMtight.nii \
    --output results/vbm/s6_vbm_final/ml_results --n-permutations 1000

What it does:
- Loads the design.json produced by the pipeline (expects 'files' entries with subject, session, group, path)
- Ensures each subject has exactly two sessions (ordered by session str -> numeric)
- Computes delta image per subject: img(session2) - img(session1)
- Vectorizes using NiftiMasker and runs a classifier to predict Group from delta maps
- Uses cross-validated StratifiedKFold and permutation_test_score for significance
- Trains final linear classifier and writes coefficient map back to NIfTI
- Saves a small JSON summary and optional visualization PNG

Notes / limitations:
- Requires Python packages: nibabel, nilearn, numpy, pandas, scikit-learn, matplotlib
- Designed as a prototype: tweak feature selection, CV, and confound handling for production

"""

import argparse
import json
import os
import re
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn.maskers import NiftiMasker
from nilearn.plotting import plot_stat_map
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC


def try_map_to_unsmoothed(path):
    """Try common substitutions to map a smoothed vbm filename to an unsmoothed counterpart (mwp1*).
    Returns the first existing candidate path, or None if none found.
    """
    if not isinstance(path, str):
        return None
    # quick check: if original exists, keep it (caller may still use it)
    if os.path.exists(path):
        # but still attempt to find mwp1 variant first
        pass

    candidates = []
    # prefer replacing 's6mwp1' -> 'mwp1' (handles filenames like 's6mwp1rsub-...')
    try:
        candidates.append(re.sub(r'(?i)s6mwp1', 'mwp1', path))
        candidates.append(re.sub(r'(?i)s6', 'mwp1', path))
    except Exception:
        pass
    # also try replacing common tokens
    candidates.append(path.replace('_s6', '_mwp1'))
    candidates.append(path.replace('-s6', '-mwp1'))
    # sometimes filenames contain 'smwc1' or similar; replace 'smwc' -> 'mwp'
    candidates.append(re.sub(r'(?i)smw?c', 'mwp', path))

    # unique candidates preserve order
    seen = set()
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        lowc = c.lower()
        # skip known white-matter or second-pass files (mwp2, mwp3) which are not desired
        if 'mwp2' in lowc or 'mwp3' in lowc or '_wm' in lowc:
            continue
        if os.path.exists(c):
            return c
    return None


def load_design(design_json):
    with open(design_json, 'r') as f:
        design = json.load(f)
    return design


def load_participants_tsv(tsv_path, group_col=None):
    # Expect simple TSV with at least subject and group columns.
    # Supports header rows. group_col may be a 1-based index or a header name.
    import csv
    participants = []
    rows = []
    with open(tsv_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if not row:
                continue
            # strip whitespace
            row = [c.strip() for c in row]
            # skip comment lines
            if row[0].startswith('#'):
                continue
            rows.append(row)

    if not rows:
        return participants

    # Heuristic: treat first row as header if it does NOT look like a subject id.
    header = None
    first_cell = rows[0][0].lower()
    if first_cell.startswith('sub-') or first_cell.startswith('sub'):
        # no header -> use positional columns
        data_rows = rows
        # create participants entries using positional assumptions
        for row in data_rows:
            if not row:
                continue
            sub = row[0]
            group = None
            if len(row) >= 3:
                group = row[2]
            elif len(row) == 2:
                group = row[1]
            participants.append({'subject': sub, 'group': group})
        return participants
    else:
        header = rows[0]
        data_rows = rows[1:]

    # create mapping from header name (lowered) to index
    lowered = [h.lower() for h in header]
    # Determine group column index if group_col provided
    group_idx = None
    if group_col is not None:
        try:
            group_idx = int(group_col) - 1
        except Exception:
            try:
                group_idx = lowered.index(group_col.lower())
            except ValueError:
                group_idx = None

    # choose default group column if not specified
    if group_idx is None:
        for cand in ('group_ml', 'group', 'groupid', 'group_id'):
            if cand in lowered:
                group_idx = lowered.index(cand)
                break

    for row in data_rows:
        if not row:
            continue
        entry = {}
        # map header names to values (when row shorter than header, missing values become None)
        for i, col in enumerate(header):
            val = row[i].strip() if i < len(row) else None
            entry[col] = val
            # also populate lowercase keys for convenience
            entry[col.lower()] = val
        # If a specific group_col was requested, prefer that column's value
        # as the canonical 'group' field even if a generic 'group' column exists.
        if group_idx is not None and group_idx < len(header):
            try:
                preferred = row[group_idx].strip() if group_idx < len(row) else entry.get(header[group_idx])
                if preferred is not None:
                    entry['group'] = preferred
            except Exception:
                # fallback: leave existing 'group' as-is
                pass
        # ensure 'subject' key exists
        entry['subject'] = entry.get('participant_id') or entry.get('participant') or entry.get(header[0])
        # ensure 'group' key exists
        if 'group' not in entry or not entry.get('group'):
            if group_idx is not None and group_idx < len(header):
                entry['group'] = entry.get(header[group_idx])
            else:
                entry['group'] = entry.get('group_ml') or entry.get('group')
        participants.append(entry)

    return participants


def find_images_for_subjects(participants, data_root, session_ids=None):
    """Walk data_root and try to find one file per requested subject/session.
    session_ids: iterable of session id strings (e.g. ['1','3']). If None, defaults to ['1','2'].
    Returns dict with key 'files' holding list of dicts: {subject, group, session, path}.
    """
    from glob import iglob
    results = []
    data_root = os.path.abspath(data_root)

    if session_ids is None:
        session_ids = ['1', '2']

    # build simple token lists for each requested session id
    sess_tokens_map = {}
    for sid in session_ids:
        try:
            ival = int(sid)
        except Exception:
            ival = sid
        # common filename tokens: ses-<id>, ses<id>, ses-0<id> (zero-padded)
        tokens = [f'ses-{sid}', f'ses{sid}']
        if isinstance(ival, int) and ival < 10:
            tokens.append(f'ses-0{ival}')
            tokens.append(f'ses0{ival}')
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
            # determine which requested session this file likely belongs to
            for sid, toks in sess_tokens_map.items():
                if any(tok in low for tok in toks):
                    matches_by_session.setdefault(sid, []).append(fp)
                    break

        # heuristic: for each requested session, pick the best match (prefer 's6' or 'mwp')
        for sid in session_ids:
            sid = str(sid)
            cand = None
            if sid in matches_by_session and matches_by_session[sid]:
                # prefer anatomically appropriate unsmoothed VBM variants (e.g. 'mwp1' or 's6' prefixed files)
                for m in matches_by_session[sid]:
                    lowm = m.lower()
                    # prefer 'mwp1' (unsmoothed) or 's6mwp1' or generic 's6' smoothed files when available
                    if 'mwp1' in lowm or 's6mwp1' in lowm or 's6' in lowm:
                        # but explicitly avoid selecting mwp2/mwp3 (white-matter maps)
                        if 'mwp2' in lowm or 'mwp3' in lowm or '_wm' in lowm:
                            continue
                        cand = m
                        break
                # if we didn't find a preferred candidate, pick the first that is not a mwp2/mwp3/WM file
                if cand is None:
                    for m in matches_by_session[sid]:
                        lowm = m.lower()
                        if 'mwp2' in lowm or 'mwp3' in lowm or '_wm' in lowm:
                            continue
                        cand = m
                        break
                if cand is None:
                    cand = matches_by_session[sid][0]
            if cand:
                results.append({'subject': subj, 'session': sid, 'group': group, 'path': cand})

    return {'files': results}


def build_subject_delta_images(design, session_a=None, session_b=None, use_unsmoothed=False, group_map=None):
    # design['files'] expected list of dicts with keys: subject, session, group, path
    files = design['files']
    # group by subject
    subj_map = {}
    for entry in files:
        subj = entry['subject']
        sess = entry['session']
        if subj not in subj_map:
            subj_map[subj] = {}
        subj_map[subj][str(sess)] = entry['path']

    subj_deltas = []
    subj_groups = []
    subj_ids = []

    # If both session_a and session_b provided, select those two sessions (if present)
    if session_a is not None and session_b is not None:
        sa = str(session_a)
        sb = str(session_b)
        for subj, sessdict in subj_map.items():
            if sa in sessdict and sb in sessdict:
                p1 = sessdict[sa]
                p2 = sessdict[sb]
                # Optionally try to substitute paths to unsmoothed images (mwp1*) if requested.
                if use_unsmoothed:
                    p1 = try_map_to_unsmoothed(p1) or p1
                    p2 = try_map_to_unsmoothed(p2) or p2
                subj_deltas.append((p1, p2))
                # find group from design['files'] (first entry for subj)
                group = None
                for e in files:
                    if e['subject'] == subj:
                        group = e.get('group')
                        break
                # if a group_map is provided (from participants.tsv), prefer that label
                if group_map is not None and subj in group_map and group_map.get(subj) is not None:
                    group = group_map.get(subj)
                subj_groups.append(group)
                subj_ids.append(subj)
            else:
                warnings.warn(f"Subject {subj} missing sessions {sa} or {sb} (skipping)")
        return subj_ids, subj_groups, subj_deltas

    # default behaviour: keep subjects with exactly two sessions (legacy behaviour)
    for subj, sessdict in subj_map.items():
        if len(sessdict) != 2:
            warnings.warn(f"Subject {subj} has {len(sessdict)} sessions (skipping)")
            continue
        # sort by session key (numeric if possible)
        keys = sorted(sessdict.keys(), key=lambda x: int(x) if x.isdigit() else x)
        p1 = sessdict[keys[0]]
        p2 = sessdict[keys[1]]
        subj_deltas.append((p1, p2))
        # find group from design['files'] (first entry for subj)
        group = None
        for e in files:
            if e['subject'] == subj:
                group = e.get('group')
                break
        # override from provided group_map if available
        if group_map is not None and subj in group_map and group_map.get(subj) is not None:
            group = group_map.get(subj)
        subj_groups.append(group)
        subj_ids.append(subj)

    return subj_ids, subj_groups, subj_deltas


def build_subject_delta_entries_for_type(design, delta_type='d21', use_unsmoothed=False, group_map=None, session_a=None, session_b=None):
    """Return subject ids, groups and entries appropriate for the requested delta_type.
    Entries format:
      - for d21/d32/d31: list of (p1,p2)
      - for dsec: list of (p1,p2,p3)
      - for concat: list of ((p1,p2),(p2,p3)) where first tuple is D21 pair and second is D32 pair
    session_a/session_b may override default mappings for simple delta types.
    """
    files = design['files']
    # group by subject -> map of session -> path
    subj_map = {}
    for entry in files:
        subj = entry['subject']
        sess = entry['session']
        if subj not in subj_map:
            subj_map[subj] = {}
        subj_map[subj][str(sess)] = entry['path']

    subj_ids = []
    subj_groups = []
    entries = []

    # helper to get group label
    def pick_group(subj):
        group = None
        for e in files:
            if e['subject'] == subj:
                group = e.get('group')
                break
        if group_map is not None and subj in group_map and group_map.get(subj) is not None:
            group = group_map.get(subj)
        return group

    # delta_type mapping defaults
    if delta_type == 'd21':
        sa, sb = ('1', '2')
    elif delta_type == 'd32':
        sa, sb = ('2', '3')
    elif delta_type == 'd31':
        sa, sb = ('1', '3')
    else:
        sa = sb = None

    # allow explicit override
    if session_a is not None and session_b is not None and delta_type in ('d21', 'd32', 'd31'):
        sa = str(session_a)
        sb = str(session_b)

    for subj, sessdict in subj_map.items():
        # need different requirements depending on delta_type
        if delta_type in ('d21', 'd32', 'd31'):
            if sa in sessdict and sb in sessdict:
                p1 = sessdict[sa]
                p2 = sessdict[sb]
                if use_unsmoothed:
                    p1 = try_map_to_unsmoothed(p1) or p1
                    p2 = try_map_to_unsmoothed(p2) or p2
                entries.append((p1, p2))
                subj_ids.append(subj)
                subj_groups.append(pick_group(subj))
            else:
                warnings.warn(f"Subject {subj} missing sessions {sa} or {sb} (skipping)")
        elif delta_type == 'dsec':
            # need sessions 1,2,3
            if '1' in sessdict and '2' in sessdict and '3' in sessdict:
                p1 = sessdict['1']
                p2 = sessdict['2']
                p3 = sessdict['3']
                if use_unsmoothed:
                    p1 = try_map_to_unsmoothed(p1) or p1
                    p2 = try_map_to_unsmoothed(p2) or p2
                    p3 = try_map_to_unsmoothed(p3) or p3
                entries.append((p1, p2, p3))
                subj_ids.append(subj)
                subj_groups.append(pick_group(subj))
            else:
                warnings.warn(f"Subject {subj} missing one of sessions 1/2/3 (skipping)")
        elif delta_type == 'concat':
            # need pairs for D21 and D32
            if '1' in sessdict and '2' in sessdict and '3' in sessdict:
                p1 = sessdict['1']
                p2 = sessdict['2']
                p3 = sessdict['3']
                if use_unsmoothed:
                    p1 = try_map_to_unsmoothed(p1) or p1
                    p2 = try_map_to_unsmoothed(p2) or p2
                    p3 = try_map_to_unsmoothed(p3) or p3
                entries.append(((p1, p2), (p2, p3)))
                subj_ids.append(subj)
                subj_groups.append(pick_group(subj))
            else:
                warnings.warn(f"Subject {subj} missing one of sessions 1/2/3 (skipping)")
        else:
            raise ValueError(f"Unknown delta_type: {delta_type}")

    return subj_ids, subj_groups, entries


def compute_delta_arrays(entries, delta_type='d21'):
    """Compute numpy arrays and representative imgs from entries built for delta_type.
    entries format depends on delta_type (see build_subject_delta_entries_for_type docstring).
    Returns (arrays, imgs, is_concat) where is_concat=True when entries represent concatenated volumes (for concat mode).
    """
    arrays = []
    imgs = []
    is_concat = False

    if delta_type in ('d21', 'd31', 'd32'):
        for p1, p2 in entries:
            img1 = nib.load(p1)
            img2 = nib.load(p2)
            a1 = img1.get_fdata(dtype=np.float32)
            a2 = img2.get_fdata(dtype=np.float32)
            delta = a2 - a1
            arrays.append(delta)
            imgs.append(img2)
    elif delta_type == 'dsec':
        for p1, p2, p3 in entries:
            img1 = nib.load(p1)
            img2 = nib.load(p2)
            img3 = nib.load(p3)
            a1 = img1.get_fdata(dtype=np.float32)
            a2 = img2.get_fdata(dtype=np.float32)
            a3 = img3.get_fdata(dtype=np.float32)
            delta = a3 - 2.0 * a2 + a1
            arrays.append(delta)
            imgs.append(img3)
    elif delta_type == 'concat':
        # entries are ((p1,p2), (p2,p3)) per subject; we will return arrays in order: all D21 then all D32
        d21_list = []
        d32_list = []
        imgs_d21 = []
        imgs_d32 = []
        for (p1p2, p2p3) in entries:
            p1, p2 = p1p2
            p2b, p3 = p2p3
            # load D21
            img1 = nib.load(p1)
            img2 = nib.load(p2)
            a1 = img1.get_fdata(dtype=np.float32)
            a2 = img2.get_fdata(dtype=np.float32)
            d21 = a2 - a1
            d21_list.append(d21)
            imgs_d21.append(img2)
            # load D32
            img2b = nib.load(p2b)
            img3 = nib.load(p3)
            a2b = img2b.get_fdata(dtype=np.float32)
            a3 = img3.get_fdata(dtype=np.float32)
            d32 = a3 - a2b
            d32_list.append(d32)
            imgs_d32.append(img3)
        # concatenate lists: all D21 then all D32
        arrays = d21_list + d32_list
        imgs = imgs_d21 + imgs_d32
        is_concat = True
    else:
        raise ValueError(f"Unknown delta_type: {delta_type}")

    return arrays, imgs, is_concat


def run_classification(X, y, mask_img, outdir, n_permutations=1000, cv_folds=5, k_best=5000, random_state=42, classifier_name='logistic', n_jobs=1, class_weight='none'):
    # X: subject x voxels numpy
    # y: labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Pipeline: scale -> selectKBest -> logistic
    # Pipeline: scale -> selectKBest -> classifier (select based on user choice)
    if classifier_name == 'logistic':
        # choose solver depending on number of classes to avoid liblinear multiclass deprecation
        n_classes = len(np.unique(y_enc))
        if n_classes > 2:
            # use a multinomial-capable solver (lbfgs) for true multiclass problems
            # do not explicitly set `multi_class` (it will default to multinomial in recent sklearn)
            if class_weight == 'balanced':
                clf_core = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=2000, class_weight='balanced')
            else:
                clf_core = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=2000)
        else:
            if class_weight == 'balanced':
                clf_core = LogisticRegression(solver='liblinear', penalty='l2', max_iter=1000, class_weight='balanced')
            else:
                clf_core = LogisticRegression(solver='liblinear', penalty='l2', max_iter=1000)
    elif classifier_name == 'svc':
        if class_weight == 'balanced':
            clf_core = LinearSVC(max_iter=10000, class_weight='balanced')
        else:
            clf_core = LinearSVC(max_iter=10000)
    elif classifier_name == 'linear_svc':
        if class_weight == 'balanced':
            clf_core = LinearSVC(max_iter=10000, class_weight='balanced')
        else:
            clf_core = LinearSVC(max_iter=10000)
    elif classifier_name == 'rbf_svc':
        if class_weight == 'balanced':
            clf_core = SVC(kernel='rbf', max_iter=10000, random_state=random_state, class_weight='balanced')
        else:
            clf_core = SVC(kernel='rbf', max_iter=10000, random_state=random_state)
    elif classifier_name == 'rf':
        if class_weight == 'balanced':
            clf_core = RandomForestClassifier(n_estimators=500, random_state=random_state, class_weight='balanced')
        else:
            clf_core = RandomForestClassifier(n_estimators=500, random_state=random_state)
    elif classifier_name == 'gb':
        if class_weight == 'balanced':
            warnings.warn('GradientBoostingClassifier does not support class_weight; ignoring --class-weight')
        clf_core = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
    elif classifier_name == 'nb':
        if class_weight == 'balanced':
            warnings.warn('GaussianNB does not support class_weight; ignoring --class-weight')
        clf_core = GaussianNB()
    elif classifier_name == 'knn':
        if class_weight == 'balanced':
            warnings.warn('KNeighborsClassifier does not support class_weight; ignoring --class-weight')
        clf_core = KNeighborsClassifier(n_neighbors=5)
    else:
        if class_weight == 'balanced':
            clf_core = LogisticRegression(solver='liblinear', penalty='l2', max_iter=1000, class_weight='balanced')
        else:
            clf_core = LogisticRegression(solver='liblinear', penalty='l2', max_iter=1000)

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(f_classif, k=min(k_best, X.shape[1]))),
        ('clf', clf_core)
    ])

    # cross-validated accuracy
    print('Running cross-validated scoring (fast)')
    scores = cross_val_score(clf, X, y_enc, cv=cv, scoring='accuracy', n_jobs=n_jobs)

    # cross-validated predictions for classification report
    y_pred_cv = cross_val_predict(clf, X, y_enc, cv=cv)

    # permutation test (custom loop so we can show progress)
    print(f'Running permutation test with n_permutations={n_permutations} (this may take some time)')
    perm_scores = []
    start_time = time.time()
    report_every = max(1, n_permutations // 20)
    rng = np.random.RandomState(random_state)
    for i in range(n_permutations):
        perm_y = rng.permutation(y_enc)
        # use cross_val_score on the same pipeline; allow inner parallelism via n_jobs
        sc = cross_val_score(clf, X, perm_y, cv=cv, scoring='accuracy', n_jobs=n_jobs)
        perm_scores.append(float(np.mean(sc)))
        if (i + 1) % report_every == 0 or i == n_permutations - 1:
            elapsed = time.time() - start_time
            per = (i + 1) / n_permutations * 100.0
            eta = elapsed / (i + 1) * (n_permutations - (i + 1)) if (i + 1) > 0 else 0
            print(f'  Permutations: {i+1}/{n_permutations} ({per:.1f}%) elapsed {elapsed:.0f}s eta {eta:.0f}s')
    perm_scores = np.array(perm_scores)
    perm_score = float(np.mean(perm_scores))
    # pvalue: proportion of permutation scores >= true score
    true_score = float(np.mean(scores))
    pvalue = float((np.sum(perm_scores >= true_score) + 1) / (len(perm_scores) + 1))

    # fit final classifier on all data
    clf.fit(X, y_enc)

    # extract coefficients/importances: from select and clf
    support_mask = clf.named_steps['select'].get_support()
    coefs = np.zeros(X.shape[1], dtype=np.float32)
    final_clf = clf.named_steps['clf']
    if hasattr(final_clf, 'coef_'):
        vals = final_clf.coef_.ravel()
        # ensure vals has the same length as support_mask.sum()
        if len(vals) == support_mask.sum():
            coefs[support_mask] = vals
        else:
            # fallback: normalize or truncate if mismatch
            coefs[support_mask] = vals[:support_mask.sum()]
    elif hasattr(final_clf, 'feature_importances_'):
        vals = final_clf.feature_importances_
        if len(vals) == support_mask.sum():
            coefs[support_mask] = vals
        else:
            coefs[support_mask] = vals[:support_mask.sum()]
    else:
        # fallback: zeros
        coefs[:] = 0.0

    # save summary
    summary = {
        'classes': le.classes_.tolist(),
        'cv_accuracy_mean': float(np.mean(scores)),
        'cv_accuracy_std': float(np.std(scores)),
        'permutation_score': float(perm_score),
        'permutation_pvalue': float(pvalue)
    }

    with open(os.path.join(outdir, 'ml_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Generate and save classification report
    class_report = classification_report(y_enc, y_pred_cv, target_names=le.classes_, digits=3)
    with open(os.path.join(outdir, 'classification_report.txt'), 'w') as f:
        f.write("Classification Report (Cross-Validated Predictions)\n")
        f.write("=" * 60 + "\n\n")
        f.write(class_report)
        f.write("\n\n")
        f.write(f"Mean CV Accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}\n")
        f.write(f"Permutation Score: {perm_score:.3f}\n")
        f.write(f"Permutation P-value: {pvalue:.4f}\n")

    # Also generate confusion matrix
    cm = confusion_matrix(y_enc, y_pred_cv)
    with open(os.path.join(outdir, 'confusion_matrix.txt'), 'w') as f:
        f.write("Confusion Matrix (Cross-Validated Predictions)\n")
        f.write("=" * 60 + "\n\n")
        f.write("Rows: True labels, Columns: Predicted labels\n")
        f.write("Classes: " + ", ".join(le.classes_) + "\n\n")
        np.savetxt(f, cm, fmt='%d', delimiter='\t')

    return coefs, summary


def main():
    parser = argparse.ArgumentParser(description='VBM ML interaction prototype (nilearn+sklearn)')
    parser.add_argument('--design-json', required=False, help='Path to design.json')
    parser.add_argument('--mask', default='templates/brainmask_GMtight.nii', help='Path to brain mask NIfTI')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--n-permutations', type=int, default=1000, help='Permutation tests')
    parser.add_argument('--cv-folds', type=int, default=5, help='CV folds')
    parser.add_argument('--k-best', type=int, default=5000, help='SelectKBest features')
    parser.add_argument('--participants-tsv', help='BIDS participants.tsv file (optional)')
    parser.add_argument('--data-root', help='Path to subject data root (used with participants.tsv). e.g. /Volumes/Thunder/129_PK01/cat12/data/cat12')
    parser.add_argument('--classifier', choices=['logistic', 'svc', 'linear_svc', 'rbf_svc', 'rf', 'gb', 'nb', 'knn'], default='logistic', help='Classifier to use.')
    parser.add_argument('--group-col', help='Column name or 1-based index for group in participants.tsv.')
    parser.add_argument('--session1-tokens', nargs='+', default=['ses-1', 'ses1', 'ses-01'], help='Tokens to identify session 1 files.')
    parser.add_argument('--session2-tokens', nargs='+', default=['ses-2', 'ses2', 'ses-02'], help='Tokens to identify session 2 files.')
    parser.add_argument('--session-a', help='Session id (as in design.json) to use as first timepoint (e.g., 1)')
    parser.add_argument('--session-b', help='Session id (as in design.json) to use as second timepoint (e.g., 2)')
    parser.add_argument('--use-unsmoothed', action='store_true', help='Prefer unsmoothed images (mwp1*) when available by attempting filename substitution')
    parser.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs to use for cross-validation (passed to cross_val_score)')
    parser.add_argument('--class-weight', choices=['balanced', 'none'], default='none', help='If "balanced", pass class_weight="balanced" to classifiers that support it')
    parser.add_argument('--delta-type', choices=['d21', 'd32', 'd31', 'dsec', 'concat'], default='d21', help='Which per-subject delta to compute: d21=(T2-T1), d32=(T3-T2), d31=(T3-T1), dsec=(T3-2*T2+T1), concat=(D21 and D32 concatenated features)')
    parser.add_argument('--merge-interventions', action='store_true', help='Merge any group containing "intervention" into a single "intervention" group')
    parser.add_argument('--min-finite-prop', type=float, default=0.9, help='Minimum proportion (0-1) of subjects with finite voxel values required to keep a voxel (default 0.9)')

    args = parser.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    print('VBM ML interaction: starting')
    print(f'Output directory: {outdir}')

    # load design: prefer explicit design.json; else, if results folder contains design.json, use it; else use participants.tsv + data_root
    design = None
    if args.design_json and os.path.isfile(args.design_json):
        design = load_design(args.design_json)
        print(f'Using design from: {args.design_json}')
    else:
        # If user provided participants TSV + data root, prefer building design from that
        if args.participants_tsv and args.data_root:
            participants = load_participants_tsv(args.participants_tsv, group_col=args.group_col)
            # determine which session ids to search for when building design from data_root
            if args.session_a is not None and args.session_b is not None:
                session_ids = [str(args.session_a), str(args.session_b)]
            else:
                # Determine which session ids to look for based on requested delta_type.
                # If delta_type requests non-default sessions (e.g. d31 or d32 or dsec/concat),
                # ensure we search for all required sessions so the design contains them.
                if getattr(args, 'delta_type', None) in ('d21', None):
                    session_ids = ['1', '2']
                elif args.delta_type == 'd31':
                    session_ids = ['1', '3']
                elif args.delta_type == 'd32':
                    session_ids = ['2', '3']
                elif args.delta_type in ('dsec', 'concat'):
                    session_ids = ['1', '2', '3']
                else:
                    # fallback to legacy behaviour
                    session_ids = ['1', '2']
            design = find_images_for_subjects(participants, args.data_root, session_ids=session_ids)
            print(f'Built design from participants TSV: {args.participants_tsv} (group_col={args.group_col})')
        else:
            # fallback: check current working dir for a design.json
            cwd_design = os.path.join(os.getcwd(), 'design.json')
            if os.path.isfile(cwd_design):
                design = load_design(cwd_design)
                print(f'Using design from cwd: {cwd_design}')
            else:
                raise SystemExit('No design.json found and participants/data-root not provided; aborting')
    # provide early feedback to the user about available files and choices
    n_files = len(design.get('files', [])) if design else 0
    print(f'Design entries: {n_files}')
    if args.mask and not os.path.isfile(args.mask):
        warnings.warn(f"Mask file not found: {args.mask}")
    if getattr(args, 'participants_tsv', None) and not os.path.isfile(args.participants_tsv):
        warnings.warn(f"Participants TSV not found: {args.participants_tsv}")

    # If a participants TSV is provided, build a subject->group map and override design groups
    if getattr(args, 'participants_tsv', None) and os.path.isfile(args.participants_tsv):
        participants = load_participants_tsv(args.participants_tsv, group_col=args.group_col)
        subj_group_map = {p['subject']: p.get('group') for p in participants}
        replaced = 0
        for e in design.get('files', []):
            sub = e.get('subject')
            if sub in subj_group_map and subj_group_map[sub] is not None:
                if e.get('group') != subj_group_map[sub]:
                    e['group'] = subj_group_map[sub]
                    replaced += 1
        print(f'Overrode group labels from participants.tsv for {replaced} design entries')

    # build a subject->group map if participants TSV provided
    subj_group_map = None
    if getattr(args, 'participants_tsv', None) and os.path.isfile(args.participants_tsv):
        participants = load_participants_tsv(args.participants_tsv, group_col=args.group_col)
        subj_group_map = {p['subject']: p.get('group') for p in participants}

    # Build appropriate subject entries for the requested delta type (d21,d32,d31,dsec,concat)
    subj_ids, subj_groups, subj_entries = build_subject_delta_entries_for_type(design, delta_type=args.delta_type, use_unsmoothed=args.use_unsmoothed, group_map=subj_group_map, session_a=args.session_a, session_b=args.session_b)

    # Optionally merge intervention groups into one label
    if args.merge_interventions:
        new_groups = []
        for g in subj_groups:
            if g is None:
                new_groups.append(g)
            else:
                if 'intervention' in str(g).lower():
                    new_groups.append('intervention')
                else:
                    new_groups.append(g)
        subj_groups = new_groups

    # Validate group labels: remove subjects with missing group labels and
    # ensure there are at least two groups. Also adjust CV folds if any
    # class has fewer samples than requested folds (to avoid StratifiedKFold errors).
    from collections import Counter

    # identify indices with missing group
    missing_idx = [i for i, g in enumerate(subj_groups) if g is None or str(g).strip() == '']
    if missing_idx:
        warnings.warn(f"Found {len(missing_idx)} subjects with missing group labels; these subjects will be skipped")
        # remove entries at these indices from subj_ids, subj_groups, subj_entries
        keep_idx = [i for i in range(len(subj_ids)) if i not in missing_idx]
        subj_ids = [subj_ids[i] for i in keep_idx]
        subj_groups = [subj_groups[i] for i in keep_idx]
        subj_entries = [subj_entries[i] for i in keep_idx]

    # ensure we have at least two groups
    grp_counts = Counter(subj_groups)
    n_groups = len(grp_counts)
    if n_groups < 2:
        raise SystemExit(f'Found {n_groups} unique group(s) after processing; need at least 2 groups to run classification; groups found: {list(grp_counts.keys())}')

    # adjust CV folds if some classes have fewer samples than requested folds
    min_class_count = min(grp_counts.values())
    cv_folds_requested = int(getattr(args, 'cv_folds', 5))
    cv_folds = cv_folds_requested
    if min_class_count < cv_folds_requested:
        warnings.warn(f"Minimum class count ({min_class_count}) is smaller than requested cv_folds ({cv_folds_requested}); reducing cv_folds to {min_class_count}")
        cv_folds = max(2, min_class_count)
    else:
        cv_folds = cv_folds_requested

    if len(subj_ids) == 0:
        raise SystemExit('No subjects with required sessions found; aborting')

    arrays, imgs, is_concat = compute_delta_arrays(subj_entries, delta_type=args.delta_type)

    # create NiftiMasker and transform
    masker = NiftiMasker(mask_img=args.mask)
    masker.fit()

    # create temporary 4D image from arrays
    # stack arrays into 4D nibabel image using affine from imgs[0]
    # stack arrays into 4D nibabel image using affine from first available img
    stacked = np.stack(arrays, axis=-1)
    # pick affine from first non-None img, else fall back to identity
    affine = None
    for im in imgs:
        if im is not None:
            affine = im.affine
            break
    if affine is None:
        affine = np.eye(4)
    temp_img = nib.Nifti1Image(stacked, affine)

    X_all = masker.transform(temp_img)

    # remove voxels that are constant across samples or contain too many NaNs/inf
    # compute validity counts across the transformed 2D array (samples x voxels)
    # this handles both normal and concat temporary stacking shapes
    finite_counts = np.isfinite(X_all).sum(axis=0)
    # require a voxel to be finite in at least `min_finite_prop` of samples
    min_prop = float(getattr(args, 'min_finite_prop', 0.9))
    min_count = max(1, int(min_prop * float(X_all.shape[0])))
    finite_mask = finite_counts >= min_count
    # variance check: compute in float64 using the numerically stable
    # mean-of-squares minus square-of-mean to avoid any platform-specific
    # nanvar/float32 issues.
    Xa = X_all.astype(np.float64)
    mean_sq = np.nanmean(Xa * Xa, axis=0)
    mean = np.nanmean(Xa, axis=0)
    var = mean_sq - (mean * mean)
    valid_var = var > 1e-8
    voxel_valid_mask = finite_mask & valid_var
    n_valid = int(np.sum(voxel_valid_mask))
    # diagnostics
    try:
        n_finite_total = int(np.sum(finite_counts > 0))
    except Exception:
        n_finite_total = 0
    print(f'Voxel filtering: total voxels={X_all.shape[1]}, finite_in_at_least_1={n_finite_total}, finite_threshold_count={min_count}, voxels_passing_finite_count={int(np.sum(finite_mask))}, voxels_passing_variance={int(np.sum(valid_var))}, voxels_final={n_valid}')
    if n_valid == 0:
        raise RuntimeError('No valid voxels found after filtering constant/NaN columns (try lowering --min-finite-prop or inspect input images for NaNs)')

    if is_concat:
        # X_all has shape (n_subjects*2, voxels) where first n are D21, next n are D32
        nsub = len(subj_ids)
        if X_all.shape[0] != 2 * nsub:
            raise RuntimeError('Unexpected shape for concat-mode transformed data')
        # filter voxels, then split and concatenate features per-subject
        X_all_filtered = X_all[:, voxel_valid_mask]
        first_half = X_all_filtered[:nsub, :]
        second_half = X_all_filtered[nsub:, :]
        X = np.concatenate([first_half, second_half], axis=1)
    else:
        X = X_all[:, voxel_valid_mask]

    coefs_reduced, summary = run_classification(X, subj_groups, args.mask, outdir,
                                                n_permutations=args.n_permutations,
                                                cv_folds=cv_folds,
                                                k_best=args.k_best,
                                                classifier_name=args.classifier,
                                                n_jobs=args.n_jobs,
                                                class_weight=args.class_weight)

    # Map reduced-space coefficients back to full masker voxel space
    # voxel_valid_mask is boolean per original voxel positions
    if is_concat:
        # coefs_reduced length should be 2 * n_valid
        m = int(np.sum(voxel_valid_mask))
        if len(coefs_reduced) != 2 * m:
            # fallback: if lengths mismatch, try to write whatever we can
            warnings.warn('Unexpected coefficient length for concat mode; saving reduced coefficients as text')
            with open(os.path.join(outdir, 'coefs_reduced.txt'), 'w') as f:
                np.savetxt(f, np.atleast_1d(coefs_reduced))
        else:
            c1 = coefs_reduced[:m]
            c2 = coefs_reduced[m:]
            full1 = np.zeros(voxel_valid_mask.shape[0], dtype=np.float32)
            full2 = np.zeros(voxel_valid_mask.shape[0], dtype=np.float32)
            full1[voxel_valid_mask] = c1
            full2[voxel_valid_mask] = c2
            # save two separate coefficient maps for D21 and D32
            coef_img1 = masker.inverse_transform(full1)
            coef_img2 = masker.inverse_transform(full2)
            coef_path1 = outdir / 'coef_map_D21.nii.gz'
            coef_path2 = outdir / 'coef_map_D32.nii.gz'
            coef_img1.to_filename(str(coef_path1))
            coef_img2.to_filename(str(coef_path2))
            # plot both
            plt.figure(figsize=(8,6))
            display = plot_stat_map(str(coef_path1), title='ML coefficient map D21', display_mode='z', cut_coords=6)
            display.savefig(str(outdir / 'coef_map_D21.png'))
            plt.figure(figsize=(8,6))
            display = plot_stat_map(str(coef_path2), title='ML coefficient map D32', display_mode='z', cut_coords=6)
            display.savefig(str(outdir / 'coef_map_D32.png'))
    else:
        # single-delta mode: map back into full voxel space
        full_coefs = np.zeros(voxel_valid_mask.shape[0], dtype=np.float32)
        if len(coefs_reduced) != int(np.sum(voxel_valid_mask)):
            # if classifier returned unexpected length, save reduced and continue
            warnings.warn('Coefficient length does not match valid voxel count; saving reduced coefficients as text')
            with open(os.path.join(outdir, 'coefs_reduced.txt'), 'w') as f:
                np.savetxt(f, np.atleast_1d(coefs_reduced))
        else:
            full_coefs[voxel_valid_mask] = coefs_reduced
            coef_img = masker.inverse_transform(full_coefs)
            coef_path = outdir / 'coef_map.nii.gz'
            coef_img.to_filename(str(coef_path))
            # plot coef map
            plt.figure(figsize=(8,6))
            display = plot_stat_map(str(coef_path), title='ML coefficient map (delta maps)', display_mode='z', cut_coords=6)
            fig_path = outdir / 'coef_map.png'
            display.savefig(str(fig_path))

    print('Saved results to:', outdir)
    print('Summary:', summary)


if __name__ == '__main__':
    main()
