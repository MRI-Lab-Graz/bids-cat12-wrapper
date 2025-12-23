#!/usr/bin/env python3
"""
Generate a sweep JSON config for ml_batch_runner covering all delta types and classifiers.

Writes `tmp/sweep_all_deltas.json` by default.

Usage:
  python utils/generate_sweep_config.py --out tmp/sweep_all_deltas.json

This script is conservative: phase1 permutations should be low (for coarse scan).
"""
import argparse
import json
from pathlib import Path


def build_config(outpath, participants_tsv='participants.tsv', data_root='/Volumes/Thunder/129_PK01/cat12/data/cat12', mask='templates/brainmask_GMtight.nii'):
    cfg = {
        'global': {
            # let the runner resolve python; use project-local venv if present
            'python_executable': '.venv_ml/bin/python',
            'script': 'utils/vbm_ml_interaction.py',
            'mask': mask,
            'participants_tsv': participants_tsv,
            'data_root': data_root,
            'group_col': 'group_ml',
            # sensible defaults for sweeps
            'class_weight': 'balanced',
            'cv_folds': 5,
        },
        'jobs': [],
        'sweeps': []
    }

    # delta types to test (TP1 vs TP2, TP1 vs TP3, TP2 vs TP3)
    delta_types = ['d21', 'd31', 'd32']

    # classifiers to try
    classifiers = ['logistic', 'svc', 'rf']

    # k-best feature counts (coarse)
    k_best_options = [500, 1000, 3000]

    # cv folds and class weights (some diversity)
    cv_folds = [5, 10]
    class_weights = ['balanced', 'none']

    # build a single sweep that expands across all combinations
    sweep = {
        'name_prefix': 'all_deltas_all_clf',
        'name_format': '{delta_type}_{classifier}_k{k_best}_cw{class_weight}_cv{cv_folds}',
        'params': {
            'delta_type': delta_types,
            'classifier': classifiers,
            'k_best': k_best_options,
            'class_weight': class_weights,
            'cv_folds': cv_folds,
            # keep group_col explicit at job level
            'group_col': ['group_ml'],
            # ensure we prefer unsmoothed images by default for these tests
            'use_unsmoothed': [True],
        }
    }

    cfg['sweeps'].append(sweep)

    # small manual job to test a baseline (k=1000 logistic)
    cfg['jobs'].append({
        'name': 'baseline_d21_logistic_k1000',
        'delta_type': 'd21',
        'classifier': 'logistic',
        'k_best': 1000,
        'n_permutations': 50,
        'class_weight': 'balanced',
        'cv_folds': 5,
        'group_col': 'group_ml',
        'use_unsmoothed': True,
    })

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(cfg, indent=2))
    print('Wrote sweep config to', str(outpath))
    return str(outpath)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='tmp/sweep_all_deltas.json')
    p.add_argument('--participants-tsv', default='participants.tsv')
    p.add_argument('--data-root', default='/Volumes/Thunder/129_PK01/cat12/data/cat12')
    p.add_argument('--mask', default='templates/brainmask_GMtight.nii')
    args = p.parse_args()
    build_config(args.out, participants_tsv=args.participants_tsv, data_root=args.data_root, mask=args.mask)


if __name__ == '__main__':
    main()
