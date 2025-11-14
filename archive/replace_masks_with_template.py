#!/usr/bin/env python3
"""Replace per-results mask files with the repo template mask.

Usage:
  python3 replace_masks_with_template.py --stats-root /path/to/stats [--template /path/to/template] [--dry-run]

This script will search under <stats-root>/results for files matching
mask*.nii* (case-insensitive), back each up with a timestamp suffix, and
copy the provided template mask into their place.
"""
import argparse
import glob
import os
import shutil
from datetime import datetime


def find_masks(stats_root):
    base = os.path.join(stats_root, 'results')
    pattern = os.path.join(base, '**', 'mask*.nii*')
    return sorted(glob.glob(pattern, recursive=True))


def main():
    p = argparse.ArgumentParser(description='Replace per-results mask files with repo template')
    p.add_argument('--stats-root', default='.', help='Path to stats/ folder (default: cwd)')
    p.add_argument('--template', default=None, help='Path to template mask (defaults to stats/templates/brainmask_GMtight.nii)')
    p.add_argument('--dry-run', action='store_true', help='Only list files that would be changed')
    args = p.parse_args()

    stats_root = os.path.abspath(args.stats_root)
    if args.template:
        template = os.path.abspath(args.template)
    else:
        template = os.path.join(stats_root, 'templates', 'brainmask_GMtight.nii')

    if not os.path.isfile(template):
        print(f'ERROR: Template mask not found: {template}')
        raise SystemExit(2)

    masks = find_masks(stats_root)
    if not masks:
        print('No per-results mask files found under results/.')
        return

    print(f'Found {len(masks)} mask file(s) under {os.path.join(stats_root, "results")}:')
    for m in masks:
        print('  -', m)

    if args.dry_run:
        print('\nDry-run complete. No files changed.')
        return

    for m in masks:
        # skip template itself if it appears under results/templates
        if os.path.abspath(m) == os.path.abspath(template):
            print(f'Skipping template file itself: {m}')
            continue
        t = datetime.now().strftime('%Y%m%d_%H%M%S')
        bak = f"{m}.bak.{t}"
        try:
            shutil.copy2(m, bak)
            shutil.copy2(template, m)
            print(f'Replaced: {m} (backup -> {bak})')
        except Exception as e:
            print(f'ERROR replacing {m}: {e}')

    print('\nReplacement complete.')


if __name__ == '__main__':
    main()
