#!/usr/bin/env python3
"""Test group label substitution"""
from scipy.io import loadmat
import json
import numpy as np
import re

# Load config
with open('config/config_14_2_26.json') as f:
    config = json.load(f)
    group_labels = config.get('analysis', {}).get('group_labels', {})

# Load SPM.mat
spm = loadmat('results/vbm/vbm_9mm_3G_tiv_sex_age/SPM.mat', struct_as_record=False, squeeze_me=True)
xCon = spm['SPM'].xCon
if not isinstance(xCon, list):
    if isinstance(xCon, np.ndarray):
        xCon = xCon.tolist() if xCon.ndim == 1 else [xCon]
    else:
        xCon = [xCon]

# Show group labels
print('Group Labels from Config:')
for code, label in group_labels.items():
    print(f'  G{code} → {label}')

print('\n\nExample Contrasts WITH Labels Applied:')
print('=' * 100)

# Show examples
example_indices = [10, 14, 16, 19, 22, 28]
for idx in example_indices:
    original = str(xCon[idx].name)
    updated = original
    for group_code, group_label in group_labels.items():
        updated = re.sub(rf'\bG{group_code}\b', group_label, updated)
    
    if original != updated:
        print(f'{idx+1:2d}. {original:50s} →')
        print(f'    {updated}')
    else:
        print(f'{idx+1:2d}. {original} (no change)')
    print()
