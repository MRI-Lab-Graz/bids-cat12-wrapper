#!/usr/bin/env python3
"""
Fix contrasts.json to match SPM.mat indices
"""
from scipy.io import loadmat
import json
import numpy as np
import shutil
import os
import sys

def fix_contrasts_json(spm_mat_path):
    """Generate contrasts.json from SPM.mat with correct indices"""
    
    # Load SPM.mat
    spm = loadmat(spm_mat_path, struct_as_record=False, squeeze_me=True)
    
    xCon = spm['SPM'].xCon
    if not isinstance(xCon, list):
        if isinstance(xCon, np.ndarray):
            xCon = xCon.tolist() if xCon.ndim == 1 else [xCon]
        else:
            xCon = [xCon]
    
    # Create corrected contrasts.json
    contrasts_list = []
    for i, con in enumerate(xCon):
        contrasts_list.append({
            'index': i + 1,
            'name': str(con.name),
            'type': str(con.STAT)
        })
    
    # Write to file with backup
    stats_dir = os.path.dirname(spm_mat_path)
    old_file = os.path.join(stats_dir, 'contrasts.json')
    backup_file = os.path.join(stats_dir, 'contrasts.json.backup')
    
    if os.path.exists(old_file):
        shutil.copy(old_file, backup_file)
        print(f'✓ Backed up old contrasts.json to: {backup_file}')
    
    with open(old_file, 'w') as f:
        json.dump(contrasts_list, f, indent=2)
    
    print(f'\n✓ Created corrected contrasts.json with {len(contrasts_list)} contrasts')
    print('\nAll contrasts:')
    print('-' * 80)
    for c in contrasts_list:
        print(f"  {c['index']:2d}. [{c['type']}] {c['name']}")
    
    return contrasts_list

if __name__ == '__main__':
    if len(sys.argv) > 1:
        spm_path = sys.argv[1]
    else:
        spm_path = 'results/vbm/vbm_9mm_3G_tiv_sex_age/SPM.mat'
    
    fix_contrasts_json(spm_path)
