#!/usr/bin/env python3
"""
Dump NIfTI/GIfTI header and affine info for beta and spm* files in a stats folder.
Usage:
  python3 dump_nifti_headers.py /path/to/stats --out screening_header_debug.txt

Outputs a simple text file listing filename, type, shape, affine matrix, voxel sizes and header details.
"""
import sys
import os
import argparse
from datetime import datetime

try:
    import nibabel as nb
except Exception as e:
    print('nibabel import failed:', e)
    sys.exit(2)

try:
    from nibabel import gifti
    HAVE_GIFTI = True
except Exception:
    HAVE_GIFTI = False

parser = argparse.ArgumentParser()
parser.add_argument('stats_folder')
parser.add_argument('--out', '-o', default='screening_header_debug.txt')
args = parser.parse_args()

stats_folder = args.stats_folder
out_file = args.out

files = []
for root, dirs, filenames in os.walk(stats_folder):
    for fn in filenames:
        if fn.startswith('beta_') and (fn.endswith('.nii') or fn.endswith('.nii.gz')):
            files.append(os.path.join(root, fn))
        if fn.startswith('spmT_') and (fn.endswith('.nii') or fn.endswith('.nii.gz')):
            files.append(os.path.join(root, fn))
        if fn.startswith('spmF_') and (fn.endswith('.nii') or fn.endswith('.nii.gz')):
            files.append(os.path.join(root, fn))
        # also check for unexpected gifti
        if fn.endswith('.gii'):
            files.append(os.path.join(root, fn))

files = sorted(files)
if not files:
    print('No beta/spmT/spmF or .gii files found under', stats_folder)

with open(out_file, 'w') as fo:
    fo.write('Header dump generated: %s\n' % datetime.now())
    fo.write('Stats folder: %s\n\n' % stats_folder)
    for f in files:
        fo.write('FILE: %s\n' % f)
        try:
            if f.endswith('.gii') and HAVE_GIFTI:
                g = gifti.read(f)
                fo.write('  TYPE: GIFTI\n')
                fo.write('  #DASETS: %d\n' % len(g.darrays))
            else:
                img = nb.load(f)
                fo.write('  TYPE: NIfTI\n')
                fo.write('  SHAPE: %s\n' % (str(getattr(img, 'shape', '<unknown>'))))
                try:
                    aff = img.affine
                    fo.write('  AFFINE:\n')
                    for row in aff:
                        fo.write('    %s\n' % (' '.join(['% .6f' % x for x in row])))
                except Exception as e:
                    fo.write('  AFFINE: <error reading affine: %s>\n' % e)
                try:
                    hdr = img.header
                    zo = hdr.get_zooms()
                    fo.write('  VOXEL_SIZES: %s\n' % (str(zo),))
                    fo.write('  DATATYPE: %s\n' % (hdr.get_data_dtype(),))
                except Exception as e:
                    fo.write('  HDR: <error reading header: %s>\n' % e)
        except Exception as e:
            fo.write('  ERROR reading file: %s\n' % e)
        fo.write('\n')

print('Wrote header dump to', out_file)
