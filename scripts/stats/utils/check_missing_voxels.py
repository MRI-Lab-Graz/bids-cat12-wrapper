#!/usr/bin/env python3
"""
check_missing_voxels.py

Detect voxels with a high fraction of missing values (NaN) across the images
listed in an SPM.mat. Saves a JSON summary, an exclusion mask NIfTI (1=exclude)
and a PNG thumbnail for quick inspection.

Usage:
    python3 check_missing_voxels.py --spm /path/to/SPM.mat --output-dir /path/to/results [--threshold 0.05]

The script is deliberately tolerant: if nibabel/matplotlib aren't available it will
emit informative messages and exit with code 0 so the pipeline can continue.
"""

import argparse
import importlib
import json
import os
import sys


def try_import(module_name, friendly_name=None):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        print(f"Warning: Could not import {friendly_name or module_name}: {e}")
        return None


nb = try_import('nibabel')
np = try_import('numpy')
plt = None
matplotlib = try_import('matplotlib')
if matplotlib is not None:
    try:
        matplotlib.use('Agg')
        plt = try_import('matplotlib.pyplot')
    except Exception as e:
        print('Warning: matplotlib backend setup failed:', e)
        plt = None


def parse_args():
    p = argparse.ArgumentParser(description='Check missing voxels across SPM.xY images')
    p.add_argument('--spm', required=True, help='Path to SPM.mat')
    p.add_argument('--output-dir', required=True, help='Results output directory')
    p.add_argument('--threshold', type=float, default=0.05, help='Fraction threshold to mark voxel as missing/excluded (default: 0.05)')
    p.add_argument('--fail-if-pct-excluded', type=float, default=None,
                   help='Exit with non-zero status if overall excluded voxels percentage exceeds this value (percent, e.g. 10)')
    p.add_argument('--gm-mask', type=str, default=None,
                   help='Path to a GM mask NIfTI. If provided, calculations are restricted to voxels inside this mask.')
    p.add_argument('--require-gm-mask', action='store_true',
                   help='Fail if a GM mask cannot be found or does not match the image grid.')
    return p.parse_args()


def load_spm_image_list(spm_path):
    # We read the SPM.mat file using scipy.io.loadmat if available as a fallback for
    # extracting SPM.xY.VY entries. If not available, we try a small MATLAB-style parse
    # by running MATLAB to print the list (but that complicates portability). For now
    # try scipy.io first.
    try:
        import scipy.io as sio
    except Exception:
        sio = None

    if sio is not None:
        try:
            m = sio.loadmat(spm_path, squeeze_me=True, struct_as_record=False)
            # navigate to SPM.xY.VY.fname - this structure can be nested; try robust access
            SPM = m.get('SPM', None)
            if SPM is None:
                raise RuntimeError('SPM variable not found in MAT file')
            # SPM might be a numpy.void with attributes
            xY = getattr(SPM, 'xY', None)
            if xY is None:
                # try dict-like access
                xY = SPM['xY'] if isinstance(SPM, dict) and 'xY' in SPM else None
            if xY is None:
                raise RuntimeError('SPM.xY not found in MAT file')

            # xY.VY is often an array of structs with field fname
            VY = None
            try:
                VY = getattr(xY, 'VY')
            except Exception:
                try:
                    VY = xY['VY']
                except Exception:
                    VY = None

            fnames = []
            if VY is None:
                raise RuntimeError('SPM.xY.VY not found')

            # VY may be a numpy object array; iterate and extract fname
            for v in np.atleast_1d(VY):
                try:
                    fn = getattr(v, 'fname')
                except Exception:
                    try:
                        fn = v['fname']
                    except Exception:
                        fn = None
                if fn is not None:
                    fnames.append(str(fn))

            return fnames
        except Exception as e:
            print(f"Warning: Failed to parse SPM.mat with scipy.io: {e}")

    # Fallback: try to call MATLAB to print filenames
    matlab = os.environ.get('MATLAB', None)
    if matlab is None:
        # try common locations
        for candidate in ["/Applications/MATLAB_R2025b.app/bin/matlab", "/usr/local/bin/matlab", "matlab"]:
            if os.path.exists(candidate):
                matlab = candidate
                break

    if matlab is not None:
        try:
            import subprocess
            cmd = [matlab, '-nodesktop', '-nodisplay', '-nosplash', '-batch', f"load('{spm_path}'); for i=1:length(SPM.xY.VY), disp(SPM.xY.VY(i).fname); end; exit;"]
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
            lines = [line.strip() for line in out.splitlines() if line.strip()]
            return lines
        except Exception as e:
            print(f"Warning: MATLAB fallback to read SPM.mat failed: {e}")

    raise RuntimeError('Could not extract image filenames from SPM.mat; install scipy or set MATLAB env')


def main():
    args = parse_args()
    spm_path = args.spm
    outdir = args.output_dir
    thresh = float(args.threshold)

    os.makedirs(outdir, exist_ok=True)

    if nb is None or np is None:
        print('Required Python packages (nibabel, numpy) are not available in this environment.')
        print('Skipping missing-voxel diagnostics.')
        return 0

    try:
        fnames = load_spm_image_list(spm_path)
    except Exception as e:
        print('Error: could not read image list from SPM.mat:', e)
        return 1

    print(f'Found {len(fnames)} images in SPM.xY.VY')

    # Load first image as template
    try:
        img0 = nb.load(fnames[0])
    except Exception as e:
        print('Error: failed to load first image', fnames[0], e)
        return 1

    data_shape = img0.shape
    affine = img0.affine if hasattr(img0, 'affine') else None

    # Attempt to load a GM mask if requested or available in templates/
    gm_mask_arr = None
    gm_mask_path = None
    if getattr(args, 'gm_mask', None):
        gm_mask_path = args.gm_mask
    else:
        # prefer a repository template if present
        default_template = os.path.join('templates', 'brainmask_GMtight.nii')
        if os.path.exists(default_template):
            gm_mask_path = default_template

    if gm_mask_path is not None:
        try:
            gm_img = nb.load(gm_mask_path)
            gm_mask_arr = gm_img.get_fdata(dtype=np.float32) != 0
            if gm_mask_arr.shape != data_shape:
                print(f'Warning: GM mask shape {gm_mask_arr.shape} != image data shape {data_shape}')
                if getattr(args, 'require_gm_mask', False):
                    print('ERROR: require-gm-mask is set but mask shape mismatches image template. Aborting.')
                    return 1
                else:
                    print('Ignoring GM mask due to shape mismatch and continuing with whole-image calculations')
                    gm_mask_arr = None
            else:
                print(f'Using GM mask: {gm_mask_path} (restricting calculations to mask voxels)')
        except Exception as e:
            print(f'Warning: failed to load GM mask {gm_mask_path}: {e}')
            if getattr(args, 'require_gm_mask', False):
                print('ERROR: require-gm-mask is set but GM mask could not be loaded. Aborting.')
                return 1
            gm_mask_arr = None

    # We'll accumulate a count of NaNs per voxel across images
    voxel_count = np.zeros(data_shape, dtype=np.int32)
    total = 0
    # Per-subject masks: mark voxels missing in any image of that subject
    subj_masks = {}
    subj_counts = {}
    for i,fn in enumerate(fnames):
        try:
            img = nb.load(fn)
            arr = img.get_fdata(dtype=np.float32)
            nanmask = np.isnan(arr)
            voxel_count += nanmask.astype(np.int32)
            total += 1
            # determine subject id from filename (look for sub-<id>)
            import re
            m = re.search(r'sub-[A-Za-z0-9_-]+', fn)
            if m:
                subj = m.group(0)
            else:
                subj = os.path.basename(fn)
            if subj not in subj_masks:
                subj_masks[subj] = np.zeros(data_shape, dtype=bool)
                subj_counts[subj] = 0
            subj_masks[subj] |= nanmask
            subj_counts[subj] += 1
            if (i+1) % 50 == 0:
                print(f'  Processed {i+1}/{len(fnames)} images...')
        except Exception as e:
            print(f'Warning: failed to read {fn}: {e}')

    if total == 0:
        print('No images processed. Exiting.')
        return 1

    frac_missing = voxel_count.astype(np.float32) / float(total)
    exclude_mask = frac_missing > thresh

    # If a GM mask is available, restrict exclusion stats and the saved mask to that region
    if gm_mask_arr is not None:
        n_voxels = int(gm_mask_arr.sum())
        # restrict exclude_mask and subject masks to GM
        exclude_mask = np.logical_and(exclude_mask, gm_mask_arr)
        n_excluded = int(exclude_mask.sum())
        pct_excluded = 100.0 * n_excluded / float(n_voxels) if n_voxels > 0 else 0.0
    else:
        n_voxels = int(np.prod(data_shape))
        n_excluded = int(exclude_mask.sum())
        pct_excluded = 100.0 * n_excluded / float(n_voxels)

    summary = {
        'n_images': int(total),
        'threshold': float(thresh),
        'n_voxels_total': int(n_voxels),
        'n_voxels_excluded': int(n_excluded),
        'pct_voxels_excluded': float(pct_excluded)
    }

    summary_file = os.path.join(outdir, 'missing_voxels_summary.json')
    with open(summary_file, 'w') as fh:
        json.dump(summary, fh, indent=2)

    print('✓ Written summary to', summary_file)

    # Save exclusion mask as NIfTI
    try:
        mask_img = nb.Nifti1Image(exclude_mask.astype(np.uint8), affine)
        mask_file = os.path.join(outdir, 'missing_voxels_mask.nii')
        nb.save(mask_img, mask_file)
        print('✓ Written mask to', mask_file)
    except Exception as e:
        print('Warning: failed to write mask NIfTI:', e)

    # Write per-subject missing summary CSV
    try:
        import csv
        subj_csv = os.path.join(outdir, 'missing_voxels_subjects.csv')
        with open(subj_csv, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['subject', 'n_images', 'n_voxels_missing_any', 'pct_voxels_missing_any'])
            for subj, mask in sorted(subj_masks.items()):
                if gm_mask_arr is not None:
                    mask = np.logical_and(mask, gm_mask_arr)
                n_missing = int(mask.sum())
                pct = 100.0 * n_missing / float(n_voxels) if n_voxels > 0 else 0.0
                writer.writerow([subj, int(subj_counts.get(subj, 0)), n_missing, round(pct, 4)])
        print('✓ Written per-subject missing CSV to', subj_csv)
    except Exception as e:
        print('Warning: failed to write per-subject CSV:', e)

    # Save a quick PNG of the middle axial slice of the fraction map
    if plt is not None:
        try:
            midz = data_shape[2] // 2
            slice_img = np.rot90(frac_missing[:, :, midz])
            plt.figure(figsize=(6,6))
            plt.imshow(slice_img, cmap='hot', vmin=0, vmax=1)
            plt.colorbar(label='Fraction missing')
            plt.title('Fraction missing (axial middle slice)')
            pngfile = os.path.join(outdir, 'missing_voxels_thumb.png')
            plt.savefig(pngfile, dpi=150, bbox_inches='tight')
            plt.close()
            print('✓ Written PNG thumbnail to', pngfile)
        except Exception as e:
            print('Warning: failed to write PNG thumbnail:', e)
    else:
        print('matplotlib not available; skipping thumbnail creation')

    # Optionally fail if overall excluded fraction exceeds user threshold
    fail_thresh = None
    try:
        # argparse sets this attribute if provided
        fail_thresh = float(args.fail_if_pct_excluded) if hasattr(args, 'fail_if_pct_excluded') and args.fail_if_pct_excluded is not None else None
    except Exception:
        # older arg name fallback
        fail_thresh = None

    if fail_thresh is not None:
        if pct_excluded > float(fail_thresh):
            print(f"ERROR: pct_voxels_excluded={pct_excluded:.3f}% > fail threshold {fail_thresh}%")
            return 2

    return 0


if __name__ == '__main__':
    sys.exit(main())
