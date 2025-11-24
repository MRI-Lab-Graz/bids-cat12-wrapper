import os
import glob
import nibabel as nib
import numpy as np
import shutil
import argparse
import csv
from scipy.ndimage import label as scipy_label

def load_atlas_labels(csv_path):
    labels = {}
    if not os.path.exists(csv_path):
        return labels
    
    with open(csv_path, 'r') as f:
        # Check delimiter
        line = f.readline()
        delimiter = ';' if ';' in line else ','
        f.seek(0)
        
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            try:
                # Try different column names if needed, but based on file content:
                # ROIid;ROIabbr;ROIname;ROIcolor
                roi_id = int(row['ROIid'])
                roi_name = row['ROIname']
                labels[roi_id] = roi_name
            except (ValueError, KeyError):
                continue
    return labels

def get_label_at_coordinate(coord, atlas_img, atlas_data, labels):
    # coord is in world coordinates (x, y, z)
    # Convert to atlas voxel coordinates
    inv_affine = np.linalg.inv(atlas_img.affine)
    voxel_coord = nib.affines.apply_affine(inv_affine, coord)
    voxel_coord = np.round(voxel_coord).astype(int)
    
    # Check bounds
    if (0 <= voxel_coord[0] < atlas_data.shape[0] and
        0 <= voxel_coord[1] < atlas_data.shape[1] and
        0 <= voxel_coord[2] < atlas_data.shape[2]):
        
        label_id = int(atlas_data[tuple(voxel_coord)])
        return labels.get(label_id, "Unknown")
    else:
        return "Outside Atlas"

def check_and_sort_significant_results(directory, threshold=1.3, atlas_names=None, min_cluster_size=0):
    # Setup output directories
    base_output_dir = os.path.join(directory, 'sig_stats')
    pos_dir = os.path.join(base_output_dir, 'pos')
    neg_dir = os.path.join(base_output_dir, 'neg')
    
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    # Load Atlases
    atlases = []
    if atlas_names:
        for name in atlas_names:
            name = name.strip()
            atlas_path = f"templates/{name}.nii.gz"
            csv_path = f"templates/{name}.csv"
            
            if os.path.exists(atlas_path):
                try:
                    img = nib.load(atlas_path)
                    data = img.get_fdata()
                    labels = {}
                    if os.path.exists(csv_path):
                        labels = load_atlas_labels(csv_path)
                    
                    atlases.append({
                        'name': name,
                        'img': img,
                        'data': data,
                        'labels': labels
                    })
                    print(f"Loaded atlas: {name}")
                except Exception as e:
                    print(f"Failed to load atlas {name}: {e}")
            else:
                print(f"Atlas file not found: {atlas_path}")

    # Pattern to match the files
    pattern = os.path.join(directory, '*_log_pFWE_*.nii')
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return

    print(f"Checking {len(files)} files in {directory}...")
    print(f"Sorting significant results (> {threshold} to 'pos', < -{threshold} to 'neg')")
    print(f"Minimum cluster size: {min_cluster_size} voxels")
    print("=" * 100)

    count_pos = 0
    count_neg = 0
    
    for file_path in files:
        try:
            img = nib.load(file_path)
            data = img.get_fdata()
            affine = img.affine
            
            # Handle NaN values if any
            data_no_nan = data[~np.isnan(data)]
            
            if data_no_nan.size == 0:
                continue
            
            filename = os.path.basename(file_path)
            
            # Collect clusters for this file
            clusters = []

            # Positive Clusters
            mask_pos = data > threshold
            if np.any(mask_pos):
                labeled_array, num_features = scipy_label(mask_pos)
                for i in range(1, num_features + 1):
                    cluster_mask = labeled_array == i
                    cluster_size = np.sum(cluster_mask)
                    
                    if cluster_size < min_cluster_size:
                        continue

                    coords = np.argwhere(cluster_mask)
                    values = data[cluster_mask]
                    
                    # Find peak
                    max_idx_local = np.argmax(values)
                    max_val = values[max_idx_local]
                    peak_idx = tuple(coords[max_idx_local])
                    
                    peak_world = nib.affines.apply_affine(affine, peak_idx)
                    
                    clusters.append({
                        'id': i,
                        'type': 'POS',
                        'size': cluster_size,
                        'val': max_val,
                        'coord': peak_world,
                        'regions': [get_label_at_coordinate(peak_world, a['img'], a['data'], a['labels']) for a in atlases]
                    })

            # Negative Clusters
            mask_neg = data < -threshold
            if np.any(mask_neg):
                labeled_array, num_features = scipy_label(mask_neg)
                for i in range(1, num_features + 1):
                    cluster_mask = labeled_array == i
                    cluster_size = np.sum(cluster_mask)
                    
                    if cluster_size < min_cluster_size:
                        continue

                    coords = np.argwhere(cluster_mask)
                    values = data[cluster_mask]
                    
                    # Find peak (min)
                    min_idx_local = np.argmin(values)
                    min_val = values[min_idx_local]
                    peak_idx = tuple(coords[min_idx_local])
                    
                    peak_world = nib.affines.apply_affine(affine, peak_idx)
                    
                    clusters.append({
                        'id': i,
                        'type': 'NEG',
                        'size': cluster_size,
                        'val': min_val,
                        'coord': peak_world,
                        'regions': [get_label_at_coordinate(peak_world, a['img'], a['data'], a['labels']) for a in atlases]
                    })

            if clusters:
                print(f"File: {filename}")
                header = f"{'ID':<4} | {'Type':<4} | {'Size':<6} | {'Peak Val':<10} | {'Coordinate (x,y,z)':<25}"
                for atlas in atlases:
                    header += f" | {atlas['name']:<25}"
                print("-" * len(header))
                print(header)
                print("-" * len(header))
                
                for c in clusters:
                    coord_str = f"{c['coord'][0]:.1f}, {c['coord'][1]:.1f}, {c['coord'][2]:.1f}"
                    row_str = f"{c['id']:<4} | {c['type']:<4} | {c['size']:<6} | {c['val']:10.4f} | {coord_str:<25}"
                    for r in c['regions']:
                        row_str += f" | {r:<25}"
                    print(row_str)
                print("\n")
                
                # Copy file if it has significant clusters
                has_pos = any(c['type'] == 'POS' for c in clusters)
                has_neg = any(c['type'] == 'NEG' for c in clusters)
                
                if has_pos:
                    shutil.copy2(file_path, os.path.join(pos_dir, filename))
                    count_pos += 1
                if has_neg:
                    shutil.copy2(file_path, os.path.join(neg_dir, filename))
                    count_neg += 1
                
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")

    print("=" * 100)
    print(f"Sorting complete.")
    print(f"Files with positive significant clusters: {count_pos}")
    print(f"Files with negative significant clusters: {count_neg}")
    print(f"Output directory: {base_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check and sort significant results based on a threshold.")
    parser.add_argument("-t", "--threshold", type=float, default=1.3, help="Threshold for significance (default: 1.3)")
    parser.add_argument("-d", "--directory", type=str, default="./results/vbm/2groups_tiv_age_sex", help="Directory to check (default: ./results/vbm/2groups_tiv_age_sex)")
    parser.add_argument("--atlas", type=str, default="aal3", help="Comma-separated list of atlas names (e.g., 'aal3, cobra'). Files must be in templates/ folder.")
    parser.add_argument("-k", "--min-cluster-size", type=int, default=0, help="Minimum cluster size in voxels (default: 0)")
    
    args = parser.parse_args()
    
    if args.atlas.lower() == 'all':
        # Find all atlases in templates folder (pairs of .nii.gz and .csv)
        atlas_files = glob.glob("templates/*.nii.gz")
        atlas_names = []
        for f in atlas_files:
            name = os.path.basename(f).replace('.nii.gz', '')
            if os.path.exists(f"templates/{name}.csv"):
                atlas_names.append(name)
        atlas_names.sort()
    else:
        atlas_names = [name.strip() for name in args.atlas.split(',')] if args.atlas else []
    
    # Check if directory exists
    if os.path.exists(args.directory):
        check_and_sort_significant_results(args.directory, threshold=args.threshold, atlas_names=atlas_names, min_cluster_size=args.min_cluster_size)
    else:
        print(f"Directory not found: {args.directory}")
