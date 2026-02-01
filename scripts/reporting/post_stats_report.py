#!/usr/bin/env python3
"""
Generate interactive HTML report for CAT12 statistical results.

Features:
- Multi-level filtering (p-value, correction method, contrast, atlas)
- Real-time visualization updates
- Cluster gallery with top peaks
- Multi-atlas support (volume & surface)
- Glass-brain + surface mesh plotting
- Base64-embedded images (fully portable)

Usage:
    python post_stats_report.py ./results/vbm/analysis report.html
    python post_stats_report.py ./results/vbm/analysis report.html --filter tfce
    python post_stats_report.py ./results/vbm/analysis report.html --spm-path /path/to/spm12
"""

import os
import argparse
import glob
import nibabel as nib
import numpy as np
import pandas as pd
import base64
from io import BytesIO
from scipy.io import loadmat
from jinja2 import Template
from nilearn import plotting
import matplotlib.pyplot as plt
import nibabel.freesurfer.io as fsio
import xml.etree.ElementTree as ET
import json
from datetime import datetime
import re
from scipy import ndimage


def get_mni_coords(affine, vox_coords):
    """Convert voxel coordinates to MNI coordinates."""
    vox_coords_homo = np.append(vox_coords, 1)
    mni_coords = np.dot(affine, vox_coords_homo)
    return mni_coords[:3]


def get_vox_coords(affine, mni_coords):
    """Convert MNI coordinates to voxel coordinates."""
    inv_affine = np.linalg.inv(affine)
    mni_coords_homo = np.append(mni_coords, 1)
    vox_coords = np.dot(inv_affine, mni_coords_homo)
    return np.round(vox_coords[:3]).astype(int)


def load_atlas(atlas_path, labels_path):
    """Load atlas image and labels from various formats."""
    try:
        if not os.path.exists(atlas_path):
            return None, None, None
        
        atlas_img = nib.load(atlas_path)
        atlas_data = atlas_img.get_fdata()
        atlas_affine = atlas_img.affine
        
        # Load labels
        labels = {}
        if not os.path.exists(labels_path):
            return atlas_data, atlas_affine, {}
        
        if labels_path.endswith('.csv'):
            df = pd.read_csv(labels_path, sep=';')
            for _, row in df.iterrows():
                labels[int(row['ROIid'])] = row['ROIname']
        elif labels_path.endswith('.txt'):
            with open(labels_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            labels[int(parts[0])] = " ".join(parts[1:])
                        except ValueError:
                            continue
        elif labels_path.endswith('.xml'):
            with open(labels_path, 'r', encoding='ISO-8859-1') as f:
                xml_content = f.read()
            # Fix unescaped ampersands in CAT12 XMLs
            xml_content = re.sub(r'&(?!(amp|lt|gt|apos|quot);)', '&amp;', xml_content)
            root = ET.fromstring(xml_content)
            for label in root.findall('.//label'):
                idx_elem = label.find('index')
                name_elem = label.find('name')
                if idx_elem is not None and name_elem is not None:
                    labels[int(idx_elem.text)] = name_elem.text
        
        return atlas_data, atlas_affine, labels
    except Exception as e:
        print(f"Warning: Could not load atlas {atlas_path} or labels {labels_path}: {e}")
        return None, None, None


def load_surface_atlas(lh_annot, rh_annot):
    """Load surface atlas labels from FreeSurfer annotation files."""
    try:
        if not os.path.exists(lh_annot) or not os.path.exists(rh_annot):
            return None, None
        lh_labels, _, lh_names = fsio.read_annot(lh_annot)
        rh_labels, _, rh_names = fsio.read_annot(rh_annot)
        lh_names = [n.decode('utf-8') for n in lh_names]
        rh_names = [n.decode('utf-8') for n in rh_names]
        return (lh_labels, lh_names), (rh_labels, rh_names)
    except Exception as e:
        print(f"Warning: Could not load surface atlas: {e}")
        return None, None


def plot_surface_to_base64(stat_map_path, mesh_lh, mesh_rh, bg_lh_data, bg_rh_data, title, threshold=1.301):
    """Generate a 4-view surface plot and return as base64 string."""
    try:
        gii = nib.load(stat_map_path)
        data = gii.darrays[0].data
        n_vertices = len(data)
        
        data_lh = data[:n_vertices//2]
        data_rh = data[n_vertices//2:]
        
        fig = plt.figure(figsize=(16, 10))
        
        # LH Lateral
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        plotting.plot_surf_stat_map(mesh_lh, data_lh, hemi='left', view='lateral', bg_map=bg_lh_data, axes=ax1, colorbar=False, threshold=threshold, darkness=0.5)
        ax1.set_title("LH Lateral", fontsize=14)
        
        # LH Medial
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        plotting.plot_surf_stat_map(mesh_lh, data_lh, hemi='left', view='medial', bg_map=bg_lh_data, axes=ax2, colorbar=False, threshold=threshold, darkness=0.5)
        ax2.set_title("LH Medial", fontsize=14)
        
        # RH Lateral
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        plotting.plot_surf_stat_map(mesh_rh, data_rh, hemi='right', view='lateral', bg_map=bg_rh_data, axes=ax3, colorbar=False, threshold=threshold, darkness=0.5)
        ax3.set_title("RH Lateral", fontsize=14)
        
        # RH Medial
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        plotting.plot_surf_stat_map(mesh_rh, data_rh, hemi='right', view='medial', bg_map=bg_rh_data, axes=ax4, colorbar=False, threshold=threshold, darkness=0.5)
        ax4.set_title("RH Medial", fontsize=14)
        
        fig.suptitle(title, fontsize=20, fontweight='bold')
        
        # Add a single colorbar
        max_val = np.nanmax(data) if np.any(~np.isnan(data)) else threshold + 1
        sm = plt.cm.ScalarMappable(cmap='cold_hot', norm=plt.Normalize(vmin=threshold, vmax=max_val if max_val > threshold else threshold + 1))
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(sm, cax=cbar_ax, label='-log10(p)')
        
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png', bbox_inches='tight', dpi=150)
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        plt.close(fig)
        return encoded
    except Exception as e:
        print(f"Warning: Could not plot surface {stat_map_path}: {e}")
        return None


def get_cluster_gallery(img, threshold, atlases=None, n_clusters=5):
    """Identify clusters and generate ortho plots for the peaks."""
    try:
        data = img.get_fdata()
        affine = img.affine
        
        # Threshold the data (use absolute for both directions)
        mask = np.abs(data) >= threshold
        labeled_array, num_features = ndimage.label(mask)
        
        if num_features == 0:
            return []
        
        # Get cluster sizes
        cluster_sizes = np.bincount(labeled_array.ravel())
        cluster_sizes[0] = 0  # ignore background
        
        # Find top clusters
        top_cluster_ids = np.argsort(cluster_sizes)[::-1]
        
        gallery = []
        count = 0
        for cid in top_cluster_ids:
            if cluster_sizes[cid] < 100 or count >= n_clusters: 
                continue
            
            # Find peak in cluster
            cluster_mask = (labeled_array == cid)
            # Use absolute values to find peak magnitude
            cluster_data = np.where(cluster_mask, np.abs(data), 0)
            peak_idx = np.unravel_index(np.argmax(cluster_data), data.shape)
            
            # Filter out peaks at the very edge of the image (often noise)
            is_edge = False
            for i, p_idx in enumerate(peak_idx):
                if p_idx <= 2 or p_idx >= data.shape[i] - 3:
                    is_edge = True
                    break
            
            if is_edge:
                continue
            
            peak_mni = get_mni_coords(affine, peak_idx)
            
            # Atlas Lookups
            region_mappings = {}
            all_unknown = True
            if atlases:
                for atl in atlases:
                    atlas_vox = get_vox_coords(atl['affine'], peak_mni)
                    region_name = "Unknown"
                    if all(0 <= atlas_vox[i] < atl['data'].shape[i] for i in range(3)):
                        region_id = int(atl['data'][tuple(atlas_vox)])
                        region_name = atl['labels'].get(region_id, f"Unknown (ID: {region_id})")
                    
                    if region_name and "Unknown" not in region_name:
                        all_unknown = False
                    region_mappings[atl['name']] = region_name
            
            # Skip if peak is in an unknown region (and we actually have atlases)
            if atlases and all_unknown:
                continue
            
            # Plot
            fig = plt.figure(figsize=(12, 3))
            plotting.plot_stat_map(img, cut_coords=peak_mni, display_mode='ortho', 
                                    colorbar=True, threshold=threshold,
                                    figure=fig,
                                    title=f"Cluster {cid} (Size: {cluster_sizes[cid]} voxels)",
                                    draw_cross=True, annotate=True)
            
            tmpfile = BytesIO()
            fig.savefig(tmpfile, format='png', bbox_inches='tight', dpi=100)
            encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
            plt.close(fig)
            
            gallery.append({
                'id': int(cid),
                'size': int(cluster_sizes[cid]),
                'peak_mni': [float(round(c, 2)) for c in peak_mni],
                'plot': encoded,
                'regions': region_mappings
            })
            count += 1
        return gallery
    except Exception as e:
        print(f"Warning: Cluster gallery generation failed: {e}")
        return []


def find_atlas_files(cat12_base, name, rel_nii, rel_xml):
    """Try to find atlas files in multiple locations."""
    nii_path = os.path.join(cat12_base, rel_nii)
    if not os.path.exists(nii_path):
        nii_path = os.path.join(cat12_base, "toolbox/cat12", rel_nii)
    
    xml_path = os.path.join(cat12_base, rel_xml)
    if not os.path.exists(xml_path):
        xml_path = os.path.join(cat12_base, "toolbox/cat12", rel_xml)
    
    return nii_path, xml_path


def generate_report(results_dir, output_html, filter_mode="all", spm_path=None):
    print(f"Generating post-stats report for: {results_dir}")
    filter_mode = (filter_mode or "all").lower()
    if filter_mode not in {"all", "tfce", "spmt", "double_threshold"}:
        print(f"Warning: Unknown filter_mode '{filter_mode}', defaulting to 'all'.")
        filter_mode = "all"
    print(f"Filter mode: {filter_mode}")
    
    if not os.path.isdir(results_dir):
        print(f"Error: {results_dir} is not a directory.")
        return
    
    # Detect if surface data
    is_surface = len(glob.glob(os.path.join(results_dir, "*.gii"))) > 0
    print(f"Mode: {'Surface' if is_surface else 'Volume'}")
    
    # Check for TFCE files (case-insensitive)
    tfce_files = glob.glob(os.path.join(results_dir, "[Tt][Ff][Cc][Ee]*"))
    
    # Fallback: check parent directory if no TFCE files found in results_dir
    search_dirs = [results_dir]
    if not tfce_files:
        parent_dir = os.path.dirname(results_dir.rstrip(os.sep))
        if parent_dir and os.path.isdir(parent_dir):
            parent_tfce = glob.glob(os.path.join(parent_dir, "[Tt][Ff][Cc][Ee]*"))
            if parent_tfce:
                print(f"Note: No TFCE files in {results_dir}, but found {len(parent_tfce)} in parent directory. Using those.")
                tfce_files = parent_tfce
                search_dirs = [results_dir, parent_dir]
    
    has_tfce = len(tfce_files) > 0
    if not has_tfce:
        print("Warning: No TFCE files found in this directory or its parent.")
    
    # Load SPM.mat
    spm_mat_path = os.path.join(results_dir, 'SPM.mat')
    contrast_names = {}
    contrast_types = {}  # T or F
    if os.path.exists(spm_mat_path):
        try:
            spm = loadmat(spm_mat_path, struct_as_record=False, squeeze_me=True)
            if hasattr(spm['SPM'], 'xCon'):
                xCon = spm['SPM'].xCon
                if not isinstance(xCon, (np.ndarray, list)):
                    xCon = [xCon]
                for i, con in enumerate(xCon):
                    contrast_names[i+1] = con.name
                    contrast_types[i+1] = con.STAT
        except Exception as e:
            print(f"Warning: Could not read SPM.mat: {e}")
    
    # Load contrasts.json if it exists
    contrasts_json_path = os.path.join(results_dir, 'contrasts.json')
    if os.path.exists(contrasts_json_path):
        try:
            with open(contrasts_json_path, 'r') as f:
                c_data = json.load(f)
                if isinstance(c_data, dict):
                    for k, v in c_data.items():
                        try:
                            contrast_names[int(k)] = v
                        except ValueError:
                            contrast_names[k] = v
                elif isinstance(c_data, list):
                    for i, item in enumerate(c_data):
                        if isinstance(item, dict) and 'name' in item:
                            idx = item.get('index', i + 1)
                            contrast_names[idx] = item['name']
                        else:
                            contrast_names[i+1] = item
        except Exception as e:
            print(f"Warning: Could not read contrasts.json: {e}")
    
    # Define Atlases
    cat12_base = spm_path if spm_path else "/Volumes/Evo/software/cat-12/external/matlab_tools/spm12"
    atlases = []
    bg_lh_data = None
    bg_rh_data = None
    mesh_lh = None
    mesh_rh = None
    
    if not is_surface:
        atlas_configs = [
            ("AAL3", "atlas/cat12_aal3.nii", "atlas/labels_cat12_aal3.xml"),
            ("Neuromorphometrics", "atlas/cat12_neuromorphometrics.nii", "atlas/labels_cat12_neuromorphometrics.xml"),
            ("Hammers", "atlas/cat12_hammers.nii", "atlas/labels_cat12_hammers.xml"),
            ("Schaefer 100", "atlas/cat12_Schaefer2018_100Parcels_17Networks_order.nii", "atlas/labels_cat12_Schaefer2018_100Parcels_17Networks_order.xml"),
            ("JulichBrain", "atlas/cat12_julichbrain.nii", "atlas/labels_cat12_julichbrain.xml")
        ]
        for name, rel_nii, rel_xml in atlas_configs:
            nii_path, xml_path = find_atlas_files(cat12_base, name, rel_nii, rel_xml)
            data, affine, labels = load_atlas(nii_path, xml_path)
            if data is not None:
                atlases.append({'name': name, 'data': data, 'affine': affine, 'labels': labels})
                print(f"Loaded atlas: {name}")
    else:
        atlas_configs = [
            ("DK40", "toolbox/cat12/atlases_surfaces_32k/lh.aparc_DK40.freesurfer.annot", "toolbox/cat12/atlases_surfaces_32k/rh.aparc_DK40.freesurfer.annot"),
            ("Destrieux", "toolbox/cat12/atlases_surfaces_32k/lh.aparc_a2009s.freesurfer.annot", "toolbox/cat12/atlases_surfaces_32k/rh.aparc_a2009s.freesurfer.annot"),
            ("HCP MMP1", "toolbox/cat12/atlases_surfaces_32k/lh.aparc_HCP_MMP1.freesurfer.annot", "toolbox/cat12/atlases_surfaces_32k/rh.aparc_HCP_MMP1.freesurfer.annot"),
            ("Schaefer 100", "toolbox/cat12/atlases_surfaces_32k/lh.Schaefer2018_100Parcels_17Networks_order.annot", "toolbox/cat12/atlases_surfaces_32k/rh.Schaefer2018_100Parcels_17Networks_order.annot")
        ]
        for name, lh_rel, rh_rel in atlas_configs:
            lh_path = os.path.join(cat12_base, lh_rel)
            rh_path = os.path.join(cat12_base, rh_rel)
            lh_atlas, rh_atlas = load_surface_atlas(lh_path, rh_path)
            if lh_atlas is not None:
                atlases.append({'name': name, 'lh': lh_atlas, 'rh': rh_atlas})
                print(f"Loaded surface atlas: {name}")
        
        # Meshes and Background maps
        mesh_lh = os.path.join(cat12_base, "toolbox/cat12/templates_surfaces_32k/lh.inflated.freesurfer.gii")
        mesh_rh = os.path.join(cat12_base, "toolbox/cat12/templates_surfaces_32k/rh.inflated.freesurfer.gii")
        bg_lh_path = os.path.join(cat12_base, "toolbox/cat12/templates_surfaces_32k/lh.sqrtsulc.freesurfer.gii")
        bg_rh_path = os.path.join(cat12_base, "toolbox/cat12/templates_surfaces_32k/rh.sqrtsulc.freesurfer.gii")
        
        try:
            bg_lh_data = nib.load(bg_lh_path).darrays[0].data
            bg_rh_data = nib.load(bg_rh_path).darrays[0].data
        except Exception as e:
            print(f"Warning: Could not load background maps: {e}")
    
    # Thresholds
    thresholds = [
        (0.01, 2.0, "Significant (p < 0.01)"),
        (0.05, 1.30103, "Significant (p < 0.05)"),
        (0.1, 1.0, "Trend (p < 0.1)"),
        (1.0, 0.0, "All Results")
    ]
    
    # Correction types
    ext_pattern = '.gii*' if is_surface else '.nii*'
    correction_patterns = {
        'FWE (Voxel)': [f'T_log_pFWE*{ext_pattern}', f'F_log_pFWE*{ext_pattern}'],
        'FDR (Voxel)': [f'T_log_pFDR*{ext_pattern}', f'F_log_pFDR*{ext_pattern}'],
        'FWE (TFCE)': [f'TFCE*FWE*{ext_pattern}'],
        'FDR (TFCE)': [f'TFCE*FDR*{ext_pattern}'],
        'Double Threshold': [f'*pk*{ext_pattern}', f'logP_*{ext_pattern}'],
        'Effect Size': [f'Cohen_d_*{ext_pattern}', f'd_map_*{ext_pattern}']
    }
    
    report_data = []
    
    # Find all relevant files
    for corr_name, patterns in correction_patterns.items():
        # Filter by mode
        if filter_mode == "double_threshold" and corr_name != "Double Threshold":
            continue
        if filter_mode == "tfce" and corr_name not in ["FWE (TFCE)", "FDR (TFCE)"]:
            continue
        if filter_mode == "spmt" and corr_name not in ["FWE (Voxel)", "FDR (Voxel)"]:
            continue
        
        files = []
        for s_dir in search_dirs:
            for p in patterns:
                found = glob.glob(os.path.join(s_dir, p))
                if found:
                    files.extend(found)
        
        # Remove duplicates and sort
        files = sorted(list(set(files)))
        
        # FILTER: If in Double Threshold, skip raw logP files
        if corr_name == "Double Threshold":
            pk_bases = [f.replace('_pkFWE5', '').replace('_pkFWE1', '').replace('_pkFWE10', '') 
                        for f in files if "PK" in f.upper()]
            files = [f for f in files if "PK" in f.upper() or f not in pk_bases]
            files = [f for f in files if "PK" in os.path.basename(f).upper() or 
                    not os.path.basename(f).upper().startswith("LOGP_")]
        
        if files:
            print(f"Found {len(files)} files for category: {corr_name}")
        
        for f in files:
            basename = os.path.basename(f)
            base_upper = basename.upper()
            
            # Determine extension of current file
            curr_ext = ""
            for e in ['.nii.gz', '.nii', '.gii']:
                if basename.lower().endswith(e):
                    curr_ext = e
                    break
            
            # Double-threshold specific parsing
            cluster_size = None
            is_bidirectional = False
            actual_p_fwe = None
            forming_threshold = None
            
            if "PK" in base_upper:
                k_match = re.search(r'_k(\d+)', basename)
                if k_match:
                    cluster_size = int(k_match.group(1))
                
                p_fwe_match = re.search(r'pkFWE(\d+)', basename)
                if p_fwe_match:
                    # In CAT12 pkFWE5 means p < 0.05
                    actual_p_fwe = int(p_fwe_match.group(1)) / 100.0
                
                forming_match = re.search(r'_p(0\.1|_001)', basename)
                if forming_match:
                    forming_threshold = "p < 0.001 (uncorr)"
                else:
                    forming_match_gen = re.search(r'_p(\d+\.?\d*)', basename)
                    if forming_match_gen:
                        try:
                            val = float(forming_match_gen.group(1))
                            forming_threshold = f"p < {val/100:.3g} (uncorr)"
                        except ValueError:
                            forming_threshold = f"p < {forming_match_gen.group(1)} (uncorr)"
                
                if "_bi" in basename.lower():
                    is_bidirectional = True
                
                display_corr = "Double Threshold"
            else:
                # Prevent pkFWE files from appearing in other lists
                if "PK" in base_upper:
                    continue
                display_corr = corr_name
            
            con_num = None
            
            # Try to parse con_num from TFCE_log_p..._0001.nii or spmT_0001.nii
            if any(x in base_upper for x in ['TFCE', 'SPMT', 'SPMF', 'T_LOG', 'F_LOG', 'COHEN', 'D_MAP']):
                try:
                    clean_name = basename
                    if curr_ext:
                        clean_name = basename[:-len(curr_ext)]
                    parts = clean_name.split('_')
                    # Handle cases like TFCE_log_pFWE_0001 or Cohen_d_0001
                    for part in reversed(parts):
                        try:
                            val = part.lstrip('0')
                            if not val and '0' in part:  # Handle '0000'
                                val = '0'
                            if val.isdigit():
                                con_num = int(val)
                                break
                        except ValueError:
                            continue
                except (ValueError, IndexError):
                    pass
            
            # If not found, try to match contrast name from filename
            if con_num is None:
                for num, name in contrast_names.items():
                    # Try exact match with underscores
                    cat12_style_name = name.replace(' ', '_')
                    if cat12_style_name in basename:
                        con_num = num
                        break
                    
                    # Try matching without colons
                    no_colon_name = name.replace(':', '_').replace(' ', '_')
                    if no_colon_name in basename:
                        con_num = num
                        break
            
            # Fallback: try more aggressive matching
            if con_num is None:
                for num, name in contrast_names.items():
                    clean_name = re.sub(r'[^a-zA-Z0-9]', '', name).lower()
                    clean_basename = re.sub(r'[^a-zA-Z0-9]', '', basename).lower()
                    if clean_name in clean_basename or clean_basename in clean_name:
                        con_num = num
                        break
            
            # Fallback for CAT12 default names
            if con_num is None:
                num_match = re.search(r'(?:condition|Group|Contrast)_(\d+)', basename)
                if num_match:
                    con_num = int(num_match.group(1))
            
            if con_num is None:
                print(f"Warning: Could not determine contrast number for {basename}. Skipping.")
                continue
            
            con_name = contrast_names.get(con_num, f"Contrast {con_num}")
            stat_type = contrast_types.get(con_num, "T")
            
            # Try to find the raw statistic file
            stat_file = None
            current_file_dir = os.path.dirname(f)
            for prefix in [f'spm{stat_type}_', f'{stat_type}_']:
                for e in [curr_ext, '.nii', '.nii.gz', '.gii']:
                    if not e:
                        continue
                    for d in [current_file_dir, results_dir]:
                        p = os.path.join(d, f"{prefix}{con_num:04d}{e}")
                        if os.path.exists(p):
                            stat_file = p
                            break
                    if stat_file:
                        break
                if stat_file:
                    break
            
            stat_img = nib.load(stat_file) if stat_file else None
            if is_surface:
                stat_data = stat_img.darrays[0].data if stat_img else None
            else:
                stat_data = stat_img.get_fdata() if stat_img else None
            
            img = nib.load(f)
            if is_surface:
                data = img.darrays[0].data
                affine = None
            else:
                data = img.get_fdata()
                affine = img.affine
            
            # Determine appropriate thresholds for this file
            current_thresholds_list = thresholds
            if actual_p_fwe is not None:
                # If double threshold (pkFWE), only show the specifically used level
                current_thresholds_list = [(actual_p_fwe, 0.0001, f"FWE (p < {actual_p_fwe})")]
            elif corr_name == "Effect Size":
                # For effect size, we only want one "threshold" (all results)
                current_thresholds_list = [(1.0, 0.2, "Cohen's d")]
            
            # For each threshold
            for p_val, log_p_thresh, p_label in current_thresholds_list:
                # Skip "All Results" for p-maps to prevent showing whole brain mask
                if p_val == 1.0 and corr_name != "Effect Size":
                    continue
                
                current_p_label = p_label
                current_log_p_thresh = log_p_thresh
                
                # Use absolute values for thresholding to catch both positive and negative effects
                abs_data = np.abs(data)
                mask = (~np.isnan(abs_data)) & (abs_data >= current_log_p_thresh)
                sig_elements = np.sum(mask)
                
                # Include all double-threshold results, even if empty, to show they were processed
                if sig_elements > 0 or display_corr == "Double Threshold":
                    region_mappings = {}
                    if sig_elements > 0:
                        # Find peak based on absolute magnitude
                        max_logp_abs = np.nanmax(abs_data[mask])
                        peak_idx = np.nanargmax(np.where(mask, abs_data, -np.inf))
                        
                        # Get the actual signed value at the peak
                        if is_surface:
                            peak_val = data[peak_idx]
                        else:
                            peak_idx_3d = np.unravel_index(peak_idx, data.shape)
                            peak_val = data[peak_idx_3d]
                        
                        if not is_surface:
                            peak_idx_3d = np.unravel_index(peak_idx, data.shape)
                            peak_mni = get_mni_coords(affine, peak_idx_3d)
                            peak_stat = stat_data[peak_idx_3d] if stat_data is not None else 0
                            
                            for atl in atlases:
                                atlas_vox = get_vox_coords(atl['affine'], peak_mni)
                                region_name = "Unknown"
                                if all(0 <= atlas_vox[i] < atl['data'].shape[i] for i in range(3)):
                                    region_id = int(atl['data'][tuple(atlas_vox)])
                                    region_name = atl['labels'].get(region_id, f"Unknown (ID: {region_id})")
                                region_mappings[atl['name']] = region_name
                        else:
                            peak_stat = stat_data[peak_idx] if stat_data is not None else 0
                            peak_mni = [0, 0, 0]
                            n_v = len(data)
                            
                            for atl in atlases:
                                region_name = "Unknown"
                                if peak_idx < n_v // 2:
                                    labels, names = atl['lh']
                                    region_id = labels[peak_idx]
                                    region_name = f"LH: {names[region_id]}"
                                else:
                                    labels, names = atl['rh']
                                    region_id = labels[peak_idx - n_v // 2]
                                    region_name = f"RH: {names[region_id]}"
                                region_mappings[atl['name']] = region_name
                    else:
                        max_logp_abs = 0.0
                        peak_val = 0.0
                        peak_stat = 0.0
                        peak_mni = [0, 0, 0]
                        for atl in atlases:
                            region_mappings[atl['name']] = "No significant clusters"
                    
                    # Direction Detection
                    has_pos = np.any(data[mask] > 1e-7) if sig_elements > 0 else False
                    has_neg = np.any(data[mask] < -1e-7) if sig_elements > 0 else False
                    
                    if has_pos and has_neg:
                        direction = "Bidirectional"
                        if peak_val > 0: direction += " (+ peak)"
                        else: direction += " (- peak)"
                    elif has_pos:
                        direction = "Positive"
                    elif has_neg:
                        direction = "Negative"
                    else:
                        direction = "No Effect"
                    
                    if corr_name == "Effect Size":
                        direction = "Positive (d)" if peak_val > 0 else "Negative (d)"
                    elif stat_type == "F" and not is_bidirectional:
                        direction = "Positive (F)"
                    
                    # Generate Cluster Gallery (Volume only)
                    cluster_gallery = []
                    if not is_surface and sig_elements > 0:
                        cluster_gallery = get_cluster_gallery(img, current_log_p_thresh, atlases=atlases, n_clusters=5)
                    
                    if display_corr == "Double Threshold" and is_bidirectional:
                        if has_pos and has_neg:
                            direction = "Two-sided (Mixed)"
                        elif has_pos:
                            direction = "Positive (Two-sided)"
                        elif has_neg:
                            direction = "Negative (Two-sided)"
                    
                    report_data.append({
                        'id': f"con_{con_num}_{corr_name.replace(' ', '_')}_{int(p_val*1000)}",
                        'con_num': con_num,
                        'con_name': con_name,
                        'correction': display_corr,
                        'orig_correction': corr_name,
                        'p_thresh': p_val,
                        'log_p_thresh': current_log_p_thresh,
                        'p_label': current_p_label,
                        'sig_voxels': int(sig_elements),
                        'max_logp': float(max_logp_abs),
                        'peak_stat': float(peak_stat),
                        'stat_type': stat_type,
                        'direction': direction,
                        'peak_mni': [float(round(c, 2)) for c in peak_mni] if not is_surface else "N/A",
                        'regions': region_mappings,
                        'cluster_size': cluster_size,
                        'forming_threshold': forming_threshold,
                        'cluster_gallery': cluster_gallery,
                        'file_path': f
                    })
    
    # Generate Plots
    plots = {}
    unique_combos = set((r['con_num'], r['correction'], r['file_path'], r['log_p_thresh']) for r in report_data)
    
    print(f"Generating {len(unique_combos)} threshold-specific plots...")
    for con_num, corr_name, f_path, log_p_thresh in unique_combos:
        img_id = f"{con_num}_{corr_name}_{log_p_thresh:.2f}"
        if not is_surface:
            try:
                img = nib.load(f_path)
                data = img.get_fdata()
                data = np.nan_to_num(data)
                clean_img = nib.Nifti1Image(data, img.affine, img.header)
                
                fig = plt.figure(figsize=(12, 5))
                plotting.plot_glass_brain(clean_img, display_mode='lyrz', colorbar=True,
                                          title=f"Con {con_num}: {corr_name} (p < {10**-log_p_thresh:.2f})",
                                          figure=fig, threshold=log_p_thresh, plot_abs=False)
                tmpfile = BytesIO()
                fig.savefig(tmpfile, format='png', bbox_inches='tight', dpi=120)
                encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
                plots[img_id] = encoded
                plt.close(fig)
            except Exception as e:
                print(f"Warning: Could not generate glass brain for {f_path}: {e}")
        else:
            encoded = plot_surface_to_base64(f_path, mesh_lh, mesh_rh, bg_lh_data, bg_rh_data, 
                                            f"Con {con_num}: {corr_name} (p < {10**-log_p_thresh:.2f})",
                                            threshold=log_p_thresh)
            if encoded:
                plots[img_id] = encoded
    
    corr_priority = {'FWE': 0, 'FDR': 1, 'Uncorrected': 2}
    report_data.sort(key=lambda x: (x['p_thresh'], corr_priority.get(x['correction'], 3), x['con_num']))
    
    # HTML Template (included inline)
    html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>CAT12 Interactive Post-Stats Report</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f8f9fa; color: #333; }
        h1 { color: #0056b3; border-bottom: 2px solid #0056b3; padding-bottom: 10px; }
        .info-container { display: flex; gap: 20px; margin-bottom: 30px; }
        .info { flex: 1; background-color: #e9ecef; padding: 15px; border-radius: 8px; }
        .plot-container { flex: 2; background-color: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; min-height: 200px; }
        #main-plot { max-width: 100%; height: auto; border-radius: 4px; }
        .controls { background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; display: flex; gap: 20px; align-items: center; flex-wrap: wrap; }
        .control-group { display: flex; flex-direction: column; gap: 5px; }
        select { padding: 8px; border-radius: 4px; border: 1px solid #ccc; min-width: 150px; background-color: #fff; }
        .warning-banner { background-color: #fff3cd; color: #856404; padding: 15px; border-radius: 8px; border: 1px solid #ffeeba; margin-bottom: 20px; font-weight: bold; }
        table { width: 100%; border-collapse: collapse; background-color: #fff; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #eee; }
        th { background-color: #007bff; color: white; font-weight: 600; text-transform: uppercase; font-size: 0.8em; cursor: pointer; }
        tr:hover { background-color: #f8f9fa; cursor: pointer; }
        tr.selected { background-color: #e7f1ff; border-left: 4px solid #007bff; }
        .badge { padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }
        .badge-fwe { background-color: #dc3545; color: white; }
        .badge-fdr { background-color: #ffc107; color: #212529; }
        .badge-dou { background-color: #6f42c1; color: white; }
        .badge-eff { background-color: #17a2b8; color: white; }
        .dir-pos { color: #28a745; font-weight: bold; }
        .dir-neg { color: #dc3545; font-weight: bold; }
        .coords { font-family: monospace; color: #666; font-size: 0.85em; }
        .region { font-weight: 500; color: #2c3e50; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <h1>CAT12 Interactive Post-Stats Report</h1>
    
    {% if not has_tfce %}
    <div class="warning-banner">
        [!] No TFCE results found. Showing standard statistic maps if available.
    </div>
    {% endif %}
    
    <div class="info-container">
        <div class="info">
            <p><strong>Results Directory:</strong> {{ results_dir }}</p>
            <p><strong>Generated on:</strong> {{ date }}</p>
            <p><strong>Mode:</strong> {{ mode }}</p>
            <p style="margin-top: 20px;"><small>Click any row in the table to update the visualization.</small></p>
        </div>
        <div class="plot-container">
            <div id="plot-title" style="font-weight: bold; margin-bottom: 10px; font-size: 1.2em;">Select a result to view plot</div>
            <img id="main-plot" src="" class="hidden">
            <div id="no-plot">No visualization available for this selection</div>
        </div>
    </div>
    
    <div id="gallery-section" class="hidden">
        <h2 style="color: #0056b3; border-bottom: 1px solid #ccc; padding-bottom: 5px;">Cluster Gallery (Top Peaks)</h2>
        <div id="gallery-content" style="display: flex; flex-direction: column; gap: 15px; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <!-- Clusters will be injected here -->
        </div>
    </div>
    
    <div class="controls" style="margin-top: 30px;">
        <div class="control-group">
            <label for="filter-p">Significance Level:</label>
            <select id="filter-p" onchange="filterTable()">
                <option value="all">All Levels</option>
                <option value="0.01">p < 0.01 (Significant)</option>
                <option value="0.05">p < 0.05 (Significant)</option>
                <option value="0.1">p < 0.1 (Trend)</option>
                <option value="1.0">p <= 1.0 (All)</option>
            </select>
        </div>
        <div class="control-group">
            <label for="filter-corr">Correction:</label>
            <select id="filter-corr" onchange="filterTable()">
                <option value="all">All Corrections</option>
                <option value="FWE (Voxel)">FWE (Voxel)</option>
                <option value="FDR (Voxel)">FDR (Voxel)</option>
                <option value="FWE (TFCE)">FWE (TFCE)</option>
                <option value="FDR (TFCE)">FDR (TFCE)</option>
                <option value="Double Threshold">Double Threshold</option>
                <option value="Effect Size">Effect Size</option>
            </select>
        </div>
        <div class="control-group">
            <label for="filter-con">Contrast:</label>
            <select id="filter-con" onchange="filterTable()">
                <option value="all">All Contrasts</option>
                {% for con_num in contrast_names.keys() | sort %}
                <option value="{{ con_num }}">{{ con_num }}: {{ contrast_names[con_num] }}</option>
                {% endfor %}
            </select>
        </div>
        <div class="control-group">
            <label for="select-atlas">Atlas Mapping:</label>
            <select id="select-atlas" onchange="updateAtlas()">
                {% for atl in atlases %}
                <option value="{{ atl.name }}">{{ atl.name }}</option>
                {% endfor %}
            </select>
        </div>
    </div>
    
    <table id="results-table">
        <thead>
            <tr>
                <th>Con #</th>
                <th>Contrast Name</th>
                <th>Correction</th>
                <th>P-Level</th>
                <th>Direction</th>
                <th>{{ 'Vertices' if is_surface else 'Voxels' }}</th>
                <th>Peak Stat</th>
                <th>Peak -log10(p)</th>
                <th>MNI Coords</th>
                <th>Region</th>
            </tr>
        </thead>
        <tbody>
            {% for row in report_data %}
            <tr class="result-row sig-{{ (row.p_thresh * 100) | int }}" 
                data-p="{{ row.p_thresh }}" 
                data-corr="{{ row.correction }}"
                data-con="{{ row.con_num }}"
                data-img-id="{{ row.con_num }}_{{ row.correction }}_{{ '%.2f'|format(row.log_p_thresh) }}"
                data-regions='{{ row.regions | tojson | safe }}'
                data-gallery='{{ row.cluster_gallery | tojson | safe }}'
                onclick="selectRow(this)">
                <td>{{ row.con_num }}</td>
                <td>{{ row.con_name }}</td>
                <td>
                    <span class="badge badge-{{ row.correction.lower()[:3] }}">{{ row.correction }}</span>
                    {% if row.cluster_size %}<br><small>k > {{ row.cluster_size }}</small>{% endif %}
                    {% if row.forming_threshold %}<br><small>Forming: {{ row.forming_threshold }}</small>{% endif %}
                </td>
                <td>{{ row.p_label }}</td>
                <td class="dir-{{ row.direction.lower()[:3] }}">{{ row.direction }}</td>
                <td>{{ row.sig_voxels }}</td>
                <td>{{ "%.2f"|format(row.peak_stat) }}</td>
                <td>{{ "%.2f"|format(row.max_logp) }}</td>
                <td class="coords">{{ row.peak_mni }}</td>
                <td class="region-cell region">Loading...</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    
    <script>
        const plots = {{ plots_json | safe }};
        
        function filterTable() {
            const pVal = document.getElementById('filter-p').value;
            const corrVal = document.getElementById('filter-corr').value;
            const conVal = document.getElementById('filter-con').value;
            
            const rows = document.querySelectorAll('.result-row');
            rows.forEach(row => {
                const pMatch = pVal === 'all' || row.getAttribute('data-p') === pVal;
                const corrMatch = corrVal === 'all' || row.getAttribute('data-corr') === corrVal;
                const conMatch = conVal === 'all' || row.getAttribute('data-con') === conVal;
                
                if (pMatch && corrMatch && conMatch) {
                    row.classList.remove('hidden');
                } else {
                    row.classList.add('hidden');
                }
            });
        }
        
        function updateAtlas() {
            const atlasName = document.getElementById('select-atlas').value;
            const rows = document.querySelectorAll('.result-row');
            rows.forEach(row => {
                const regions = JSON.parse(row.getAttribute('data-regions'));
                const cell = row.querySelector('.region-cell');
                cell.innerText = regions[atlasName] || 'N/A';
            });
            
            const selectedRow = document.querySelector('.result-row.selected');
            if (selectedRow) selectRow(selectedRow);
        }
        
        function selectRow(row) {
            document.querySelectorAll('.result-row').forEach(r => r.classList.remove('selected'));
            row.classList.add('selected');
            
            const imgId = row.getAttribute('data-img-id');
            const plotImg = document.getElementById('main-plot');
            const noPlot = document.getElementById('no-plot');
            const plotTitle = document.getElementById('plot-title');
            
            if (plots[imgId]) {
                plotImg.src = 'data:image/png;base64,' + plots[imgId];
                plotImg.classList.remove('hidden');
                noPlot.classList.add('hidden');
                plotTitle.innerText = row.cells[1].innerText + ' (' + row.cells[2].innerText + ' @ ' + row.cells[3].innerText + ')';
            } else {
                plotImg.classList.add('hidden');
                noPlot.classList.remove('hidden');
                plotTitle.innerText = 'No visualization available';
            }
            
            const galleryData = JSON.parse(row.getAttribute('data-gallery'));
            const gallerySection = document.getElementById('gallery-section');
            const galleryContent = document.getElementById('gallery-content');
            
            if (galleryData && galleryData.length > 0) {
                const activeAtlas = document.getElementById('select-atlas').value;
                gallerySection.classList.remove('hidden');
                galleryContent.innerHTML = galleryData.map(c => `
                    <div style="border-bottom: 1px solid #eee; padding-bottom: 10px;">
                        <div style="font-weight: bold; margin-bottom: 5px;">
                            Cluster ${c.id} - ${c.regions[activeAtlas] || 'N/A'} 
                            <span style="font-weight: normal; color: #666; font-size: 0.9em; margin-left: 10px;">
                                MNI: [${c.peak_mni.join(', ')}] | Size: ${c.size} voxels
                            </span>
                        </div>
                        <img src="data:image/png;base64,${c.plot}" style="max-width: 100%; height: auto; border-radius: 4px;">
                    </div>
                `).join('');
            } else {
                gallerySection.classList.add('hidden');
                galleryContent.innerHTML = '';
            }
        }
        
        window.onload = () => {
            updateAtlas();
            const firstRow = document.querySelector('.result-row:not(.hidden)');
            if (firstRow) selectRow(firstRow);
        };
    </script>
</body>
</html>"""
    
    template = Template(html_template)
    html_content = template.render(
        results_dir=results_dir,
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        mode='Surface' if is_surface else 'Volume',
        is_surface=is_surface,
        has_tfce=has_tfce,
        contrast_names=contrast_names,
        report_data=report_data,
        plots_json=json.dumps(plots),
        atlases=[{'name': a['name']} for a in atlases]
    )
    
    with open(output_html, 'w') as f:
        f.write(html_content)
    
    print(f"Report saved to: {output_html}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML report for CAT12 statistical results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python post_stats_report.py ./stats_results ./report.html
  python post_stats_report.py ./stats_results ./report.html --filter tfce
  python post_stats_report.py ./stats_results ./report.html --spm-path /path/to/spm12

Filter Modes:
  all              - Include all available results (TFCE, SPM{T}, etc.)
  tfce             - Only include TFCE results
  spmt             - Only include standard SPM T-maps
  double_threshold - Only include double-threshold results
        """
    )
    
    parser.add_argument("results_dir", help="Directory containing CAT12/SPM statistical results")
    parser.add_argument("output_html", help="Path where the HTML report will be saved")
    parser.add_argument("--filter", "-f", choices=["all", "tfce", "spmt", "double_threshold"], default="all",
                        help="Filter the types of results included (default: all)")
    parser.add_argument("--spm-path", help="Path to SPM installation (for loading atlases)")
    
    args = parser.parse_args()
    
    generate_report(args.results_dir, args.output_html, args.filter, spm_path=args.spm_path)
