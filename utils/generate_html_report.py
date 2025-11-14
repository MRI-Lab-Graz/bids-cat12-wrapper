#!/usr/bin/env python3
"""
Generate fMRIPrep-style HTML report for CAT12 longitudinal analysis
"""

import json
import os
from datetime import datetime

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CAT12 Longitudinal Analysis Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}
        
        .header .subtitle {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        .summary {{
            background: white;
            border-radius: 8px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .summary h2 {{
            color: #667eea;
            margin-bottom: 1rem;
            border-bottom: 2px solid #667eea;
            padding-bottom: 0.5rem;
        }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}
        
        .info-item {{
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 6px;
            border-left: 4px solid #667eea;
        }}
        
        .info-label {{
            font-weight: bold;
            color: #666;
            font-size: 0.9rem;
            margin-bottom: 0.3rem;
        }}
        
        .info-value {{
            color: #333;
            font-size: 1.1rem;
        }}
        
        .section {{
            background: white;
            border-radius: 8px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .section h3 {{
            color: #764ba2;
            margin-bottom: 1rem;
            font-size: 1.5rem;
        }}
        
        .design-matrix {{
            width: 100%;
            max-width: 600px;
            margin: 1rem 0;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        th {{
            background: #667eea;
            color: white;
            font-weight: 600;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        .status {{
            display: inline-block;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        
        .status.complete {{
            background: #d4edda;
            color: #155724;
        }}
        
        .status.running {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .status.pending {{
            background: #e7e7e7;
            color: #666;
        }}
        
        .step-list {{
            list-style: none;
            counter-reset: step-counter;
        }}
        
        .step-list li {{
            counter-increment: step-counter;
            margin-bottom: 1.5rem;
            padding-left: 3rem;
            position: relative;
        }}
        
        .step-list li:before {{
            content: counter(step-counter);
            position: absolute;
            left: 0;
            top: 0;
            width: 2rem;
            height: 2rem;
            background: #667eea;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }}
        
        .footer {{
            text-align: center;
            padding: 2rem;
            color: #666;
            font-size: 0.9rem;
        }}
        
        code {{
            background: #f4f4f4;
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 0.9rem;
        }}
        
        .code-block {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 1rem;
            border-radius: 6px;
            overflow-x: auto;
            margin: 1rem 0;
        }}
        
        .warning {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>CAT12 Longitudinal Analysis</h1>
        <div class="subtitle">Statistical Parametric Mapping Analysis Report</div>
    </div>
    
    <div class="container">
        <!-- Summary -->
        <div class="summary">
            <h2>Analysis Summary</h2>
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Analysis Name</div>
                    <div class="info-value">{analysis_name}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Modality</div>
                    <div class="info-value">{modality}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Smoothing</div>
                    <div class="info-value">{smoothing} mm</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Total Scans</div>
                    <div class="info-value">{total_scans}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Generated</div>
                    <div class="info-value">{timestamp}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Output Directory</div>
                    <div class="info-value" style="font-size: 0.85rem; word-break: break-all;">{output_dir}</div>
                </div>
            </div>
        </div>
        
        <!-- Design Matrix -->
        <div class="section">
            <h3>Experimental Design</h3>
            <p><strong>Design Type:</strong> Flexible Factorial ({n_groups} groups x {n_sessions} timepoints)</p>
            
            <h4 style="margin-top: 1.5rem; color: #333;">Factors:</h4>
            <ul style="margin-left: 2rem; margin-top: 0.5rem;">
                <li><strong>Group</strong> (between-subject, independent): {groups_list}</li>
                <li><strong>Time</strong> (within-subject, repeated measures): {sessions_list}</li>
            </ul>
            
            {covariates_section}
            
            {design_matrix_image}

            {missing_section}
            
            <h4 style="margin-top: 1.5rem; color: #333;">Sample Distribution:</h4>
            <table>
                <thead>
                    <tr>
                        <th>Group</th>
                        {session_headers}
                        <th>Total</th>
                    </tr>
                </thead>
                <tbody>
                    {sample_distribution_rows}
                </tbody>
            </table>
        </div>
        
        <!-- Processing Steps -->
        <div class="section">
            <h3>Processing Pipeline</h3>
            <ol class="step-list">
                <li>
                    <strong>Participant Data Parsing</strong> <span class="status complete">✓ Complete</span>
                    <div style="margin-top: 0.5rem; color: #666;">
                        Matched {total_scans} scans across {n_subjects} subjects from participants.tsv
                    </div>
                </li>
                <li>
                    <strong>SPM Factorial Design Specification</strong> <span class="status complete">✓ Complete</span>
                    <div style="margin-top: 0.5rem; color: #666;">
                        Generated SPM batch with well-conditioned design matrix (no subject factors)
                    </div>
                </li>
                <li>
                    <strong>GLM Estimation</strong> <span class="status complete">✓ Complete</span>
                    <div style="margin-top: 0.5rem; color: #666;">
                        Estimated general linear model with classical ReML
                    </div>
                </li>
                <li>
                    <strong>Contrast Specification</strong> <span class="status complete">✓ Complete</span>
                    <div style="margin-top: 0.5rem; color: #666;">
                        Added {n_contrasts} contrasts for longitudinal comparisons
                    </div>
                </li>
                {screening_step}
                {tfce_step}
            </ol>
        </div>
        
        <!-- TFCE Results -->
        {tfce_results_section}
        
        <!-- File Locations -->
        <div class="section">
            <h3>Output Files</h3>
            <table>
                <thead>
                    <tr>
                        <th>File</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><code>SPM.mat</code></td>
                        <td>SPM design and estimation structure</td>
                    </tr>
                    <tr>
                        <td><code>spmT_*.nii</code></td>
                        <td>T-statistic maps for each contrast</td>
                    </tr>
                    <tr>
                        <td><code>spmF_*.nii</code></td>
                        <td>F-statistic maps for factorial effects</td>
                    </tr>
                    <tr>
                        <td><code>beta_*.nii</code></td>
                        <td>Parameter estimate images</td>
                    </tr>
                    <tr>
                        <td><code>ResMS.nii</code></td>
                        <td>Residual mean squares</td>
                    </tr>
                    <tr>
                        <td><code>mask.nii</code></td>
                        <td>Analysis mask</td>
                    </tr>
                    {tfce_files}
                </tbody>
            </table>
        </div>
        
        <!-- Reproducibility -->
        <div class="section">
            <h3>Reproducibility Information</h3>
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">SPM Version</div>
                    <div class="info-value">SPM25</div>
                </div>
                <div class="info-item">
                    <div class="info-label">CAT12 Version</div>
                    <div class="info-value">CAT12.9</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Pipeline Script</div>
                    <div class="info-value">cat12_longitudinal_analysis.sh</div>
                </div>
            </div>
            
            <h4 style="margin-top: 1.5rem; color: #333;">Command Line:</h4>
            <div class="code-block">
                <code>{command_line}</code>
            </div>
            {tfce_thumbs_section}
            <h4 style="margin-top: 1.5rem; color: #333;">Full Pipeline Log:</h4>
            <div class="code-block" style="max-height: 500px; overflow-y: auto;">
                <code>{pipeline_log}</code>
            </div>
        </div>
    </div>
    
    <div class="footer">
        Generated by CAT12 Longitudinal Analysis Pipeline<br>
        {timestamp}
    </div>
</body>
</html>
"""


def generate_report(design_json_path, output_html_path, **kwargs):
    """Generate HTML report from design JSON and analysis parameters"""
    
    # Load design structure
    with open(design_json_path) as f:
        design = json.load(f)
    
    # Check for design matrix image
    output_dir = kwargs.get('output_dir', '')
    design_matrix_img = None
    if output_dir:
        img_path = f"{output_dir}/design_matrix.png"
        if os.path.exists(img_path):
            design_matrix_img = "design_matrix.png"  # Relative path for HTML
    
    # Load TFCE summary if available
    tfce_summary = None
    if output_dir:
        tfce_summary_path = os.path.join(output_dir, 'tfce_summary.json')
        if os.path.exists(tfce_summary_path):
            with open(tfce_summary_path) as f:
                tfce_summary = json.load(f)
    
    # Extract parameters
    modality = design.get('modality', 'vbm')
    smoothing = design.get('smoothing', 'auto')
    groups = design.get('groups', {})
    
    # Count samples
    total_scans = 0
    n_subjects_per_group = {}
    sample_dist = {}
    
    for group, data in groups.items():
        sessions = data.get('sessions', {})
        group_total = sum(len(files) for files in sessions.values())
        total_scans += group_total
        n_subjects_per_group[group] = group_total // len(sessions) if sessions else 0
        sample_dist[group] = {sess: len(files) for sess, files in sessions.items()}
    
    n_groups = len(groups)
    n_sessions = len(list(groups.values())[0].get('sessions', {})) if groups else 0
    n_subjects = sum(n_subjects_per_group.values())
    
    # Build sample distribution table
    all_sessions = sorted(list(groups.values())[0].get('sessions', {}).keys()) if groups else []
    session_headers = ''.join(f'<th>Session {s}</th>' for s in all_sessions)
    
    rows = []
    for group in sorted(groups.keys()):
        row_cells = [f'<td><strong>{group}</strong></td>']
        group_total = 0
        for sess in all_sessions:
            count = sample_dist.get(group, {}).get(sess, 0)
            row_cells.append(f'<td>{count}</td>')
            group_total += count
        row_cells.append(f'<td><strong>{group_total}</strong></td>')
        rows.append('<tr>' + ''.join(row_cells) + '</tr>')
    
    # Add total row
    total_row = ['<tr style="background: #f0f0f0; font-weight: bold;"><td>Total</td>']
    for sess in all_sessions:
        sess_total = sum(sample_dist.get(g, {}).get(sess, 0) for g in groups.keys())
        total_row.append(f'<td>{sess_total}</td>')
    total_row.append(f'<td>{total_scans}</td></tr>')
    rows.append(''.join(total_row))
    
    sample_distribution_rows = '\n'.join(rows)
    
    # Covariates section
    covariates = design.get('covariates', {})
    if covariates:
        cov_list = ', '.join(f'<code>{cov}</code>' for cov in covariates.keys())
        covariates_section = f"""
            <h4 style="margin-top: 1.5rem; color: #333;">Covariates:</h4>
            <p>{cov_list}</p>
        """
    else:
        covariates_section = ""
    
    # Design matrix image section
    if design_matrix_img:
        design_matrix_image = f"""
            <h4 style="margin-top: 1.5rem; color: #333;">Design Matrix Visualization:</h4>
            <img src="{design_matrix_img}" alt="Design Matrix" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; margin-top: 1rem;">
            <p style="margin-top: 0.5rem; color: #666; font-size: 0.9rem;">
                Visual representation of the SPM design matrix showing parameter estimates across all scans.
            </p>
        """
    else:
        design_matrix_image = ""

    # Collect pipeline log (full terminal output) if present
    pipeline_log = None
    if output_dir:
        cand_log = os.path.join(output_dir, 'logs', 'pipeline.log')
        if os.path.exists(cand_log):
            try:
                with open(cand_log, 'r') as lf:
                    pipeline_log = lf.read()
            except Exception:
                pipeline_log = None

    # Generate thumbnails for TFCE result NIfTIs if possible
    tfce_thumbnails = []
    if output_dir:
        # search for TFCE_log_pFWE_*.nii in output_dir and subdirs
        nifti_paths = []
        for root, dirs, files in os.walk(output_dir):
            for fn in files:
                if fn.startswith('TFCE_log_pFWE') and (fn.endswith('.nii') or fn.endswith('.nii.gz')):
                    nifti_paths.append(os.path.join(root, fn))

        # Try to create thumbnails using nibabel + matplotlib if available
        if nifti_paths:
            try:
                import nibabel as nb
                import numpy as np
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                for npth in nifti_paths:
                    try:
                        img = nb.load(npth)
                        data = img.get_fdata()
                        # choose middle axial slice
                        if data.ndim == 3:
                            z = data.shape[2] // 2
                            slice_img = np.rot90(data[:, :, z])
                        else:
                            # fallback to flatten
                            slice_img = np.rot90(np.squeeze(data))

                        thumb_name = os.path.join(output_dir, os.path.basename(npth).replace('.nii.gz', '').replace('.nii', '') + '_thumb.png')
                        plt.figure(figsize=(6, 4))
                        plt.imshow(slice_img, cmap='hot', interpolation='nearest')
                        plt.axis('off')
                        plt.tight_layout()
                        plt.savefig(thumb_name, dpi=150, bbox_inches='tight', pad_inches=0)
                        plt.close()
                        tfce_thumbnails.append({'nifti': npth, 'thumb': os.path.relpath(thumb_name, output_dir)})
                    except Exception:
                        continue
            except Exception:
                # matplotlib/nibabel not available - skip thumbnails
                tfce_thumbnails = []

    
    # Processing steps
    screening_step = kwargs.get('screening_step', '')
    tfce_step = kwargs.get('tfce_step', '')
    tfce_files = kwargs.get('tfce_files', '')
    
    # Build TFCE results section if summary available
    tfce_results_section = ""
    if tfce_summary and tfce_summary.get('contrasts'):
        tfce_contrasts = tfce_summary['contrasts']
        n_tfce = len([c for c in tfce_contrasts if c.get('has_results')])
        fwe_thresh = tfce_summary.get('fwe_threshold', 0.05)
        
        tfce_table_rows = []
        for c in tfce_contrasts:
            if c.get('has_results'):
                status_icon = '✓'
                status_class = 'complete'
            else:
                status_icon = '○'
                status_class = 'pending'
            
            tfce_table_rows.append(f"""
                <tr>
                    <td>{c['index']}</td>
                    <td>{c['name']}</td>
                    <td><span class="status {status_class}">{status_icon}</span></td>
                </tr>
            """)
        
        tfce_results_section = f"""
        <div class="section">
            <h3>TFCE Multiple Comparison Correction Results</h3>
            <p>
                <strong>FWE-corrected threshold:</strong> p &lt; {fwe_thresh}<br>
                <strong>Contrasts with TFCE results:</strong> {n_tfce} / {len(tfce_contrasts)}
            </p>
            
            <table>
                <thead>
                    <tr>
                        <th>Contrast</th>
                        <th>Name</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(tfce_table_rows)}
                </tbody>
            </table>
            
            <p style="margin-top: 1.5rem; color: #666; font-size: 0.9rem;">
                <strong>Note:</strong> TFCE (Threshold-Free Cluster Enhancement) results provide family-wise error (FWE) corrected p-values. 
                Results files are located in the output directory as <code>TFCE_*_log_pFWE.nii</code> or in <code>TFCE_*/</code> subdirectories.
            </p>
        </div>
        """

    # Build TFCE thumbnails section if thumbnails were generated
    tfce_thumbs_section = ""
    if tfce_thumbnails:
        thumb_items = []
        for t in tfce_thumbnails:
            # t['thumb'] is relative path under output_dir
            thumb_items.append(f'<div style="display:inline-block;margin:8px;text-align:center;"><a href="{os.path.relpath(t["nifti"], output_dir)}"><img src="{t["thumb"]}" style="width:220px;height:auto;border:1px solid #ddd;border-radius:4px"></a><div style="font-size:0.85rem;margin-top:0.3rem">{os.path.basename(t["nifti"])}</div></div>')
        tfce_thumbs_section = f"""
        <div class="section">
            <h3>Result Thumbnails</h3>
            <p>Quick visual summaries of TFCE FWE-corrected maps. Click a thumbnail to download the full NIfTI.</p>
            <div style="margin-top:1rem;">{''.join(thumb_items)}</div>
        </div>
        """
    
    # Format parameters
    params = {
        'analysis_name': kwargs.get('analysis_name', 'Unnamed Analysis'),
        'modality': modality.upper(),
        'smoothing': smoothing,
        'total_scans': total_scans,
        'n_subjects': n_subjects,
        'n_groups': n_groups,
        'n_sessions': n_sessions,
        'groups_list': ', '.join(sorted(groups.keys())),
        'sessions_list': ', '.join(f'Session {s}' for s in all_sessions),
        'session_headers': session_headers,
        'sample_distribution_rows': sample_distribution_rows,
        'covariates_section': covariates_section,
        'design_matrix_image': design_matrix_image,
    'missing_section': '',
        'screening_step': screening_step,
        'tfce_step': tfce_step,
        'tfce_results_section': tfce_results_section,
        'tfce_files': tfce_files,
        'n_contrasts': kwargs.get('n_contrasts', 'N/A'),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'output_dir': kwargs.get('output_dir', 'N/A'),
        'command_line': kwargs.get('command_line', 'N/A'),
        'pipeline_log': pipeline_log or '',
        'tfce_thumbs_section': tfce_thumbs_section,
    }

    # Missing voxels diagnostics: include summary, thumb, links if present
    missing_section = ''
    if output_dir:
        msum = os.path.join(output_dir, 'missing_voxels_summary.json')
        mthumb = os.path.join(output_dir, 'missing_voxels_thumb.png')
        mmask = os.path.join(output_dir, 'missing_voxels_mask.nii')
        msub = os.path.join(output_dir, 'missing_voxels_subjects.csv')
        if os.path.exists(msum):
            try:
                with open(msum) as fh:
                    mdata = json.load(fh)
                pct = mdata.get('pct_voxels_excluded', None)
                n_exc = mdata.get('n_voxels_excluded', None)
                n_tot = mdata.get('n_voxels_total', None)
                missing_section = '<h4 style="margin-top: 1.5rem; color: #333;">Missing-voxel Diagnostics</h4>'
                missing_section += '<p style="color: #666; font-size: 0.95rem;">This analysis includes a diagnostic scan for voxels excluded across images (NaNs).</p>'
                if pct is not None:
                    missing_section += f'<p><strong>Excluded voxels:</strong> {n_exc} / {n_tot} ({pct}%)</p>'
                # include thumbnail if present
                if os.path.exists(mthumb):
                    rel_thumb = os.path.relpath(mthumb, output_dir)
                    missing_section += f'<div style="margin-top:0.75rem;"><img src="{rel_thumb}" alt="Missing voxels thumbnail" style="max-width:400px;border:1px solid #ddd;border-radius:4px"></div>'
                # links to mask and per-subject CSV
                links = []
                if os.path.exists(mmask):
                    links.append(f'<a href="{os.path.relpath(mmask, output_dir)}">missing_voxels_mask.nii</a>')
                if os.path.exists(msub):
                    links.append(f'<a href="{os.path.relpath(msub, output_dir)}">missing_voxels_subjects.csv</a>')
                if links:
                    missing_section += '<p style="margin-top:0.5rem;">Download: ' + ' | '.join(links) + '</p>'
            except Exception:
                missing_section = ''

    params['missing_section'] = missing_section
    
    # Generate HTML
    html_content = HTML_TEMPLATE.format(**params)
    
    # Write to file
    with open(output_html_path, 'w') as f:
        f.write(html_content)
    
    print(f"✓ HTML report generated: {output_html_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate HTML analysis report')
    parser.add_argument('--design-json', required=True, help='Path to design.json')
    parser.add_argument('--output', required=True, help='Output HTML file path')
    parser.add_argument('--analysis-name', default='CAT12 Analysis', help='Analysis name')
    parser.add_argument('--output-dir', help='Results output directory')
    parser.add_argument('--command-line', help='Command line used')
    parser.add_argument('--n-contrasts', type=int, help='Number of contrasts')
    
    args = parser.parse_args()
    
    generate_report(
        args.design_json,
        args.output,
        analysis_name=args.analysis_name,
        output_dir=args.output_dir,
        command_line=args.command_line,
        n_contrasts=args.n_contrasts
    )
