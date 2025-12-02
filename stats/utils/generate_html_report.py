#!/usr/bin/env python3
"""
Generate fMRIPrep-style HTML report for CAT12 longitudinal analysis
"""

import json
import os
import html
import numpy as _np
import matplotlib

import matplotlib.colors
import matplotlib.cm
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as _plt  # noqa: E402
from datetime import datetime

# Optional dependency for thumbnails
try:
    import nibabel as nb
except ImportError:
    nb = None

# Import the new template
try:
    from report_template import HTML_TEMPLATE
except ImportError:
    # Fallback if not in path (e.g. running from different dir)
    import sys

    sys.path.append(os.path.dirname(__file__))
    from report_template import HTML_TEMPLATE  # noqa: E402


def generate_methods_text(
    design,
    n_subjects,
    n_groups,
    n_sessions,
    smoothing,
    modality,
    n_perm=5000,
    cluster_size=50,
    uncorrected_p=0.001,
):
    """Generate boilerplate methods text for publication."""
    groups = sorted(design.get("groups", {}).keys())
    covariates = sorted(design.get("covariates", {}).keys())

    # Determine if TFCE was used or standard cluster correction
    # For this pipeline, we assume TFCE is the primary output
    inference_text = f"""
    <p><strong>Inference</strong></p>
    <p>Statistical inference was performed using Threshold-Free Cluster Enhancement (TFCE) (Smith and Nichols, 2009)
    as implemented in the TFCE toolbox for SPM. Permutation testing ({n_perm} permutations) was used to control the
    Family-Wise Error (FWE) rate at p < 0.05. This approach provides increased sensitivity for spatially extended
    signals without defining an arbitrary cluster-forming threshold.</p>
    <p>For initial screening of contrasts, an uncorrected threshold of p < {uncorrected_p} with a minimum cluster size
    of {cluster_size} voxels was used.</p>
    """

    text = f"""
    <h3 class="mt-4">Methods</h3>
    <div class="card card-body bg-light">
        <p><strong>Preprocessing and Statistical Analysis</strong></p>
        <p>Longitudinal data processing was performed using the CAT12 toolbox (Gaser et al., 2022) within SPM12.
        Structural images were processed using the longitudinal pipeline to ensure consistent intra-subject registration.
        The analysis focused on {modality} measures smoothed with a {smoothing}mm FWHM Gaussian kernel.</p>

        <p><strong>Statistical Modeling</strong></p>
        <p>A flexible factorial design was specified in SPM12 to model longitudinal changes.
        The design included {n_subjects} subjects across {n_groups} groups ({', '.join(groups)}) and {n_sessions} time points.
        Subject-specific intercepts were modeled to account for repeated measures (within-subject factor: Time; between-subject factor: Group).
        {'Covariates of no interest included: ' + ', '.join(covariates) + '.' if covariates else 'No additional covariates were included.'}</p>

        {inference_text}

        <p><strong>References</strong></p>
        <ul>
            <li>Gaser C, Dahnke R, Thompson PM, Kurth F, Luders E, Alzheimer's Disease Neuroimaging Initiative. CAT – A Computational Anatomy Toolbox for the Analysis of Structural MRI Data. NeuroImage. 2022.</li>
            <li>Smith SM, Nichols TE. Threshold-free cluster enhancement: addressing problems of smoothing, threshold dependence and localisation in cluster inference. NeuroImage. 2009;44(1):83-98.</li>
        </ul>
    </div>
    """
    return text


def generate_report(design_json_path, output_html_path, **kwargs):
    """Generate HTML report from design JSON and analysis parameters"""

    # Load design structure
    with open(design_json_path) as f:
        design = json.load(f)

    # Check for design matrix image (generate a simple visualization if missing)
    output_dir = kwargs.get("output_dir", "")
    design_matrix_img = None
    # create a dedicated report directory inside the analysis output
    report_dir = None
    if output_dir:
        report_dir = os.path.join(output_dir, "report")
        os.makedirs(report_dir, exist_ok=True)
        
        # Prefer the MATLAB-generated design matrix image if available
        matlab_img_path = os.path.join(output_dir, "design_matrix.png")
        report_img_path = os.path.join(report_dir, "design_matrix.png")
        
        if os.path.exists(matlab_img_path):
            # Copy it to the report directory
            import shutil
            shutil.copy2(matlab_img_path, report_img_path)
            design_matrix_img = "design_matrix.png"
        elif os.path.exists(report_img_path):
            design_matrix_img = "design_matrix.png"
        else:
            # Try to construct a compact design matrix visualization from the design JSON
            try:
                # Build rows in consistent order: groups -> sessions -> file order
                groups = design.get("groups", {})
                rows = []
                row_meta = []
                for gname in sorted(groups.keys()):
                    sessions = groups[gname].get("sessions", {})
                    for sname in sorted(sessions.keys()):
                        files = sessions[sname]
                        for f in files:
                            rows.append((gname, sname))
                            row_meta.append(f)

                if rows:
                    n_rows = len(rows)
                    gnames = sorted(groups.keys())
                    snames = (
                        sorted(list(groups.values())[0].get("sessions", {}).keys())
                        if groups
                        else []
                    )
                    covs = design.get("covariates", {})

                    # columns: one-hot groups, one-hot sessions, then covariates (if numeric)
                    cols = []
                    for g in gnames:
                        cols.append(("group", g))
                    for s in snames:
                        cols.append(("session", s))
                    cov_names = [c for c in covs.keys()]
                    for c in cov_names:
                        cols.append(("cov", c))

                    mat = _np.zeros((n_rows, len(cols)), dtype=float)
                    for i, (g, s) in enumerate(rows):
                        # group columns
                        if g in gnames:
                            mat[i, gnames.index(g)] = 1.0
                        # session columns
                        if s in snames:
                            mat[i, len(gnames) + snames.index(s)] = 1.0
                    # covariate columns
                    for j, cname in enumerate(cov_names):
                        vals = covs.get(cname, [])
                        if len(vals) == n_rows:
                            try:
                                arr = _np.array(vals, dtype=float)
                                # zscore for display
                                arr = (arr - _np.nanmean(arr)) / (
                                    _np.nanstd(arr) + 1e-9
                                )
                                mat[:, len(gnames) + len(snames) + j] = arr
                            except Exception:
                                pass

                    # create a figure
                    fig, ax = _plt.subplots(figsize=(8, max(2, n_rows * 0.02)))
                    # Use nearest interpolation to avoid "convolved" look
                    # Use binary colormap (Greys) or similar to show presence/absence clearly
                    ax.imshow(mat, aspect="auto", cmap="Greys", interpolation="nearest")
                    ax.set_xlabel("Design columns")
                    ax.set_ylabel("Scans")
                    ax.set_title("Design matrix (groups | sessions | covariates)")
                    # _plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02) # Colorbar not needed for binary
                    _plt.tight_layout()
                    _plt.savefig(report_img_path, dpi=150)
                    _plt.close(fig)
                    design_matrix_img = os.path.basename(report_img_path)
            except Exception:
                design_matrix_img = None

    # Load contrast names from contrasts.json if available
    contrast_names = {}
    if output_dir:
        cjson = os.path.join(output_dir, "contrasts.json")
        if os.path.exists(cjson):
            try:
                with open(cjson) as f:
                    cdata = json.load(f)
                    # Handle both list of dicts and dict
                    if isinstance(cdata, list):
                        for item in cdata:
                            if "index" in item and "name" in item:
                                contrast_names[int(item["index"])] = item["name"]
            except Exception:
                pass

    # Load TFCE summary if available
    tfce_summary = None
    if output_dir:
        tfce_summary_path = os.path.join(output_dir, "tfce_summary.json")
        if os.path.exists(tfce_summary_path):
            with open(tfce_summary_path) as f:
                tfce_summary = json.load(f)
                # Update names from contrasts.json if available
                if tfce_summary and "contrasts" in tfce_summary:
                    for c in tfce_summary["contrasts"]:
                        idx = c.get("index")
                        if idx in contrast_names:
                            c["name"] = contrast_names[idx]

    # Extract parameters
    modality = design.get("modality", "vbm")
    smoothing = design.get("smoothing", "auto")
    groups = design.get("groups", {})

    # Normalize groups structure if it's the simple format (name -> index)
    if groups and isinstance(next(iter(groups.values())), int):
        # Rebuild from 'files' list
        files_list = design.get("files", [])
        new_groups = {}
        for g_name in groups.keys():
            new_groups[g_name] = {"sessions": {}}
        
        for f in files_list:
            g = f.get("group")
            s = f.get("session")
            p = f.get("path")
            if g in new_groups:
                if s not in new_groups[g]["sessions"]:
                    new_groups[g]["sessions"][s] = []
                new_groups[g]["sessions"][s].append(p)
        groups = new_groups

    # Count samples
    total_scans = 0
    n_subjects_per_group = {}
    sample_dist = {}

    for group, data in groups.items():
        sessions = data.get("sessions", {})
        group_total = sum(len(files) for files in sessions.values())
        total_scans += group_total
        n_subjects_per_group[group] = group_total // len(sessions) if sessions else 0
        sample_dist[group] = {sess: len(files) for sess, files in sessions.items()}

    n_groups = len(groups)
    n_sessions = len(list(groups.values())[0].get("sessions", {})) if groups else 0
    n_subjects = sum(n_subjects_per_group.values())

    # Build sample distribution table
    all_sessions = (
        sorted(list(groups.values())[0].get("sessions", {}).keys()) if groups else []
    )
    session_headers = "".join(f"<th>Session {s}</th>" for s in all_sessions)

    rows = []
    for group in sorted(groups.keys()):
        row_cells = [f"<td><strong>{group}</strong></td>"]
        group_total = 0
        for sess in all_sessions:
            count = sample_dist.get(group, {}).get(sess, 0)
            row_cells.append(f"<td>{count}</td>")
            group_total += count
        row_cells.append(f"<td><strong>{group_total}</strong></td>")
        rows.append("<tr>" + "".join(row_cells) + "</tr>")

    # Add total row
    total_row = ['<tr style="background: #f0f0f0; font-weight: bold;"><td>Total</td>']
    for sess in all_sessions:
        sess_total = sum(sample_dist.get(g, {}).get(sess, 0) for g in groups.keys())
        total_row.append(f"<td>{sess_total}</td>")
    total_row.append(f"<td>{total_scans}</td></tr>")
    rows.append("".join(total_row))

    sample_distribution_rows = "\n".join(rows)

    # Covariates section
    covariates = design.get("covariates", {})
    if covariates:
        cov_list = ", ".join(f"<code>{cov}</code>" for cov in covariates.keys())
        
        # Build summary table for covariates
        cov_table_rows = []
        for cov_name, values in covariates.items():
            try:
                # Convert to numeric array, handling potential non-numeric values gracefully
                vals = _np.array(values, dtype=float)
                mean_val = _np.nanmean(vals)
                std_val = _np.nanstd(vals)
                min_val = _np.nanmin(vals)
                max_val = _np.nanmax(vals)
                
                cov_table_rows.append(f"""
                <tr>
                    <td><strong>{cov_name}</strong></td>
                    <td>{mean_val:.2f} ± {std_val:.2f}</td>
                    <td>{min_val:.2f} - {max_val:.2f}</td>
                </tr>
                """)
            except Exception:
                # Fallback for non-numeric covariates if any slip through
                cov_table_rows.append(f"""
                <tr>
                    <td><strong>{cov_name}</strong></td>
                    <td colspan="2"><em>Non-numeric or mixed data</em></td>
                </tr>
                """)

        covariates_section = f"""
            <h4 style="margin-top: 1.5rem; color: #333;">Covariates:</h4>
            <p>Included covariates: {cov_list}</p>
            <table class="table table-sm table-bordered" style="width: auto; min-width: 50%;">
                <thead class="thead-light">
                    <tr>
                        <th>Covariate</th>
                        <th>Mean ± SD</th>
                        <th>Range</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(cov_table_rows)}
                </tbody>
            </table>
        """
    else:
        covariates_section = ""

    # Design matrix image section
    if design_matrix_img:
        design_matrix_image = f"""
            <img src="{design_matrix_img}" alt="Design Matrix" class="img-fluid rounded">
        """
    else:
        design_matrix_image = (
            '<div class="text-muted">No design matrix visualization available</div>'
        )

    # Collect pipeline log (summarize head/tail) if present
    pipeline_log = None
    if output_dir:
        cand_log = os.path.join(output_dir, "logs", "pipeline.log")
        if os.path.exists(cand_log):
            try:
                with open(cand_log, "r") as lf:
                    full_log = lf.read()
                # summarize: head (first 30 lines) and tail (last 30 lines)
                lines = full_log.splitlines()
                head = lines[:30]
                tail = lines[-30:] if len(lines) > 30 else []
                n_warn = sum(
                    1 for L in lines if ("warn" in L.lower() or "warning" in L.lower())
                )
                n_err = sum(
                    1
                    for L in lines
                    if ("error" in L.lower() or "traceback" in L.lower())
                )
                summary = []
                summary.append(
                    f"Log lines: {len(lines)} | warnings: {n_warn} | errors: {n_err}"
                )
                summary.append("--- HEAD ---")
                summary.extend(head)
                if tail:
                    summary.append("--- TAIL ---")
                    summary.extend(tail)
                pipeline_log = html.escape("\n".join(summary))
            except Exception:
                pipeline_log = ""

    # Generate thumbnails / montages for TFCE result NIfTIs if possible
    tfce_thumbnails = []
    start_time = kwargs.get("start_time")
    
    if output_dir and report_dir:
        # search for TFCE_log_pFWE_*.nii in output_dir and subdirs
        nifti_paths = []
        for root, dirs, files in os.walk(output_dir):
            for fn in files:
                if fn.startswith("TFCE_log_pFWE") and (
                    fn.endswith(".nii") or fn.endswith(".nii.gz")
                ):
                    fpath = os.path.join(root, fn)
                    # Filter by start time if provided
                    if start_time is not None:
                        try:
                            mtime = os.path.getmtime(fpath)
                            if mtime < start_time:
                                continue
                        except OSError:
                            continue
                    nifti_paths.append(fpath)

        # Try to create montages using nibabel + matplotlib if available
        if nifti_paths and nb:
            try:
                # Determine threshold from summary or default to 0.05
                fwe_p_thresh = 0.05
                if tfce_summary:
                    fwe_p_thresh = tfce_summary.get("fwe_threshold", 0.05)
                
                # Convert p-value to -log10(p)
                log_threshold = -_np.log10(fwe_p_thresh)

                def make_moire_single(
                    nifti_path,
                    out_png,
                    ncols=8,
                    nrows=1,
                    dpi=100,
                    threshold=1.301,
                    data=None,
                    cmap="hot",
                    title=None,
                ):
                    if data is None:
                        img = nb.load(nifti_path)
                        data = img.get_fdata()

                    data = _np.array(data, copy=True)
                    data_masked = _np.ma.masked_less(data, threshold)
                    if _np.all(_np.isnan(data_masked)):
                        return False

                    if data.ndim < 3:
                        slice_imgs = [_np.rot90(_np.squeeze(data_masked))]
                    else:
                        zsize = data.shape[2]
                        idxs = _np.linspace(0, zsize - 1, ncols * nrows, dtype=int)
                        slice_imgs = [_np.rot90(data_masked[:, :, z]) for z in idxs]

                    fig = _plt.figure(figsize=(ncols * 1.2 + 1, nrows * 1.2), dpi=dpi)
                    gs = fig.add_gridspec(nrows, ncols + 1, width_ratios=[1] * ncols + [0.1])

                    vmax = _np.nanmax(data_masked)
                    if _np.isnan(vmax) or vmax < threshold:
                        vmax = threshold + 1

                    for i, sl in enumerate(slice_imgs):
                        if i >= ncols * nrows:
                            break
                        r, c = divmod(i, ncols)
                        ax = fig.add_subplot(gs[r, c])
                        ax.imshow(sl, cmap=cmap, interpolation="nearest", vmin=threshold, vmax=vmax)
                        ax.axis("off")

                    cax = fig.add_subplot(gs[:, -1])
                    norm = matplotlib.colors.Normalize(vmin=threshold, vmax=vmax)
                    cb = _plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
                    cb.set_label("-log10(p)")

                    if title:
                        fig.suptitle(title, fontsize=10)

                    _plt.subplots_adjust(wspace=0.01, hspace=0.01, right=0.9)
                    _plt.savefig(out_png, bbox_inches="tight", pad_inches=0.1)
                    _plt.close(fig)
                    return True

                for npth in nifti_paths:
                    try:
                        base = (
                            os.path.basename(npth)
                            .replace(".nii.gz", "")
                            .replace(".nii", "")
                        )

                        c_name = base
                        c_idx = None
                        parts = base.split('_')
                        for p in parts:
                            if p.isdigit():
                                c_idx = int(p)
                                break

                        if c_idx in contrast_names:
                            c_name = f"{c_idx}: {contrast_names[c_idx]}"

                        t_map_path = (
                            os.path.join(output_dir, f"spmT_{c_idx:04d}.nii")
                            if c_idx is not None
                            else None
                        )

                        tfce_img = nb.load(npth)
                        tfce_data = tfce_img.get_fdata()

                        # Skip if no significant voxels
                        vmax = _np.nanmax(tfce_data)
                        if _np.isnan(vmax) or vmax <= log_threshold:
                            continue

                        t_data = None
                        if t_map_path and os.path.exists(t_map_path):
                            try:
                                t_data = nb.load(t_map_path).get_fdata()
                            except Exception:
                                t_data = None

                        def _append_thumb(mask, suffix, direction_label, cmap):
                            if not _np.any(mask):
                                return
                            data_masked = _np.where(mask, tfce_data, _np.nan)
                            outpng = os.path.join(
                                report_dir,
                                f"{base}_{suffix}_fwe.png",
                            )
                            updated = make_moire_single(
                                npth,
                                outpng,
                                ncols=8,
                                nrows=1,
                                threshold=log_threshold,
                                data=data_masked,
                                cmap=cmap,
                                title=f"{c_name} ({direction_label})",
                            )
                            if updated:
                                tfce_thumbnails.append(
                                    {
                                        "nifti": os.path.relpath(npth, report_dir),
                                        "thumb": os.path.basename(outpng),
                                        "name": f"{c_name} ({direction_label})",
                                        "direction": direction_label,
                                    }
                                )

                        if t_data is not None:
                            pos_mask = (tfce_data > log_threshold) & (t_data > 0)
                            neg_mask = (tfce_data > log_threshold) & (t_data < 0)
                            _append_thumb(pos_mask, "pos", "+", "hot")
                            _append_thumb(neg_mask, "neg", "−", "winter")

                        # Fallback: if no directional thumbnails were produced, keep combined view
                        if not any(t.get("nifti") == os.path.relpath(npth, report_dir) for t in tfce_thumbnails):
                            outpng = os.path.join(report_dir, base + "_fwe_moire.png")
                            needs_update = (not os.path.exists(outpng)) or (
                                os.path.getmtime(npth) > os.path.getmtime(outpng)
                            )
                            if needs_update:
                                make_moire_single(npth, outpng, ncols=8, nrows=1, threshold=log_threshold)
                            tfce_thumbnails.append(
                                {
                                    "nifti": os.path.relpath(npth, report_dir),
                                    "thumb": os.path.basename(outpng),
                                    "name": c_name,
                                    "direction": "±",
                                }
                            )
                    except Exception as exc:
                        print(f"Warning: Failed to build TFCE thumbnail for {npth}: {exc}")
                        continue
            except Exception:
                tfce_thumbnails = []

    # Uncorrected contrast maps are no longer generated/displayed as per user request
    contrast_items = []


    # Build HTML items for contrasts (Bootstrap columns)
    contrast_items_html = ""
    if contrast_items:
        for it in contrast_items:
            contrast_items_html += f"""
            <div class="col-md-4 col-lg-3">
                <div class="card contrast-card">
                    <a href="{it['nifti']}" target="_blank">
                        <img src="{it['thumb']}" class="card-img-top" alt="{it['name']}">
                    </a>
                    <div class="card-body p-2 text-center">
                        <small class="text-muted">{it['name']}</small>
                    </div>
                </div>
            </div>
            """
    else:
        contrast_items_html = '<div class="col-12 text-muted">No contrast images found to generate previews.</div>'

    # Processing steps
    screening_step = kwargs.get("screening_step", "")
    tfce_step = kwargs.get("tfce_step", "")
    tfce_files = kwargs.get("tfce_files", "")

    # Build TFCE results section if summary available
    tfce_results_section = ""
    if tfce_summary and tfce_summary.get("contrasts"):
        tfce_contrasts = tfce_summary["contrasts"]
        n_tfce = len([c for c in tfce_contrasts if c.get("has_results")])
        fwe_thresh = tfce_summary.get("fwe_threshold", 0.05)

        tfce_table_rows = []
        for c in tfce_contrasts:
            # Only show contrasts that have significant results
            if not c.get("has_results"):
                continue
                
            status_icon = "✓"
            status_class = "bg-success text-white"

            tfce_table_rows.append(
                f"""
                <tr>
                    <td>{c['index']}</td>
                    <td>{c['name']}</td>
                    <td><span class="badge {status_class}">{status_icon}</span></td>
                </tr>
            """
            )
        
        if not tfce_table_rows:
            tfce_table_rows.append('<tr><td colspan="3" class="text-center text-muted">No significant clusters found at FWE < 0.05</td></tr>')

        tfce_results_section = f"""
        <div class="alert alert-info">
            <strong>FWE-corrected threshold:</strong> p &lt; {fwe_thresh} | 
            <strong>Contrasts with TFCE results:</strong> {n_tfce} / {len(tfce_contrasts)}
        </div>
        
        <table class="table table-hover">
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
        
        <p class="text-muted small mt-2">
            <strong>Note:</strong> TFCE (Threshold-Free Cluster Enhancement) results provide family-wise error (FWE) corrected p-values. 
            Results files are located in the output directory as <code>TFCE_*_log_pFWE.nii</code> or in <code>TFCE_*/</code> subdirectories.
        </p>
        """

    # Build TFCE thumbnails section if thumbnails were generated
    tfce_thumbs_section = ""
    if tfce_thumbnails:
        thumb_items = []
        for t in tfce_thumbnails:
            direction_badge = t.get("direction", "")
            badge_text = direction_badge if direction_badge in {"+", "-", "±"} else direction_badge

            thumb_items.append(
                f"""
            <div class="col-md-6 mb-3">
                <div class="card">
                    <a href="{t['nifti']}" target="_blank">
                        <img src="{t['thumb']}" class="card-img-top" alt="{t.get('name', 'TFCE Map')}">
                    </a>
                    <div class="card-footer text-center small text-muted">
                        {t.get("name", os.path.basename(t["nifti"]))}
                        {f"<span class='badge bg-secondary ms-2'>{badge_text}</span>" if badge_text else ''}
                    </div>
                </div>
            </div>
            """
            )
        
        if thumb_items:
            tfce_thumbs_section = f"""
            <div class="row mt-4">
                {''.join(thumb_items)}
            </div>
            """
        tfce_results_section += tfce_thumbs_section

    # Generate methods text
    methods_text = generate_methods_text(
        design,
        n_subjects,
        n_groups,
        n_sessions,
        smoothing,
        modality,
        n_perm=kwargs.get("n_perm", 5000),
        cluster_size=kwargs.get("cluster_size", 50),
        uncorrected_p=kwargs.get("uncorrected_p", 0.001),
    )

    # Format parameters
    # ensure the command_line is HTML-escaped for safe display
    cmdline = kwargs.get("command_line", "") or ""
    cmdline = html.escape(cmdline)

    params = {
        "analysis_name": kwargs.get("analysis_name", "Unnamed Analysis"),
        "modality": modality.upper(),
        "smoothing": smoothing,
        "total_scans": total_scans,
        "n_subjects": n_subjects,
        "n_groups": n_groups,
        "n_sessions": n_sessions,
        "groups_list": ", ".join(sorted(groups.keys())),
        "sessions_list": ", ".join(f"Session {s}" for s in all_sessions),
        "session_headers": session_headers,
        "sample_distribution_rows": sample_distribution_rows,
        "covariates_section": covariates_section,
        "design_matrix_image": design_matrix_image,
        "missing_section": "",
        "screening_step": screening_step,
        "tfce_step": tfce_step,
        "tfce_results_section": tfce_results_section,
        "tfce_files": tfce_files,
        "n_contrasts": kwargs.get("n_contrasts", "N/A"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": kwargs.get("output_dir", "N/A"),
        "command_line": cmdline or "N/A",
        "pipeline_log": pipeline_log or "",
        "tfce_thumbs_section": tfce_thumbs_section,
        "contrast_items": contrast_items_html,
        "methods_text": methods_text,
    }

    # Missing voxels diagnostics: include summary, thumb, links if present
    missing_section = ""
    if output_dir:
        msum = os.path.join(output_dir, "missing_voxels_summary.json")
        mthumb = os.path.join(output_dir, "missing_voxels_thumb.png")
        mmask = os.path.join(output_dir, "missing_voxels_mask.nii")
        msub = os.path.join(output_dir, "missing_voxels_subjects.csv")
        if os.path.exists(msum):
            try:
                with open(msum) as fh:
                    mdata = json.load(fh)
                pct = mdata.get("pct_voxels_excluded", None)
                n_exc = mdata.get("n_voxels_excluded", None)
                n_tot = mdata.get("n_voxels_total", None)
                missing_section = '<h4 style="margin-top: 1.5rem; color: #333;">Missing-voxel Diagnostics</h4>'
                missing_section += '<p style="color: #666; font-size: 0.95rem;">This analysis includes a diagnostic scan for voxels excluded across images (NaNs).</p>'
                if pct is not None:
                    missing_section += f"<p><strong>Excluded voxels:</strong> {n_exc} / {n_tot} ({pct}%)</p>"
                # include thumbnail if present
                if os.path.exists(mthumb):
                    rel_thumb = os.path.relpath(mthumb, output_dir)
                    missing_section += f'<div style="margin-top:0.75rem;"><img src="{rel_thumb}" alt="Missing voxels thumbnail" style="max-width:400px;border:1px solid #ddd;border-radius:4px"></div>'
                # links to mask and per-subject CSV
                links = []
                if os.path.exists(mmask):
                    links.append(
                        f'<a href="{os.path.relpath(mmask, output_dir)}">missing_voxels_mask.nii</a>'
                    )
                if os.path.exists(msub):
                    links.append(
                        f'<a href="{os.path.relpath(msub, output_dir)}">missing_voxels_subjects.csv</a>'
                    )
                if links:
                    missing_section += (
                        '<p style="margin-top:0.5rem;">Download: '
                        + " | ".join(links)
                        + "</p>"
                    )
            except Exception:
                missing_section = ""

    params["missing_section"] = missing_section

    # Generate HTML
    html_content = HTML_TEMPLATE.format(**params)

    # Write to file inside report directory so report + images are colocated
    if report_dir:
        html_out = os.path.join(
            report_dir, os.path.basename(output_html_path) or "report.html"
        )
    else:
        html_out = output_html_path
    with open(html_out, "w") as f:
        f.write(html_content)

    print(f"✓ HTML report generated: {html_out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate HTML analysis report")
    parser.add_argument("--design-json", required=True, help="Path to design.json")
    parser.add_argument("--output", required=True, help="Output HTML file path")
    parser.add_argument(
        "--analysis-name", default="CAT12 Analysis", help="Analysis name"
    )
    parser.add_argument("--output-dir", help="Results output directory")
    parser.add_argument("--command-line", help="Command line used")
    parser.add_argument("--n-contrasts", type=int, help="Number of contrasts")
    parser.add_argument(
        "--n-perm", type=int, default=5000, help="Number of permutations"
    )
    parser.add_argument(
        "--cluster-size", type=int, default=50, help="Cluster size threshold"
    )
    parser.add_argument(
        "--uncorrected-p", type=float, default=0.001, help="Uncorrected p threshold"
    )
    parser.add_argument(
        "--start-time", type=float, default=None, help="Unix timestamp of pipeline start"
    )

    args = parser.parse_args()

    generate_report(
        args.design_json,
        args.output,
        analysis_name=args.analysis_name,
        output_dir=args.output_dir,
        command_line=args.command_line,
        n_contrasts=args.n_contrasts,
        n_perm=args.n_perm,
        cluster_size=args.cluster_size,
        uncorrected_p=args.uncorrected_p,
        start_time=args.start_time,
    )
