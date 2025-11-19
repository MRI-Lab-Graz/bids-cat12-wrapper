
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CAT12 Analysis Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{
            background-color: #f8f9fa;
            padding-top: 20px;
            padding-bottom: 40px;
        }}
        .container {{
            max-width: 1200px;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        .header {{
            border-bottom: 2px solid #e9ecef;
            margin-bottom: 30px;
            padding-bottom: 20px;
        }}
        .header h1 {{
            color: #2c3e50;
            font-weight: 700;
        }}
        .meta-info {{
            color: #6c757d;
            font-size: 0.9rem;
        }}
        .section-title {{
            color: #2c3e50;
            border-left: 4px solid #0d6efd;
            padding-left: 15px;
            margin-top: 40px;
            margin-bottom: 20px;
        }}
        .boilerplate {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            padding: 20px;
            border-radius: 4px;
            font-family: 'Times New Roman', serif;
            font-size: 1.1rem;
            line-height: 1.6;
            color: #333;
        }}
        .boilerplate h5 {{
            font-family: sans-serif;
            font-size: 0.9rem;
            text-transform: uppercase;
            color: #6c757d;
            margin-bottom: 10px;
            letter-spacing: 1px;
        }}
        .img-thumbnail {{
            max-width: 100%;
            height: auto;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 4px;
        }}
        .status-badge {{
            font-size: 0.85em;
            padding: 5px 10px;
            border-radius: 20px;
        }}
        .table-sm td, .table-sm th {{
            padding: .5rem;
        }}
        code {{
            color: #d63384;
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
        }}
        .nav-pills .nav-link.active {{
            background-color: #0d6efd;
        }}
        .contrast-card {{
            transition: transform 0.2s;
            margin-bottom: 20px;
        }}
        .contrast-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="row align-items-center">
                <div class="col-md-8">
                    <h1>CAT12 Longitudinal Analysis</h1>
                    <p class="lead text-muted">Statistical Report</p>
                </div>
                <div class="col-md-4 text-end meta-info">
                    <div><strong>Date:</strong> {timestamp}</div>
                    <div><strong>Analysis:</strong> {analysis_name}</div>
                    <div><strong>Modality:</strong> {modality}</div>
                </div>
            </div>
        </div>

        <!-- Navigation -->
        <ul class="nav nav-pills mb-4" id="pills-tab" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="pills-summary-tab" data-bs-toggle="pill" data-bs-target="#pills-summary" type="button" role="tab">Summary</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="pills-methods-tab" data-bs-toggle="pill" data-bs-target="#pills-methods" type="button" role="tab">Methods</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="pills-results-tab" data-bs-toggle="pill" data-bs-target="#pills-results" type="button" role="tab">Results</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="pills-qc-tab" data-bs-toggle="pill" data-bs-target="#pills-qc" type="button" role="tab">Quality Control</button>
            </li>
        </ul>

        <div class="tab-content" id="pills-tabContent">
            
            <!-- Summary Tab -->
            <div class="tab-pane fade show active" id="pills-summary" role="tabpanel">
                <h3 class="section-title">Analysis Overview</h3>
                <div class="row">
                    <div class="col-md-6">
                        <div class="card mb-4">
                            <div class="card-header">Parameters</div>
                            <div class="card-body">
                                <table class="table table-sm table-borderless">
                                    <tr><td><strong>Smoothing:</strong></td><td>{smoothing} mm</td></tr>
                                    <tr><td><strong>Subjects:</strong></td><td>{n_subjects}</td></tr>
                                    <tr><td><strong>Total Scans:</strong></td><td>{total_scans}</td></tr>
                                    <tr><td><strong>Groups:</strong></td><td>{groups_list}</td></tr>
                                    <tr><td><strong>Sessions:</strong></td><td>{n_sessions} ({sessions_list})</td></tr>
                                </table>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card mb-4">
                            <div class="card-header">Design Matrix</div>
                            <div class="card-body text-center">
                                {design_matrix_image}
                                <div class="text-muted small mt-2">Flexible Factorial Design</div>
                            </div>
                        </div>
                    </div>
                </div>

                <h3 class="section-title">Sample Distribution</h3>
                <div class="table-responsive">
                    <table class="table table-striped table-hover">
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
            </div>

            <!-- Methods Tab -->
            <div class="tab-pane fade" id="pills-methods" role="tabpanel">
                <h3 class="section-title">Methods Boilerplate</h3>
                <div class="boilerplate">
                    <h5>Suggested Text for Publication</h5>
                    {methods_text}
                </div>
                <div class="mt-3 text-muted small">
                    <p><strong>Note:</strong> This text is auto-generated based on the pipeline parameters. Please verify all details against your actual processing steps and add references where appropriate.</p>
                </div>
            </div>

            <!-- Results Tab -->
            <div class="tab-pane fade" id="pills-results" role="tabpanel">
                <h3 class="section-title">TFCE Results (FWE Corrected)</h3>
                {tfce_results_section}
                
                <h3 class="section-title mt-5">Contrast Maps (Uncorrected)</h3>
                <div class="row">
                    {contrast_items}
                </div>
            </div>

            <!-- QC Tab -->
            <div class="tab-pane fade" id="pills-qc" role="tabpanel">
                <h3 class="section-title">Missing Data Diagnostics</h3>
                {missing_section}
                
                <h3 class="section-title mt-5">Execution Log</h3>
                <pre class="bg-light p-3 border rounded" style="max-height: 400px; overflow-y: auto;">{pipeline_log}</pre>
            </div>
        </div>

        <footer class="mt-5 pt-4 border-top text-center text-muted">
            <p>Generated by CAT12 Longitudinal Analysis Pipeline | {timestamp}</p>
        </footer>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
