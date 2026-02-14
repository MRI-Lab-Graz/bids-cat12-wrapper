#!/usr/bin/env python3
"""
Auto-generate post-stats reports based on config file.

Reads reporting settings from config.json and automatically generates
HTML reports for all configured modalities.

Usage:
    python scripts/reporting/auto_report.py [config_path]

Examples:
    python scripts/reporting/auto_report.py config/config_14_2_26.json
    python scripts/reporting/auto_report.py  # Uses config/config.json by default
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path


def load_config(config_path=None):
    """Load configuration from JSON file."""
    candidates = [
        config_path,
        "config/config_14_2_26.json",
        "config/config.json",
        os.path.join(os.path.dirname(__file__), "../../config/config.json"),
    ]
    
    for cfg in candidates:
        if cfg and os.path.exists(cfg):
            with open(cfg) as f:
                return json.load(f), cfg
    
    raise FileNotFoundError("Could not find config file")


def generate_reports(config_path=None):
    """Generate reports for all configured modalities."""
    config, cfg_used = load_config(config_path)
    
    print(f"📋 Loaded config from: {cfg_used}\n")
    
    # Check if reporting is enabled
    reporting_config = config.get("reporting", {})
    if not reporting_config.get("auto_generate", False):
        print("⚠️  Reporting is disabled (auto_generate=false). Enable it in config.json to auto-generate reports.")
        return
    
    analysis_config = config.get("analysis", {})
    output_config = config.get("output", {})
    modalities = analysis_config.get("modalities", [])
    analysis_name = output_config.get("analysis_name", "analysis")
    
    if not modalities:
        print("❌ No modalities configured in analysis.modalities")
        return
    
    quality = reporting_config.get("quality", "low")
    filter_mode = reporting_config.get("filter", "no_tfce")
    output_template = reporting_config.get("output_filename", "report_{date}.html")
    
    # Get base results directory
    base_results_dir = os.path.join(os.getcwd(), "results")
    if not os.path.exists(base_results_dir):
        print(f"❌ Results directory not found: {base_results_dir}")
        return
    
    # Process each modality
    for modality in modalities:
        modality_name = modality.get("name", "unknown")
        folder_name = modality.get("folder_name", modality_name)
        
        results_dir = os.path.join(base_results_dir, modality_name, folder_name)
        
        if not os.path.exists(results_dir):
            print(f"⚠️  Skipping {modality_name}: directory not found at {results_dir}")
            continue
        
        # Generate output filename with substitutions
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_filename = output_template.format(
            date=date_str,
            modality=modality_name,
            analysis=analysis_name
        )
        output_path = os.path.join(results_dir, output_filename)
        
        print(f"🔄 Generating report for: {modality_name}/{folder_name}")
        print(f"   Quality: {quality}")
        print(f"   Filter: {filter_mode}")
        print(f"   Output: {output_path}\n")
        
        # Build command
        cmd = [
            "python",
            "scripts/reporting/post_stats_report.py",
            results_dir,
            output_path,
            "--quality", quality,
            "--filter", filter_mode,
            "--config", cfg_used,
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=False, text=True)
            if result.returncode == 0:
                print(f"✅ Report saved: {output_path}\n")
            else:
                print(f"❌ Failed to generate report for {modality_name}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_reports(config_path)
