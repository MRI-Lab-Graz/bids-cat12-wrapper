#!/usr/bin/env python3
"""
Extract modality configurations from config.json for multi-modality pipeline.

Usage:
  python3 extract_modalities.py <config.json> [--modality <name>]
  
Returns JSON array of modality configs, or single modality if --modality specified.
"""

import json
import sys
import argparse
from pathlib import Path


def extract_modalities(config_path, single_modality=None):
    """
    Extract modality configurations from config.json.
    
    Parameters:
    -----------
    config_path : str
        Path to config.json
    single_modality : str, optional
        If provided, return only this modality config
        
    Returns:
    --------
    list or dict
        List of modality dicts, or single dict if single_modality specified
    """
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse config.json: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Check if new multi-modality format exists
    if "analysis" in config and "modalities" in config["analysis"]:
        modalities = config["analysis"]["modalities"]
        
        # If no modalities defined, provide defaults
        if not modalities or len(modalities) == 0:
            print("Error: 'analysis.modalities' is empty. Define at least one modality.", file=sys.stderr)
            sys.exit(1)
    else:
        # Fallback to old single-modality format
        print("Warning: Using legacy single-modality config format", file=sys.stderr)
        
        analysis = config.get("analysis", {})
        modality_name = analysis.get("modality", "vbm")
        
        modalities = [{
            "name": modality_name,
            "smoothing_kernel": analysis.get("smoothing_kernel"),
            "covariates": analysis.get("covariates", []),
            "mask": analysis.get("mask")
        }]
    
    # Filter by single modality if requested
    if single_modality:
        matching = [m for m in modalities if m.get("name") == single_modality]
        if not matching:
            print(f"Error: Modality '{single_modality}' not found in config", file=sys.stderr)
            sys.exit(1)
        return matching[0]
    
    return modalities


def main():
    parser = argparse.ArgumentParser(
        description="Extract modality configurations from config.json"
    )
    parser.add_argument("config_json", help="Path to config.json")
    parser.add_argument(
        "--modality",
        help="Extract only this modality (returns single object instead of array)",
        default=None
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON (default: pretty-printed)"
    )
    
    args = parser.parse_args()
    
    result = extract_modalities(args.config_json, args.modality)
    
    if args.json:
        # Compact JSON
        print(json.dumps(result, separators=(',', ':')))
    else:
        # Pretty-printed JSON
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
