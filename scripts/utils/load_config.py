#!/usr/bin/env python3
"""
Configuration file parser for CAT12 pipeline.
Reads config.json and provides easy access to settings.
"""

import json
from pathlib import Path


def load_config(config_file=None):
    """Load config.json from config directory"""
    if config_file is None:
        workspace_root = Path(__file__).resolve().parents[2]
        config_file = workspace_root / "config" / "config.json"
    else:
        config_file = Path(config_file)

    if config_file.exists():
        try:
            with open(config_file) as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}

    return {}


def get_matlab_exe(config=None):
    """Get MATLAB executable path"""
    if config is None:
        config = load_config()

    if "matlab" in config and config["matlab"].get("executable"):
        exe = config["matlab"]["executable"].strip()
        if exe:
            return exe

    # Auto-detect on macOS
    import subprocess

    try:
        result = subprocess.run(
            ["find", "/Applications", "-name", "MATLAB_R*.app", "-maxdepth", "1"],
            capture_output=True,
            text=True,
        )
        matches = sorted(result.stdout.strip().split("\n"), reverse=True)
        if matches and matches[0]:
            return f"{matches[0]}/bin/matlab"
    except Exception:
        pass

    return "matlab"


def get_python_exe(config=None):
    """Get Python 3 executable"""
    if config is None:
        config = load_config()

    if "python" in config and config["python"].get("executable"):
        exe = config["python"]["executable"].strip()
        if exe:
            return exe

    return "python3"


def get_spm_path(config=None):
    """Get SPM installation path"""
    if config is None:
        config = load_config()

    if "spm" in config and config["spm"].get("path"):
        path = config["spm"]["path"].strip()
        if path:
            return path

    return None


def get_parallel_jobs(config=None):
    """Get number of parallel jobs for TFCE"""
    if config is None:
        config = load_config()

    if "performance" in config:
        try:
            return int(config["performance"].get("parallel_jobs", 4))
        except (ValueError, TypeError):
            pass

    return 4


if __name__ == "__main__":
    # Test config loading
    cfg = load_config()
    print("MATLAB:", get_matlab_exe(cfg))
    print("Python:", get_python_exe(cfg))
    print("SPM:", get_spm_path(cfg))
    print("Parallel jobs:", get_parallel_jobs(cfg))
