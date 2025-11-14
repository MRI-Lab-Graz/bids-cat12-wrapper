#!/usr/bin/env python3
"""
Configuration file parser for CAT12 pipeline.
Reads config.ini and provides easy access to settings.
"""

import configparser
from pathlib import Path


def load_config():
    """Load config.ini from script directory"""
    config = configparser.ConfigParser()
    
    # Try to find config.ini in common locations
    script_dir = Path(__file__).parent.parent  # utils/../
    config_file = script_dir / 'config.ini'
    
    if config_file.exists():
        config.read(config_file)
    
    return config


def get_matlab_exe(config=None):
    """Get MATLAB executable path"""
    if config is None:
        config = load_config()
    
    if config.has_option('MATLAB', 'exe'):
        exe = config.get('MATLAB', 'exe').strip()
        if exe and exe != '':
            return exe
    
    # Auto-detect on macOS
    import subprocess
    try:
        result = subprocess.run(
            ["find", "/Applications", "-name", "MATLAB_R*.app", "-maxdepth", "1"],
            capture_output=True, text=True
        )
        matches = sorted(result.stdout.strip().split('\n'), reverse=True)
        if matches and matches[0]:
            return f"{matches[0]}/bin/matlab"
    except Exception:
        pass
    
    # Default fallback
    return "matlab"


def get_python_exe(config=None):
    """Get Python 3 executable"""
    if config is None:
        config = load_config()
    
    if config.has_option('PYTHON', 'exe'):
        exe = config.get('PYTHON', 'exe').strip()
        if exe and exe != '':
            return exe
    
    return "python3"


def get_spm_path(config=None):
    """Get SPM installation path"""
    if config is None:
        config = load_config()
    
    if config.has_option('SPM', 'path'):
        path = config.get('SPM', 'path').strip()
        if path and path != '':
            return path
    
    return None


def get_parallel_jobs(config=None):
    """Get number of parallel jobs for TFCE"""
    if config is None:
        config = load_config()
    
    if config.has_option('PERFORMANCE', 'parallel_jobs'):
        try:
            return int(config.get('PERFORMANCE', 'parallel_jobs'))
        except (ValueError, TypeError):
            pass
    
    return 4


if __name__ == '__main__':
    # Test config loading
    cfg = load_config()
    print("MATLAB:", get_matlab_exe(cfg))
    print("Python:", get_python_exe(cfg))
    print("SPM:", get_spm_path(cfg))
    print("Parallel jobs:", get_parallel_jobs(cfg))
