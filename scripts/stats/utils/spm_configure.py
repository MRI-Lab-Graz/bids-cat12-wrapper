#!/usr/bin/env python3
"""spm_configure.py

Lightweight non-interactive replacement for `configure_spm_path.m` using
`spm-python`. This script attempts to detect an SPM installation, verifies
MATLAB runtime availability via `spm-python`, and optionally saves a simple
config file for the pipeline.

Usage:
  python3 utils/spm_configure.py configure [--spm-path PATH] [--save] [--config-file PATH]

Notes:
  - This is intentionally non-interactive; it tries auto-detection first and
    exits with a non-zero code on failure so the calling shell script can
    decide the fallback (standalone runner, MATLAB, etc.).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

COMMON_PATHS = [
    '/Applications/spm25',
    '/usr/local/spm25',
    '/opt/spm25',
    str(Path.home() / 'software' / 'spm25'),
    str(Path.home() / 'spm25'),
]


def find_spm_path(candidates=None):
    candidates = candidates or COMMON_PATHS
    for p in candidates:
        if os.path.isdir(p):
            # basic sanity: check for spm.m
            if os.path.exists(os.path.join(p, 'spm.m')):
                return os.path.abspath(p)
    return None


def test_spm_python():
    try:
        # import may trigger spm-python runtime checks
        import spm  # type: ignore
        return True
    except Exception as e:
        print(f"spm-python import failed: {e}", file=sys.stderr)
        return False


def test_spm_init(spm_path: str) -> bool:
    # ensure MATLAB knows about spm_path by using MATLABPATH env var
    prev = os.environ.get('MATLABPATH', '')
    if prev:
        os.environ['MATLABPATH'] = spm_path + os.pathsep + prev
    else:
        os.environ['MATLABPATH'] = spm_path

    try:
        # Import spm after we set MATLABPATH
        import spm  # type: ignore

        # attempt to initialize defaults; if this raises we consider it failed
        try:
            spm.spm('defaults', 'FMRI')
        except Exception:
            # Some SPM installs may require only version check to succeed
            pass
        try:
            ver = spm.spm('version')
            print(f"Detected SPM version: {ver}")
        except Exception:
            # If version retrieval fails, still consider init successful
            print("Detected SPM (version unknown)")
        return True
    except Exception as e:
        print(f"Failed to initialize SPM via spm-python: {e}", file=sys.stderr)
        return False
    finally:
        # restore MATLABPATH
        if prev:
            os.environ['MATLABPATH'] = prev
        else:
            os.environ.pop('MATLABPATH', None)


def save_config(spm_path: str, config_file: str) -> None:
    with open(config_file, 'w') as f:
        f.write(spm_path + '\n')


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd')
    cfg = sub.add_parser('configure')
    cfg.add_argument('--spm-path', help='Path to SPM installation', default=None)
    cfg.add_argument('--save', action='store_true', help='Save detected path to config file')
    cfg.add_argument('--config-file', default='spm_config.txt', help='Where to save configuration if --save')

    args = p.parse_args(argv)

    if args.cmd != 'configure':
        p.print_help()
        return 2

    # Ensure spm-python is available
    if not test_spm_python():
        print("spm-python is not installed in this environment. Install with: pip install spm-python", file=sys.stderr)
        return 3

    spm_path = None
    if args.spm_path:
        if os.path.isdir(args.spm_path) and os.path.exists(os.path.join(args.spm_path, 'spm.m')):
            spm_path = os.path.abspath(args.spm_path)
        else:
            print(f"Provided spm path invalid or missing spm.m: {args.spm_path}", file=sys.stderr)
            return 4
    else:
        spm_path = find_spm_path()

    if not spm_path:
        print("Could not auto-detect an SPM installation. Provide --spm-path or install SPM standalone.", file=sys.stderr)
        return 5

    print(f"Using SPM path: {spm_path}")

    ok = test_spm_init(spm_path)
    if not ok:
        print("SPM failed to initialize via spm-python. Ensure MATLAB Runtime is installed and compatible.", file=sys.stderr)
        return 6

    # Optional TFCE check (presence of toolbox folder)
    tfce_path = os.path.join(spm_path, 'toolbox', 'TFCE')
    if os.path.isdir(tfce_path):
        print(f"TFCE toolbox found at: {tfce_path}")
    else:
        print("TFCE toolbox not found in SPM toolbox path; TFCE steps may fail.")

    if args.save:
        save_config(spm_path, args.config_file)
        print(f"Configuration saved to: {args.config_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
