#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(
        description="Run MATLAB commands using CAT12 Standalone (MCR).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python run_matlab_standalone.py "cat_standalone_segment('T1.nii')"
  python run_matlab_standalone.py "my_custom_script(1, 2)" --mcr /path/to/mcr --standalone /path/to/run_spm12.sh

This utility creates a temporary MATLAB script that adds the specified utils path,
executes your command within a try-catch block, and handles exit codes properly.
        """
    )
    parser.add_argument("command", help="MATLAB command or function call to execute (e.g. \"disp('hello')\")")
    parser.add_argument("--mcr", help="Path to MATLAB Runtime (MCR) directory (e.g., .../v913)", default="/data/local/software/cat-12/external/MCR/v93")
    parser.add_argument("--standalone", help="Path to the run_spm12.sh execution script", default="/data/local/software/cat-12/external/cat12/run_spm12.sh")
    parser.add_argument("--utils", help="Path to additional MATLAB utilities to add to path", default="/data/local/software/cat-12/stats/utils_clean")
    
    args = parser.parse_args()
    
    # Create a temporary MATLAB script
    # We use a fixed name in /tmp to avoid clutter, or a temp file
    # Using a temp file is safer for concurrency
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.m', delete=False) as tmp:
        script_path = tmp.name
        tmp.write(f"addpath('{args.utils}');\n")
        tmp.write("try\n")
        tmp.write(f"    {args.command};\n")
        tmp.write("catch e\n")
        tmp.write("    fprintf('ERROR: %s\\n', e.message);\n")
        tmp.write("    exit(1);\n")
        tmp.write("end\n")
        tmp.write("exit(0);\n")
        
    try:
        cmd = [args.standalone, args.mcr, "script", script_path]
        # print(f"Executing: {' '.join(cmd)}")
        subprocess.check_call(cmd)
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)

if __name__ == "__main__":
    main()
