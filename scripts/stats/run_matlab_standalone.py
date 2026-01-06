#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import tempfile


def main():
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
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
    parser.add_argument("--mcr", help="Path to MATLAB Runtime (MCR) directory", default=os.path.join(workspace_root, "external/MCR/v232"))
    parser.add_argument("--standalone", help="Path to the run_spm12.sh execution script", default=os.path.join(workspace_root, "external/cat12/run_spm12.sh"))
    parser.add_argument("--utils", help="Path to additional MATLAB utilities to add to path", default=os.path.join(workspace_root, "scripts/stats/utils"))
    parser.add_argument("--use-matlab", action="store_true", help="Use local MATLAB installation instead of standalone")
    parser.add_argument("--matlab-exe", help="Path to MATLAB executable", default="matlab")
    parser.add_argument("--spm-path", help="Path to SPM12 installation (required for local MATLAB)")
    
    args = parser.parse_args()
    
    # Create a temporary MATLAB script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.m', delete=False) as tmp:
        script_path = tmp.name
        script_dir = os.path.dirname(script_path)
        script_name = os.path.basename(script_path).replace('.m', '')

        if args.use_matlab:
            if args.spm_path and os.path.exists(args.spm_path):
                tmp.write(f"addpath('{args.spm_path}');\n")
            tmp.write("try, spm('defaults','FMRI'); end;\n")

        tmp.write(f"addpath('{args.utils}');\n")
        tmp.write("try\n")
        tmp.write(f"    {args.command};\n")
        tmp.write("catch e\n")
        tmp.write("    fprintf('ERROR: %s\\n', e.message);\n")
        tmp.write("    exit(1);\n")
        tmp.write("end\n")
        tmp.write("exit(0);\n")
        
    try:
        if args.use_matlab:
            # Use local MATLAB -batch mode (supported since R2019a)
            # We must be in the directory of the script or add it to path
            cmd = [args.matlab_exe, "-batch", f"addpath('{script_dir}'); {script_name}"]
            print(f"Executing via local MATLAB: {' '.join(cmd)}")
        else:
            if not os.path.exists(args.standalone):
                raise FileNotFoundError(f"Standalone script not found at {args.standalone}. Use --use-matlab if you have a local MATLAB installation.")
            cmd = [args.standalone, args.mcr, "script", script_path]
            # print(f"Executing via Standalone: {' '.join(cmd)}")
        subprocess.check_call(cmd)
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)

if __name__ == "__main__":
    main()
