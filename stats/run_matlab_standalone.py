#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import tempfile

def main():
    parser = argparse.ArgumentParser(description="Run MATLAB commands using CAT12 Standalone")
    parser.add_argument("command", help="MATLAB command to execute (e.g. \"my_func('arg')\")")
    parser.add_argument("--mcr", help="Path to MCR", default="/data/local/software/cat-12/external/MCR/v93")
    parser.add_argument("--standalone", help="Path to run_spm12.sh", default="/data/local/software/cat-12/external/cat12/run_spm12.sh")
    parser.add_argument("--utils", help="Path to utils folder", default="/data/local/software/cat-12/stats/utils_clean")
    
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
