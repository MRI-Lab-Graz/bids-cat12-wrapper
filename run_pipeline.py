#!/usr/bin/env python3
"""Root launcher for unified pipeline script."""

from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts" / "pipeline" / "run_pipeline.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
