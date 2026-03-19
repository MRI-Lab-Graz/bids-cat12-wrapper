#!/usr/bin/env python3
from pathlib import Path
import os
import runpy
import sys

WORKSPACE = Path(__file__).resolve().parent
VENV_PYTHON = WORKSPACE / ".venv" / "bin" / "python3.11"

if __name__ == "__main__":
    # Re-exec under the project venv if we're not already using it.
    if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

    target = WORKSPACE / "scripts" / "webui" / "app.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
