#!/bin/bash
# Inspect an SPM.mat in headless MATLAB and print a concise design audit
# Usage: ./inspect_spm_headless.sh <stats_folder>

if [ $# -lt 1 ]; then
  echo "Usage: $0 <stats_folder>"
  exit 1
fi

STATS_FOLDER="$1"
if [[ ! "$STATS_FOLDER" = /* ]]; then
  STATS_FOLDER="$(pwd)/$STATS_FOLDER"
fi

if [ ! -d "$STATS_FOLDER" ]; then
  echo "Error: Folder not found: $STATS_FOLDER"
  exit 1
fi
if [ ! -f "$STATS_FOLDER/SPM.mat" ]; then
  echo "Error: SPM.mat not found in: $STATS_FOLDER"
  exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export DISPLAY=
export JAVA_TOOL_OPTIONS="-Djava.awt.headless=true"

if [ -x "/Applications/MATLAB_R2025b.app/bin/matlab" ]; then
  MATLAB_EXE="/Applications/MATLAB_R2025b.app/bin/matlab"
else
  MATLAB_EXE=$(find /Applications -path "*/bin/matlab" -type f 2>/dev/null | head -1)
fi

if [ -z "$MATLAB_EXE" ]; then
  echo "Error: MATLAB not found"
  exit 1
fi

CMD="cd '$SCRIPT_DIR'; inspect_spm_mat('$STATS_FOLDER')"
"$MATLAB_EXE" -nodesktop -nodisplay -nosplash -batch "$CMD" 2>&1

exit $?
