#!/bin/bash
# Script to compile SPM12 MEX files for Apple Silicon (arm64)

# Paths
SPM_DIR="/Volumes/Evo/software/cat-12/external/matlab_tools/spm12"
MEX_BIN="/Applications/MATLAB_R2025b.app/bin/mex"

echo "=========================================================="
echo "Compiling SPM12 MEX files for Apple Silicon (arm64)"
echo "=========================================================="

# Check if MEX exists
if [ ! -f "$MEX_BIN" ]; then
    echo "ERROR: MATLAB mex compiler not found at $MEX_BIN"
    exit 1
fi

# Check if source exists
if [ ! -d "$SPM_DIR/src" ]; then
    echo "ERROR: SPM12 src directory not found at $SPM_DIR/src"
    exit 1
fi

echo "Checking for Xcode license..."
if ! xcrun clang --version >/dev/null 2>&1; then
    echo "----------------------------------------------------------"
    echo "CRITICAL: Xcode license not accepted or compiler missing."
    echo "Please run the following command in your terminal first:"
    echo "    sudo xcodebuild -license accept"
    echo "----------------------------------------------------------"
fi

echo "Step 1: Compiling core SPM12 MEX files..."
cd "$SPM_DIR/src"
make MEX="$MEX_BIN" install

echo "Step 2: Checking CAT12 MEX files..."
# CAT12 usually ships with maca64, but we can try to compile if needed.
# For now, SPM12 is the primary blocker for spm_sample_vol.
echo "SPM12 compilation finished. Check for errors above."

echo "Step 3: Verifying maca64 files..."
ls "$SPM_DIR"/*.mexmaca64 2>/dev/null | head -n 5

echo "=========================================================="
echo "Done. If you saw 'Error 255', please check the Xcode license."
echo "=========================================================="
