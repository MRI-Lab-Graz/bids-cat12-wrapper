#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# CAT12 Complete Installation Script
# =============================================================================
# Detects OS and installs CAT12 standalone (Linux/macOS) or guides for Windows
# Automatically downloads correct versions and sets up environment
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXTERNAL_DIR="$REPO_ROOT/external"
VENV_DIR="$REPO_ROOT/.venv"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}CAT12 Complete Installation${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# =============================================================================
# STEP 1: Detect Operating System
# =============================================================================

OS_TYPE=$(uname -s)
ARCH=$(uname -m)

echo -e "${YELLOW}System Detection:${NC}"
echo "  OS: $OS_TYPE"
echo "  Architecture: $ARCH"
echo ""

case "$OS_TYPE" in
    Linux)
        echo -e "${GREEN}✓ Linux detected - using CAT12 standalone${NC}"
        OS_NAME="Linux"
        MCR_ARCH="glnxa64"
        CAT12_ARCH="Linux"
        MCR_FILENAME="MATLAB_Runtime_R2023b_Update_10_glnxa64.zip"
        CAT12_FILENAME="cat12_latest_R2023b_MCR_Linux.zip"
        ;;
    Darwin)
        echo -e "${GREEN}✓ macOS detected - using CAT12 standalone${NC}"
        OS_NAME="macOS"
        if [ "$ARCH" = "arm64" ]; then
            echo -e "${GREEN}✓ Apple Silicon (ARM64) detected${NC}"
            MCR_ARCH="maca64"
            CAT12_FILENAME="CAT12.9_R2023b_MCR_Mac_arm64.zip"
            MCR_FILENAME="MATLAB_Runtime_R2023b_Update_10_maca64.dmg"
            IS_ARM64=true
        else
            echo -e "${GREEN}✓ Intel (x86_64) detected${NC}"
            MCR_ARCH="maci64"
            CAT12_FILENAME="cat12_latest_R2023b_MCR_Mac.zip"
            MCR_FILENAME="MATLAB_Runtime_R2023b_Update_10_maci64.dmg.zip"
            IS_ARM64=false
        fi
        ;;
    MINGW*|MSYS*|CYGWIN*)
        echo -e "${YELLOW}⚠ Windows detected${NC}"
        echo -e "${RED}Windows support is not yet implemented.${NC}"
        echo ""
        echo "CAT12 standalone is available for Windows, but requires:"
        echo "  1. Manual download from: https://www.neuro.uni-jena.de/cat12/"
        echo "  2. CAT12.9_R2023b_MCR_Win.zip"
        echo "  3. MATLAB Runtime R2023b for Windows"
        echo ""
        echo "For now, use MATLAB directly or ask for Windows support."
        exit 1
        ;;
    *)
        echo -e "${RED}✗ Unknown OS: $OS_TYPE${NC}"
        exit 1
        ;;
esac

echo ""

# =============================================================================
# STEP 2: Create directories
# =============================================================================

echo -e "${YELLOW}Setting up directories:${NC}"
mkdir -p "$EXTERNAL_DIR"
echo -e "${GREEN}✓ Created $EXTERNAL_DIR${NC}"

# =============================================================================
# STEP 3: Download CAT12 Standalone
# =============================================================================

echo ""
echo -e "${YELLOW}Downloading CAT12 Standalone:${NC}"

CAT12_URL="https://www.neuro.uni-jena.de/cat12/$CAT12_FILENAME"
CAT12_ZIP="$EXTERNAL_DIR/$CAT12_FILENAME"

if [ -f "$CAT12_ZIP" ]; then
    echo -e "${GREEN}✓ CAT12 already downloaded${NC}"
else
    echo "  Downloading from: $CAT12_URL"
    echo "  Size: ~700MB (may take 1-2 minutes)"
    echo ""
    
    if ! curl -L -o "$CAT12_ZIP" "$CAT12_URL"; then
        echo -e "${RED}✗ Download failed!${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Downloaded successfully${NC}"
fi

# =============================================================================
# STEP 4: Extract CAT12
# =============================================================================

echo ""
echo -e "${YELLOW}Extracting CAT12:${NC}"

if [ "$OS_NAME" = "Linux" ]; then
    CAT12_DIR="$EXTERNAL_DIR/cat12"
    if [ ! -d "$CAT12_DIR" ]; then
        echo "  Extracting to: $EXTERNAL_DIR"
        unzip -q "$CAT12_ZIP" -d "$EXTERNAL_DIR"
        # Find and rename extracted directory to 'cat12'
        EXTRACTED_DIR=$(ls -d "$EXTERNAL_DIR"/CAT12* 2>/dev/null | head -1)
        if [ -n "$EXTRACTED_DIR" ] && [ -d "$EXTRACTED_DIR" ]; then
            mv "$EXTRACTED_DIR" "$CAT12_DIR"
        fi
        echo -e "${GREEN}✓ Extracted${NC}"
    else
        echo -e "${GREEN}✓ Already extracted${NC}"
    fi
elif [ "$OS_NAME" = "macOS" ]; then
    if [ "$IS_ARM64" = true ]; then
        CAT12_DIR="$EXTERNAL_DIR/CAT12.9_R2023b_MCR_MAC_arm64"
        EXPECTED_NAME="CAT12.9_R2023b_MCR_MAC_arm64"
    else
        CAT12_DIR="$EXTERNAL_DIR/cat12"
        EXPECTED_NAME="cat12_*"
    fi
    
    if [ ! -d "$CAT12_DIR" ]; then
        echo "  Extracting to: $EXTERNAL_DIR"
        unzip -n -q "$CAT12_ZIP" -d "$EXTERNAL_DIR"
        echo -e "${GREEN}✓ Extracted${NC}"
    else
        echo -e "${GREEN}✓ Already extracted${NC}"
    fi
fi

# Verify extraction
if [ -f "$CAT12_DIR/standalone/cat_standalone.sh" ]; then
    echo -e "${GREEN}✓ CAT12 standalone executable found${NC}"
else
    echo -e "${RED}✗ CAT12 extraction may have failed${NC}"
    exit 1
fi

# =============================================================================
# STEP 5: Download and Prepare MATLAB Runtime
# =============================================================================

echo ""
echo -e "${YELLOW}MATLAB Runtime R2023b:${NC}"

MCR_URL="https://ssd.mathworks.com/supportfiles/downloads/R2023b/Release/10/deployment_files/installer/complete/$MCR_ARCH/$MCR_FILENAME"
MCR_FILE="$EXTERNAL_DIR/$MCR_FILENAME"

if [ -f "$MCR_FILE" ]; then
    echo -e "${GREEN}✓ MCR already downloaded${NC}"
else
    echo "  Downloading from: $MCR_URL"
    echo "  Size: ~2GB (may take 3-5 minutes)"
    echo ""
    
    if ! curl -L -o "$MCR_FILE" "$MCR_URL"; then
        echo -e "${YELLOW}⚠ MCR download failed (will be needed for standalone)${NC}"
    else
        echo -e "${GREEN}✓ Downloaded successfully${NC}"
    fi
fi

# =============================================================================
# STEP 6: Python Environment (with UV)
# =============================================================================

echo ""
echo -e "${YELLOW}Setting up Python environment with UV:${NC}"

# Check for UV and install if needed
if ! command -v uv >/dev/null 2>&1; then
    echo -e "${YELLOW}UV not found. Installing UV...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo -e "${GREEN}✓ UV installed${NC}"
else
    echo -e "${GREEN}✓ UV found: $(command -v uv)${NC}"
fi

# Create venv using UV (or standard if UV unavailable)
if [ ! -d "$VENV_DIR" ]; then
    if command -v uv >/dev/null 2>&1; then
        echo "  Creating virtual environment with UV..."
        uv venv "$VENV_DIR"
        echo -e "${GREEN}✓ Virtual environment created with UV${NC}"
    elif command -v python3 &> /dev/null; then
        echo "  Creating virtual environment with Python..."
        python3 -m venv "$VENV_DIR"
        echo -e "${GREEN}✓ Virtual environment created${NC}"
    else
        echo -e "${RED}✗ Python3 not found${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Virtual environment exists at $VENV_DIR${NC}"
fi

# =============================================================================
# STEP 6b: Install Python packages with UV
# =============================================================================

echo ""
echo -e "${YELLOW}Installing Python packages with UV:${NC}"

if [ -f "$REPO_ROOT/requirements.txt" ]; then
    if command -v uv >/dev/null 2>&1; then
        export PATH="$HOME/.local/bin:$PATH"
        echo "  Installing packages from requirements.txt with UV..."
        uv pip install -r "$REPO_ROOT/requirements.txt"
        echo -e "${GREEN}✓ Python packages installed${NC}"
    else
        echo -e "${YELLOW}⚠ UV not available, skipping package installation${NC}"
        echo "   You can install packages later with:"
        echo "   source $VENV_DIR/bin/activate && pip install -r requirements.txt"
    fi
else
    echo -e "${YELLOW}⚠ requirements.txt not found, skipping package installation${NC}"
fi

# =============================================================================
# STEP 7: Configure Environment
# =============================================================================

echo ""
echo -e "${YELLOW}Configuring environment:${NC}"

# Create/update .env file
ENV_FILE="$REPO_ROOT/.env"
cat > "$ENV_FILE" << EOF
# CAT12 Standalone Configuration
# Generated by install.sh on $(date)

# Operating System
OS_NAME=$OS_NAME
ARCH=$ARCH

# CAT12 Standalone paths
CAT12_STANDALONE=$CAT12_DIR
SPMROOT=$CAT12_DIR
export CAT12_STANDALONE
export SPMROOT

# MATLAB Runtime (needed for standalone)
EOF

if [ "$OS_NAME" = "Linux" ]; then
    cat >> "$ENV_FILE" << 'EOF'
MCR_ROOT=/opt/MATLAB/MATLAB_Runtime/R2023b
export MCR_ROOT
EOF
elif [ "$OS_NAME" = "macOS" ]; then
    cat >> "$ENV_FILE" << 'EOF'
MCR_ROOT=/Applications/MATLAB/MATLAB_Runtime/R2023b
export MCR_ROOT
EOF
fi

cat >> "$ENV_FILE" << 'EOF'

# Use standalone version
USE_STANDALONE=true
export USE_STANDALONE

# Python
PYTHON_BIN=$(which python3)
export PYTHON_BIN
EOF

echo -e "${GREEN}✓ Created/updated .env file${NC}"

# =============================================================================
# STEP 8: Verification
# =============================================================================

echo ""
echo -e "${YELLOW}Verification:${NC}"

# Check CAT12
if [ -f "$CAT12_DIR/standalone/cat_standalone.sh" ]; then
    echo -e "${GREEN}✓ CAT12 standalone: $CAT12_DIR${NC}"
else
    echo -e "${RED}✗ CAT12 not found${NC}"
fi

# Check MCR
if [ -f "$MCR_FILE" ]; then
    echo -e "${GREEN}✓ MCR downloaded: $MCR_FILE${NC}"
    
    if [ "$OS_NAME" = "Linux" ]; then
        echo "  Installation: Extract and run installer manually"
        echo "    unzip $MCR_FILE -d /opt/MATLAB/"
    elif [ "$OS_NAME" = "macOS" ]; then
        if [ "$IS_ARM64" = true ]; then
            echo "  Installation: Double-click $MCR_FILE and run InstallForMacOSAppleSilicon.app"
        else
            echo "  Installation: Extract and run installer"
        fi
    fi
else
    echo -e "${YELLOW}⚠ MCR not found (will be needed for standalone)${NC}"
fi

# Check Python
if [ -d "$VENV_DIR" ]; then
    echo -e "${GREEN}✓ Python environment: $VENV_DIR${NC}"
else
    echo -e "${RED}✗ Python environment not found${NC}"
fi

# =============================================================================
# STEP 9: Installation Instructions
# =============================================================================

echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}NEXT STEPS:${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo ""

if [ "$OS_NAME" = "Linux" ]; then
    echo "1. Install MATLAB Runtime R2023b:"
    echo "   cd $EXTERNAL_DIR"
    echo "   unzip MATLAB_Runtime_R2023b_Update_10_glnxa64.zip"
    echo "   ./install -destinationFolder /opt/MATLAB/MATLAB_Runtime/R2023b -agreeToLicense yes"
    echo ""
elif [ "$OS_NAME" = "macOS" ]; then
    echo "1. Install MATLAB Runtime R2023b:"
    echo "   open $MCR_FILE"
    echo "   Run InstallForMacOSAppleSilicon.app (or InstallForMacOSX.app for Intel)"
    echo "   Install to: /Applications/MATLAB/MATLAB_Runtime/R2023b"
    echo ""
fi

echo "2. Activate Python environment:"
echo "   source $VENV_DIR/bin/activate"
echo ""

echo "3. Install Python dependencies:"
echo "   $REPO_ROOT/scripts/setup/install.sh"
echo ""

echo "4. Run CAT12 preprocessing:"
echo "   ./cat12_prepro openneuro/ds000114 projects/demo/derivatives/cat12 participant \\"
echo "     --preproc --participant-label 01 --session-label test \\"
echo "     --smooth-volume 6 --smooth-surface 12 --qa --tiv --no-validate"
echo ""

echo -e "${GREEN}✓ Installation setup complete!${NC}"
echo ""
echo "For more information, see:"
echo "  - INSTALL_GUIDE.md"
echo "  - STANDALONE_SETUP.md"
echo "  - DEMO_RUN.md"
echo ""
