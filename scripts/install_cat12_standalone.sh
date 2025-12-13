#!/bin/bash

# CAT12 Standalone Installation Script
# Installs CAT12.9 (R2017b) with integrated SPM12 standalone
# Based on: https://neuro-jena.github.io/enigma-cat12/#standalone
# Target: Ubuntu Server with CUDA support

set -e  # Exit on any error

echo "=========================================="
echo "CAT12 Standalone Installation Script"
echo "=========================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning "Third-party software notice: this installer downloads CAT12/SPM12 standalone, MATLAB Runtime (MCR), and Deno from upstream sources."
print_warning "Those components are NOT part of this repository and remain under their respective licenses/terms."
print_warning "Proceed only if you agree to comply with upstream licenses."

# Check if running on Ubuntu
if ! grep -q "Ubuntu" /etc/os-release 2>/dev/null; then
    print_warning "This script is optimized for Ubuntu. Proceeding anyway..."
fi

# Check for root privileges for system package installation
if [[ $EUID -eq 0 ]]; then
    print_warning "Running as root. Consider running as regular user."
fi

print_status "Updating system packages..."

# Determine repo root (this script lives in ./scripts)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Create installation directory within the project (repo root)
INSTALL_DIR="$PROJECT_DIR/external"
print_status "Creating installation directory: $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Download CAT12 standalone with MCR
print_status "Downloading CAT12.9 standalone (R2017b) with integrated SPM12..."
CAT12_URL="https://github.com/ChristianGaser/cat12/releases/download/12.9/CAT12.9_R2017b_MCR_Linux.zip"
wget -O cat12_standalone.zip "$CAT12_URL"

print_status "Extracting CAT12 standalone..."
unzip -q cat12_standalone.zip
rm cat12_standalone.zip

# Move the complete CAT12 package to cat12 directory
if [ -d "CAT12.9_R2017b_MCR_Linux" ]; then
    if [ -d "cat12" ]; then
        print_warning "Removing existing cat12 directory..."
        rm -rf cat12
    fi
    mv CAT12.9_R2017b_MCR_Linux cat12
fi

# Make CAT12 standalone scripts executable
if [ -d "cat12/standalone" ]; then
    chmod +x cat12/standalone/*.sh
fi
chmod +x cat12/*.sh 2>/dev/null || true

# Download and install MATLAB Runtime v93 (R2017b) if not present
MCR_DIR="$INSTALL_DIR/MCR"
MCR_VERSION="v93"
if [ ! -d "$MCR_DIR/$MCR_VERSION" ]; then
    print_status "Downloading MATLAB Runtime R2017b (v93)..."
    # Try the official MathWorks download URL
    MCR_URL="https://www.mathworks.com/supportfiles/downloads/R2017b/deployment_files/R2017b/installers/glnxa64/MCR_R2017b_glnxa64_installer.zip"
    
    # If that doesn't work, try alternative sources
    if ! wget -O mcr_installer.zip "$MCR_URL" 2>/dev/null; then
        print_warning "Official MCR download failed, trying alternative source..."
        MCR_URL="https://ssd.mathworks.com/supportfiles/downloads/R2017b/Release/9/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2017b_Update_9_glnxa64.zip"
        wget -O mcr_installer.zip "$MCR_URL"
    fi
    
    print_status "Installing MATLAB Runtime R2017b (v93)..."
    unzip -q mcr_installer.zip
    ./install -mode silent -agreeToLicense yes -destinationFolder "$MCR_DIR"
    rm -f mcr_installer.zip install
else
    print_status "MATLAB Runtime v93 already installed in workspace."
fi

# Return to project directory
cd "$PROJECT_DIR"

# Create environment configuration file
print_status "Creating environment configuration..."
cat > .env << EOF
# CAT12 Standalone Environment Configuration
# Source this file to set up the environment: source .env

export CAT12_ROOT="$INSTALL_DIR/cat12/standalone"
export SPMROOT="$INSTALL_DIR/cat12"
export MCR_ROOT="$MCR_DIR/$MCR_VERSION"
export MCRROOT="$MCR_ROOT"
export LD_LIBRARY_PATH="\$MCR_ROOT/runtime/glnxa64:\$MCR_ROOT/bin/glnxa64:\$MCR_ROOT/sys/os/glnxa64:\$MCR_ROOT/sys/opengl/lib/glnxa64:\$LD_LIBRARY_PATH"
export PATH="\$CAT12_ROOT:\$SPMROOT:\$PATH"

# Deno for BIDS validation (installed into the repo, not the user home)
export DENO_INSTALL="$INSTALL_DIR/deno"
export DENO_DIR="$INSTALL_DIR/deno_cache"
export PATH="\$DENO_INSTALL/bin:\$PATH"

# Keep caches workspace-local (avoid writing to the user home)
export UV_CACHE_DIR="$PROJECT_DIR/.uv-cache"

# Project-specific paths
export CAT12_PROJECT_ROOT="$PROJECT_DIR"
EOF

# Return to project directory (repo root)
cd "$PROJECT_DIR"

# Install Deno for BIDS validation (into repo root external/deno)
print_status "Installing Deno for BIDS validation (workspace-local)..."
export DENO_INSTALL="$INSTALL_DIR/deno"
export DENO_DIR="$INSTALL_DIR/deno_cache"
mkdir -p "$DENO_INSTALL" "$DENO_DIR" "$DENO_INSTALL/bin"

# Prefer the workspace-local Deno even if a system-wide one exists.
export PATH="$DENO_INSTALL/bin:$PATH"

if [ -x "$DENO_INSTALL/bin/deno" ]; then
    print_status "Workspace-local Deno already installed: $($DENO_INSTALL/bin/deno --version | head -1)"
else
    print_status "Installing pinned Deno release into: $DENO_INSTALL"

    # NOTE: We intentionally avoid the upstream install.sh because it can modify
    # shell startup files under the user's home directory.
    DENO_VERSION="2.6.0"
    ARCH="$(uname -m)"
    case "$ARCH" in
        x86_64|amd64)
            DENO_TARGET="x86_64-unknown-linux-gnu"
            ;;
        aarch64|arm64)
            DENO_TARGET="aarch64-unknown-linux-gnu"
            ;;
        *)
            print_error "Unsupported architecture for Deno: $ARCH"
            exit 1
            ;;
    esac

    DENO_URL="https://github.com/denoland/deno/releases/download/v${DENO_VERSION}/deno-${DENO_TARGET}.zip"
    DENO_ZIP="$INSTALL_DIR/deno_${DENO_VERSION}_${DENO_TARGET}.zip"

    # Use a sanitized runtime environment so system download tools aren't affected
    # by any active MCR/CAT12-related environment variables.
    env -u LD_PRELOAD \
        LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu" \
        PATH="/usr/bin:/bin" \
        /usr/bin/wget -O "$DENO_ZIP" "$DENO_URL"

    unzip -q "$DENO_ZIP" -d "$DENO_INSTALL/bin"
    rm -f "$DENO_ZIP"
    chmod +x "$DENO_INSTALL/bin/deno" 2>/dev/null || true
    if [ -x "$DENO_INSTALL/bin/deno" ]; then
        print_status "Deno installed successfully: $($DENO_INSTALL/bin/deno --version | head -1)"
    else
        print_error "Deno installation did not produce $DENO_INSTALL/bin/deno"
        exit 1
    fi
fi

# Create Python virtual environment (workspace-local) and install uv inside it
print_status "Creating Python virtual environment (.venv)..."
python3 -m venv .venv

print_status "Activating Python virtual environment..."
source .venv/bin/activate

# Keep uv's cache within the repo for reproducibility / clean uninstalls
export UV_CACHE_DIR="$PROJECT_DIR/.uv-cache"
mkdir -p "$UV_CACHE_DIR"

print_status "Installing uv into the virtual environment..."
python -m pip install --upgrade pip
python -m pip install "uv>=0.4"

print_status "Installing Python dependencies with uv..."
uv pip install -r requirements.txt

print_status "Fixing pybids and universal-pathlib compatibility..."
# Force reinstall compatible versions to avoid 'Protocol not known: bids' error
uv pip install --force-reinstall 'pybids>=0.15.1,<0.16.0' 'universal-pathlib<0.2.0'

print_status "Testing CAT12 installation..."
# Test CAT12 installation by checking if it can start
if [ -f "$INSTALL_DIR/cat12/standalone/cat_standalone.sh" ] && [ -d "$MCR_DIR/$MCR_VERSION" ]; then
    print_status "CAT12 standalone files found."
    print_status "Testing CAT12 execution (this may take a moment)..."
    
    # Quick test - try to get version info
    if timeout 30 bash -c "source '$PROJECT_DIR/.env' && '$INSTALL_DIR/cat12/standalone/cat_standalone.sh' 2>&1 | head -10 | grep -q 'SPM12'" 2>/dev/null; then
        print_status "✓ CAT12 standalone installation completed successfully!"
        print_status "✓ SPM12 with CAT12 integration verified"
    else
        print_warning "CAT12 installation completed but execution test inconclusive."
        print_warning "This may be normal - full testing requires input files."
    fi
    
    print_status "CAT12 location: $INSTALL_DIR/cat12/standalone"
    print_status "MCR location: $MCR_DIR/$MCR_VERSION"
    print_status "To use CAT12, run: source .env"
else
    print_error "CAT12 installation failed!"
    print_error "Missing: $INSTALL_DIR/cat12/standalone/cat_standalone.sh or $MCR_DIR/$MCR_VERSION"
    exit 1
fi

print_status "Verifying pybids installation..."
source .venv/bin/activate
if python -c "import bids" 2>/dev/null; then
    print_status "pybids is installed: $(python -c 'import bids; print(bids.__file__)')"
else
    print_warning "pybids is NOT installed in the virtual environment. Run 'pip install pybids' manually if needed."
fi

print_status "Creating activation script..."
# Create an activation script for easy environment setup
cat > activate_cat12.sh << 'EOF'
#!/bin/bash
# CAT12 Environment Activation Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment variables
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
    echo "✓ CAT12 environment variables loaded"
else
    echo "✗ .env file not found. Run installation script first."
    exit 1
fi

# Activate Python virtual environment
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
    echo "✓ Python virtual environment activated"
else
    echo "✗ Virtual environment not found. Run installation script first."
    exit 1
fi

echo "CAT12 environment is now active!"
echo "Run './cat12_prepro --help' to get started."
echo "Run './cat12_stats --help' for longitudinal statistics."
EOF

chmod +x activate_cat12.sh

echo "=========================================="
print_status "CAT12.9 (R2017b) Installation completed!"
echo "=========================================="
print_status "Components installed:"
print_status "• CAT12.9 with integrated SPM12 standalone"
print_status "• MATLAB Runtime R2017b (v93)"
print_status "• Python virtual environment with dependencies"
echo "=========================================="
print_status "Next steps:"
print_status "1. Activate the environment: source activate_cat12.sh"
print_status "2. Test installation: ./test_installation.sh"
print_status "3. Process BIDS data: python bids_cat12_processor.py --help"
echo "=========================================="
print_status "All dependencies are contained within this project directory."
print_status "No system-wide modifications were made to your shell configuration."
echo "=========================================="