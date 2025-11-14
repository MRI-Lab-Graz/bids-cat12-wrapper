# SPM Path Configuration Guide

This document explains how to configure the SPM (Statistical Parametric Mapping) path for analysis scripts in this project. The system now automatically detects SPM installations across different operating systems and provides multiple configuration options.

## 🚀 Quick Start

### Option 1: Automatic Detection (Recommended)
Simply run your analysis script (e.g., `run_screen_and_tfce.m`). The system will automatically try to find your SPM installation.

### Option 2: Use Configuration Tool
```matlab
configure_spm_path()
```
This interactive tool will guide you through the setup process.

### Option 3: Manual Configuration
Choose one of these methods:

#### Environment Variable (System-wide)
```bash
# macOS/Linux
export SPM_PATH="/path/to/your/spm"

# Windows
set SPM_PATH="C:\path\to\your\spm"
```

#### Configuration File (Project-specific)
Create a file named `spm_config.txt` in your project directory:
```
/path/to/your/spm
```

## 🔍 How Auto-Detection Works

The system searches for SPM in this order:

1. **Environment Variable**: Checks `SPM_PATH` environment variable
2. **Configuration File**: Looks for `spm_config.txt` in current directory  
3. **Common Locations**: Searches standard installation directories
4. **MATLAB Path**: Checks if SPM is already in MATLAB's path
5. **Interactive Input**: Prompts user if all else fails

## 📁 Common SPM Installation Locations

### macOS
- `/Applications/spm25`
- `/Applications/spm24`
- `/usr/local/spm25`
- `/opt/spm25`
- `~/software/spm25`
- `~/Documents/MATLAB/spm25`

### Linux
- `/usr/local/spm25`
- `/opt/spm25`
- `~/software/spm25`
- `~/spm25`
- `/software/spm25`

### Windows
- `C:\Program Files\spm25`
- `C:\spm25`
- `C:\software\spm25`
- `%USERPROFILE%\Documents\MATLAB\spm25`

## ⚙️ System Requirements

The system validates that your SPM installation includes:

### Critical Files (Required)
- `spm.m` - Main SPM function
- `spm_get_defaults.m` - SPM configuration
- `spm_vol.m` - Volume reading
- `spm_read_vols.m` - Volume data reading

### TFCE Toolbox (Required for TFCE Analysis)
- `toolbox/TFCE/tfce_estimate_stat.m` - TFCE main function

### Recommended Components
- CAT12 toolbox (`toolbox/cat12/`)
- GIfTI support (`external/gifti/`)

## 🛠️ Troubleshooting

### Problem: "SPM path not found"
**Solutions:**
1. Run `configure_spm_path()` to set up interactively
2. Set environment variable: `export SPM_PATH="/your/spm/path"`
3. Create `spm_config.txt` with your SPM path
4. Install SPM in a standard location

### Problem: "SPM installation incomplete"
**Solutions:**
1. Verify all required files exist in SPM directory
2. Re-download and reinstall SPM
3. Check file permissions

### Problem: "TFCE toolbox not found"
**Solutions:**
1. Download TFCE toolbox from SPM website
2. Extract to `[SPM_PATH]/toolbox/TFCE/`
3. Restart MATLAB

### Problem: Configuration not persisting
**Solutions:**
1. Save to config file: `echo "/path/to/spm" > spm_config.txt`
2. Set system environment variable (permanent)
3. Add to MATLAB startup script

## 🔧 Advanced Configuration

### Multiple SPM Versions
If you have multiple SPM versions, specify the preferred one:
```bash
export SPM_PATH="/usr/local/spm25"  # Use SPM25 specifically
```

### Network/Shared Installations
For shared network installations:
```bash
export SPM_PATH="/network/shared/software/spm25"
```

### Docker/Container Environments
For containerized environments, mount SPM and set the path:
```bash
docker run -v /host/spm:/container/spm -e SPM_PATH="/container/spm" ...
```

## 📝 Testing Your Configuration

After configuration, test it:

```matlab
% Quick test
spm_path = find_spm_path();
fprintf('SPM found at: %s\n', spm_path);

% Full test with configuration tool
configure_spm_path();
```

## 🔄 Migration from Old System

If you were using the old hardcoded system:

1. **No changes needed** - the new system is backward compatible
2. **Optional**: Remove hardcoded paths and use auto-detection
3. **Recommended**: Run `configure_spm_path()` to set up properly

## 💡 Best Practices

1. **Use environment variables** for system-wide configuration
2. **Use config files** for project-specific setups
3. **Test configuration** after changes
4. **Keep TFCE toolbox updated** for best results
5. **Document your setup** for team members

## 🆘 Getting Help

If you encounter issues:

1. Run the configuration tool: `configure_spm_path()`
2. Check the error messages for specific guidance
3. Verify SPM installation is complete
4. Check MATLAB and system permissions
5. Contact system administrator for network/shared installations

---

*This flexible path system ensures compatibility across different operating systems and installation setups, making the analysis scripts more portable and user-friendly.*