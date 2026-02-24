#!/usr/bin/env python3
"""
Generate SPM contrasts programmatically from design structure.

This replaces add_contrasts_longitudinal.m but works in standalone mode
(no addpath() needed, pure Python + SPM batch syntax).

Usage:
    python3 generate_contrasts.py /path/to/stats_dir [--output batch_file.m]
"""

import sys
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


def extract_design_info_via_matlab(stats_dir: str) -> List[str]:
    """
    Use MATLAB to extract parameter names (more robust than scipy.io).
    Writes spm_design_info.json which we then read.
    """
    stats_path = Path(stats_dir)
    json_file = stats_path / 'spm_design_info.json'
    temp_path = stats_path.parent / '.tmp_contrast_gen'
    
    # Determine MCR path from environment or use default
    mcr_path = Path('/data/local/software/cat-12/external/MCR/v232/R2023b')
    spm_script = Path('/data/local/software/cat-12/external/cat12/run_spm25.sh')
    
    if not spm_script.exists():
        raise FileNotFoundError(f"SPM script not found: {spm_script}")
    
    # Create temporary MATLAB script
    temp_script = temp_path / 'extract_design_temp.m'
    script_code = f"""
load('{stats_path}/SPM.mat', 'SPM');
param_names = SPM.xX.name;
fid = fopen('{json_file}', 'w');
fprintf(fid, '{{"\\n');
fprintf(fid, '  "parameters": [\\n');
for i = 1:length(param_names)
    if i > 1
        fprintf(fid, ',\\n');
    end
    name = param_names{{i}};
    name = strrep(name, '"', '\\\\"');
    fprintf(fid, '    "%s"', name);
end
fprintf(fid, '\\n  ]\\n');
fprintf(fid, '}}\\n');
fclose(fid);
exit;
"""
    
    temp_script.parent.mkdir(parents=True, exist_ok=True)
    temp_script.write_text(script_code)
    
    # Run SPM script
    try:
        result = subprocess.run(
            [str(spm_script), str(mcr_path), 'script', str(temp_script)],
            capture_output=True,
            timeout=30,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Warning: MATLAB extraction failed: {result.stderr[:200]}", file=sys.stderr)
            return []
    except Exception as e:
        print(f"Warning: Could not run MATLAB extraction: {e}", file=sys.stderr)
        return []
    
    # Try to read generated JSON
    if json_file.exists():
        try:
            with open(json_file) as f:
                data = json.load(f)
                return data.get('parameters', [])
        except Exception as e:
            print(f"Warning: Could not read generated JSON: {e}", file=sys.stderr)
    
    return []


def load_spm_mat(spm_file: str) -> Dict:
    """Load SPM.mat and extract design information."""
    try:
        import scipy.io as sio
        spm_data = sio.loadmat(spm_file, squeeze_me=True, struct_as_record=False)
        return spm_data
    except Exception as e:
        print(f"Error loading {spm_file}: {e}", file=sys.stderr)
        return {}



def parse_design_structure(param_names: List[str]) -> Tuple[int, int, List[str], List[int]]:
    """
    Parse parameter names to extract groups, timepoints.
    
    Returns:
        (n_groups, n_times, param_names, timepoints)
    """
    if not param_names:
        print("Error: No parameter names provided", file=sys.stderr)
        return 0, 0, [], []
    
    print(f"Parameters ({len(param_names)}):", file=sys.stderr)
    for i, name in enumerate(param_names[:10]):
        print(f"  {i+1}: {name}", file=sys.stderr)
    if len(param_names) > 10:
        print(f"  ... and {len(param_names) - 10} more", file=sys.stderr)
    
    # Extract groups and timepoints from parameter names
    # Expected format: "Group*Time_{group,time}" or similar
    groups = set()
    timepoints = set()
    
    for name in param_names:
        # Try to extract {group,time} pattern
        match = re.search(r'\{(\d+),(\d+)\}', name)
        if match:
            g_idx = int(match.group(1))
            t_idx = int(match.group(2))
            groups.add(g_idx)
            timepoints.add(t_idx)
    
    # Sort for consistent ordering
    groups = sorted(list(groups))
    timepoints = sorted(list(timepoints))
    
    n_groups = len(groups)
    n_times = len(timepoints)
    
    print(f"\nDetected design:", file=sys.stderr)
    print(f"  Groups: {n_groups} ({groups})", file=sys.stderr)
    print(f"  Timepoints: {n_times} ({timepoints})", file=sys.stderr)
    print(f"  Total parameters: {len(param_names)}\n", file=sys.stderr)
    
    return n_groups, n_times, param_names, timepoints


def generate_contrasts(param_names: List[str], n_groups: int, n_times: List[int]) -> List[Dict]:
    """
    Generate contrast specifications (weights vectors).
    
    Returns list of contrast dicts with 'name' and 'weights' keys.
    """
    n_params = len(param_names)
    contrasts = []
    
    print("Generating contrasts:", file=sys.stderr)
    
    # 1. MAIN EFFECT OF TIME (within each group)
    for g_idx in range(1, n_groups + 1):
        if len(n_times) >= 2:
            first_t = n_times[0]
            last_t = n_times[-1]
            
            weights = [0] * n_params
            
            for p_idx, name in enumerate(param_names):
                match = re.search(r'\{(\d+),(\d+)\}', name)
                if match:
                    g = int(match.group(1))
                    t = int(match.group(2))
                    
                    if g == g_idx:
                        if t == last_t:
                            weights[p_idx] = 1
                        elif t == first_t:
                            weights[p_idx] = -1
            
            # Only add if has non-zero weights
            if any(weights):
                contrast_name = f"Group{g_idx}: Time_linear"
                contrasts.append({
                    'name': contrast_name,
                    'weights': weights,
                    'type': 'T'  # T-contrast
                })
                print(f"  [{len(contrasts)}] {contrast_name}", file=sys.stderr)
    
    # 2. MAIN EFFECT OF GROUP (average over time)
    if n_groups >= 2:
        for g_idx in range(1, n_groups + 1):
            weights = [0] * n_params
            count = 0
            
            for p_idx, name in enumerate(param_names):
                match = re.search(r'\{(\d+),(\d+)\}', name)
                if match:
                    g = int(match.group(1))
                    if g == g_idx:
                        weights[p_idx] = 1 / len(n_times)  # average over time
                        count += 1
            
            if count > 0:
                contrast_name = f"Group{g_idx}: Main_effect"
                contrasts.append({
                    'name': contrast_name,
                    'weights': weights,
                    'type': 'T'
                })
                print(f"  [{len(contrasts)}] {contrast_name}", file=sys.stderr)
    
    # 3. GROUP COMPARISONS (if 2+ groups)
    groups_list = sorted(list(set(
        int(m.group(1)) for name in param_names 
        if (m := re.search(r'\{(\d+),\d+\}', name))
    )))
    
    if len(groups_list) >= 2:
        for i in range(len(groups_list) - 1):
            g1 = groups_list[i]
            g2 = groups_list[i + 1]
            
            weights = [0] * n_params
            
            for p_idx, name in enumerate(param_names):
                match = re.search(r'\{(\d+),(\d+)\}', name)
                if match:
                    g = int(match.group(1))
                    if g == g1:
                        weights[p_idx] = 1 / len(n_times)
                    elif g == g2:
                        weights[p_idx] = -1 / len(n_times)
            
            if any(weights):
                contrast_name = f"Group{g1}_vs_Group{g2}"
                contrasts.append({
                    'name': contrast_name,
                    'weights': weights,
                    'type': 'T'
                })
                print(f"  [{len(contrasts)}] {contrast_name}", file=sys.stderr)
    
    print(f"\nTotal contrasts generated: {len(contrasts)}\n", file=sys.stderr)
    
    return contrasts


def generate_batch_file(spm_dir: str, contrasts: List[Dict]) -> str:
    """
    Generate SPM batch file for contrast estimation.
    
    Returns MATLAB batch code as string.
    """
    batch_lines = [
        "% SPM Contrasts Batch File",
        "% Auto-generated by generate_contrasts.py",
        "",
        f"spm_dir = '{spm_dir}';",
        "load(fullfile(spm_dir, 'SPM.mat'));",
        "",
        "% Clear any existing contrasts",
        "SPM.xCon = [];",
        "",
    ]
    
    # Add each contrast
    for idx, contrast in enumerate(contrasts, 1):
        weights = contrast['weights']
        name = contrast['name'].replace("'", "''")
        
        weights_str = '[' + ', '.join(f'{w:.6f}' for w in weights) + ']'
        
        batch_lines.extend([
            f"% Contrast {idx}: {name}",
            f"SPM.xCon({idx}).name = '{name}';",
            f"SPM.xCon({idx}).STAT = 'T';",
            f"SPM.xCon({idx}).c = {weights_str};",
            f"SPM.xCon({idx}).orth = 0;",
            "",
        ])
    
    batch_lines.extend([
        "% Estimate contrasts",
        "spm_contrasts(SPM, 1:length(SPM.xCon));",
        "",
        "% Save SPM.mat with contrasts",
        "save(fullfile(spm_dir, 'SPM.mat'), 'SPM');",
        "",
        "fprintf('✓ Contrasts estimated and saved\\n');",
        "exit;",
    ])
    
    return '\n'.join(batch_lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 generate_contrasts.py <stats_dir> [--output <batch_file>]", 
              file=sys.stderr)
        sys.exit(1)
    
    stats_dir = sys.argv[1]
    output_file = None
    
    # Parse optional output file
    if '--output' in sys.argv:
        idx = sys.argv.index('--output')
        if idx + 1 < len(sys.argv):
            output_file = sys.argv[idx + 1]
    
    stats_path = Path(stats_dir)
    spm_mat = stats_path / 'SPM.mat'
    
    if not spm_mat.exists():
        print(f"Error: SPM.mat not found at {spm_mat}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading: {spm_mat}\n", file=sys.stderr)
    
    # Try to extract parameter names using MATLAB first (more reliable)
    param_names = extract_design_info_via_matlab(str(stats_path))
    
    # Fallback if MATLAB method didn't work
    if not param_names:
        print("Attempting fallback parameter extraction...", file=sys.stderr)
        # This fallback is simpler - just try scipy directly
        try:
            import scipy.io as sio
            spm_data = sio.loadmat(str(spm_mat))
            if 'SPM' in spm_data:
                spm = spm_data['SPM']
                # Try various ways to extract names
                if hasattr(spm, 'item'):
                    spm = spm.item()
                if isinstance(spm, dict) and 'xX' in spm:
                    xx = spm['xX']
                    if hasattr(xx, 'item'):
                        xx = xx.item()
                    if isinstance(xx, dict) and 'name' in xx:
                        names = xx['name']
                        if hasattr(names, 'tolist'):
                            param_names = names.tolist()
                        else:
                            param_names = list(names) if hasattr(names, '__iter__') else [str(names)]
        except Exception as e:
            print(f"Fallback extraction failed: {e}", file=sys.stderr)
    
    if not param_names:
        print("Error: Could not extract parameter names from SPM.mat", file=sys.stderr)
        print("This SPM.mat may be from incomplete model estimation", file=sys.stderr)
        sys.exit(1)
    
    # Parse and generate contrasts
    n_groups, n_times, param_names, timepoints = parse_design_structure(param_names)
    
    if n_groups < 1 or n_times < 1:
        print("Error: Could not detect design structure", file=sys.stderr)
        sys.exit(1)
    
    contrasts = generate_contrasts(param_names, n_groups, timepoints)
    
    if not contrasts:
        print("Warning: No contrasts generated", file=sys.stderr)
        return 1
    
    # Generate batch file
    batch_code = generate_batch_file(str(stats_path), contrasts)
    
    # Write to file or stdout
    if output_file:
        Path(output_file).write_text(batch_code)
        print(f"✓ Batch file written to: {output_file}", file=sys.stderr)
    else:
        print(batch_code)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
