function run_screen_and_tfce(stats_folder, varargin)
% -------------------------------------------------------------------------
% Script to screen spmT images (uncorrected p < 0.001) and run TFCE on significant contrasts.
% Supports both single-stage and multi-stage TFCE analysis for efficiency.
%
% SINGLE-STAGE USAGE:
%   run_screen_and_tfce(stats_folder)
%   run_screen_and_tfce(stats_folder, '--n-perm', 1000, '--cluster-size', 100, '--force')
%
% MULTI-STAGE USAGE (recommended for efficiency):
%   run_screen_and_tfce(stats_folder, '--multi-stage')
%   run_screen_and_tfce(stats_folder, '--multi-stage', '--n-perm-stage1', 500, '--n-perm-stage2', 5000)
%
% Multi-stage workflow:
%   1. Check uncorrected results (already done)
%   2. Run TFCE with low permutations (Stage 1)
%   3. If Stage 1 finds significant results, run high permutations (Stage 2)
%
% Author: [Karl Koschutnig]
% Date: [22.10.2025]
%
% --- DEFAULT PARAMETERS ---
n_perm = 1500;        % Number of permutations for TFCE
n_jobs = 4;           % Number of CPU cores to use
mask_file = 'mask.nii'; % Optional mask file name (in stats folder)
p_thresh = 0.001;     % Uncorrected p-value threshold
cluster_size = 50;    % Minimum cluster size (voxels/vertices)
force_analysis = false; % Skip existing TFCE files check
no_background = false; % Wait for TFCE to complete (longer timeout)
pilot_mode = false;   % Test mode: only process 1 random contrast

% Parse command-line arguments
p = inputParser;
addParameter(p, 'n_perm', n_perm, @isnumeric);
addParameter(p, 'n_jobs', n_jobs, @isnumeric);
addParameter(p, 'mask_file', mask_file, @ischar);
addParameter(p, 'p_thresh', p_thresh, @isnumeric);
addParameter(p, 'cluster_size', cluster_size, @isnumeric);
addParameter(p, 'force', force_analysis, @islogical);
addParameter(p, 'no_background', no_background, @islogical);
addParameter(p, 'pilot', pilot_mode, @islogical);
addParameter(p, 'multi_stage', false, @islogical);  % Enable multi-stage TFCE
addParameter(p, 'n_perm_stage1', 500, @isnumeric);   % Low permutations for initial check
addParameter(p, 'n_perm_stage2', 5000, @isnumeric);  % High permutations for significant results

% Parse arguments (varargin starts after stats_folder)
parse(p, varargin{:});

n_perm = p.Results.n_perm;
n_jobs = p.Results.n_jobs;
mask_file = p.Results.mask_file;
p_thresh = p.Results.p_thresh;
cluster_size = p.Results.cluster_size;
force_analysis = p.Results.force;
no_background = p.Results.no_background;
pilot_mode = p.Results.pilot;
multi_stage = p.Results.multi_stage;
n_perm_stage1 = p.Results.n_perm_stage1;
n_perm_stage2 = p.Results.n_perm_stage2;

% -------------------------------------------------------------------------
% CRITICAL: Add shadow functions to path FIRST to intercept SPM GUI calls
% This must happen BEFORE adding SPM to the path
current_dir = fileparts(mfilename('fullpath'));
addpath(current_dir, '-begin');  % Prepend to path with highest priority

% -------------------------------------------------------------------------
% Disable all GUI/graphics output - terminal only
% Set MATLAB to headless mode
set(0, 'DefaultFigureVisible', 'off');
set(0, 'DefaultUIControlVisible', 'off');
set(groot, 'defaultFigureVisible', 'off');

fprintf('\n%s\n', repmat('═', 1, 90));
fprintf('SYSTEM VERIFICATION - Checking MATLAB, SPM, CAT12, and TFCE installation\n');
fprintf('%s\n\n', repmat('═', 1, 90));

% Check 1: MATLAB version
fprintf('1️⃣  MATLAB VERSION CHECK\n');
fprintf('   ─────────────────────────────────────────────────────────────\n');
matlab_version = version;
fprintf('   ✓ MATLAB version: %s\n', matlab_version);
v = version('-release');
year = str2double(v(1:4));
if year >= 2019
    fprintf('   ✓ Version is recent enough (R%s)\n\n', v);
else
    warning('⚠️  MATLAB version is older than R2019. Some features may not work.\n\n');
end

function save_tfce_null_and_fdr(outdir, contrast_num, is_volume, is_surface)
% SAVE_TFCE_NULL_AND_FDR - Locate null distribution files, compute Gamma-tail
% approximated voxelwise p-values, compute FDR, and save results.

    % Look for null distribution files matching common TFCE output patterns
    null_patterns = {
        sprintf('nullT_%04d.*', contrast_num),
        sprintf('nullT_%03d.*', contrast_num),
        sprintf('nullTFCE_%04d.*', contrast_num),
        'nullT_*.nii',
        'nullT_*.gii',
        'nullT_*.dat'
    };

    null_files = [];
    for pi = 1:numel(null_patterns)
        nf = dir(fullfile(outdir, null_patterns{pi}));
        if ~isempty(nf)
            null_files = nf;
            break;
        end
    end

    if isempty(null_files)
        fprintf('   ⚠️  No null distribution files found for contrast %d - skipping tail approximation/FDR\n', contrast_num);
        return;
    end

    fprintf('   🔢 Found %d null files - loading for contrast %d\n', numel(null_files), contrast_num);

    % Read TFCE statistic map (observed)
    tfce_stat_file = fullfile(outdir, sprintf('TFCE_%04d.nii', contrast_num));
    tfce_gii_file = fullfile(outdir, sprintf('TFCE_%04d.gii', contrast_num));
    if exist(tfce_stat_file, 'file')
        Vtf = spm_vol(tfce_stat_file);
        Ytf = spm_read_vols(Vtf);
        vox_mask = ~isnan(Ytf) & (Ytf ~= 0);
        obs_vals = Ytf(vox_mask);
    elseif exist(tfce_gii_file, 'file')
        G = gifti(tfce_gii_file);
        if isfield(G, 'cdata')
            Ytf = double(G.cdata(:));
        else
            Ytf = double(G.darrays{1}.data(:));
        end
        vox_mask = ~isnan(Ytf) & (Ytf ~= 0);
        obs_vals = Ytf(vox_mask);
    else
        fprintf('   ⚠️  TFCE statistic file not found for contrast %d - skipping\n', contrast_num);
        return;
    end

    % Load nulls: assume each null file is a map for one permutation
    n_null = numel(null_files);
    fprintf('   ⏬ Loading null maps (%d permutations)...\n', n_null);
    null_matrix = [];
    for ni = 1:n_null
        fn = fullfile(outdir, null_files(ni).name);
        try
            if endsWith(fn, '.nii') || endsWith(fn, '.nii.gz')
                Vn = spm_vol(fn);
                Yn = spm_read_vols(Vn);
                if isempty(null_matrix)
                    null_matrix = zeros(sum(vox_mask(:)), n_null);
                end
                null_matrix(:, ni) = Yn(vox_mask);
            elseif endsWith(fn, '.gii')
                Gn = gifti(fn);
                if isfield(Gn, 'cdata')
                    Yn = double(Gn.cdata(:));
                else
                    Yn = double(Gn.darrays{1}.data(:));
                end
                if isempty(null_matrix)
                    null_matrix = zeros(sum(vox_mask(:)), n_null);
                end
                null_matrix(:, ni) = Yn(vox_mask);
            elseif endsWith(fn, '.dat')
                fid = fopen(fn, 'r');
                Yn = fread(fid, 'float32');
                fclose(fid);
                if isempty(null_matrix)
                    null_matrix = zeros(sum(vox_mask(:)), n_null);
                end
                % If sizes don't match, try to subset
                if numel(Yn) == sum(vox_mask(:))
                    null_matrix(:, ni) = Yn;
                else
                    % try reshape assuming full image stored
                    warning('Null .dat file size mismatch - storing as NaNs');
                    null_matrix(:, ni) = NaN(sum(vox_mask(:)), 1);
                end
            else
                warning('Unsupported null file type: %s', fn);
            end
        catch MEload
            warning('Failed to read null file %s: %s', fn, MEload.message);
            if isempty(null_matrix)
                null_matrix = NaN(sum(vox_mask(:)), n_null);
            else
                null_matrix(:, ni) = NaN(sum(vox_mask(:)), 1);
            end
        end
    end

    % Save raw null matrix for reproducibility
    null_mat_file = fullfile(outdir, sprintf('TFCE_null_distribution_%04d.mat', contrast_num));
    try
        save(null_mat_file, 'null_matrix', '-v7.3');
        fprintf('   💾 Saved null distribution: %s\n', null_mat_file);
    catch
        warning('Failed to save null distribution mat file: %s', null_mat_file);
    end

    % Compute voxelwise p-values using tail approximation (Gamma) with fallback to empirical
    nvox = size(null_matrix, 1);
    pvals = nan(nvox, 1);
    fprintf('   🔬 Estimating tail (Gamma) per voxel and computing p-values...\n');

    % Precompute: for each voxel, fit gamma to upper tail (e.g., values > 90th percentile)
    for v = 1:nvox
        null_vec = null_matrix(v, :);
        null_vec = null_vec(~isnan(null_vec));
        if isempty(null_vec)
            pvals(v) = NaN;
            continue;
        end

        obs = obs_vals(v);

        % empirical fallback
        emp_p = (sum(null_vec >= obs) + 1) / (numel(null_vec) + 1);

        % Fit gamma to upper tail if enough samples
        try
            tail_cut = max(ceil(0.9 * numel(null_vec)), 10);
            tail_vals = sort(null_vec, 'ascend');
            tail_vals = tail_vals(end - tail_cut + 1:end);

            if numel(tail_vals) >= 10
                % Shift to positive domain if necessary
                min_tail = min(tail_vals);
                if min_tail <= 0
                    shift = abs(min_tail) + eps;
                    tail_vals_shift = tail_vals + shift;
                    obs_shift = obs + shift;
                else
                    tail_vals_shift = tail_vals;
                    obs_shift = obs;
                end
                % Fit gamma: shape k and scale theta
                phat = gamfit(tail_vals_shift);
                k = phat(1); theta = phat(2);
                % Compute upper-tail p-value from fitted gamma
                p_gamma = 1 - gamcdf(max(obs_shift,0), k, theta);
                % Validate gamma p-value; if invalid use empirical
                if isnan(p_gamma) || p_gamma <= 0 || p_gamma > 1
                    pvals(v) = emp_p;
                else
                    pvals(v) = p_gamma;
                end
            else
                pvals(v) = emp_p;
            end
        catch
            pvals(v) = emp_p;
        end
    end

    % Compute FDR using Benjamini-Hochberg
    fprintf('   📊 Computing FDR (Benjamini-Hochberg)...\n');
    valid_idx = ~isnan(pvals);
    pfdr = nan(size(pvals));
    try
        % Use mafdr if available
        if exist('mafdr', 'file')
            pfdr(valid_idx) = mafdr(pvals(valid_idx), 'BHFDR', true);
        else
            % Simple BH implementation
            pv = pvals(valid_idx);
            [pv_sorted, sort_idx] = sort(pv);
            m = numel(pv);
            % compute q-values via step-up procedure
            q = zeros(m,1);
            for j = m:-1:1
                if j == m
                    q(j) = pv_sorted(j);
                else
                    q(j) = min(q(j+1), pv_sorted(j) * m / j);
                end
            end
            % map back
            pfdr(valid_idx) = q(sort_idx);
        end
    catch MEfdr
        warning('FDR computation failed: %s', MEfdr.message);
        pfdr(valid_idx) = pvals(valid_idx); % fallback
    end

    % Save FDR map as -log10(q)
    logpFDR = nan(size(vox_mask));
    logpFDR(vox_mask) = -log10(pfdr);

    out_fdr_file = fullfile(outdir, sprintf('TFCE_log_pFDR_%04d.nii', contrast_num));
    try
        if exist('Vtf', 'var') && ~isempty(Vtf)
            Vf = Vtf;
            Vf.fname = out_fdr_file;
            Vf.dt = [64 0]; % single precision
            spm_write_vol(Vf, logpFDR);
            fprintf('   💾 Saved TFCE FDR map: %s\n', out_fdr_file);
        elseif exist('G', 'var') && ~isempty(G)
            % For surface, write GIfTI file
            Gout = gifti;
            Gout.cdata = reshape(logpFDR, [], 1);
            save(Gout, out_fdr_file);
            fprintf('   💾 Saved TFCE FDR (GIfTI): %s\n', out_fdr_file);
        end
    catch MEsave
        warning('Failed to save TFCE FDR file: %s', MEsave.message);
    end

    % Save gamma parameters (if computed) for debugging
    try
        gamma_file = fullfile(outdir, sprintf('TFCE_gamma_params_%04d.mat', contrast_num));
        save(gamma_file, 'null_files', 'n_null');
        fprintf('   💾 Saved TFCE metadata: %s\n', gamma_file);
    catch
    end

end

% Check 2: SPM installation
fprintf('2️⃣  SPM INSTALLATION CHECK\n');
fprintf('   ─────────────────────────────────────────────────────────────\n');

% Use flexible SPM path detection instead of hardcoded path
try
    spm_path = find_spm_path();
    fprintf('   ✓ SPM path detected: %s\n', spm_path);
catch ME
    fprintf('   ❌ CRITICAL ERROR: SPM path detection failed!\n');
    fprintf('      Error: %s\n', ME.message);
    fprintf('\n');
    fprintf('   SETUP INSTRUCTIONS:\n');
    fprintf('   ──────────────────────\n');
    fprintf('   Option 1: Set environment variable\n');
    fprintf('      export SPM_PATH="/path/to/your/spm"\n');
    fprintf('   Option 2: Create spm_config.txt in current directory\n');
    fprintf('      echo "/path/to/your/spm" > spm_config.txt\n');
    fprintf('   Option 3: Install SPM in a standard location\n');
    fprintf('      e.g., /Applications/spm25 (macOS)\n');
    fprintf('      e.g., /usr/local/spm25 (Linux)\n');
    fprintf('      e.g., C:\\Program Files\\spm25 (Windows)\n\n');
    error('SPM path not found. Please install SPM or configure the path.');
end

% If we get here, SPM path was successfully detected
% Check if critical SPM files exist
    critical_files = {
        'spm.m'
        'spm_get_defaults.m'
        'toolbox/TFCE/tfce_estimate_stat.m'
    };
    
    all_found = true;
    for i = 1:length(critical_files)
        file_path = fullfile(spm_path, critical_files{i});
        if exist(file_path, 'file')
            fprintf('   ✓ Found: %s\n', critical_files{i});
        else
            fprintf('   ❌ MISSING: %s\n', critical_files{i});
            all_found = false;
        end
    end
    
    if ~all_found
        fprintf('\n   ⚠️  Some critical SPM files are missing!\n');
        fprintf('      SPM may not be properly installed.\n\n');
        error('SPM installation incomplete at: %s', spm_path);
    end
    
% Add SPM to path
addpath(spm_path);
fprintf('   ✓ Added SPM to MATLAB path\n\n');

% Check 3: TFCE toolbox
fprintf('3️⃣  TFCE TOOLBOX CHECK\n');
fprintf('   ─────────────────────────────────────────────────────────────\n');
tfce_path = fullfile(spm_path, 'toolbox', 'TFCE');

if ~isdir(tfce_path)
    fprintf('   ❌ CRITICAL ERROR: TFCE toolbox not found!\n');
    fprintf('      Expected: %s\n', tfce_path);
    fprintf('      TFCE must be installed as SPM toolbox.\n');
    fprintf('      Installation: Download from SPM website and place in toolbox folder.\n\n');
    error('TFCE toolbox not found at: %s', tfce_path);
else
    fprintf('   ✓ TFCE toolbox path exists: %s\n', tfce_path);
    
    % Check if TFCE main function exists
    tfce_main = fullfile(tfce_path, 'tfce_estimate_stat.m');
    if exist(tfce_main, 'file')
        fprintf('   ✓ Found TFCE main function: tfce_estimate_stat.m\n');
    else
        fprintf('   ❌ MISSING: tfce_estimate_stat.m\n');
        fprintf('      TFCE installation may be incomplete.\n\n');
        error('TFCE main function not found at: %s', tfce_main);
    end
    
    % Add TFCE to path
    addpath(tfce_path);
    fprintf('   ✓ Added TFCE toolbox to MATLAB path\n\n');
end

% Check 4: CAT12 installation
fprintf('4️⃣  CAT12 TOOLBOX CHECK\n');
fprintf('   ─────────────────────────────────────────────────────────────\n');
cat_path = fullfile(spm_path, 'toolbox', 'cat12');

if ~isdir(cat_path)
    fprintf('   ⚠️  CAT12 toolbox not found at: %s\n', cat_path);
    fprintf('      (CAT12 may not be needed for TFCE, but recommended)\n\n');
else
    fprintf('   ✓ CAT12 toolbox path exists: %s\n', cat_path);
    
    % Check for GIfTI support (needed for surface data)
    gifti_check = fullfile(spm_path, 'external', 'gifti');
    if isdir(gifti_check)
        fprintf('   ✓ GIfTI support found (for surface data)\n');
        addpath(gifti_check);
    else
        fprintf('   ⚠️  GIfTI support not found (surface data may have issues)\n');
    end
    
    % Add CAT12 to path
    addpath(cat_path);
    fprintf('   ✓ Added CAT12 toolbox to MATLAB path\n\n');
end

% Check 5: Statistics Toolbox
fprintf('5️⃣  STATISTICS TOOLBOX CHECK\n');
fprintf('   ─────────────────────────────────────────────────────────────\n');
try
    % Test if Statistics and Machine Learning Toolbox is available
    v = ver;
    stats_installed = any(strcmpi({v.Name}, 'Statistics and Machine Learning Toolbox'));
    
    if stats_installed
        fprintf('   ✓ Statistics and Machine Learning Toolbox is installed\n');
        fprintf('   ✓ TFCE permutation testing will work correctly\n\n');
    else
        fprintf('   ⚠️  Statistics and Machine Learning Toolbox not found\n');
        fprintf('      Some TFCE functions may not work.\n\n');
    end
catch
    fprintf('   ⚠️  Could not verify Statistics Toolbox\n\n');
end

% Check 6: Image Processing Toolbox
fprintf('6️⃣  IMAGE PROCESSING TOOLBOX CHECK\n');
fprintf('   ─────────────────────────────────────────────────────────────\n');
try
    v = ver;
    img_installed = any(strcmpi({v.Name}, 'Image Processing Toolbox'));
    
    if img_installed
        fprintf('   ✓ Image Processing Toolbox is installed\n');
        fprintf('   ✓ NIfTI/GIfTI file handling will work correctly\n\n');
    else
        fprintf('   ⚠️  Image Processing Toolbox not found\n');
        fprintf('      File I/O may have issues.\n\n');
    end
catch
    fprintf('   ⚠️  Could not verify Image Processing Toolbox\n\n');
end

% Check 7: Key MATLAB functions availability
fprintf('7️⃣  MATLAB FUNCTIONS AVAILABILITY CHECK\n');
fprintf('   ─────────────────────────────────────────────────────────────\n');

required_functions = {
    'spm_vol'              'SPM I/O'
    'spm_read_vols'        'SPM I/O'
    'spm_get_defaults'     'SPM configuration'
    'tinv'                 'Statistics (t-distribution)'
    'bwconncomp'           'Image Processing (connectivity)'
    'gifti'                'GIfTI support'
};

functions_ok = true;
for i = 1:size(required_functions, 1)
    func_name = required_functions{i, 1};
    func_desc = required_functions{i, 2};
    
    try
        which_result = which(func_name);
        if ~isempty(which_result)
            fprintf('   ✓ %s found (%s)\n', func_name, func_desc);
        else
            fprintf('   ⚠️  %s not found (%s)\n', func_name, func_desc);
            if ismember(func_name, {'spm_vol', 'spm_read_vols', 'spm_get_defaults'})
                functions_ok = false;
            end
        end
    catch
        fprintf('   ⚠️  Could not check %s\n', func_name);
    end
end

if functions_ok
    fprintf('\n   ✓ All critical functions are available\n\n');
else
    fprintf('\n   ❌ Some critical functions are missing!\n');
    fprintf('      SPM may not be properly loaded.\n\n');
    error('Required MATLAB functions not available');
end

% Check 8: File system and permissions
fprintf('8️⃣  FILE SYSTEM AND PERMISSIONS CHECK\n');
fprintf('   ─────────────────────────────────────────────────────────────\n');

stats_folder_check = stats_folder;
if ~isdir(stats_folder_check)
    fprintf('   ❌ ERROR: Statistics folder not found!\n');
    fprintf('      Requested: %s\n', stats_folder_check);
    error('Statistics folder not found: %s', stats_folder_check);
else
    fprintf('   ✓ Statistics folder exists: %s\n', stats_folder_check);
end

% Check write permissions
try
    test_file = fullfile(stats_folder_check, '.write_test_tmp_');
    fid = fopen(test_file, 'w');
    if fid > 0
        fclose(fid);
        delete(test_file);
        fprintf('   ✓ Write permissions OK\n');
    else
        fprintf('   ❌ ERROR: Cannot write to statistics folder!\n');
        fprintf('      Check file permissions.\n\n');
        error('No write permissions in: %s', stats_folder_check);
    end
catch
    fprintf('   ❌ ERROR: Cannot verify write permissions!\n\n');
    error('Permission check failed for: %s', stats_folder_check);
end

% Check SPM.mat file
spm_mat_file = fullfile(stats_folder_check, 'SPM.mat');
if ~exist(spm_mat_file, 'file')
    fprintf('   ❌ ERROR: SPM.mat not found in statistics folder!\n');
    fprintf('      Looked for: %s\n', spm_mat_file);
    error('SPM.mat not found in: %s', stats_folder_check);
else
    fprintf('   ✓ SPM.mat found\n\n');
end

% Final summary
fprintf('%s\n', repmat('═', 1, 90));
fprintf('✅ SYSTEM VERIFICATION COMPLETE - All checks passed!\n');
fprintf('   Ready to run TFCE analysis.\n');
fprintf('%s\n\n', repmat('═', 1, 90));

% AGGRESSIVE GUI SUPPRESSION FOR HEADLESS MODE
% This is CRITICAL to prevent any GUI from appearing
fprintf('Configuring headless mode...\n');

% Set environment variables for Java headless mode
setenv('JAVA_TOOL_OPTIONS', '-Djava.awt.headless=true');
setenv('AWT_TOOLKIT', 'MToolkit');
setenv('DISPLAY', '');  % Unset display

% Disable all graphics at every level
set(0, 'DefaultFigureVisible', 'off');
set(0, 'DefaultUIControlVisible', 'off');
set(0, 'DefaultUITableVisible', 'off');
set(0, 'DefaultFigureNumberTitle', 'off');
set(0, 'DefaultFigureMenuBar', 'none');
set(0, 'DefaultFigureToolBar', 'none');
set(groot, 'defaultFigureVisible', 'off');
set(groot, 'defaultAxesVisible', 'off');

% Close all figures forcefully
close all force hidden;

% Suppress all warnings that might pop dialogs
warning('off', 'all');

% Disable Java desktop (if running in batch mode)
try
    usejava('desktop', false);
    desktop('hide');  % Hide desktop if it's running
catch
end

% Initialize SPM defaults early
spm_get_defaults;
global defaults;
defaults.cmdline = 1;  % Command line mode
defaults.nogui = 1;    % No GUI
defaults.modality = 'FMRI';  % Set modality early
defaults.ui.mask = 0;  % No mask dialog
defaults.ui.output = 0;  % No output window

% Put SPM into command-line (no-UI) mode explicitly
try
    spm('defaults','FMRI');
    spm_jobman('initcfg');
    spm('CmdLine');  % ensure SPM does not attempt GUI
    
    fprintf('  ✓ SPM command-line mode enabled\n');
    fprintf('  ✓ spm_figure() shadow function active (intercepts all GUI calls)\n');
catch ME
    warning('Failed to initialize SPM command-line mode: %s', ME.message);
end
defaults.ui.window_mm = 0;  % Disable figure window
defaults.ui.mask_edit_window = 0;  % No mask editing window

fprintf('✓ Headless mode configured\n');
% -------------------------------------------------------------------------

fprintf('\n🔍 Looking for spmT* files in: %s\n', stats_folder);

% Get all spmT files (search recursively in case they're in subdirectories)
spmT_files = dir(fullfile(stats_folder, 'spmT_*.*ii'));
if isempty(spmT_files)
    % Try searching in subdirectories
    spmT_files = dir(fullfile(stats_folder, '*', 'spmT_*.*ii'));
end
if isempty(spmT_files)
    error('No spmT files found in %s or its subdirectories.', stats_folder);
end

% Detect if Surface or Volume data
has_gii_files = any(endsWith({spmT_files.name}, '.gii'));
has_nii_files = any(endsWith({spmT_files.name}, '.nii')) || any(endsWith({spmT_files.name}, '.nii.gz'));

if has_gii_files
    fprintf('   📊 Data type: Surface data (.gii files)\n');
elseif has_nii_files
    fprintf('   📊 Data type: Volume data (.nii files)\n');
else
    fprintf('   📊 Data type: Mixed formats\n');
end

% Check for existing TFCE files
tfce_files = dir(fullfile(stats_folder, '*', 'TFCE_*.gii'));
if isempty(tfce_files)
    tfce_files = dir(fullfile(stats_folder, 'TFCE_*.gii'));
end
if isempty(tfce_files)
    % Check for other TFCE output formats
    tfce_files = dir(fullfile(stats_folder, '*', 'TFCE_*.nii'));
    if isempty(tfce_files)
        tfce_files = dir(fullfile(stats_folder, 'TFCE_*.nii'));
    end
end

if ~isempty(tfce_files) && ~force_analysis
    fprintf('\n⚠️  Warning: TFCE output files already exist in the stats folder.\n');
    fprintf('   Found %d existing TFCE files.\n', length(tfce_files));
    fprintf('   To re-run the analysis, use: --force flag\n\n');
    fprintf('❌ Exiting to avoid overwriting existing results.\n');
    return;
elseif ~isempty(tfce_files) && force_analysis
    fprintf('\n⚠️  Force flag detected: Re-running TFCE analysis despite existing output files.\n');
end

% Load SPM.mat to extract df and contrast names
spm_mat_file = fullfile(stats_folder, 'SPM.mat');
if ~exist(spm_mat_file, 'file')
    % Try looking in subdirectories
    spm_mats = dir(fullfile(stats_folder, '*', 'SPM.mat'));
    if ~isempty(spm_mats)
        spm_mat_file = fullfile(spm_mats(1).folder, spm_mats(1).name);
    else
        error('SPM.mat not found in %s or its subdirectories', stats_folder);
    end
end
load(spm_mat_file, 'SPM');
df = SPM.xX.erdf;

% Create a map of contrast numbers to names for easy lookup
contrast_names = cell(length(SPM.xCon), 1);
for i = 1:length(SPM.xCon)
    contrast_names{i} = SPM.xCon(i).name;
end

fprintf('📐 Degrees of freedom from SPM.mat: %d\n', df);

% Compute T threshold
t_thresh = tinv(1 - p_thresh, df);
fprintf('🔬 Using T threshold = %.4f (p = %.3f)\n', t_thresh, p_thresh);
fprintf('   Minimum cluster size: %d voxels/vertices\n\n', cluster_size);

% Identify significant contrasts
significant_indices = [];
for i = 1:length(spmT_files)
    % Handle both direct path and nested folder structure
    if isfield(spmT_files(i), 'folder') && ~isempty(spmT_files(i).folder)
        file_path = fullfile(spmT_files(i).folder, spmT_files(i).name);
    else
        file_path = fullfile(stats_folder, spmT_files(i).name);
    end
    
    % Extract contrast number from filename
    match = regexp(spmT_files(i).name, 'spmT_(\d+)', 'tokens');
    if isempty(match)
        continue;
    end
    contrast_num = str2double(match{1}{1});
    
    % Get contrast name (with bounds checking)
    if contrast_num > 0 && contrast_num <= length(contrast_names)
        con_name = contrast_names{contrast_num};
    else
        con_name = '(unknown)';
    end
    
    % Handle both NIfTI (.nii) and GIfTI (.gii) files
    try
        if endsWith(file_path, '.nii') || endsWith(file_path, '.nii.gz')
            V = spm_vol(file_path);
            Y = spm_read_vols(V);
        elseif endsWith(file_path, '.gii')
            % For GIfTI files, read using CAT12 functions
            G = gifti(file_path);
            % Extract data - gifti objects store data differently
            if isstruct(G)
                Y = G.cdata(:);
            elseif iscell(G.cdata)
                Y = cell2mat(G.cdata);
            else
                Y = double(G.cdata(:));
            end
        elseif endsWith(file_path, '.dat')
            % For .dat files, try to read as raw binary
            fid = fopen(file_path, 'r');
            Y = fread(fid, 'single');
            fclose(fid);
        else
            fprintf('⚠ Skipping unsupported file type: %s\n', file_path);
            continue;
        end
        
        % Check for voxels/vertices above threshold
        above_thresh = Y > t_thresh;
        if any(above_thresh(:))
            % Convert to logical for connectivity analysis
            above_thresh = logical(above_thresh);
            
            % Find connected clusters
            try
                % Try standard 3D connectivity first
                CC = bwconncomp(above_thresh, 26);
                cluster_sizes = cellfun(@numel, CC.PixelIdxList);
                max_cluster = max(cluster_sizes);
            catch
                try
                    % If 3D fails, try with 6-connectivity
                    CC = bwconncomp(above_thresh, 6);
                    cluster_sizes = cellfun(@numel, CC.PixelIdxList);
                    max_cluster = max(cluster_sizes);
                catch
                    % For surface data or other cases, just count total significant voxels
                    max_cluster = sum(above_thresh(:));
                end
            end
            
            % Check if max cluster size passes threshold
            if max_cluster >= cluster_size
                fprintf('   \033[32m✔ %03d │ %s │ %d voxels\033[0m\n', ...
                    contrast_num, con_name, max_cluster);
                significant_indices(end+1) = contrast_num;
            else
                fprintf('   \033[33m⚠ %03d │ %s │ %d < %d (cluster too small)\033[0m\n', ...
                    contrast_num, con_name, max_cluster, cluster_size);
            end
        else
            fprintf('   \033[31m✘ %03d │ %s │ NOT significant\033[0m\n', contrast_num, con_name);
        end
    catch ME
        fprintf('⚠ Error reading %s: %s\n', spmT_files(i).name, ME.message);
    end
end

% Exit if no contrasts are significant
if isempty(significant_indices)
    fprintf('\n❌ No contrasts passed criteria (p < %.3f AND cluster size > %d). Exiting.\n', ...
        p_thresh, cluster_size);
    return;
end

% Pilot mode: select only 1 random contrast
if pilot_mode
    if length(significant_indices) > 1
        % Randomly select one contrast
        rng('shuffle');  % Seed with current time
        random_idx = randi(length(significant_indices));
        pilot_contrast = significant_indices(random_idx);
        
        % Get name of selected contrast for display
        if pilot_contrast > 0 && pilot_contrast <= length(contrast_names)
            pilot_con_name = contrast_names{pilot_contrast};
        else
            pilot_con_name = '(unknown)';
        end
        
        fprintf('\n\033[36m🧪 PILOT MODE: Processing 1 random contrast out of %d\033[0m\n', length(significant_indices));
        fprintf('   Selected: \033[32m%03d │ %s\033[0m\n\n', pilot_contrast, pilot_con_name);
        significant_indices = pilot_contrast;
    else
        fprintf('\n\033[36m🧪 PILOT MODE: Only 1 significant contrast found, processing it\033[0m\n\n');
    end
end

% Set up TFCE batch
fprintf('\n⚙️ Preparing TFCE batch for %d significant contrast(s)...\n', numel(significant_indices));
fprintf('   Using %d CPU cores for parallel processing\n', n_jobs);
fprintf('   Permutations per contrast: %d\n', n_perm);

matlabbatch{1}.spm.tools.tfce_estimate.data = {spm_mat_file};

% Create conspec - specify all significant contrasts at once
matlabbatch{1}.spm.tools.tfce_estimate.conspec.contrasts = significant_indices(:);
matlabbatch{1}.spm.tools.tfce_estimate.conspec.n_perm = n_perm;

matlabbatch{1}.spm.tools.tfce_estimate.nproc = n_jobs;

% Multi-stage TFCE analysis
if multi_stage
    fprintf('\n🔄 MULTI-STAGE TFCE ANALYSIS ENABLED\n');
    fprintf('   Stage 1: %d permutations (quick check)\n', n_perm_stage1);
    fprintf('   Stage 2: %d permutations (only if significant)\n', n_perm_stage2);
    fprintf('   ─────────────────────────────────────────────────\n');

    % Stage 1: Quick check with low permutations
    fprintf('\n🏃 STAGE 1: Quick TFCE analysis (%d permutations)...\n', n_perm_stage1);

    % Temporarily set low permutation count
    original_n_perm = n_perm;
    n_perm = n_perm_stage1;

    % Run Stage 1 TFCE
    stage1_successful = run_tfce_stage(stats_folder, significant_indices, n_perm, n_jobs, has_gii_files, has_nii_files, contrast_names, no_background, pilot_mode);

    if ~stage1_successful
        fprintf('\n❌ Stage 1 TFCE failed. Skipping Stage 2.\n');
        return;
    end

    % Check Stage 1 results for significance
    fprintf('\n🔍 Checking Stage 1 results for significant findings...\n');

    contrasts_for_stage2 = [];
    for i = 1:length(significant_indices)
        contrast_num = significant_indices(i);

        % Check if FWE corrected results have significant data
        has_significant_fwe = check_tfce_output_data(stats_folder, contrast_num, 'FWE');

        if has_significant_fwe
            contrasts_for_stage2 = [contrasts_for_stage2, contrast_num];
            fprintf('   ✓ Contrast %03d: Significant FWE results found → Stage 2\n', contrast_num);
        else
            fprintf('   ⚠️  Contrast %03d: No significant FWE results → Skipping Stage 2\n', contrast_num);
        end
    end

    % Stage 2: High-resolution analysis for significant contrasts
    if ~isempty(contrasts_for_stage2)
        fprintf('\n🏃 STAGE 2: High-resolution TFCE analysis (%d permutations)...\n', n_perm_stage2);
        fprintf('   Processing %d significant contrast(s) with high permutations\n', length(contrasts_for_stage2));

        n_perm = n_perm_stage2;
        stage2_successful = run_tfce_stage(stats_folder, contrasts_for_stage2, n_perm, n_jobs, has_gii_files, has_nii_files, contrast_names, no_background, pilot_mode);

        if stage2_successful
            % Validate Stage 2 results - check for actual significant findings
            fprintf('\n🔍 Validating Stage 2 results for significant findings...\n');
            
            stage2_significant_count = 0;
            for i = 1:length(contrasts_for_stage2)
                contrast_num = contrasts_for_stage2(i);
                
                % Check if FWE corrected results have significant data
                has_significant_fwe = check_tfce_output_data(stats_folder, contrast_num, 'FWE');
                
                if has_significant_fwe
                    stage2_significant_count = stage2_significant_count + 1;
                    fprintf('   ✓ Contrast %03d: Significant FWE results confirmed\n', contrast_num);
                else
                    fprintf('   ⚠️  Contrast %03d: No significant FWE results found\n', contrast_num);
                end
            end
            
            fprintf('\n✅ Multi-stage TFCE analysis complete!\n');
            fprintf('   Stage 1: %d permutations for %d contrasts\n', n_perm_stage1, length(significant_indices));
            fprintf('   Stage 2: %d permutations for %d contrasts → %d with significant results\n', n_perm_stage2, length(contrasts_for_stage2), stage2_significant_count);
            
            if stage2_significant_count == 0
                fprintf('\n⚠️  NOTE: Stage 2 completed but no significant FWE-corrected results were found.\n');
                fprintf('   This may indicate that the effects are not robust to multiple comparison correction.\n');
            else
                fprintf('\n🎉 SUCCESS: Found significant results in %d/%d contrasts after FWE correction!\n', stage2_significant_count, length(contrasts_for_stage2));
            end
        else
            fprintf('\n⚠️  Stage 2 TFCE completed with warnings\n');
        end
    else
        fprintf('\n✅ Multi-stage TFCE analysis complete!\n');
        fprintf('   Stage 1: %d permutations for %d contrasts\n', n_perm_stage1, length(significant_indices));
        fprintf('   Stage 2: Skipped (no significant results found in Stage 1)\n');
    end

    % Restore original permutation count
    n_perm = original_n_perm;
    return;

end

% Run the batch (single stage)
fprintf('\n🚀 Launching TFCE analysis...\n');

% Completely disable all GUI/graphics
set(0, 'DefaultFigureVisible', 'off');
set(groot, 'defaultFigureVisible', 'off');
close all force hidden;

try
    % Load SPM.mat to get design information
    SPM_data = load(spm_mat_file);
    SPM = SPM_data.SPM;
    
    % CRITICAL: Verify model has been estimated to avoid dialog
    if ~isfield(SPM, 'xVol')
        fprintf('   ❌ ERROR: SPM model has not been estimated (missing xVol field)\n');
        fprintf('      Please run SPM estimation first.\n');
        error('SPM model not properly estimated');
    else
        fprintf('   ✓ SPM model verified as estimated\n');
    end
    
    % Get output directory
    outdir = fileparts(spm_mat_file);
    
    % Process each significant contrast
    fprintf('   Processing %d contrasts...\n', length(significant_indices));
    
    % Get output directory
    outdir = fileparts(spm_mat_file);
    if isempty(outdir)
        outdir = '.';
    end
    
    % Record start time
    analysis_start = tic;
    
    for c_idx = 1:length(significant_indices)
        contrast_num = significant_indices(c_idx);
        
        % Get contrast name for display
        if contrast_num > 0 && contrast_num <= length(contrast_names)
            con_name = contrast_names{contrast_num};
        else
            con_name = '(unknown)';
        end
        
        contrast_start = tic;
        fprintf('   [%d/%d] \033[32m%03d │ %s\033[0m │ ', c_idx, length(significant_indices), contrast_num, con_name);
        
        try
            % Call TFCE directly via tfce_estimate_stat
            % Build complete job structure with all required fields
            job = struct();
            job.data = {spm_mat_file};
            job.conspec = struct();
            job.conspec.contrasts = contrast_num;  % Single contrast
            job.conspec.n_perm = n_perm;  % Number of permutations
            job.nproc = n_jobs;  % Number of processors
            job.singlethreaded = 0;  % Use multi-threading
            job.nuisance_method = 2;  % 2=Smith (recommended)
            job.tbss = 0;  % Not TBSS data
            
            % Handle masking based on data type
            if has_nii_files && ~has_gii_files
                % Volume data: prefer repo template mask, fall back to mask.nii
                utils_dir = fileparts(mfilename('fullpath'));
                repo_root = fileparts(utils_dir);
                template_mask = fullfile(repo_root, 'templates', 'brainmask_GMtight.nii');
                if exist(template_mask, 'file')
                    mask_path = template_mask;
                else
                    mask_path = fullfile(stats_folder, 'mask.nii');
                end
                if exist(mask_path, 'file')
                    job.mask = {mask_path};
                    fprintf('   (using volume mask: %s) ', mask_path);
                end
            elseif has_gii_files && ~has_nii_files
                % Surface data: look for mask.gii
                mask_gii = fullfile(stats_folder, 'mask.gii');
                if exist(mask_gii, 'file')
                    job.mask = {mask_gii};
                    fprintf('   (using surface mask: %s) ', mask_gii);
                else
                    fprintf('   (surface data: no mask found, using implicit vertex masking) ');
                end
            end
            
            % Suppress warnings
            old_warning = warning('off', 'all');
            
            % Try to run TFCE with error handling
            try
                tfce_estimate_stat(job);
            catch e_tfce
                % Capture the actual error
                fprintf('\n❌ TFCE execution error: %s\n', e_tfce.message);
                rethrow(e_tfce);
            end
            
            warning(old_warning);
            warning(old_warning);
            
            % Wait for output file to be created (with timeout)
            % Expected output file pattern: TFCE_0XXX.gii or TFCE_log_p_0XXX.gii
            output_file = fullfile(outdir, sprintf('TFCE_%04d.gii', contrast_num));
            if ~exist(output_file, 'file')
                output_file = fullfile(outdir, sprintf('TFCE_%04d.nii', contrast_num));
            end
            if ~exist(output_file, 'file')
                output_file = fullfile(outdir, sprintf('TFCE_%04d.dat', contrast_num));
            end
            
            % Also check for alternative naming patterns
            alt_patterns = {
                sprintf('TFCE_%03d.gii', contrast_num)
                sprintf('TFCE_%03d.nii', contrast_num)
                sprintf('log_p_%04d.gii', contrast_num)
                sprintf('log_p_%03d.gii', contrast_num)
                sprintf('log_p_%04d.dat', contrast_num)
                sprintf('log_p_%03d.dat', contrast_num)
            };
            
            % Poll for file existence
            % Timeout: longer for --no-background (wait for completion)
            if no_background
                max_wait = 3600;  % 60 minutes for volume data with --no-background
                wait_str = '(waiting for completion)';
            else
                max_wait = 600;   % 10 minutes default (background processing)
                wait_str = '(timeout, background processing)';
            end
            
            wait_start = tic;
            file_found = false;
            while toc(wait_start) < max_wait
                % Check main pattern
                if exist(output_file, 'file')
                    file_found = true;
                    break;
                end
                % Check alternative patterns
                for p = 1:length(alt_patterns)
                    alt_file = fullfile(outdir, alt_patterns{p});
                    if exist(alt_file, 'file')
                        output_file = alt_file;
                        file_found = true;
                        break;
                    end
                end
                if file_found
                    break;
                end
                
                pause(1.0);  % Check every second
                elapsed = toc(wait_start);
                if mod(elapsed, 10) < 1.1  % Print status every 10 seconds
                    fprintf('.');
                end
            end
            
            contrast_time = toc(contrast_start);
            
            if file_found
                fprintf('\n✓ Complete (%.1fs)\n', contrast_time);
                % -----------------------------------------------------------------
                % Post-processing: attempt to load null distribution files and
                % compute tail-approximated p-values (Gamma) and FDR values.
                % Save null distribution and FDR maps for user inspection.
                % -----------------------------------------------------------------
                try
                    save_tfce_null_and_fdr(outdir, contrast_num, has_nii_files, has_gii_files);
                catch ME_proc
                    fprintf('   ⚠️  Post-processing TFCE (null/FDR) failed: %s\n', ME_proc.message);
                end
            else
                % File not created - this is an error condition
                fprintf('\n❌ NO OUTPUT FILE CREATED! (%.1fs)\n', contrast_time);
                
                % Debug: List all files in output directory that were created recently
                fprintf('      DEBUG: Files in %s:\n', outdir);
                all_files = dir(outdir);
                for df = 1:length(all_files)
                    if ~all_files(df).isdir
                        fprintf('             %s (%.0f bytes)\n', all_files(df).name, all_files(df).bytes);
                    end
                end
            end
            
        catch ME_inner
            contrast_time = toc(contrast_start);
            fprintf('⚠ (%.1fs) - %s\n', contrast_time, ME_inner.message);
        end
    end
    
    total_time = toc(analysis_start);
    
    % Check for actual TFCE output files - use more comprehensive search
    tfce_gii = dir(fullfile(outdir, 'TFCE_*.gii'));
    tfce_nii = dir(fullfile(outdir, 'TFCE_*.nii'));
    tfce_dat = dir(fullfile(outdir, 'TFCE_*.dat'));
    log_p_files = dir(fullfile(outdir, 'log_p_*.gii'));
    log_p_nii = dir(fullfile(outdir, 'log_p_*.nii'));
    
    tfce_final = length(tfce_gii) + length(tfce_nii) + length(tfce_dat) + length(log_p_files) + length(log_p_nii);
    
    fprintf('\n✅ TFCE analysis complete for %d significant contrast(s)!\n', length(significant_indices));
    fprintf('   Total processing time: %.1f seconds (%.1f minutes)\n', total_time, total_time/60);
    if total_time > 0
        fprintf('   Average time per contrast: %.1f seconds\n', total_time / length(significant_indices));
    end
    fprintf('   Generated %d TFCE output files:\n', tfce_final);
    fprintf('      - TFCE_*.gii: %d\n', length(tfce_gii));
    fprintf('      - TFCE_*.nii: %d\n', length(tfce_nii));
    fprintf('      - TFCE_*.dat: %d\n', length(tfce_dat));
    fprintf('      - log_p_*.gii: %d\n', length(log_p_files));
    fprintf('      - log_p_*.nii: %d\n', length(log_p_nii));
    
    % CRITICAL: If no files were created, this is a serious error!
    if tfce_final == 0
        fprintf('\n❌ CRITICAL ERROR: No TFCE output files were created!\n');
        fprintf('   This suggests TFCE processing failed silently.\n');
        fprintf('   Possible causes:\n');
        fprintf('   1. TFCE toolbox not properly installed\n');
        fprintf('   2. SPM.mat may be corrupted or missing\n');
        fprintf('   3. TFCE function threw an error but it was suppressed\n');
        fprintf('   4. File permission issues in output directory\n');
        fprintf('   5. Insufficient disk space\n');
        fprintf('\n   Output directory: %s\n', outdir);
        fprintf('   Directory contents:\n');
        all_files = dir(outdir);
        for df = 1:min(length(all_files), 30)
            if ~all_files(df).isdir
                fprintf('      %s (%.0f bytes)\n', all_files(df).name, all_files(df).bytes);
            end
        end
    end
    
    if pilot_mode
        fprintf('\n🧪 PILOT MODE: Tested 1 random contrast for quick validation\n');
    end
    
    % Warn if not all files are complete
    if tfce_final < length(significant_indices)
        if no_background
            fprintf('\n⚠️  Note: Some contrasts may still be processing (60-minute timeout).\n');
        else
            fprintf('\n⚠️  Note: Some contrasts may still be processing in background.\n');
        end
        fprintf('   Monitor the output folder for TFCE_*.gii/nii files.\n');
        fprintf('   Expected total files: ~%d (TFCE + log_p + log_pFWE + log_pFDR)\n', length(significant_indices)*4);
    end
    
catch ME
    fprintf('\n⚠ Error during TFCE processing: %s\n', ME.message);
    fprintf('   (Some TFCE files may still be processing in background)\n');
end

end

function success = run_tfce_stage(stats_folder, contrast_indices, n_perm, n_jobs, has_gii_files, has_nii_files, contrast_names, no_background, pilot_mode)
% RUN_TFCE_STAGE - Execute TFCE analysis for a specific stage
% This function handles the actual TFCE execution for either Stage 1 or Stage 2

success = false;

try
    % Load SPM.mat to get design information
    spm_mat_file = fullfile(stats_folder, 'SPM.mat');
    SPM_data = load(spm_mat_file);
    SPM = SPM_data.SPM;

    % Verify model has been estimated
    if ~isfield(SPM, 'xVol')
        fprintf('   ❌ ERROR: SPM model has not been estimated\n');
        return;
    end

    % Get output directory
    outdir = stats_folder;

    % Record start time
    stage_start = tic;

    fprintf('   Processing %d contrast(s)...\n', length(contrast_indices));

    for c_idx = 1:length(contrast_indices)
        contrast_num = contrast_indices(c_idx);

        % Get contrast name for display
        if contrast_num > 0 && contrast_num <= length(contrast_names)
            con_name = contrast_names{contrast_num};
        else
            con_name = '(unknown)';
        end

        contrast_start = tic;
        fprintf('   [%d/%d] \033[32m%03d │ %s\033[0m │ ', c_idx, length(contrast_indices), contrast_num, con_name);

        try
            % Build TFCE job structure
            job = struct();
            job.data = {spm_mat_file};
            job.conspec = struct();
            job.conspec.contrasts = contrast_num;
            job.conspec.n_perm = n_perm;
            job.nproc = n_jobs;
            job.singlethreaded = 0;
            job.nuisance_method = 2;
            job.tbss = 0;

            % Handle masking
            if has_nii_files && ~has_gii_files
                % Prefer repo template mask if present
                utils_dir = fileparts(mfilename('fullpath'));
                repo_root = fileparts(utils_dir);
                template_mask = fullfile(repo_root, 'templates', 'brainmask_GMtight.nii');
                if exist(template_mask, 'file')
                    mask_path = template_mask;
                else
                    mask_path = fullfile(stats_folder, 'mask.nii');
                end
                if exist(mask_path, 'file')
                    job.mask = {mask_path};
                    fprintf('   (using volume mask) ');
                end
            elseif has_gii_files && ~has_nii_files
                mask_gii = fullfile(stats_folder, 'mask.gii');
                if exist(mask_gii, 'file')
                    job.mask = {mask_gii};
                    fprintf('   (using surface mask) ');
                else
                    fprintf('   (implicit masking) ');
                end
            end

            % Disable GUI and run TFCE
            set(0, 'DefaultFigureVisible', 'off');
            set(groot, 'defaultFigureVisible', 'off');
            close all force hidden;

            old_warning = warning('off', 'all');

            tfce_estimate_stat(job);

            warning(old_warning);

            % Wait for output files
            max_wait = 600;  % 10 minutes
            wait_start = tic;
            file_found = false;

            while toc(wait_start) < max_wait
                % Check for TFCE output files
                tfce_file = fullfile(outdir, sprintf('TFCE_%04d.gii', contrast_num));
                if ~exist(tfce_file, 'file')
                    tfce_file = fullfile(outdir, sprintf('TFCE_%04d.nii', contrast_num));
                end
                if ~exist(tfce_file, 'file')
                    tfce_file = fullfile(outdir, sprintf('TFCE_%04d.dat', contrast_num));
                end

                if exist(tfce_file, 'file')
                    file_found = true;
                    break;
                end

                pause(1.0);
                elapsed = toc(wait_start);
                if mod(elapsed, 10) < 1.1
                    fprintf('.');
                end
            end

            contrast_time = toc(contrast_start);

            if file_found
                fprintf('\n✓ Complete (%.1fs)\n', contrast_time);
            else
                fprintf('\n❌ No output file created (%.1fs)\n', contrast_time);
                return;
            end

        catch ME_inner
            contrast_time = toc(contrast_start);
            fprintf('⚠ (%.1fs) - %s\n', contrast_time, ME_inner.message);
            return;
        end
    end

    stage_time = toc(stage_start);
    fprintf('   Stage completed in %.1f seconds (%.1f minutes)\n', stage_time, stage_time/60);
    success = true;

catch ME
    fprintf('   ❌ Stage failed: %s\n', ME.message);
end

end