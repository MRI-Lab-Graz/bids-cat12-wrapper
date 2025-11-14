function mask_file = generate_vbm_mask(cat12_dir, smoothing, output_dir, modality)
% GENERATE_VBM_MASK - Create VBM-specific mask from random smoothed subject
%
% For VBM analysis, we want to use a mask based on actual gray matter
% distribution rather than the whole-brain mask. This function:
% 1. Finds all smoothed smoothed images
% 2. Selects one at random
% 3. Creates a binary mask for values > 0.1
% 4. Saves as mask_vbm.nii
%
% Usage:
%   mask_file = generate_vbm_mask('/path/to/cat12', 6, '/path/to/results', 'vbm')

if ~exist('smoothing','var') || isempty(smoothing)
    smoothing = 6;  % Default smoothing
end

if ~exist('modality','var') || isempty(modality)
    modality = 'vbm';
end

fprintf('\n%s\n', repmat('═', 1, 80));
fprintf('Generating VBM-Specific Mask\n');
fprintf('%s\n\n', repmat('═', 1, 80));

% DEPRECATION/CHANGE: The pipeline now prefers the canonical repo-level
% mask at stats/templates/brainmask_GMtight.nii. If that file exists we copy
% it into the output directory and return that path instead of creating a
% per-results 'mask_vbm.nii'. This helps keep masking consistent across
% analyses and avoids generating multiple divergent masks.
utils_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(utils_dir);
template_mask = fullfile(repo_root, 'templates', 'brainmask_GMtight.nii');
if exist(template_mask, 'file')
    fprintf('NOTICE: Found repo template mask: %s\n', template_mask);
    dest = fullfile(output_dir, 'brainmask_GMtight.nii');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    try
        copyfile(template_mask, dest);
        fprintf('Copied template mask to output dir: %s\n', dest);
        mask_file = dest;
        return;
    catch ME
        warning('Failed to copy template mask (%s): %s\nFalling back to generation.', template_mask, ME.message);
    end
end

% Create output directory if needed
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Find smoothed images based on modality
fprintf('Searching for smoothed %s images in:\n%s\n\n', modality, cat12_dir);

if strcmp(modality, 'vbm')
    % For VBM, look for s<N>mwp1r*.nii files in mri/
    search_pattern = fullfile(cat12_dir, 'data', 'cat12', '**', 'mri', ...
                             sprintf('s%dmwp1r*.nii', smoothing));
    
    files = dir(search_pattern);
    
    if isempty(files)
        % Try alternative pattern without 'r'
        search_pattern = fullfile(cat12_dir, 'data', 'cat12', '**', 'mri', ...
                                 sprintf('s%dmwp1*.nii', smoothing));
        files = dir(search_pattern);
    end
else
    error('Mask generation only supported for VBM modality currently');
end

if isempty(files)
    error('No smoothed %s images found with smoothing=%dmm in %s', ...
          modality, smoothing, cat12_dir);
end

fprintf('✓ Found %d smoothed %s images\n\n', length(files), modality);

% Select random image
rand_idx = randi(length(files));
selected_file = fullfile(files(rand_idx).folder, files(rand_idx).name);

fprintf('Selected random image (index %d of %d):\n%s\n\n', ...
    rand_idx, length(files), selected_file);

% Load the image
try
    V = spm_vol(selected_file);
    img = spm_read_vols(V);
catch
    error('Failed to read image: %s', selected_file);
end

% Use ASCII 'x' for dimension display to avoid non-ASCII characters
fprintf('Image dimensions: %d x %d x %d\n', size(img, 1), size(img, 2), size(img, 3));
fprintf('Value range: [%.4f, %.4f]\n', min(img(:)), max(img(:)));

% Create binary mask: values > 0.1
mask = img > 0.1;

fprintf('\nCreated mask with values > 0.1\n');
fprintf('Voxels in mask: %d / %d (%.2f%%)\n\n', ...
    sum(mask(:)), numel(mask), 100*sum(mask(:))/numel(mask));

% Save mask as NIfTI
mask_file = fullfile(output_dir, 'mask_vbm.nii');

% Prepare volume header for writing. Use the same data type as the
% original image to reduce the chance of platform-specific write issues.
V_mask = V;
V_mask.fname = mask_file;
% Use original data type and pinfo; cast mask to double for writing which
% is generally safe across SPM builds.
V_mask.dt = V.dt;
V_mask.pinfo = [1 0 0]';

try
    % Write as double to avoid unexpected dtype edge-cases in spm_write_vol
    spm_write_vol(V_mask, double(mask));
catch ME
    % If writing fails, attempt a safe fallback: save mask as MATLAB .mat
    % so the user can convert it later. Also rethrow a warning to the
    % console so the calling script can handle it.
    warning('spm_write_vol failed: %s\nFalling back to saving mask as MAT file: %s.mat', ME.message, mask_file);
    try
        save([mask_file '.mat'], 'mask', '-v7.3');
        fprintf('Saved fallback MAT mask to %s.mat\n', mask_file);
    catch ME2
        warning('Failed to save fallback MAT mask: %s', ME2.message);
    end
end

fprintf('✓ Mask saved to:\n%s\n\n', mask_file);

% Print statistics
fprintf('Mask Statistics:\n');
fprintf('  Mask file: %s\n', mask_file);
fprintf('  Voxels included: %d\n', sum(mask(:)));
fprintf('  Voxels excluded: %d\n', sum(~mask(:)));
fprintf('  Threshold: > 0.1\n\n');

end
