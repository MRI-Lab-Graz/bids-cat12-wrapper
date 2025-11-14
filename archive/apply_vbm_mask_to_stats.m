function masked_files = apply_vbm_mask_to_stats(stats_folder, mask_file)
% APPLY_VBM_MASK_TO_STATS  Create masked copies of spmT/spmF maps
%
% masked_files = apply_vbm_mask_to_stats(stats_folder, mask_file)
%
% This utility will locate `spmT_*.nii` and `spmF_*.nii` files in
% `stats_folder` (recursively), apply the provided binary mask and
% write masked copies named `spmT_XXXX_masked.nii` next to the originals.
% If `mask_file` is omitted, the function will prefer the canonical
% repo template mask at `stats/templates/brainmask_GMtight.nii`. The
% original files are left untouched. The function returns a cell array
% of written filenames.
%
% Example:
%   apply_vbm_mask_to_stats('/path/to/stats')  % uses repo template mask
%   apply_vbm_mask_to_stats('/path/to/stats', '/path/to/custom_mask.nii')

if nargin < 1 || isempty(stats_folder)
    stats_folder = pwd;
end
if nargin < 2 || isempty(mask_file)
    % Prefer the repo-level template mask over any per-results mask.
    utils_dir = fileparts(mfilename('fullpath'));
    repo_root = fileparts(utils_dir);
    template_mask = fullfile(repo_root, 'templates', 'brainmask_GMtight.nii');
    if exist(template_mask, 'file')
        mask_file = template_mask;
    else
        error('Mask file not provided and repo template mask not found: %s', template_mask);
    end
end

if ~exist(mask_file, 'file')
    error('Mask file not found: %s', mask_file);
end

fprintf('Applying mask: %s\n', mask_file);

% Use spm to read mask
try
    Vm = spm_vol(mask_file);
    M = spm_read_vols(Vm) > 0;
catch ME
    error('Failed to read mask via SPM: %s', ME.message);
end

% Find spmT and spmF files under stats_folder
files = spm_select('FPListRec', stats_folder, '^spm(T|F)_\d{4}\.nii(\.gz)?$');
masked_files = {};

for i = 1:size(files,1)
    fn = strtrim(files(i,:));
    [p,n,e] = fileparts(fn);
    outname = fullfile(p, [n, '_masked', e]);

    try
        V = spm_vol(fn);
        Y = spm_read_vols(V);

        % Check dimension compatibility
        if ~isequal(size(Y), size(M))
            warning('Mask and image size differ for %s --> skipping', fn);
            continue;
        end

        % Apply mask (set outside mask to 0)
        Y(~M) = 0;

        % Prepare output header (preserve original dtype/affine)
        Vout = V;
        Vout.fname = outname;
        Vout = rmfield_if_exists(Vout, 'pinfo');
        spm_write_vol(Vout, Y);

        masked_files{end+1,1} = outname; %#ok<AGROW>
        fprintf('Wrote masked file: %s\n', outname);
    catch ME
        warning('Failed to mask %s: %s', fn, ME.message);
    end
end

if isempty(masked_files)
    fprintf('No files were masked. Check patterns and mask compatibility.\n');
else
    fprintf('Masking complete: %d files written.\n', numel(masked_files));
end

end

function S = rmfield_if_exists(S, fname)
if isfield(S, fname)
    S = rmfield(S, fname);
end
end
