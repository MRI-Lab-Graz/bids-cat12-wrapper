function out_mask = replace_vbm_mask(stats_folder, src_mask, varargin)
% REPLACE_VBM_MASK  Safely replace the per-results mask with a provided mask
%
% Usage:
%   out_mask = replace_vbm_mask(stats_folder, src_mask)
%   out_mask = replace_vbm_mask(stats_folder, src_mask, 'threshold', 0.1)
%
% This helper will:
%  - back up existing brainmask_GMtight.nii -> brainmask_GMtight.nii.bak.TIMESTAMP
%  - load src_mask (NIfTI or GIfTI converted to volume) and optionally
%    threshold it (absolute threshold) to produce a binary mask
%  - write the resulting mask to results folder as brainmask_GMtight.nii
%
% Parameters:
%  stats_folder - folder containing SPM.mat and where the template mask will be written
%  src_mask     - path to source mask (e.g. CAT12/brainmask_GMtight.nii)
%  'threshold'  - numeric absolute threshold to apply to src_mask (optional)

p = inputParser;
addRequired(p,'stats_folder',@ischar);
addRequired(p,'src_mask',@ischar);
addParameter(p,'threshold',[],@(x) isempty(x) || isnumeric(x));
parse(p,stats_folder,src_mask,varargin{:});

stats_folder = p.Results.stats_folder;
src_mask = p.Results.src_mask;
thresh = p.Results.threshold;

% Destination filename in results folder: prefer canonical name
out_mask = fullfile(stats_folder, 'brainmask_GMtight.nii');

if ~exist(src_mask,'file')
    error('Source mask not found: %s', src_mask);
end

% backup existing mask if present
if exist(out_mask,'file')
    tstamp = datestr(now,'yyyymmdd_HHMMSS');
    bak = [out_mask,'.bak.',tstamp];
    fprintf('Backing up existing mask to: %s\n', bak);
    copyfile(out_mask, bak);
end

% Try to load with SPM first (NIfTI)
try
    Vm = spm_vol(src_mask);
    M = spm_read_vols(Vm);
catch
    % Try to convert GIfTI or other formats: if not NIfTI, try using
    % SPM's nifti function on gzipped or different ext
    try
        % attempt to use spm_vol on gzipped variants
        [pdir, name, ext] = fileparts(src_mask);
        if strcmp(ext,'.gii')
            error('GIfTI input - please convert to NIfTI before calling this helper.');
        else
            error('Could not read source mask via spm_vol');
        end
    catch ME
        rethrow(ME);
    end
end

% If threshold provided, binarize accordingly
if ~isempty(thresh)
    fprintf('Applying absolute threshold %.4g to source mask\n', thresh);
    Mbin = M > thresh;
else
    % If source is already binary-like, just use >0
    Mbin = M > 0;
end

% Ensure mask is same orientation / dims as stats images: prefer to
% write mask with same header as first beta image if available
beta_glob = spm_select('FPList', stats_folder, '^beta_.*\.nii(\.gz)?$');
if ~isempty(beta_glob)
    Vref = spm_vol(beta_glob(1,:));
    Vout = Vref;
    Vout.fname = out_mask;
    fprintf('Writing mask using beta reference header: %s\n', Vref.fname);
else
    % fallback: use source header
    Vout = Vm;
    Vout.fname = out_mask;
    fprintf('Writing mask using source header: %s\n', src_mask);
end

% Ensure datatype is unsigned char for mask
Vout.dt = [2 0]; % UINT8

spm_write_vol(Vout, double(Mbin));

fprintf('Wrote new mask: %s\n', out_mask);

end
