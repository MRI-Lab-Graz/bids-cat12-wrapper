function out_mat = export_longitudinal_job(stats_folder, out_mat)
% EXPORT_LONGITUDINAL_JOB  Create an SPM Batch (.mat) to inspect contrasts
%
%   out_mat = export_longitudinal_job(stats_folder, out_mat)
%
% This helper builds a minimal `matlabbatch` containing one
% `spm.stats.results` job per contrast found in the SPM.mat located in
% `stats_folder`. The generated .mat can be loaded into the SPM Batch
% Editor (File -> Load) or loaded into MATLAB and opened interactively
% with `spm_jobman(''interactive'', matlabbatch)`.
%
% Inputs:
%   stats_folder - folder containing SPM.mat (default: pwd)
%   out_mat      - output .mat filename (optional). Default:
%                  fullfile(stats_folder,'longitudinal_pipeline_job.mat')
%
% Output:
%   out_mat      - path to written .mat file containing variable
%                  `matlabbatch`.
%
% Notes:
% - This file only *saves* the jobs; it does not run them. Use the SPM
%   Batch Editor to inspect or edit parameters before running.
% - The job uses an uncorrected threshold of p<0.001 and extent 0; you
%   can edit thresholds in the Batch Editor.

if nargin < 1 || isempty(stats_folder)
    stats_folder = pwd;
end
if nargin < 2 || isempty(out_mat)
    out_mat = fullfile(stats_folder, 'longitudinal_pipeline_job.mat');
end

spm_mat = fullfile(stats_folder, 'SPM.mat');
if ~exist(spm_mat, 'file')
    error('SPM.mat not found in %s', stats_folder);
end

% Load SPM to count contrasts
s = load(spm_mat, 'SPM');
SPM = s.SPM;

ncon = length(SPM.xCon);
matlabbatch = cell(1, max(1, ncon));

for i = 1:max(1,ncon)
    % Minimal results job for contrast i
    matlabbatch{i}.spm.stats.results.spmmat = {spm_mat};
    matlabbatch{i}.spm.stats.results.conspec(1).titlestr = {''};
    % If there are no contrasts (unlikely) set 1, else use i
    if ncon >= 1
        matlabbatch{i}.spm.stats.results.conspec(1).contrasts = i;
    else
        matlabbatch{i}.spm.stats.results.conspec(1).contrasts = 1;
    end
    matlabbatch{i}.spm.stats.results.conspec(1).threshdesc = 'none';
    matlabbatch{i}.spm.stats.results.conspec(1).thresh = 0.001;
    matlabbatch{i}.spm.stats.results.conspec(1).extent = 0;
    matlabbatch{i}.spm.stats.results.conspec(1).conjunction = 1;
    % Prefer the canonical repo template mask (CAT12 tight) if available.
    % This avoids per-results 'mask_vbm.nii' files and ensures a single
    % canonical mask is used across analyses.
    utils_dir = fileparts(mfilename('fullpath'));
    repo_root = fileparts(utils_dir);
    template_mask = fullfile(repo_root, 'templates', 'brainmask_GMtight.nii');
    if exist(template_mask, 'file')
        matlabbatch{i}.spm.stats.results.conspec(1).mask.em = {template_mask};
    else
        % No explicit mask provided; use default implicit masking
        matlabbatch{i}.spm.stats.results.conspec(1).mask.none = 1;
    end
    % Optional: display units (not critical)
    matlabbatch{i}.spm.stats.results.units = 1;
    matlabbatch{i}.spm.stats.results.print = false;
end

% Save the matlabbatch to disk
save(out_mat, 'matlabbatch');
fprintf('Wrote SPM Batch file: %s\n', out_mat);
end
