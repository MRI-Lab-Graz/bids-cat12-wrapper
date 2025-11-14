function inspect_spm_mat(stats_folder)
% INSPECT_SPM_MAT - Print a concise audit of SPM.mat design and inputs
% Usage:
%   inspect_spm_mat('/path/to/stats_folder')
%
% This function does NOT require SPM functions; it only loads SPM.mat and
% reports design size, rank, covariates, potential collinearity, and mask presence.
%
% Author: Auto-generated helper
% Date: 2025-10-29

if nargin < 1 || ~ischar(stats_folder)
    error('Provide a stats folder path containing SPM.mat');
end

spm_mat = fullfile(stats_folder, 'SPM.mat');
if ~exist(spm_mat, 'file')
    error('SPM.mat not found at: %s', spm_mat);
end

S = load(spm_mat);
if ~isfield(S, 'SPM')
    error('Invalid SPM.mat: variable SPM missing');
end
SPM = S.SPM; %#ok<NODEF>

fprintf('\n══════════════════════════════════════════════════════════════════════\n');
fprintf('SPM.mat DESIGN SUMMARY: %s\n', stats_folder);
fprintf('══════════════════════════════════════════════════════════════════════\n');

%% Scans and basic design info
try
    P = SPM.xY.P; % can be char array or cellstr
    if ischar(P)
        n_scans = size(P,1);
    elseif iscell(P)
        n_scans = numel(P);
    else
        n_scans = numel(P);
    end
catch
    n_scans = NaN;
end

try
    X = SPM.xX.X;
    [n_obs, n_reg] = size(X);
    rX = rank(full(X));
    condX = NaN;
    try
        condX = cond(full(X));
    catch
    end
catch ME
    warning('Could not access SPM.xX.X: %s', ME.message);
    n_obs = NaN; n_reg = NaN; rX = NaN; condX = NaN; X = [];
end

erdf = NaN;
if isfield(SPM, 'xX') && isfield(SPM.xX, 'erdf')
    erdf = SPM.xX.erdf;
end

fprintf('Scans (xY.P):               %s\n', num2str(n_scans));
fprintf('Design matrix size (X):     %s x %s\n', num2str(n_obs), num2str(n_reg));
fprintf('Rank(X):                    %s\n', num2str(rX));
if ~isnan(condX)
    fprintf('Condition number(X):        %.3e\n', condX);
end
fprintf('Error degrees of freedom:   %s\n', num2str(erdf));

%% Sessions (if present)
if isfield(SPM, 'Sess')
    fprintf('Sessions detected:          %d\n', numel(SPM.Sess));
end

%% Contrast names
if isfield(SPM, 'xCon')
    fprintf('Contrasts defined:          %d\n', numel(SPM.xCon));
    for k = 1:min(10, numel(SPM.xCon))
        fprintf('  [%02d] %s\n', k, SPM.xCon(k).name);
    end
end

%% Column names
if isfield(SPM, 'xX') && isfield(SPM.xX, 'name') && iscell(SPM.xX.name)
    fprintf('Regressor names (first 20):\n');
    for i = 1:min(20, numel(SPM.xX.name))
        fprintf('   %3d: %s\n', i, SPM.xX.name{i});
    end
end

%% Covariates
if isfield(SPM, 'xC') && ~isempty(SPM.xC)
    C = SPM.xC; if isstruct(C), C = num2cell(C); end
    fprintf('Covariates found:           %d\n', numel(C));
    for i = 1:numel(C)
        ci = C{i};
        cname = '<unnamed>';
        if isfield(ci, 'name') && ~isempty(ci.name), cname = ci.name; end
        cvec = [];
        if isfield(ci, 'C') && ~isempty(ci.C), cvec = ci.C; end
        m = NaN; s = NaN;
        if ~isempty(cvec)
            m = mean(cvec(:), 'omitnan');
            s = std(cvec(:), 0, 'omitnan');
        end
        fprintf('  - %s (mean=%.4g, sd=%.4g, n=%d)\n', cname, m, s, numel(cvec));
        if ~isempty(cvec) && (s < 1e-6)
            fprintf('    ⚠ Covariate has near-zero variance → may be uninformative.\n');
        end
    end
else
    fprintf('Covariates found:           0\n');
end

%% Attempt to infer subject/block indicators from names
block_like = {};
if isfield(SPM, 'xX') && isfield(SPM.xX, 'name') && iscell(SPM.xX.name)
    for i = 1:numel(SPM.xX.name)
        nm = lower(SPM.xX.name{i});
        if contains(nm, 'subject') || contains(nm, 'subj') || contains(nm, 'sess(')
            block_like{end+1} = SPM.xX.name{i}; %#ok<AGROW>
        end
    end
end
if ~isempty(block_like)
    fprintf('Potential subject/block columns: %d\n', numel(block_like));
    for i = 1:min(10, numel(block_like))
        fprintf('   · %s\n', block_like{i});
    end
end

%% Mask presence in stats folder
mask_gii = fullfile(stats_folder, 'mask.gii');
mask_nii = fullfile(stats_folder, 'mask.nii');
if exist(mask_gii, 'file')
    d = dir(mask_gii); fprintf('Mask (surface):             %s (%.1f KB)\n', mask_gii, d.bytes/1024);
elseif exist(mask_nii, 'file')
    d = dir(mask_nii); fprintf('Mask (volume):              %s (%.1f KB)\n', mask_nii, d.bytes/1024);
else
    fprintf('Mask:                       NOT FOUND in stats folder\n');
end

%% Heuristics and warnings
fprintf('\nHeuristics / Potential Issues:\n');
if ~isempty(X)
    if rX < n_reg
        fprintf('  ⚠ Rank deficiency: rank(X)=%d < %d columns → collinearity/redundant columns.\n', rX, n_reg);
    end
    if ~isnan(condX) && condX > 1e10
        fprintf('  ⚠ Very ill-conditioned design (cond>1e10) → inflated/unstable stats likely.\n');
    end
end
if ~exist(mask_gii, 'file') && ~exist(mask_nii, 'file')
    fprintf('  ⚠ No explicit mask detected → consider providing mask.gii/mask.nii to avoid edge/non-brain voxels.\n');
end
if isfield(SPM, 'xX') && isfield(SPM.xX, 'DT') && ~isempty(SPM.xX.DT)
    % Nothing specific here; placeholder if needed later
end

fprintf('\nDone.\n');
end
