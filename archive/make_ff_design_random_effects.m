function make_ff_design_random_effects(csv_file, out_dir, varargin)
% MAKE_FF_DESIGN_RANDOM_EFFECTS
% Build a flexible factorial design with subject as TRUE random effect (no dummies).
% Uses SPM's native factorial_design (not CAT12 wrapper) to model subject variance
% as a random component rather than fixed dummies.
%
% Compare this to make_ff_design_surface_from_csv to validate design choices.
%
% Usage:
%   make_ff_design_random_effects('design_mri_s6.csv', '/path/to/stats_random')

p = inputParser;
p.addRequired('csv_file', @(s)ischar(s) || isstring(s));
p.addRequired('out_dir', @(s)ischar(s) || isstring(s));
p.addParameter('keepIncomplete', false, @islogical);
p.addParameter('overwrite', true, @islogical);
p.parse(csv_file, out_dir, varargin{:});

csv_file = char(csv_file); out_dir = char(out_dir);

if ~exist(csv_file, 'file'), error('CSV not found: %s', csv_file); end
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

T = readtable(csv_file, 'TextType', 'string');
subj = string(T.subject);
grp  = string(T.group);
tim  = string(T.time);
pth  = string(T.path);

G_levels = unique(grp, 'stable');
T_levels = unique(tim, 'stable');

% Build per-group subject lists (complete cases only for simplicity)
groups = cell(numel(G_levels),1);
for g = 1:numel(G_levels)
    gmask = grp == G_levels(g);
    Sg = unique(subj(gmask), 'stable');
    keep_subj = true(size(Sg));
    for s = 1:numel(Sg)
        smask = subj == Sg(s) & gmask;
        hasAll = true;
        for t = 1:numel(T_levels)
            if ~any(smask & tim == T_levels(t)), hasAll = false; break; end
        end
        if ~p.Results.keepIncomplete && ~hasAll, keep_subj(s) = false; end
    end
    groups{g} = Sg(keep_subj);
end

fprintf('Random-effects design (from %s)\n', csv_file);
for g = 1:numel(G_levels)
    fprintf('  Group %s: %d subjects\n', G_levels(g), numel(groups{g}));
end
fprintf('  Time levels: %s\n', strjoin(cellstr(T_levels), ', '));

% Use SPM's standard factorial_design (NOT cat.factorial_design)
matlabbatch{1}.spm.stats.factorial_design.dir = {out_dir};

% Factors: group (between), time (within/repeated)
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(1).name = 'group';
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(1).dept = 0;
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(1).variance = 1; % unequal variance
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(1).gmsca = 0;
matlabbatch{1}.spm.tools.cat.factorial_design.des.fblock.fac(1).ancova = 0;

matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(2).name = 'time';
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(2).dept = 1; % repeated
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(2).variance = 1;
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(2).gmsca = 0;
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(2).ancova = 0;

% Subject as random factor (key difference: this models subject variance, no dummies)
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(3).name = 'subject';
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(3).dept = 0;
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(3).variance = 1; % random variance
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(3).gmsca = 0;
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(3).ancova = 0;

% Populate scans by group and time
for g = 1:numel(G_levels)
    for t = 1:numel(T_levels)
        file_list = strings(0,1);
        subj_list = [];
        for s = 1:numel(groups{g})
            row = find(subj==groups{g}(s) & grp==G_levels(g) & tim==T_levels(t));
            if isempty(row), continue; end
            file_list(end+1,1) = pth(row(1));
            subj_list(end+1) = s; % subject index within group
        end
        matlabbatch{1}.spm.stats.factorial_design.des.fblock.fsuball.fsubject(s).scans = cellstr(file_list);
        matlabbatch{1}.spm.stats.factorial_design.des.fblock.fsuball.fsubject(s).conds = [g*ones(size(file_list)); t*ones(size(file_list)); subj_list'];
    end
end

% Interactions
matlabbatch{1}.spm.stats.factorial_design.des.fblock.maininters{1}.inter.fnums = [1 2]; % group x time

% Masking/globals
matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

% Estimate
matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('Factorial design specification: SPM.mat File', ...
    substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

if exist(fullfile(out_dir,'SPM.mat'),'file') && ~p.Results.overwrite
    error('SPM.mat exists in %s', out_dir);
end

spm('defaults','FMRI'); spm_jobman('initcfg');
spm_jobman('run', matlabbatch);

S = load(fullfile(out_dir,'SPM.mat')); X = S.SPM.xX.X;
[rX, cX] = size(X); rk = rank(full(X));
fprintf('\nRandom-effects design: X = %d x %d, rank = %d, cond = %.3e\n', rX, cX, rk, cond(full(X)));
end
