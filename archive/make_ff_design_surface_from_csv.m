function make_ff_design_surface_from_csv(csv_file, out_dir, varargin)
% MAKE_FF_DESIGN_SURFACE_FROM_CSV
% Build and estimate a CAT12 flexible factorial design (surface data) from a CSV.
%
% CSV columns required (case-insensitive):
%   subject  - subject id (string/char)
%   group    - between-subject group label (numeric or string)
%   time     - within-subject time label (numeric or string)
%   path     - absolute path to the GIfTI surface metric for that subject/time
%
% Usage:
%   make_ff_design_surface_from_csv('design.csv', '/path/to/stats', 'keepIncomplete', false)
%
% Options (name/value):
%   'keepIncomplete'  true/false (default false): include subjects missing any timepoint
%   'overwrite'       true/false (default true): overwrite existing SPM.mat in out_dir
%
% Notes:
%   - Factors: Subject (random), Group (between), Time (within, repeated)
%   - This produces the minimal, numerically stable FF setup used by CAT12.
%   - Rank deficiency of ~3 is expected (constraints for full dummy coding). SPM handles this.
%
% Author: Auto-generated helper
% Date: 2025-10-29

p = inputParser;
p.addRequired('csv_file', @(s)ischar(s) || isstring(s));
p.addRequired('out_dir', @(s)ischar(s) || isstring(s));
p.addParameter('keepIncomplete', false, @islogical);
p.addParameter('overwrite', true, @islogical);
p.parse(csv_file, out_dir, varargin{:});
keepIncomplete = p.Results.keepIncomplete;
overwrite = p.Results.overwrite;

csv_file = char(csv_file); out_dir = char(out_dir);
if ~exist(csv_file, 'file')
    error('CSV not found: %s', csv_file);
end
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

T = readtable(csv_file, 'TextType', 'string');
req = {'subject','group','time','path'};
for k = 1:numel(req)
    if ~any(strcmpi(T.Properties.VariableNames, req{k}))
        error('CSV missing required column: %s', req{k});
    end
end

% Normalize columns
subj = string(T{:, find(strcmpi(T.Properties.VariableNames,'subject'))});
grp  = string(T{:, find(strcmpi(T.Properties.VariableNames,'group'))});
tim  = string(T{:, find(strcmpi(T.Properties.VariableNames,'time'))});
pth  = string(T{:, find(strcmpi(T.Properties.VariableNames,'path'))});

% Ensure absolute paths and existence
for i = 1:numel(pth)
    if ~isfile(pth(i))
        error('File not found: %s (row %d)', pth(i), i);
    end
end

% Factor levels
G_levels = unique(grp, 'stable');
T_levels = unique(tim, 'stable');

% Build per-group complete-case subject lists (unless keepIncomplete)
matlabbatch = [];

groups = cell(numel(G_levels),1);

for g = 1:numel(G_levels)
    gmask = grp == G_levels(g);
    Sg = unique(subj(gmask), 'stable');
    keep_subj = true(size(Sg));
    for s = 1:numel(Sg)
        smask = subj == Sg(s) & gmask;
        hasAll = true;
        for t = 1:numel(T_levels)
            if ~any(smask & tim == T_levels(t))
                hasAll = false; break; %#ok<*BREAK>
            end
        end
        if ~keepIncomplete && ~hasAll
            keep_subj(s) = false;
        end
    end
    groups{g} = Sg(keep_subj);
end

% Informative summary
fprintf('Design summary (from %s)\n', csv_file);
for g = 1:numel(G_levels)
    fprintf('  Group %s: %d subjects\n', G_levels(g), numel(groups{g}));
end
fprintf('  Time levels: %s\n', strjoin(cellstr(T_levels), ', '));

% Prepare fsuball.group(k).timepoint{t}
for g = 1:numel(G_levels)
    for t = 1:numel(T_levels)
        file_list = strings(0,1);
        for s = 1:numel(groups{g})
            row = find(subj==groups{g}(s) & grp==G_levels(g) & tim==T_levels(t));
            if isempty(row)
                if keepIncomplete
                    continue; % skip this subject for this time
                else
                    error('Subject %s missing time %s in group %s', groups{g}(s), T_levels(t), G_levels(g));
                end
            end
            file_list(end+1,1) = pth(row(1)); %#ok<AGROW>
        end
        % Assign into batch (cellstr)
        matlabbatch{1}.spm.tools.cat.factorial_design.des.fblock.fsuball.group(g).timepoint{t} = cellstr(file_list);
    end
end

% Output dir
matlabbatch{1}.spm.tools.cat.factorial_design.dir = {out_dir};

% Factors: Subject, Group, Time
% Use settings aligned with CAT12 recommendations
matlabbatch{1}.spm.tools.cat.factorial_design.des.fblock.fac(1).name = 'subject';
matlabbatch{1}.spm.tools.cat.factorial_design.des.fblock.fac(1).dept = 0;
matlabbatch{1}.spm.tools.cat.factorial_design.des.fblock.fac(1).variance = 0;
matlabbatch{1}.spm.tools.cat.factorial_design.des.fblock.fac(1).gmsca = 0;
matlabbatch{1}.spm.tools.cat.factorial_design.des.fblock.fac(1).ancova = 0;

matlabbatch{1}.spm.tools.cat.factorial_design.des.fblock.fac(2).name = 'group';
matlabbatch{1}.spm.tools.cat.factorial_design.des.fblock.fac(2).dept = 0;
matlabbatch{1}.spm.tools.cat.factorial_design.des.fblock.fac(2).variance = 0;
matlabbatch{1}.spm.tools.cat.factorial_design.des.fblock.fac(2).gmsca = 0;
matlabbatch{1}.spm.tools.cat.factorial_design.des.fblock.fac(2).ancova = 0;

matlabbatch{1}.spm.tools.cat.factorial_design.des.fblock.fac(3).name = 'time';
matlabbatch{1}.spm.tools.cat.factorial_design.des.fblock.fac(3).dept = 1;  % repeated measure
matlabbatch{1}.spm.tools.cat.factorial_design.des.fblock.fac(3).variance = 0;
matlabbatch{1}.spm.tools.cat.factorial_design.des.fblock.fac(3).gmsca = 0;
matlabbatch{1}.spm.tools.cat.factorial_design.des.fblock.fac(3).ancova = 0;

% Interactions: Group x Time only (no subject main effect to avoid perfect collinearity)
matlabbatch{1}.spm.tools.cat.factorial_design.des.fblock.maininters{1}.inter.fnums = [2 3];

% Defaults / masking
matlabbatch{1}.spm.tools.cat.factorial_design.des.fblock.voxel_cov.files = {''};
matlabbatch{1}.spm.tools.cat.factorial_design.des.fblock.voxel_cov.iCFI = 1;
matlabbatch{1}.spm.tools.cat.factorial_design.des.fblock.voxel_cov.iCC = 1;
matlabbatch{1}.spm.tools.cat.factorial_design.des.fblock.voxel_cov.globals.g_omit = 1;
matlabbatch{1}.spm.tools.cat.factorial_design.des.fblock.voxel_cov.consess = cell(1,0);

matlabbatch{1}.spm.tools.cat.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.tools.cat.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.tools.cat.factorial_design.masking.tm.tm_none = 1;
matlabbatch{1}.spm.tools.cat.factorial_design.masking.im = 1;
matlabbatch{1}.spm.tools.cat.factorial_design.masking.em = {''};
matlabbatch{1}.spm.tools.cat.factorial_design.globals.g_omit = 1;
% Disable quality checks to avoid headless graphics errors (Renderer property issue)
matlabbatch{1}.spm.tools.cat.factorial_design.check_SPM.check_SPM_zscore.none = 1;
matlabbatch{1}.spm.tools.cat.factorial_design.check_SPM.check_SPM_ortho = 0;

% Estimate model
matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('Basic models: SPM.mat File', ...
    substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), ...
    substruct('.','spmmat'));
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

% If SPM.mat exists and overwrite=false, stop
if exist(fullfile(out_dir,'SPM.mat'),'file') && ~overwrite
    error('SPM.mat already exists in %s (set ''overwrite'',true to replace).', out_dir);
end

% Initialize SPM and run
spm('defaults','FMRI'); spm_jobman('initcfg');
spm_jobman('run', matlabbatch);

% Brief design audit
S = load(fullfile(out_dir,'SPM.mat')); X = S.SPM.xX.X;
[rX, cX] = size(X); rk = rank(full(X));
try cd(out_dir); end
fprintf('\nDesign built: X = %d x %d, rank = %d, cond = %.3e\n', rX, cX, rk, cond(full(X)));
end
