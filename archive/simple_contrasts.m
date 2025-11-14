spm_file = 'vol_mri_s6_fixed/SPM.mat';

% Load SPM
load(spm_file, 'SPM');

fprintf('SPM loaded: %d existing contrasts\n', numel(SPM.xCon));

% Keep the first contrast if it exists, or create template
if isempty(SPM.xCon)
    error('No contrasts in SPM.mat');
end

con_template = SPM.xCon(1);
SPM.xCon = []; % Clear all

% Define all contrasts
contrasts = {
    struct('name', 'G1: T3-T1', 'c', [-1 0 1 0 0 0 0 0 0], 'STAT', 'T');
    struct('name', 'G1: T2-T1', 'c', [-1 1 0 0 0 0 0 0 0], 'STAT', 'T');
    struct('name', 'G1: T3-T2', 'c', [0 -1 1 0 0 0 0 0 0], 'STAT', 'T');
    struct('name', 'G1: Linear', 'c', [-1 0 1 0 0 0 0 0 0], 'STAT', 'T');
    struct('name', 'G2: T3-T1', 'c', [0 0 0 -1 0 1 0 0 0], 'STAT', 'T');
    struct('name', 'G2: T2-T1', 'c', [0 0 0 -1 1 0 0 0 0], 'STAT', 'T');
    struct('name', 'G2: T3-T2', 'c', [0 0 0 0 -1 1 0 0 0], 'STAT', 'T');
    struct('name', 'G2: Linear', 'c', [0 0 0 -1 0 1 0 0 0], 'STAT', 'T');
    struct('name', 'G3: T3-T1', 'c', [0 0 0 0 0 0 -1 0 1], 'STAT', 'T');
    struct('name', 'G3: T2-T1', 'c', [0 0 0 0 0 0 -1 1 0], 'STAT', 'T');
    struct('name', 'G3: T3-T2', 'c', [0 0 0 0 0 0 0 -1 1], 'STAT', 'T');
    struct('name', 'G3: Linear', 'c', [0 0 0 0 0 0 -1 0 1], 'STAT', 'T');
    struct('name', 'T1: G1-G2', 'c', [1 0 0 -1 0 0 0 0 0], 'STAT', 'T');
    struct('name', 'T1: G1-G3', 'c', [1 0 0 0 0 0 -1 0 0], 'STAT', 'T');
    struct('name', 'T1: G2-G3', 'c', [0 0 0 1 0 0 -1 0 0], 'STAT', 'T');
    struct('name', 'T3: G1-G2', 'c', [0 0 1 0 0 -1 0 0 0], 'STAT', 'T');
    struct('name', 'T3: G1-G3', 'c', [0 0 1 0 0 0 0 0 -1], 'STAT', 'T');
    struct('name', 'T3: G2-G3', 'c', [0 0 0 0 0 1 0 0 -1], 'STAT', 'T');
    struct('name', 'GxT: G1vG2', 'c', [-1 0 1 1 0 -1 0 0 0], 'STAT', 'T');
    struct('name', 'GxT: G1vG3', 'c', [-1 0 1 0 0 0 1 0 -1], 'STAT', 'T');
    struct('name', 'GxT: G2vG3', 'c', [0 0 0 -1 0 1 1 0 -1], 'STAT', 'T');
    struct('name', 'All: T3-T1', 'c', [-1 0 1 -1 0 1 -1 0 1]/3, 'STAT', 'T');
    struct('name', 'All: Time', 'c', [-1 0 1 -1 0 1 -1 0 1], 'STAT', 'T');
    struct('name', 'Mean', 'c', ones(1,9)/9, 'STAT', 'T');
    struct('name', 'F: Time', 'c', [-1 0 1 -1 0 1 -1 0 1; 0 -1 1 0 -1 1 0 -1 1], 'STAT', 'F');
    struct('name', 'F: Group', 'c', [1 -1 0 -1 1 0; 1 -1 0 -1 1 0; 1 -1 0 -1 1 0], 'STAT', 'F');
    struct('name', 'F: GxT', 'c', [-1 0 1 1 0 -1 0 0 0; -1 0 1 0 0 0 1 0 -1], 'STAT', 'F');
};

% Add each contrast to SPM.xCon
for k = 1:numel(contrasts)
    con = contrasts{k};
    SPM.xCon(k) = con_template;
    SPM.xCon(k).name = con.name;
    SPM.xCon(k).STAT = con.STAT;
    SPM.xCon(k).c = con.c(:);
    SPM.xCon(k).eidf = 1;
end

save(spm_file, 'SPM', '-append');
fprintf('✓ Added %d contrasts to SPM.xCon\n', numel(contrasts));
quit;
