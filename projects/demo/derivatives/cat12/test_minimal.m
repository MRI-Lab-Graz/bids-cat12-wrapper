% Minimal CAT12 longitudinal batch test
addpath(genpath(spm('dir')));
spm_jobman('initcfg');

% Create proper batch structure for cat.long
matlabbatch{1}.spm.tools.cat.long.datalong.subjects = struct('mov', {}, 'timepoints', {});
matlabbatch{1}.spm.tools.cat.long.datalong.subjects(1).mov = {
    '/Volumes/Thunder/129_PK01/cat12/stats/projects/demo/derivatives/cat12/sub-01/sub-01_ses-retest_T1w.nii'
    '/Volumes/Thunder/129_PK01/cat12/stats/projects/demo/derivatives/cat12/sub-01/sub-01_ses-test_T1w.nii'
};
matlabbatch{1}.spm.tools.cat.long.datalong.subjects(1).timepoints = [1 2];

% Check configuration
cfg_util('initjob', matlabbatch);
fprintf('Batch structure is valid!\n');
