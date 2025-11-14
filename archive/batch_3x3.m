% List of open inputs
% Basic models: Vector - cfg_entry
nrun = X; % enter the number of runs here
jobfile = {'/Volumes/Thunder/129_PK01/cat12/stats/batch_3x3_job.m'};
jobs = repmat(jobfile, 1, nrun);
inputs = cell(1, nrun);
for crun = 1:nrun
    inputs{1, crun} = MATLAB_CODE_TO_FILL_INPUT; % Basic models: Vector - cfg_entry
end
spm('defaults', 'PET');
spm_jobman('run', jobs, inputs{:});
