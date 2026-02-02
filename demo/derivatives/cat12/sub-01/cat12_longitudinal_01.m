
% CAT12 Longitudinal Processing Script
% Subject: 01
% Generated: 2026-02-02 20:33:33

% Initialize CAT12
addpath(genpath(spm('dir')));
spm_jobman('initcfg');

% Subject files
files = {
    'demo/derivatives/cat12/sub-01/sub-01_ses-retest_T1w.nii'
    'demo/derivatives/cat12/sub-01/sub-01_ses-test_T1w.nii'
};

% Output directory
output_dir = 'demo/derivatives/cat12/sub-01';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% CAT12 batch configuration
matlabbatch = struct();

% Longitudinal processing job
if true
    % Longitudinal processing
    matlabbatch{1}.spm.tools.cat.long.datalong.subjects{1}.mov = files;
    matlabbatch{1}.spm.tools.cat.long.datalong.subjects{1}.timepoints = 1:length(files);

    % Processing options
    matlabbatch{1}.spm.tools.cat.long.opts.tpm = {fullfile(spm('dir'),'tpm','TPM.nii')};
    matlabbatch{1}.spm.tools.cat.long.opts.affreg = 'mni';
    matlabbatch{1}.spm.tools.cat.long.opts.biasstr = 0.5;

    % Surface processing
    if true
        matlabbatch{1}.spm.tools.cat.long.surface.pbtres = 0.5;
        matlabbatch{1}.spm.tools.cat.long.surface.pbtmethod = 'pbt2x';
        matlabbatch{1}.spm.tools.cat.long.surface.SRP = 22;
        matlabbatch{1}.spm.tools.cat.long.surface.reduce_mesh = 1;
        matlabbatch{1}.spm.tools.cat.long.surface.vdist = 2;
        matlabbatch{1}.spm.tools.cat.long.surface.scale_cortex = 0.7;
        matlabbatch{1}.spm.tools.cat.long.surface.add_parahipp = 0.1;
        matlabbatch{1}.spm.tools.cat.long.surface.close_parahipp = 0;
    else
        matlabbatch{1}.spm.tools.cat.long.surface = struct();
    end

    % Volume processing
    if true
        matlabbatch{1}.spm.tools.cat.long.output.surface = 1;
        matlabbatch{1}.spm.tools.cat.long.output.ROImenu.atlases.neuromorphometrics = 1;
        matlabbatch{1}.spm.tools.cat.long.output.ROImenu.atlases.lpba40 = 1;
        matlabbatch{1}.spm.tools.cat.long.output.ROImenu.atlases.cobra = 1;
        matlabbatch{1}.spm.tools.cat.long.output.ROImenu.atlases.hammers = 1;
        matlabbatch{1}.spm.tools.cat.long.output.GM.native = 0;
        matlabbatch{1}.spm.tools.cat.long.output.GM.mod = 1;
        matlabbatch{1}.spm.tools.cat.long.output.GM.dartel = 0;
        matlabbatch{1}.spm.tools.cat.long.output.WM.native = 0;
        matlabbatch{1}.spm.tools.cat.long.output.WM.mod = 1;
        matlabbatch{1}.spm.tools.cat.long.output.WM.dartel = 0;
    end

else
    % Cross-sectional processing
    matlabbatch{1}.spm.tools.cat.estwrite.data = files;

    % Standard CAT12 settings
    matlabbatch{1}.spm.tools.cat.estwrite.opts.tpm = {fullfile(spm('dir'),'tpm','TPM.nii')};
    matlabbatch{1}.spm.tools.cat.estwrite.opts.affreg = 'mni';
    matlabbatch{1}.spm.tools.cat.estwrite.opts.biasstr = 0.5;

    % Output options
    matlabbatch{1}.spm.tools.cat.estwrite.output.surface = true;
    matlabbatch{1}.spm.tools.cat.estwrite.output.GM.native = 0;
    matlabbatch{1}.spm.tools.cat.estwrite.output.GM.mod = 1;
    matlabbatch{1}.spm.tools.cat.estwrite.output.GM.dartel = 0;
    matlabbatch{1}.spm.tools.cat.estwrite.output.WM.native = 0;
    matlabbatch{1}.spm.tools.cat.estwrite.output.WM.mod = 1;
    matlabbatch{1}.spm.tools.cat.estwrite.output.WM.dartel = 0;
end

% Run the job
fprintf('Starting CAT12 processing for subject 01...\n');
try
    spm_jobman('run', matlabbatch);
    fprintf('CAT12 processing completed successfully for 01\n');

    % Save processing log
    log_file = fullfile(output_dir, 'cat12_processing_log.mat');
    save(log_file, 'matlabbatch', 'files');

catch ME
    fprintf('Error during CAT12 processing: %s\n', ME.message);
    error_file = fullfile(output_dir, 'cat12_error_log.mat');
    save(error_file, 'ME', 'matlabbatch', 'files');
    rethrow(ME);
end

fprintf('Processing log saved to: %s\n', output_dir);
exit;
