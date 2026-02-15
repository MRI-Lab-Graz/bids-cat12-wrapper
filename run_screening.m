%% Run screening contrasts only
clear all; close all;

OUTPUT_DIR = '/Volumes/Thunder/129_PK01/cat12/stats/results/vbm/vbm_9mm_3G_2TP_tiv_sex_age';
LOG_DIR = fullfile(OUTPUT_DIR, 'logs');
STATS_DIR = '/Volumes/Thunder/129_PK01/cat12/stats';

warning('off','MATLAB:dispatcher:nameConflict');
warning('off','all');
set(0,'DefaultFigureVisible','off');
set(0,'DefaultFigureCreateFcn',@(h,ev)[]);

addpath(fullfile(STATS_DIR,'scripts','utils'));

fprintf('Starting SPM initialization...\n');
spm('defaults','FMRI');
spm_jobman('initcfg');

fprintf('Running screening contrasts...\n');
try
    significant_contrasts = screen_contrasts(OUTPUT_DIR,'p_thresh',0.001,'cluster_size',10);
    fprintf('\n✓ Screening complete with %d significant contrasts\n\n', length(significant_contrasts));
    
    % Save significant contrasts to file
    fid = fopen(fullfile(LOG_DIR,'significant_contrasts.txt'),'w');
    if fid > 0
        for ii = 1:numel(significant_contrasts)
            fprintf(fid,'%d\n',significant_contrasts(ii));
        end
        fclose(fid);
        fprintf('Saved significant contrasts to: %s\n', fullfile(LOG_DIR,'significant_contrasts.txt'));
    end
catch e
    fprintf('ERROR in screening: %s\n', e.message);
end

exit;
