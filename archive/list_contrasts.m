load('vol_mri_s6_fixed/SPM.mat', 'SPM');
fprintf('\n=== Contrast List ===\n');
fprintf('Total contrasts: %d\n\n', numel(SPM.xCon));
for k = 1:numel(SPM.xCon)
    fprintf('%02d. [%s] %s\n', k, SPM.xCon(k).STAT, SPM.xCon(k).name);
end
quit;
