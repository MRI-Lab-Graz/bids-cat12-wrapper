load('vol_mri_s6_fixed/SPM.mat', 'SPM');

fprintf('\n=== Design Structure ===\n');
fprintf('Design matrix: %d scans × %d parameters\n', size(SPM.xX.X,1), size(SPM.xX.X,2));
fprintf('Rank: %d\n', rank(SPM.xX.X));

fprintf('\n=== Variance Components (ReML) ===\n');
if isfield(SPM, 'xVi')
    fprintf('Non-sphericity modeling: %s\n', SPM.xVi.form);
    fprintf('Variance components: %d\n', numel(SPM.xVi.Vi));
    
    % Check if subject structure is preserved
    if isfield(SPM, 'xX') && isfield(SPM.xX, 'I')
        fprintf('\n=== Subject Structure ===\n');
        subj_ids = unique(SPM.xX.I);
        fprintf('Number of subjects detected: %d\n', numel(subj_ids));
        fprintf('Scans per subject range: %d to %d\n', ...
            min(histc(SPM.xX.I, subj_ids)), max(histc(SPM.xX.I, subj_ids)));
    end
end

fprintf('\n=== Factors ===\n');
fprintf('Factor 1 (Group): %d levels\n', numel(unique(SPM.xX.iB)));
fprintf('Factor 2 (Time): %d levels\n', numel(unique(SPM.xX.iC)));

quit;
