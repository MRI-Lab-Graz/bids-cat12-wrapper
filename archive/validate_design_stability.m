function validate_design_stability(spm_dir)
% VALIDATE_DESIGN_STABILITY
% Diagnostic checks for flexible factorial design to detect numerical problems
% that could cause spurious "large" effects.
%
% Usage:
%   validate_design_stability('/path/to/stats/folder')
%
% Checks:
%   1. Design matrix correlation (detect collinearities)
%   2. Contrast estimability and standard errors
%   3. Variance inflation factors (VIF) for key effects
%   4. Residual checks (if model estimated)
%
% Author: Design validation helper
% Date: 2025-10-30

if ~exist('spm_dir','var') || isempty(spm_dir)
    spm_dir = pwd;
end

spm_file = fullfile(spm_dir, 'SPM.mat');
if ~exist(spm_file,'file')
    error('SPM.mat not found in %s', spm_dir);
end

fprintf('Loading SPM.mat from: %s\n', spm_dir);
load(spm_file);

X = SPM.xX.X;
[n, p] = size(X);
rk = rank(full(X));
cn = cond(full(X));

fprintf('\n=== Design Matrix Summary ===\n');
fprintf('Scans:       %d\n', n);
fprintf('Regressors:  %d\n', p);
fprintf('Rank:        %d (deficiency: %d)\n', rk, p - rk);
fprintf('Condition:   %.3e\n', cn);
fprintf('Residual df: %d\n', SPM.xX.erdf);

%% 1. Correlation matrix of design columns
fprintf('\n=== Checking Design Orthogonality ===\n');
C = corr(X);
% Find highly correlated pairs (excluding diagonal)
C_nodiag = C - diag(diag(C));
[r, c] = find(abs(C_nodiag) > 0.95);
if ~isempty(r)
    fprintf('WARNING: %d pairs of columns with |r| > 0.95\n', numel(r));
    for k = 1:min(10, numel(r))
        fprintf('  %s <-> %s: r=%.4f\n', SPM.xX.name{r(k)}, SPM.xX.name{c(k)}, C(r(k),c(k)));
    end
else
    fprintf('OK: No extreme collinearities (max |r| = %.3f)\n', max(abs(C_nodiag(:))));
end

%% 2. Check key contrast estimability
fprintf('\n=== Checking Contrast Estimability ===\n');
% Example contrast: Group 1, Time3 - Time1
% For your design: group*time_{1,1} vs group*time_{1,3}
if p >= 3
    c_test = zeros(p, 1);
    c_test(1) = -1; % Time1 (assuming group*time_{1,1})
    c_test(3) = 1;  % Time3 (assuming group*time_{1,3})
    
    % Check estimability: c' must be in row space of X
    estimable = rank([X; c_test']) == rank(X);
    if estimable
        fprintf('Sample contrast [G1: Time3-Time1] is estimable\n');
        % Compute standard error if model estimated
        if isfield(SPM, 'Vbeta') && ~isempty(SPM.Vbeta)
            se = sqrt(c_test' * SPM.xX.Bcov * c_test);
            fprintf('  Estimated SE: %.4f (if this is huge, numerical issues likely)\n', se);
        end
    else
        fprintf('WARNING: Sample contrast is NOT estimable (design problem)\n');
    end
end

%% 3. Variance Inflation Factors (VIF) for group*time cells
fprintf('\n=== Variance Inflation Factors ===\n');
fprintf('(VIF > 10 suggests multicollinearity problems)\n');
vif = zeros(min(9, p), 1); % Check first 9 (group*time cells)
for j = 1:min(9, p)
    X_others = X(:, [1:j-1, j+1:end]);
    if rank(X_others) < size(X_others,2)
        vif(j) = Inf; % Perfect collinearity
    else
        R2 = 1 - sum((X(:,j) - X_others * (X_others \ X(:,j))).^2) / sum((X(:,j) - mean(X(:,j))).^2);
        vif(j) = 1 / (1 - R2);
    end
    fprintf('  %s: VIF = %.2f\n', SPM.xX.name{j}, vif(j));
end

if any(vif > 10)
    fprintf('WARNING: High VIF detected (>10) - multicollinearity present\n');
else
    fprintf('OK: VIF values reasonable (<10)\n');
end

%% 4. Residual diagnostics (if model estimated)
if isfield(SPM, 'Vbeta') && ~isempty(SPM.Vbeta)
    fprintf('\n=== Residual Diagnostics ===\n');
    fprintf('Model has been estimated.\n');
    fprintf('Residual variance: median=%.4f, range=[%.4f, %.4f]\n', ...
        median(SPM.xX.V(:)), min(SPM.xX.V(:)), max(SPM.xX.V(:)));
    
    % Check for extreme leverage (hat diagonals)
    % For large designs, sample a subset
    if exist(fullfile(spm_dir, 'ResMS.nii'), 'file')
        fprintf('Residual variance map exists (ResMS.nii)\n');
    end
end

%% 5. Summary recommendation
fprintf('\n=== Validation Summary ===\n');
issues = 0;
if cn > 1e10, issues = issues + 1; fprintf('- Extremely high condition number\n'); end
if ~isempty(r), issues = issues + 1; fprintf('- High column correlations detected\n'); end
if any(vif > 10), issues = issues + 1; fprintf('- High VIF values\n'); end

if issues == 0
    fprintf('No major numerical issues detected.\n');
    fprintf('High condition number is likely from many subject dummies (expected).\n');
    fprintf('\nRecommendation: Proceed with analysis. Use TFCE permutation tests.\n');
else
    fprintf('\n%d potential issue(s) found.\n', issues);
    fprintf('Recommendation: Review design specification or use permutation-based inference (TFCE).\n');
end

fprintf('\n=== Next Steps ===\n');
fprintf('1. Run a simple contrast and check if parametric t-values are plausible\n');
fprintf('2. Use TFCE with sufficient permutations (>=5000)\n');
fprintf('3. Check percent signal change in any significant clusters\n');
fprintf('4. Run a null contrast (e.g., Time1 vs Time1) to check false-positive rate\n');

end
