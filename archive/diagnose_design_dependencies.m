function diagnose_design_dependencies(stats_folder)
% DIAGNOSE_DESIGN_DEPENDENCIES - Identify linear dependencies in SPM.xX.X
% Prints the null space vectors and their largest coefficients mapped to regressor names.
%
% Usage:
%   diagnose_design_dependencies('/path/to/stats_folder')
%
S = load(fullfile(stats_folder,'SPM.mat'));
SPM = S.SPM;
X = full(SPM.xX.X);
names = SPM.xX.name;

fprintf('\nDesign dependency diagnosis: %s\n', stats_folder);
[n_obs, n_reg] = size(X);
fprintf('X size: %d x %d\n', n_obs, n_reg);

% Compute a basis for the null space of X
Z = null(X,'r');  % rational basis when possible
if isempty(Z)
    fprintf('No exact linear dependencies detected (full column rank).\n');
    return;
end

k = size(Z,2);
fprintf('Null space dimension: %d (number of exact linear dependencies)\n', k);

for j = 1:k
    v = Z(:,j);
    % Normalize by max abs coefficient for readability
    [m, idx] = max(abs(v)); %#ok<ASGLU>
    if m == 0, m = 1; end
    v = v / m;
    % Show the top 20 contributors in absolute value
    [~, order] = sort(abs(v), 'descend');
    top = order(1:min(20, numel(order)));
    fprintf('\nDependency #%d (top coefficients):\n', j);
    for t = 1:numel(top)
        ii = top(t);
        if abs(v(ii)) < 1e-6, break; end
        coef = v(ii);
        nm = '<unnamed>';
        if iscell(names) && ii <= numel(names)
            nm = names{ii};
        end
        fprintf('   %+8.4f  [%3d] %s\n', coef, ii, nm);
    end
end

fprintf('\nInterpretation tips:\n');
fprintf('  • Dependencies with many subject_{i} columns and a nearly opposite constant/cell-sum indicate\n');
fprintf('    the expected constraints (sum-to-zero over factors or redundant intercept).\n');
fprintf('  • For repeated measures with Subject, Group, Time, expect ~3 dependencies:\n');
fprintf('    one for overall mean, and one for each factor coding (Group, Time) when fully dummy-coded.\n');
end
