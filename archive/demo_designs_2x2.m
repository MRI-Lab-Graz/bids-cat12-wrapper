% Demo: 2x2 longitudinal design matrices
% - Manual-style: Subject main effect + full Group×Time cells (rank-deficient in nested design)
% - fsuball-style: Only Group×Time cells; repeated-measures via ReML (full rank)

% Setup: 2 groups (A,B), 2 time points (T1,T2), 6 subjects (3 per group)
clear; clc;

nG = 2;         % groups
nT = 2;         % timepoints
nS = 6;         % subjects (3 per group)
scansPerSubj = nT;
N = nS * scansPerSubj;  % total scans/rows

% Subject-to-group assignment (subjects 1-3 in group 1, 4-6 in group 2)
subjGroup = [1 1 1 2 2 2];

% Row mapping: rows are [S1_T1, S1_T2, S2_T1, S2_T2, ..., S6_T1, S6_T2]
rows_subj = repelem(1:nS, scansPerSubj)';
rows_time = repmat(1:nT, 1, nS)';

% Build Group×Time cell coding (4 columns: A_T1, A_T2, B_T1, B_T2)
X_cells = zeros(N, nG*nT);
for i = 1:N
    s = rows_subj(i);
    t = rows_time(i);
    g = subjGroup(s);
    col = (g-1)*nT + t;
    X_cells(i, col) = 1;
end

% Build Subject dummies (6 columns)
X_subject = zeros(N, nS);
for i = 1:N
    s = rows_subj(i);
    X_subject(i, s) = 1;
end

% Combine for manual-style design (Subject main effect + Group×Time cells)
X_manual = [X_cells, X_subject];

% fsuball-style design is just the cell coding
X_fsub = X_cells;

% Compute ranks and conditions
r_manual = rank(X_manual);
r_fsub   = rank(X_fsub);

c_manual = cond(X_manual);  % will be Inf if singular
c_fsub   = cond(X_fsub);

% Print summary
fprintf('Manual-style: %dx%d, rank=%d, cond=%s\n', size(X_manual,1), size(X_manual,2), r_manual, mat2str(c_manual));
fprintf('fsuball-style: %dx%d, rank=%d, cond=%.2e\n', size(X_fsub,1), size(X_fsub,2), r_fsub, c_fsub);

% Visualize
fig = figure('Color','w','Units','pixels','Position',[100 100 1200 480],'Visible','off');

subplot(1,2,1);
imagesc(X_manual);
colormap(gray);
axis tight; box on;
title(sprintf('Manual: %d x %d, rank=%d, cond=%s', size(X_manual,1), size(X_manual,2), r_manual, char(string(c_manual))))
xlabel('columns'); ylabel('rows (scans)');

subplot(1,2,2);
imagesc(X_fsub);
colormap(gray);
axis tight; box on;
title(sprintf('fsuball: %d x %d, rank=%d, cond=%.2e', size(X_fsub,1), size(X_fsub,2), r_fsub, c_fsub))
xlabel('columns'); ylabel('rows (scans)');

% Save side-by-side image
outPng = fullfile(pwd, 'design_matrix_2x2_compare.png');
exportgraphics(fig, outPng, 'Resolution', 200);

% Also save matrices if needed
save('demo_designs_2x2.mat', 'X_manual', 'X_fsub', 'rows_subj', 'rows_time', 'subjGroup');

fprintf('Saved figure to %s\n', outPng);
