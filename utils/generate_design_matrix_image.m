function img_path = generate_design_matrix_image(spm_mat_path, output_path)
% GENERATE_DESIGN_MATRIX_IMAGE - Create visual representation of design matrix
%
% Usage:
%   img_path = generate_design_matrix_image(spm_mat_path, output_path)
%
% Inputs:
%   spm_mat_path - Path to SPM.mat file
%   output_path  - Path for output PNG image
%
% Outputs:
%   img_path - Path to generated image
%
% Example:
%   generate_design_matrix_image('results/SPM.mat', 'results/design_matrix.png');

% Load SPM.mat
load(spm_mat_path, 'SPM');

% Extract design matrix
X = SPM.xX.X;
[n_scans, n_params] = size(X);

% Create figure (invisible for batch processing)
fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 800]);
set(fig, 'Color', 'white');

% Create subplot layout
subplot(2, 3, [1 2 4 5]);

% Plot design matrix as image
imagesc(X);
colormap(gray);
colorbar;

% Labels
xlabel('Parameters', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Scans', 'FontSize', 12, 'FontWeight', 'bold');
title('Design Matrix', 'FontSize', 14, 'FontWeight', 'bold');

% Add grid for better readability
hold on;
% Vertical lines to separate parameter groups
for i = 1:n_params
    line([i i]+0.5, [0.5 n_scans+0.5], 'Color', [0.3 0.3 0.3 0.3], 'LineWidth', 0.5);
end
hold off;

% Set axis properties
set(gca, 'XTick', 1:n_params);
set(gca, 'XTickLabel', SPM.xX.name, 'FontSize', 8);
xtickangle(90);
set(gca, 'TickLength', [0 0]);

% Plot parameter names separately for better readability
subplot(2, 3, [3 6]);
axis off;

% Create text labels for each parameter
y_start = 0.95;
y_step = 0.9 / n_params;

text(0.05, y_start, 'Parameters:', 'FontSize', 12, 'FontWeight', 'bold');

for i = 1:n_params
    % Color code by parameter type
    param_name = SPM.xX.name{i};
    
    if contains(param_name, 'Sn(')
        color = [0.2 0.4 0.8]; % Blue for main effects
    elseif contains(param_name, 'constant')
        color = [0.5 0.5 0.5]; % Gray for constants
    else
        color = [0.8 0.2 0.2]; % Red for other
    end
    
    y_pos = y_start - 0.05 - (i * y_step);
    text(0.05, y_pos, sprintf('%d. %s', i, param_name), ...
        'FontSize', 8, 'Color', color, 'Interpreter', 'none');
end

% Add design info
text(0.05, 0.05, sprintf('Scans: %d | Parameters: %d', n_scans, n_params), ...
    'FontSize', 10, 'FontWeight', 'bold');

% Save figure
print(fig, output_path, '-dpng', '-r300');
close(fig);

img_path = output_path;

fprintf('✓ Design matrix image saved: %s\n', output_path);

end
