function has_tfce_data = check_tfce_output_data(stats_folder, contrast_num, correction_type)
% CHECK_TFCE_OUTPUT_DATA - Check if TFCE output file contains actual data
% This function verifies that TFCE output files exist and have reasonable file sizes
% indicating they contain meaningful statistical data.
%
% Usage:
%   has_data = check_tfce_output_data(stats_folder, contrast_num, correction_type)
%
% Inputs:
%   stats_folder - Path to the statistics folder containing TFCE output
%   contrast_num - Contrast number (e.g., 2 for contrast 0002)
%   correction_type - 'FWE', 'FDR', or 'uncorrected'
%
% Returns:
%   has_tfce_data - true if file exists and has reasonable size (>10KB for surface data)
%
% Author: GitHub Copilot
% Date: 2025-10-22

has_tfce_data = false;

% Determine file pattern based on correction type
switch lower(correction_type)
    case 'fwe'
        file_pattern = sprintf('TFCE_log_pFWE_%04d', contrast_num);
    case 'fdr'
        file_pattern = sprintf('TFCE_log_pFDR_%04d', contrast_num);
    case 'uncorrected'
        file_pattern = sprintf('TFCE_log_p_%04d', contrast_num);
    otherwise
        warning('Unknown correction type: %s. Using uncorrected.', correction_type);
        file_pattern = sprintf('TFCE_log_p_%04d', contrast_num);
end

% Try .gii file first (surface data), then .nii (volume data), then .dat
file_extensions = {'.gii', '.nii', '.dat'};
tfce_file = '';

for i = 1:length(file_extensions)
    candidate_file = fullfile(stats_folder, [file_pattern file_extensions{i}]);
    if exist(candidate_file, 'file')
        tfce_file = candidate_file;
        break;
    end
end

if isempty(tfce_file)
    fprintf('   ⚠️  TFCE file not found: %s.*\n', file_pattern);
    return;
end

% Check file size - TFCE output files should be reasonably large if they contain data
file_info = dir(tfce_file);
file_size_kb = file_info.bytes / 1024;

% Minimum size thresholds (in KB)
if endsWith(tfce_file, '.gii')
    min_size_kb = 10;  % GIfTI files should be at least 10KB for surface data
elseif endsWith(tfce_file, '.nii')
    min_size_kb = 100;  % NIfTI files should be at least 100KB for volume data
else
    min_size_kb = 1;   % Other formats just need to be non-empty
end

if file_size_kb < min_size_kb
    fprintf('   ⚠️  TFCE file too small (%.1f KB < %.1f KB): %s\n', file_size_kb, min_size_kb, tfce_file);
    return;
end

% File exists and has reasonable size
has_tfce_data = true;
fprintf('   ✓ TFCE %s file valid (%.1f KB): %s\n', upper(correction_type), file_size_kb, tfce_file);

end