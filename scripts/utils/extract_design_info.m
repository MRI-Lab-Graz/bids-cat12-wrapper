% Extract design info from SPM.mat and save as JSON
% Usage: matlab_extract_design.m /path/to/stats_dir

function matlab_extract_design(stats_dir)
    if ~exist('stats_dir','var') || isempty(stats_dir)
        error('stats_dir required');
    end
    
    spm_file = fullfile(stats_dir, 'SPM.mat');
    if ~exist(spm_file, 'file')
        error('SPM.mat not found: %s', spm_file);
    end
    
    load(spm_file, 'SPM');
    
    % Extract parameter namesin cell array
    param_names = SPM.xX.name;
    
    % Write to JSON file
    json_file = fullfile(stats_dir, 'spm_design_info.json');
    
    % Create simple JSON output
    fid = fopen(json_file, 'w');
    
    fprintf(fid, '{\n');
    fprintf(fid, '  "parameters": [\n');
    
    for i = 1:length(param_names)
        if i > 1
            fprintf(fid, ',\n');
        end
        % Escape quotes in parameter name
        name = param_names{i};
        name = strrep(name, '"', '\"');
        fprintf(fid, '    "%s"', name);
    end
    
    fprintf(fid, '\n  ]\n');
    fprintf(fid, '}\n');
    
    fclose(fid);
    
    fprintf('✓ Design info saved to: %s\n', json_file);
    exit;
end
