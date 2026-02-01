function add_contrasts_longitudinal(stats_dir)
% ADD_CONTRASTS_LONGITUDINAL - Add dynamic contrasts for flexible factorial designs
%
% Automatically generates contrasts based on the actual design structure
% (any number of groups × any number of timepoints)
%
% Usage:
%   add_contrasts_longitudinal('/path/to/stats/folder')

if ~exist('stats_dir','var') || isempty(stats_dir)
    stats_dir = pwd;
end

% Convert to absolute path
if stats_dir(1) ~= '/'
    stats_dir = fullfile(pwd, stats_dir);
end

spm_file = fullfile(stats_dir, 'SPM.mat');
if ~exist(spm_file,'file')
    error('SPM.mat not found in %s', stats_dir);
end

fprintf('\n%s\n', repmat('═', 1, 80));
fprintf('Adding Dynamic Contrasts\n');
fprintf('%s\n\n', repmat('═', 1, 80));

% Load SPM.mat
load(spm_file, 'SPM');

% Extract design information
n_params = size(SPM.xX.X, 2);
fprintf('Design matrix: %d scans × %d parameters\n\n', size(SPM.xX.X, 1), n_params);

% Parse parameter names to extract groups and timepoints
% Expected format: Group*Time_{group,time}
param_names = SPM.xX.name;
fprintf('Parameters:\n');
for i = 1:length(param_names)
    fprintf('  %d: %s\n', i, param_names{i});
end
fprintf('\n');

% Extract unique groups and timepoints from parameter names
groups = {};
timepoints = [];

for i = 1:length(param_names)
    name = param_names{i};
    
    % Parse: Group*Time_{group,time}
    % Extract numbers using regex
    tokens = regexp(name, '{(\d+),(\d+)}', 'tokens');
    if ~isempty(tokens)
        group_idx = str2double(tokens{1}{1});
        time_idx = str2double(tokens{1}{2});
        
        % Store unique groups and timepoints
        if ~ismember(group_idx, str2double(groups))
            groups{end+1} = num2str(group_idx);
        end
        if ~ismember(time_idx, timepoints)
            timepoints = [timepoints, time_idx];
        end
    end
end

n_groups = length(groups);
n_times = length(timepoints);
timepoints = sort(timepoints);

fprintf('Detected: %d groups × %d timepoints\n', n_groups, n_times);
fprintf('Groups: %s\n', strjoin(groups, ', '));
fprintf('Timepoints: %s\n\n', sprintf('%d ', timepoints));

% Initialize contrasts
contrasts = {};
con_idx = 1;

% ============================================================================
% 1. WITHIN-GROUP TIME EFFECTS
% ============================================================================
fprintf('Adding within-group timepoint contrasts...\n');

for g = 1:n_groups
    group_str = num2str(g);
    
    % Linear contrast: Last timepoint vs First timepoint
    if n_times >= 2
        first_time = timepoints(1);
        last_time = timepoints(end);
        
        weights = zeros(1, n_params);
        for p = 1:n_params
            tokens = regexp(param_names{p}, '{(\d+),(\d+)}', 'tokens');
            if ~isempty(tokens)
                g_idx = str2double(tokens{1}{1});
                t_idx = str2double(tokens{1}{2});
                
                if g_idx == g
                    if t_idx == last_time
                        weights(p) = 1;
                    elseif t_idx == first_time
                        weights(p) = -1;
                    end
                end
            end
        end
        
        if any(weights ~= 0)
            contrasts{con_idx} = struct('name', sprintf('G%d: T%d - T%d', g, last_time, first_time), ...
                                        'weights', weights);
            con_idx = con_idx + 1;
        end
    end
    
    % All pairwise timepoint comparisons
    for t1 = 1:n_times
        for t2 = (t1+1):n_times
            time_1 = timepoints(t1);
            time_2 = timepoints(t2);
            
            weights = zeros(1, n_params);
            for p = 1:n_params
                tokens = regexp(param_names{p}, '{(\d+),(\d+)}', 'tokens');
                if ~isempty(tokens)
                    g_idx = str2double(tokens{1}{1});
                    t_idx = str2double(tokens{1}{2});
                    
                    if g_idx == g
                        if t_idx == time_2
                            weights(p) = 1;
                        elseif t_idx == time_1
                            weights(p) = -1;
                        end
                    end
                end
            end
            
            if any(weights ~= 0)
                contrasts{con_idx} = struct('name', sprintf('G%d: T%d - T%d', g, time_2, time_1), ...
                                            'weights', weights);
                con_idx = con_idx + 1;
            end
        end
    end
end

% ============================================================================
% 2. BETWEEN-GROUP COMPARISONS (at each timepoint)
% ============================================================================
fprintf('Adding between-group contrasts at each timepoint...\n');

for t = 1:n_times
    time_idx = timepoints(t);
    
    % Pairwise group comparisons
    for g1 = 1:n_groups
        for g2 = (g1+1):n_groups
            weights = zeros(1, n_params);
            
            for p = 1:n_params
                tokens = regexp(param_names{p}, '{(\d+),(\d+)}', 'tokens');
                if ~isempty(tokens)
                    g_idx = str2double(tokens{1}{1});
                    t_idx = str2double(tokens{1}{2});
                    
                    if t_idx == time_idx
                        if g_idx == g1
                            weights(p) = 1;
                        elseif g_idx == g2
                            weights(p) = -1;
                        end
                    end
                end
            end
            
            if any(weights ~= 0)
                contrasts{con_idx} = struct('name', sprintf('T%d: G%d - G%d', time_idx, g1, g2), ...
                                            'weights', weights);
                con_idx = con_idx + 1;
            end
        end
    end
end

% ============================================================================
% 3. INTERACTION CONTRASTS (Group × Time trajectories)
% ============================================================================
if n_groups >= 2 && n_times >= 2
    fprintf('Adding Group×Time interaction contrasts...\n');
    
    for g1 = 1:n_groups
        for g2 = (g1+1):n_groups
            % Trajectory difference: (G2_last - G2_first) - (G1_last - G1_first)
            first_time = timepoints(1);
            last_time = timepoints(end);
            
            weights = zeros(1, n_params);
            
            for p = 1:n_params
                tokens = regexp(param_names{p}, '{(\d+),(\d+)}', 'tokens');
                if ~isempty(tokens)
                    g_idx = str2double(tokens{1}{1});
                    t_idx = str2double(tokens{1}{2});
                    
                    if g_idx == g1
                        if t_idx == last_time
                            weights(p) = -1;
                        elseif t_idx == first_time
                            weights(p) = 1;
                        end
                    elseif g_idx == g2
                        if t_idx == last_time
                            weights(p) = 1;
                        elseif t_idx == first_time
                            weights(p) = -1;
                        end
                    end
                end
            end
            
            if any(weights ~= 0)
                contrasts{con_idx} = struct('name', sprintf('Interaction: (G%d vs G%d) trajectory', g1, g2), ...
                                            'weights', weights);
                con_idx = con_idx + 1;
                
                % Add inverse interaction contrast (G2 vs G1)
                contrasts{con_idx} = struct('name', sprintf('Interaction: (G%d vs G%d) trajectory', g2, g1), ...
                                            'weights', -weights);
                con_idx = con_idx + 1;
            end
        end
    end
end

% ============================================================================
% 4. OVERALL EFFECTS
% ============================================================================
fprintf('Adding overall effect contrasts...\n');

% Overall time effect (pooled across groups)
if n_times >= 2
    first_time = timepoints(1);
    last_time = timepoints(end);
    
    weights = zeros(1, n_params);
    for p = 1:n_params
        tokens = regexp(param_names{p}, '{(\d+),(\d+)}', 'tokens');
        if ~isempty(tokens)
            t_idx = str2double(tokens{1}{2});
            
            if t_idx == last_time
                weights(p) = 1;
            elseif t_idx == first_time
                weights(p) = -1;
            end
        end
    end
    
    if any(weights ~= 0)
        contrasts{con_idx} = struct('name', sprintf('Overall: T%d - T%d', last_time, first_time), ...
                                    'weights', weights);
        con_idx = con_idx + 1;
    end
end

% Overall group effect (pooled across timepoints)
if n_groups >= 2
    for g1 = 1:n_groups
        for g2 = (g1+1):n_groups
            weights = zeros(1, n_params);
            
            for p = 1:n_params
                tokens = regexp(param_names{p}, '{(\d+),(\d+)}', 'tokens');
                if ~isempty(tokens)
                    g_idx = str2double(tokens{1}{1});
                    
                    if g_idx == g1
                        weights(p) = 1;
                    elseif g_idx == g2
                        weights(p) = -1;
                    end
                end
            end
            
            if any(weights ~= 0)
                contrasts{con_idx} = struct('name', sprintf('Overall: G%d - G%d', g1, g2), ...
                                            'weights', weights);
                con_idx = con_idx + 1;
            end
        end
    end
end

% ============================================================================
% APPLY CONTRASTS TO SPM.mat
% ============================================================================

fprintf('\nApplying %d contrasts to SPM.mat...\n\n', numel(contrasts));

% Diagnostic: if no contrasts generated, write SPM parameter names and existing
% contrasts to a diagnostics file to aid debugging (saved next to SPM.mat).
if numel(contrasts) == 0
    diag_file = fullfile(fileparts(spm_file),'contrasts_diagnostics.txt');
    try
        fid = fopen(diag_file,'w');
        fprintf(fid,'No contrasts generated by add_contrasts_longitudinal on %s\n', datestr(now));
        fprintf(fid,'SPM.xX.name entries (%d):\n', numel(param_names));
        for i = 1:length(param_names)
            fprintf(fid,'  %03d: %s\n', i, param_names{i});
        end
        if isfield(SPM,'xCon') && ~isempty(SPM.xCon)
            fprintf(fid,'\nExisting SPM.xCon (%d):\n', length(SPM.xCon));
            for i = 1:length(SPM.xCon)
                try
                    fprintf(fid,'  %03d: %s\n', i, SPM.xCon(i).name);
                catch
                    fprintf(fid,'  %03d: <unable to read name>\n', i);
                end
            end
        else
            fprintf(fid,'\nNo existing SPM.xCon entries found.\n');
        end
        fclose(fid);
        fprintf('WARNING: No contrasts were auto-generated. Diagnostics written to: %s\n', diag_file);
    catch
        try, if exist('fid','var') && fid>0, fclose(fid); end; end
    end
end

%% Build matlabbatch
matlabbatch{1}.spm.stats.con.spmmat = {spm_file};

for k = 1:numel(contrasts)
    if isfield(contrasts{k}, 'ftest') && contrasts{k}.ftest
        % F-contrast
        matlabbatch{1}.spm.stats.con.consess{k}.fcon.name = contrasts{k}.name;
        matlabbatch{1}.spm.stats.con.consess{k}.fcon.weights = contrasts{k}.weights;
        matlabbatch{1}.spm.stats.con.consess{k}.fcon.sessrep = 'none';
    else
        % T-contrast
        matlabbatch{1}.spm.stats.con.consess{k}.tcon.name = contrasts{k}.name;
        matlabbatch{1}.spm.stats.con.consess{k}.tcon.weights = contrasts{k}.weights;
        matlabbatch{1}.spm.stats.con.consess{k}.tcon.sessrep = 'none';
    end
end

% By default, do not delete existing contrasts: use SPM's 'add' behaviour
% so newly generated contrasts are appended to any existing ones.
matlabbatch{1}.spm.stats.con.delete = 0;

spm('defaults','FMRI');
spm_jobman('initcfg');
spm_jobman('run', matlabbatch);

fprintf('\n✓ Successfully added %d contrasts.\n', numel(contrasts));
fprintf('Contrast list:\n');
for k = 1:numel(contrasts)
    fprintf('  %02d. %s\n', k, contrasts{k}.name);
end

% Save contrast list to JSON for reporting
json_file = fullfile(stats_dir, 'contrasts.json');
fid = fopen(json_file, 'w');
if fid > 0
    fprintf(fid, '[\n');
    for k = 1:numel(contrasts)
        % Escape quotes in name
        safe_name = strrep(contrasts{k}.name, '"', '\"');
        if k < numel(contrasts)
            fprintf(fid, '  {"index": %d, "name": "%s"},\n', k, safe_name);
        else
            fprintf(fid, '  {"index": %d, "name": "%s"}\n', k, safe_name);
        end
    end
    fprintf(fid, ']\n');
    fclose(fid);
    fprintf('Saved contrast list to: %s\n', json_file);
end

% IMPORTANT: spm_jobman may update and save SPM.mat as part of the job.
% Do NOT overwrite SPM.mat with the (potentially stale) local `SPM` variable
% that was loaded at the start of this function. Instead, reload the saved
% SPM.mat to report the up-to-date SPM.xCon contents (avoids clobbering
% newly-added contrasts written by SPM itself).
if exist(spm_file, 'file')
    try
        load(spm_file, 'SPM');
        if isfield(SPM, 'xCon')
            fprintf('\nSPM.mat now contains %d contrasts (SPM.xCon length)\n', length(SPM.xCon));
        else
            fprintf('\nSPM.mat does not contain SPM.xCon after running contrasts.\n');
        end
    catch ME
        fprintf('\nWarning: could not reload SPM.mat to verify contrasts: %s\n', ME.message);
    end
else
    fprintf('\nWarning: SPM.mat not found after running contrasts.\n');
end

end

