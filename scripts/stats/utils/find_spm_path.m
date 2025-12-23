function spm_path = find_spm_path()
% FIND_SPM_PATH - Automatically detect SPM installation path
% This function attempts to locate SPM in common installation directories
% across different operating systems and provides fallback options.
%
% Usage:
%   spm_path = find_spm_path()
%
% Returns:
%   spm_path - String containing the path to SPM installation
%
% Detection Strategy:
%   1. Check environment variable SPM_PATH
%   2. Check for spm_config.txt in current directory
%   3. Search common installation directories
%   4. Search MATLAB path for existing SPM
%   5. Interactive user input as fallback
%
% Author: GitHub Copilot
% Date: 2025-10-22

fprintf('\n🔍 AUTO-DETECTING SPM INSTALLATION PATH...\n');
fprintf('   ─────────────────────────────────────────────────\n');

spm_path = '';

% Method 1: Check environment variable
fprintf('   1. Checking environment variable SPM_PATH... ');
env_spm_path = getenv('SPM_PATH');
if ~isempty(env_spm_path) && isdir(env_spm_path)
    if check_spm_installation(env_spm_path)
        spm_path = env_spm_path;
        fprintf('✓ Found\n');
        fprintf('      → %s\n', spm_path);
        return;
    else
        fprintf('✗ Invalid (SPM files missing)\n');
    end
else
    fprintf('✗ Not set\n');
end

% Method 2: Check for local configuration file
fprintf('   2. Checking for spm_config.txt... ');
config_file = fullfile(pwd, 'spm_config.txt');
if exist(config_file, 'file')
    try
        fid = fopen(config_file, 'r');
        config_path = strtrim(fgetl(fid));
        fclose(fid);
        
        if isdir(config_path) && check_spm_installation(config_path)
            spm_path = config_path;
            fprintf('✓ Found\n');
            fprintf('      → %s\n', spm_path);
            return;
        else
            fprintf('✗ Invalid path in config\n');
        end
    catch
        fprintf('✗ Error reading config\n');
    end
else
    fprintf('✗ Not found\n');
end

% Method 3: Search common installation directories
fprintf('   3. Searching common directories... ');
common_paths = get_common_spm_paths();

for i = 1:length(common_paths)
    candidate_path = common_paths{i};
    if isdir(candidate_path) && check_spm_installation(candidate_path)
        spm_path = candidate_path;
        fprintf('✓ Found\n');
        fprintf('      → %s\n', spm_path);
        return;
    end
end
fprintf('✗ Not found in common locations\n');

% Method 4: Check if SPM is already in MATLAB path
fprintf('   4. Checking MATLAB path for existing SPM... ');
try
    smp_main = which('spm.m');
    if ~isempty(spm_main)
        % Extract directory path
        [spm_dir, ~, ~] = fileparts(spm_main);
        if check_spm_installation(spm_dir)
            spm_path = spm_dir;
            fprintf('✓ Found in MATLAB path\n');
            fprintf('      → %s\n', spm_path);
            return;
        end
    end
    fprintf('✗ Not in MATLAB path\n');
catch
    fprintf('✗ Error checking MATLAB path\n');
end

% Method 5: Interactive fallback
fprintf('   5. Interactive input (fallback)...\n\n');
fprintf('❌ SPM NOT FOUND AUTOMATICALLY\n\n');
fprintf('Please provide your SPM installation path manually.\n');
fprintf('Common locations might be:\n');
for i = 1:min(5, length(common_paths))
    fprintf('   • %s\n', common_paths{i});
end
fprintf('\n');

% Interactive input with validation
max_attempts = 3;
for attempt = 1:max_attempts
    user_path = input(sprintf('Enter SPM path (attempt %d/%d): ', attempt, max_attempts), 's');
    
    if isempty(user_path)
        fprintf('   ⚠️  Empty input. Please provide a valid path.\n\n');
        continue;
    end
    
    % Expand ~ to home directory
    if startsWith(user_path, '~')
        if ispc
            home_dir = [getenv('HOMEDRIVE') getenv('HOMEPATH')];
        else
            home_dir = getenv('HOME');
        end
        user_path = strrep(user_path, '~', home_dir);
    end
    
    if isdir(user_path)
        if check_spm_installation(user_path)
            spm_path = user_path;
            fprintf('   ✓ Valid SPM installation found!\n');
            
            % Offer to save for future use
            save_config = input('   Save this path for future use? (y/n): ', 's');
            if strcmpi(save_config, 'y') || strcmpi(save_config, 'yes')
                try
                    fid = fopen(config_file, 'w');
                    fprintf(fid, '%s\n', spm_path);
                    fclose(fid);
                    fprintf('   ✓ Saved to spm_config.txt\n');
                catch
                    fprintf('   ⚠️  Could not save config file\n');
                end
            end
            return;
        else
            fprintf('   ❌ Directory exists but SPM installation appears incomplete.\n');
            fprintf('      Missing critical files. Please check the installation.\n\n');
        end
    else
        fprintf('   ❌ Directory does not exist: %s\n\n', user_path);
    end
end

% If we get here, all attempts failed
error('SPM path detection failed after %d attempts. Please check your SPM installation.', max_attempts);

end

function is_valid = check_spm_installation(path_to_check)
% Check if the given path contains a valid SPM installation
% Returns true if all critical SPM files are found

is_valid = false;

if ~isdir(path_to_check)
    return;
end

% Critical files that must exist for a valid SPM installation
critical_files = {
    'spm.m'
    'spm_get_defaults.m' 
    'spm_vol.m'
    'spm_read_vols.m'
};

% Check for critical files
for i = 1:length(critical_files)
    file_path = fullfile(path_to_check, critical_files{i});
    if ~exist(file_path, 'file')
        return; % Missing critical file
    end
end

% Additional check for TFCE toolbox (recommended but not critical)
tfce_path = fullfile(path_to_check, 'toolbox', 'TFCE', 'tfce_estimate_stat.m');
has_tfce = exist(tfce_path, 'file');

% Valid SPM installation found
is_valid = true;

% Warn if TFCE is missing (since our script needs it)
if ~has_tfce
    fprintf('\n      ⚠️  TFCE toolbox not found in SPM installation.\n');
    fprintf('         TFCE is required for this analysis.\n');
    fprintf('         Please install TFCE toolbox in: %s\n', fullfile(path_to_check, 'toolbox', 'TFCE'));
end

end

function paths = get_common_spm_paths()
% Return list of common SPM installation paths for different operating systems

paths = {};

if ismac
    % macOS common paths
    paths = [paths, {
        '/Volumes/Evo/software/spm25'   % User's actual SPM path
        '/Applications/spm25'
        '/Applications/spm24' 
        '/Applications/spm23'
        '/Applications/SPM/spm25'
        '/Applications/SPM/spm24'
        '/usr/local/spm25'
        '/usr/local/spm24'
        '/opt/spm25'
        '/opt/spm24'
        '~/Applications/spm25'
        '~/Applications/spm24'
        '~/software/spm25'
        '~/software/spm24'
        '/Volumes/Evo/software/spm24'   % External drive variants
        '/Volumes/Thunder/software/spm25'
        '/Volumes/Thunder/software/spm24'
        fullfile(getenv('HOME'), 'Documents', 'MATLAB', 'spm25')
        fullfile(getenv('HOME'), 'Documents', 'MATLAB', 'spm24')
    }];
elseif isunix
    % Linux common paths
    paths = [paths, {
        '/usr/local/spm25'
        '/usr/local/spm24'
        '/opt/spm25'
        '/opt/spm24'
        '/home/*/software/spm25'
        '/home/*/software/spm24'
        '~/software/spm25'
        '~/software/spm24'
        '~/spm25'
        '~/spm24'
        '/software/spm25'
        '/software/spm24'
        fullfile(getenv('HOME'), 'Documents', 'MATLAB', 'spm25')
        fullfile(getenv('HOME'), 'Documents', 'MATLAB', 'spm24')
    }];
elseif ispc
    % Windows common paths
    paths = [paths, {
        'C:\Program Files\spm25'
        'C:\Program Files\spm24'
        'C:\Program Files (x86)\spm25'
        'C:\Program Files (x86)\spm24'
        'C:\spm25'
        'C:\spm24'
        'C:\software\spm25'
        'C:\software\spm24'
        fullfile(getenv('USERPROFILE'), 'Documents', 'MATLAB', 'spm25')
        fullfile(getenv('USERPROFILE'), 'Documents', 'MATLAB', 'spm24')
        fullfile(getenv('USERPROFILE'), 'software', 'spm25')
        fullfile(getenv('USERPROFILE'), 'software', 'spm24')
    }];
end

% Remove empty paths and expand ~
valid_paths = {};
for i = 1:length(paths)
    path = paths{i};
    if ~isempty(path)
        % Expand ~ to home directory
        if startsWith(path, '~')
            if ispc
                home_dir = [getenv('HOMEDRIVE') getenv('HOMEPATH')];
            else
                home_dir = getenv('HOME');
            end
            path = strrep(path, '~', home_dir);
        end
        valid_paths{end+1} = path;  %#ok<AGROW>
    end
end

paths = valid_paths;

end