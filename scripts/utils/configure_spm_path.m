function configure_spm_path()
% CONFIGURE_SPM_PATH - Helper script to configure SPM path for analysis scripts
% This script helps users set up their SPM path configuration for use with
% TFCE analysis scripts and other SPM-dependent analysis tools.
%
% Usage:
%   configure_spm_path()
%
% The script will:
%   1. Attempt to auto-detect SPM installation
%   2. Allow manual path specification
%   3. Validate the SPM installation
%   4. Save configuration for future use
%   5. Test the configuration
%
% Author: GitHub Copilot
% Date: 2025-10-22

fprintf('\n%s\n', repmat('═', 1, 70));
fprintf('SPM PATH CONFIGURATION TOOL\n');
fprintf('%s\n\n', repmat('═', 1, 70));

fprintf('This tool will help you configure the SPM path for analysis scripts.\n\n');

% Try auto-detection first
fprintf('🔍 STEP 1: AUTO-DETECTION\n');
fprintf('   ────────────────────────────────────────\n');

try
    auto_spm_path = find_spm_path();
    fprintf('   ✅ SUCCESS: SPM automatically detected!\n');
    fprintf('      Path: %s\n\n', auto_spm_path);
    
    % Test the detected path
    test_result = test_spm_configuration(auto_spm_path);
    if test_result
        fprintf('   ✅ Configuration test passed!\n\n');
        fprintf('🎉 SETUP COMPLETE!\n');
        fprintf('   Your SPM path is properly configured.\n');
        fprintf('   You can now run analysis scripts like run_screen_and_tfce.m\n\n');
        return;
    else
        fprintf('   ⚠️  Configuration test failed. Trying manual setup...\n\n');
    end
    
catch ME
    fprintf('   ⚠️  Auto-detection failed: %s\n\n', ME.message);
end

% Manual configuration
fprintf('🔧 STEP 2: MANUAL CONFIGURATION\n');
fprintf('   ────────────────────────────────────────\n');

fprintf('Please help us locate your SPM installation.\n');
fprintf('Common SPM installation directories:\n\n');

% Show common paths for current system
common_paths = get_system_specific_paths();
for i = 1:min(length(common_paths), 8)
    fprintf('   %d. %s\n', i, common_paths{i});
end
fprintf('\n');

% Interactive input
max_attempts = 5;
for attempt = 1:max_attempts
    fprintf('Attempt %d/%d:\n', attempt, max_attempts);
    
    % Show options
    fprintf('Options:\n');
    fprintf('   • Enter full path to SPM directory\n');
    fprintf('   • Type "browse" to use file browser (if available)\n');
    fprintf('   • Type "quit" to exit\n\n');
    
    user_input = input('Enter SPM path or option: ', 's');
    
    if strcmpi(user_input, 'quit')
        fprintf('\n❌ Configuration cancelled by user.\n\n');
        return;
    end
    
    if strcmpi(user_input, 'browse')
        try
            spm_path = uigetdir('', 'Select SPM Installation Directory');
            if spm_path == 0
                fprintf('   ⚠️  Directory selection cancelled.\n\n');
                continue;
            end
        catch
            fprintf('   ❌ File browser not available in this environment.\n');
            fprintf('      Please enter the path manually.\n\n');
            continue;
        end
    else
        spm_path = strtrim(user_input);
    end
    
    if isempty(spm_path)
        fprintf('   ⚠️  Empty input. Please provide a valid path.\n\n');
        continue;
    end
    
    % Expand ~ to home directory
    if startsWith(spm_path, '~')
        if ispc
            home_dir = [getenv('HOMEDRIVE') getenv('HOMEPATH')];
        else
            home_dir = getenv('HOME');
        end
        spm_path = strrep(spm_path, '~', home_dir);
    end
    
    fprintf('\n   🔍 Checking: %s\n', spm_path);
    
    % Validate path
    if ~isdir(spm_path)
        fprintf('   ❌ Directory does not exist.\n\n');
        continue;
    end
    
    % Check SPM installation
    if ~check_spm_installation(spm_path)
        fprintf('   ❌ Invalid SPM installation (missing critical files).\n\n');
        
        % Show what's missing
        missing_files = check_missing_spm_files(spm_path);
        if ~isempty(missing_files)
            fprintf('      Missing files:\n');
            for j = 1:length(missing_files)
                fprintf('         • %s\n', missing_files{j});
            end
            fprintf('\n');
        end
        continue;
    end
    
    % Valid SPM path found
    fprintf('   ✅ Valid SPM installation found!\n\n');
    
    % Test configuration
    fprintf('🧪 STEP 3: TESTING CONFIGURATION\n');
    fprintf('   ────────────────────────────────────────\n');
    
    test_result = test_spm_configuration(spm_path);
    
    if test_result
        fprintf('   ✅ All tests passed!\n\n');
        
        % Save configuration
        fprintf('💾 STEP 4: SAVING CONFIGURATION\n');
        fprintf('   ────────────────────────────────────────\n');
        
        save_config = input('   Save this configuration for future use? (y/n): ', 's');
        if strcmpi(save_config, 'y') || strcmpi(save_config, 'yes') || isempty(save_config)
            try
                config_file = fullfile(pwd, 'spm_config.txt');
                fid = fopen(config_file, 'w');
                fprintf(fid, '%s\n', spm_path);
                fclose(fid);
                fprintf('   ✅ Configuration saved to: %s\n', config_file);
                
                % Also suggest environment variable
                fprintf('\n   💡 TIP: You can also set a system environment variable:\n');
                if ispc
                    fprintf('      set SPM_PATH=%s\n', spm_path);
                else
                    fprintf('      export SPM_PATH="%s"\n', spm_path);
                end
                
            catch ME
                fprintf('   ⚠️  Could not save config file: %s\n', ME.message);
            end
        end
        
        fprintf('\n🎉 SETUP COMPLETE!\n');
        fprintf('   SPM path: %s\n', spm_path);
        fprintf('   You can now run analysis scripts!\n\n');
        return;
        
    else
        fprintf('   ❌ Configuration test failed.\n');
        fprintf('      SPM path is valid but there may be issues with the installation.\n\n');
    end
end

% If we get here, all attempts failed
fprintf('❌ CONFIGURATION FAILED\n');
fprintf('   Could not configure SPM path after %d attempts.\n\n', max_attempts);
fprintf('   Troubleshooting suggestions:\n');
fprintf('   • Verify SPM is properly installed\n');
fprintf('   • Check file permissions\n');
fprintf('   • Try running MATLAB as administrator (if on Windows)\n');
fprintf('   • Contact system administrator for help\n\n');

end

function paths = get_system_specific_paths()
% Get common SPM paths for the current operating system

if ismac
    paths = {
        '/Applications/spm25'
        '/Applications/spm24'
        '/usr/local/spm25'
        '/opt/spm25'
        '~/software/spm25'
        '~/Documents/MATLAB/spm25'
        '/Volumes/*/software/spm25'
    };
elseif isunix
    paths = {
        '/usr/local/spm25'
        '/opt/spm25'
        '~/software/spm25'
        '~/spm25'
        '/software/spm25'
        '~/Documents/MATLAB/spm25'
    };
elseif ispc
    paths = {
        'C:\Program Files\spm25'
        'C:\spm25'
        'C:\software\spm25'
        fullfile(getenv('USERPROFILE'), 'Documents', 'MATLAB', 'spm25')
        fullfile(getenv('USERPROFILE'), 'software', 'spm25')
    };
else
    paths = {'~/spm25', '~/software/spm25'};
end

% Expand ~ paths
for i = 1:length(paths)
    if startsWith(paths{i}, '~')
        if ispc
            home_dir = [getenv('HOMEDRIVE') getenv('HOMEPATH')];
        else
            home_dir = getenv('HOME');
        end
        paths{i} = strrep(paths{i}, '~', home_dir);
    end
end

end

function missing = check_missing_spm_files(spm_path)
% Check which critical SPM files are missing

critical_files = {
    'spm.m'
    'spm_get_defaults.m'
    'spm_vol.m'
    'spm_read_vols.m'
    'toolbox/TFCE/tfce_estimate_stat.m'
};

missing = {};
for i = 1:length(critical_files)
    file_path = fullfile(spm_path, critical_files{i});
    if ~exist(file_path, 'file')
        missing{end+1} = critical_files{i}; %#ok<AGROW>
    end
end

end

function success = test_spm_configuration(spm_path)
% Test the SPM configuration by trying to initialize SPM

success = false;

fprintf('   Testing SPM initialization... ');

try
    % Add SPM to path temporarily
    original_path = path;
    addpath(spm_path);
    
    % Try to initialize SPM
    old_warning = warning('off', 'all');
    
    % Test basic SPM functions
    spm_get_defaults;
    
    % Test if we can access SPM version
    spm_ver = spm('version');
    fprintf('✓ (SPM %s)\n', spm_ver);
    
    % Test TFCE toolbox
    fprintf('   Testing TFCE toolbox... ');
    tfce_path = fullfile(spm_path, 'toolbox', 'TFCE');
    if isdir(tfce_path)
        addpath(tfce_path);
        if exist('tfce_estimate_stat', 'file')
            fprintf('✓\n');
            success = true;
        else
            fprintf('❌ (tfce_estimate_stat not found)\n');
        end
    else
        fprintf('❌ (TFCE toolbox not found)\n');
    end
    
    warning(old_warning);
    
catch ME
    fprintf('❌ (%s)\n', ME.message);
end

% Restore original path
if exist('original_path', 'var')
    path(original_path);
end

end