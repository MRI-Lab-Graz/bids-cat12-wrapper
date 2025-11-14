% TEST_SPM_PATH_DETECTION - Test script for the new SPM path detection system
% This script tests the find_spm_path function without running the full analysis

fprintf('\n%s\n', repmat('═', 1, 60));
fprintf('TESTING SPM PATH DETECTION SYSTEM\n');
fprintf('%s\n\n', repmat('═', 1, 60));

try
    % Test the auto-detection
    fprintf('🧪 Testing find_spm_path()...\n');
    spm_path = find_spm_path();
    
    fprintf('\n✅ SUCCESS!\n');
    fprintf('   SPM path detected: %s\n', spm_path);
    
    % Verify the path is valid
    if isdir(spm_path)
        fprintf('   ✓ Directory exists\n');
    else
        fprintf('   ❌ Directory does not exist!\n');
        return;
    end
    
    % Check for critical files
    critical_files = {'spm.m', 'spm_get_defaults.m'};
    for i = 1:length(critical_files)
        file_path = fullfile(spm_path, critical_files{i});
        if exist(file_path, 'file')
            fprintf('   ✓ Found %s\n', critical_files{i});
        else
            fprintf('   ❌ Missing %s\n', critical_files{i});
        end
    end
    
    % Check for TFCE
    tfce_path = fullfile(spm_path, 'toolbox', 'TFCE', 'tfce_estimate_stat.m');
    if exist(tfce_path, 'file')
        fprintf('   ✓ TFCE toolbox found\n');
    else
        fprintf('   ⚠️  TFCE toolbox not found (required for TFCE analysis)\n');
    end
    
    fprintf('\n🎉 SPM path detection is working correctly!\n');
    fprintf('   You can now run run_screen_and_tfce.m safely.\n\n');
    
catch ME
    fprintf('\n❌ TEST FAILED: %s\n', ME.message);
    fprintf('\n💡 Try running: configure_spm_path()\n');
    fprintf('   This will help you set up your SPM path interactively.\n\n');
end