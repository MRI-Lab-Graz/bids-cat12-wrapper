function cat12_threshold_maps(stats_dir, varargin)
% CAT12_THRESHOLD_MAPS - Threshold and transform SPM-maps using CAT12
%
% This function uses the CAT12 "Threshold and transform SPM-maps" tool to
% generate (log-scaled) p-maps or correlation maps from SPM.mat.
%
% Usage:
%   cat12_threshold_maps(stats_dir)
%   cat12_threshold_maps(stats_dir, 'p_unc', 0.001, 'p_fwe', 0.05, 'both', true)
%
% Inputs:
%   stats_dir - Path to SPM.mat directory
%   'p_unc'   - Uncorrected p-value threshold (default: 0.001)
%   'p_fwe'   - FWE corrected p-value threshold (default: 0.05)
%   'both'    - Both directions (default: true)
%   'log'     - Log-scaled p-maps (default: true)

    % Parse inputs
    p = inputParser;
    addRequired(p, 'stats_dir', @ischar);
    addParameter(p, 'p_unc', 0.001, @isnumeric);
    addParameter(p, 'p_fwe', 0.05, @isnumeric);
    addParameter(p, 'both', true, @islogical);
    addParameter(p, 'log', true, @islogical);
    parse(p, stats_dir, varargin{:});

    stats_dir = p.Results.stats_dir;
    p_unc = p.Results.p_unc;
    p_fwe = p.Results.p_fwe;
    both = p.Results.both;
    log_scaled = p.Results.log;

    fprintf('\n%s\n', repmat('═', 1, 80));
    fprintf('CAT12 DOUBLE THRESHOLDING\n');
    fprintf('%s\n\n', repmat('═', 1, 80));

    spm_file = fullfile(stats_dir, 'SPM.mat');
    if ~exist(spm_file, 'file')
        error('SPM.mat not found in %s', stats_dir);
    end

    fprintf('Settings:\n');
    fprintf('  Stats folder:     %s\n', stats_dir);
    fprintf('  p (uncorrected):  %.3f\n', p_unc);
    fprintf('  p (FWE):          %.3f\n', p_fwe);
    fprintf('  Both directions:  %d\n', both);
    fprintf('  Log-scaled:       %d\n\n', log_scaled);

    % Load SPM.mat to get T-maps
    load(spm_file);
    t_maps = {};
    for i = 1:length(SPM.xCon)
        % Check if it's a T-contrast
        if strcmp(SPM.xCon(i).STAT, 'T')
            t_maps{end+1} = fullfile(stats_dir, [SPM.xCon(i).Vspm.fname ',1']);
        end
    end
    
    if isempty(t_maps)
        error('No T-contrasts found in SPM.mat');
    end

    % Initialize matlabbatch
    matlabbatch{1}.spm.tools.cat.tools.T2x.data_T2x = t_maps';
    
    if log_scaled
        matlabbatch{1}.spm.tools.cat.tools.T2x.conversion.sel = 2; % Log-scaled p-maps
    else
        matlabbatch{1}.spm.tools.cat.tools.T2x.conversion.sel = 1; % p-maps
    end

    % Handle threshold
    % Note: CAT12 uses specific field names for common thresholds
    if p_unc == 0.001
        matlabbatch{1}.spm.tools.cat.tools.T2x.conversion.threshdesc.uncorr.thresh001 = 0.001;
    elseif p_unc == 0.01
        matlabbatch{1}.spm.tools.cat.tools.T2x.conversion.threshdesc.uncorr.thresh01 = 0.01;
    elseif p_unc == 0.05
        matlabbatch{1}.spm.tools.cat.tools.T2x.conversion.threshdesc.uncorr.thresh05 = 0.05;
    else
        fprintf('Warning: p_unc = %.4f not explicitly supported by this script version. Using 0.001.\n', p_unc);
        matlabbatch{1}.spm.tools.cat.tools.T2x.conversion.threshdesc.uncorr.thresh001 = 0.001;
    end

    matlabbatch{1}.spm.tools.cat.tools.T2x.conversion.inverse = 0;
    matlabbatch{1}.spm.tools.cat.tools.T2x.conversion.cluster.none = 1;
    matlabbatch{1}.spm.tools.cat.tools.T2x.atlas = 'None';

    % Run the batch
    spm('defaults', 'FMRI');
    spm_jobman('initcfg');
    try
        spm_jobman('run', matlabbatch);
        fprintf('\n✓ Double thresholding complete.\n');
    catch ME
        fprintf('\n❌ Double thresholding failed: %s\n', ME.message);
        rethrow(ME);
    end
end
