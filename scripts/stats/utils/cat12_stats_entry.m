function cat12_stats_entry(cmd, varargin)
% CAT12_STATS_ENTRY Dispatcher for compiled standalone application
%
% This function serves as the entry point for a compiled standalone
% application. It takes a command string and dispatches it to the
% appropriate function in the stats/utils folder.
%
% Usage from shell:
%   ./cat12_stats_standalone <command> <arg1> <arg2> ...
%
% Examples:
%   ./cat12_stats_standalone add_contrasts /path/to/stats
%   ./cat12_stats_standalone run_tfce /path/to/stats n_perm 5000
%

    if nargin < 1
        error('Usage: cat12_stats_standalone <command> [args...]');
    end

    % Convert string arguments to numbers if possible
    % This is necessary because shell arguments are always strings
    args = varargin;
    for i = 1:length(args)
        val = args{i};
        if ischar(val) || isstring(val)
            % Try to convert to number
            num_val = str2double(val);
            if ~isnan(num_val)
                % Check if it really looks like a number (str2double is aggressive)
                % e.g. '1e-3' is a number. 'folder/1' is not (returns NaN).
                % But we need to be careful not to convert filenames that look like numbers
                % (unlikely for full paths, but possible for '1').
                % Heuristic: if it converts to a number, use the number.
                % Exception: if the argument is expected to be a string (like 'vbm'),
                % str2double returns NaN, so we are safe.
                args{i} = num_val;
            elseif strcmpi(val, 'true')
                args{i} = true;
            elseif strcmpi(val, 'false')
                args{i} = false;
            end
        end
    end

    fprintf('CAT12 Stats Standalone Dispatcher\n');
    fprintf('Command: %s\n', cmd);
    
    switch cmd
        case 'add_contrasts'
            % add_contrasts_longitudinal(stats_dir)
            if isempty(args)
                error('add_contrasts requires stats_dir argument');
            end
            add_contrasts_longitudinal(args{1});

        case 'threshold_maps'
            % cat12_threshold_maps(stats_dir, varargin)
            if isempty(args)
                error('threshold_maps requires stats_dir argument');
            end
            cat12_threshold_maps(args{1}, args{2:end});

        case 'screen_contrasts'
            % screen_contrasts(stats_dir, varargin)
            if isempty(args)
                error('screen_contrasts requires stats_dir argument');
            end
            screen_contrasts(args{1}, args{2:end});

        case 'run_tfce'
            % run_tfce_correction(stats_dir, varargin)
            if isempty(args)
                error('run_tfce requires stats_dir argument');
            end
            run_tfce_correction(args{1}, args{2:end});
            
        case 'configure_spm'
             % Just run the configuration (useful for testing)
             configure_spm_path();
             
        otherwise
            error('Unknown command: %s. Available: add_contrasts, screen_contrasts, run_tfce', cmd);
    end
end
