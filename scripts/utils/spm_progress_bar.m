function varargout = spm_progress_bar(varargin)
% SPM_PROGRESS_BAR (HEADLESS SHADOW)
% Suppresses progress bar figure creation in headless mode.
%
% In headless/batch mode with -nodisplay, figure windows cannot be created
% properly. This shadow function prevents SPM from trying to create and
% update progress bars, avoiding "Invalid or deleted object" errors.
%
% Syntax:
%   spm_progress_bar('Init', x, str, y)  - Initialize
%   spm_progress_bar('Set', x)            - Update
%   spm_progress_bar('Clear')             - Clear
%
% Returns empty/dummy values to satisfy SPM's calls without creating figures.

if nargin == 0
    varargout{1} = [];
    return;
end

action = lower(varargin{1});

switch action
    case 'init'
        % SPM calls this to initialize the progress bar
        % We return empty to indicate no progress bar was created
        varargout{1} = [];
        
    case 'set'
        % SPM calls this to update progress
        % Do nothing, return empty
        varargout{1} = [];
        
    case 'clear'
        % SPM calls this to close progress bar
        % Do nothing, return empty
        varargout{1} = [];
        
    otherwise
        % For any other action, return empty
        varargout{1} = [];
end

end
