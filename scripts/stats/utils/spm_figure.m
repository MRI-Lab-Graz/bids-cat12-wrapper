function varargout = spm_figure(varargin)
% Shadow function for headless SPM figure creation
% This file must be in the MATLAB path BEFORE the real spm_figure
%
% Returns valid invisible figure handles for headless MATLAB operation.
% In headless mode (-nodisplay -nojvm), we can't show figures but we CAN
% create them as objects, so SPM's internal figure operations work.

% Return appropriate values based on the requested operation
if nargin == 0
    % No arguments - return empty
    varargout{1} = [];
    return;
end

action = varargin{1};

switch lower(action)
    case {'getwin', 'findwin'}
        % Create an invisible figure and return its handle
        % SPM uses this to draw progress bars, etc.
        try
            h = figure('Visible', 'off', 'NumberTitle', 'off', 'Name', 'SPM');
            varargout{1} = h;
        catch
            % If figure creation fails, return a minimal handle
            % (should rarely/never happen)
            varargout{1} = [];
        end
        
    case 'create'
        % Create and return invisible figure
        try
            h = figure('Visible', 'off', 'NumberTitle', 'off', 'Name', 'SPM');
            varargout{1} = h;
        catch
            varargout{1} = [];
        end
        
    case 'clear'
        % Close all SPM figures
        try
            close(findobj('Name','SPM'));
        catch
        end
        varargout{1} = [];
        
    case 'print'
        % Don't actually print, return empty
        varargout{1} = [];
        
    case 'focus'
        % Don't need to focus in headless, return empty
        varargout{1} = [];
        
    case {'watermark', 'colormap'}
        % Don't need these in headless, return empty
        varargout{1} = [];
        
    otherwise
        % For any other operation, try to create a figure
        try
            h = figure('Visible', 'off', 'NumberTitle', 'off', 'Name', 'SPM');
            varargout{1} = h;
        catch
            varargout{1} = [];
        end
end

% Ensure we always have at least one output
if nargout > 0 && isempty(varargout)
    varargout{1} = [];
end
