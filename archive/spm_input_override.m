function varargout = spm_input_override(varargin)
% Override for spm_input - returns default values without showing GUI
% This prevents dialogs from appearing in headless mode

% Parse the input to find the default value
% spm_input(str,fig,type,choices,values,def)
%          ( 1 , 2  , 3    , 4      , 5     , 6 )

if nargin < 6
    % Not enough arguments - return empty or try to extract default
    varargout{1} = [];
    return;
end

default_value = varargin{6};

% For yes/no dialogs with [1,0] values, return 1 (yes - estimate the model)
if nargin >= 5
    values = varargin{5};
    if isequal(values, [1, 0])
        % This is likely the "estimate model" dialog - answer yes
        varargout{1} = 1;
        return;
    end
end

% For other cases, return the default value
varargout{1} = default_value;

end
