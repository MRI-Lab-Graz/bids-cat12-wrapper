function varargout = spm_input(varargin)
% spm_input (HEADLESS OVERRIDE)
% Always returns default values without showing any UI.
% Signature: spm_input(str,fig,type,choices,values,def)
%
% Handles prompts like "Overwrite existing SPM.mat?" by returning 'yes'
% to allow the analysis to proceed without user interaction.

% Default empty output
varargout = {[]};

% If a default (6th arg) is provided, return it
if nargin >= 6
    varargout{1} = varargin{6};
    return
end

% Check the prompt (1st arg) for keywords
if nargin >= 1 && (ischar(varargin{1}) || isstring(varargin{1}))
    prompt = lower(char(varargin{1}));
    
    % For "Overwrite?" prompts, answer 'yes' (1) to allow overwriting
    if contains(prompt, {'overwrite', 'replace', 'confirm'})
        varargout{1} = 1;
        return
    end
end

% If 'values' provided (5th arg) is [1 0], auto-answer 'yes' (1)
if nargin >= 5
    vals = varargin{5};
    if isnumeric(vals) && isequal(vals, [1 0])
        varargout{1} = 1;
        return
    end
end

% If 'choices' provided (4th arg) is a string list, pick the first
if nargin >= 4
    ch = varargin{4};
    if ischar(ch)
        % First token before '|'
        parts = strsplit(ch, '|');
        varargout{1} = strtrim(parts{1});
        return
    elseif iscellstr(ch) && ~isempty(ch)
        varargout{1} = ch{1};
        return
    end
end

% Fallback to a benign default
varargout{1} = [];

end
