function result = input(prompt, varargin)
% SHADOW INPUT: Intercepts input() calls for headless MATLAB operation
% 
% This shadow function prevents interactive prompts from blocking
% headless batch execution. Instead of waiting for user input, it returns
% sensible defaults:
% - For 's' mode (string input): returns empty string or 'y' for yes/no prompts
% - For numeric mode: returns 0
%
% Usage:
%   result = input('Enter something: ')       -> returns ''
%   result = input('Continue? (y/n): ', 's')  -> returns 'y'
%   result = input('Enter number: ')          -> returns 0

% This function is only on the path in headless mode, so we always return defaults
% (We don't try to call the real input() because that would cause recursion issues)

if nargin > 1 && strcmpi(varargin{1}, 's')
    % String input mode
    prompt_lower = lower(prompt);
    
    % Check if this looks like a yes/no question
    if contains(prompt_lower, {'yes', 'no', '(y/n)', '(yes/no)', 'continue', 'save', 'future', 'configuration'})
        % For yes/no questions, return 'y' by default
        result = 'y';
    else
        % For other string inputs, return empty
        result = '';
    end
else
    % Numeric input mode - return 0 as default
    result = 0;
end

end
