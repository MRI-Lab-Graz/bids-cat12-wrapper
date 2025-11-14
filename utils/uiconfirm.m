function selection = uiconfirm(varargin)
% UICONFIRM - Shadow function to suppress interactive confirmation dialogs
%
% Returns the default/first option without showing dialog.
% For "Overwrite?" type dialogs, returns "Yes" to allow proceeding.

% Look for keywords in the message (2nd arg) that suggest overwrite/confirm
if nargin >= 2 && (ischar(varargin{2}) || isstring(varargin{2}))
    msg = lower(char(varargin{2}));
    if contains(msg, {'overwrite', 'replace', 'confirm', 'already exists'})
        % For overwrite dialogs, return "Yes" (first option usually)
        if nargin >= 3 && isfield(varargin{3}, 'Options')
            opts = varargin{3}.Options;
            % Prefer "Yes" if available, otherwise first option
            for i = 1:length(opts)
                if strcmpi(opts{i}, 'yes')
                    selection = 'Yes';
                    return
                end
            end
            selection = opts{1};
        else
            selection = 'Yes';
        end
        return
    end
end

% Default: return first option if structure provided, else "OK"
if nargin >= 3 && isfield(varargin{3}, 'Options')
    selection = varargin{3}.Options{1};
else
    selection = 'OK';
end

end
