function selection = uiconfirm(varargin)
% UICONFIRM - Shadow function to suppress interactive confirmation dialogs
%
% Returns the default/first option without showing dialog

if nargin >= 3 && isfield(varargin{3}, 'Options')
    selection = varargin{3}.Options{1};
else
    selection = 'OK';
end

end
