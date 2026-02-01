function ButtonName = questdlg(varargin)
% QUESTDLG - Shadow function to suppress interactive dialog boxes
% 
% This shadow function intercepts all questdlg calls and returns
% a default answer without showing any dialog boxes. This enables
% fully headless/batch execution of SPM scripts.
%
% Usage:
%   Place this file in a directory that appears BEFORE MATLAB's
%   built-in uitools in the path (e.g., add to path at script start).
%
% Returns:
%   ButtonName - Always returns the first/default button option

% Default to first option (usually "Yes" or "OK")
if nargin >= 3
    % Return the first button option provided
    ButtonName = varargin{3};
else
    % Fallback to generic "Yes"
    ButtonName = 'Yes';
end

end
