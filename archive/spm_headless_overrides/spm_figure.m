function F = spm_figure(varargin)
% spm_figure (HEADLESS OVERRIDE)
% Minimal no-op replacement to prevent SPM from opening any GUI windows.
% This shadow function is placed earlier on the MATLAB path than SPM's.
% It returns [] or 0 to indicate no figure should be used/created.
%
% Common SPM usages:
%   spm_figure('GetWin','Graphics'|'Interactive') -> return []
%   spm_figure('Create','Graphics'|'Interactive') -> return [] (don't create)
%   spm_figure('Clear','Graphics') -> no-op
%   spm_clf('Graphics') -> no-op (SPM sometimes calls via spm_figure)

F = [];

if nargin == 0
    return
end

cmd = varargin{1};
if ischar(cmd)
    switch lower(cmd)
        case {'getwin','create','clear','get'}
            % Always indicate there is no window in headless mode
            F = [];
        otherwise
            % Any other calls are treated as no-op
            F = [];
    end
else
    % Unexpected usage, still return empty
    F = [];
end

end
