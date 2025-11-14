% startup.m - Temporary startup script for headless TFCE execution
% This patches SPM functions to be non-interactive

% Disable all GUI features immediately
set(0, 'DefaultFigureVisible', 'off');
set(0, 'DefaultUIControlVisible', 'off');
set(0, 'DefaultUITableVisible', 'off');
set(0, 'DefaultFigureNumberTitle', 'off');
set(0, 'DefaultFigureMenuBar', 'none');
set(0, 'DefaultFigureToolBar', 'none');
set(groot, 'defaultFigureVisible', 'off');
set(groot, 'defaultAxesVisible', 'off');

% Disable Java
try
    usejava('desktop', false);
    usejava('swing', false);
    usejava('awt', false);
catch
end

% Set headless environment variables
setenv('DISPLAY', '');
setenv('JAVA_TOOL_OPTIONS', '-Djava.awt.headless=true');
setenv('AWT_TOOLKIT', 'MToolkit');

% Close all figures
close all force hidden;

% Suppress warnings
warning('off', 'all');

% Override spm_input to never show dialogs
% We do this by creating a function in the current path that takes precedence
% But since we can't modify the function directly, we'll use onCleanup and preferences

% Disable any graphics rendering
set(0, 'DefaultFigureRenderer', 'painters');

% Force command-line mode for SPM
try
    spm('defaults', 'FMRI');
    global defaults;
    defaults.cmdline = 1;
    defaults.nogui = 1;
catch
end
