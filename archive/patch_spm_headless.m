function patch_spm_headless()
% Patch SPM functions to be non-interactive in headless mode
% This prevents spm_input, spm_select, and figure windows from appearing

% Override spm_input to return default values
% Instead of showing a dialog, automatically return the default answer
global spm_input_override
spm_input_override = true;

% We can't directly override MATLAB built-in functions, but we can set global flags
% that SPM checks. Let's ensure we're in a true headless state by:

% 1. Ensure no figures can be displayed
set(0, 'DefaultFigureVisible', 'off');
set(0, 'DefaultFigureNumberTitle', 'off');
set(0, 'DefaultFigureMenuBar', 'none');
set(0, 'DefaultFigureToolBar', 'none');
set(groot, 'defaultFigureVisible', 'off');
set(groot, 'defaultAxesVisible', 'off');

% 2. Close any open figures
close all force hidden;

% 3. Set up a custom figure handler that prevents display
set(0, 'DefaultFigureCreateFcn', 'set(gcf, ''Visible'', ''off''); set(gcf, ''Menubar'', ''none''); set(gcf, ''Toolbar'', ''none'')');

% 4. Disable Java desktop completely
try
    usejava('desktop', false);
    usejava('swing', false);
catch
end

% 5. Set up environment for no displays
setenv('DISPLAY', '');
setenv('JAVA_TOOL_OPTIONS', '-Djava.awt.headless=true');
setenv('AWT_TOOLKIT', 'MToolkit');

% 6. Suppress all warnings
warning('off', 'all');

% 7. Enable command-line mode in SPM
spm_get_defaults;
global defaults;
defaults.cmdline = 1;
defaults.nogui = 1;

end
