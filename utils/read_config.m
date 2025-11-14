
function config = read_config(file_path)
% READ_CONFIG - Reads a simple INI file and returns a struct
%
% Usage:
%   config = read_config('config.ini');
%
% Input:
%   file_path - Path to the INI file
%
% Output:
%   config    - Struct with sections and key-value pairs

    config = struct();
    fid = fopen(file_path, 'r');
    if fid == -1
        error('Cannot open config file: %s', file_path);
    end
    
    current_section = '';
    
    while ~feof(fid)
        line = strtrim(fgetl(fid));
        
        % Skip empty lines and comments
        if isempty(line) || startsWith(line, '#') || startsWith(line, ';')
            continue;
        end
        
        % Section header
        if startsWith(line, '[') && endsWith(line, ']')
            current_section = lower(line(2:end-1));
            config.(current_section) = struct();
            continue;
        end
        
        % Key-value pair
        parts = split(line, '=');
        if length(parts) == 2
            key = strtrim(parts{1});
            value_str = strtrim(parts{2});
            
            % Convert to number if possible, otherwise keep as string
            [num, status] = str2num(value_str);
            if status
                value = num;
            else
                % Handle boolean strings
                if strcmpi(value_str, 'true')
                    value = true;
                elseif strcmpi(value_str, 'false')
                    value = false;
                else
                    value = value_str;
                end
            end
            
            if ~isempty(current_section)
                config.(current_section).(key) = value;
            end
        end
    end
    
    fclose(fid);
end
