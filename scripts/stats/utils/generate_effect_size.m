function generate_effect_size(stats_dir)
% GENERATE_EFFECT_SIZE - Calculate Cohen's d maps from SPM T-maps
% d = t / sqrt(df_error)

    spm_file = fullfile(stats_dir, 'SPM.mat');
    if ~exist(spm_file, 'file')
        error('SPM.mat not found in %s', stats_dir);
    end
    
    load(spm_file);
    df = SPM.xX.erdf;
    
    fprintf('Generating Effect Size (Cohen''s d) maps for %d contrasts (df=%0.1f)...\n', length(SPM.xCon), df);
    
    for i = 1:length(SPM.xCon)
        if strcmp(SPM.xCon(i).STAT, 'T')
            t_file = fullfile(stats_dir, SPM.xCon(i).Vspm.fname);
            [path, name, ext] = fileparts(t_file);
            out_name = fullfile(path, sprintf('Cohen_d_%04d%s', i, ext));
            
            % Load T-map
            V = spm_vol(t_file);
            img = spm_read_vols(V);
            
            % Calculate d
            d_img = img / sqrt(df);
            
            % Write output
            V_out = V;
            V_out.fname = out_name;
            V_out.descrip = sprintf('Cohen''s d (t/sqrt(%0.1f))', df);
            spm_write_vol(V_out, d_img);
        end
    end
    fprintf('Done.\n');
end
