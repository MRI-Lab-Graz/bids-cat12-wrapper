function significant_contrasts = screen_contrasts(stats_folder, varargin)
% SCREEN_CONTRASTS - Screen contrasts for significant clusters (uncorrected)
%
% This function screens all contrasts in a statistical analysis folder
% for significant clusters using uncorrected thresholding. Only contrasts
% with significant clusters will be passed to TFCE correction.
%
% Usage:
%   significant_contrasts = screen_contrasts(stats_folder)
%   significant_contrasts = screen_contrasts(stats_folder, 'p_thresh', 0.001, 'cluster_size', 50)
%
% Inputs:
%   stats_folder  - Path to SPM.mat directory
%   'p_thresh'    - Uncorrected p-value threshold (default: 0.001)
%   'cluster_size' - Minimum cluster size in voxels/vertices (default: 50)
%
% Outputs:
%   significant_contrasts - Array of contrast numbers with significant clusters
%
% Example:
%   sig_cons = screen_contrasts('/path/to/stats/s9_int_control');
%   % Returns: [1 3 5 7 ...] (contrast indices with significant clusters)

% Parse inputs
p = inputParser;
addRequired(p, 'stats_folder', @ischar);
addParameter(p, 'p_thresh', 0.001, @isnumeric);
addParameter(p, 'cluster_size', 50, @isnumeric);
parse(p, stats_folder, varargin{:});

stats_folder = p.Results.stats_folder;
p_thresh = p.Results.p_thresh;
cluster_size = p.Results.cluster_size;

fprintf('\n%s\n', repmat('═', 1, 80));
fprintf('CONTRAST SCREENING - Uncorrected Cluster Thresholding\n');
fprintf('%s\n\n', repmat('═', 1, 80));

fprintf('Settings:\n');
fprintf('  Stats folder:     %s\n', stats_folder);
fprintf('  p-value:          p < %.3f (uncorrected)\n', p_thresh);
fprintf('  Cluster size:     k ≥ %d voxels/vertices\n\n', cluster_size);

% Check if SPM.mat exists
spm_mat_file = fullfile(stats_folder, 'SPM.mat');
if ~exist(spm_mat_file, 'file')
    error('SPM.mat not found in: %s', stats_folder);
end

% Load SPM.mat
fprintf('Loading SPM.mat...\n');
load(spm_mat_file, 'SPM');

n_contrasts = length(SPM.xCon);
fprintf('Found %d contrasts\n\n', n_contrasts);

% Initialize results
significant_contrasts = [];
screening_results = struct();

fprintf('%s\n', repmat('─', 1, 80));
fprintf('Screening contrasts:\n');
fprintf('%s\n\n', repmat('─', 1, 80));

% Determine file extension for statistic images (NIfTI vs GIfTI)
% Check for any GIfTI statistic files (spmT_*.gii or spmF_*.gii)
if ~isempty(dir(fullfile(stats_folder, 'spm*.gii')))
    stat_ext = '.gii';
else
    stat_ext = '.nii';
end

% Screen each contrast
for con_idx = 1:n_contrasts
    con_name = SPM.xCon(con_idx).name;
    fprintf('[%2d/%2d] %s\n', con_idx, n_contrasts, con_name);
    
    % Determine if this is a T or F contrast using the STAT field when
    % available. Some SPM versions set an F field or a STAT field; STAT is
    % the reliable indicator ('T' or 'F').
    if isfield(SPM.xCon(con_idx), 'STAT') && ~isempty(SPM.xCon(con_idx).STAT)
        is_f_contrast = strcmpi(SPM.xCon(con_idx).STAT, 'F');
    else
        % Fallback: older SPM variants may store an F matrix - treat that
        % as an indicator of an F-contrast if present and non-empty.
        is_f_contrast = false;
        if isfield(SPM.xCon(con_idx), 'F')
            try
                is_f_contrast = ~isempty(SPM.xCon(con_idx).F);
            catch
                is_f_contrast = false;
            end
        end
    end
    
    if is_f_contrast
        % F-contrast - use spmF file
        stat_file = fullfile(stats_folder, sprintf('spmF_%04d%s', con_idx, stat_ext));
        stat_type = 'F';
    else
        % T-contrast - use spmT file
        stat_file = fullfile(stats_folder, sprintf('spmT_%04d%s', con_idx, stat_ext));
        stat_type = 'T';
    end
    
    if ~exist(stat_file, 'file')
        fprintf('        ⚠ spm%s_%04d%s not found, skipping\n\n', stat_type, con_idx, stat_ext);
        continue;
    end
    
    % Load statistic map
    if strcmpi(stat_ext, '.gii')
        % GIfTI handling
        try
            fprintf('        Loading GIfTI: %s\n', stat_file);
            g = gifti(stat_file);
            % Ensure Y is a numeric array, not a file_array
            Y = double(g.cdata);
            fprintf('        Loaded GIfTI data. Size: %s, Class: %s\n', mat2str(size(Y)), class(Y));
            % For surfaces, Y is typically N x 1
        catch ME
            fprintf('        ⚠ Error loading GIfTI: %s\n', ME.message);
            continue;
        end
    else
        % NIfTI handling
        V = spm_vol(stat_file);
        % Log header and dimension info to help debug orientation/affine issues
        try
            debug_file = fullfile(stats_folder, 'screening_header_debug.txt');
            fid = fopen(debug_file, 'a');
            fprintf(fid, 'Contrast %d (%s) - %s\n', con_idx, con_name, datestr(now));
            fprintf(fid, '  stat_file: %s\n', stat_file);
            fprintf(fid, '  dims: [%d %d %d]\n', V.dim(1), V.dim(2), V.dim(3));
            % print datatype (V.dt can be a pair)
            if isfield(V, 'dt') && ~isempty(V.dt)
                try
                    dt_str = mat2str(V.dt);
                catch
                    dt_str = '<unknown dt>';
                end
            else
                dt_str = '<no dt>'; 
            end
            fprintf(fid, '  dt: %s\n', dt_str);
            fprintf(fid, '  mat:\n');
            for r = 1:size(V.mat,1)
                fprintf(fid, '    %s\n', mat2str(V.mat(r,:)));
            end
            fprintf(fid, '\n');
            fclose(fid);
        catch
            % If logging fails, don't crash screening - just continue
            try
                if exist('fid','var') && fid > 0, fclose(fid); end
            catch, end
        end

        [Y, XYZ] = spm_read_vols(V);
    end
    
    % Get degrees of freedom and convert to threshold
    fprintf('        Calculating threshold (Type: %s)...\n', stat_type);
    if is_f_contrast
        % For F-contrasts, compute numerator and denominator degrees of
        % freedom defensively. Some SPM versions store the F-contrast
        % information under different subfields; avoid direct indexing
        % that can raise an error. If numerator df cannot be determined
        % reliably, fall back to 1 (conservative).
        df_den = SPM.xX.erdf;  % Denominator df (residual df)
        df_num = 1;
        if isfield(SPM.xCon(con_idx), 'STAT') && strcmpi(SPM.xCon(con_idx).STAT, 'F')
            % Try to infer numerator df from the contrast definition
            % Commonly, SPM.xCon(con).c is a matrix for F-contrasts
            if isfield(SPM.xCon(con_idx), 'c') && ~isempty(SPM.xCon(con_idx).c)
                try
                    cmat = SPM.xCon(con_idx).c;
                    if ismatrix(cmat)
                        df_num = size(cmat, 1);
                    end
                catch
                    df_num = 1;
                end
            end
        else
            % Fallback: if an F field exists and is non-empty, try to use its size
            if isfield(SPM.xCon(con_idx), 'F') && ~isempty(SPM.xCon(con_idx).F)
                try
                    df_num = size(SPM.xCon(con_idx).F, 2);
                catch
                    df_num = 1;
                end
            end
        end

        fprintf('        F-contrast: df_num=%d, df_den=%f\n', df_num, df_den);

        % Compute F threshold and threshold the map
        try
            f_thresh = finv(1 - p_thresh, df_num, df_den);
            fprintf('        F-threshold: %f\n', f_thresh);
        catch ME
            fprintf('        ⚠ Error in finv: %s\n', ME.message);
            % If finv fails for any reason, set a very high threshold so no voxels pass
            f_thresh = Inf;
        end
        Y_thresh = Y;
        Y_thresh(Y_thresh < f_thresh) = 0;
    else
        % For T-contrasts, use T-distribution
        df = SPM.xX.erdf;
        fprintf('        T-contrast: df=%f\n', df);
        t_thresh = spm_invTcdf(1 - p_thresh, df);
        fprintf('        T-threshold: %f\n', t_thresh);
        Y_thresh = Y;
        Y_thresh(abs(Y_thresh) < t_thresh) = 0;
    end
    
    fprintf('        Thresholding complete.\n');
    
    % Find clusters using connected components
    if any(Y_thresh(:) ~= 0)
        % Create binary mask of significant voxels
        sig_mask = Y_thresh > 0;
        
        if strcmpi(stat_ext, '.gii')
            % For surfaces, we can't easily do topological clustering without the mesh.
            % As a screening proxy, we just count the total number of significant vertices.
            % If total_sig_vertices >= cluster_size, we pass it.
            n_sig_vertices = sum(sig_mask(:));
            
            if n_sig_vertices >= cluster_size
                n_sig_clusters = 1; % Treat as "at least one cluster"
                significant_contrasts = [significant_contrasts, con_idx];
                
                fprintf('        ✓ SIGNIFICANT: %d significant vertices (≥ %d)\n', ...
                        n_sig_vertices, cluster_size);
                
                screening_results(con_idx).name = con_name;
                screening_results(con_idx).n_clusters = 1; % Dummy value
                screening_results(con_idx).max_t = max(abs(Y(:)));
                screening_results(con_idx).significant = true;
            else
                fprintf('        ○ %d significant vertices (below threshold %d)\n', ...
                        n_sig_vertices, cluster_size);
                screening_results(con_idx).name = con_name;
                screening_results(con_idx).n_clusters = 0;
                screening_results(con_idx).max_t = max(abs(Y(:)));
                screening_results(con_idx).significant = false;
            end
        else
            % Use spm_bwlabel for 3D connectivity analysis (NIfTI) - Standalone compatible
            try
                % Find connected components (6-connectivity for 3D)
                [L, num] = spm_bwlabel(double(sig_mask), 6);
                
                % Get cluster sizes
                if num > 0
                    % Count voxels per label
                    cluster_sizes = zeros(1, num);
                    for i = 1:num
                        cluster_sizes(i) = sum(L(:) == i);
                    end
                else
                    cluster_sizes = [];
                end
                
                % Find clusters above size threshold
                large_clusters = cluster_sizes(cluster_sizes >= cluster_size);
                n_sig_clusters = length(large_clusters);
                
                if n_sig_clusters > 0
                    % This contrast has significant clusters
                    significant_contrasts = [significant_contrasts, con_idx];
                    
                    fprintf('        ✓ SIGNIFICANT: %d cluster(s) ≥ %d voxels\n', ...
                            n_sig_clusters, cluster_size);
                    
                    % Store detailed results
                    screening_results(con_idx).name = con_name;
                    screening_results(con_idx).n_clusters = n_sig_clusters;
                    screening_results(con_idx).max_t = max(abs(Y(:)));
                    screening_results(con_idx).significant = true;
                else
                    fprintf('        ○ No clusters ≥ %d voxels\n', cluster_size);
                    screening_results(con_idx).name = con_name;
                    screening_results(con_idx).n_clusters = 0;
                    screening_results(con_idx).max_t = max(abs(Y(:)));
                    screening_results(con_idx).significant = false;
                end
            catch ME
                fprintf('        ⚠ Error in cluster analysis: %s\n', ME.message);
                fprintf('        ○ Could not analyze clusters\n');
                screening_results(con_idx).name = con_name;
                screening_results(con_idx).n_clusters = 0;
                screening_results(con_idx).max_t = max(abs(Y(:)));
                screening_results(con_idx).significant = false;
            end
        end
    else
        fprintf('        ○ No voxels above threshold\n');
        screening_results(con_idx).name = con_name;
        screening_results(con_idx).n_clusters = 0;
        screening_results(con_idx).max_t = max(abs(Y(:)));
        screening_results(con_idx).significant = false;
    end
    
    fprintf('\n');
end

% Summary
fprintf('%s\n', repmat('═', 1, 80));
fprintf('SCREENING SUMMARY\n');
fprintf('%s\n\n', repmat('═', 1, 80));

fprintf('Total contrasts:        %d\n', n_contrasts);
fprintf('Significant contrasts:  %d\n', length(significant_contrasts));
fprintf('Pass rate:              %.1f%%\n\n', ...
        100 * length(significant_contrasts) / n_contrasts);

if ~isempty(significant_contrasts)
    fprintf('Significant contrast indices:\n');
    fprintf('  ');
    for i = 1:length(significant_contrasts)
        fprintf('%d ', significant_contrasts(i));
        if mod(i, 15) == 0
            fprintf('\n  ');
        end
    end
    fprintf('\n\n');
    
    fprintf('These contrasts will proceed to TFCE correction.\n');
else
    fprintf('⚠ No contrasts passed screening!\n');
    fprintf('  Consider:\n');
    fprintf('    - Lowering p-threshold (current: p < %.3f)\n', p_thresh);
    fprintf('    - Reducing cluster size (current: k ≥ %d)\n', cluster_size);
    fprintf('    - Checking data quality\n');
end

fprintf('%s\n', repmat('═', 1, 80));

% Save results
output_file = fullfile(stats_folder, 'screening_results.mat');
save(output_file, 'significant_contrasts', 'screening_results', ...
     'p_thresh', 'cluster_size');
fprintf('\n✓ Results saved to: %s\n\n', output_file);

end
