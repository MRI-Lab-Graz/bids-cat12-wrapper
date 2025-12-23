function export_tfce_probe_metadata(stats_folder, out_json)
% EXPORT_TFCE_PROBE_METADATA  Emit per-contrast metadata for TFCE heuristics.
%
% This helper inspects SPM.xCon entries and writes a JSON file with
% per-contrast diagnostics (degrees of freedom, weight sparsity, etc.).
% The JSON is later consumed by the TFCE two-stage wrapper to make
% DF-aware decisions about permutation counts and nuisance handling.
%
% Usage:
%   export_tfce_probe_metadata('/path/to/stats');
%   export_tfce_probe_metadata('/path/to/stats','/tmp/meta.json');
%
% Inputs:
%   stats_folder - Folder containing SPM.mat
%   out_json     - Optional destination JSON file. Defaults to
%                  <stats_folder>/logs/tfce_contrast_metadata.json
%
% The JSON output is a list of structs with fields such as:
%   contrast, name, stat, error_df, design_rows, design_cols, design_rank,
%   nnz_weights, n_positive_weights, n_negative_weights, weight_min,
%   weight_max, df_num, contrast_rank.
%
% The function is intentionally lightweight (no toolboxes beyond SPM).

arguments
    stats_folder (1, :) char
    out_json (1, :) char = ""
end

spm_mat = fullfile(stats_folder, 'SPM.mat');
if ~exist(spm_mat, 'file')
    error('SPM.mat not found at %s', spm_mat);
end

load(spm_mat, 'SPM');

if isempty(out_json)
    logs_dir = fullfile(stats_folder, 'logs');
    if ~exist(logs_dir, 'dir')
        mkdir(logs_dir);
    end
    out_json = fullfile(logs_dir, 'tfce_contrast_metadata.json');
end

X = SPM.xX.X;
if issparse(X)
    X = full(X);
end

n_observations = size(X, 1);
n_regressors = size(X, 2);
error_df = SPM.xX.erdf;
try
    design_rank = rank(X);
catch ME
    warning('export_tfce_probe_metadata:rankFailed', ...
        'Could not compute design rank (%s). Using n_regressors.', ME.message);
    design_rank = n_regressors;
end

factor_summary = collect_factor_summary(SPM);

entries = cell(1, numel(SPM.xCon));
for idx = 1:numel(SPM.xCon)
    con = SPM.xCon(idx);
    entry = struct();
    entry.contrast = idx;
    entry.name = strtrim(char(con.name));
    entry.stat = detect_stat(con);
    entry.error_df = error_df;
    entry.design_rows = n_observations;
    entry.design_cols = n_regressors;
    entry.design_rank = design_rank;
    entry.factor_summary = factor_summary;

    weights = extract_weights(con);
    entry.nnz_weights = nnz(abs(weights) > 1e-6);
    entry.n_positive_weights = nnz(weights > 1e-6);
    entry.n_negative_weights = nnz(weights < -1e-6);
    if isempty(weights)
        entry.weight_min = 0;
        entry.weight_max = 0;
        entry.weight_sum = 0;
        entry.weight_abs_sum = 0;
        entry.contrast_rank = 0;
    else
        entry.weight_min = min(weights(:));
        entry.weight_max = max(weights(:));
        entry.weight_sum = sum(weights(:));
        entry.weight_abs_sum = sum(abs(weights(:)));
        entry.contrast_rank = safe_rank(weights);
    end

    entry.df_num = infer_numerator_df(con, entry.stat);
    entries{idx} = entry;
end

json_text = jsonencode(entries, 'PrettyPrint', true);
fid = fopen(out_json, 'w');
if fid == -1
    error('Cannot open %s for writing', out_json);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid, json_text, 'char');
fclose(fid);

fprintf('Saved TFCE contrast metadata to %s\n', out_json);
end

function summary = collect_factor_summary(SPM)
summary = struct('name', {}, 'levels', {});
if ~isfield(SPM, 'factor') || isempty(SPM.factor)
    return;
end
try
    for ii = 1:numel(SPM.factor)
        f = SPM.factor(ii);
        name = '<unnamed>';
        if isfield(f, 'name') && ~isempty(f.name)
            name = char(f.name);
        end
        levels = NaN;
        if isfield(f, 'levels') && ~isempty(f.levels)
            levels = double(f.levels);
        end
        summary(end + 1).name = strtrim(name); %#ok<AGROW>
        summary(end).levels = levels;
    end
catch ME
    warning('export_tfce_probe_metadata:factorSummary', ...
        'Could not collect factor summary: %s', ME.message);
end
end

function stat = detect_stat(con)
stat = 'T';
if isfield(con, 'STAT') && ~isempty(con.STAT)
    stat = upper(strtrim(con.STAT));
elseif isfield(con, 'stat') && ~isempty(con.stat)
    stat = upper(strtrim(con.stat));
end
if isempty(stat)
    stat = 'T';
end
end

function weights = extract_weights(con)
weights = [];
if isfield(con, 'c') && ~isempty(con.c)
    weights = double(con.c);
elseif isfield(con, 'weights') && ~isempty(con.weights)
    weights = double(con.weights);
elseif isfield(con, 'F') && ~isempty(con.F)
    weights = double(con.F);
end
end

function df_num = infer_numerator_df(con, stat)
df_num = 1;
try
    if strcmpi(stat, 'F')
        if isfield(con, 'c') && ~isempty(con.c)
            df_num = size(con.c, 1);
        elseif isfield(con, 'eidf') && ~isempty(con.eidf)
            df_num = double(con.eidf);
        elseif isfield(con, 'F') && ~isempty(con.F)
            df_num = size(con.F, 2);
        end
    end
catch
    df_num = 1;
end
df_num = max(1, df_num);
end

function r = safe_rank(matrix_values)
if isempty(matrix_values)
    r = 0;
    return;
end
try
    if issparse(matrix_values)
        r = rank(full(matrix_values));
    else
        r = rank(matrix_values);
    end
catch
    r = size(matrix_values, 2);
end
end
