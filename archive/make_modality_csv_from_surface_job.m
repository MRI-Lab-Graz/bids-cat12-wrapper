function out_csv = make_modality_csv_from_surface_job(job_file, modality, smoothPrefix, varargin)
% MAKE_MODALITY_CSV_FROM_SURFACE_JOB
% Build a CSV listing [subject, group, time, path] for a chosen modality and smoothing prefix
% using group/time assignments from an existing CAT12 surface factorial-design job.
%
% Inputs:
%   job_file     - path to a surface job (MATLAB .m that defines matlabbatch, or .mat with matlabbatch)
%   modality     - 'mri' (for volume) OR one of CAT12 surface metrics, e.g. 'thickness','gyrification',
%                  'fractaldimension','depth' (explicit surface name)
%   smoothPrefix - smoothing prefix string, e.g. 's6' (volume), 's15' (surface)
%
% Name/Value options:
%   'dataRoot'        - base path to CAT12 data (default: '/Volumes/Thunder/129_PK01/cat12/data/cat12')
%   'outCsv'          - output CSV path (default: fullfile(pwd, sprintf('design_%s_%s.csv', modality, smoothPrefix)))
%   'filenamePattern' - when modality='mri', optional extra pattern to narrow matches (e.g., '*mwp1*.nii*')
%   'requireComplete' - true/false (default true). If true, error if any subject/time is missing the modality file.
%   'preferExt'       - preferred extensions (cellstr), default {'.gii','.nii','.nii.gz'}
%
% Output:
%   out_csv      - path to the written CSV
%
% Notes:
%   - Group and time levels are taken from the job's fsuball.group(k).timepoint{t} structure.
%   - Time labels in the CSV are the 1..T indices of timepoints in the job (not the ses- label), ensuring consistency.
%   - For MRI volume mode, the script searches sub-<id>/mri/ for files starting with smoothPrefix (e.g., 's6*'). If multiple
%     matches exist, the first lexicographically sorted match is used unless 'filenamePattern' is specified to narrow it.
%   - For surface mode, the script searches sub-<id>/surf/ for files matching [smoothPrefix '.mesh.' modality '*.gii'] and
%     containing the expected session tag (e.g., '_ses-2_') inferred from time index.
%
% Example:
%   out = make_modality_csv_from_surface_job('template_surface_job.m', 'mri', 's6', ...
%       'dataRoot','/Volumes/Thunder/129_PK01/cat12/data/cat12', 'outCsv','design_mri_s6.csv', 'requireComplete',false);
%
% Author: Auto-generated helper
% Date: 2025-10-29

p = inputParser;
p.addRequired('job_file', @(s)ischar(s) || isstring(s));
p.addRequired('modality', @(s)ischar(s) || isstring(s));
p.addRequired('smoothPrefix', @(s)ischar(s) || isstring(s));
p.addParameter('dataRoot','/Volumes/Thunder/129_PK01/cat12/data/cat12', @(s)ischar(s) || isstring(s));
p.addParameter('outCsv','', @(s)ischar(s) || isstring(s));
p.addParameter('filenamePattern','', @(s)ischar(s) || isstring(s));
p.addParameter('requireComplete', true, @islogical);
p.addParameter('preferExt', {'.gii','.nii','.nii.gz'}, @(c)iscellstr(c) || (isstring(c) && isvector(c)));
p.parse(job_file, modality, smoothPrefix, varargin{:});

job_file = char(p.Results.job_file);
modality = char(lower(string(p.Results.modality)));
smoothPrefix = char(p.Results.smoothPrefix);
dataRoot = char(p.Results.dataRoot);
outCsv = char(p.Results.outCsv);
filenamePattern = char(p.Results.filenamePattern);
requireComplete = p.Results.requireComplete;
preferExt = cellstr(string(p.Results.preferExt));

if isempty(outCsv)
    outCsv = fullfile(pwd, sprintf('design_%s_%s.csv', modality, smoothPrefix));
end

if ~exist(job_file,'file')
    error('Job file not found: %s', job_file);
end

% Load matlabbatch from job file (.mat or .m)
clear matlabbatch;
[~,~,ext] = fileparts(job_file);
switch lower(ext)
    case '.mat'
        S = load(job_file);
        if isfield(S,'matlabbatch')
            matlabbatch = S.matlabbatch; %#ok<NASGU>
        else
            error('MAT file does not contain matlabbatch: %s', job_file);
        end
    otherwise
        run(job_file); % expects to populate matlabbatch in workspace
        if ~exist('matlabbatch','var')
            error('Job script did not create variable matlabbatch: %s', job_file);
        end
end

try
    fsuball = matlabbatch{1}.spm.tools.cat.factorial_design.des.fblock.fsuball;
catch
    error('Unexpected job structure. Cannot find fsuball under matlabbatch{1}.spm.tools.cat.factorial_design.des.fblock');
end

groups = fsuball.group;
G = numel(groups);
if G < 1
    error('No groups found in job.');
end

% Build records from group/time lists in job
records = struct('subject',{},'group',{},'time',{},'path',{});

for g = 1:G
    tp = groups(g).timepoint;
    if ~iscell(tp)
        error('Expected group(%d).timepoint to be a cell array of length T.', g);
    end
    T = numel(tp);
    for t = 1:T
        files_t = tp{t};
        if isempty(files_t); continue; end
        files_t = cellstr(files_t);
        for i = 1:numel(files_t)
            src = string(files_t{i});
            % Parse subject id and session id from the surface path
            subj = parse_token(src, 'sub-([A-Za-z0-9]+)');
            ses  = parse_token(src, 'ses-([0-9]+)');
            if subj == ""; warning('Could not parse subject from %s', files_t{i}); continue; end
            % Find target file for requested modality/smoothing
            target = find_modality_file(dataRoot, subj, ses, modality, smoothPrefix, filenamePattern, preferExt);
            if target == ""
                msg = sprintf('Missing %s %s for subject %s time %d (ses-%s)', modality, smoothPrefix, subj, t, ses);
                if requireComplete
                    error(msg);
                else
                    warning('%s', msg);
                    continue;
                end
            end
            rec.subject = char(subj);
            rec.group   = char(string(g)); % group label by index (1..G)
            rec.time    = char(string(t)); % time label by index (1..T)
            rec.path    = char(target);
            records(end+1) = rec; %#ok<AGROW>
        end
    end
end

% Convert to table and write CSV
if isempty(records)
    error('No records assembled. Nothing to write.');
end

T = struct2table(records);
% Ensure consistent column order
T = T(:,{'subject','group','time','path'});

% Deduplicate rows in case of accidental duplicates
T = unique(T,'rows');

% Sort by group, subject, time for readability
T = sortrows(T, {'group','subject','time'});

writetable(T, outCsv);
fprintf('Wrote %d rows to %s\n', height(T), outCsv);

out_csv = outCsv;

end % main function

function token = parse_token(str, pattern)
% Return the first capture group for the given pattern, or empty string
str = char(str);
tok = regexp(str, pattern, 'tokens', 'once');
if isempty(tok)
    token = "";
else
    token = string(tok{1});
end
end

function target = find_modality_file(dataRoot, subj, ses, modality, smoothPrefix, filenamePattern, preferExt)
% Find a single file path for the requested modality and smoothing for a given subject/session.
% Returns "" if not found.

subdir = fullfile(dataRoot, sprintf('sub-%s', subj));
if strcmp(modality, 'mri')
    mdir = fullfile(subdir, 'mri');
    if ~exist(mdir,'dir'), target = ""; return; end
    patt = [smoothPrefix '*'];
    if ~isempty(filenamePattern)
        % Allow user to narrow further, e.g., '*mwp1*.nii*'
        patt = [smoothPrefix filenamePattern];
    end
    dd = dir(fullfile(mdir, patt));
    % Filter by preferred extensions
    dd = dd(~[dd.isdir]);
    dd = dd(endsWith({dd.name}, preferExt));
    % If session is known, prefer filenames containing ses-<n>
    if ses ~= ""
        ses_tag = sprintf('_ses-%s', char(ses));
        dd_ses = dd(contains({dd.name}, ses_tag));
        if ~isempty(dd_ses); dd = dd_ses; end
    end
    if isempty(dd)
        target = ""; return;
    end
    names = sort({dd.name});
    target = string(fullfile(mdir, names{1}));
else
    % Surface modality: look under sub-*/surf/
    sdir = fullfile(subdir, 'surf');
    if ~exist(sdir,'dir'), target = ""; return; end
    patt = sprintf('%s.mesh.%s*.gii', smoothPrefix, modality);
    dd = dir(fullfile(sdir, patt));
    dd = dd(~[dd.isdir]);
    % Filter by session occurrence in filename if ses known
    if ses ~= ""
        ses_tag = sprintf('_ses-%s', char(ses));
        dd = dd(contains({dd.name}, ses_tag));
    end
    if isempty(dd)
        target = ""; return;
    end
    % Prefer filenames that include the subject id explicitly after the 'r'
    rid = ['rsub-' char(subj)];
    dd_pref = dd(contains({dd.name}, rid));
    if ~isempty(dd_pref); dd = dd_pref; end
    names = sort({dd.name});
    target = string(fullfile(sdir, names{1}));
end
end
