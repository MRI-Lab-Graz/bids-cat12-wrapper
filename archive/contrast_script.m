%==========================================================================
% Define all valid contrasts for 3×3 longitudinal model with subject effects
%==========================================================================
% Author: ChatGPT
% Date: 2025-10-15
%==========================================================================
feature('DefaultCharacterSet', 'UTF-8');

spm_path = '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control';  % <-- CHANGE this to your actual path

% -------------------------------------------------------------------------
% Constants
% -------------------------------------------------------------------------
nRegressors = 9;     % 3 groups × 3 timepoints
offset = 0;          % task regressors start at column 1
nTotal = offset + nRegressors;

% -------------------------------------------------------------------------
% Initialize batch
% -------------------------------------------------------------------------
clear matlabbatch
matlabbatch{1}.spm.stats.con.spmmat = {spm_path};

% Helper: create empty contrast vector
% orig zero_c = @(v) [v, zeros(1, nTotal - length(v))];
zero_c = @(v) [v, zeros(size(v,1), nTotal - size(v,2))];

% -------------------------------------------------------------------------
% G1 Linear time effects (T-contrasts)
% -------------------------------------------------------------------------

% G1 Time3 - Time1
c = zero_c([-1 0 1]);
matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = 'G1: Time3 - Time1';
matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = c;
matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

% G1 Time1 - Time3
c = zero_c([1 0 -1]);
matlabbatch{1}.spm.stats.con.consess{2}.tcon.name = 'G1: Time1 - Time3';
matlabbatch{1}.spm.stats.con.consess{2}.tcon.weights = c;
matlabbatch{1}.spm.stats.con.consess{2}.tcon.sessrep = 'none';

% G1 Time3 - Time2
c = zero_c([0 -1 1]);
matlabbatch{1}.spm.stats.con.consess{3}.tcon.name = 'G1: Time3 - Time2';
matlabbatch{1}.spm.stats.con.consess{3}.tcon.weights = c;
matlabbatch{1}.spm.stats.con.consess{3}.tcon.sessrep = 'none';

% G1 Time2 - Time3
c = zero_c([0 1 -1]);
matlabbatch{1}.spm.stats.con.consess{4}.tcon.name = 'G1: Time2 - Time3';
matlabbatch{1}.spm.stats.con.consess{4}.tcon.weights = c;
matlabbatch{1}.spm.stats.con.consess{4}.tcon.sessrep = 'none';

% G1 Time3 - Time2
c = zero_c([-1 1 0]);
matlabbatch{1}.spm.stats.con.consess{5}.tcon.name = 'G1: Time2 - Time1';
matlabbatch{1}.spm.stats.con.consess{5}.tcon.weights = c;
matlabbatch{1}.spm.stats.con.consess{5}.tcon.sessrep = 'none';

% G1 Time2 - Time3
c = zero_c([1 -1 0]);
matlabbatch{1}.spm.stats.con.consess{6}.tcon.name = 'G1: Time1 - Time2';
matlabbatch{1}.spm.stats.con.consess{6}.tcon.weights = c;
matlabbatch{1}.spm.stats.con.consess{6}.tcon.sessrep = 'none';

% -------------------------------------------------------------------------
% G2 Linear time effects (T-contrasts)
% -------------------------------------------------------------------------

% G2 Time3 - Time1
c = zero_c([0 0 0 -1 0 1]);
matlabbatch{1}.spm.stats.con.consess{7}.tcon.name = 'G2: Time3 - Time1';
matlabbatch{1}.spm.stats.con.consess{7}.tcon.weights = c;
matlabbatch{1}.spm.stats.con.consess{7}.tcon.sessrep = 'none';

% G2 Time1 - Time3
c = zero_c([0 0 0 1 0 -1]);
matlabbatch{1}.spm.stats.con.consess{8}.tcon.name = 'G2: Time1 - Time3';
matlabbatch{1}.spm.stats.con.consess{8}.tcon.weights = c;
matlabbatch{1}.spm.stats.con.consess{8}.tcon.sessrep = 'none';

% G2 Time3 - Time2
c = zero_c([0 0 0  0 -1 1]);
matlabbatch{1}.spm.stats.con.consess{9}.tcon.name = 'G2: Time3 - Time2';
matlabbatch{1}.spm.stats.con.consess{9}.tcon.weights = c;
matlabbatch{1}.spm.stats.con.consess{9}.tcon.sessrep = 'none';

% G2 Time2 - Time3
c = zero_c([0 0 0 0 1 -1]);
matlabbatch{1}.spm.stats.con.consess{10}.tcon.name = 'G2: Time2 - Time3';
matlabbatch{1}.spm.stats.con.consess{10}.tcon.weights = c;
matlabbatch{1}.spm.stats.con.consess{10}.tcon.sessrep = 'none';

% G2 Time3 - Time2
c = zero_c([0 0 0  -1 1 0]);
matlabbatch{1}.spm.stats.con.consess{11}.tcon.name = 'G2: Time2 - Time1';
matlabbatch{1}.spm.stats.con.consess{11}.tcon.weights = c;
matlabbatch{1}.spm.stats.con.consess{11}.tcon.sessrep = 'none';

% G2 Time2 - Time3
c = zero_c([0 0 0  1 -1 0]);
matlabbatch{1}.spm.stats.con.consess{12}.tcon.name = 'G2: Time1 - Time2';
matlabbatch{1}.spm.stats.con.consess{12}.tcon.weights = c;
matlabbatch{1}.spm.stats.con.consess{12}.tcon.sessrep = 'none';

% -------------------------------------------------------------------------
% G3 Linear time effects (T-contrasts)
% -------------------------------------------------------------------------

% G3 Time3 - Time1
c = zero_c([0 0 0 0 0 0 -1 0 1]);
matlabbatch{1}.spm.stats.con.consess{13}.tcon.name = 'G3: Time3 - Time1';
matlabbatch{1}.spm.stats.con.consess{13}.tcon.weights = c;
matlabbatch{1}.spm.stats.con.consess{13}.tcon.sessrep = 'none';

% G3 Time1 - Time3
c = zero_c([0 0 0 0 0 0 1 0 -1]);
matlabbatch{1}.spm.stats.con.consess{14}.tcon.name = 'G3: Time1 - Time3';
matlabbatch{1}.spm.stats.con.consess{14}.tcon.weights = c;
matlabbatch{1}.spm.stats.con.consess{14}.tcon.sessrep = 'none';

% G3 Time3 - Time2
c = zero_c([0 -1 1]);
matlabbatch{1}.spm.stats.con.consess{15}.tcon.name = 'G1: Time3 - Time2';
matlabbatch{1}.spm.stats.con.consess{15}.tcon.weights = c;
matlabbatch{1}.spm.stats.con.consess{15}.tcon.sessrep = 'none';

% G3 Time2 - Time3
c = zero_c([0 0 0 0 0 0 0 1 -1]);
matlabbatch{1}.spm.stats.con.consess{16}.tcon.name = 'G3: Time2 - Time3';
matlabbatch{1}.spm.stats.con.consess{16}.tcon.weights = c;
matlabbatch{1}.spm.stats.con.consess{16}.tcon.sessrep = 'none';

% G3 Time2 - Time2
c = zero_c([0 0 0 0 0 0 -1 1 0]);
matlabbatch{1}.spm.stats.con.consess{17}.tcon.name = 'G3: Time2 - Time1';
matlabbatch{1}.spm.stats.con.consess{17}.tcon.weights = c;
matlabbatch{1}.spm.stats.con.consess{17}.tcon.sessrep = 'none';

% G3 Time2 - Time1
c = zero_c([0 0 0 0 0 0 1 -1 0]);
matlabbatch{1}.spm.stats.con.consess{18}.tcon.name = 'G3: Time1 - Time2';
matlabbatch{1}.spm.stats.con.consess{18}.tcon.weights = c;
matlabbatch{1}.spm.stats.con.consess{18}.tcon.sessrep = 'none';

% -------------------------------------------------------------------------
% 2. Group × Time interaction contrasts (T)
% -------------------------------------------------------------------------

% G1 vs G2
c = zero_c([-1 0 1  1 0 -1  0 0 0]);
matlabbatch{1}.spm.stats.con.consess{19}.tcon.name = 'Interaction: G1 vs G2 (Time)';
matlabbatch{1}.spm.stats.con.consess{19}.tcon.weights = c;
matlabbatch{1}.spm.stats.con.consess{19}.tcon.sessrep = 'none';

% G1 vs G3
c = zero_c([-1 0 1  0 0 0  1 0 -1]);
matlabbatch{1}.spm.stats.con.consess{20}.tcon.name = 'Interaction: G1 vs G3 (Time)';
matlabbatch{1}.spm.stats.con.consess{20}.tcon.weights = c;
matlabbatch{1}.spm.stats.con.consess{20}.tcon.sessrep = 'none';

% G2 vs G3
c = zero_c([0 0 0  -1 0 1  1 0 -1]);
matlabbatch{1}.spm.stats.con.consess{21}.tcon.name = 'Interaction: G2 vs G3 (Time)';
matlabbatch{1}.spm.stats.con.consess{21}.tcon.weights = c;
matlabbatch{1}.spm.stats.con.consess{21}.tcon.sessrep = 'none';

% -------------------------------------------------------------------------
% 3. quadratic-contrasts
% -------------------------------------------------------------------------

matlabbatch{1}.spm.stats.con.consess{22}.tcon.name = 'G1: pos Quadratic Time';
matlabbatch{1}.spm.stats.con.consess{22}.tcon.weights = [1 -2 1 0 0 0 0 0 0];
matlabbatch{1}.spm.stats.con.consess{22}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{23}.tcon.name = 'G2: pos Quadratic Time';
matlabbatch{1}.spm.stats.con.consess{23}.tcon.weights = [0 0 0 1 -2 1 0 0 0];
matlabbatch{1}.spm.stats.con.consess{23}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{24}.tcon.name = 'G3: pos Quadratic Time';
matlabbatch{1}.spm.stats.con.consess{24}.tcon.weights = [0 0 0 0 0 0 1 -2 1 ];
matlabbatch{1}.spm.stats.con.consess{24}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{25}.tcon.name = 'G1: neg Quadratic Time';
matlabbatch{1}.spm.stats.con.consess{25}.tcon.weights = [-1 2 -1 0 0 0 0 0 0];
matlabbatch{1}.spm.stats.con.consess{25}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{26}.tcon.name = 'G2: neg Quadratic Time';
matlabbatch{1}.spm.stats.con.consess{26}.tcon.weights = [0 0 0 -1 2 -1 0 0 0];
matlabbatch{1}.spm.stats.con.consess{26}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{27}.tcon.name = 'G3: neg Quadratic Time';
matlabbatch{1}.spm.stats.con.consess{27}.tcon.weights = [0 0 0 0 0 0 -1 2 -1 ];
matlabbatch{1}.spm.stats.con.consess{27}.tcon.sessrep = 'none';



% -------------------------------------------------------------------------
% 3. F-contrasts
% -------------------------------------------------------------------------

% F: Time effects in all groups
F1 = zero_c([-1 0 1  0 0 0  0 0 0]);
F2 = zero_c([0 0 0  -1 0 1  0 0 0]);
F3 = zero_c([0 0 0  0 0 0  -1 0 1]);
F_time = [F1; F2; F3];
matlabbatch{1}.spm.stats.con.consess{28}.fcon.name = 'F: Time effects (all groups)';
matlabbatch{1}.spm.stats.con.consess{28}.fcon.weights = F_time;
matlabbatch{1}.spm.stats.con.consess{28}.fcon.sessrep = 'none';

% F: Time × Group Interaction
F_int1 = zero_c([-1 0 1  1 0 -1  0 0 0]);  % G1 vs G2
F_int2 = zero_c([-1 0 1  0 0 0  1 0 -1]);  % G1 vs G3
F_int3 = zero_c([0 0 0  -1 0 1  1 0 -1]);  % G2 vs G3
F_int = [F_int1; F_int2; F_int3];
matlabbatch{1}.spm.stats.con.consess{29}.fcon.name = 'F: Time by Group interaction';
matlabbatch{1}.spm.stats.con.consess{29}.fcon.weights = F_int;
matlabbatch{1}.spm.stats.con.consess{29}.fcon.sessrep = 'none';

% G1 vs G2 late effect (TP2 to TP3)
c = zero_c([0 -1 1  0 1 -1  0 0 0]);
matlabbatch{1}.spm.stats.con.consess{30}.tcon.name = 'Interaction: G1 vs G2 (Late Time)';
matlabbatch{1}.spm.stats.con.consess{30}.tcon.weights = c;
matlabbatch{1}.spm.stats.con.consess{30}.tcon.sessrep = 'none';

% G2 late vs G1 early
c = zero_c([1 -1 0   0 -1 1  0 0 0]);
matlabbatch{1}.spm.stats.con.consess{31}.tcon.name = 'G2 Late vs G1 Early';
matlabbatch{1}.spm.stats.con.consess{31}.tcon.weights = c;
matlabbatch{1}.spm.stats.con.consess{31}.tcon.sessrep = 'none';

% G1 vs G2 early effect
c = zero_c([-1 1 0  1 -1 0  0 0 0]);
matlabbatch{1}.spm.stats.con.consess{32}.tcon.name = 'G1 vs G2 Early (should be ~0)';
matlabbatch{1}.spm.stats.con.consess{32}.tcon.weights = c;
matlabbatch{1}.spm.stats.con.consess{32}.tcon.sessrep = 'none';

F_qint1 = zero_c([1 -2 1  -1 2 -1  0 0 0]);  % G1 vs G2 (quadratic)
F_qint2 = zero_c([1 -2 1   0 0 0  -1 2 -1]); % G1 vs G3
F_qint3 = zero_c([0 0 0  1 -2 1  -1 2 -1]);  % G2 vs G3
F_qint = [F_qint1; F_qint2; F_qint3];
matlabbatch{1}.spm.stats.con.consess{33}.fcon.name = 'F: Quadratic by Group interaction';
matlabbatch{1}.spm.stats.con.consess{33}.fcon.weights = F_qint;
matlabbatch{1}.spm.stats.con.consess{33}.fcon.sessrep = 'none';

F_lq1 = zero_c([-1 0 1; 1 -2 1]);   % G1: linear + quadratic
F_lq2 = zero_c([0 0 0 -1 0 1; 0 0 0 1 -2 1]); % G2
F_lq3 = zero_c([0 0 0 0 0 0 -1 0 1; 0 0 0 0 0 0 1 -2 1]); % G3
F_lq_all = [F_lq1; F_lq2; F_lq3];
matlabbatch{1}.spm.stats.con.consess{34}.fcon.name = 'F: Linear + Quadratic (all groups)';
matlabbatch{1}.spm.stats.con.consess{34}.fcon.weights = F_lq_all;
matlabbatch{1}.spm.stats.con.consess{34}.fcon.sessrep = 'none';

c = zero_c([-1 0 1  -1 0 1  -1 0 1]);
matlabbatch{1}.spm.stats.con.consess{35}.tcon.name = 'All Groups: Time3 - Time1';
matlabbatch{1}.spm.stats.con.consess{35}.tcon.weights = c;
matlabbatch{1}.spm.stats.con.consess{35}.tcon.sessrep = 'none';


% -------------------------------------------------------------------------
% Delete existing contrasts (optional)
% -------------------------------------------------------------------------
matlabbatch{1}.spm.stats.con.delete = 1;

% -------------------------------------------------------------------------
% Run batch
% -------------------------------------------------------------------------

spm_jobman('run', matlabbatch)