%-----------------------------------------------------------------------
% Job saved on 17-Oct-2025 08:24:01 by cfg_util (rev $Rev: 8183 $)
% spm SPM - SPM25 (25.01.02)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
%%
matlabbatch{1}.spm.tools.cat.tools.T2x.data_T2x = {
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0001.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0002.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0003.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0004.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0005.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0006.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0007.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0008.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0009.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0010.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0011.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0012.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0013.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0014.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0015.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0016.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0017.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0018.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0019.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0020.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0021.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0022.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0023.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0024.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0025.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0026.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0027.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0030.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0031.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0032.nii,1'
                                                   '/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/spmT_0035.nii,1'
                                                   };
%%
matlabbatch{1}.spm.tools.cat.tools.T2x.conversion.sel = 6;
matlabbatch{1}.spm.tools.cat.tools.T2x.conversion.threshdesc.uncorr.thresh001 = 0.001;
matlabbatch{1}.spm.tools.cat.tools.T2x.conversion.inverse = 0;
matlabbatch{1}.spm.tools.cat.tools.T2x.conversion.cluster.none = 1;
matlabbatch{1}.spm.tools.cat.tools.T2x.atlas = 'None';
matlabbatch{2}.spm.tools.tfce_estimate.data = {'/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/SPM.mat'};
matlabbatch{2}.spm.tools.tfce_estimate.nproc = 0;
matlabbatch{2}.spm.tools.tfce_estimate.mask = {'/Volumes/Thunder/129_PK01/cat12/stats/s9_int_control/mask.nii,1'};
matlabbatch{2}.spm.tools.tfce_estimate.conspec.titlestr = '';
matlabbatch{2}.spm.tools.tfce_estimate.conspec.contrasts(1) = cfg_dep('Threshold and transform spmT images: Transform & Threshold spm volumes', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Pname'));
matlabbatch{2}.spm.tools.tfce_estimate.conspec.n_perm = 1500;
matlabbatch{2}.spm.tools.tfce_estimate.nuisance_method = 2;
matlabbatch{2}.spm.tools.tfce_estimate.tbss = 0;
matlabbatch{2}.spm.tools.tfce_estimate.singlethreaded = 0;
