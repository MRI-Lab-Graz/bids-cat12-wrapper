import nibabel as nib
import os

def check_geometry():
    atlas_path = 'templates/aal3.nii.gz'
    stat_files = sorted(os.listdir('results/vbm/2groups_tiv_age_sex'))
    stat_file = [f for f in stat_files if 'log_pFWE' in f][0]
    stat_path = os.path.join('results/vbm/2groups_tiv_age_sex', stat_file)
    
    img_atlas = nib.load(atlas_path)
    img_stat = nib.load(stat_path)
    
    print(f"Atlas shape: {img_atlas.shape}")
    print(f"Atlas affine:\n{img_atlas.affine}")
    print("-" * 20)
    print(f"Stat shape: {img_stat.shape}")
    print(f"Stat affine:\n{img_stat.affine}")

if __name__ == "__main__":
    check_geometry()
