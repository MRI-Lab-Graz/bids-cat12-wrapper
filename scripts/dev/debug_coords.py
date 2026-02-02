import nibabel as nib
import numpy as np


def debug_coords():
    stat_file = "results/vbm/2groups_tiv_age_sex/TFCE_log_pFWE_0001.nii"
    atlas_file = "templates/aal3.nii.gz"

    img_stat = nib.load(stat_file)
    data_stat = img_stat.get_fdata()

    img_atlas = nib.load(atlas_file)
    data_atlas = img_atlas.get_fdata()

    # Find peak
    if np.isnan(data_stat).any():
        print("Data contains NaNs")
        peak_idx = np.unravel_index(np.nanargmax(data_stat), data_stat.shape)
    else:
        peak_idx = np.unravel_index(np.argmax(data_stat), data_stat.shape)

    print(f"Peak voxel index (stat): {peak_idx}")

    # Convert to world
    peak_world = nib.affines.apply_affine(img_stat.affine, peak_idx)
    print(f"Peak world coord: {peak_world}")

    # Convert to atlas voxel
    inv_affine_atlas = np.linalg.inv(img_atlas.affine)
    voxel_coord_atlas = nib.affines.apply_affine(inv_affine_atlas, peak_world)
    print(f"Voxel coord in atlas (float): {voxel_coord_atlas}")
    voxel_coord_atlas_int = np.round(voxel_coord_atlas).astype(int)
    print(f"Voxel coord in atlas (int): {voxel_coord_atlas_int}")

    print(f"Atlas shape: {data_atlas.shape}")

    if (
        0 <= voxel_coord_atlas_int[0] < data_atlas.shape[0]
        and 0 <= voxel_coord_atlas_int[1] < data_atlas.shape[1]
        and 0 <= voxel_coord_atlas_int[2] < data_atlas.shape[2]
    ):
        val = data_atlas[tuple(voxel_coord_atlas_int)]
        print(f"Value at atlas voxel: {val}")
    else:
        print("Coordinate is outside atlas bounds.")


if __name__ == "__main__":
    debug_coords()
