#!/usr/bin/env python3
import scipy.io as sio
import numpy as np

spm_mat_path = "./results/vbm/3groups_vbm_smooth9_tiv/SPM.mat"
try:
    spm = sio.loadmat(spm_mat_path, struct_as_record=False, squeeze_me=True)
    if hasattr(spm["SPM"], "xCon"):
        xCon = spm["SPM"].xCon
        if not isinstance(xCon, (np.ndarray, list)):
            xCon = [xCon]
        
        print(f"Number of contrasts: {len(xCon)}")
        print()
        
        # Also check regressor names
        if hasattr(spm["SPM"], "xX") and hasattr(spm["SPM"].xX, "name"):
            regressor_names = [str(n) for n in spm["SPM"].xX.name]
            print(f"Regressors ({len(regressor_names)}):")
            for i, name in enumerate(regressor_names):
                print(f"  {i}: {name}")
            print()
        
        for i, con in enumerate(xCon[:3]):  # Show first 3 contrasts
            print(f"Contrast {i+1}: {con.name}")
            print(f"  STAT: {con.STAT}")
            weights = None
            if hasattr(con, "c"):
                weights = con.c
                print(f"  c shape: {np.array(weights).shape}")
                print(f"  c values: {weights}")
            elif hasattr(con, "weights"):
                weights = con.weights
                print(f"  weights shape: {np.array(weights).shape}")
                print(f"  weights values: {weights}")
            elif hasattr(con, "F"):
                weights = con.F
                print(f"  F shape: {np.array(weights).shape}")
                print(f"  F values: {weights}")
            print()
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
