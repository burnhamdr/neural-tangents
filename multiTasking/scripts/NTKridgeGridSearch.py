import os
import json
from tqdm import tqdm
import numpy as np
import jax.numpy as jnp
import jax
from scipy.stats import wilcoxon
from pathlib import Path
import sys

try:
    import neural_tangents as nt
    from neural_tangents import stax
except Exception:
    nt = None
    stax = None
    
p = Path(__file__).resolve() if "__file__" in globals() else Path.cwd().resolve()
for parent in (p.parent, *p.parents):
    if (parent / "multiTasking").is_dir():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break
else:
    raise RuntimeError("Could not locate repo root containing 'multiTasking/'")

from multiTasking.utils.utils import KernelParams, make_effective_capacity_panels

def main():
    addn_save_str = "cntrl_tot_pts_out_bias_valset_64ppt"
    T_list = [1,2,3,4,5,10,15,20,25,30,35,40,50]#,60,70,80,90,100]
    reg_list = np.logspace(-10, -2, 40).astype(np.float64)
    kernels = ["full_ntk", "bias_only"]

    par = KernelParams(sigma_w2=1.0**2, sigma_b2=1.0**2, sigma_v2=1.0, sigma_bout2=1.0)

    nT = len(T_list)
    nL = len(reg_list)
    
    # Store tuning grids as dense arrays: (T, lambda)
    mse_grid_A = {k: np.full((nT, nL), np.nan, dtype=np.float64) for k in kernels}
    mse_star_A = {k: np.full((nT,), np.nan, dtype=np.float64) for k in kernels}
    lam_star = {k: np.full((nT,), np.nan, dtype=np.float64) for k in kernels}
    N_tot = np.max(T_list) * 64  # approx total number of data points
    for ti, T in enumerate(tqdm(T_list, desc="Per-T lambda tuning")):
        pts_per_circle_T = max(16, N_tot // T)
        for li, lam in enumerate(reg_list):
            out = make_effective_capacity_panels(
                T_list=[T],
                n_seeds=10,
                pts_per_circle=pts_per_circle_T,
                m=4,
                normals_mode="fibonacci",
                random_phase=True,
                W_std=1.0,
                b_std=1.0,
                reg_scale={
                    "full_ntk": lam,
                    "bias_only": lam,
                },
                backend="kappa",
                generalization="within",
                kappa_params=par,
                kappa_scale=1.0,
                show_seed_progress=False,
                plot=False,
                selection_split="A",
                report_split=None,
            )

            # Make sure these are scalars; .item() avoids the NumPy 1.25 warning.
            for k in kernels:
                mse_grid_A[k][ti, li] = np.asarray(out[k]["mse_mean"]).item()

        # Choose best lambda using replicate A.
        for k in kernels:
            j = int(np.nanargmin(mse_grid_A[k][ti, :]))
            lam_star[k][ti] = reg_list[j]
            mse_star_A[k][ti] = mse_grid_A[k][ti, j]

    # Output directory + save
    out_dir = "ridge_tuning_results"
    os.makedirs(out_dir, exist_ok=True)

    npz_path = os.path.join(out_dir, f"ridge_tuning_by_T_{addn_save_str}.npz")
    np.savez_compressed(
        npz_path,
        T_list=np.array(T_list, dtype=np.int32),
        reg_list=reg_list,
        mse_grid_A_full_ntk=mse_grid_A["full_ntk"],
        mse_grid_A_bias_only=mse_grid_A["bias_only"],
        mse_star_A_full_ntk=mse_star_A["full_ntk"],
        mse_star_A_bias_only=mse_star_A["bias_only"],
        # Legacy aliases point to the tuning/validation split A.
        mse_grid_full_ntk=mse_grid_A["full_ntk"],
        mse_grid_bias_only=mse_grid_A["bias_only"],
        mse_star_full_ntk=mse_star_A["full_ntk"],
        mse_star_bias_only=mse_star_A["bias_only"],
        lam_star_full_ntk=lam_star["full_ntk"],
        lam_star_bias_only=lam_star["bias_only"],
    )

    # Optional: also save a small JSON summary for quick inspection
    summary = {
        "T_list": T_list,
        "reg_list_min": float(reg_list.min()),
        "reg_list_max": float(reg_list.max()),
        "selection_split": "A",
        "report_split": None,
        "lam_star_full_ntk": {str(T): float(lam_star["full_ntk"][i]) for i, T in enumerate(T_list)},
        "lam_star_bias_only": {str(T): float(lam_star["bias_only"][i]) for i, T in enumerate(T_list)},
        "mse_star_A_full_ntk": {str(T): float(mse_star_A["full_ntk"][i]) for i, T in enumerate(T_list)},
        "mse_star_A_bias_only": {str(T): float(mse_star_A["bias_only"][i]) for i, T in enumerate(T_list)},
    }
    with open(os.path.join(out_dir, f"ridge_tuning_summary_{addn_save_str}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {npz_path}")
    print(f"Saved: {os.path.join(out_dir, f'ridge_tuning_summary_{addn_save_str}.json')}")

if __name__ == "__main__":
    main()
