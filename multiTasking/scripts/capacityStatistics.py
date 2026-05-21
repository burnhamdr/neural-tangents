import os
import json
import numpy as np
import sys
from pathlib import Path

p = Path(__file__).resolve() if "__file__" in globals() else Path.cwd().resolve()
for parent in (p.parent, *p.parents):
    if (parent / "multiTasking").is_dir():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break
else:
    raise RuntimeError("Could not locate repo root containing 'multiTasking/'")

from multiTasking.utils.utils import run_snr_sweep_and_save, interpolate_lambda_star, KernelParams

def main():
    # ----------------------------
    # Load ridge tuning results (per-T lambda*)
    # ----------------------------
    addn_save_str = "cntrl_tot_pts_out_bias"#"test"
    addn_ridge_str = 'cntrl_tot_pts_out_bias'#"cntrl_tot_pts"
    ridge_path = f"/home/dburnham/Documents/NTK/neural-tangents/multiTasking/results/ridge_tuning_results/ridge_tuning_by_T_{addn_ridge_str}.npz"
    if not os.path.exists(ridge_path):
        raise FileNotFoundError(f"Could not find: {ridge_path}")

    D = np.load(ridge_path)

    # Original ridge-tuning grid
    T_list_ridge = np.asarray(D["T_list"]).astype(int)
    lam_star_full = np.asarray(D["lam_star_full_ntk"], dtype=float)
    lam_star_bias = np.asarray(D["lam_star_bias_only"], dtype=float)

    # ----------------------------
    # New sweep settings
    # ----------------------------
    T_list = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 50]#, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100

    # Interpolate lambda*(T) onto the new T_list
    lam_full = interpolate_lambda_star(T_list=T_list_ridge, lam_star=lam_star_full, T_new=T_list, clip=True)
    lam_bias = interpolate_lambda_star(T_list=T_list_ridge, lam_star=lam_star_bias, T_new=T_list, clip=True)

    # IMPORTANT: make these plain Python lists so your isinstance(..., list) check works as intended
    reg_scale = {
        "full_ntk": np.asarray(lam_full, dtype=float).tolist(),
        "bias_only": np.asarray(lam_bias, dtype=float).tolist(),
    }

    # ----------------------------
    # Kernel parameters / backend config
    # ----------------------------
    backend = "kappa"         # "kappa" or "nt"
    generalization = "within" # "within" or "across"
    W_std = 1.0
    b_std = 1.0

    par = KernelParams(sigma_w2=W_std**2, sigma_b2=b_std**2, sigma_v2=1.0, sigma_bout2=1.0)
    kappa_scale = 1.0  # keep 1.0 unless you decide to calibrate

    # ----------------------------
    # SNR sweep configuration
    # ----------------------------
    # Interpreted as: noise_var = signal_var / snr
    snr_list = [0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 5.0, 10.0, 20.0]
    noise_on = "both"  # "train", "test", "both"

    # Experiment sizes
    n_seeds = 20
    pts_per_circle = 128
    cntrl_pts_per_task = True
    test_pts_per_circle = None  # defaults to 2*pts_per_circle inside runner
    m = 4

    # Stats
    alpha = 0.05
    n_perm = 5000
    bh_correct = True

    # ----------------------------
    # Output directory + save metadata
    # ----------------------------
    out_dir = "/home/dburnham/Documents/NTK/neural-tangents/multiTasking/results/snr_sweep_results"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"snr_sweep_by_T_{addn_save_str}.npz")

    meta = dict(
        ridge_path=ridge_path,
        T_list=T_list,
        snr_list=snr_list,
        backend=backend,
        generalization=generalization,
        n_seeds=n_seeds,
        pts_per_circle=pts_per_circle,
        cntrl_pts_per_task=cntrl_pts_per_task,
        test_pts_per_circle=test_pts_per_circle,
        m=m,
        W_std=W_std,
        b_std=b_std,
        kappa_scale=kappa_scale,
        alpha=alpha,
        n_perm=n_perm,
        bh_correct=bh_correct,
        reg_scale=dict(
            full_ntk=reg_scale["full_ntk"],
            bias_only=reg_scale["bias_only"],
        ),
        noise_on=noise_on,
    )
    with open(os.path.join(out_dir, f"snr_sweep_by_T_{addn_save_str}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ----------------------------
    # Run sweep + save .npz
    # ----------------------------
    print(f"Saving results to: {out_path}")
    _ = run_snr_sweep_and_save(
        out_path=out_path,
        snr_list=snr_list,
        T_list=T_list,
        reg_scale=reg_scale,
        n_seeds=n_seeds,
        pts_per_circle=pts_per_circle,
        cntrl_pts_per_task=cntrl_pts_per_task,
        test_pts_per_circle=test_pts_per_circle,
        m=m,
        backend=backend,
        generalization=generalization,
        normals_mode="fibonacci",
        random_phase=True,
        W_std=W_std,
        b_std=b_std,
        kappa_params=par,
        kappa_scale=kappa_scale,
        alpha=alpha,
        n_perm=n_perm,
        bh_correct=bh_correct,
        show_snr_progress=True,
        noise_on=noise_on,
    )

    print("Done.")
    print(f"Wrote: {out_path}")
    print(f"Wrote: {os.path.join(out_dir, f'snr_sweep_by_T_{addn_save_str}_meta.json')}")

if __name__ == "__main__":
    main()