import os
import json
import numpy as np
from pathlib import Path
import sys

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
    addn_ridge_str = "cntrl_tot_pts_out_bias_valset_verify"
    addn_save_str = addn_ridge_str
    ridge_path = (
        f"/home/dburnham/Documents/NTK/neural-tangents/multiTasking/results/"
        f"ridge_tuning_results/ridge_tuning_by_T_{addn_ridge_str}.npz"
    )
    if not os.path.exists(ridge_path):
        raise FileNotFoundError(f"Could not find: {ridge_path}")

    D = np.load(ridge_path)
    T_list = np.asarray(D["T_list"]).astype(int).tolist()
    reg_scale = {
        "full_ntk": np.asarray(D["lam_star_full_ntk"], dtype=float).tolist(),
        "bias_only": np.asarray(D["lam_star_bias_only"], dtype=float).tolist(),
    }

    W_std = 1.0
    b_std = 1.0
    par = KernelParams(sigma_w2=W_std**2, sigma_b2=b_std**2, sigma_v2=1.0, sigma_bout2=1.0)

    out_dir = "/home/dburnham/Documents/NTK/neural-tangents/multiTasking/results/ridge_eval_results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"frozen_ridge_eval_by_T_{addn_save_str}.npz")

    meta = dict(
        ridge_path=ridge_path,
        T_list=T_list,
        backend="kappa",
        generalization="within",
        normals_mode="fibonacci",
        random_phase=True,
        n_seeds=20,
        pts_per_circle=128,
        cntrl_pts_per_task=True,
        test_pts_per_circle=None,
        m=4,
        W_std=W_std,
        b_std=b_std,
        kappa_scale=1.0,
        selection_split="A",
        report_split="B",
        reg_scale=reg_scale,
        compute_spectrum=False,
    )
    with open(os.path.join(out_dir, f"frozen_ridge_eval_by_T_{addn_save_str}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Evaluating frozen ridge schedule from: {ridge_path}")
    out = make_effective_capacity_panels(
        T_list=T_list,
        n_seeds=20,
        pts_per_circle=128,
        cntrl_pts_per_task=True,
        test_pts_per_circle=None,
        m=4,
        normals_mode="fibonacci",
        random_phase=True,
        W_std=W_std,
        b_std=b_std,
        reg_scale=reg_scale,
        backend="kappa",
        generalization="within",
        kappa_params=par,
        kappa_scale=1.0,
        show_seed_progress=False,
        plot=False,
        selection_split="B",
        report_split=None,
        alpha=None,
        compute_spectrum=False,
    )
    np.savez_compressed(out_path, **out)

    summary = {
        "ridge_path": ridge_path,
        "selection_split": "A",
        "report_split": "B",
        "T_list": T_list,
        "mse_full_ntk_B": {str(T): float(np.asarray(out["full_ntk"]["mse_mean"]).reshape(-1)[i]) for i, T in enumerate(T_list)},
        "mse_bias_only_B": {str(T): float(np.asarray(out["bias_only"]["mse_mean"]).reshape(-1)[i]) for i, T in enumerate(T_list)},
        "nmse_full_ntk_B": {str(T): float(np.asarray(out["full_ntk"]["nmse_mean"]).reshape(-1)[i]) for i, T in enumerate(T_list)},
        "nmse_bias_only_B": {str(T): float(np.asarray(out["bias_only"]["nmse_mean"]).reshape(-1)[i]) for i, T in enumerate(T_list)},
    }
    with open(os.path.join(out_dir, f"frozen_ridge_eval_summary_{addn_save_str}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {out_path}")
    print(f"Wrote: {os.path.join(out_dir, f'frozen_ridge_eval_by_T_{addn_save_str}_meta.json')}")
    print(f"Wrote: {os.path.join(out_dir, f'frozen_ridge_eval_summary_{addn_save_str}.json')}")


if __name__ == "__main__":
    main()
