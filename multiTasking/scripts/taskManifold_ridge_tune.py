import argparse
import json
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
from jax import jacrev, random, vmap


p = Path(__file__).resolve() if "__file__" in globals() else Path.cwd().resolve()
for parent in (p.parent, *p.parents):
    if (parent / "multiTasking").is_dir():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break
else:
    raise RuntimeError("Could not locate repo root containing 'multiTasking/'")

import taskManifold_update as tmu


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "ridge_tuning_results"

REGIME_SPECS = {
    "emp_bias": ("emp_bias_train", "emp_bias_testA", "emp_bias_testB"),
    "emp_full": ("emp_full_train", "emp_full_testA", "emp_full_testB"),
    "inf_bias": ("inf_bias_train", "inf_bias_testA", "inf_bias_testB"),
    "inf_full": ("inf_full_train", "inf_full_testA", "inf_full_testB"),
}


def parse_int_list(text):
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_float_list(text):
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def build_reg_list(args):
    if args.reg_list is not None:
        reg_list = np.asarray(parse_float_list(args.reg_list), dtype=np.float64)
    else:
        reg_list = np.logspace(args.reg_min_exp, args.reg_max_exp, args.reg_num).astype(np.float64)
    if reg_list.ndim != 1 or reg_list.size == 0:
        raise ValueError("reg_list must contain at least one value.")
    return reg_list


def sample_single_axis_pair_data(
    *,
    pts_per_circle,
    test_pts_per_circle,
    m,
    seed,
    target_phase,
    target_mode,
    global_field_name,
    pair_angle,
    base_normal=(0.0, 0.0, 1.0),
    rotation_axis=(1.0, 0.0, 0.0),
):
    rngA = np.random.default_rng(seed + 10_000_001)
    rngB = np.random.default_rng(seed + 10_000_002)
    base_n = jnp.asarray(base_normal, dtype=float)
    rot_ax = jnp.asarray(rotation_axis, dtype=float)
    angles = np.asarray([0.0, float(pair_angle)], dtype=float)

    normals = []
    Xs = []
    ys = []
    testA_Xs = []
    testA_ys = []
    testB_Xs = []
    testB_ys = []
    phases = np.zeros(2, dtype=float)

    for angle in angles:
        normal = np.asarray(tmu.rotate_normal_around_axis(base_n, rot_ax, angle), dtype=float)
        X, y, _ = tmu.generate_circle_samples(
            normal,
            pts_per_circle,
            m,
            phase=0.0,
            target_phase=target_phase,
            target_mode=target_mode,
            global_field_name=global_field_name,
            mode="grid",
        )
        X_testA, y_testA, _ = tmu.generate_circle_samples(
            normal,
            test_pts_per_circle,
            m,
            phase=0.0,
            target_phase=target_phase,
            target_mode=target_mode,
            global_field_name=global_field_name,
            rng=rngA,
            mode="random",
        )
        X_testB, y_testB, _ = tmu.generate_circle_samples(
            normal,
            test_pts_per_circle,
            m,
            phase=0.0,
            target_phase=target_phase,
            target_mode=target_mode,
            global_field_name=global_field_name,
            rng=rngB,
            mode="random",
        )
        normals.append(normal)
        Xs.append(np.asarray(X))
        ys.append(np.asarray(y))
        testA_Xs.append(np.asarray(X_testA))
        testA_ys.append(np.asarray(y_testA))
        testB_Xs.append(np.asarray(X_testB))
        testB_ys.append(np.asarray(y_testB))

    return {
        "Xs": Xs,
        "ys": ys,
        "testA_Xs": testA_Xs,
        "testA_ys": testA_ys,
        "testB_Xs": testB_Xs,
        "testB_ys": testB_ys,
        "normals": np.asarray(normals, dtype=float),
        "phases": phases,
    }


def sample_tuning_task_data(
    *,
    K_tasks,
    pts_per_circle,
    test_pts_per_circle,
    m,
    seed,
    task_family,
    normals_mode,
    random_phase,
    target_phase,
    target_mode,
    global_field_name,
    single_axis_pair_angle,
):
    if task_family == "single_axis" and K_tasks == 2:
        return sample_single_axis_pair_data(
            pts_per_circle=pts_per_circle,
            test_pts_per_circle=test_pts_per_circle,
            m=m,
            seed=seed,
            target_phase=target_phase,
            target_mode=target_mode,
            global_field_name=global_field_name,
            pair_angle=single_axis_pair_angle,
        )

    return tmu.sample_task_family_data(
        K_tasks=K_tasks,
        pts_per_circle=pts_per_circle,
        test_pts_per_circle=test_pts_per_circle,
        m=m,
        seed=seed,
        task_family=task_family,
        normals_mode=normals_mode,
        random_phase=random_phase,
        target_phase=target_phase,
        target_mode=target_mode,
        global_field_name=global_field_name,
    )


def make_batched_kernel_fns():
    if tmu.nt is None:
        raise ImportError("neural_tangents not available; install neural_tangents to run ridge tuning.")

    par = tmu.KernelParams(sigma_w2=1.0, sigma_b2=1.0, sigma_v2=1.0, sigma_bout2=1.0)
    kernel_fn_full = tmu.nt.batch(
        tmu.make_kappa_kernel_fn(par, which="full", kappa_scale=1.0),
        device_count=-1,
    )
    kernel_fn_bias = tmu.nt.batch(
        tmu.make_kappa_kernel_fn(par, which="bias", kappa_scale=1.0),
        device_count=-1,
    )
    return kernel_fn_full, kernel_fn_bias


def build_seed_cache(
    *,
    T,
    seed,
    dim,
    bias_std,
    pts,
    test_pts,
    m,
    task_family,
    normals_mode,
    random_phase,
    target_phase,
    target_mode,
    global_field_name,
    single_axis_pair_angle,
    kernel_fn_full,
    kernel_fn_bias,
):
    key = random.PRNGKey(seed)
    params0 = tmu.init_mlp_params(key, 3, dim, bias_std=bias_std)
    task_data = sample_tuning_task_data(
        K_tasks=T,
        pts_per_circle=pts,
        test_pts_per_circle=test_pts,
        m=m,
        seed=seed,
        task_family=task_family,
        normals_mode=normals_mode,
        random_phase=random_phase,
        target_phase=target_phase,
        target_mode=target_mode,
        global_field_name=global_field_name,
        single_axis_pair_angle=single_axis_pair_angle,
    )

    task_cache = []
    for X_np, y_np, X_testA_np, y_testA_np, X_testB_np, y_testB_np in zip(
        task_data["Xs"],
        task_data["ys"],
        task_data["testA_Xs"],
        task_data["testA_ys"],
        task_data["testB_Xs"],
        task_data["testB_ys"],
    ):
        X = jnp.asarray(X_np)
        y = jnp.asarray(y_np)
        X_testA = jnp.asarray(X_testA_np)
        y_testA = jnp.asarray(y_testA_np)
        X_testB = jnp.asarray(X_testB_np)
        y_testB = jnp.asarray(y_testB_np)

        f0 = tmu.mlp_apply(params0, X)
        f0_testA = tmu.mlp_apply(params0, X_testA)
        f0_testB = tmu.mlp_apply(params0, X_testB)

        J_tree = vmap(jacrev(tmu.f_single), (None, 0))(params0, X)
        Jb = jnp.concatenate([J_tree["b1"], J_tree["b2"][:, None]], axis=1)
        Jfull = tmu.flatten_pytree_jac(J_tree)

        J_tree_testA = vmap(jacrev(tmu.f_single), (None, 0))(params0, X_testA)
        Jb_testA = jnp.concatenate([J_tree_testA["b1"], J_tree_testA["b2"][:, None]], axis=1)
        Jfull_testA = tmu.flatten_pytree_jac(J_tree_testA)

        J_tree_testB = vmap(jacrev(tmu.f_single), (None, 0))(params0, X_testB)
        Jb_testB = jnp.concatenate([J_tree_testB["b1"], J_tree_testB["b2"][:, None]], axis=1)
        Jfull_testB = tmu.flatten_pytree_jac(J_tree_testB)

        task_cache.append(
            {
                "y": y,
                "f0": f0,
                "y_testA": y_testA,
                "f0_testA": f0_testA,
                "y_testB": y_testB,
                "f0_testB": f0_testB,
                "emp_bias_train": Jb @ Jb.T,
                "emp_bias_testA": Jb_testA @ Jb.T,
                "emp_bias_testB": Jb_testB @ Jb.T,
                "emp_full_train": Jfull @ Jfull.T,
                "emp_full_testA": Jfull_testA @ Jfull.T,
                "emp_full_testB": Jfull_testB @ Jfull.T,
                "inf_bias_train": jnp.array(kernel_fn_bias(X, X, get="ntk")),
                "inf_bias_testA": jnp.array(kernel_fn_bias(X_testA, X, get="ntk")),
                "inf_bias_testB": jnp.array(kernel_fn_bias(X_testB, X, get="ntk")),
                "inf_full_train": jnp.array(kernel_fn_full(X, X, get="ntk")),
                "inf_full_testA": jnp.array(kernel_fn_full(X_testA, X, get="ntk")),
                "inf_full_testB": jnp.array(kernel_fn_full(X_testB, X, get="ntk")),
            }
        )

    return task_cache


def evaluate_regime_mse(task, regime_name, reg, ridge_mode):
    train_key, testA_key, testB_key = REGIME_SPECS[regime_name]
    alpha = tmu.solve_tangent_update_kernel(
        task[train_key],
        task["y"],
        task["f0"],
        reg,
        ridge_mode=ridge_mode,
    )
    pred_testA = task["f0_testA"] + task[testA_key] @ alpha
    pred_testB = task["f0_testB"] + task[testB_key] @ alpha
    mse_testA = tmu.centered_mse(pred_testA, task["y_testA"])
    mse_testB = tmu.centered_mse(pred_testB, task["y_testB"])
    return 0.5 * (mse_testA + mse_testB)


def evaluate_seed_cache(task_cache, reg, ridge_mode):
    scores = {name: [] for name in REGIME_SPECS}
    for task in task_cache:
        for regime_name in REGIME_SPECS:
            scores[regime_name].append(evaluate_regime_mse(task, regime_name, reg, ridge_mode))
    return {name: float(np.mean(values)) for name, values in scores.items()}


def auto_output_stem(args):
    stem = (
        f"taskmanifold_ridge_{tmu.sanitize_name_token(args.task_family)}"
        f"_m{args.m}_pts{args.pts}_dim{args.dim}"
        f"_tphase{tmu.format_phase_tag(args.target_phase)}"
    )
    if args.task_family == "single_axis":
        stem += f"_pairang{tmu.format_phase_tag(args.single_axis_pair_angle)}"
    elif args.task_family == "sampled":
        stem += f"_normals{tmu.sanitize_name_token(args.normals_mode)}"
    return stem


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Retune trace- or eig-normalized ridge scales for the exact taskManifold_update setup, "
            "separately for empirical finite-width and infinite-width kernels."
        )
    )
    parser.add_argument("--T-list", type=str, default="1,2", help="Comma-separated task counts to tune, e.g. '1,2'.")
    parser.add_argument("--n-seeds", type=int, default=10, help="Number of seeds to average over for each T and ridge value.")
    parser.add_argument(
        "--reg-list",
        type=str,
        default=None,
        help="Optional comma-separated ridge scales. If omitted, uses a logspace grid from --reg-min-exp to --reg-max-exp.",
    )
    parser.add_argument("--reg-min-exp", type=float, default=-10.0, help="Lower log10 endpoint for the ridge grid.")
    parser.add_argument("--reg-max-exp", type=float, default=-2.0, help="Upper log10 endpoint for the ridge grid.")
    parser.add_argument("--reg-num", type=int, default=40, help="Number of ridge values when building a logspace grid.")
    parser.add_argument("--ridge-mode", choices=["trace", "max_eig"], default="trace")
    parser.add_argument("--dim", type=int, default=8192)
    parser.add_argument("--bias-std", type=float, default=0.0)
    parser.add_argument("--pts", type=int, default=300)
    parser.add_argument("--test-pts", type=int, default=None)
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=0, help="Base seed offset. Each tuning run uses seed_base + 1000*T + seed_index.")
    parser.add_argument("--task-family", choices=["sampled", "single_axis"], default="sampled")
    parser.add_argument("--normals-mode", choices=["fibonacci", "random"], default="fibonacci")
    parser.add_argument("--random-phase", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-mode", choices=["local_harmonic", "global_field"], default="local_harmonic")
    parser.add_argument("--target-phase", type=tmu.parse_angle_arg, default=0.0)
    parser.add_argument("--global-field", choices=["xz", "x2_minus_y2", "z2_legendre", "yz"], default="xz")
    parser.add_argument(
        "--single-axis-pair-angle",
        type=tmu.parse_angle_arg,
        default=np.pi / 4.0,
        help=(
            "When task_family=single_axis and T=2, tune the pair on angles [0, pair_angle] instead of the default "
            "evenly spaced [0, pi], which would produce the same great circle twice."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-stem",
        type=str,
        default=None,
        help="Optional filename stem. Defaults to an auto-generated name that includes family, m, dim, and target phase.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    T_list = parse_int_list(args.T_list)
    reg_list = build_reg_list(args)
    test_pts = args.test_pts if args.test_pts is not None else 2 * args.pts
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_stem or auto_output_stem(args)

    kernel_fn_full, kernel_fn_bias = make_batched_kernel_fns()

    nT = len(T_list)
    nL = len(reg_list)
    mse_grid = {name: np.full((nT, nL), np.nan, dtype=np.float64) for name in REGIME_SPECS}
    mse_star = {name: np.full((nT,), np.nan, dtype=np.float64) for name in REGIME_SPECS}
    lam_star = {name: np.full((nT,), np.nan, dtype=np.float64) for name in REGIME_SPECS}

    print(
        "Retuning ridge with "
        f"task_family={args.task_family}, target_mode={args.target_mode}, "
        f"target_phase={float(args.target_phase):.6f}, ridge_mode={args.ridge_mode}, "
        f"T_list={T_list}, n_seeds={args.n_seeds}, n_reg={nL}"
    )
    if args.task_family == "single_axis":
        print(f"Single-axis T=2 pair angle: {float(args.single_axis_pair_angle):.6f} rad")

    for ti, T in enumerate(T_list):
        print(f"\nPreparing caches for T={T}")
        seed_caches = []
        for seed_index in range(args.n_seeds):
            seed = args.seed_base + 1000 * int(T) + seed_index
            seed_caches.append(
                build_seed_cache(
                    T=T,
                    seed=seed,
                    dim=args.dim,
                    bias_std=args.bias_std,
                    pts=args.pts,
                    test_pts=test_pts,
                    m=args.m,
                    task_family=args.task_family,
                    normals_mode=args.normals_mode,
                    random_phase=args.random_phase,
                    target_phase=float(args.target_phase),
                    target_mode=args.target_mode,
                    global_field_name=args.global_field,
                    single_axis_pair_angle=float(args.single_axis_pair_angle),
                    kernel_fn_full=kernel_fn_full,
                    kernel_fn_bias=kernel_fn_bias,
                )
            )

        for li, reg in enumerate(reg_list):
            seed_scores = {name: [] for name in REGIME_SPECS}
            for cache in seed_caches:
                scores = evaluate_seed_cache(cache, float(reg), args.ridge_mode)
                for regime_name in REGIME_SPECS:
                    seed_scores[regime_name].append(scores[regime_name])

            for regime_name in REGIME_SPECS:
                mse_grid[regime_name][ti, li] = float(np.mean(seed_scores[regime_name]))

        for regime_name in REGIME_SPECS:
            best_idx = int(np.nanargmin(mse_grid[regime_name][ti]))
            lam_star[regime_name][ti] = reg_list[best_idx]
            mse_star[regime_name][ti] = mse_grid[regime_name][ti, best_idx]
            print(
                f"T={T:>2} | {regime_name:>8} | best_reg={lam_star[regime_name][ti]:.6g} "
                f"| mean_test_centered_mse={mse_star[regime_name][ti]:.6g}"
            )

    npz_path = output_dir / f"{output_stem}.npz"
    payload = {
        "T_list": np.asarray(T_list, dtype=np.int32),
        "reg_list": reg_list,
        "target_phase": np.asarray(float(args.target_phase), dtype=np.float64),
        "single_axis_pair_angle": np.asarray(float(args.single_axis_pair_angle), dtype=np.float64),
    }
    for regime_name in REGIME_SPECS:
        payload[f"mse_grid_{regime_name}"] = mse_grid[regime_name]
        payload[f"mse_star_{regime_name}"] = mse_star[regime_name]
        payload[f"lam_star_{regime_name}"] = lam_star[regime_name]
    np.savez_compressed(npz_path, **payload)

    summary = {
        "task_family": args.task_family,
        "normals_mode": args.normals_mode,
        "random_phase": bool(args.random_phase),
        "target_mode": args.target_mode,
        "target_phase": float(args.target_phase),
        "target_phase_over_pi": float(float(args.target_phase) / np.pi),
        "global_field_name": args.global_field,
        "single_axis_pair_angle": float(args.single_axis_pair_angle),
        "ridge_mode": args.ridge_mode,
        "T_list": [int(T) for T in T_list],
        "n_seeds": int(args.n_seeds),
        "dim": int(args.dim),
        "pts": int(args.pts),
        "test_pts": int(test_pts),
        "m": int(args.m),
        "reg_list_min": float(np.min(reg_list)),
        "reg_list_max": float(np.max(reg_list)),
        "lam_star_emp_bias": {str(T): float(lam_star["emp_bias"][i]) for i, T in enumerate(T_list)},
        "lam_star_emp_full": {str(T): float(lam_star["emp_full"][i]) for i, T in enumerate(T_list)},
        "lam_star_inf_bias": {str(T): float(lam_star["inf_bias"][i]) for i, T in enumerate(T_list)},
        "lam_star_inf_full": {str(T): float(lam_star["inf_full"][i]) for i, T in enumerate(T_list)},
        "mse_star_emp_bias": {str(T): float(mse_star["emp_bias"][i]) for i, T in enumerate(T_list)},
        "mse_star_emp_full": {str(T): float(mse_star["emp_full"][i]) for i, T in enumerate(T_list)},
        "mse_star_inf_bias": {str(T): float(mse_star["inf_bias"][i]) for i, T in enumerate(T_list)},
        "mse_star_inf_full": {str(T): float(mse_star["inf_full"][i]) for i, T in enumerate(T_list)},
    }
    json_path = output_dir / f"{output_stem}.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {npz_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
