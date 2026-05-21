import argparse
import csv
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation, Slerp

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
except Exception:
    plt = None
    Normalize = None

import taskManifold_update as tmu


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "transport_models"


def parse_csv_list(text):
    if text is None:
        return []
    return [item.strip() for item in str(text).split(",") if item.strip()]


def write_csv_rows(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def unit_normalize(vec, eps=1e-12):
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm <= eps:
        return np.zeros_like(vec)
    return vec / norm


def format_axis_short(vec, atol=1e-6):
    vec = np.asarray(vec, dtype=float).reshape(-1)
    rounded = np.rint(vec).astype(int)
    if vec.size == 3 and np.allclose(vec, rounded, atol=atol):
        return "".join(str(int(v)) for v in rounded)
    return "(" + ",".join(f"{float(v):.2f}" for v in vec) + ")"


def multi_r2_score(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rss = float(np.sum((y_true - y_pred) ** 2))
    centered = y_true - np.mean(y_true, axis=0, keepdims=True)
    tss = float(np.sum(centered ** 2))
    if tss <= eps:
        return np.nan
    return 1.0 - rss / tss


def cosine_similarity_rows(a, b, eps=1e-12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + eps
    return np.sum(a * b, axis=1) / denom


def compute_shared_subspace_metrics(coords_true, coords_pred, basis, centered_updates, eps=1e-12):
    coords_true = np.asarray(coords_true, dtype=float)
    coords_pred = np.asarray(coords_pred, dtype=float)
    basis = np.asarray(basis, dtype=float)
    centered_updates = np.asarray(centered_updates, dtype=float)
    k_max = min(coords_true.shape[1], coords_pred.shape[1], basis.shape[1])
    if k_max <= 0:
        empty = np.zeros((0,), dtype=float)
        return {
            "coord_r2_topk": empty,
            "update_r2_topk": empty,
            "update_cosine_topk": empty,
            "update_rmse_topk": empty,
            "energy_frac_topk": empty,
        }

    total_energy = float(np.sum(centered_updates ** 2))
    coord_r2_topk = []
    update_r2_topk = []
    update_cosine_topk = []
    update_rmse_topk = []
    energy_frac_topk = []

    for k in range(1, k_max + 1):
        basis_k = basis[:, :k]
        true_centered_k = coords_true[:, :k] @ basis_k.T
        pred_centered_k = coords_pred[:, :k] @ basis_k.T
        coord_r2_topk.append(
            multi_r2_score(coords_true[:, :k], coords_pred[:, :k], eps=eps)
        )
        update_r2_topk.append(
            multi_r2_score(true_centered_k, pred_centered_k, eps=eps)
        )
        update_cosine_topk.append(
            float(np.mean(cosine_similarity_rows(true_centered_k, pred_centered_k, eps=eps)))
        )
        update_rmse_topk.append(
            float(np.sqrt(np.mean((true_centered_k - pred_centered_k) ** 2)))
        )
        energy_frac_topk.append(
            float(np.sum(true_centered_k ** 2) / (total_energy + eps))
            if total_energy > eps
            else np.nan
        )

    return {
        "coord_r2_topk": np.asarray(coord_r2_topk, dtype=float),
        "update_r2_topk": np.asarray(update_r2_topk, dtype=float),
        "update_cosine_topk": np.asarray(update_cosine_topk, dtype=float),
        "update_rmse_topk": np.asarray(update_rmse_topk, dtype=float),
        "energy_frac_topk": np.asarray(energy_frac_topk, dtype=float),
    }


def compute_shared_subspace_metrics_from_full_coords(full_coords_true, coords_pred, eps=1e-12):
    full_coords_true = np.asarray(full_coords_true, dtype=float)
    coords_pred = np.asarray(coords_pred, dtype=float)
    k_max = min(full_coords_true.shape[1], coords_pred.shape[1])
    if k_max <= 0:
        empty = np.zeros((0,), dtype=float)
        return {
            "coord_r2_topk": empty,
            "update_r2_topk": empty,
            "update_cosine_topk": empty,
            "update_rmse_topk": empty,
            "energy_frac_topk": empty,
        }

    total_energy = float(np.sum(full_coords_true ** 2))
    coord_r2_topk = []
    update_r2_topk = []
    update_cosine_topk = []
    update_rmse_topk = []
    energy_frac_topk = []
    for k in range(1, k_max + 1):
        true_k = full_coords_true[:, :k]
        pred_k = coords_pred[:, :k]
        coord_r2_topk.append(multi_r2_score(true_k, pred_k, eps=eps))
        update_r2_topk.append(multi_r2_score(true_k, pred_k, eps=eps))
        update_cosine_topk.append(float(np.mean(cosine_similarity_rows(true_k, pred_k, eps=eps))))
        update_rmse_topk.append(float(np.sqrt(np.mean((true_k - pred_k) ** 2))))
        energy_frac_topk.append(
            float(np.sum(true_k ** 2) / (total_energy + eps))
            if total_energy > eps
            else np.nan
        )

    return {
        "coord_r2_topk": np.asarray(coord_r2_topk, dtype=float),
        "update_r2_topk": np.asarray(update_r2_topk, dtype=float),
        "update_cosine_topk": np.asarray(update_cosine_topk, dtype=float),
        "update_rmse_topk": np.asarray(update_rmse_topk, dtype=float),
        "energy_frac_topk": np.asarray(energy_frac_topk, dtype=float),
    }


def load_task_update_vectors(results_dir):
    results_dir = Path(results_dir)
    path = results_dir / "task_update_vectors.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Re-run taskManifold_update.py after the raw update export change."
        )
    obj = np.load(path, allow_pickle=True)
    task_normals = np.asarray(obj["task_normals"], dtype=float)
    if "task_frame_v1" in obj and "task_frame_v2" in obj and "task_rotations" in obj:
        task_frame_v1 = np.asarray(obj["task_frame_v1"], dtype=float)
        task_frame_v2 = np.asarray(obj["task_frame_v2"], dtype=float)
        task_rotations = np.asarray(obj["task_rotations"], dtype=float)
    else:
        frame_v1_rows = []
        frame_v2_rows = []
        rotation_rows = []
        for normal in task_normals:
            _, v1, v2 = tmu.orthonormal_circle_frame(normal)
            v1_np = np.asarray(v1, dtype=float)
            v2_np = np.asarray(v2, dtype=float)
            frame_v1_rows.append(v1_np)
            frame_v2_rows.append(v2_np)
            rotation_rows.append(tmu.rotation_matrix_from_frame(v1_np, v2_np, normal))
        task_frame_v1 = np.asarray(frame_v1_rows, dtype=float)
        task_frame_v2 = np.asarray(frame_v2_rows, dtype=float)
        task_rotations = np.asarray(rotation_rows, dtype=float)
    return {
        "results_dir": str(results_dir.resolve()),
        "task_normals": task_normals,
        "task_frame_v1": task_frame_v1,
        "task_frame_v2": task_frame_v2,
        "task_rotations": task_rotations,
        "task_phases": np.asarray(obj["task_phases"], dtype=float),
        "task_color_values": np.asarray(obj["task_color_values"], dtype=float),
        "delta_bias": np.asarray(obj["delta_bias"], dtype=float),
        "delta_full": np.asarray(obj["delta_full"], dtype=float),
        "target_phase": float(np.asarray(obj["target_phase"], dtype=float).reshape(-1)[0]),
        "target_mode": str(np.asarray(obj["target_mode"]).reshape(-1)[0]),
        "global_field_name": str(np.asarray(obj["global_field_name"]).reshape(-1)[0]),
        "m": int(np.asarray(obj["m"], dtype=int).reshape(-1)[0]),
        "task_family": str(np.asarray(obj["task_family"]).reshape(-1)[0]),
        "task_representation_version": int(np.asarray(obj["task_representation_version"], dtype=int).reshape(-1)[0]) if "task_representation_version" in obj else 1,
        "base_normal": np.asarray(obj["base_normal"], dtype=float),
        "rotation_axis": np.asarray(obj["rotation_axis"], dtype=float),
    }


def infer_random_phase(task_phases, target_mode):
    task_phases = np.asarray(task_phases, dtype=float)
    return bool(target_mode == "local_harmonic" and not np.allclose(task_phases, 0.0, atol=1e-10))


def reconstruct_task_family_data(saved_run, *, run_meta):
    task_family = str(saved_run["task_family"])
    normals_mode_candidates = ["fibonacci"] if task_family == "single_axis" else ["fibonacci", "random"]
    last_error = None
    for normals_mode in normals_mode_candidates:
        try:
            task_data = tmu.sample_task_family_data(
                K_tasks=int(run_meta.get("K_tasks", len(saved_run["task_normals"]))),
                pts_per_circle=int(run_meta["pts"]),
                m=int(saved_run["m"]),
                seed=int(run_meta.get("seed", 0)),
                task_family=task_family,
                normals_mode=normals_mode,
                random_phase=infer_random_phase(saved_run["task_phases"], saved_run["target_mode"]),
                target_phase=float(saved_run["target_phase"]),
                target_mode=str(saved_run["target_mode"]),
                global_field_name=str(saved_run["global_field_name"]),
                base_normal=np.asarray(saved_run["base_normal"], dtype=float),
                rotation_axis=np.asarray(saved_run["rotation_axis"], dtype=float),
            )
        except Exception as exc:
            last_error = exc
            continue
        normals_ok = np.allclose(
            np.asarray(task_data["normals"], dtype=float),
            np.asarray(saved_run["task_normals"], dtype=float),
            atol=1e-7,
        )
        rotations_ok = np.allclose(
            np.asarray(task_data["task_rotations"], dtype=float),
            np.asarray(saved_run["task_rotations"], dtype=float),
            atol=1e-7,
        )
        if normals_ok and rotations_ok:
            return task_data
        last_error = ValueError(
            f"Reconstructed task data with normals_mode={normals_mode} did not match the saved task frame."
        )
    raise ValueError(
        f"Could not reconstruct task family data for {saved_run['results_dir']}: {last_error}"
    )


def reconstruct_infinite_width_rkhs_run(results_dir, *, which="bias", solo_inf_reg=1e-10, ridge_mode="trace"):
    which = str(which)
    if which not in {"bias", "full"}:
        raise ValueError(f"Unsupported infinite-width correction kind: {which}")
    saved_run = load_task_update_vectors(results_dir)
    if int(saved_run.get("task_representation_version", 1)) < 2:
        raise ValueError(
            f"{saved_run['results_dir']} uses task representation v1. "
            "Re-run taskManifold_update.py so task_update_vectors.npz includes the saved transported rotations."
        )
    run_meta = tmu.parse_run_metadata_from_name(Path(results_dir).name)
    if "pts" not in run_meta or "dim" not in run_meta:
        raise ValueError(f"Could not parse pts/dim from results dir name: {Path(results_dir).name}")
    task_data = reconstruct_task_family_data(saved_run, run_meta=run_meta)

    if tmu.nt is None:
        raise ImportError(
            "neural_tangents is required to reconstruct infinite-width RKHS corrections."
        )

    params0 = tmu.init_mlp_params(
        tmu.random.PRNGKey(int(run_meta.get("seed", 0))),
        3,
        int(run_meta["dim"]),
        bias_std=0.0,
    )
    kappa_params = tmu.KernelParams(sigma_w2=1.0, sigma_b2=1.0, sigma_v2=1.0, sigma_bout2=1.0)
    kernel_fn = tmu.make_kappa_kernel_fn(kappa_params, which=which, kappa_scale=1.0)
    kernel_fn = tmu.nt.batch(kernel_fn, device_count=-1)

    Xs = []
    ys = []
    f0s = []
    alphas = []
    norms = []
    for X_np, y_np in zip(task_data["Xs"], task_data["ys"]):
        X = tmu.jnp.asarray(X_np)
        y = tmu.jnp.asarray(y_np)
        f0 = tmu.mlp_apply(params0, X)
        K_inf = tmu.jnp.array(kernel_fn(X, X, get="ntk"))
        alpha = tmu.solve_tangent_update_kernel(
            K_inf,
            y,
            f0,
            float(solo_inf_reg),
            ridge_mode=ridge_mode,
        )
        alpha_np = np.asarray(alpha, dtype=float)
        K_np = np.asarray(K_inf, dtype=float)
        norm_sq = float(alpha_np @ K_np @ alpha_np)
        Xs.append(np.asarray(X, dtype=float))
        ys.append(np.asarray(y, dtype=float))
        f0s.append(np.asarray(f0, dtype=float))
        alphas.append(alpha_np)
        norms.append(np.sqrt(max(norm_sq, 0.0)))

    return {
        **saved_run,
        "run_meta": run_meta,
        "task_data": task_data,
        "kernel_fn": kernel_fn,
        "Xs": Xs,
        "ys": ys,
        "f0s": f0s,
        "alphas_inf_rkhs": alphas,
        "correction_norms": np.asarray(norms, dtype=float),
        "within_similarity_rkhs": np.asarray(
            tmu.load_matrix_csv(Path(results_dir) / f"infinite_width_rkhs_similarity_{which}.csv"),
            dtype=float,
        ),
        "correction_kind": which,
    }


def pool_reconstructed_run_geometry(reconstructed_runs):
    normals = []
    frame_v1 = []
    frame_v2 = []
    rotations = []
    color_values = []
    source_dirs = []
    source_run = []
    task_index = []
    rotation_axes = []

    for run_idx, item in enumerate(reconstructed_runs):
        run_normals = np.asarray(item["task_normals"], dtype=float)
        normals.append(run_normals)
        frame_v1.append(np.asarray(item["task_frame_v1"], dtype=float))
        frame_v2.append(np.asarray(item["task_frame_v2"], dtype=float))
        rotations.append(np.asarray(item["task_rotations"], dtype=float))
        run_colors = np.asarray(item["task_color_values"], dtype=float)
        color_values.append(run_colors)
        source_dirs.extend([item["results_dir"]] * len(run_normals))
        source_run.extend([run_idx] * len(run_normals))
        task_index.extend(range(len(run_normals)))
        rotation_axes.extend([np.asarray(item["rotation_axis"], dtype=float)] * len(run_normals))

    return {
        "normals": np.vstack(normals) if normals else np.zeros((0, 3)),
        "frame_v1": np.vstack(frame_v1) if frame_v1 else np.zeros((0, 3)),
        "frame_v2": np.vstack(frame_v2) if frame_v2 else np.zeros((0, 3)),
        "task_rotations": np.concatenate(rotations, axis=0) if rotations else np.zeros((0, 3, 3)),
        "updates": np.zeros((sum(len(np.asarray(run["task_normals"])) for run in reconstructed_runs), 0), dtype=float),
        "color_values": np.concatenate(color_values) if color_values else np.zeros((0,), dtype=float),
        "source_dirs": np.asarray(source_dirs, dtype=object),
        "source_run": np.asarray(source_run, dtype=int),
        "task_index": np.asarray(task_index, dtype=int),
        "rotation_axes": np.asarray(rotation_axes, dtype=float) if rotation_axes else np.zeros((0, 3)),
        "runs": reconstructed_runs,
    }


def compute_cross_run_similarity_block(run_a, run_b, eps=1e-12):
    n_a = len(run_a["Xs"])
    n_b = len(run_b["Xs"])
    block = np.zeros((n_a, n_b), dtype=float)
    kernel_fn = run_a["kernel_fn"]
    for i in range(n_a):
        Xa = run_a["Xs"][i]
        alpha_a = np.asarray(run_a["alphas_inf_rkhs"][i], dtype=float)
        norm_a = float(run_a["correction_norms"][i])
        for j in range(n_b):
            Xb = run_b["Xs"][j]
            alpha_b = np.asarray(run_b["alphas_inf_rkhs"][j], dtype=float)
            norm_b = float(run_b["correction_norms"][j])
            K_cross = np.asarray(kernel_fn(Xa, Xb, get="ntk"), dtype=float)
            numer = float(alpha_a @ K_cross @ alpha_b)
            block[i, j] = numer / (norm_a * norm_b + eps)
    return block


def build_similarity_matrix_blocks(train_runs, other_runs=None):
    train_runs = list(train_runs)
    if other_runs is None:
        sizes = [len(run["Xs"]) for run in train_runs]
        total = int(sum(sizes))
        gram = np.zeros((total, total), dtype=float)
        offsets = np.cumsum([0] + sizes)
        for i, run_i in enumerate(train_runs):
            si = slice(offsets[i], offsets[i + 1])
            gram[si, si] = np.asarray(run_i["within_similarity_rkhs"], dtype=float)
            for j in range(i + 1, len(train_runs)):
                run_j = train_runs[j]
                sj = slice(offsets[j], offsets[j + 1])
                block = compute_cross_run_similarity_block(run_i, run_j)
                gram[si, sj] = block
                gram[sj, si] = block.T
        return 0.5 * (gram + gram.T)

    other_runs = list(other_runs)
    train_sizes = [len(run["Xs"]) for run in train_runs]
    other_sizes = [len(run["Xs"]) for run in other_runs]
    cross = np.zeros((int(sum(other_sizes)), int(sum(train_sizes))), dtype=float)
    train_offsets = np.cumsum([0] + train_sizes)
    other_offsets = np.cumsum([0] + other_sizes)
    for i, run_other in enumerate(other_runs):
        si = slice(other_offsets[i], other_offsets[i + 1])
        for j, run_train in enumerate(train_runs):
            sj = slice(train_offsets[j], train_offsets[j + 1])
            cross[si, sj] = compute_cross_run_similarity_block(run_other, run_train)
    return cross


def _check_run_compatibility(loaded_runs):
    if not loaded_runs:
        return
    ref = loaded_runs[0]
    ref_keys = ("target_phase", "target_mode", "global_field_name", "m")
    for item in loaded_runs[1:]:
        for key in ref_keys:
            lhs = ref[key]
            rhs = item[key]
            if isinstance(lhs, float):
                if not np.isclose(lhs, rhs):
                    raise ValueError(f"Incompatible {key}: {lhs} vs {rhs}")
            elif lhs != rhs:
                raise ValueError(f"Incompatible {key}: {lhs} vs {rhs}")


def pool_results_dirs(results_dirs, update_kind="bias"):
    loaded = [load_task_update_vectors(path) for path in results_dirs]
    _check_run_compatibility(loaded)

    normals = []
    frame_v1 = []
    frame_v2 = []
    rotations = []
    updates = []
    color_values = []
    source_dirs = []
    source_run = []
    task_index = []
    rotation_axes = []

    key = "delta_bias" if update_kind == "bias" else "delta_full"
    for run_idx, item in enumerate(loaded):
        run_normals = np.asarray(item["task_normals"], dtype=float)
        run_frame_v1 = np.asarray(item["task_frame_v1"], dtype=float)
        run_frame_v2 = np.asarray(item["task_frame_v2"], dtype=float)
        run_rotations = np.asarray(item["task_rotations"], dtype=float)
        run_updates = np.asarray(item[key], dtype=float)
        run_colors = np.asarray(item["task_color_values"], dtype=float)
        normals.append(run_normals)
        frame_v1.append(run_frame_v1)
        frame_v2.append(run_frame_v2)
        rotations.append(run_rotations)
        updates.append(run_updates)
        color_values.append(run_colors)
        source_dirs.extend([item["results_dir"]] * len(run_normals))
        source_run.extend([run_idx] * len(run_normals))
        task_index.extend(range(len(run_normals)))
        rotation_axes.extend([np.asarray(item["rotation_axis"], dtype=float)] * len(run_normals))

    return {
        "normals": np.vstack(normals) if normals else np.zeros((0, 3)),
        "frame_v1": np.vstack(frame_v1) if frame_v1 else np.zeros((0, 3)),
        "frame_v2": np.vstack(frame_v2) if frame_v2 else np.zeros((0, 3)),
        "task_rotations": np.concatenate(rotations, axis=0) if rotations else np.zeros((0, 3, 3)),
        "updates": np.vstack(updates) if updates else np.zeros((0, 0)),
        "color_values": np.concatenate(color_values) if color_values else np.zeros((0,), dtype=float),
        "source_dirs": np.asarray(source_dirs, dtype=object),
        "source_run": np.asarray(source_run, dtype=int),
        "task_index": np.asarray(task_index, dtype=int),
        "rotation_axes": np.asarray(rotation_axes, dtype=float) if rotation_axes else np.zeros((0, 3)),
        "runs": loaded,
    }


def build_feature_powers(n_vars, max_degree):
    n_vars = int(n_vars)
    max_degree = int(max_degree)
    if n_vars <= 0:
        return np.zeros((0, 0), dtype=int)
    if max_degree < 1 or max_degree > 4:
        raise ValueError(f"Unsupported degree={max_degree}. Expected 1, 2, 3, or 4.")
    powers = []
    for degree in range(1, max_degree + 1):
        for combo in combinations_with_replacement(range(n_vars), degree):
            exp = np.zeros(n_vars, dtype=int)
            for idx in combo:
                exp[idx] += 1
            powers.append(exp)
    return np.asarray(powers, dtype=int)


def polynomial_feature_names(base_labels, powers):
    labels = [str(label) for label in base_labels]
    names = []
    for power in np.asarray(powers, dtype=int):
        factors = []
        for label, exponent in zip(labels, power):
            exponent = int(exponent)
            if exponent <= 0:
                continue
            if exponent == 1:
                factors.append(label)
            else:
                factors.append(f"{label}^{exponent}")
        names.append(" ".join(factors) if factors else "1")
    return np.asarray(names, dtype=object)


def polynomial_features(base_features, powers):
    base_features = np.asarray(base_features, dtype=float)
    if base_features.ndim != 2:
        raise ValueError("base_features must be an N x D array.")
    cols = []
    for power in np.asarray(powers, dtype=int):
        col = np.ones(base_features.shape[0], dtype=float)
        nz = np.where(power > 0)[0]
        for idx in nz:
            col *= base_features[:, idx] ** int(power[idx])
        cols.append(col)
    return np.column_stack(cols) if cols else np.zeros((base_features.shape[0], 0), dtype=float)


def build_rotation_feature_powers(rotation_degree):
    return build_feature_powers(9, rotation_degree)


def rotation_feature_names(powers):
    return polynomial_feature_names([f"R{i}{j}" for i in range(3) for j in range(3)], powers)


def rotation_polynomial_features(task_rotations, powers):
    task_rotations = np.asarray(task_rotations, dtype=float)
    if task_rotations.ndim != 3 or task_rotations.shape[1:] != (3, 3):
        raise ValueError("task_rotations must be an N x 3 x 3 array.")
    flat = task_rotations.reshape(task_rotations.shape[0], 9)
    return polynomial_features(flat, powers)


def fit_coordinate_ridge_regression(features, coords, coord_ridge=1e-3):
    features = np.asarray(features, dtype=float)
    coords = np.asarray(coords, dtype=float)
    feat_mean = np.mean(features, axis=0, keepdims=True)
    feat_scale = np.std(features, axis=0, keepdims=True)
    feat_scale = np.where(feat_scale > 1e-12, feat_scale, 1.0)
    features_std = (features - feat_mean) / feat_scale
    design = np.column_stack([np.ones(features_std.shape[0], dtype=float), features_std])
    ridge = float(coord_ridge)
    gram = design.T @ design
    rhs = design.T @ coords
    reg = ridge * np.eye(gram.shape[0], dtype=float)
    reg[0, 0] = 0.0
    beta = np.linalg.solve(gram + reg, rhs)
    pred_coords = design @ beta
    return {
        "coord_beta": beta,
        "feature_mean": feat_mean.reshape(-1),
        "feature_scale": feat_scale.reshape(-1),
        "coords_pred": pred_coords,
        "coord_rank": int(np.linalg.matrix_rank(design)),
    }


def global_field_quadratic_matrix(field_name):
    field_name = str(field_name)
    if field_name == "xz":
        return np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    if field_name == "yz":
        return np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=float)
    if field_name == "x2_minus_y2":
        return np.asarray([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)
    if field_name == "z2_legendre":
        return np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], dtype=float)
    raise ValueError(
        "global_field_name must be one of {'xz', 'yz', 'x2_minus_y2', 'z2_legendre'}"
    )


def global_field_support_coeff_features(task_normals, global_field_name):
    task_normals = np.asarray(task_normals, dtype=float)
    if task_normals.ndim != 2 or task_normals.shape[1] != 3:
        raise ValueError("task_normals must be an N x 3 array.")
    A = global_field_quadratic_matrix(global_field_name)
    eye = np.eye(3, dtype=float)
    rows = []
    for normal in task_normals:
        n = unit_normalize(normal)
        P = eye - np.outer(n, n)
        B = 0.5 * (P @ A @ P + (P @ A @ P).T)
        rows.append(
            [
                float(B[0, 0]),
                float(B[1, 1]),
                float(B[2, 2]),
                float(B[0, 1]),
                float(B[0, 2]),
                float(B[1, 2]),
            ]
        )
    names = np.asarray(
        ["B_xx", "B_yy", "B_zz", "B_xy", "B_xz", "B_yz"],
        dtype=object,
    )
    return np.asarray(rows, dtype=float), names


def global_field_fourier_features(task_rotations, global_field_name):
    task_rotations = np.asarray(task_rotations, dtype=float)
    if task_rotations.ndim != 3 or task_rotations.shape[1:] != (3, 3):
        raise ValueError("task_rotations must be an N x 3 x 3 array.")
    A = global_field_quadratic_matrix(global_field_name)
    rows = []
    for R in task_rotations:
        v1 = np.asarray(R[:, 0], dtype=float)
        v2 = np.asarray(R[:, 1], dtype=float)
        a = float(v1 @ A @ v1)
        b = float(v1 @ A @ v2)
        d = float(v2 @ A @ v2)
        rows.append(
            [
                0.5 * (a + d),
                0.5 * (a - d),
                b,
            ]
        )
    names = np.asarray(["f_const", "f_cos2", "f_sin2"], dtype=object)
    return np.asarray(rows, dtype=float), names


def descriptor_base_features(descriptor_kind, task_rotations, task_normals, *, global_field_name=None):
    descriptor_kind = str(descriptor_kind)
    task_rotations = np.asarray(task_rotations, dtype=float)
    task_normals = np.asarray(task_normals, dtype=float)
    if descriptor_kind == "rotation":
        if task_rotations.ndim != 3 or task_rotations.shape[1:] != (3, 3):
            raise ValueError("task_rotations must be an N x 3 x 3 array for rotation descriptors.")
        return task_rotations.reshape(task_rotations.shape[0], 9), np.asarray(
            [f"R{i}{j}" for i in range(3) for j in range(3)],
            dtype=object,
        )
    if descriptor_kind == "global_field_coeffs":
        if not global_field_name:
            raise ValueError("global_field_name is required for descriptor_kind='global_field_coeffs'.")
        return global_field_support_coeff_features(task_normals, global_field_name)
    if descriptor_kind == "global_field_fourier":
        if not global_field_name:
            raise ValueError("global_field_name is required for descriptor_kind='global_field_fourier'.")
        return global_field_fourier_features(task_rotations, global_field_name)
    raise ValueError(f"Unsupported descriptor_kind={descriptor_kind}")


def evaluate_rotation_features(model, task_rotations, task_normals):
    powers = np.asarray(model["rotation_feature_powers"], dtype=int)
    descriptor_base, _ = descriptor_base_features(
        model.get("descriptor_kind", "rotation"),
        task_rotations,
        task_normals,
        global_field_name=model.get("global_field_name"),
    )
    raw = polynomial_features(descriptor_base, powers)
    feat_mean = np.asarray(model["feature_mean"], dtype=float).reshape(1, -1)
    feat_scale = np.asarray(model["feature_scale"], dtype=float).reshape(1, -1)
    return (raw - feat_mean) / feat_scale


def fit_joint_pca_transport_model(
    task_rotations,
    task_normals,
    updates,
    n_components=3,
    rotation_degree=3,
    coord_ridge=1e-3,
    descriptor_kind="rotation",
    global_field_name=None,
    eps=1e-12,
):
    task_rotations = np.asarray(task_rotations, dtype=float)
    task_normals = np.asarray(task_normals, dtype=float)
    updates = np.asarray(updates, dtype=float)
    if task_rotations.ndim != 3 or task_rotations.shape[1:] != (3, 3):
        raise ValueError("task_rotations must be an N x 3 x 3 array.")
    if task_normals.ndim != 2 or task_normals.shape[0] != task_rotations.shape[0] or task_normals.shape[1] != 3:
        raise ValueError("task_normals must be an N x 3 array with the same N as task_rotations.")
    if updates.ndim != 2 or updates.shape[0] != task_rotations.shape[0]:
        raise ValueError("Updates must be an N x P array with the same N as task_rotations.")

    mean_update = np.mean(updates, axis=0, keepdims=True)
    centered = updates - mean_update
    _, sing_vals, Vt = np.linalg.svd(centered, full_matrices=False)
    denom = max(centered.shape[0] - 1, 1)
    evals = (sing_vals ** 2) / denom
    total = np.sum(evals)
    var_ratio = evals / (total + eps) if total > 0 else np.zeros_like(evals)

    k = min(int(n_components), Vt.shape[0])
    basis = Vt[:k].T
    coords = centered @ basis

    descriptor_base, descriptor_names = descriptor_base_features(
        descriptor_kind,
        task_rotations,
        task_normals,
        global_field_name=global_field_name,
    )
    rotation_feature_powers = build_feature_powers(descriptor_base.shape[1], rotation_degree)
    features = polynomial_features(descriptor_base, rotation_feature_powers)
    reg_fit = fit_coordinate_ridge_regression(features, coords, coord_ridge=coord_ridge)
    pred_coords = np.asarray(reg_fit["coords_pred"], dtype=float)
    pred_updates = mean_update + pred_coords @ basis.T

    coord_r2_per_pc = []
    for pc_idx in range(k):
        coord_r2_per_pc.append(multi_r2_score(coords[:, [pc_idx]], pred_coords[:, [pc_idx]], eps=eps))

    topk_metrics = compute_shared_subspace_metrics(
        coords,
        pred_coords,
        basis,
        centered,
        eps=eps,
    )

    return {
        "descriptor_kind": str(descriptor_kind),
        "global_field_name": None if global_field_name is None else str(global_field_name),
        "representation_kind": "empirical_update",
        "mean_update": mean_update.reshape(-1),
        "basis": basis,
        "coords_true": coords,
        "coords_pred": pred_coords,
        "updates_pred": pred_updates,
        "coord_beta": np.asarray(reg_fit["coord_beta"], dtype=float),
        "coord_rank": int(reg_fit["coord_rank"]),
        "rotation_degree": int(rotation_degree),
        "coord_ridge": float(coord_ridge),
        "rotation_feature_powers": np.asarray(rotation_feature_powers, dtype=int),
        "feature_names": polynomial_feature_names(descriptor_names, rotation_feature_powers),
        "descriptor_base_feature_names": np.asarray(descriptor_names, dtype=object),
        "feature_mean": np.asarray(reg_fit["feature_mean"], dtype=float),
        "feature_scale": np.asarray(reg_fit["feature_scale"], dtype=float),
        "evals": evals,
        "var_ratio": var_ratio,
        "n_components": int(k),
        "coord_r2_per_pc": np.asarray(coord_r2_per_pc, dtype=float),
        "coord_r2_overall": multi_r2_score(coords, pred_coords, eps=eps),
        "update_r2_overall": multi_r2_score(updates, pred_updates, eps=eps),
        "update_cosine_mean": float(np.mean(cosine_similarity_rows(updates, pred_updates))),
        "update_rmse": float(np.sqrt(np.mean((updates - pred_updates) ** 2))),
        "coord_r2_topk": np.asarray(topk_metrics["coord_r2_topk"], dtype=float),
        "subspace_update_r2_topk": np.asarray(topk_metrics["update_r2_topk"], dtype=float),
        "subspace_update_cosine_topk": np.asarray(topk_metrics["update_cosine_topk"], dtype=float),
        "subspace_update_rmse_topk": np.asarray(topk_metrics["update_rmse_topk"], dtype=float),
        "subspace_energy_frac_topk": np.asarray(topk_metrics["energy_frac_topk"], dtype=float),
    }


def fit_joint_kpca_transport_model(
    task_rotations,
    task_normals,
    similarity_gram,
    n_components=3,
    rotation_degree=3,
    coord_ridge=1e-3,
    representation_kind="inf_bias_rkhs",
    descriptor_kind="rotation",
    global_field_name=None,
    eps=1e-12,
):
    task_rotations = np.asarray(task_rotations, dtype=float)
    task_normals = np.asarray(task_normals, dtype=float)
    similarity_gram = np.asarray(similarity_gram, dtype=float)
    if task_rotations.ndim != 3 or task_rotations.shape[1:] != (3, 3):
        raise ValueError("task_rotations must be an N x 3 x 3 array.")
    if task_normals.ndim != 2 or task_normals.shape[0] != task_rotations.shape[0] or task_normals.shape[1] != 3:
        raise ValueError("task_normals must be an N x 3 array with the same N as task_rotations.")
    if similarity_gram.ndim != 2 or similarity_gram.shape[0] != similarity_gram.shape[1]:
        raise ValueError("similarity_gram must be square.")
    if similarity_gram.shape[0] != task_rotations.shape[0]:
        raise ValueError("similarity_gram size must match the number of task rotations.")

    G = 0.5 * (similarity_gram + similarity_gram.T)
    n_tasks = G.shape[0]
    H = np.eye(n_tasks) - np.ones((n_tasks, n_tasks), dtype=float) / float(n_tasks)
    Gc = H @ G @ H
    evals, evecs = np.linalg.eigh(Gc)
    order = np.argsort(evals)[::-1]
    evals = np.asarray(evals[order], dtype=float)
    evecs = np.asarray(evecs[:, order], dtype=float)
    pos_mask = evals > eps
    evals_pos = evals[pos_mask]
    evecs_pos = evecs[:, pos_mask]
    if evals_pos.size == 0:
        raise ValueError("No positive kernel PCA eigenvalues found in the similarity Gram matrix.")

    full_coords_true = evecs_pos * np.sqrt(evals_pos[None, :] + eps)
    total = float(np.sum(evals_pos))
    var_ratio = evals_pos / (total + eps) if total > 0 else np.zeros_like(evals_pos)

    k = min(int(n_components), full_coords_true.shape[1])
    coords = full_coords_true[:, :k]
    descriptor_base, descriptor_names = descriptor_base_features(
        descriptor_kind,
        task_rotations,
        task_normals,
        global_field_name=global_field_name,
    )
    rotation_feature_powers = build_feature_powers(descriptor_base.shape[1], rotation_degree)
    features = polynomial_features(descriptor_base, rotation_feature_powers)
    reg_fit = fit_coordinate_ridge_regression(features, coords, coord_ridge=coord_ridge)
    pred_coords = np.asarray(reg_fit["coords_pred"], dtype=float)

    full_coords_pred = np.zeros_like(full_coords_true)
    full_coords_pred[:, :k] = pred_coords
    basis = np.zeros((full_coords_true.shape[1], k), dtype=float)
    if k > 0:
        basis[:k, :k] = np.eye(k, dtype=float)

    coord_r2_per_pc = []
    for pc_idx in range(k):
        coord_r2_per_pc.append(multi_r2_score(coords[:, [pc_idx]], pred_coords[:, [pc_idx]], eps=eps))

    topk_metrics = compute_shared_subspace_metrics_from_full_coords(
        full_coords_true,
        pred_coords,
        eps=eps,
    )

    return {
        "representation_kind": str(representation_kind),
        "descriptor_kind": str(descriptor_kind),
        "global_field_name": None if global_field_name is None else str(global_field_name),
        "mean_update": np.zeros((full_coords_true.shape[1],), dtype=float),
        "basis": basis,
        "coords_true": coords,
        "coords_pred": pred_coords,
        "updates_pred": full_coords_pred,
        "full_coords_true": full_coords_true,
        "full_coords_pred": full_coords_pred,
        "coord_beta": np.asarray(reg_fit["coord_beta"], dtype=float),
        "coord_rank": int(reg_fit["coord_rank"]),
        "rotation_degree": int(rotation_degree),
        "coord_ridge": float(coord_ridge),
        "rotation_feature_powers": np.asarray(rotation_feature_powers, dtype=int),
        "feature_names": polynomial_feature_names(descriptor_names, rotation_feature_powers),
        "descriptor_base_feature_names": np.asarray(descriptor_names, dtype=object),
        "feature_mean": np.asarray(reg_fit["feature_mean"], dtype=float),
        "feature_scale": np.asarray(reg_fit["feature_scale"], dtype=float),
        "evals": np.asarray(evals_pos, dtype=float),
        "var_ratio": np.asarray(var_ratio, dtype=float),
        "n_components": int(k),
        "coord_r2_per_pc": np.asarray(coord_r2_per_pc, dtype=float),
        "coord_r2_overall": multi_r2_score(coords, pred_coords, eps=eps),
        "update_r2_overall": multi_r2_score(full_coords_true, full_coords_pred, eps=eps),
        "update_cosine_mean": float(np.mean(cosine_similarity_rows(full_coords_true, full_coords_pred, eps=eps))),
        "update_rmse": float(np.sqrt(np.mean((full_coords_true - full_coords_pred) ** 2))),
        "coord_r2_topk": np.asarray(topk_metrics["coord_r2_topk"], dtype=float),
        "subspace_update_r2_topk": np.asarray(topk_metrics["update_r2_topk"], dtype=float),
        "subspace_update_cosine_topk": np.asarray(topk_metrics["update_cosine_topk"], dtype=float),
        "subspace_update_rmse_topk": np.asarray(topk_metrics["update_rmse_topk"], dtype=float),
        "subspace_energy_frac_topk": np.asarray(topk_metrics["energy_frac_topk"], dtype=float),
        "kpca_evecs": np.asarray(evecs_pos, dtype=float),
        "train_similarity_gram": G,
        "train_similarity_col_mean": np.mean(G, axis=0),
        "train_similarity_mean": float(np.mean(G)),
    }


def predict_transport_coords(model, task_rotations, task_normals=None):
    task_rotations = np.asarray(task_rotations, dtype=float)
    if task_normals is None:
        task_normals = task_rotations[:, :, 2]
    features_std = evaluate_rotation_features(model, task_rotations, task_normals)
    design = np.column_stack([np.ones(features_std.shape[0], dtype=float), features_std])
    return design @ np.asarray(model["coord_beta"], dtype=float)


def predict_transport_updates(model, task_rotations, task_normals=None):
    coords = predict_transport_coords(model, task_rotations, task_normals=task_normals)
    basis = np.asarray(model["basis"], dtype=float)
    mean_update = np.asarray(model["mean_update"], dtype=float)
    return mean_update[None, :] + coords @ basis.T


def project_cross_similarity_to_train_kpca(model, cross_similarity, eps=1e-12):
    cross_similarity = np.asarray(cross_similarity, dtype=float)
    if cross_similarity.ndim != 2:
        raise ValueError("cross_similarity must be a 2D array.")
    col_mean = np.asarray(model["train_similarity_col_mean"], dtype=float)[None, :]
    row_mean = np.mean(cross_similarity, axis=1, keepdims=True)
    grand_mean = float(model["train_similarity_mean"])
    centered_cross = cross_similarity - col_mean - row_mean + grand_mean
    evecs = np.asarray(model["kpca_evecs"], dtype=float)
    evals = np.asarray(model["evals"], dtype=float)
    return centered_cross @ evecs / np.sqrt(evals[None, :] + eps)


def transport_coord_jacobian(model, task_rotation, eps=1e-5):
    task_rotation = np.asarray(task_rotation, dtype=float)
    if task_rotation.shape != (3, 3):
        raise ValueError("task_rotation must be a 3 x 3 matrix.")
    base_normal = task_rotation[:, 2][None, :]
    base_coords = predict_transport_coords(model, task_rotation[None, :, :], task_normals=base_normal)[0]
    jac = np.zeros((base_coords.shape[0], 3), dtype=float)
    for axis_idx in range(3):
        step = np.zeros(3, dtype=float)
        step[axis_idx] = eps
        plus = (SciRotation.from_rotvec(step) * SciRotation.from_matrix(task_rotation)).as_matrix()
        minus = (SciRotation.from_rotvec(-step) * SciRotation.from_matrix(task_rotation)).as_matrix()
        coords_plus = predict_transport_coords(model, plus[None, :, :], task_normals=plus[None, :, 2])[0]
        coords_minus = predict_transport_coords(model, minus[None, :, :], task_normals=minus[None, :, 2])[0]
        jac[:, axis_idx] = (coords_plus - coords_minus) / (2.0 * eps)
    return jac


def rotation_geodesic_distance(start_rotation, end_rotation):
    start_rotation = np.asarray(start_rotation, dtype=float)
    end_rotation = np.asarray(end_rotation, dtype=float)
    rel = SciRotation.from_matrix(end_rotation) * SciRotation.from_matrix(start_rotation).inv()
    return float(np.linalg.norm(rel.as_rotvec()))


def rotation_geodesic_path(start_rotation, end_rotation, n_steps=129):
    key_rots = SciRotation.from_matrix(
        np.asarray([start_rotation, end_rotation], dtype=float)
    )
    slerp = Slerp([0.0, 1.0], key_rots)
    ts = np.linspace(0.0, 1.0, int(n_steps))
    return slerp(ts).as_matrix()


def local_bias_transport_rate(model, task_rotation, generator_rotvec):
    generator_rotvec = np.asarray(generator_rotvec, dtype=float).reshape(3)
    coord_rate = transport_coord_jacobian(model, task_rotation) @ generator_rotvec
    bias_rate = np.asarray(model["basis"], dtype=float) @ coord_rate
    return bias_rate, coord_rate


def integrate_transport_along_geodesic(model, start_rotation, end_rotation, start_update, n_steps=257):
    path = rotation_geodesic_path(start_rotation, end_rotation, n_steps=n_steps)
    pred = np.asarray(start_update, dtype=float).copy()
    prev_coords = predict_transport_coords(model, path[0:1], task_normals=path[0:1, :, 2])[0]
    for current_rotation in path[1:]:
        curr_coords = predict_transport_coords(model, current_rotation[None, :, :], task_normals=current_rotation[None, :, 2])[0]
        pred = pred + np.asarray(model["basis"], dtype=float) @ (curr_coords - prev_coords)
        prev_coords = curr_coords
    return pred, path


def choose_farthest_pair(task_rotations):
    task_rotations = np.asarray(task_rotations, dtype=float)
    best_pair = (0, min(1, len(task_rotations) - 1))
    best_dist = -np.inf
    for i in range(len(task_rotations)):
        for j in range(i + 1, len(task_rotations)):
            dist = rotation_geodesic_distance(task_rotations[i], task_rotations[j])
            if dist > best_dist:
                best_dist = dist
                best_pair = (i, j)
    return best_pair


def evaluate_model_on_pool(model, pooled):
    if pooled["task_rotations"].shape[0] == 0:
        return None
    coords_pred = predict_transport_coords(model, pooled["task_rotations"], task_normals=pooled["normals"])
    pred_updates = predict_transport_updates(model, pooled["task_rotations"], task_normals=pooled["normals"])
    coords_true = (pooled["updates"] - model["mean_update"][None, :]) @ model["basis"]
    coord_r2_per_pc = []
    for pc_idx in range(coords_true.shape[1]):
        coord_r2_per_pc.append(multi_r2_score(coords_true[:, [pc_idx]], coords_pred[:, [pc_idx]]))
    centered_updates = pooled["updates"] - model["mean_update"][None, :]
    topk_metrics = compute_shared_subspace_metrics(
        coords_true,
        coords_pred,
        model["basis"],
        centered_updates,
    )
    return {
        "coords_true": coords_true,
        "coords_pred": coords_pred,
        "updates_pred": pred_updates,
        "coord_r2_per_pc": np.asarray(coord_r2_per_pc, dtype=float),
        "coord_r2_overall": multi_r2_score(coords_true, coords_pred),
        "update_r2_overall": multi_r2_score(pooled["updates"], pred_updates),
        "update_cosine_mean": float(np.mean(cosine_similarity_rows(pooled["updates"], pred_updates))),
        "update_rmse": float(np.sqrt(np.mean((pooled["updates"] - pred_updates) ** 2))),
        "coord_r2_topk": np.asarray(topk_metrics["coord_r2_topk"], dtype=float),
        "subspace_update_r2_topk": np.asarray(topk_metrics["update_r2_topk"], dtype=float),
        "subspace_update_cosine_topk": np.asarray(topk_metrics["update_cosine_topk"], dtype=float),
        "subspace_update_rmse_topk": np.asarray(topk_metrics["update_rmse_topk"], dtype=float),
        "subspace_energy_frac_topk": np.asarray(topk_metrics["energy_frac_topk"], dtype=float),
    }


def format_vector_text(vec):
    vec = np.asarray(vec, dtype=float).reshape(-1)
    return ",".join(f"{float(value):.6f}" for value in vec)


def summarize_eval_by_run(split, pooled, eval_result, model):
    if pooled is None or eval_result is None or pooled["normals"].shape[0] == 0:
        return []

    coords_true = np.asarray(eval_result["coords_true"], dtype=float)
    coords_pred = np.asarray(eval_result["coords_pred"], dtype=float)
    updates_true = np.asarray(pooled["updates"], dtype=float)
    updates_pred = np.asarray(eval_result["updates_pred"], dtype=float)
    source_runs = np.asarray(pooled["source_run"], dtype=int)
    rotation_axes = np.asarray(pooled["rotation_axes"], dtype=float)
    source_dirs = np.asarray(pooled["source_dirs"], dtype=object)

    rows = []
    for run_id in np.unique(source_runs):
        mask = source_runs == run_id
        first_idx = int(np.where(mask)[0][0])
        base = {
            "split": split,
            "source_run": int(run_id),
            "rotation_axis": format_vector_text(rotation_axes[first_idx]),
            "source_dir": str(source_dirs[first_idx]),
            "n_tasks": int(np.sum(mask)),
        }
        rows.extend(
            [
                {
                    **base,
                    "metric": "coord_r2_overall",
                    "value": float(multi_r2_score(coords_true[mask], coords_pred[mask])),
                },
                {
                    **base,
                    "metric": "update_r2_overall",
                    "value": float(multi_r2_score(updates_true[mask], updates_pred[mask])),
                },
                {
                    **base,
                    "metric": "update_cosine_mean",
                    "value": float(np.mean(cosine_similarity_rows(updates_true[mask], updates_pred[mask]))),
                },
                {
                    **base,
                    "metric": "update_rmse",
                    "value": float(np.sqrt(np.mean((updates_true[mask] - updates_pred[mask]) ** 2))),
                },
            ]
        )
        for pc_idx in range(coords_true.shape[1]):
            rows.append(
                {
                    **base,
                    "metric": f"coord_r2_pc{pc_idx + 1}",
                    "value": float(
                        multi_r2_score(
                            coords_true[mask, [pc_idx]],
                            coords_pred[mask, [pc_idx]],
                        )
                    ),
                }
            )
        centered_updates = updates_true[mask] - np.asarray(model["mean_update"], dtype=float)[None, :]
        topk_metrics = compute_shared_subspace_metrics(
            coords_true[mask],
            coords_pred[mask],
            np.asarray(model["basis"], dtype=float),
            centered_updates,
        )
        for topk_idx, value in enumerate(np.asarray(topk_metrics["coord_r2_topk"], dtype=float), start=1):
            rows.append({**base, "metric": f"coord_r2_top{topk_idx}", "value": float(value)})
        for topk_idx, value in enumerate(np.asarray(topk_metrics["update_r2_topk"], dtype=float), start=1):
            rows.append({**base, "metric": f"subspace_update_r2_top{topk_idx}", "value": float(value)})
        for topk_idx, value in enumerate(np.asarray(topk_metrics["update_cosine_topk"], dtype=float), start=1):
            rows.append({**base, "metric": f"subspace_update_cosine_top{topk_idx}", "value": float(value)})
        for topk_idx, value in enumerate(np.asarray(topk_metrics["energy_frac_topk"], dtype=float), start=1):
            rows.append({**base, "metric": f"subspace_energy_frac_top{topk_idx}", "value": float(value)})
    return rows


def subset_pool_rows(pooled, mask):
    if pooled is None:
        return None
    mask = np.asarray(mask, dtype=bool)
    n_rows = int(np.asarray(pooled["task_rotations"]).shape[0])
    out = {}
    for key, value in pooled.items():
        if key == "runs":
            source_runs = np.asarray(pooled["source_run"], dtype=int)[mask]
            out["runs"] = [pooled["runs"][int(run_id)] for run_id in np.unique(source_runs)]
            continue
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == n_rows:
            out[key] = value[mask]
        else:
            out[key] = value
    return out


def independent_axis_diagnostics(
    train_pool,
    *,
    update_kind,
    n_components,
    rotation_degree,
    coord_ridge,
    descriptor_kind,
    output_dir,
):
    train_pool = train_pool or {}
    if "task_rotations" not in train_pool or np.asarray(train_pool["task_rotations"]).size == 0:
        return []

    rows = []
    source_runs = np.asarray(train_pool["source_run"], dtype=int)
    for run_id in np.unique(source_runs):
        mask = source_runs == run_id
        axis_pool = subset_pool_rows(train_pool, mask)
        axis_model = fit_joint_pca_transport_model(
            axis_pool["task_rotations"],
            axis_pool["normals"],
            axis_pool["updates"],
            n_components=n_components,
            rotation_degree=rotation_degree,
            coord_ridge=coord_ridge,
            descriptor_kind=descriptor_kind,
            global_field_name=axis_pool["runs"][0].get("global_field_name") if axis_pool.get("runs") else None,
        )
        first_idx = int(np.where(mask)[0][0])
        base = {
            "source_run": int(run_id),
            "rotation_axis": format_vector_text(train_pool["rotation_axes"][first_idx]),
            "source_dir": str(train_pool["source_dirs"][first_idx]),
            "n_tasks": int(np.sum(mask)),
            "rotation_degree": int(rotation_degree),
            "coord_ridge": float(coord_ridge),
            "n_components": int(n_components),
        }
        rows.extend(
            [
                {**base, "metric": "coord_r2_overall", "value": float(axis_model["coord_r2_overall"])},
                {**base, "metric": "update_r2_overall", "value": float(axis_model["update_r2_overall"])},
                {**base, "metric": "update_cosine_mean", "value": float(axis_model["update_cosine_mean"])},
                {**base, "metric": "update_rmse", "value": float(axis_model["update_rmse"])},
            ]
        )
        for pc_idx, value in enumerate(np.asarray(axis_model["coord_r2_per_pc"], dtype=float), start=1):
            rows.append({**base, "metric": f"coord_r2_pc{pc_idx}", "value": float(value)})
        for topk_idx, value in enumerate(np.asarray(axis_model["coord_r2_topk"], dtype=float), start=1):
            rows.append({**base, "metric": f"coord_r2_top{topk_idx}", "value": float(value)})
        for topk_idx, value in enumerate(np.asarray(axis_model["subspace_update_r2_topk"], dtype=float), start=1):
            rows.append({**base, "metric": f"subspace_update_r2_top{topk_idx}", "value": float(value)})
        for topk_idx, value in enumerate(np.asarray(axis_model["subspace_update_cosine_topk"], dtype=float), start=1):
            rows.append({**base, "metric": f"subspace_update_cosine_top{topk_idx}", "value": float(value)})
        for topk_idx, value in enumerate(np.asarray(axis_model["subspace_energy_frac_topk"], dtype=float), start=1):
            rows.append({**base, "metric": f"subspace_energy_frac_top{topk_idx}", "value": float(value)})

        stem = output_dir / f"{update_kind}_independent_axis{int(run_id)}"
        plot_coordinate_fit(
            axis_model,
            holdout_eval=None,
            save_path=stem.with_name(stem.name + "_coord_fit.png"),
        )
        plot_transport_coordinate_profiles(
            axis_model,
            axis_pool,
            holdout_pool=None,
            holdout_eval=None,
            save_path=stem.with_name(stem.name + "_coord_profiles.png"),
        )
        export_transport_coordinate_rows(
            axis_model,
            axis_pool,
            holdout_pool=None,
            holdout_eval=None,
            save_path=stem.with_name(stem.name + "_coords.csv"),
        )
    return rows


def plot_coordinate_fit(model, holdout_eval=None, save_path=None):
    if plt is None:
        return None

    coords_true = np.asarray(model["coords_true"], dtype=float)
    coords_pred = np.asarray(model["coords_pred"], dtype=float)
    k = min(3, coords_true.shape[1])
    fig, axes = plt.subplots(1, k, figsize=(5.0 * k, 4.6))
    if k == 1:
        axes = [axes]

    for idx in range(k):
        ax = axes[idx]
        y_true = coords_true[:, idx]
        y_pred = coords_pred[:, idx]
        ax.scatter(y_true, y_pred, s=28, alpha=0.65, color="#4c72b0", edgecolors="none", label="Train")
        if holdout_eval is not None:
            hold_true = np.asarray(holdout_eval["coords_true"], dtype=float)[:, idx]
            hold_pred = np.asarray(holdout_eval["coords_pred"], dtype=float)[:, idx]
            ax.scatter(
                hold_true,
                hold_pred,
                s=32,
                alpha=0.85,
                color="#c44e52",
                marker="^",
                edgecolors="none",
                label="Holdout",
            )
        lims = np.asarray([np.min([ax.get_xlim()[0], ax.get_ylim()[0]]), np.max([ax.get_xlim()[1], ax.get_ylim()[1]])], dtype=float)
        ax.plot(lims, lims, "k--", lw=1.2, alpha=0.8)
        ax.set_xlabel(f"True PC{idx + 1}")
        ax.set_ylabel(f"Predicted PC{idx + 1}")
        ax.set_title(
            f"PC{idx + 1} fit\ntrain $R^2$={model['coord_r2_per_pc'][idx]:.3f}"
        )
        ax.grid(True, alpha=0.22)
        if idx == 0 and holdout_eval is not None:
            ax.legend(frameon=False, loc="best")

    fig.suptitle("Joint PCA coordinate regression for bias transport", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fig


def _cyclic_color_norm(values):
    values = np.asarray(values, dtype=float)
    if Normalize is None:
        return None
    if values.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmin >= -1e-9 and vmax <= 2.0 * np.pi + 1e-9:
        return Normalize(vmin=0.0, vmax=2.0 * np.pi)
    return Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1.0)


def _set_equal_3d_limits(ax, points, pad=0.08):
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    radius = max(radius, 1e-6)
    radius *= (1.0 + float(pad))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _sample_task_indices(n_tasks, n_show):
    n_tasks = int(n_tasks)
    n_show = max(1, int(n_show))
    if n_tasks <= n_show:
        return np.arange(n_tasks, dtype=int)
    return np.unique(np.linspace(0, n_tasks - 1, n_show, dtype=int))


def plot_task_geometry_overview(
    train_pool,
    holdout_pool=None,
    *,
    tasks_per_run=2,
    save_path=None,
):
    if plt is None:
        return None

    fig = plt.figure(figsize=(8.8, 7.6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    tmu._draw_unit_sphere_wireframe(ax, color="0.86", alpha=0.35)
    phi = np.linspace(0.0, 2.0 * np.pi, 240)
    train_colors = plt.get_cmap("tab10")

    plotted_points = []
    plotted_handles = []

    def _plot_pool(pool, *, is_holdout):
        if pool is None or np.asarray(pool.get("normals", np.zeros((0, 3)))).size == 0:
            return
        source_runs = np.asarray(pool["source_run"], dtype=int)
        rotation_axes = np.asarray(pool["rotation_axes"], dtype=float)
        normals = np.asarray(pool["normals"], dtype=float)
        frame_v1 = np.asarray(pool["frame_v1"], dtype=float)
        frame_v2 = np.asarray(pool["frame_v2"], dtype=float)
        for run_id in np.unique(source_runs):
            mask = source_runs == run_id
            idx_all = np.where(mask)[0]
            idx_show = idx_all[_sample_task_indices(len(idx_all), tasks_per_run)]
            axis_vec = unit_normalize(rotation_axes[idx_all[0]])
            color = "black" if is_holdout else train_colors(int(run_id % 10))
            line_alpha = 0.95 if is_holdout else 0.50
            line_lw = 2.2 if is_holdout else 1.2
            axis_lw = 2.3 if is_holdout else 1.8
            axis_label = "Holdout axis" if is_holdout else f"Train axis {format_axis_short(axis_vec)}"

            ax.quiver(
                0.0, 0.0, 0.0,
                axis_vec[0], axis_vec[1], axis_vec[2],
                color=color,
                linewidth=axis_lw,
                arrow_length_ratio=0.10,
            )
            plotted_points.append(axis_vec)
            if not plotted_handles or axis_label not in [h.get_label() for h in plotted_handles]:
                plotted_handles.append(ax.plot([], [], [], color=color, lw=axis_lw, label=axis_label)[0])

            for local_j, task_idx in enumerate(idx_show):
                circle = np.asarray(
                    tmu.circle_points_from_phi(
                        normals[task_idx],
                        phi,
                        frame=(frame_v1[task_idx], frame_v2[task_idx]),
                    ),
                    dtype=float,
                )
                ax.plot(
                    circle[:, 0],
                    circle[:, 1],
                    circle[:, 2],
                    color=color,
                    lw=line_lw,
                    alpha=line_alpha if local_j > 0 else min(1.0, line_alpha + 0.1),
                )
                anchor = circle[0]
                ax.scatter(
                    [anchor[0]],
                    [anchor[1]],
                    [anchor[2]],
                    color=color,
                    s=28 if is_holdout else 18,
                    alpha=0.95,
                    marker="o" if is_holdout else ".",
                )
                plotted_points.append(circle)

    _plot_pool(train_pool, is_holdout=False)
    _plot_pool(holdout_pool, is_holdout=True)

    if plotted_points:
        _set_equal_3d_limits(ax, np.vstack([np.reshape(p, (-1, 3)) for p in plotted_points]), pad=0.12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Task geometry on the sphere\nSampled great circles from train families; holdout family in black")
    if plotted_handles:
        ax.legend(handles=plotted_handles, frameon=False, loc="upper left")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fig


def plot_phi_anchor_overview(train_pool, holdout_pool=None, save_path=None):
    if plt is None:
        return None

    fig = plt.figure(figsize=(8.8, 7.6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    tmu._draw_unit_sphere_wireframe(ax, color="0.86", alpha=0.35)
    train_colors = plt.get_cmap("tab10")
    plotted_points = []
    legend_handles = []

    def _plot_pool(pool, *, is_holdout):
        if pool is None or np.asarray(pool.get("frame_v1", np.zeros((0, 3)))).size == 0:
            return
        anchors = np.asarray(pool["frame_v1"], dtype=float)
        tangents = np.asarray(pool["frame_v2"], dtype=float)
        source_runs = np.asarray(pool["source_run"], dtype=int)
        colors = np.asarray(pool["color_values"], dtype=float)
        rotation_axes = np.asarray(pool["rotation_axes"], dtype=float)
        for run_id in np.unique(source_runs):
            mask = source_runs == run_id
            order = np.argsort(colors[mask])
            idx = np.where(mask)[0][order]
            axis_vec = unit_normalize(rotation_axes[idx[0]])
            color = "black" if is_holdout else train_colors(int(run_id % 10))
            label = "Holdout phi=0 path" if is_holdout else f"Train axis {format_axis_short(axis_vec)}"
            ax.plot(
                anchors[idx, 0],
                anchors[idx, 1],
                anchors[idx, 2],
                color=color,
                lw=2.2 if is_holdout else 1.6,
                alpha=0.95 if is_holdout else 0.75,
            )
            ax.scatter(
                anchors[idx, 0],
                anchors[idx, 1],
                anchors[idx, 2],
                color=color,
                s=20 if is_holdout else 14,
                alpha=0.95,
            )
            step = max(1, len(idx) // 6)
            for j in idx[::step]:
                tangent = unit_normalize(tangents[j]) * 0.12
                ax.quiver(
                    anchors[j, 0],
                    anchors[j, 1],
                    anchors[j, 2],
                    tangent[0],
                    tangent[1],
                    tangent[2],
                    color=color,
                    linewidth=1.0,
                    arrow_length_ratio=0.28,
                    alpha=0.7,
                )
            plotted_points.append(anchors[idx])
            if label not in [h.get_label() for h in legend_handles]:
                legend_handles.append(ax.plot([], [], [], color=color, lw=2.0, label=label)[0])

    _plot_pool(train_pool, is_holdout=False)
    _plot_pool(holdout_pool, is_holdout=True)
    if plotted_points:
        _set_equal_3d_limits(ax, np.vstack(plotted_points), pad=0.12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(r"Phase anchor $x_R(0)$ moves smoothly across each rotation family")
    if legend_handles:
        ax.legend(handles=legend_handles, frameon=False, loc="upper left")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fig


def plot_single_family_frame_trace(pool, *, source_run=None, save_path=None):
    if plt is None or pool is None:
        return None
    source_runs = np.asarray(pool.get("source_run", np.zeros((0,), dtype=int)), dtype=int)
    if source_runs.size == 0:
        return None
    if source_run is None:
        source_run = int(np.unique(source_runs)[0])
    mask = source_runs == int(source_run)
    if not np.any(mask):
        return None

    colors = np.asarray(pool["color_values"], dtype=float)[mask]
    order = np.argsort(colors)
    theta = colors[order]
    normals = np.asarray(pool["normals"], dtype=float)[mask][order]
    anchors = np.asarray(pool["frame_v1"], dtype=float)[mask][order]
    axis_vec = unit_normalize(np.asarray(pool["rotation_axes"], dtype=float)[mask][0])

    fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.2), sharex=True)
    component_specs = [
        (axes[0], normals, "Normal components", ("$n_x$", "$n_y$", "$n_z$")),
        (axes[1], anchors, r"Phase-anchor components $v_1(\theta)=x_R(0)$", ("$v_{1x}$", "$v_{1y}$", "$v_{1z}$")),
    ]
    colors_xyz = ["#c44e52", "#55a868", "#4c72b0"]

    for ax, values, title, labels in component_specs:
        for dim_idx, (label, color) in enumerate(zip(labels, colors_xyz)):
            ax.plot(theta, values[:, dim_idx], marker="o", ms=3.5, lw=1.8, color=color, label=label)
        ax.set_ylabel("Component value")
        ax.set_title(title)
        ax.grid(True, alpha=0.22)
        ax.legend(frameon=False, loc="best", ncol=3)

    axes[1].set_xlabel("Task angle / distance label")
    fig.suptitle(
        f"Smooth frame variation along one rotation family\naxis {format_axis_short(axis_vec)}",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fig


def plot_transport_coordinate_profiles(model, train_pool, holdout_pool=None, holdout_eval=None, save_path=None):
    if plt is None:
        return None

    k = int(np.asarray(model["basis"], dtype=float).shape[1])
    ncols = 2 if holdout_pool is not None and holdout_eval is not None else 1
    fig, axes = plt.subplots(k, ncols, figsize=(6.2 * ncols, 3.6 * k), squeeze=False)
    cmap = plt.get_cmap("twilight_shifted")
    markers = ["o", "s", "^", "D", "P", "X"]

    train_true = np.asarray(model["coords_true"], dtype=float)
    train_pred = np.asarray(model["coords_pred"], dtype=float)
    train_colors = np.asarray(train_pool["color_values"], dtype=float)
    train_runs = np.asarray(train_pool["source_run"], dtype=int)
    train_norm = _cyclic_color_norm(train_colors)

    hold_true = hold_pred = hold_colors = hold_runs = hold_norm = None
    if holdout_pool is not None and holdout_eval is not None:
        hold_true = np.asarray(holdout_eval["coords_true"], dtype=float)
        hold_pred = np.asarray(holdout_eval["coords_pred"], dtype=float)
        hold_colors = np.asarray(holdout_pool["color_values"], dtype=float)
        hold_runs = np.asarray(holdout_pool["source_run"], dtype=int)
        hold_norm = _cyclic_color_norm(hold_colors)

    unique_train_runs = np.unique(train_runs)
    unique_hold_runs = np.unique(hold_runs) if hold_runs is not None else np.asarray([], dtype=int)

    def _panel(ax, pc_idx, *, coords_true, coords_pred, colors, runs, run_ids, norm, title, pred_label):
        for local_idx, run_id in enumerate(run_ids):
            mask = runs == run_id
            order = np.argsort(colors[mask])
            x = colors[mask][order]
            y_true = coords_true[mask, pc_idx][order]
            y_pred = coords_pred[mask, pc_idx][order]
            marker = markers[local_idx % len(markers)]
            ax.plot(x, y_pred, color="black", lw=1.7, alpha=0.95, label=pred_label if local_idx == 0 else None)
            ax.scatter(
                x,
                y_true,
                c=x,
                cmap=cmap,
                norm=norm,
                s=38,
                alpha=0.9,
                marker=marker,
                edgecolors="none",
                label=f"Axis {int(run_id)} true" if local_idx < 3 else None,
            )
        ax.set_title(title)
        ax.set_xlabel("Task angle / distance label")
        ax.set_ylabel(f"PC{pc_idx + 1} coordinate")
        ax.grid(True, alpha=0.22)

    for pc_idx in range(k):
        train_r2 = float(np.asarray(model["coord_r2_per_pc"], dtype=float)[pc_idx])
        _panel(
            axes[pc_idx, 0],
            pc_idx,
            coords_true=train_true,
            coords_pred=train_pred,
            colors=train_colors,
            runs=train_runs,
            run_ids=unique_train_runs,
            norm=train_norm,
            title=f"Train axes: PC{pc_idx + 1} fit ($R^2$={train_r2:.3f})",
            pred_label="Predicted function",
        )
        if ncols > 1:
            hold_r2 = float(np.asarray(holdout_eval["coord_r2_per_pc"], dtype=float)[pc_idx])
            _panel(
                axes[pc_idx, 1],
                pc_idx,
                coords_true=hold_true,
                coords_pred=hold_pred,
                colors=hold_colors,
                runs=hold_runs,
                run_ids=unique_hold_runs,
                norm=hold_norm,
                title=f"Holdout axes: PC{pc_idx + 1} prediction ($R^2$={hold_r2:.3f})",
                pred_label="Predicted function",
            )
        for col in range(ncols):
            if axes[pc_idx, col].get_legend_handles_labels()[0]:
                axes[pc_idx, col].legend(frameon=False, loc="best")

    cax = fig.add_axes([0.92, 0.17, 0.015, 0.68])
    sm = plt.cm.ScalarMappable(norm=train_norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Task angle / distance label")
    fig.suptitle("Smooth transport functions in shared PCA coordinates", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 0.90, 0.98])
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fig


def plot_transport_manifold_predictions(model, train_pool, holdout_pool=None, holdout_eval=None, save_path=None):
    if plt is None:
        return None

    coords_train_true = np.asarray(model["coords_true"], dtype=float)
    coords_train_pred = np.asarray(model["coords_pred"], dtype=float)
    if coords_train_true.shape[1] < 3:
        coords_train_true = np.pad(coords_train_true, ((0, 0), (0, 3 - coords_train_true.shape[1])), mode="constant")
        coords_train_pred = np.pad(coords_train_pred, ((0, 0), (0, 3 - coords_train_pred.shape[1])), mode="constant")

    cmap = plt.get_cmap("twilight_shifted")
    colors_train = np.asarray(train_pool["color_values"], dtype=float)
    norm = _cyclic_color_norm(colors_train)

    fig = plt.figure(figsize=(13.5, 6.0))
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")

    train_runs = np.asarray(train_pool["source_run"], dtype=int)
    for run_id in np.unique(train_runs):
        mask = train_runs == run_id
        order = np.argsort(colors_train[mask])
        idx = np.where(mask)[0][order]
        ax0.plot(
            coords_train_true[idx, 0],
            coords_train_true[idx, 1],
            coords_train_true[idx, 2],
            color="0.55",
            lw=1.0,
            alpha=0.35,
        )
    ax0.scatter(
        coords_train_true[:, 0],
        coords_train_true[:, 1],
        coords_train_true[:, 2],
        c=colors_train,
        cmap=cmap,
        norm=norm,
        s=34,
        alpha=0.92,
        edgecolors="none",
        label="Train true",
    )
    ax0.scatter(
        coords_train_pred[:, 0],
        coords_train_pred[:, 1],
        coords_train_pred[:, 2],
        c=colors_train,
        cmap=cmap,
        norm=norm,
        s=26,
        alpha=0.78,
        marker="x",
        label="Train predicted",
    )
    ax0.set_title("Training axes in shared PCA space")
    ax0.set_xlabel("PC1")
    ax0.set_ylabel("PC2")
    ax0.set_zlabel("PC3")
    ax0.legend(frameon=False, loc="best")

    if holdout_pool is not None and holdout_eval is not None:
        coords_hold_true = np.asarray(holdout_eval["coords_true"], dtype=float)
        coords_hold_pred = np.asarray(holdout_eval["coords_pred"], dtype=float)
        if coords_hold_true.shape[1] < 3:
            coords_hold_true = np.pad(coords_hold_true, ((0, 0), (0, 3 - coords_hold_true.shape[1])), mode="constant")
            coords_hold_pred = np.pad(coords_hold_pred, ((0, 0), (0, 3 - coords_hold_pred.shape[1])), mode="constant")
        colors_hold = np.asarray(holdout_pool["color_values"], dtype=float)
        hold_runs = np.asarray(holdout_pool["source_run"], dtype=int)
        for run_id in np.unique(hold_runs):
            mask = hold_runs == run_id
            order = np.argsort(colors_hold[mask])
            idx = np.where(mask)[0][order]
            ax1.plot(
                coords_hold_true[idx, 0],
                coords_hold_true[idx, 1],
                coords_hold_true[idx, 2],
                color="0.55",
                lw=1.0,
                alpha=0.35,
            )
            ax1.plot(
                coords_hold_pred[idx, 0],
                coords_hold_pred[idx, 1],
                coords_hold_pred[idx, 2],
                color="black",
                lw=1.4,
                alpha=0.85,
            )
        ax1.scatter(
            coords_hold_true[:, 0],
            coords_hold_true[:, 1],
            coords_hold_true[:, 2],
            c=colors_hold,
            cmap=cmap,
            norm=norm,
            s=42,
            alpha=0.92,
            marker="^",
            edgecolors="none",
            label="Holdout true",
        )
        ax1.scatter(
            coords_hold_pred[:, 0],
            coords_hold_pred[:, 1],
            coords_hold_pred[:, 2],
            c=colors_hold,
            cmap=cmap,
            norm=norm,
            s=28,
            alpha=0.82,
            marker="o",
            facecolors="none",
            linewidths=1.0,
            label="Holdout predicted",
        )
        hold_title = (
            "Holdout axis prediction in shared PCA space\n"
            f"overall $R^2$={holdout_eval['coord_r2_overall']:.3f}"
        )
    else:
        hold_title = "No holdout axis provided"
    ax1.set_title(hold_title)
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("PC3")
    ax1.legend(frameon=False, loc="best")

    cax = fig.add_axes([0.92, 0.18, 0.015, 0.66])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Task angle / distance label")
    fig.suptitle("Shared PCA transport manifold: train fit and holdout prediction", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 0.90, 0.95])
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fig


def plot_shared_pca_spectrum(model, save_path=None, max_components_display=25):
    if plt is None:
        return None

    evals = np.asarray(model.get("evals", np.zeros((0,), dtype=float)), dtype=float)
    evals = evals[evals > 1e-12]
    if evals.size == 0:
        return None
    frac = evals / np.sum(evals)
    cum = np.cumsum(frac)
    xs = np.arange(1, len(frac) + 1, dtype=int)
    k80 = int(np.searchsorted(cum, 0.80) + 1)
    pr = float(tmu.participation_ratio(evals))
    n_show = min(len(frac), max(int(max_components_display), int(k80)))
    xs_show = xs[:n_show]
    frac_show = frac[:n_show]
    cum_show = cum[:n_show]

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.8))
    line_var, = ax.plot(xs_show, frac_show, marker="o", lw=1.9, color="#4c72b0", label="Variance fraction")
    ax.set_xlabel("Shared PCA component")
    ax.set_ylabel("Variance fraction")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.22)
    ax.set_xlim(1, xs_show[-1])

    ax2 = ax.twinx()
    line_cum, = ax2.plot(xs_show, cum_show, marker="s", lw=1.7, color="#c44e52", label="Cumulative variance")
    ax2.axhline(0.80, color="0.25", lw=1.1, ls="--", label="_nolegend_")
    ax2.axvline(k80, color="0.25", lw=1.1, ls=":", label="_nolegend_")
    ax2.set_ylabel("Cumulative variance")
    ax2.set_ylim(0.0, 1.02)

    xticks = list(ax.get_xticks())
    if 1 <= k80 <= xs_show[-1]:
        xticks.append(k80)
    xticks = np.unique(np.asarray(np.round(xticks), dtype=int))
    xticks = xticks[(xticks >= 1) & (xticks <= xs_show[-1])]
    ax.set_xticks(xticks)
    ax.legend([line_var, line_cum], ["Variance fraction", "Cumulative variance"], frameon=False, loc="best")
    ax.set_title(
        f"Shared update spectrum\nPR={pr:.2f}, 80% variance by PC {k80}, rank={len(evals)}"
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fig


def plot_shared_pca_geometry(model, train_pool, holdout_pool=None, holdout_eval=None, save_path=None):
    if plt is None:
        return None

    coords_train = np.asarray(model["coords_true"], dtype=float)
    if coords_train.shape[1] < 3:
        coords_train = np.pad(coords_train, ((0, 0), (0, 3 - coords_train.shape[1])), mode="constant")
    colors_train = np.asarray(train_pool["color_values"], dtype=float)
    norm = _cyclic_color_norm(colors_train)
    cmap = plt.get_cmap("twilight_shifted")

    fig = plt.figure(figsize=(8.0, 6.8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    train_runs = np.asarray(train_pool["source_run"], dtype=int)
    for run_id in np.unique(train_runs):
        mask = train_runs == run_id
        order = np.argsort(colors_train[mask])
        idx = np.where(mask)[0][order]
        ax.plot(
            coords_train[idx, 0],
            coords_train[idx, 1],
            coords_train[idx, 2],
            color="0.65",
            lw=1.0,
            alpha=0.35,
        )
    ax.scatter(
        coords_train[:, 0],
        coords_train[:, 1],
        coords_train[:, 2],
        c=colors_train,
        cmap=cmap,
        norm=norm,
        s=34,
        alpha=0.92,
        edgecolors="none",
        label="Train",
    )

    points_for_limits = [coords_train]
    if holdout_pool is not None and holdout_eval is not None:
        coords_hold = np.asarray(holdout_eval["coords_true"], dtype=float)
        if coords_hold.shape[1] < 3:
            coords_hold = np.pad(coords_hold, ((0, 0), (0, 3 - coords_hold.shape[1])), mode="constant")
        colors_hold = np.asarray(holdout_pool["color_values"], dtype=float)
        hold_runs = np.asarray(holdout_pool["source_run"], dtype=int)
        for run_id in np.unique(hold_runs):
            mask = hold_runs == run_id
            order = np.argsort(colors_hold[mask])
            idx = np.where(mask)[0][order]
            ax.plot(
                coords_hold[idx, 0],
                coords_hold[idx, 1],
                coords_hold[idx, 2],
                color="#1f77b4",
                lw=1.0,
                alpha=0.30,
            )
        ax.scatter(
            coords_hold[:, 0],
            coords_hold[:, 1],
            coords_hold[:, 2],
            c=colors_hold,
            cmap=cmap,
            norm=norm,
            s=42,
            alpha=0.92,
            marker="^",
            edgecolors="none",
            label="Holdout",
        )
        points_for_limits.append(coords_hold)

    _set_equal_3d_limits(ax, np.vstack(points_for_limits), pad=0.10)
    vr = np.asarray(model.get("var_ratio", np.zeros((0,), dtype=float)), dtype=float)
    ax.set_xlabel(f"PC1 ({100.0 * vr[0]:.1f}%)" if len(vr) > 0 else "PC1")
    ax.set_ylabel(f"PC2 ({100.0 * vr[1]:.1f}%)" if len(vr) > 1 else "PC2")
    ax.set_zlabel(f"PC3 ({100.0 * vr[2]:.1f}%)" if len(vr) > 2 else "PC3")
    ax.set_title("Bias updates in the shared PCA subspace")
    ax.legend(frameon=False, loc="best")
    cax = fig.add_axes([0.92, 0.18, 0.015, 0.64])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Task angle / distance label")
    fig.tight_layout(rect=[0, 0, 0.90, 1.0])
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fig


def plot_topk_transport_summary(model, train_eval, holdout_eval=None, save_path=None):
    if plt is None:
        return None

    train_r2 = np.asarray(train_eval["subspace_update_r2_topk"], dtype=float)
    train_cos = np.asarray(train_eval["subspace_update_cosine_topk"], dtype=float)
    train_energy = np.asarray(train_eval["subspace_energy_frac_topk"], dtype=float)
    ks = np.arange(1, len(train_r2) + 1, dtype=int)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    metric_specs = [
        (
            axes[0],
            train_energy,
            None if holdout_eval is None else np.asarray(holdout_eval["subspace_energy_frac_topk"], dtype=float),
            "Update energy inside the shared train subspace",
            "Fraction of total update energy",
            (0.0, 1.05),
        ),
        (
            axes[1],
            train_r2,
            None if holdout_eval is None else np.asarray(holdout_eval["subspace_update_r2_topk"], dtype=float),
            "Top-k subspace transport $R^2$",
            "$R^2$",
            None,
        ),
        (
            axes[2],
            train_cos,
            None if holdout_eval is None else np.asarray(holdout_eval["subspace_update_cosine_topk"], dtype=float),
            "Directional agreement in that subspace",
            "Cosine(true, predicted) after projection",
            (-0.05, 1.05),
        ),
    ]

    for ax, train_vals, hold_vals, title, ylabel, ylim in metric_specs:
        ax.plot(ks, train_vals, color="#4c72b0", marker="o", lw=2.0, label="Train")
        if hold_vals is not None and hold_vals.size:
            ax.plot(ks, hold_vals, color="#c44e52", marker="^", lw=2.0, label="Holdout")
        ax.set_title(title)
        ax.set_xlabel("Top-k shared train PCs")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ks)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.22)
        ax.legend(frameon=False, loc="best")

    var_ratio = np.asarray(model.get("var_ratio", np.zeros((0,), dtype=float)), dtype=float)
    subtitle_bits = []
    for idx in range(min(len(ks), len(var_ratio))):
        subtitle_bits.append(f"PC{idx + 1}: {100.0 * float(var_ratio[idx]):.1f}%")
    subtitle = " | ".join(subtitle_bits)
    fig.suptitle(
        "Shared-subspace transport quality across top-k modes"
        + (f"\nTrain PCA variance: {subtitle}" if subtitle else ""),
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fig


def export_transport_coordinate_rows(model, train_pool, holdout_pool=None, holdout_eval=None, save_path=None):
    rows = []

    def _append_rows(split, pool, coords_true, coords_pred):
        if pool is None:
            return
        for idx in range(coords_true.shape[0]):
            row = {
                "split": split,
                "source_run": int(pool["source_run"][idx]),
                "task_index": int(pool["task_index"][idx]),
                "color_value": float(pool["color_values"][idx]),
            }
            for pc_idx in range(coords_true.shape[1]):
                row[f"pc{pc_idx + 1}_true"] = float(coords_true[idx, pc_idx])
                row[f"pc{pc_idx + 1}_pred"] = float(coords_pred[idx, pc_idx])
            rows.append(row)

    _append_rows("train", train_pool, np.asarray(model["coords_true"], dtype=float), np.asarray(model["coords_pred"], dtype=float))
    if holdout_pool is not None and holdout_eval is not None:
        _append_rows("holdout", holdout_pool, np.asarray(holdout_eval["coords_true"], dtype=float), np.asarray(holdout_eval["coords_pred"], dtype=float))

    if save_path is not None:
        write_csv_rows(save_path, list(rows[0].keys()) if rows else [], rows)
    return rows


def task_rotation_display_coords(pooled):
    if pooled is None or np.asarray(pooled.get("task_rotations", np.zeros((0, 3, 3)))).size == 0:
        return np.zeros((0, 3), dtype=float)
    if all(str(run.get("task_family", "")) == "single_axis" for run in pooled.get("runs", [])):
        axes = np.asarray(pooled["rotation_axes"], dtype=float)
        axis_norms = np.linalg.norm(axes, axis=1, keepdims=True)
        axis_norms = np.where(axis_norms > 1e-12, axis_norms, 1.0)
        unit_axes = axes / axis_norms
        return unit_axes * np.asarray(pooled["color_values"], dtype=float)[:, None]
    return SciRotation.from_matrix(np.asarray(pooled["task_rotations"], dtype=float)).as_rotvec()


def choose_cross_split_pair(train_rotations, holdout_rotations):
    train_rotations = np.asarray(train_rotations, dtype=float)
    holdout_rotations = np.asarray(holdout_rotations, dtype=float)
    best_pair = (0, 0)
    best_dist = -np.inf
    for i in range(len(train_rotations)):
        for j in range(len(holdout_rotations)):
            dist = rotation_geodesic_distance(train_rotations[i], holdout_rotations[j])
            if dist > best_dist:
                best_dist = dist
                best_pair = (i, j)
    return best_pair


def plot_rotation_transport_path(
    model,
    train_pool,
    *,
    source_index,
    target_index,
    holdout_pool=None,
    path_steps=257,
    save_path=None,
):
    train_rotations = np.asarray(train_pool["task_rotations"], dtype=float)
    source_rotation = train_rotations[int(source_index)]
    target_rotation = train_rotations[int(target_index)]
    source_update = np.asarray(train_pool["updates"], dtype=float)[int(source_index)]
    integrated_target_update, path = integrate_transport_along_geodesic(
        model,
        source_rotation,
        target_rotation,
        source_update,
        n_steps=path_steps,
    )
    direct_target_update = predict_transport_updates(model, target_rotation[None, :, :])[0]
    true_target_update = np.asarray(train_pool["updates"], dtype=float)[int(target_index)]
    source_coords = predict_transport_coords(model, source_rotation[None, :, :])[0]
    target_coords = predict_transport_coords(model, target_rotation[None, :, :])[0]
    anchored_target_update = source_update + np.asarray(model["basis"], dtype=float) @ (target_coords - source_coords)

    out = {
        "source_index": int(source_index),
        "target_index": int(target_index),
        "path": path,
        "direct_target_update": direct_target_update,
        "integrated_target_update": integrated_target_update,
        "anchored_target_update": anchored_target_update,
        "true_target_update": true_target_update,
    }
    if plt is None:
        return out

    fig = plt.figure(figsize=(14.0, 6.0))
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")

    train_points = task_rotation_display_coords(train_pool)
    run_ids = np.asarray(train_pool["source_run"], dtype=int)
    cmap_runs = plt.get_cmap("tab10")
    unique_runs = np.unique(run_ids)
    for run_id in unique_runs:
        mask = run_ids == run_id
        ax0.scatter(
            train_points[mask, 0],
            train_points[mask, 1],
            train_points[mask, 2],
            s=28,
            alpha=0.9,
            color=cmap_runs(int(run_id % 10)),
            label=f"Train axis {int(run_id)}",
        )
    if holdout_pool is not None and holdout_pool["task_rotations"].size:
        hold_points = task_rotation_display_coords(holdout_pool)
        ax0.scatter(
            hold_points[:, 0],
            hold_points[:, 1],
            hold_points[:, 2],
            s=38,
            alpha=0.9,
            color="#c44e52",
            marker="^",
            label="Holdout tasks",
        )
    path_points = SciRotation.from_matrix(path).as_rotvec()
    ax0.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], color="black", lw=2.2, alpha=0.85)
    ax0.scatter([train_points[int(source_index), 0]], [train_points[int(source_index), 1]], [train_points[int(source_index), 2]], color="#2f6b2f", s=95, marker="o", label="Source")
    ax0.scatter([train_points[int(target_index), 0]], [train_points[int(target_index), 1]], [train_points[int(target_index), 2]], color="#7f1d1d", s=95, marker="*", label="Target")
    ax0.set_title("Rotation-task coordinates with source-to-target SO(3) path")
    ax0.set_xlabel(r"$\omega_x$")
    ax0.set_ylabel(r"$\omega_y$")
    ax0.set_zlabel(r"$\omega_z$")
    ax0.legend(frameon=False, loc="upper left")
    ax0.set_box_aspect((1, 1, 1))

    path_ts = np.linspace(0.0, 1.0, path.shape[0])
    path_coords = predict_transport_coords(model, path)
    for pc_idx in range(min(3, path_coords.shape[1])):
        ax1.plot(path_ts, path_coords[:, pc_idx], lw=2.0, label=f"PC{pc_idx + 1}")
    ax1.set_title("Predicted PCA coordinates along the SO(3) transport path")
    ax1.set_xlabel("Path parameter")
    ax1.set_ylabel("Predicted coordinate")
    ax1.grid(True, alpha=0.22)
    ax1.legend(frameon=False, loc="best")

    title = (
        "Rotation-based smooth bias transport\n"
        f"direct endpoint cos={cosine_similarity_rows(true_target_update[None, :], direct_target_update[None, :])[0]:.3f}, "
        f"anchored endpoint cos={cosine_similarity_rows(true_target_update[None, :], anchored_target_update[None, :])[0]:.3f}"
    )
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_cross_split_transport_summary(
    model,
    train_pool,
    train_eval,
    *,
    source_index,
    holdout_pool,
    holdout_eval,
    holdout_target_index,
    path_steps=257,
    save_path=None,
):
    if plt is None or holdout_pool is None or holdout_eval is None:
        return None

    source_index = int(source_index)
    holdout_target_index = int(holdout_target_index)
    train_rotations = np.asarray(train_pool["task_rotations"], dtype=float)
    holdout_rotations = np.asarray(holdout_pool["task_rotations"], dtype=float)
    source_rotation = train_rotations[source_index]
    target_rotation = holdout_rotations[holdout_target_index]
    path = rotation_geodesic_path(source_rotation, target_rotation, n_steps=path_steps)
    path_coords = predict_transport_coords(model, path)

    source_update_true = np.asarray(train_pool["updates"], dtype=float)[source_index]
    source_update_pred = np.asarray(train_eval["updates_pred"], dtype=float)[source_index]
    source_coords_true = np.asarray(train_eval["coords_true"], dtype=float)[source_index]
    source_coords_pred = np.asarray(train_eval["coords_pred"], dtype=float)[source_index]

    target_update_true = np.asarray(holdout_pool["updates"], dtype=float)[holdout_target_index]
    target_update_pred = np.asarray(holdout_eval["updates_pred"], dtype=float)[holdout_target_index]
    target_coords_true = np.asarray(holdout_eval["coords_true"], dtype=float)[holdout_target_index]
    target_coords_pred = np.asarray(holdout_eval["coords_pred"], dtype=float)[holdout_target_index]

    k_plot = min(
        6,
        int(path_coords.shape[1]),
        int(len(source_coords_true)),
        int(len(source_coords_pred)),
        int(len(target_coords_true)),
        int(len(target_coords_pred)),
    )
    if k_plot <= 0:
        return None
    pc_axis = np.arange(1, k_plot + 1, dtype=int)

    source_top2_cos = float(
        cosine_similarity_rows(
            source_coords_true[None, : min(2, len(source_coords_true))],
            source_coords_pred[None, : min(2, len(source_coords_pred))],
        )[0]
    )
    target_top2_cos = float(
        cosine_similarity_rows(
            target_coords_true[None, : min(2, len(target_coords_true))],
            target_coords_pred[None, : min(2, len(target_coords_pred))],
        )[0]
    )
    source_full_cos = float(cosine_similarity_rows(source_update_true[None, :], source_update_pred[None, :])[0])
    target_full_cos = float(cosine_similarity_rows(target_update_true[None, :], target_update_pred[None, :])[0])

    fig = plt.figure(figsize=(14.0, 10.5))
    ax_geom = fig.add_subplot(2, 2, 1, projection="3d")
    ax_path = fig.add_subplot(2, 2, 2)
    ax_source = fig.add_subplot(2, 2, 3)
    ax_target = fig.add_subplot(2, 2, 4)

    tmu._draw_unit_sphere_wireframe(ax_geom, color="0.86", alpha=0.35)
    phi = np.linspace(0.0, 2.0 * np.pi, 240)
    source_normal = np.asarray(train_pool["normals"], dtype=float)[source_index]
    source_frame = (
        np.asarray(train_pool["frame_v1"], dtype=float)[source_index],
        np.asarray(train_pool["frame_v2"], dtype=float)[source_index],
    )
    target_normal = np.asarray(holdout_pool["normals"], dtype=float)[holdout_target_index]
    target_frame = (
        np.asarray(holdout_pool["frame_v1"], dtype=float)[holdout_target_index],
        np.asarray(holdout_pool["frame_v2"], dtype=float)[holdout_target_index],
    )
    source_circle = np.asarray(tmu.circle_points_from_phi(source_normal, phi, frame=source_frame), dtype=float)
    target_circle = np.asarray(tmu.circle_points_from_phi(target_normal, phi, frame=target_frame), dtype=float)
    path_normals = np.asarray(path[:, :, 2], dtype=float)
    ax_geom.plot(source_circle[:, 0], source_circle[:, 1], source_circle[:, 2], color="#2f6b2f", lw=2.2, label="Source train circle")
    ax_geom.plot(target_circle[:, 0], target_circle[:, 1], target_circle[:, 2], color="#1f77b4", lw=2.2, label="Target holdout circle")
    ax_geom.plot(path_normals[:, 0], path_normals[:, 1], path_normals[:, 2], color="black", lw=1.8, alpha=0.9, label="Rotation path (normal view)")
    ax_geom.scatter([source_normal[0]], [source_normal[1]], [source_normal[2]], color="#2f6b2f", s=70)
    ax_geom.scatter([target_normal[0]], [target_normal[1]], [target_normal[2]], color="#1f77b4", s=78, marker="^")
    _set_equal_3d_limits(ax_geom, np.vstack([source_circle, target_circle, path_normals]), pad=0.12)
    ax_geom.set_title("Selected transport from a train task to a holdout task")
    ax_geom.set_xlabel("x")
    ax_geom.set_ylabel("y")
    ax_geom.set_zlabel("z")
    ax_geom.legend(frameon=False, loc="upper left")

    train_coords = np.asarray(train_eval["coords_true"], dtype=float)
    holdout_coords = np.asarray(holdout_eval["coords_true"], dtype=float)
    ax_path.scatter(train_coords[:, 0], train_coords[:, 1], s=18, color="0.70", alpha=0.6, label="Train tasks")
    ax_path.scatter(holdout_coords[:, 0], holdout_coords[:, 1], s=28, color="#9ecae1", alpha=0.85, marker="^", label="Holdout tasks")
    ax_path.plot(path_coords[:, 0], path_coords[:, 1], color="black", lw=2.0, alpha=0.95, label="Predicted transport path")
    step = max(1, path_coords.shape[0] // 12)
    for idx in range(0, path_coords.shape[0] - step, step):
        dx = path_coords[idx + step, 0] - path_coords[idx, 0]
        dy = path_coords[idx + step, 1] - path_coords[idx, 1]
        ax_path.arrow(
            path_coords[idx, 0],
            path_coords[idx, 1],
            dx,
            dy,
            color="black",
            alpha=0.65,
            width=0.0,
            length_includes_head=True,
            head_width=0.015 * max(1.0, np.max(np.ptp(path_coords[:, :2], axis=0))),
        )
    ax_path.scatter([source_coords_true[0]], [source_coords_true[1]], color="#2f6b2f", s=60, label="Source true")
    ax_path.scatter([target_coords_true[0]], [target_coords_true[1]], color="#1f77b4", s=70, marker="^", label="Target true")
    ax_path.scatter([target_coords_pred[0]], [target_coords_pred[1]], color="#c44e52", s=70, marker="x", label="Target predicted")
    ax_path.set_xlabel("PC1")
    ax_path.set_ylabel("PC2")
    ax_path.set_title("Inferred flow field along the selected transport path")
    ax_path.grid(True, alpha=0.22)
    ax_path.legend(frameon=False, loc="best")

    def _plot_selected(ax, label, coords_true, coords_pred, full_cos, top2_cos, color):
        ax.plot(pc_axis, coords_true, marker="o", lw=2.0, color=color, label=f"{label} true")
        ax.plot(pc_axis, coords_pred, marker="s", lw=1.8, ls="--", color="black", label=f"{label} predicted")
        ax.set_xlabel("Shared PCA coordinate")
        ax.set_ylabel("Coordinate value")
        ax.set_xticks(pc_axis[:k_plot])
        ax.set_xlim(1, k_plot)
        ax.grid(True, alpha=0.22)
        ax.legend(frameon=False, loc="best")
        ax.text(
            0.03,
            0.97,
            f"full-space cosine = {full_cos:.3f}\nshared top-2 cosine = {top2_cos:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.92),
        )

    _plot_selected(ax_source, "Source train task", source_coords_true[:k_plot], source_coords_pred[:k_plot], source_full_cos, source_top2_cos, "#2f6b2f")
    ax_source.set_title("Selected train-task transport fit")
    _plot_selected(ax_target, "Target holdout task", target_coords_true[:k_plot], target_coords_pred[:k_plot], target_full_cos, target_top2_cos, "#1f77b4")
    ax_target.set_title("Selected holdout-task transport prediction")

    fig.suptitle("Rotation-based bias transport: selected train-to-holdout example", fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "source_index": source_index,
        "holdout_target_index": holdout_target_index,
        "source_full_cos": source_full_cos,
        "target_full_cos": target_full_cos,
        "source_top2_cos": source_top2_cos,
        "target_top2_cos": target_top2_cos,
    }


def fit_bias_transport_model_from_results(
    *,
    train_results_dirs,
    holdout_results_dirs=None,
    update_kind="bias",
    n_components=3,
    rotation_degree=3,
    coord_ridge=1e-3,
    descriptor_kind="rotation",
    solo_inf_bias_reg=1e-10,
    solo_inf_full_reg=1e-10,
    ridge_mode="trace",
    path_steps=257,
    source_index=None,
    target_index=None,
    holdout_target_index=None,
    output_dir=None,
):
    train_results_dirs = [Path(path) for path in train_results_dirs]
    holdout_results_dirs = [Path(path) for path in (holdout_results_dirs or [])]
    if output_dir is None:
        train_tag = "__".join(path.name for path in train_results_dirs[:3])
        if len(train_results_dirs) > 3:
            train_tag += f"__plus{len(train_results_dirs) - 3}"
        output_dir = DEFAULT_OUTPUT_ROOT / (
            f"{update_kind}_{descriptor_kind}_{tmu.sanitize_name_token(train_tag)}_rotdeg{int(rotation_degree)}_ridge{tmu.sanitize_name_token(f'{float(coord_ridge):.0e}')}"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if update_kind in {"inf_bias_rkhs", "inf_full_rkhs"}:
        inf_kind = "bias" if update_kind == "inf_bias_rkhs" else "full"
        solo_inf_reg = float(solo_inf_bias_reg if inf_kind == "bias" else solo_inf_full_reg)
        train_runs = [
            reconstruct_infinite_width_rkhs_run(
                path,
                which=inf_kind,
                solo_inf_reg=solo_inf_reg,
                ridge_mode=ridge_mode,
            )
            for path in train_results_dirs
        ]
        _check_run_compatibility(train_runs)
        if descriptor_kind in {"global_field_coeffs", "global_field_fourier"} and str(train_runs[0].get("target_mode")) != "global_field":
            raise ValueError(f"--task-descriptor {descriptor_kind} requires runs with target_mode=global_field.")
        train_pool = pool_reconstructed_run_geometry(train_runs)
        if train_pool["task_rotations"].shape[0] == 0:
            raise ValueError("No training tasks were loaded.")
        train_similarity = build_similarity_matrix_blocks(train_runs)
        model = fit_joint_kpca_transport_model(
            train_pool["task_rotations"],
            train_pool["normals"],
            train_similarity,
            n_components=n_components,
            rotation_degree=rotation_degree,
            coord_ridge=coord_ridge,
            representation_kind=update_kind,
            descriptor_kind=descriptor_kind,
            global_field_name=train_runs[0].get("global_field_name") if train_runs else None,
        )
        train_pool["updates"] = np.asarray(model["full_coords_true"], dtype=float)
        train_eval = evaluate_model_on_pool(model, train_pool)

        holdout_pool = None
        holdout_eval = None
        if holdout_results_dirs:
            holdout_runs = [
                reconstruct_infinite_width_rkhs_run(
                    path,
                    which=inf_kind,
                    solo_inf_reg=solo_inf_reg,
                    ridge_mode=ridge_mode,
                )
                for path in holdout_results_dirs
            ]
            _check_run_compatibility(train_runs + holdout_runs)
            holdout_pool = pool_reconstructed_run_geometry(holdout_runs)
            if holdout_pool["task_rotations"].shape[0] > 0:
                holdout_cross = build_similarity_matrix_blocks(train_runs, holdout_runs)
                holdout_pool["updates"] = project_cross_similarity_to_train_kpca(model, holdout_cross)
                holdout_eval = evaluate_model_on_pool(model, holdout_pool)
    else:
        train_pool = pool_results_dirs(train_results_dirs, update_kind=update_kind)
        if train_pool["task_rotations"].shape[0] == 0:
            raise ValueError("No training tasks were loaded.")
        if descriptor_kind in {"global_field_coeffs", "global_field_fourier"} and str(train_pool["runs"][0].get("target_mode")) != "global_field":
            raise ValueError(f"--task-descriptor {descriptor_kind} requires runs with target_mode=global_field.")
        for run in train_pool["runs"]:
            if int(run.get("task_representation_version", 1)) < 2:
                raise ValueError(
                    f"{run['results_dir']} uses task representation v1. "
                    "Re-run taskManifold_update.py so task_update_vectors.npz includes the saved transported rotations."
                )
        model = fit_joint_pca_transport_model(
            train_pool["task_rotations"],
            train_pool["normals"],
            train_pool["updates"],
            n_components=n_components,
            rotation_degree=rotation_degree,
            coord_ridge=coord_ridge,
            descriptor_kind=descriptor_kind,
            global_field_name=train_pool["runs"][0].get("global_field_name") if train_pool.get("runs") else None,
        )
        train_eval = evaluate_model_on_pool(model, train_pool)
        holdout_pool = pool_results_dirs(holdout_results_dirs, update_kind=update_kind) if holdout_results_dirs else None

        holdout_eval = None
        if holdout_pool is not None and holdout_pool["task_rotations"].shape[0] > 0:
            for run in holdout_pool["runs"]:
                if int(run.get("task_representation_version", 1)) < 2:
                    raise ValueError(
                        f"{run['results_dir']} uses task representation v1. "
                        "Re-run taskManifold_update.py so task_update_vectors.npz includes the saved transported rotations."
                    )
            holdout_eval = evaluate_model_on_pool(model, holdout_pool)
    axis_metric_rows = summarize_eval_by_run("train", train_pool, train_eval, model)
    axis_metric_rows.extend(summarize_eval_by_run("holdout", holdout_pool, holdout_eval, model))

    if holdout_eval is not None and holdout_pool is not None and holdout_pool["task_rotations"].shape[0] > 0:
        if source_index is None or holdout_target_index is None:
            source_index_auto, holdout_target_index_auto = choose_cross_split_pair(
                train_pool["task_rotations"],
                holdout_pool["task_rotations"],
            )
            if source_index is None:
                source_index = source_index_auto
            if holdout_target_index is None:
                holdout_target_index = holdout_target_index_auto
    if source_index is None or target_index is None:
        source_index_train_auto, target_index_auto = choose_farthest_pair(train_pool["task_rotations"])
        if source_index is None:
            source_index = source_index_train_auto
        if target_index is None:
            target_index = target_index_auto

    flow_data = plot_rotation_transport_path(
        model,
        train_pool,
        source_index=source_index,
        target_index=target_index,
        holdout_pool=holdout_pool,
        path_steps=path_steps,
        save_path=output_dir / f"{update_kind}_transport_rotation_path.png",
    )
    flow_field_data = plot_cross_split_transport_summary(
        model,
        train_pool,
        train_eval,
        source_index=source_index,
        holdout_pool=holdout_pool,
        holdout_eval=holdout_eval,
        holdout_target_index=holdout_target_index if holdout_target_index is not None else 0,
        path_steps=path_steps,
        save_path=output_dir / f"{update_kind}_transport_flow_field.png",
    )
    plot_task_geometry_overview(
        train_pool,
        holdout_pool=holdout_pool,
        save_path=output_dir / f"{update_kind}_transport_task_geometry.png",
    )
    plot_single_family_frame_trace(
        train_pool,
        save_path=output_dir / f"{update_kind}_transport_frame_trace.png",
    )
    plot_phi_anchor_overview(
        train_pool,
        holdout_pool=holdout_pool,
        save_path=output_dir / f"{update_kind}_transport_phi_anchors.png",
    )
    plot_shared_pca_geometry(
        model,
        train_pool,
        holdout_pool=holdout_pool,
        holdout_eval=holdout_eval,
        save_path=output_dir / f"{update_kind}_transport_shared_pca.png",
    )
    plot_shared_pca_spectrum(
        model,
        save_path=output_dir / f"{update_kind}_transport_spectrum.png",
    )
    plot_coordinate_fit(
        model,
        holdout_eval=holdout_eval,
        save_path=output_dir / f"{update_kind}_transport_coord_fit.png",
    )
    plot_transport_coordinate_profiles(
        model,
        train_pool,
        holdout_pool=holdout_pool,
        holdout_eval=holdout_eval,
        save_path=output_dir / f"{update_kind}_transport_coord_profiles.png",
    )
    plot_transport_manifold_predictions(
        model,
        train_pool,
        holdout_pool=holdout_pool,
        holdout_eval=holdout_eval,
        save_path=output_dir / f"{update_kind}_transport_manifold_predictions.png",
    )
    plot_topk_transport_summary(
        model,
        train_eval,
        holdout_eval=holdout_eval,
        save_path=output_dir / f"{update_kind}_transport_topk_summary.png",
    )
    export_transport_coordinate_rows(
        model,
        train_pool,
        holdout_pool=holdout_pool,
        holdout_eval=holdout_eval,
        save_path=output_dir / f"{update_kind}_transport_coords.csv",
    )
    independent_rows = independent_axis_diagnostics(
        train_pool,
        update_kind=update_kind,
        n_components=n_components,
        rotation_degree=rotation_degree,
        coord_ridge=coord_ridge,
        descriptor_kind=descriptor_kind,
        output_dir=output_dir,
    )

    direct_endpoint_cos = float(
        cosine_similarity_rows(
            flow_data["true_target_update"][None, :],
            flow_data["direct_target_update"][None, :],
        )[0]
    )
    integrated_endpoint_cos = float(
        cosine_similarity_rows(
            flow_data["true_target_update"][None, :],
            flow_data["anchored_target_update"][None, :],
        )[0]
    )
    summary_rows = [
        {
            "split": "train",
            "metric": "coord_r2_overall",
            "value": float(train_eval["coord_r2_overall"]),
        },
        {
            "split": "train",
            "metric": "update_r2_overall",
            "value": float(train_eval["update_r2_overall"]),
        },
        {
            "split": "train",
            "metric": "update_cosine_mean",
            "value": float(train_eval["update_cosine_mean"]),
        },
        {
            "split": "train",
            "metric": "update_rmse",
            "value": float(train_eval["update_rmse"]),
        },
        {
            "split": "train",
            "metric": "direct_endpoint_cosine",
            "value": direct_endpoint_cos,
        },
        {
            "split": "train",
            "metric": "integrated_endpoint_cosine",
            "value": integrated_endpoint_cos,
        },
    ]
    for idx, value in enumerate(np.asarray(train_eval["coord_r2_per_pc"], dtype=float), start=1):
        summary_rows.append(
            {
                "split": "train",
                "metric": f"coord_r2_pc{idx}",
                "value": float(value),
            }
        )
    for idx, value in enumerate(np.asarray(train_eval["coord_r2_topk"], dtype=float), start=1):
        summary_rows.append(
            {
                "split": "train",
                "metric": f"coord_r2_top{idx}",
                "value": float(value),
            }
        )
    for idx, value in enumerate(np.asarray(train_eval["subspace_update_r2_topk"], dtype=float), start=1):
        summary_rows.append(
            {
                "split": "train",
                "metric": f"subspace_update_r2_top{idx}",
                "value": float(value),
            }
        )
    for idx, value in enumerate(np.asarray(train_eval["subspace_update_cosine_topk"], dtype=float), start=1):
        summary_rows.append(
            {
                "split": "train",
                "metric": f"subspace_update_cosine_top{idx}",
                "value": float(value),
            }
        )
    for idx, value in enumerate(np.asarray(train_eval["subspace_energy_frac_topk"], dtype=float), start=1):
        summary_rows.append(
            {
                "split": "train",
                "metric": f"subspace_energy_frac_top{idx}",
                "value": float(value),
            }
        )
    if holdout_eval is not None:
        summary_rows.extend(
            [
                {
                    "split": "holdout",
                    "metric": "coord_r2_overall",
                    "value": float(holdout_eval["coord_r2_overall"]),
                },
                {
                    "split": "holdout",
                    "metric": "update_r2_overall",
                    "value": float(holdout_eval["update_r2_overall"]),
                },
                {
                    "split": "holdout",
                    "metric": "update_cosine_mean",
                    "value": float(holdout_eval["update_cosine_mean"]),
                },
                {
                    "split": "holdout",
                    "metric": "update_rmse",
                    "value": float(holdout_eval["update_rmse"]),
                },
            ]
        )
        for idx, value in enumerate(np.asarray(holdout_eval["coord_r2_per_pc"], dtype=float), start=1):
            summary_rows.append(
                {
                    "split": "holdout",
                    "metric": f"coord_r2_pc{idx}",
                    "value": float(value),
                }
            )
        for idx, value in enumerate(np.asarray(holdout_eval["coord_r2_topk"], dtype=float), start=1):
            summary_rows.append(
                {
                    "split": "holdout",
                    "metric": f"coord_r2_top{idx}",
                    "value": float(value),
                }
            )
        for idx, value in enumerate(np.asarray(holdout_eval["subspace_update_r2_topk"], dtype=float), start=1):
            summary_rows.append(
                {
                    "split": "holdout",
                    "metric": f"subspace_update_r2_top{idx}",
                    "value": float(value),
                }
            )
        for idx, value in enumerate(np.asarray(holdout_eval["subspace_update_cosine_topk"], dtype=float), start=1):
            summary_rows.append(
                {
                    "split": "holdout",
                    "metric": f"subspace_update_cosine_top{idx}",
                    "value": float(value),
                }
            )
        for idx, value in enumerate(np.asarray(holdout_eval["subspace_energy_frac_topk"], dtype=float), start=1):
            summary_rows.append(
                {
                    "split": "holdout",
                    "metric": f"subspace_energy_frac_top{idx}",
                    "value": float(value),
                }
            )

    write_csv_rows(
        output_dir / f"{update_kind}_transport_summary.csv",
        ["split", "metric", "value"],
        summary_rows,
    )
    write_csv_rows(
        output_dir / f"{update_kind}_transport_axis_metrics.csv",
        ["split", "source_run", "rotation_axis", "source_dir", "n_tasks", "metric", "value"],
        axis_metric_rows,
    )
    write_csv_rows(
        output_dir / f"{update_kind}_independent_axis_summary.csv",
        [
            "source_run",
            "rotation_axis",
            "source_dir",
            "n_tasks",
            "rotation_degree",
            "coord_ridge",
            "n_components",
            "metric",
            "value",
        ],
        independent_rows,
    )
    np.savez(
        output_dir / f"{update_kind}_transport_model.npz",
        representation_kind=np.asarray([str(model.get("representation_kind", update_kind))], dtype=object),
        mean_update=np.asarray(model["mean_update"], dtype=float),
        basis=np.asarray(model["basis"], dtype=float),
        coord_beta=np.asarray(model["coord_beta"], dtype=float),
        descriptor_kind=np.asarray([str(descriptor_kind)], dtype=object),
        rotation_degree=np.asarray([int(model["rotation_degree"])], dtype=int),
        coord_ridge=np.asarray([float(model["coord_ridge"])], dtype=float),
        solo_inf_bias_reg=np.asarray([float(solo_inf_bias_reg)], dtype=float),
        solo_inf_full_reg=np.asarray([float(solo_inf_full_reg)], dtype=float),
        ridge_mode=np.asarray([str(ridge_mode)], dtype=object),
        rotation_feature_powers=np.asarray(model["rotation_feature_powers"], dtype=int),
        feature_names=np.asarray(model["feature_names"], dtype=object),
        feature_mean=np.asarray(model["feature_mean"], dtype=float),
        feature_scale=np.asarray(model["feature_scale"], dtype=float),
        evals=np.asarray(model["evals"], dtype=float),
        var_ratio=np.asarray(model["var_ratio"], dtype=float),
        train_coord_r2_topk=np.asarray(train_eval["coord_r2_topk"], dtype=float),
        train_subspace_update_r2_topk=np.asarray(train_eval["subspace_update_r2_topk"], dtype=float),
        train_subspace_update_cosine_topk=np.asarray(train_eval["subspace_update_cosine_topk"], dtype=float),
        train_subspace_energy_frac_topk=np.asarray(train_eval["subspace_energy_frac_topk"], dtype=float),
        train_normals=np.asarray(train_pool["normals"], dtype=float),
        train_task_rotations=np.asarray(train_pool["task_rotations"], dtype=float),
        train_updates=np.asarray(train_pool["updates"], dtype=float),
        train_color_values=np.asarray(train_pool["color_values"], dtype=float),
        train_source_run=np.asarray(train_pool["source_run"], dtype=int),
        train_task_index=np.asarray(train_pool["task_index"], dtype=int),
        train_results_dirs=np.asarray([str(path.resolve()) for path in train_results_dirs]),
        holdout_results_dirs=np.asarray([str(path.resolve()) for path in holdout_results_dirs]),
        source_index=np.asarray([int(source_index)], dtype=int),
        target_index=np.asarray([int(target_index if target_index is not None else -1)], dtype=int),
        holdout_target_index=np.asarray([int(holdout_target_index if holdout_target_index is not None else -1)], dtype=int),
    )

    print(
        f"[{update_kind}] rotation degree={int(model['rotation_degree'])} | "
        f"coord ridge={float(model['coord_ridge']):.2e} | "
        f"train coord R2={train_eval['coord_r2_overall']:.4f} | "
        f"train update R2={train_eval['update_r2_overall']:.4f} | "
        f"train cosine={train_eval['update_cosine_mean']:.4f}"
    )
    if holdout_eval is not None:
        print(
            f"[{update_kind}] holdout coord R2={holdout_eval['coord_r2_overall']:.4f} | "
            f"holdout update R2={holdout_eval['update_r2_overall']:.4f} | "
            f"holdout cosine={holdout_eval['update_cosine_mean']:.4f}"
        )
    for split_name, eval_result in [("train", train_eval), ("holdout", holdout_eval)]:
        if eval_result is None:
            continue
        for topk_idx in range(len(np.asarray(eval_result["coord_r2_topk"], dtype=float))):
            print(
                f"[{update_kind}] {split_name} top{topk_idx + 1} "
                f"coord R2={float(eval_result['coord_r2_topk'][topk_idx]):.4f} | "
                f"subspace update R2={float(eval_result['subspace_update_r2_topk'][topk_idx]):.4f} | "
                f"subspace cosine={float(eval_result['subspace_update_cosine_topk'][topk_idx]):.4f} | "
                f"energy frac={float(eval_result['subspace_energy_frac_topk'][topk_idx]):.4f}"
            )
    for row in axis_metric_rows:
        if row["metric"] in {"coord_r2_overall", "update_r2_overall", "subspace_update_r2_top1", "subspace_update_r2_top2", "subspace_update_r2_top3"}:
            print(
                f"[{update_kind}] {row['split']} axis {row['source_run']} "
                f"({row['rotation_axis']}) {row['metric']}={row['value']:.4f}"
            )
    for row in independent_rows:
        if row["metric"] in {"coord_r2_overall", "update_r2_overall", "subspace_update_r2_top1", "subspace_update_r2_top2", "subspace_update_r2_top3"}:
            print(
                f"[{update_kind}] independent axis {row['source_run']} "
                f"({row['rotation_axis']}) {row['metric']}={row['value']:.4f}"
            )
    print(
        f"[{update_kind}] source={source_index} -> target={target_index} | "
        f"direct endpoint cosine={direct_endpoint_cos:.4f} | "
        f"anchored endpoint cosine={integrated_endpoint_cos:.4f}"
    )
    if flow_field_data is not None:
        print(
            f"[{update_kind}] selected train-to-holdout example: "
            f"train cosine={flow_field_data['source_full_cos']:.4f}, "
            f"holdout cosine={flow_field_data['target_full_cos']:.4f}, "
            f"holdout top2 cosine={flow_field_data['target_top2_cos']:.4f}"
        )
    print(f"Saved transport outputs to {output_dir}")

    return {
        "model": model,
        "train_eval": train_eval,
        "train_pool": train_pool,
        "holdout_pool": holdout_pool,
        "holdout_eval": holdout_eval,
        "axis_metric_rows": axis_metric_rows,
        "independent_axis_rows": independent_rows,
        "flow_data": flow_data,
        "flow_field_data": flow_field_data,
        "output_dir": output_dir,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Fit a pooled smooth transport model for empirical task updates saved by taskManifold_update.py."
        )
    )
    parser.add_argument(
        "--train-results-dirs",
        type=str,
        required=True,
        help="Comma-separated list of results directories used to fit the shared transport model.",
    )
    parser.add_argument(
        "--holdout-results-dirs",
        type=str,
        default=None,
        help="Optional comma-separated list of held-out results directories used only for evaluation.",
    )
    parser.add_argument(
        "--update-kind",
        choices=["bias", "full", "inf_bias_rkhs", "inf_full_rkhs"],
        default="bias",
        help="Which update/correction representation to model.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=3,
        help="Number of PCA transport coordinates to fit.",
    )
    parser.add_argument(
        "--rotation-degree",
        type=int,
        choices=[1, 2, 3, 4],
        default=3,
        help="Maximum monomial degree in the rotation-matrix chart F(R).",
    )
    parser.add_argument(
        "--coord-ridge",
        type=float,
        default=1e-3,
        help="Ridge penalty used when regressing PCA coordinates on rotation features.",
    )
    parser.add_argument(
        "--task-descriptor",
        choices=["rotation", "global_field_coeffs", "global_field_fourier"],
        default="rotation",
        help="Task descriptor used before the polynomial regression stage.",
    )
    parser.add_argument(
        "--solo-inf-bias-reg",
        type=float,
        default=1e-10,
        help="Dimensionless ridge scale used to reconstruct infinite-width bias RKHS corrections.",
    )
    parser.add_argument(
        "--solo-inf-full-reg",
        type=float,
        default=1e-10,
        help="Dimensionless ridge scale used to reconstruct infinite-width full RKHS corrections.",
    )
    parser.add_argument(
        "--ridge-mode",
        choices=["trace", "absolute"],
        default="trace",
        help="How to convert the infinite-width ridge scale into a kernel regularizer.",
    )
    parser.add_argument(
        "--path-steps",
        type=int,
        default=257,
        help="Number of steps used for the geodesic integration path.",
    )
    parser.add_argument(
        "--source-index",
        type=int,
        default=None,
        help="Optional pooled training-task index used as the transport source.",
    )
    parser.add_argument(
        "--target-index",
        type=int,
        default=None,
        help="Optional pooled training-task index used as the transport target.",
    )
    parser.add_argument(
        "--holdout-target-index",
        type=int,
        default=None,
        help="Optional pooled holdout-task index used in the train-to-holdout transport summary figure.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Directory for model files and figures. Defaults to {DEFAULT_OUTPUT_ROOT}.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    fit_bias_transport_model_from_results(
        train_results_dirs=parse_csv_list(args.train_results_dirs),
        holdout_results_dirs=parse_csv_list(args.holdout_results_dirs),
        update_kind=args.update_kind,
        n_components=args.n_components,
        rotation_degree=args.rotation_degree,
        coord_ridge=args.coord_ridge,
        descriptor_kind=args.task_descriptor,
        solo_inf_bias_reg=args.solo_inf_bias_reg,
        solo_inf_full_reg=args.solo_inf_full_reg,
        ridge_mode=args.ridge_mode,
        path_steps=args.path_steps,
        source_index=args.source_index,
        target_index=args.target_index,
        holdout_target_index=args.holdout_target_index,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
