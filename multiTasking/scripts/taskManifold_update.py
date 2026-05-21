import argparse
import ast
import csv
from pathlib import Path
import re
import sys
import jax.numpy as jnp
import numpy as np
import jax
from jax import grad, jit, random, jacrev, vmap
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
except Exception:
    plt = None

    class Normalize:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class TwoSlopeNorm(Normalize):
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
MULTITASKING_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = MULTITASKING_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multiTasking.utils.utils import (
    KernelParams,
    make_kappa_kernel_fn,
    sample_random_normals_s2,
    normals_fibonacci_s2,
)
try:
    import neural_tangents as nt
except Exception:
    nt = None


DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results"
DEFAULT_FIGURES_ROOT = MULTITASKING_DIR / "figures"


def orthonormal_circle_frame(normal):
    # Normalize just in case
    n = normal / jnp.linalg.norm(normal)
    v1 = jnp.array([1., 0., 0.]) if abs(n[0]) < 0.9 else jnp.array([0., 1., 0.])
    v1 = v1 - n * jnp.dot(n, v1)
    v1 /= jnp.linalg.norm(v1)
    v2 = jnp.cross(n, v1)
    return n, v1, v2


def circle_points_from_phi(normal, phi, frame=None):
    if frame is None:
        _, v1, v2 = orthonormal_circle_frame(normal)
    else:
        v1, v2 = frame
        v1 = jnp.asarray(v1, dtype=float)
        v2 = jnp.asarray(v2, dtype=float)

    # Geometry on the sphere
    X = jnp.outer(jnp.cos(phi), v1) + jnp.outer(jnp.sin(phi), v2)
    return X


def evaluate_global_sphere_field(X, field_name="xz"):
    X = jnp.asarray(X)
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]

    if field_name == "xz":
        return 2.0 * x * z
    if field_name == "x2_minus_y2":
        return x**2 - y**2
    if field_name == "z2_legendre":
        return 0.5 * (3.0 * z**2 - 1.0)
    if field_name == "yz":
        return 2.0 * y * z
    raise ValueError(
        "global_field_name must be one of "
        "{'xz', 'x2_minus_y2', 'z2_legendre', 'yz'}"
    )


def evaluate_task_targets(
    X,
    phi,
    m,
    *,
    target_phase=0.0,
    target_mode="local_harmonic",
    global_field_name="xz",
):
    if target_mode == "local_harmonic":
        return jnp.sin(m * phi + target_phase)
    if target_mode == "global_field":
        return evaluate_global_sphere_field(X, field_name=global_field_name)
    raise ValueError("target_mode must be 'local_harmonic' or 'global_field'")


def generate_circle_from_phi(
    normal,
    phi,
    m,
    target_phase=0.0,
    frame=None,
    *,
    target_mode="local_harmonic",
    global_field_name="xz",
):
    X = circle_points_from_phi(normal, phi, frame=frame)

    y = evaluate_task_targets(
        X,
        phi,
        m,
        target_phase=target_phase,
        target_mode=target_mode,
        global_field_name=global_field_name,
    )

    return X, y


def generate_single_circle(
    normal,
    pts,
    m,
    phase=0.0,
    target_phase=0.0,
    frame=None,
    *,
    target_mode="local_harmonic",
    global_field_name="xz",
):
    phi = jnp.linspace(0, 2*jnp.pi, pts, endpoint=False) + phase
    phi = phi % (2*jnp.pi)
    return generate_circle_from_phi(
        normal,
        phi,
        m,
        target_phase=target_phase,
        frame=frame,
        target_mode=target_mode,
        global_field_name=global_field_name,
    )


def generate_circle_samples(
    normal,
    pts,
    m,
    *,
    phase=0.0,
    target_phase=0.0,
    frame=None,
    target_mode="local_harmonic",
    global_field_name="xz",
    rng=None,
    mode="grid",
):
    if mode == "grid":
        phi = np.linspace(0.0, 2 * np.pi, pts, endpoint=False) + phase
    elif mode == "random":
        if rng is None:
            raise ValueError("rng must be provided when mode='random'.")
        phi = rng.uniform(0.0, 2 * np.pi, size=pts) + phase
    else:
        raise ValueError("mode must be 'grid' or 'random'")
    phi = np.mod(phi, 2 * np.pi)
    X, y = generate_circle_from_phi(
        normal,
        jnp.asarray(phi),
        m,
        target_phase=target_phase,
        frame=frame,
        target_mode=target_mode,
        global_field_name=global_field_name,
    )
    return np.asarray(X), np.asarray(y), np.asarray(phi, dtype=float)

def generate_normals(K):

    # Fibonacci sphere sampling to get normals
    idx = jnp.arange(0, K, dtype=float) + 0.5
    phi = jnp.arccos(1 - 2 * idx / K)
    theta = jnp.pi * (1 + 5**0.5) * idx
    
    x = jnp.sin(phi) * jnp.cos(theta)
    y = jnp.sin(phi) * jnp.sin(theta)
    z = jnp.cos(phi)
    
    return jnp.vstack([x, y, z]).T

def generate_multi_great_circle_tasks(
    K_tasks=2,
    pts_per_circle=512,
    m=4,
    phase=0.0,
    random_phase=False,
    target_phase=0.0,
    target_mode="local_harmonic",
    global_field_name="xz",
):
    normals = generate_normals(K_tasks)

    X_list, y_list, slices = [], [], []
    start = 0

    for j, n in enumerate(normals):
        if random_phase:
            phase_j = jnp.random.rand() * 2*jnp.pi
        else:
            phase_j = phase

        X, _ = generate_single_circle(
            n,
            pts_per_circle,
            m,
            phase=phase_j,
            target_phase=target_phase,
            target_mode=target_mode,
            global_field_name=global_field_name,
        )
        y = np.zeros((X.shape[0],K_tasks))
        y[:,j]=1

        X_list.append(X)
        y_list.append(y)
        slices.append(slice(start, start + pts_per_circle))
        start += pts_per_circle

    return (
        jnp.vstack(X_list),
        jnp.concatenate(y_list),
        slices,
        normals,
    )


def sample_task_family_data(
    *,
    K_tasks,
    pts_per_circle,
    test_pts_per_circle=None,
    m,
    seed=0,
    task_family="sampled",
    normals_mode="fibonacci",
    random_phase=True,
    target_phase=0.0,
    target_mode="local_harmonic",
    global_field_name="xz",
    base_normal=(0.0, 0.0, 1.0),
    rotation_axis=(1.0, 0.0, 0.0),
):
    rng = np.random.default_rng(seed)
    if test_pts_per_circle is None:
        test_pts_per_circle = 2 * pts_per_circle
    rngA = np.random.default_rng(seed + 10_000_001)
    rngB = np.random.default_rng(seed + 10_000_002)

    if task_family == "single_axis":
        base_n = jnp.asarray(base_normal, dtype=float)
        rot_ax = jnp.asarray(rotation_axis, dtype=float)
        _, base_v1, base_v2 = orthonormal_circle_frame(base_n)
        angles = np.linspace(0.0, 2 * np.pi, K_tasks, endpoint=False)
        normals = []
        frame_v1_rows = []
        frame_v2_rows = []
        rotation_rows = []
        Xs = []
        ys = []
        testA_Xs = []
        testA_ys = []
        testB_Xs = []
        testB_ys = []
        phases = np.zeros(K_tasks)

        for angle in angles:
            n = np.array(rotate_normal_around_axis(base_n, rot_ax, angle))
            v1 = np.array(rotate_vector_around_axis(base_v1, rot_ax, angle))
            v2 = np.array(rotate_vector_around_axis(base_v2, rot_ax, angle))
            frame = (v1, v2)
            X, y, _ = generate_circle_samples(
                n,
                pts_per_circle,
                m,
                phase=0.0,
                target_phase=target_phase,
                frame=frame,
                target_mode=target_mode,
                global_field_name=global_field_name,
                mode="grid",
            )
            X_testA, y_testA, _ = generate_circle_samples(
                n,
                test_pts_per_circle,
                m,
                phase=0.0,
                target_phase=target_phase,
                frame=frame,
                target_mode=target_mode,
                global_field_name=global_field_name,
                rng=rngA,
                mode="random",
            )
            X_testB, y_testB, _ = generate_circle_samples(
                n,
                test_pts_per_circle,
                m,
                phase=0.0,
                target_phase=target_phase,
                frame=frame,
                target_mode=target_mode,
                global_field_name=global_field_name,
                rng=rngB,
                mode="random",
            )
            normals.append(n)
            frame_v1_rows.append(np.asarray(v1, dtype=float))
            frame_v2_rows.append(np.asarray(v2, dtype=float))
            rotation_rows.append(rotation_matrix_from_frame(v1, v2, n))
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
            "frame_v1": np.asarray(frame_v1_rows, dtype=float),
            "frame_v2": np.asarray(frame_v2_rows, dtype=float),
            "task_rotations": np.asarray(rotation_rows, dtype=float),
            "phases": phases,
            "color_values": angles,
            "color_label": "Task angle (rad)",
            "color_is_cyclic": True,
            "align_manifold_to_colors": True,
            "connect_manifold": True,
            "manifold_cmap": "twilight_shifted",
            "distance_label": "Task angular distance (rad)",
            "distance_mode": "angles",
        }

    if task_family == "sampled":
        if normals_mode == "random":
            normals = sample_random_normals_s2(K_tasks, rng=rng)
        elif normals_mode == "fibonacci":
            normals = normals_fibonacci_s2(K_tasks)
        else:
            raise ValueError("normals_mode must be 'random' or 'fibonacci'")

        phases = (
            rng.uniform(0.0, 2 * np.pi, size=K_tasks)
            if random_phase and target_mode == "local_harmonic"
            else np.zeros(K_tasks)
        )
        Xs = []
        ys = []
        testA_Xs = []
        testA_ys = []
        testB_Xs = []
        testB_ys = []
        frame_v1_rows = []
        frame_v2_rows = []
        rotation_rows = []
        for normal, phase in zip(normals, phases):
            _, v1, v2 = orthonormal_circle_frame(jnp.asarray(normal, dtype=float))
            X, y, _ = generate_circle_samples(
                normal,
                pts_per_circle,
                m,
                phase=float(phase),
                target_phase=target_phase,
                target_mode=target_mode,
                global_field_name=global_field_name,
                mode="grid",
            )
            X_testA, y_testA, _ = generate_circle_samples(
                normal,
                test_pts_per_circle,
                m,
                phase=float(phase),
                target_phase=target_phase,
                target_mode=target_mode,
                global_field_name=global_field_name,
                rng=rngA,
                mode="random",
            )
            X_testB, y_testB, _ = generate_circle_samples(
                normal,
                test_pts_per_circle,
                m,
                phase=float(phase),
                target_phase=target_phase,
                target_mode=target_mode,
                global_field_name=global_field_name,
                rng=rngB,
                mode="random",
            )
            frame_v1_rows.append(np.asarray(v1, dtype=float))
            frame_v2_rows.append(np.asarray(v2, dtype=float))
            rotation_rows.append(
                rotation_matrix_from_frame(
                    np.asarray(v1, dtype=float),
                    np.asarray(v2, dtype=float),
                    np.asarray(normal, dtype=float),
                )
            )
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
            "frame_v1": np.asarray(frame_v1_rows, dtype=float),
            "frame_v2": np.asarray(frame_v2_rows, dtype=float),
            "task_rotations": np.asarray(rotation_rows, dtype=float),
            "phases": np.asarray(phases, dtype=float),
            "color_values": np.arange(K_tasks, dtype=float),
            "color_label": "Task index",
            "color_is_cyclic": False,
            "align_manifold_to_colors": False,
            "connect_manifold": False,
            "manifold_cmap": "viridis",
            "distance_label": "Task support distance (rad)",
            "distance_mode": "normals",
        }

    raise ValueError("task_family must be 'sampled' or 'single_axis'")


# Function to create parameters, we apply the NTK scaling here
def init_mlp_params(
    key,
    in_dim,
    hidden_dim,
    bias_mean=0.0,
    bias_std=0.0,
    weight_mean=0.0,
    weight_std=1.0,
):
    k1, k2, kb1, kb2 = random.split(key, 4)

    W1 = (random.normal(k1, (in_dim, hidden_dim)) * weight_std + weight_mean) / jnp.sqrt(in_dim)
    W2 = (random.normal(k2, (hidden_dim,)) * weight_std + weight_mean) / jnp.sqrt(hidden_dim)

    b1 = random.normal(kb1, (hidden_dim,)) * bias_std + bias_mean
    b2 = random.normal(kb2, ()) * bias_std + bias_mean

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

# Actual forward pass
def mlp_apply(params, x):
    h = jnp.maximum(0, x @ params["W1"] + params["b1"])
    return h @ params["W2"] + params["b2"]

# One pass through the net, this is needed for JAX computation reasons in automatic differentiation
def f_single(params, x):
    return mlp_apply(params, x)


# Uses Rodrigues formula
def rotate_normal_around_axis(base_normal, axis, angle):
    return rotate_vector_around_axis(base_normal, axis, angle, normalize_output=True)


def rotate_vector_around_axis(vec, axis, angle, normalize_output=False):
    axis = axis / jnp.linalg.norm(axis)
    v = jnp.asarray(vec, dtype=float)
    v_rot = (
        v * jnp.cos(angle)
        + jnp.cross(axis, v) * jnp.sin(angle)
        + axis * jnp.dot(axis, v) * (1 - jnp.cos(angle))
    )
    if normalize_output:
        return v_rot / jnp.linalg.norm(v_rot)
    return v_rot


def rotation_matrix_from_frame(v1, v2, normal):
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    normal = np.asarray(normal, dtype=float)
    return np.column_stack([v1, v2, normal])

def cosine_matrix(V, eps=1e-12):
    V = np.asarray(np.stack(V), dtype=np.float64)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    safe_norms = np.where(norms > eps, norms, 1.0)
    Vn = np.where(norms > eps, V / safe_norms, 0.0)
    G = np.clip(Vn @ Vn.T, -1.0, 1.0)
    np.fill_diagonal(G, 1.0)
    return G

def great_circle_distance(n1, n2):
    c = np.abs(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2)))
    c = np.clip(c, -1.0, 1.0)
    return np.arccos(c)

def empirical_rkhs_task_similarity(Jbs, alphas, eps=1e-12):
    # For empirical tangent kernels, alpha^T J J^T alpha = ||J^T alpha||^2,
    # so the RKHS cosine can be computed directly in update space.
    deltas = [
        np.asarray(Ji, dtype=np.float64).T @ np.asarray(ai, dtype=np.float64)
        for Ji, ai in zip(Jbs, alphas)
    ]
    return cosine_matrix(deltas, eps=eps)

def rkhs_task_similarity_from_kernel(Xs, alphas, kernel_fn, eps=1e-12):
    """
    Computes G[i,j] = <delta f_i, delta f_j>_H /
                     (||delta f_i||_H ||delta f_j||_H)

    where delta f_i(.) = K(., X_i) alpha_i.
    """
    T = len(Xs)
    G = np.zeros((T, T))

    # cache diagonal norms
    norms = []
    for i in range(T):
        Kii = np.asarray(kernel_fn(Xs[i], Xs[i]), dtype=np.float64)
        Kii = 0.5 * (Kii + Kii.T)
        ai = np.asarray(alphas[i], dtype=np.float64)
        ni = float(ai @ Kii @ ai)
        norms.append(max(ni, 0.0))

    for i in range(T):
        ai = np.asarray(alphas[i], dtype=np.float64)
        for j in range(T):
            aj = np.asarray(alphas[j], dtype=np.float64)
            Kij = np.asarray(kernel_fn(Xs[i], Xs[j]), dtype=np.float64)
            num = float(ai @ Kij @ aj)
            denom = float(np.sqrt(norms[i] * norms[j]))
            if denom <= eps:
                G[i, j] = 1.0 if i == j else 0.0
            else:
                G[i, j] = np.clip(num / denom, -1.0, 1.0)

    return G

def flatten_pytree_jac(J_tree):
    leaves, _ = jax.tree_util.tree_flatten(J_tree)
    return jnp.concatenate([leaf.reshape(leaf.shape[0], -1) for leaf in leaves], axis=1)


def compute_ridge_lambda_jax(K, reg_scale, ridge_mode="trace", eps=1e-12):
    n = K.shape[0]
    if ridge_mode == "trace":
        s_trace = jnp.trace(K) / max(n, 1)
        return reg_scale * n * s_trace + eps
    if ridge_mode == "max_eig":
        return reg_scale * jnp.max(jnp.linalg.eigvalsh(K)) + eps
    raise ValueError(f"Unsupported ridge_mode={ridge_mode}")


def compute_ridge_lambda_np(K, reg_scale, ridge_mode="trace", eps=1e-12):
    K = np.asarray(K, dtype=float)
    n = K.shape[0]
    if ridge_mode == "trace":
        s_trace = np.trace(K) / max(n, 1)
        return float(reg_scale * n * s_trace + eps)
    if ridge_mode == "max_eig":
        return float(reg_scale * np.max(np.linalg.eigvalsh(K)) + eps)
    raise ValueError(f"Unsupported ridge_mode={ridge_mode}")


def solve_tangent_update(J, y, f0, reg, ridge_mode="trace"):
    K = J @ J.T
    lam = compute_ridge_lambda_jax(K, reg, ridge_mode=ridge_mode)
    alpha = jnp.linalg.solve(K + lam * jnp.eye(K.shape[0]), y - f0)
    delta = J.T @ alpha
    return alpha, delta, K

def solve_tangent_update_kernel(K, y, f0, reg, ridge_mode="trace"):
    lam = compute_ridge_lambda_jax(K, reg, ridge_mode=ridge_mode)
    alpha = jnp.linalg.solve(K + lam * jnp.eye(K.shape[0]), y - f0)
    return alpha


def centered_mse(pred, y):
    pred = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=float)
    pred_centered = pred - np.mean(pred)
    y_centered = y - np.mean(y)
    return float(np.mean((pred_centered - y_centered) ** 2))


def solve_joint_kernel_prediction(Kii, Kjj, Kij, yi, yj, f0i, f0j, reg, ridge_mode="trace", eps=1e-12):
    Kii = np.asarray(Kii, dtype=float)
    Kjj = np.asarray(Kjj, dtype=float)
    Kij = np.asarray(Kij, dtype=float)
    yi = np.asarray(yi, dtype=float)
    yj = np.asarray(yj, dtype=float)
    f0i = np.asarray(f0i, dtype=float)
    f0j = np.asarray(f0j, dtype=float)

    K_joint = np.block([[Kii, Kij], [Kij.T, Kjj]])
    K_joint = 0.5 * (K_joint + K_joint.T)
    rhs = np.concatenate([yi - f0i, yj - f0j])
    lam = compute_ridge_lambda_np(K_joint, reg, ridge_mode=ridge_mode, eps=eps)
    alpha = np.linalg.solve(K_joint + lam * np.eye(K_joint.shape[0]), rhs)

    ni = len(yi)
    alpha_i = alpha[:ni]
    alpha_j = alpha[ni:]
    pred_i = f0i + Kii @ alpha_i + Kij @ alpha_j
    pred_j = f0j + Kij.T @ alpha_i + Kjj @ alpha_j
    return pred_i, pred_j, alpha


def solve_joint_kernel_alpha(Kii, Kjj, Kij, yi, yj, f0i, f0j, reg, ridge_mode="trace", eps=1e-12):
    Kii = np.asarray(Kii, dtype=float)
    Kjj = np.asarray(Kjj, dtype=float)
    Kij = np.asarray(Kij, dtype=float)
    yi = np.asarray(yi, dtype=float)
    yj = np.asarray(yj, dtype=float)
    f0i = np.asarray(f0i, dtype=float)
    f0j = np.asarray(f0j, dtype=float)

    K_joint = np.block([[Kii, Kij], [Kij.T, Kjj]])
    K_joint = 0.5 * (K_joint + K_joint.T)
    rhs = np.concatenate([yi - f0i, yj - f0j])
    lam = compute_ridge_lambda_np(K_joint, reg, ridge_mode=ridge_mode, eps=eps)
    alpha = np.linalg.solve(K_joint + lam * np.eye(K_joint.shape[0]), rhs)

    ni = len(yi)
    return alpha[:ni], alpha[ni:], alpha


def predict_joint_from_alpha(K_eval_ii, K_eval_ij, f0_eval_i, alpha_i, alpha_j):
    return np.asarray(f0_eval_i, dtype=float) + np.asarray(K_eval_ii, dtype=float) @ np.asarray(alpha_i, dtype=float) + np.asarray(K_eval_ij, dtype=float) @ np.asarray(alpha_j, dtype=float)


def pairwise_joint_interference_analysis(
    *,
    pairs,
    dists,
    train_ys,
    train_f0s,
    train_within_kernels,
    eval_ys,
    eval_f0s,
    eval_self_kernels,
    solo_eval_preds,
    similarity_matrix,
    train_cross_kernel_fn,
    eval_cross_kernel_fn,
    reg,
    ridge_mode="trace",
):
    solo_errors = np.asarray([centered_mse(pred, y) for pred, y in zip(solo_eval_preds, eval_ys)], dtype=float)
    pair_similarity = upper_tri_values(similarity_matrix, pairs)

    joint_error_i = []
    joint_error_j = []
    drop_i = []
    drop_j = []
    pair_drop_mean = []
    pair_drop_max = []

    for i, j in pairs:
        alpha_i, alpha_j, _ = solve_joint_kernel_alpha(
            train_within_kernels[i],
            train_within_kernels[j],
            train_cross_kernel_fn(i, j),
            train_ys[i],
            train_ys[j],
            train_f0s[i],
            train_f0s[j],
            reg,
            ridge_mode=ridge_mode,
        )
        pred_i = predict_joint_from_alpha(
            eval_self_kernels[i],
            eval_cross_kernel_fn(i, j),
            eval_f0s[i],
            alpha_i,
            alpha_j,
        )
        pred_j = predict_joint_from_alpha(
            eval_self_kernels[j],
            eval_cross_kernel_fn(j, i),
            eval_f0s[j],
            alpha_j,
            alpha_i,
        )

        err_i = centered_mse(pred_i, eval_ys[i])
        err_j = centered_mse(pred_j, eval_ys[j])
        di = err_i - solo_errors[i]
        dj = err_j - solo_errors[j]

        joint_error_i.append(err_i)
        joint_error_j.append(err_j)
        drop_i.append(di)
        drop_j.append(dj)
        pair_drop_mean.append(0.5 * (di + dj))
        pair_drop_max.append(max(di, dj))

    return {
        "pairs": pairs,
        "dists": np.asarray(dists, dtype=float),
        "pair_similarity": np.asarray(pair_similarity, dtype=float),
        "solo_errors": solo_errors,
        "joint_error_i": np.asarray(joint_error_i, dtype=float),
        "joint_error_j": np.asarray(joint_error_j, dtype=float),
        "drop_i": np.asarray(drop_i, dtype=float),
        "drop_j": np.asarray(drop_j, dtype=float),
        "pair_drop_mean": np.asarray(pair_drop_mean, dtype=float),
        "pair_drop_max": np.asarray(pair_drop_max, dtype=float),
    }


def pairwise_kernel_difference_decomposition(
    *,
    pairs,
    dists,
    within_kernels_bias,
    within_kernels_full,
    alphas_bias,
    alphas_full,
    cross_kernel_bias_fn,
    cross_kernel_full_fn,
):
    rows = []
    for pair_index, (i, j) in enumerate(pairs):
        Kii_bias = np.asarray(within_kernels_bias[i], dtype=float)
        Kjj_bias = np.asarray(within_kernels_bias[j], dtype=float)
        Kii_full = np.asarray(within_kernels_full[i], dtype=float)
        Kjj_full = np.asarray(within_kernels_full[j], dtype=float)
        Kij_bias = np.asarray(cross_kernel_bias_fn(i, j), dtype=float)
        Kij_full = np.asarray(cross_kernel_full_fn(i, j), dtype=float)
        delta_Kij = Kij_full - Kij_bias

        alpha_i_bias = np.asarray(alphas_bias[i], dtype=float)
        alpha_j_bias = np.asarray(alphas_bias[j], dtype=float)
        alpha_i_full = np.asarray(alphas_full[i], dtype=float)
        alpha_j_full = np.asarray(alphas_full[j], dtype=float)

        self_bias_i = float(alpha_i_bias @ Kii_bias @ alpha_i_bias)
        self_bias_j = float(alpha_j_bias @ Kjj_bias @ alpha_j_bias)
        self_full_i = float(alpha_i_full @ Kii_full @ alpha_i_full)
        self_full_j = float(alpha_j_full @ Kjj_full @ alpha_j_full)
        bias_scale = float(np.sqrt(max(self_bias_i * self_bias_j, 0.0)) + 1e-12)
        full_scale = float(np.sqrt(max(self_full_i * self_full_j, 0.0)) + 1e-12)

        num_bb = float(alpha_i_bias @ Kij_bias @ alpha_j_bias)
        num_bf = float(alpha_i_bias @ Kij_full @ alpha_j_bias)
        num_ff = float(alpha_i_full @ Kij_full @ alpha_j_full)
        num_fb = float(alpha_i_full @ Kij_bias @ alpha_j_full)
        total_numerator_shift = float(num_ff - num_bb)

        rows.append(
            {
                "pair_index": int(pair_index),
                "task_i": int(i),
                "task_j": int(j),
                "distance": float(dists[pair_index]),
                "kij_bias_fro": float(np.linalg.norm(Kij_bias, ord="fro") / np.sqrt(Kij_bias.size)),
                "kij_full_fro": float(np.linalg.norm(Kij_full, ord="fro") / np.sqrt(Kij_full.size)),
                "delta_kij_fro": float(np.linalg.norm(delta_Kij, ord="fro") / np.sqrt(delta_Kij.size)),
                "delta_kij_mean": float(np.mean(delta_Kij)),
                "delta_kij_mean_abs": float(np.mean(np.abs(delta_Kij))),
                "delta_kij_max_abs": float(np.max(np.abs(delta_Kij))),
                "trace_norm_bias": float(np.trace(Kii_bias) / Kii_bias.shape[0]),
                "trace_norm_full": float(np.trace(Kii_full) / Kii_full.shape[0]),
                "self_trace_gap_i": float(np.trace(Kii_full - Kii_bias) / Kii_bias.shape[0]),
                "self_trace_gap_j": float(np.trace(Kjj_full - Kjj_bias) / Kjj_bias.shape[0]),
                "self_norm_bias_i": self_bias_i,
                "self_norm_bias_j": self_bias_j,
                "self_norm_full_i": self_full_i,
                "self_norm_full_j": self_full_j,
                "num_bb": num_bb,
                "num_bf": num_bf,
                "num_fb": num_fb,
                "num_ff": num_ff,
                "direct_block_shift": float(num_bf - num_bb),
                "alpha_shift": float(num_ff - num_bf),
                "full_alpha_on_bias_block_shift": float(num_fb - num_bb),
                "total_numerator_shift": total_numerator_shift,
                "normalized_total_numerator_shift_bias_scale": float(total_numerator_shift / bias_scale),
                "exact_similarity_shift": float((num_ff / full_scale) - (num_bb / bias_scale)),
            }
        )

    return rows


def _plot_scatter_with_binned_mean(ax, x, y, *, title, xlabel, ylabel, color, n_bins=14):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ax.scatter(x, y, s=18, alpha=0.35, color=color, edgecolors="none")
    if x.size:
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        if x_max > x_min:
            edges = np.linspace(x_min, x_max, n_bins + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            means = []
            valid_centers = []
            for lo, hi, center in zip(edges[:-1], edges[1:], centers):
                if hi == edges[-1]:
                    mask = (x >= lo) & (x <= hi)
                else:
                    mask = (x >= lo) & (x < hi)
                if np.any(mask):
                    valid_centers.append(center)
                    means.append(float(np.mean(y[mask])))
            if valid_centers:
                ax.plot(valid_centers, means, color="black", lw=2.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22)


def plot_kernel_difference_decomposition_summary(
    *,
    empirical_rows,
    infinite_rows,
    distance_label,
    title_prefix,
    save_path=None,
):
    configs = [
        ("delta_kij_fro", r"$||K_{ij}^{full} - K_{ij}^{bias}||_F / \sqrt{n_i n_j}$", "Added cross-task block magnitude"),
        ("direct_block_shift", r"$\alpha_b^\top (K_{ij}^{full} - K_{ij}^{bias}) \alpha_b$", "Direct cross-block effect"),
        ("alpha_shift", r"$\alpha_f^\top K_{ij}^{full} \alpha_f - \alpha_b^\top K_{ij}^{full} \alpha_b$", "Self-conditioning / alpha effect"),
        ("total_numerator_shift", r"$\alpha_f^\top K_{ij}^{full} \alpha_f - \alpha_b^\top K_{ij}^{bias} \alpha_b$", "Total numerator shift"),
        ("normalized_total_numerator_shift_bias_scale", r"$\Delta$ numerator / $\sqrt{s_i^{bias} s_j^{bias}}$", "Bias-scale normalized total shift"),
        ("exact_similarity_shift", r"$\cos_{full}(i,j) - \cos_{bias}(i,j)$", "Exact cosine-similarity shift"),
    ]
    row_specs = [
        ("Empirical tangent", empirical_rows, "#4c72b0"),
        ("Infinite width", infinite_rows, "#55a868"),
    ]
    available_keys = set(configs_i[0] for configs_i in configs)
    for _, rows, _ in row_specs:
        if not rows:
            continue
        available_keys &= set(rows[0].keys())
    filtered_configs = [cfg for cfg in configs if cfg[0] in available_keys]
    if not filtered_configs:
        return None

    fig, axes = plt.subplots(2, len(filtered_configs), figsize=(4.8 * len(filtered_configs), 9.2))
    axes = np.asarray(axes, dtype=object)
    if axes.ndim == 1:
        axes = axes[:, None]

    for row_idx, (row_label, rows, color) in enumerate(row_specs):
        for col_idx, (key, ylabel, subtitle) in enumerate(filtered_configs):
            x = [row["distance"] for row in rows]
            y = [row[key] for row in rows]
            _plot_scatter_with_binned_mean(
                axes[row_idx, col_idx],
                x,
                y,
                title=f"{row_label}: {subtitle}",
                xlabel=distance_label,
                ylabel=ylabel,
                color=color,
            )

    fig.suptitle(title_prefix, fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


def save_pairwise_results_npz(save_path, results_by_name, extras=None):
    payload = {}
    for name, result in results_by_name.items():
        for key, value in result.items():
            payload[f"{name}__{key}"] = np.asarray(value)
    if extras is not None:
        for key, value in extras.items():
            payload[key] = np.asarray(value)
    np.savez(save_path, **payload)


def save_task_update_vectors_npz(
    save_path,
    *,
    normals,
    phases,
    color_values,
    frame_v1=None,
    frame_v2=None,
    task_rotations=None,
    delta_bias,
    delta_full,
    target_phase,
    target_mode,
    global_field_name,
    m,
    task_family,
    base_normal=(0.0, 0.0, 1.0),
    rotation_axis=(1.0, 0.0, 0.0),
):
    normals = np.asarray(normals, dtype=float)
    if frame_v1 is None or frame_v2 is None or task_rotations is None:
        frame_v1_rows = []
        frame_v2_rows = []
        rotation_rows = []
        for normal in normals:
            _, v1, v2 = orthonormal_circle_frame(jnp.asarray(normal, dtype=float))
            v1_np = np.asarray(v1, dtype=float)
            v2_np = np.asarray(v2, dtype=float)
            frame_v1_rows.append(v1_np)
            frame_v2_rows.append(v2_np)
            rotation_rows.append(rotation_matrix_from_frame(v1_np, v2_np, normal))
        frame_v1 = np.asarray(frame_v1_rows, dtype=float)
        frame_v2 = np.asarray(frame_v2_rows, dtype=float)
        task_rotations = np.asarray(rotation_rows, dtype=float)

    np.savez(
        save_path,
        task_normals=normals,
        task_phases=np.asarray(phases, dtype=float),
        task_color_values=np.asarray(color_values, dtype=float),
        task_frame_v1=np.asarray(frame_v1, dtype=float),
        task_frame_v2=np.asarray(frame_v2, dtype=float),
        task_rotations=np.asarray(task_rotations, dtype=float),
        delta_bias=np.asarray(delta_bias, dtype=float),
        delta_full=np.asarray(delta_full, dtype=float),
        target_phase=np.asarray([target_phase], dtype=float),
        target_mode=np.asarray([str(target_mode)]),
        global_field_name=np.asarray([str(global_field_name)]),
        m=np.asarray([int(m)], dtype=int),
        task_family=np.asarray([str(task_family)]),
        task_representation_version=np.asarray([2], dtype=int),
        base_normal=np.asarray(base_normal, dtype=float),
        rotation_axis=np.asarray(rotation_axis, dtype=float),
    )


def format_phase_tag(target_phase):
    return f"{float(target_phase):.3f}".replace("-", "m").replace(".", "p")


def sanitize_name_token(token):
    return re.sub(r"[^A-Za-z0-9]+", "", str(token))


def parse_vector_arg(value):
    if isinstance(value, np.ndarray):
        vec = np.asarray(value, dtype=float).reshape(-1)
        if vec.size != 3:
            raise argparse.ArgumentTypeError("Expected a 3-vector.")
        return tuple(float(x) for x in vec)
    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise argparse.ArgumentTypeError("Expected a 3-vector.")
        return tuple(float(x) for x in value)

    items = [item.strip() for item in str(value).replace(";", ",").split(",") if item.strip()]
    if len(items) != 3:
        raise argparse.ArgumentTypeError(
            "Vector arguments must be comma-separated triples like '1,0,0'."
        )
    try:
        return tuple(float(item) for item in items)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Could not parse vector '{value}'. Use a triple like '0,1,0'."
        ) from exc


def _format_signed_float_token(value):
    return f"{float(value):+.3f}".replace("+", "pos").replace("-", "neg").replace(".", "p")


def format_vector_tag(vector, prefix):
    vec = np.asarray(vector, dtype=float).reshape(3)
    return (
        f"_{sanitize_name_token(prefix)}"
        f"x{_format_signed_float_token(vec[0])}"
        f"y{_format_signed_float_token(vec[1])}"
        f"z{_format_signed_float_token(vec[2])}"
    )


def parse_angle_arg(value):
    if isinstance(value, (int, float)):
        return float(value)

    expr = str(value).strip().lower()
    if not expr:
        raise argparse.ArgumentTypeError("Angle value cannot be empty.")

    node = ast.parse(expr, mode="eval")

    def _eval(node_):
        if isinstance(node_, ast.Expression):
            return _eval(node_.body)
        if isinstance(node_, ast.Constant) and isinstance(node_.value, (int, float)):
            return float(node_.value)
        if isinstance(node_, ast.Name) and node_.id == "pi":
            return float(np.pi)
        if isinstance(node_, ast.UnaryOp) and isinstance(node_.op, (ast.UAdd, ast.USub)):
            val = _eval(node_.operand)
            return val if isinstance(node_.op, ast.UAdd) else -val
        if isinstance(node_, ast.BinOp) and isinstance(node_.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left = _eval(node_.left)
            right = _eval(node_.right)
            if isinstance(node_.op, ast.Add):
                return left + right
            if isinstance(node_.op, ast.Sub):
                return left - right
            if isinstance(node_.op, ast.Mult):
                return left * right
            return left / right
        raise argparse.ArgumentTypeError(
            f"Unsupported angle expression '{value}'. Use a float like 1.5708 or a simple pi expression like pi/2."
        )

    try:
        return float(_eval(node))
    except ZeroDivisionError as exc:
        raise argparse.ArgumentTypeError("Angle expression divides by zero.") from exc


def make_run_name(
    *,
    task_family,
    seed,
    K_tasks,
    m,
    pts,
    dim,
    target_phase,
    target_mode="local_harmonic",
    global_field_name="xz",
    base_normal=(0.0, 0.0, 1.0),
    rotation_axis=(1.0, 0.0, 0.0),
):
    middle = ""
    if target_mode != "local_harmonic":
        middle = (
            f"_tmode{sanitize_name_token(target_mode)}"
            f"_gfield{sanitize_name_token(global_field_name)}"
        )
    if task_family == "single_axis":
        base_default = np.array([0.0, 0.0, 1.0], dtype=float)
        axis_default = np.array([1.0, 0.0, 0.0], dtype=float)
        if not np.allclose(np.asarray(base_normal, dtype=float), base_default):
            middle += format_vector_tag(base_normal, "bn")
        if not np.allclose(np.asarray(rotation_axis, dtype=float), axis_default):
            middle += format_vector_tag(rotation_axis, "rax")
    return (
        f"{task_family}_seed{seed}_K{K_tasks}_m{m}_pts{pts}_dim{dim}"
        f"{middle}_"
        f"tphase{format_phase_tag(target_phase)}"
    )


def ensure_output_dir(
    output_dir,
    *,
    task_family,
    seed,
    K_tasks,
    m,
    pts,
    dim,
    target_phase,
    target_mode="local_harmonic",
    global_field_name="xz",
    base_normal=(0.0, 0.0, 1.0),
    rotation_axis=(1.0, 0.0, 0.0),
    default_root,
):
    if output_dir is None:
        output_dir = Path(default_root) / make_run_name(
            task_family=task_family,
            seed=seed,
            K_tasks=K_tasks,
            m=m,
            pts=pts,
            dim=dim,
            target_phase=target_phase,
            target_mode=target_mode,
            global_field_name=global_field_name,
            base_normal=base_normal,
            rotation_axis=rotation_axis,
        )
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def ensure_results_dir(
    results_dir,
    *,
    task_family,
    seed,
    K_tasks,
    m,
    pts,
    dim,
    target_phase,
    target_mode="local_harmonic",
    global_field_name="xz",
    base_normal=(0.0, 0.0, 1.0),
    rotation_axis=(1.0, 0.0, 0.0),
):
    return ensure_output_dir(
        results_dir,
        task_family=task_family,
        seed=seed,
        K_tasks=K_tasks,
        m=m,
        pts=pts,
        dim=dim,
        target_phase=target_phase,
        target_mode=target_mode,
        global_field_name=global_field_name,
        base_normal=base_normal,
        rotation_axis=rotation_axis,
        default_root=DEFAULT_RESULTS_ROOT,
    )


def ensure_figures_dir(
    figures_dir,
    *,
    task_family,
    seed,
    K_tasks,
    m,
    pts,
    dim,
    target_phase,
    target_mode="local_harmonic",
    global_field_name="xz",
    base_normal=(0.0, 0.0, 1.0),
    rotation_axis=(1.0, 0.0, 0.0),
):
    return ensure_output_dir(
        figures_dir,
        task_family=task_family,
        seed=seed,
        K_tasks=K_tasks,
        m=m,
        pts=pts,
        dim=dim,
        target_phase=target_phase,
        target_mode=target_mode,
        global_field_name=global_field_name,
        base_normal=base_normal,
        rotation_axis=rotation_axis,
        default_root=DEFAULT_FIGURES_ROOT,
    )


def load_ridge_tuning_scales(ridge_path, *, single_task_T=1, pairwise_task_T=2):
    ridge_path = Path(ridge_path)
    obj = np.load(ridge_path, allow_pickle=True)
    T_list = np.asarray(obj["T_list"], dtype=int)
    files = set(obj.files)

    def _lookup(T, key):
        matches = np.where(T_list == int(T))[0]
        if matches.size == 0:
            raise ValueError(
                f"T={T} not found in ridge tuning file {ridge_path}. "
                f"Available T values: {T_list.tolist()}"
            )
        return float(np.asarray(obj[key], dtype=float)[matches[0]])

    separate_keys = {
        "lam_star_emp_bias",
        "lam_star_emp_full",
        "lam_star_inf_bias",
        "lam_star_inf_full",
    }
    legacy_keys = {
        "lam_star_bias_only",
        "lam_star_full_ntk",
    }

    if separate_keys.issubset(files):
        single_emp_bias = _lookup(single_task_T, "lam_star_emp_bias")
        single_emp_full = _lookup(single_task_T, "lam_star_emp_full")
        single_inf_bias = _lookup(single_task_T, "lam_star_inf_bias")
        single_inf_full = _lookup(single_task_T, "lam_star_inf_full")
        pair_emp_bias = _lookup(pairwise_task_T, "lam_star_emp_bias")
        pair_emp_full = _lookup(pairwise_task_T, "lam_star_emp_full")
        pair_inf_bias = _lookup(pairwise_task_T, "lam_star_inf_bias")
        pair_inf_full = _lookup(pairwise_task_T, "lam_star_inf_full")
        return {
            "single_emp_bias": single_emp_bias,
            "single_emp_full": single_emp_full,
            "single_inf_bias": single_inf_bias,
            "single_inf_full": single_inf_full,
            "pair_emp_bias": pair_emp_bias,
            "pair_emp_full": pair_emp_full,
            "pair_inf_bias": pair_inf_bias,
            "pair_inf_full": pair_inf_full,
            "single_task_T": int(single_task_T),
            "pairwise_task_T": int(pairwise_task_T),
            "path": str(ridge_path),
            "format": "separate_empirical_infinite",
        }

    if legacy_keys.issubset(files):
        single_bias = _lookup(single_task_T, "lam_star_bias_only")
        single_full = _lookup(single_task_T, "lam_star_full_ntk")
        pair_bias = _lookup(pairwise_task_T, "lam_star_bias_only")
        pair_full = _lookup(pairwise_task_T, "lam_star_full_ntk")
        return {
            "single_emp_bias": single_bias,
            "single_emp_full": single_full,
            "single_inf_bias": single_bias,
            "single_inf_full": single_full,
            "pair_emp_bias": pair_bias,
            "pair_emp_full": pair_full,
            "pair_inf_bias": pair_bias,
            "pair_inf_full": pair_full,
            "single_bias": single_bias,
            "single_full": single_full,
            "pair_bias": pair_bias,
            "pair_full": pair_full,
            "single_task_T": int(single_task_T),
            "pairwise_task_T": int(pairwise_task_T),
            "path": str(ridge_path),
            "format": "shared_bias_full",
        }

    raise ValueError(
        f"Unrecognized ridge tuning format in {ridge_path}. "
        "Expected either legacy keys "
        "{'lam_star_bias_only', 'lam_star_full_ntk'} or separate empirical/infinite keys "
        "{'lam_star_emp_bias', 'lam_star_emp_full', 'lam_star_inf_bias', 'lam_star_inf_full'}."
    )


def write_csv_rows(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_matrix_csv(path, matrix):
    matrix = np.asarray(matrix, dtype=float)
    header = ",".join([f"task_{j}" for j in range(matrix.shape[1])])
    np.savetxt(path, matrix, delimiter=",", header=header, comments="")


def wrapped_phase_difference(a, b):
    return np.angle(np.exp(1j * (float(a) - float(b))))


def pairwise_phase_metrics(phases, pairs, m):
    wrapped = []
    abs_wrapped = []
    compat = []
    for i, j in pairs:
        delta = wrapped_phase_difference(phases[i], phases[j])
        wrapped.append(delta)
        abs_wrapped.append(abs(delta))
        compat.append(np.cos(m * delta))
    return {
        "phase_delta_wrapped": np.asarray(wrapped, dtype=float),
        "phase_delta_abs": np.asarray(abs_wrapped, dtype=float),
        "phase_compat": np.asarray(compat, dtype=float),
    }


def best_circular_target_correlation(y_i, y_j, eps=1e-12):
    y_i = np.asarray(y_i, dtype=float)
    y_j = np.asarray(y_j, dtype=float)
    if y_i.shape != y_j.shape:
        raise ValueError("Target vectors must have matching shapes for circular correlation.")

    yi = y_i - np.mean(y_i)
    denom_i = np.linalg.norm(yi)
    if denom_i < eps:
        return 0.0

    best_corr = 0.0
    for candidate in (y_j, y_j[::-1]):
        yc = np.asarray(candidate, dtype=float) - np.mean(candidate)
        denom = denom_i * np.linalg.norm(yc)
        if denom < eps:
            continue
        for shift in range(len(y_i)):
            corr = float(np.dot(yi, np.roll(yc, shift)) / denom)
            if abs(corr) > abs(best_corr):
                best_corr = corr
    return best_corr


def pairwise_teacher_correlation(
    *,
    normals,
    phases,
    m,
    pairs,
    target_phase=0.0,
    target_mode="local_harmonic",
    global_field_name="xz",
    pts_per_circle=256,
):
    normals = np.asarray(normals, dtype=float)
    phases = np.asarray(phases, dtype=float)
    task_targets = []
    for normal, phase in zip(normals, phases):
        _, y, _ = generate_circle_samples(
            normal,
            pts_per_circle,
            int(m),
            phase=float(phase),
            target_phase=float(target_phase),
            target_mode=target_mode,
            global_field_name=global_field_name,
            mode="grid",
        )
        task_targets.append(np.asarray(y, dtype=float))

    return np.asarray(
        [best_circular_target_correlation(task_targets[i], task_targets[j]) for i, j in pairs],
        dtype=float,
    )


def save_task_metadata_csv(
    path,
    *,
    normals,
    phases,
    target_phase,
    target_mode,
    global_field_name,
    m,
    task_family,
    color_values,
    delta_b_norms,
    delta_full_norms,
    solo_mse_emp_bias,
    solo_mse_emp_full,
    solo_mse_inf_bias,
    solo_mse_inf_full,
):
    rows = []
    normals = np.asarray(normals, dtype=float)
    phases = np.asarray(phases, dtype=float)
    color_values = np.asarray(color_values, dtype=float)
    for idx in range(len(phases)):
        rows.append(
            {
                "task_index": idx,
                "normal_x": normals[idx, 0],
                "normal_y": normals[idx, 1],
                "normal_z": normals[idx, 2],
                "phase": phases[idx],
                "target_phase": float(target_phase),
                "target_mode": str(target_mode),
                "global_field_name": str(global_field_name),
                "m": int(m),
                "task_family": str(task_family),
                "color_value": color_values[idx],
                "delta_norm_bias": float(delta_b_norms[idx]),
                "delta_norm_full": float(delta_full_norms[idx]),
                "solo_mse_emp_bias": float(solo_mse_emp_bias[idx]),
                "solo_mse_emp_full": float(solo_mse_emp_full[idx]),
                "solo_mse_inf_bias": float(solo_mse_inf_bias[idx]),
                "solo_mse_inf_full": float(solo_mse_inf_full[idx]),
            }
        )
    write_csv_rows(path, list(rows[0].keys()) if rows else [], rows)


def save_pairwise_metrics_csv(path, rows):
    if not rows:
        write_csv_rows(path, [], [])
        return
    write_csv_rows(path, list(rows[0].keys()), rows)


def save_summary_stats_csv(path, rows):
    if not rows:
        write_csv_rows(path, [], [])
        return
    write_csv_rows(path, list(rows[0].keys()), rows)


def save_csv_preserving_existing(path, save_fn, rows, *, preserve_existing=False, label=None):
    path = Path(path)
    if preserve_existing and path.exists():
        print(f"Preserving existing {label or path.name}: {path}")
        return
    save_fn(path, rows)


def read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_matrix_csv(path):
    return np.loadtxt(path, delimiter=",", skiprows=1)


def load_task_metadata(path):
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    rows = sorted(rows, key=lambda row: int(row["task_index"]))
    return {
        "task_index": np.asarray([int(row["task_index"]) for row in rows], dtype=int),
        "normals": np.asarray([[float(row["normal_x"]), float(row["normal_y"]), float(row["normal_z"])] for row in rows], dtype=float),
        "phases": np.asarray([float(row["phase"]) for row in rows], dtype=float),
        "target_phase": float(rows[0]["target_phase"]),
        "target_mode": rows[0].get("target_mode", "local_harmonic") or "local_harmonic",
        "global_field_name": rows[0].get("global_field_name", "xz") or "xz",
        "m": int(float(rows[0]["m"])) if rows[0].get("m", "") not in {"", None} else None,
        "task_family": rows[0].get("task_family", "") or None,
        "color_values": np.asarray([float(row["color_value"]) for row in rows], dtype=float),
        "delta_norm_bias": np.asarray([float(row["delta_norm_bias"]) for row in rows], dtype=float),
        "delta_norm_full": np.asarray([float(row["delta_norm_full"]) for row in rows], dtype=float),
        "solo_mse_emp_bias": np.asarray([float(row["solo_mse_emp_bias"]) for row in rows], dtype=float),
        "solo_mse_emp_full": np.asarray([float(row["solo_mse_emp_full"]) for row in rows], dtype=float),
        "solo_mse_inf_bias": np.asarray([float(row["solo_mse_inf_bias"]) for row in rows], dtype=float),
        "solo_mse_inf_full": np.asarray([float(row["solo_mse_inf_full"]) for row in rows], dtype=float),
    }


def load_manifold_analysis_csvs(results_dir, stem):
    coord_rows = read_csv_rows(Path(results_dir) / f"{stem}_coords.csv")
    spec_rows = read_csv_rows(Path(results_dir) / f"{stem}_spectrum.csv")

    coord_groups = {}
    for row in coord_rows:
        kind = row["kind"]
        coord_groups.setdefault(kind, []).append(
            (
                int(row["task_index"]),
                np.asarray(
                    [
                        float(row.get("coord_1", 0.0)),
                        float(row.get("coord_2", 0.0)),
                        float(row.get("coord_3", 0.0)),
                    ],
                    dtype=float,
                ),
            )
        )

    spec_groups = {}
    for row in spec_rows:
        kind = row["kind"]
        spec_groups.setdefault(kind, []).append(
            (
                int(row["component"]),
                float(row["eigenvalue"]),
                float(row["variance_ratio"]),
            )
        )

    out = {}
    for kind in ["bias", "full"]:
        coord_entries = sorted(coord_groups.get(kind, []), key=lambda item: item[0])
        spec_entries = sorted(spec_groups.get(kind, []), key=lambda item: item[0])
        out[kind] = {
            "coords": np.vstack([coords for _, coords in coord_entries]) if coord_entries else np.zeros((0, 3)),
            "evals": np.asarray([entry[1] for entry in spec_entries], dtype=float),
            "var_ratio": np.asarray([entry[2] for entry in spec_entries], dtype=float),
        }
    return out


def load_kernel_decomposition_csv(path):
    rows = read_csv_rows(path)
    grouped = {"empirical": [], "infinite_width": []}
    for row in rows:
        regime = row["regime"]
        parsed = {}
        for key, value in row.items():
            if key == "regime":
                continue
            if key in {"pair_index", "task_i", "task_j"}:
                parsed[key] = int(value)
            else:
                parsed[key] = float(value)
        grouped.setdefault(regime, []).append(parsed)
    return grouped


def save_manifold_analysis_csvs(results_dir, stem, coords_bias, evals_bias, vr_bias, coords_full, evals_full, vr_full):
    coord_rows = []
    for label, coords in [("bias", coords_bias), ("full", coords_full)]:
        coords = np.asarray(coords, dtype=float)
        if coords.shape[1] < 3:
            coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])), mode="constant")
        for idx in range(coords.shape[0]):
            coord_rows.append(
                {
                    "kind": label,
                    "task_index": idx,
                    "coord_1": coords[idx, 0],
                    "coord_2": coords[idx, 1],
                    "coord_3": coords[idx, 2],
                }
            )
    save_pairwise_metrics_csv(results_dir / f"{stem}_coords.csv", coord_rows)

    spec_rows = []
    for label, evals, vr in [("bias", evals_bias, vr_bias), ("full", evals_full, vr_full)]:
        evals = np.asarray(evals, dtype=float)
        vr = np.asarray(vr, dtype=float)
        for idx in range(max(len(evals), len(vr))):
            spec_rows.append(
                {
                    "kind": label,
                    "component": idx + 1,
                    "eigenvalue": evals[idx] if idx < len(evals) else np.nan,
                    "variance_ratio": vr[idx] if idx < len(vr) else np.nan,
                }
            )
    save_pairwise_metrics_csv(results_dir / f"{stem}_spectrum.csv", spec_rows)


def append_summary_row(rows, analysis, group, values):
    stats = summary_stats(values)
    rows.append(
        {
            "analysis": analysis,
            "group": group,
            "mean": stats["mean"],
            "std": stats["std"],
            "mean_abs": stats["mean_abs"],
            "rms": stats["rms"],
            "min": stats["min"],
            "max": stats["max"],
            "median": stats["median"],
        }
    )

# -----------------------------------------------------------------------------
# Plotting utilities for task-manifold analyses
# -----------------------------------------------------------------------------
def pairwise_distances_from_angles(angles):
    """Pairwise circular distance for one-axis great-circle rotations."""
    angles = np.asarray(angles)
    T = len(angles)
    dists = []
    pairs = []
    for i in range(T):
        for j in range(i + 1, T):
            diff = np.abs(angles[i] - angles[j])
            d = np.minimum(diff, 2 * np.pi - diff)
            dists.append(d)
            pairs.append((i, j))
    return np.asarray(dists), pairs


def pairwise_distances_from_normals(normals):
    """Pairwise distance between great circles using their unoriented normals."""
    normals = np.asarray(normals)
    T = len(normals)
    dists = []
    pairs = []
    for i in range(T):
        for j in range(i + 1, T):
            dists.append(great_circle_distance(normals[i], normals[j]))
            pairs.append((i, j))
    return np.asarray(dists), pairs


def upper_tri_values(M, pairs):
    M = np.asarray(M)
    return np.asarray([M[i, j] for i, j in pairs])


def offdiag_values(M):
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("Expected a square similarity matrix.")
    tri = np.triu_indices(M.shape[0], k=1)
    return M[tri]


def linear_fit_stats(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    r2 = np.corrcoef(x, y)[0, 1] ** 2 if len(x) > 1 else np.nan
    return slope, intercept, r2


def summary_stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "mean_abs": np.nan,
            "rms": np.nan,
        }
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "mean_abs": float(np.mean(np.abs(values))),
        "rms": float(np.sqrt(np.mean(values ** 2))),
    }


def format_summary_line(label, stats):
    return (
        f"{label}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
        f"|.|={stats['mean_abs']:.3f}, rms={stats['rms']:.3f}, "
        f"range=[{stats['min']:.3f}, {stats['max']:.3f}]"
    )


def format_compact_summary_line(label, stats):
    return (
        f"{label}: mean={stats['mean']:.3f}, "
        f"|.|={stats['mean_abs']:.3f}, rms={stats['rms']:.3f}"
    )


def print_comparison_summary(title, label_a, values_a, label_b, values_b):
    stats_a = summary_stats(values_a)
    stats_b = summary_stats(values_b)
    print(title)
    print(f"  {format_summary_line(label_a, stats_a)}")
    print(f"  {format_summary_line(label_b, stats_b)}")
    return stats_a, stats_b


def print_difference_summary(title, values):
    stats = summary_stats(values)
    print(title)
    print(f"  {format_summary_line('Full - bias', stats)}")
    return stats


def print_distance_fit_summary(title, dists, values):
    slope, intercept, r2 = linear_fit_stats(dists, values)
    print(title)
    print(f"  slope={slope:.4f}, intercept={intercept:.4f}, R^2={r2:.4f}")
    return slope, intercept, r2


def paired_mean_difference_test(bias_values, full_values, *, seed=0, n_perm=20000, n_boot=10000, chunk_size=2000):
    bias_values = np.asarray(bias_values, dtype=float)
    full_values = np.asarray(full_values, dtype=float)
    diff = full_values - bias_values
    n = diff.size

    if n == 0:
        return {
            "n": 0,
            "mean_diff": np.nan,
            "median_diff": np.nan,
            "std_diff": np.nan,
            "mean_abs_diff": np.nan,
            "positive_frac": np.nan,
            "negative_frac": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "perm_pvalue": np.nan,
        }

    rng = np.random.default_rng(seed)
    mean_diff = float(np.mean(diff))
    median_diff = float(np.median(diff))
    std_diff = float(np.std(diff))
    mean_abs_diff = float(np.mean(np.abs(diff)))
    positive_frac = float(np.mean(diff > 0))
    negative_frac = float(np.mean(diff < 0))

    exceed_count = 0
    draws_done = 0
    while draws_done < n_perm:
        batch = min(chunk_size, n_perm - draws_done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(batch, n))
        perm_means = np.mean(signs * diff[None, :], axis=1)
        exceed_count += int(np.sum(np.abs(perm_means) >= abs(mean_diff)))
        draws_done += batch
    perm_pvalue = (exceed_count + 1.0) / (n_perm + 1.0)

    boot_idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = np.mean(diff[boot_idx], axis=1)
    ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975])

    return {
        "n": int(n),
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "std_diff": std_diff,
        "mean_abs_diff": mean_abs_diff,
        "positive_frac": positive_frac,
        "negative_frac": negative_frac,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "perm_pvalue": float(perm_pvalue),
    }


def print_paired_mean_test_summary(title, bias_values, full_values, *, seed=0):
    stats = paired_mean_difference_test(bias_values, full_values, seed=seed)
    print(title)
    print(
        "  "
        f"mean(full-bias)={stats['mean_diff']:.6f}, "
        f"median={stats['median_diff']:.6f}, "
        f"95% CI=[{stats['ci_low']:.6f}, {stats['ci_high']:.6f}], "
        f"perm-p={stats['perm_pvalue']:.6g}, "
        f"neg-frac={stats['negative_frac']:.3f}"
    )
    return stats


def participation_ratio(evals, eps=1e-12):
    evals = np.asarray(evals)
    evals = evals[evals > eps]
    if evals.size == 0:
        return 0.0
    return (np.sum(evals) ** 2) / (np.sum(evals ** 2) + eps)


def wrapped_task_angles(angle_values):
    """Map task angles to a consistent [0, 2pi) range for cyclic coloring."""
    angle_values = np.asarray(angle_values, dtype=float)
    if angle_values.size == 0:
        return angle_values
    if np.nanmin(angle_values) >= -1e-9 and np.nanmax(angle_values) <= 2 * np.pi + 1e-9:
        return np.mod(angle_values, 2 * np.pi)
    amin = np.nanmin(angle_values)
    span = np.nanmax(angle_values) - amin
    if span < 1e-12:
        return np.zeros_like(angle_values)
    return 2 * np.pi * (angle_values - amin) / span


def angle_color_norm(angle_values):
    angle_values = np.asarray(angle_values, dtype=float)
    if angle_values.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    if np.nanmin(angle_values) >= -1e-9 and np.nanmax(angle_values) <= 2 * np.pi + 1e-9:
        return Normalize(vmin=0.0, vmax=2 * np.pi)
    return Normalize(vmin=np.nanmin(angle_values), vmax=np.nanmax(angle_values))


def pca_coords_from_vectors(V, n_plot_components=3, eps=1e-12):
    """3D PCA coordinates plus the full task-space variance spectrum."""
    V = np.asarray(V, dtype=float)
    if V.ndim != 2:
        raise ValueError("Expected a task-by-parameter matrix for PCA.")

    Vc = V - np.mean(V, axis=0, keepdims=True)
    U, sing_vals, _ = np.linalg.svd(Vc, full_matrices=False)
    denom = max(V.shape[0] - 1, 1)
    evals = (sing_vals ** 2) / denom
    total = np.sum(evals)
    var_ratio = evals / (total + eps) if total > 0 else np.zeros_like(evals)
    k = min(n_plot_components, U.shape[1])
    coords = U[:, :k] * sing_vals[:k]
    return coords, evals, var_ratio


def coords_from_similarity_gram(G, n_components=3, eps=1e-12):
    """
    Kernel-PCA-style coordinates from a task Gram/similarity matrix.

    G should be an inner-product Gram matrix between task corrections. If G is a
    normalized cosine matrix, this embeds the normalized correction vectors.
    """
    G = np.asarray(G)
    G = 0.5 * (G + G.T)

    # Center in feature space for PCA-style manifold visualization.
    T = G.shape[0]
    H = np.eye(T) - np.ones((T, T)) / T
    Gc = H @ G @ H

    evals, evecs = np.linalg.eigh(Gc)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    evals_pos = np.clip(evals, 0.0, None)
    k = min(n_components, T)
    coords = evecs[:, :k] * np.sqrt(evals_pos[:k] + eps)
    total = np.sum(evals_pos) + eps
    var_ratio = evals_pos / total
    return coords, evals_pos, var_ratio


def align_coords_to_task_angles(coords, angle_values, eps=1e-12):
    """
    Canonicalize the embedding phase so the same task angle lands at the same
    position around the loop across related plots.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[0] == 0:
        return coords

    aligned = coords - np.mean(coords, axis=0, keepdims=True)
    theta = wrapped_task_angles(angle_values)

    if theta.shape[0] != aligned.shape[0]:
        return aligned

    if aligned.shape[1] >= 2:
        ref = np.column_stack([np.cos(theta), np.sin(theta)])
        cross = aligned[:, :2].T @ ref
        if np.linalg.norm(cross) > eps:
            U, _, Vt = np.linalg.svd(cross, full_matrices=False)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt
            aligned[:, :2] = aligned[:, :2] @ R
    elif aligned.shape[1] == 1 and np.dot(aligned[:, 0], np.cos(theta)) < 0:
        aligned[:, 0] *= -1

    return aligned


def _plot_similarity_scatter(ax, dists, sim_bias, sim_full, title, ylabel, xlabel, *, show_fit=True):
    colors = {"bias": "#c44e52", "full": "0.25"}
    labels = {"bias": "Bias-only", "full": "Full"}

    for sims, key in [(sim_bias, "bias"), (sim_full, "full")]:
        ax.scatter(dists, sims, s=22, alpha=0.48, color=colors[key], label=f"{labels[key]} pairs")
        if show_fit:
            slope, intercept, r2 = linear_fit_stats(dists, sims)
            xr = np.linspace(np.min(dists), np.max(dists), 200)
            ax.plot(
                xr,
                slope * xr + intercept,
                color=colors[key],
                lw=2.2,
                ls="--",
                label=f"{labels[key]} fit, $R^2={r2:.2f}$",
            )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize="small")


def _plot_paired_difference_distribution(ax, diff_values, title):
    diff_values = np.asarray(diff_values, dtype=float)
    stats = summary_stats(diff_values)
    bins = min(24, max(8, int(np.sqrt(max(diff_values.size, 1)))))
    ax.hist(diff_values, bins=bins, color="#4c72b0", alpha=0.75, edgecolor="white")
    ax.axvline(0.0, color="0.15", lw=1.2, ls="--")
    ax.axvline(stats["mean"], color="#c44e52", lw=1.6)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(r"$\Delta$ similarity", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.2)
    ax.text(
        0.03,
        0.97,
        f"mean={stats['mean']:.3f}\n|.|={stats['mean_abs']:.3f}\nrms={stats['rms']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.8", alpha=0.92),
    )


def _plot_difference_vs_distance(ax, dists, diff_values, title, *, show_fit=True):
    dists = np.asarray(dists, dtype=float)
    diff_values = np.asarray(diff_values, dtype=float)
    ax.scatter(dists, diff_values, s=16, alpha=0.55, color="#4c72b0", edgecolors="none")
    ax.axhline(0.0, color="0.15", lw=1.1, ls=":")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Task dist.", fontsize=8)
    ax.set_ylabel(r"$\Delta$ sim", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.2)
    if show_fit:
        slope, intercept, r2 = linear_fit_stats(dists, diff_values)
        xr = np.linspace(np.min(dists), np.max(dists), 200)
        ax.plot(xr, slope * xr + intercept, color="#c44e52", lw=1.8, ls="--")
        ax.text(
            0.03,
            0.97,
            f"slope={slope:.3f}\n$R^2$={r2:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.8", alpha=0.92),
        )


def _plot_interference_scatter(ax, x_bias, y_bias, x_full, y_full, title, xlabel, ylabel, *, show_fit=True):
    colors = {"bias": "#c44e52", "full": "0.25"}
    labels = {"bias": "Bias-only", "full": "Full"}

    for x, y, key in [(x_bias, y_bias, "bias"), (x_full, y_full, "full")]:
        ax.scatter(x, y, s=18, alpha=0.45, color=colors[key], label=f"{labels[key]} pairs")
        if show_fit:
            slope, intercept, r2 = linear_fit_stats(x, y)
            xr = np.linspace(np.min(x), np.max(x), 200)
            ax.plot(
                xr,
                slope * xr + intercept,
                color=colors[key],
                lw=2.0,
                ls="--",
                label=f"{labels[key]} fit, $R^2={r2:.2f}$",
            )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize="small")


def _group_distance_summary(x, y, *, max_exact_groups=20, n_bins=8):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    uniq = np.unique(np.round(x, 10))
    if uniq.size <= max_exact_groups:
        centers = uniq
        groups = [y[np.isclose(x, val, atol=1e-9)] for val in centers]
        return centers, groups, True

    edges = np.linspace(float(np.min(x)), float(np.max(x)), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    groups = []
    used_centers = []
    for idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (x >= lo) & (x <= hi) if idx == n_bins - 1 else (x >= lo) & (x < hi)
        if np.any(mask):
            used_centers.append(centers[idx])
            groups.append(y[mask])
    return np.asarray(used_centers, dtype=float), groups, False


def _plot_interference_distance_summary(ax, x_bias, y_bias, x_full, y_full, title, xlabel, ylabel):
    colors = {"bias": "#c44e52", "full": "0.25"}
    labels = {"bias": "Bias-only", "full": "Full"}

    for x, y, key in [(x_bias, y_bias, "bias"), (x_full, y_full, "full")]:
        centers, groups, used_exact = _group_distance_summary(x, y)
        means = np.asarray([np.mean(group) for group in groups], dtype=float)
        sems = np.asarray(
            [np.std(group, ddof=1) / np.sqrt(len(group)) if len(group) > 1 else 0.0 for group in groups],
            dtype=float,
        )
        ax.errorbar(
            centers,
            means,
            yerr=sems,
            color=colors[key],
            lw=2.0,
            marker="o",
            ms=5.0,
            capsize=3.0,
            label=labels[key],
        )
        if used_exact and centers.size <= 20:
            ax.set_xticks(centers)
            if np.max(centers) <= 1.000001:
                ax.set_xticklabels([f"{val:.2f}" if val > 1e-9 else "0" for val in centers], rotation=45, ha="right")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize="small")


def _plot_phase_conditioned_scatter(ax, x, y, phase_compat, title, xlabel, ylabel):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    phase_compat = np.asarray(phase_compat, dtype=float)
    sc = ax.scatter(
        x,
        y,
        c=phase_compat,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        s=20,
        alpha=0.75,
        edgecolors="none",
    )
    slope, intercept, r2 = linear_fit_stats(x, y)
    xr = np.linspace(np.min(x), np.max(x), 200)
    ax.plot(xr, slope * xr + intercept, color="0.15", lw=1.5, ls="--")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
    ax.text(
        0.03,
        0.97,
        f"slope={slope:.3f}\n$R^2$={r2:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.8", alpha=0.92),
    )
    return sc


def _plot_task_manifold(
    ax,
    coords,
    color_values,
    var_ratio,
    evals,
    title,
    norm,
    *,
    cmap="twilight_shifted",
    align_to_cycle=True,
    connect_ordered=True,
):
    color_values = np.asarray(color_values, dtype=float)
    if align_to_cycle:
        coords = align_coords_to_task_angles(coords, color_values)
        plot_colors = wrapped_task_angles(color_values)
        plot_order = np.argsort(plot_colors)
    else:
        coords = np.asarray(coords, dtype=float) - np.mean(coords, axis=0, keepdims=True)
        plot_colors = color_values
        plot_order = np.argsort(plot_colors)

    # pad coords when only 1 or 2 components exist
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])), mode="constant")

    sc = ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=plot_colors,
        cmap=cmap,
        norm=norm,
        s=60,
        edgecolors="k",
        linewidth=0.5,
    )
    if connect_ordered and plot_order.size:
        line_order = np.append(plot_order, plot_order[0]) if align_to_cycle else plot_order
        ax.plot(coords[line_order, 0], coords[line_order, 1], coords[line_order, 2], "k-", alpha=0.25, lw=1.0)
    pr = participation_ratio(evals)
    total3 = 100 * np.sum(var_ratio[:3])
    rank = np.sum(np.asarray(evals) > 1e-12)
    ax.set_title(f"{title}\n3-PC var={total3:.1f}%, PR={pr:.2f}, rank={rank}")
    ax.set_xlabel(f"PC1 ({100 * var_ratio[0]:.1f}%)" if len(var_ratio) > 0 else "PC1")
    ax.set_ylabel(f"PC2 ({100 * var_ratio[1]:.1f}%)" if len(var_ratio) > 1 else "PC2")
    ax.set_zlabel(f"PC3 ({100 * var_ratio[2]:.1f}%)" if len(var_ratio) > 2 else "PC3")
    ax.view_init(elev=20, azim=45)
    return sc


def _plot_pc_spectrum(ax, evals, title):
    evals = np.asarray(evals)
    evals = evals[evals > 1e-12]
    if evals.size == 0:
        ax.text(0.5, 0.5, "No positive eigenvalues", ha="center", va="center")
        ax.set_axis_off()
        return
    frac = evals / np.sum(evals)
    pr = participation_ratio(evals)
    xs = np.arange(1, len(frac) + 1)
    ax.plot(xs, frac, marker="o", lw=1.8)
    ax.set_title(f"{title}\nPR={pr:.2f}, rank={evals.size}")
    ax.set_xlabel("Component")
    ax.set_ylabel("Explained variance fraction")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)


def _plot_pc_spectra_overlay(ax, evals_bias, evals_full, title):
    series = [
        ("Bias-only", np.asarray(evals_bias), "#c44e52"),
        ("Full", np.asarray(evals_full), "0.25"),
    ]
    pr_parts = []
    rank_parts = []
    for label, evals, color in series:
        evals = evals[evals > 1e-12]
        if evals.size == 0:
            continue
        frac = evals / np.sum(evals)
        xs = np.arange(1, len(frac) + 1)
        ax.plot(xs, frac, marker="o", lw=1.8, ms=4, color=color, label=label)
        pr_parts.append(f"{label} PR={participation_ratio(evals):.2f}")
        rank_parts.append(f"{label} rank={evals.size}")
    ax.set_title(title + ("\n" + " | ".join(pr_parts) if pr_parts else ""))
    ax.set_xlabel("Component")
    ax.set_ylabel("Explained variance fraction")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)


def task_family_plot_config(task_family):
    if task_family == "single_axis":
        return {
            "color_label": "Task angle (rad)",
            "color_is_cyclic": True,
            "align_manifold_to_colors": True,
            "connect_manifold": True,
            "manifold_cmap": "twilight_shifted",
            "distance_label": "Task angular distance (rad)",
            "distance_mode": "angles",
        }
    if task_family == "sampled":
        return {
            "color_label": "Task index",
            "color_is_cyclic": False,
            "align_manifold_to_colors": False,
            "connect_manifold": False,
            "manifold_cmap": "viridis",
            "distance_label": "Task support distance (rad)",
            "distance_mode": "normals",
        }
    raise ValueError(f"Unsupported task_family={task_family}")


def _draw_unit_sphere_wireframe(ax, *, color="0.84", alpha=0.32):
    u = np.linspace(0.0, 2 * np.pi, 60)
    v = np.linspace(0.0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, rstride=3, cstride=3, color=color, linewidth=0.45, alpha=alpha)


def _set_equal_sphere_axes(ax, *, lim=1.18):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xticks([-1.0, 0.0, 1.0])
    ax.set_yticks([-1.0, 0.0, 1.0])
    ax.set_zticks([-1.0, 0.0, 1.0])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))


def select_task_geometry_highlights(task_family, color_values, max_highlights=5):
    color_values = np.asarray(color_values, dtype=float)
    n_tasks = color_values.size
    if n_tasks <= max_highlights:
        return list(range(n_tasks))

    if task_family == "single_axis":
        anchor_angles = np.array([0.0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, 1.0 * np.pi], dtype=float)
        chosen = []
        for angle in anchor_angles:
            idx = int(np.argmin(np.abs(color_values - angle)))
            if idx not in chosen:
                chosen.append(idx)
        return chosen

    return list(np.linspace(0, n_tasks - 1, min(max_highlights, n_tasks), dtype=int))


def format_task_geometry_label(task_family, color_values, task_index):
    if task_family == "single_axis":
        angle_deg = float(np.degrees(np.mod(color_values[task_index], 2 * np.pi)))
        return rf"$\theta={angle_deg:.0f}^\circ$"
    return f"task {task_index}"


def paper_task_color_encoding(task_family, color_values):
    color_values = np.asarray(color_values, dtype=float)
    if task_family == "single_axis":
        return {
            "values": np.mod(color_values, np.pi) / np.pi,
            "norm": Normalize(vmin=0.0, vmax=1.0),
            "cmap": "viridis",
            "label": r"Task support angle $\theta / \pi$",
            "ticks": [0.0, 0.5, 1.0],
            "ticklabels": ["0", "0.5", "1.0"],
        }
    vmin = float(np.nanmin(color_values)) if color_values.size else 0.0
    vmax = float(np.nanmax(color_values)) if color_values.size else 1.0
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return {
        "values": color_values,
        "norm": Normalize(vmin=vmin, vmax=vmax),
        "cmap": "viridis",
        "label": "Task index",
        "ticks": None,
        "ticklabels": None,
    }


def plot_task_family_geometry_overview(
    *,
    normals,
    phases,
    target_phase,
    target_mode="local_harmonic",
    global_field_name="xz",
    m,
    task_family,
    color_values,
    title_prefix,
    save_path=None,
    highlight_indices=None,
    rotation_axis=(1.0, 0.0, 0.0),
):
    normals = np.asarray(normals, dtype=float)
    phases = np.asarray(phases, dtype=float)
    color_values = np.asarray(color_values, dtype=float)
    plot_cfg = task_family_plot_config(task_family)

    if normals.shape[0] == 0:
        return None

    if highlight_indices is None:
        highlight_indices = select_task_geometry_highlights(task_family, color_values)
    highlight_indices = [int(idx) for idx in highlight_indices]
    highlight_set = set(highlight_indices)

    if plot_cfg["color_is_cyclic"]:
        manifold_colors = wrapped_task_angles(color_values)
        task_color_norm = angle_color_norm(manifold_colors)
    else:
        manifold_colors = color_values
        vmin = float(np.nanmin(manifold_colors)) if manifold_colors.size else 0.0
        vmax = float(np.nanmax(manifold_colors)) if manifold_colors.size else 1.0
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        task_color_norm = Normalize(vmin=vmin, vmax=vmax)
    task_cmap = plt.get_cmap(plot_cfg["manifold_cmap"])

    fig = plt.figure(figsize=(15.6, 7.6))
    ax_tasks = fig.add_subplot(1, 2, 1, projection="3d")
    ax_normals = fig.add_subplot(1, 2, 2, projection="3d")

    _draw_unit_sphere_wireframe(ax_tasks)
    _draw_unit_sphere_wireframe(ax_normals, color="0.86", alpha=0.26)

    rotation_axis = np.asarray(rotation_axis, dtype=float)
    if np.linalg.norm(rotation_axis) > 0:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    else:
        rotation_axis = np.array([1.0, 0.0, 0.0], dtype=float)

    if task_family == "single_axis":
        for ax in [ax_tasks, ax_normals]:
            ax.plot(
                [-1.18 * rotation_axis[0], 1.18 * rotation_axis[0]],
                [-1.18 * rotation_axis[1], 1.18 * rotation_axis[1]],
                [-1.18 * rotation_axis[2], 1.18 * rotation_axis[2]],
                ls="--",
                lw=1.2,
                color="0.15",
                alpha=0.75,
            )
            axis_text_pos = 1.28 * rotation_axis
            ax.text(
                axis_text_pos[0],
                axis_text_pos[1],
                axis_text_pos[2],
                "rotation axis",
                fontsize=9,
                color="0.15",
                ha="center",
                va="center",
            )

    target_scatter = None
    for idx, (normal, phase) in enumerate(zip(normals, phases)):
        X, y, _ = generate_circle_samples(
            normal,
            240,
            int(m),
            phase=float(phase),
            target_phase=float(target_phase),
            target_mode=target_mode,
            global_field_name=global_field_name,
            mode="grid",
        )
        task_color = task_cmap(task_color_norm(manifold_colors[idx]))
        if idx in highlight_set:
            ax_tasks.plot(X[:, 0], X[:, 1], X[:, 2], color=task_color, lw=1.3, alpha=0.82)
            target_scatter = ax_tasks.scatter(
                X[:, 0],
                X[:, 1],
                X[:, 2],
                c=y,
                cmap="coolwarm",
                vmin=-1.0,
                vmax=1.0,
                s=14,
                alpha=0.98,
                edgecolors="none",
                depthshade=False,
            )
            label_idx = int(np.argmax(X[:, 2] + 0.25 * X[:, 0]))
            label_pos = 1.10 * X[label_idx]
            ax_tasks.text(
                label_pos[0],
                label_pos[1],
                label_pos[2],
                format_task_geometry_label(task_family, color_values, idx),
                fontsize=9,
                color=task_color,
                ha="center",
                va="center",
            )
        else:
            ax_tasks.plot(X[:, 0], X[:, 1], X[:, 2], color="0.58", lw=0.8, alpha=0.18)

    if task_family == "single_axis":
        order = np.argsort(color_values)
        ax_normals.plot(
            normals[order, 0],
            normals[order, 1],
            normals[order, 2],
            color="0.65",
            lw=1.2,
            alpha=0.75,
        )
    ax_normals.scatter(normals[:, 0], normals[:, 1], normals[:, 2], color="0.67", s=26, alpha=0.22, depthshade=False)
    for idx in highlight_indices:
        normal = normals[idx]
        task_color = task_cmap(task_color_norm(manifold_colors[idx]))
        ax_normals.plot([0.0, normal[0]], [0.0, normal[1]], [0.0, normal[2]], color=task_color, lw=1.2, alpha=0.9)
        ax_normals.scatter(
            [normal[0]],
            [normal[1]],
            [normal[2]],
            color=[task_color],
            s=60,
            edgecolors="k",
            linewidth=0.45,
            depthshade=False,
        )
        label_pos = 1.10 * normal
        ax_normals.text(
            label_pos[0],
            label_pos[1],
            label_pos[2],
            format_task_geometry_label(task_family, color_values, idx),
            fontsize=9,
            color=task_color,
            ha="center",
            va="center",
        )

    ax_tasks.set_title("Task supports on the sphere\nHighlighted circles colored by target value")
    ax_normals.set_title("Great-circle normals generating the family")
    ax_tasks.view_init(elev=20, azim=36)
    ax_normals.view_init(elev=18, azim=36)
    _set_equal_sphere_axes(ax_tasks)
    _set_equal_sphere_axes(ax_normals)

    if target_scatter is not None:
        cbar = fig.colorbar(target_scatter, ax=[ax_tasks], shrink=0.75, pad=0.04)
        cbar.set_label("Target value on highlighted circles")

    fig.suptitle(title_prefix, fontsize=16, y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


def _resolve_manifold_embedding(*, manifold_kind, manifold_bias, manifold_full, precomputed_bias=None, precomputed_full=None):
    if precomputed_bias is not None and precomputed_full is not None:
        return (
            np.asarray(precomputed_bias["coords"], dtype=float),
            np.asarray(precomputed_bias["evals"], dtype=float),
            np.asarray(precomputed_bias["var_ratio"], dtype=float),
            np.asarray(precomputed_full["coords"], dtype=float),
            np.asarray(precomputed_full["evals"], dtype=float),
            np.asarray(precomputed_full["var_ratio"], dtype=float),
        )
    if manifold_kind == "vectors":
        coords_bias, evals_bias, vr_bias = pca_coords_from_vectors(np.asarray(manifold_bias))
        coords_full, evals_full, vr_full = pca_coords_from_vectors(np.asarray(manifold_full))
        return coords_bias, evals_bias, vr_bias, coords_full, evals_full, vr_full
    if manifold_kind == "gram":
        coords_bias, evals_bias, vr_bias = coords_from_similarity_gram(np.asarray(manifold_bias))
        coords_full, evals_full, vr_full = coords_from_similarity_gram(np.asarray(manifold_full))
        return coords_bias, evals_bias, vr_bias, coords_full, evals_full, vr_full
    raise ValueError("manifold_kind must be 'vectors' or 'gram'")


def _plot_clean_task_manifold_panel(
    ax,
    coords,
    color_values,
    color_norm,
    *,
    cmap,
    title,
    highlight_indices=None,
    align_angle_values=None,
    connect_ordered=True,
):
    coords = np.asarray(coords, dtype=float)
    if align_angle_values is not None:
        coords = align_coords_to_task_angles(coords, align_angle_values)
        plot_order = np.argsort(np.asarray(align_angle_values, dtype=float))
    else:
        coords = coords - np.mean(coords, axis=0, keepdims=True)
        plot_order = np.argsort(np.asarray(color_values, dtype=float))
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])), mode="constant")

    if connect_ordered and plot_order.size:
        ax.plot(
            coords[plot_order, 0],
            coords[plot_order, 1],
            coords[plot_order, 2],
            color="0.35",
            lw=1.1,
            alpha=0.35,
        )

    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=color_values,
        cmap=cmap,
        norm=color_norm,
        s=46,
        edgecolors="none",
        alpha=0.82,
        depthshade=False,
    )

    if highlight_indices is not None:
        hi = np.asarray(highlight_indices, dtype=int)
        ax.scatter(
            coords[hi, 0],
            coords[hi, 1],
            coords[hi, 2],
            c=color_values[hi],
            cmap=cmap,
            norm=color_norm,
            s=76,
            edgecolors="k",
            linewidth=0.7,
            alpha=1.0,
            depthshade=False,
        )

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.view_init(elev=20, azim=42)
    return sc


def _plot_similarity_shift_panel(fig, outer_spec, *, dists, bias_vals, full_vals, color, title, distance_label, seed=0):
    subgs = outer_spec.subgridspec(1, 2, width_ratios=[4.1, 1.0], wspace=0.04)
    ax_scatter = fig.add_subplot(subgs[0, 0])
    ax_hist = fig.add_subplot(subgs[0, 1], sharey=ax_scatter)

    dists = np.asarray(dists, dtype=float)
    bias_vals = np.asarray(bias_vals, dtype=float)
    full_vals = np.asarray(full_vals, dtype=float)
    diff_vals = full_vals - bias_vals
    stats = paired_mean_difference_test(bias_vals, full_vals, seed=seed)

    ax_scatter.scatter(dists, diff_vals, s=18, alpha=0.42, color=color, edgecolors="none")
    ax_scatter.axhline(0.0, color="0.2", lw=1.1, ls=":")
    x_min = float(np.min(dists))
    x_max = float(np.max(dists))
    if x_max > x_min:
        edges = np.linspace(x_min, x_max, 13)
        centers = 0.5 * (edges[:-1] + edges[1:])
        mean_x = []
        mean_y = []
        for lo, hi, center in zip(edges[:-1], edges[1:], centers):
            mask = (dists >= lo) & (dists <= hi) if hi == edges[-1] else (dists >= lo) & (dists < hi)
            if np.any(mask):
                mean_x.append(center)
                mean_y.append(float(np.mean(diff_vals[mask])))
        if mean_x:
            ax_scatter.plot(mean_x, mean_y, color="black", lw=2.0)
    ax_scatter.set_title(title)
    ax_scatter.set_xlabel(distance_label)
    ax_scatter.set_ylabel(r"Off-diagonal $\Delta$ cosine similarity")
    ax_scatter.grid(True, alpha=0.22)
    ax_scatter.text(
        0.03,
        0.97,
        (
            f"mean={stats['mean_diff']:.3f}\n"
            f"95% CI=[{stats['ci_low']:.3f}, {stats['ci_high']:.3f}]\n"
            f"perm-p={stats['perm_pvalue']:.2g}"
        ),
        transform=ax_scatter.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="0.8", alpha=0.94),
    )

    bins = min(24, max(10, int(np.sqrt(max(diff_vals.size, 1)))))
    ax_hist.hist(diff_vals, bins=bins, orientation="horizontal", color=color, alpha=0.78, edgecolor="white")
    ax_hist.axhline(0.0, color="0.2", lw=1.1, ls=":")
    ax_hist.axhline(stats["mean_diff"], color="black", lw=1.6)
    ax_hist.set_xlabel("Count")
    ax_hist.grid(True, axis="x", alpha=0.18)
    ax_hist.tick_params(axis="y", labelleft=False)
    for spine in ["top", "right"]:
        ax_hist.spines[spine].set_visible(False)

    return ax_scatter, ax_hist


def _format_distance_bin_labels(edges, *, use_pi=False):
    labels = []
    if use_pi:
        pretty = {
            0.0: "0",
            0.25 * np.pi: r"$\pi/4$",
            0.5 * np.pi: r"$\pi/2$",
            0.75 * np.pi: r"$3\pi/4$",
            1.0 * np.pi: r"$\pi$",
        }

        def _pretty(v):
            for key, label in pretty.items():
                if np.isclose(v, key, atol=1e-6):
                    return label
            return f"{v:.2f}"

        for lo, hi in zip(edges[:-1], edges[1:]):
            labels.append(f"{_pretty(lo)}-{_pretty(hi)}")
    else:
        for lo, hi in zip(edges[:-1], edges[1:]):
            labels.append(f"{lo:.2f}-{hi:.2f}")
    return labels


def _format_distance_value_labels(values, *, use_pi=False):
    values = np.asarray(values, dtype=float)
    if use_pi:
        return [f"{(val / np.pi):.2f}" if val > 1e-9 else "0" for val in values]
    return [f"{val:.2f}" for val in values]


def _plot_similarity_boxplot_panel(
    ax,
    *,
    dists,
    series,
    distance_label,
    title,
    use_pi_labels=False,
    use_exact_groups=False,
):
    dists = np.asarray(dists, dtype=float)
    if use_exact_groups:
        group_values = np.unique(np.round(dists, 10))
        group_labels = _format_distance_value_labels(group_values, use_pi=use_pi_labels)
        group_masks = [np.isclose(dists, gv, atol=1e-9) for gv in group_values]
    else:
        n_bins = 4
        x_min = float(np.min(dists))
        x_max = float(np.max(dists))
        edges = np.linspace(x_min, x_max, n_bins + 1)
        group_labels = _format_distance_bin_labels(edges, use_pi=use_pi_labels)
        group_masks = []
        for bin_idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            group_masks.append((dists >= lo) & (dists <= hi) if bin_idx == n_bins - 1 else (dists >= lo) & (dists < hi))

    n_series = len(series)
    cluster_spacing = n_series + 1.8
    centers = []
    legend_handles = []
    legend_labels = []

    for bin_idx, mask in enumerate(group_masks):
        centers.append(bin_idx * cluster_spacing + 0.5 * (n_series - 1))
        for series_idx, (label, values, color) in enumerate(series):
            values = np.asarray(values, dtype=float)
            bin_values = values[mask]
            if bin_values.size == 0:
                continue
            pos = bin_idx * cluster_spacing + series_idx
            bp = ax.boxplot(
                [bin_values],
                positions=[pos],
                widths=0.78,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="black", linewidth=1.2),
                whiskerprops=dict(color=color, linewidth=1.0),
                capprops=dict(color=color, linewidth=1.0),
                boxprops=dict(facecolor=color, edgecolor=color, alpha=0.38, linewidth=1.1),
            )
            if bin_idx == 0:
                legend_handles.append(bp["boxes"][0])
                legend_labels.append(label)

    ax.set_xticks(centers)
    ax.set_xticklabels(group_labels, rotation=45 if use_exact_groups else 0, ha="right" if use_exact_groups else "center")
    ax.tick_params(axis="x", labelsize=8 if use_exact_groups else 9)
    ax.set_title(title)
    ax.set_xlabel(distance_label if use_exact_groups else distance_label + " bins")
    ax.set_ylabel("Task-pair cosine similarity")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(
        legend_handles,
        legend_labels,
        fontsize=8,
        frameon=True,
        facecolor="white",
        edgecolor="0.85",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        columnspacing=1.2,
        handletextpad=0.5,
    )


def plot_paper_intro_summary_figure(
    *,
    task_family,
    normals,
    phases,
    target_phase,
    target_mode="local_harmonic",
    global_field_name="xz",
    m,
    color_values,
    finite_bias_sim,
    finite_full_sim,
    finite_rkhs_bias_sim,
    finite_rkhs_full_sim,
    infinite_bias_sim,
    infinite_full_sim,
    distance_label,
    title_prefix,
    finite_manifold_bias=None,
    finite_manifold_full=None,
    finite_precomputed_bias=None,
    finite_precomputed_full=None,
    save_path=None,
    seed=0,
):
    highlight_indices = select_task_geometry_highlights(task_family, color_values)
    paper_colors = paper_task_color_encoding(task_family, color_values)
    display_color_values = np.asarray(paper_colors["values"], dtype=float)
    display_color_norm = paper_colors["norm"]
    display_cmap = paper_colors["cmap"]

    if task_family == "single_axis":
        dists, pairs = pairwise_distances_from_angles(color_values)
        align_angle_values = np.asarray(color_values, dtype=float)
        display_dists = dists / np.pi
        paper_distance_label = r"Task angular distance $/ \pi$"
    else:
        dists, pairs = pairwise_distances_from_normals(normals)
        align_angle_values = None
        display_dists = dists
        paper_distance_label = distance_label

    finite_bias_vals = upper_tri_values(finite_bias_sim, pairs)
    finite_full_vals = upper_tri_values(finite_full_sim, pairs)
    infinite_bias_vals = upper_tri_values(infinite_bias_sim, pairs)
    infinite_full_vals = upper_tri_values(infinite_full_sim, pairs)

    coords_bias, evals_bias, vr_bias, coords_full, evals_full, vr_full = _resolve_manifold_embedding(
        manifold_kind="vectors",
        manifold_bias=finite_manifold_bias,
        manifold_full=finite_manifold_full,
        precomputed_bias=finite_precomputed_bias,
        precomputed_full=finite_precomputed_full,
    )

    fig = plt.figure(figsize=(20.5, 10.8))
    outer = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.0, 1.28, 1.28, 1.0],
        height_ratios=[1.02, 0.98],
        left=0.04,
        right=0.985,
        top=0.93,
        bottom=0.08,
        wspace=0.28,
        hspace=0.34,
    )

    ax_tasks = fig.add_subplot(outer[0, 0], projection="3d")
    ax_summary = fig.add_subplot(outer[0, 1:3])
    _draw_unit_sphere_wireframe(ax_tasks)

    target_scatter = None
    if task_family == "single_axis":
        ax_tasks.plot([-1.18, 1.18], [0.0, 0.0], [0.0, 0.0], ls="--", lw=1.15, color="0.15", alpha=0.75)
        ax_tasks.annotate(
            "",
            xy=(0.60, 0.885),
            xytext=(0.44, 0.885),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", lw=1.1, color="0.15", connectionstyle="arc3,rad=-0.22"),
        )

    highlight_set = set(highlight_indices)
    for idx, (normal, phase) in enumerate(zip(np.asarray(normals, dtype=float), np.asarray(phases, dtype=float))):
        X, y, _ = generate_circle_samples(
            normal,
            240,
            int(m),
            phase=float(phase),
            target_phase=float(target_phase),
            target_mode=target_mode,
            global_field_name=global_field_name,
            mode="grid",
        )
        if idx in highlight_set:
            target_scatter = ax_tasks.scatter(
                X[:, 0], X[:, 1], X[:, 2],
                c=y, cmap="coolwarm", vmin=-1.0, vmax=1.0,
                s=13, alpha=0.98, edgecolors="none", depthshade=False,
            )
        else:
            ax_tasks.plot(X[:, 0], X[:, 1], X[:, 2], color="0.62", lw=0.85, alpha=0.16)

    ax_tasks.set_title("Task family on the sphere")
    ax_tasks.view_init(elev=20, azim=36)
    _set_equal_sphere_axes(ax_tasks)

    ax_full = fig.add_subplot(outer[0, 3], projection="3d")
    ax_bias = fig.add_subplot(outer[1, 3], projection="3d")
    sc_manifold = _plot_clean_task_manifold_panel(
        ax_full,
        coords_full,
        display_color_values,
        display_color_norm,
        cmap=display_cmap,
        title="Full: finite-width parameter-update manifold",
        highlight_indices=highlight_indices,
        align_angle_values=align_angle_values,
    )
    _plot_clean_task_manifold_panel(
        ax_bias,
        coords_bias,
        display_color_values,
        display_color_norm,
        cmap=display_cmap,
        title="Bias-only: finite-width parameter-update manifold",
        highlight_indices=highlight_indices,
        align_angle_values=align_angle_values,
    )

    similarity_series = [
        (r"finite bias $\Delta b$", finite_bias_vals, "#c44e52"),
        (r"finite full $\Delta \theta$", finite_full_vals, "0.25"),
        (r"finite bias $\delta f$", upper_tri_values(finite_rkhs_bias_sim, pairs), "#dd8452"),
        (r"finite full $\delta f$", upper_tri_values(finite_rkhs_full_sim, pairs), "#8172b2"),
        (r"infinite bias $\delta f$", infinite_bias_vals, "#64b5cd"),
        (r"infinite full $\delta f$", infinite_full_vals, "#55a868"),
    ]
    _plot_similarity_boxplot_panel(
        ax_summary,
        dists=display_dists,
        series=similarity_series,
        distance_label=paper_distance_label,
        title="Task-pair similarity vs distance",
        use_pi_labels=False,
        use_exact_groups=(task_family == "single_axis"),
    )

    bottom_pair = outer[1, 0:3].subgridspec(1, 2, wspace=0.18)
    _plot_similarity_shift_panel(
        fig,
        bottom_pair[0, 0],
        dists=display_dists,
        bias_vals=finite_bias_vals,
        full_vals=finite_full_vals,
        color="#4c72b0",
        title="Empirical finite width: off-diagonal full - bias similarity",
        distance_label=paper_distance_label,
        seed=seed,
    )
    _plot_similarity_shift_panel(
        fig,
        bottom_pair[0, 1],
        dists=display_dists,
        bias_vals=infinite_bias_vals,
        full_vals=infinite_full_vals,
        color="#55a868",
        title="Infinite width: off-diagonal full - bias similarity",
        distance_label=paper_distance_label,
        seed=seed,
    )

    task_bbox = ax_tasks.get_position()
    manifolds_left = min(ax_full.get_position().x0, ax_bias.get_position().x0)
    manifolds_right = max(ax_full.get_position().x1, ax_bias.get_position().x1)
    cbar_height = 0.018
    task_cbar_y = ax_tasks.get_position().y0 - 0.043
    manifold_cbar_y = ax_bias.get_position().y0 - 0.052

    if target_scatter is not None:
        target_width = min(0.20, 0.74 * task_bbox.width)
        cax_target = fig.add_axes([task_bbox.x0 + 0.5 * (task_bbox.width - target_width), task_cbar_y, target_width, cbar_height])
        cbar_target = fig.colorbar(target_scatter, cax=cax_target, orientation="horizontal")
        cbar_target.set_label("Target value on highlighted circles")

    manifold_width = min(0.22, 0.78 * (manifolds_right - manifolds_left))
    cax_task = fig.add_axes([manifolds_left + 0.5 * ((manifolds_right - manifolds_left) - manifold_width), manifold_cbar_y, manifold_width, cbar_height])
    cbar_task = fig.colorbar(sc_manifold, cax=cax_task, orientation="horizontal")
    cbar_task.set_label(paper_colors["label"])
    if paper_colors["ticks"] is not None:
        cbar_task.set_ticks(paper_colors["ticks"])
        cbar_task.set_ticklabels(paper_colors["ticklabels"])

    fig.suptitle(title_prefix, fontsize=17, y=0.975)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


def plot_task_manifold_summary(
    *,
    angles,
    sim_bias,
    sim_full,
    title_prefix,
    manifold_bias=None,
    manifold_full=None,
    manifold_kind="vectors",
    normals=None,
    vector_norms_bias=None,
    vector_norms_full=None,
    color_label="Task angle (rad)",
    color_is_cyclic=True,
    align_manifold_to_colors=True,
    connect_manifold=True,
    manifold_cmap="twilight_shifted",
    distance_label="Task angular distance (rad)",
    export_dir=None,
    export_stem=None,
    save_path=None,
    precomputed_bias=None,
    precomputed_full=None,
):
    """
    Summary figure requested for task-similarity/update-manifold analysis.

    Parameters
    ----------
    angles : array-like
        Task angle values for coloring. For one-axis rotations these are also
        used for pairwise distances unless normals is provided.
    sim_bias, sim_full : (T,T) arrays
        Pairwise cosine/similarity matrices for bias-only and full kernels.
    manifold_bias, manifold_full : arrays
        If manifold_kind='vectors', task-by-feature matrices or lists of vectors.
        If manifold_kind='gram', task Gram/similarity matrices.
    manifold_kind : {'vectors', 'gram'}
        Use explicit PCA for finite update vectors or kernel-PCA coordinates for
        infinite-width task-correction Gram matrices.
    normals : optional array, shape (T,3)
        If provided, pairwise great-circle distances are arccos(|n_i^T n_j|).
    """
    angles = np.asarray(angles, dtype=float)
    color_values = angles

    if normals is None:
        dists, pairs = pairwise_distances_from_angles(angles)
    else:
        dists, pairs = pairwise_distances_from_normals(normals)

    sim_bias_vals = upper_tri_values(sim_bias, pairs)
    sim_full_vals = upper_tri_values(sim_full, pairs)
    offdiag_bias = offdiag_values(sim_bias)
    offdiag_full = offdiag_values(sim_full)
    sim_diff_vals = sim_full_vals - sim_bias_vals

    if manifold_bias is None:
        manifold_bias = sim_bias if manifold_kind == "gram" else None
    if manifold_full is None:
        manifold_full = sim_full if manifold_kind == "gram" else None

    if precomputed_bias is not None and precomputed_full is not None:
        coords_bias = np.asarray(precomputed_bias["coords"], dtype=float)
        evals_bias = np.asarray(precomputed_bias["evals"], dtype=float)
        vr_bias = np.asarray(precomputed_bias["var_ratio"], dtype=float)
        coords_full = np.asarray(precomputed_full["coords"], dtype=float)
        evals_full = np.asarray(precomputed_full["evals"], dtype=float)
        vr_full = np.asarray(precomputed_full["var_ratio"], dtype=float)
        sim_ylabel = r"$\Delta$ cosine similarity" if manifold_kind == "vectors" else r"RKHS correction cosine"
    elif manifold_kind == "vectors":
        coords_bias, evals_bias, vr_bias = pca_coords_from_vectors(np.asarray(manifold_bias))
        coords_full, evals_full, vr_full = pca_coords_from_vectors(np.asarray(manifold_full))
        sim_ylabel = r"$\Delta$ cosine similarity"
    elif manifold_kind == "gram":
        coords_bias, evals_bias, vr_bias = coords_from_similarity_gram(np.asarray(manifold_bias))
        coords_full, evals_full, vr_full = coords_from_similarity_gram(np.asarray(manifold_full))
        sim_ylabel = r"RKHS correction cosine"
    else:
        raise ValueError("manifold_kind must be 'vectors' or 'gram'")

    if color_is_cyclic:
        plot_color_values = wrapped_task_angles(color_values)
        color_norm = angle_color_norm(plot_color_values)
    else:
        plot_color_values = color_values
        vmin = np.nanmin(plot_color_values) if plot_color_values.size else 0.0
        vmax = np.nanmax(plot_color_values) if plot_color_values.size else 1.0
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        color_norm = Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(17.4, 10.2))
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.22, 0.82, 1.0, 0.82],
        height_ratios=[1, 1],
        left=0.05,
        right=0.98,
        top=0.90,
        bottom=0.18,
        wspace=0.35,
        hspace=0.35,
    )

    ax_scatter = fig.add_subplot(gs[:, 0])
    _plot_similarity_scatter(
        ax_scatter,
        dists,
        sim_bias_vals,
        sim_full_vals,
        title=f"{title_prefix}: task similarity vs distance",
        ylabel=sim_ylabel,
        xlabel=distance_label,
        show_fit=False,
    )

    diag_lines = [
        format_compact_summary_line("Bias offdiag", summary_stats(offdiag_bias)),
        format_compact_summary_line("Full offdiag", summary_stats(offdiag_full)),
        format_compact_summary_line("Full-bias", summary_stats(sim_diff_vals)),
    ]
    if vector_norms_bias is not None and vector_norms_full is not None:
        diag_lines.extend(
            [
                format_compact_summary_line(r"Bias ||Δ||", summary_stats(vector_norms_bias)),
                format_compact_summary_line(r"Full ||Δ||", summary_stats(vector_norms_full)),
            ]
        )
    scatter_bbox = ax_scatter.get_position()
    diag_height = 0.068
    diag_y0 = 0.092
    ax_diag = fig.add_axes([scatter_bbox.x0, diag_y0, scatter_bbox.width, diag_height])
    ax_diag.axis("off")
    ax_diag.text(
        0.0,
        1.0,
        "\n".join(diag_lines),
        transform=ax_diag.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8", alpha=0.92),
    )

    ax_diff = fig.add_subplot(gs[0, 1])
    _plot_paired_difference_distribution(
        ax_diff,
        sim_diff_vals,
        "Off-diagonal task-pair full - bias similarity",
    )

    ax_delta = fig.add_subplot(gs[1, 1])
    _plot_difference_vs_distance(
        ax_delta,
        dists,
        sim_diff_vals,
        r"Off-diagonal $\Delta$ similarity vs distance",
        show_fit=False,
    )

    ax_full_3d = fig.add_subplot(gs[0, 2], projection="3d")
    sc_full = _plot_task_manifold(
        ax_full_3d,
        coords_full,
        plot_color_values,
        vr_full,
        evals_full,
        "Full",
        color_norm,
        cmap=manifold_cmap,
        align_to_cycle=align_manifold_to_colors,
        connect_ordered=connect_manifold,
    )

    ax_bias_3d = fig.add_subplot(gs[1, 2], projection="3d")
    _plot_task_manifold(
        ax_bias_3d,
        coords_bias,
        plot_color_values,
        vr_bias,
        evals_bias,
        "Bias-only",
        color_norm,
        cmap=manifold_cmap,
        align_to_cycle=align_manifold_to_colors,
        connect_ordered=connect_manifold,
    )

    ax_spec = fig.add_subplot(gs[:, 3])
    _plot_pc_spectra_overlay(ax_spec, evals_bias, evals_full, "PC spectrum")

    cax = fig.add_axes([0.355, 0.085, 0.22, 0.022])
    cbar = fig.colorbar(sc_full, cax=cax, orientation="horizontal")
    cbar.set_label(color_label)
    if color_is_cyclic and np.nanmax(plot_color_values) <= 2 * np.pi + 1e-6:
        cbar.set_ticks([0, np.pi, 2 * np.pi])
        cbar.set_ticklabels(["0", r"$\pi$", r"$2\pi$"])

    fig.suptitle(title_prefix, fontsize=16, y=0.965)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    if export_dir is not None and export_stem is not None:
        save_manifold_analysis_csvs(
            Path(export_dir),
            export_stem,
            coords_bias,
            evals_bias,
            vr_bias,
            coords_full,
            evals_full,
            vr_full,
        )
    plt.show()
    return fig


def plot_phase_conditioned_pairwise_summary(
    *,
    pair_dists,
    phase_compat,
    empirical_update_diff,
    inf_update_diff,
    empirical_full_joint_drop,
    inf_full_joint_drop,
    distance_label,
    title_prefix,
    save_path=None,
):
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.0))

    sc = _plot_phase_conditioned_scatter(
        axes[0, 0],
        pair_dists,
        empirical_update_diff,
        phase_compat,
        "Empirical tangent: full-bias similarity diff",
        distance_label,
        r"$\Delta$ similarity",
    )
    _plot_phase_conditioned_scatter(
        axes[0, 1],
        pair_dists,
        empirical_full_joint_drop,
        phase_compat,
        "Empirical tangent: full joint excess MSE",
        distance_label,
        "Avg joint excess MSE",
    )
    _plot_phase_conditioned_scatter(
        axes[1, 0],
        pair_dists,
        inf_update_diff,
        phase_compat,
        "Infinite width: full-bias similarity diff",
        distance_label,
        r"$\Delta$ similarity",
    )
    _plot_phase_conditioned_scatter(
        axes[1, 1],
        pair_dists,
        inf_full_joint_drop,
        phase_compat,
        "Infinite width: full joint excess MSE",
        distance_label,
        "Avg joint excess MSE",
    )

    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
    cbar.set_label(r"Phase compatibility $\cos(m \Delta \phi)$")

    fig.suptitle(title_prefix, fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


def _exact_or_binned_edges(values, *, max_exact_groups=20, n_bins=12):
    values = np.asarray(values, dtype=float)
    uniq = np.unique(np.round(values, 10))
    if uniq.size <= max_exact_groups:
        centers = np.sort(uniq)
        if centers.size == 1:
            delta = max(1e-6, 0.05 * (abs(float(centers[0])) + 1.0))
            edges = np.asarray([centers[0] - delta, centers[0] + delta], dtype=float)
        else:
            mids = 0.5 * (centers[:-1] + centers[1:])
            left = centers[0] - 0.5 * (centers[1] - centers[0])
            right = centers[-1] + 0.5 * (centers[-1] - centers[-2])
            edges = np.concatenate([[left], mids, [right]])
        return edges, centers, True

    lo = float(np.min(values))
    hi = float(np.max(values))
    if np.isclose(lo, hi):
        pad = max(1e-6, 0.05 * (abs(lo) + 1.0))
        lo -= pad
        hi += pad
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers, False


def _binned_mean_2d(x, y, z, *, x_edges, y_edges):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if not np.any(mask):
        counts = np.zeros((len(x_edges) - 1, len(y_edges) - 1), dtype=float)
        means = np.full_like(counts, np.nan, dtype=float)
        return means, counts

    counts, _, _ = np.histogram2d(x[mask], y[mask], bins=[x_edges, y_edges])
    sums, _, _ = np.histogram2d(x[mask], y[mask], bins=[x_edges, y_edges], weights=z[mask])
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
    means[counts < 1] = np.nan
    return means, counts


def plot_teacher_conditioned_pairwise_summary(
    *,
    pair_dists,
    teacher_corr,
    empirical_bias_similarity,
    empirical_full_similarity,
    inf_bias_similarity,
    inf_full_similarity,
    empirical_bias_joint_drop,
    empirical_full_joint_drop,
    inf_bias_joint_drop,
    inf_full_joint_drop,
    distance_label,
    title_prefix,
    save_path=None,
):
    pair_dists = np.asarray(pair_dists, dtype=float)
    teacher_corr = np.asarray(teacher_corr, dtype=float)

    x_edges, x_centers, used_exact = _exact_or_binned_edges(pair_dists)
    y_edges = np.linspace(-1.0, 1.0, 13)

    fig, axes = plt.subplots(2, 4, figsize=(21.0, 9.8))

    similarity_arrays = [
        np.asarray(empirical_bias_similarity, dtype=float),
        np.asarray(empirical_full_similarity, dtype=float),
        np.asarray(inf_bias_similarity, dtype=float),
        np.asarray(inf_full_similarity, dtype=float),
    ]
    joint_arrays = [
        np.asarray(empirical_bias_joint_drop, dtype=float),
        np.asarray(empirical_full_joint_drop, dtype=float),
        np.asarray(inf_bias_joint_drop, dtype=float),
        np.asarray(inf_full_joint_drop, dtype=float),
    ]

    similarity_norm = Normalize(vmin=-1.0, vmax=1.0)
    joint_all = np.concatenate([arr[np.isfinite(arr)] for arr in joint_arrays if np.any(np.isfinite(arr))])
    if joint_all.size:
        joint_lim = max(abs(float(np.min(joint_all))), abs(float(np.max(joint_all))))
    else:
        joint_lim = 1.0
    joint_norm = TwoSlopeNorm(vmin=-joint_lim, vcenter=0.0, vmax=joint_lim)

    def _heatmap(ax, z_values, *, title, ylabel, cmap, norm):
        means, counts = _binned_mean_2d(pair_dists, teacher_corr, z_values, x_edges=x_edges, y_edges=y_edges)
        means = means.T
        mesh = ax.pcolormesh(x_edges, y_edges, means, shading="auto", cmap=cmap, norm=norm)
        ax.set_title(title)
        ax.set_xlabel(distance_label)
        ax.set_ylabel(ylabel)
        if used_exact and x_centers.size <= 20:
            ax.set_xticks(x_centers)
            if np.max(x_centers) <= 1.000001:
                ax.set_xticklabels(
                    [f"{val:.2f}" if val > 1e-9 else "0" for val in x_centers],
                    rotation=45,
                    ha="right",
                )
        ax.set_ylim(-1.0, 1.0)
        ax.grid(False)
        return mesh

    sim_mesh = _heatmap(
        axes[0, 0],
        similarity_arrays[0],
        title="Empirical RKHS: bias-only correction similarity",
        ylabel="Teacher correlation",
        cmap="coolwarm",
        norm=similarity_norm,
    )
    _heatmap(
        axes[0, 1],
        similarity_arrays[1],
        title="Empirical RKHS: full correction similarity",
        ylabel="Teacher correlation",
        cmap="coolwarm",
        norm=similarity_norm,
    )
    _heatmap(
        axes[0, 2],
        similarity_arrays[2],
        title="Infinite width: bias-only correction similarity",
        ylabel="Teacher correlation",
        cmap="coolwarm",
        norm=similarity_norm,
    )
    _heatmap(
        axes[0, 3],
        similarity_arrays[3],
        title="Infinite width: full correction similarity",
        ylabel="Teacher correlation",
        cmap="coolwarm",
        norm=similarity_norm,
    )
    joint_mesh = _heatmap(
        axes[1, 0],
        joint_arrays[0],
        title="Empirical held-out joint effect: bias-only",
        ylabel="Teacher correlation",
        cmap="RdBu_r",
        norm=joint_norm,
    )
    _heatmap(
        axes[1, 1],
        joint_arrays[1],
        title="Empirical held-out joint effect: full",
        ylabel="Teacher correlation",
        cmap="RdBu_r",
        norm=joint_norm,
    )
    _heatmap(
        axes[1, 2],
        joint_arrays[2],
        title="Infinite-width held-out joint effect: bias-only",
        ylabel="Teacher correlation",
        cmap="RdBu_r",
        norm=joint_norm,
    )
    _heatmap(
        axes[1, 3],
        joint_arrays[3],
        title="Infinite-width held-out joint effect: full",
        ylabel="Teacher correlation",
        cmap="RdBu_r",
        norm=joint_norm,
    )

    cbar_top = fig.colorbar(sim_mesh, ax=axes[0, :].tolist(), shrink=0.96, pad=0.02)
    cbar_top.set_label("Mean correction cosine similarity")
    cbar_bot = fig.colorbar(joint_mesh, ax=axes[1, :].tolist(), shrink=0.96, pad=0.02)
    cbar_bot.set_label("Mean held-out joint excess MSE")

    fig.suptitle(title_prefix, fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


def _hist_bins_from_values(*arrays, num_bins=28):
    arrays = [np.asarray(arr, dtype=float).ravel() for arr in arrays if arr is not None]
    arrays = [arr[np.isfinite(arr)] for arr in arrays if arr.size]
    if not arrays:
        return np.linspace(-0.5, 0.5, 11)
    values = np.concatenate(arrays)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if np.isclose(lo, hi):
        pad = max(1e-6, 0.05 * (abs(lo) + 1.0))
        lo -= pad
        hi += pad
    return np.linspace(lo, hi, num_bins + 1)


def _plot_distribution_overlay(ax, bias_values, full_values, title, xlabel):
    bias_values = np.asarray(bias_values, dtype=float)
    full_values = np.asarray(full_values, dtype=float)
    bins = _hist_bins_from_values(bias_values, full_values)

    ax.hist(
        bias_values,
        bins=bins,
        density=True,
        alpha=0.45,
        color="tab:blue",
        label="Bias-only",
        edgecolor="none",
    )
    ax.hist(
        full_values,
        bins=bins,
        density=True,
        alpha=0.45,
        color="tab:orange",
        label="Full",
        edgecolor="none",
    )
    ax.axvline(float(np.mean(bias_values)), color="tab:blue", linestyle="--", linewidth=1.6)
    ax.axvline(float(np.mean(full_values)), color="tab:orange", linestyle="--", linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=9)


def _plot_difference_hist(ax, diff_values, title, xlabel, *, stats=None):
    diff_values = np.asarray(diff_values, dtype=float)
    bins = _hist_bins_from_values(diff_values)

    ax.hist(
        diff_values,
        bins=bins,
        density=True,
        color="0.35",
        alpha=0.8,
        edgecolor="none",
    )
    ax.axvline(0.0, color="black", linestyle=":", linewidth=1.5)
    ax.axvline(float(np.mean(diff_values)), color="crimson", linestyle="--", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.22)

    if stats is not None:
        lines = [
            f"mean={stats['mean_diff']:.4f}",
            f"95% CI=[{stats['ci_low']:.4f}, {stats['ci_high']:.4f}]",
            f"perm-p={stats['perm_pvalue']:.3g}",
            f"neg-frac={stats['negative_frac']:.3f}",
        ]
        ax.text(
            0.03,
            0.97,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8", alpha=0.92),
        )


def plot_joint_excess_distribution_summary(
    *,
    empirical_bias,
    empirical_full,
    inf_bias,
    inf_full,
    title_prefix,
    seed=0,
    save_path=None,
):
    emp_bias_vals = np.asarray(empirical_bias["pair_drop_mean"], dtype=float)
    emp_full_vals = np.asarray(empirical_full["pair_drop_mean"], dtype=float)
    inf_bias_vals = np.asarray(inf_bias["pair_drop_mean"], dtype=float)
    inf_full_vals = np.asarray(inf_full["pair_drop_mean"], dtype=float)

    emp_stats = paired_mean_difference_test(emp_bias_vals, emp_full_vals, seed=seed)
    inf_stats = paired_mean_difference_test(inf_bias_vals, inf_full_vals, seed=seed)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.0))

    _plot_distribution_overlay(
        axes[0, 0],
        emp_bias_vals,
        emp_full_vals,
        "Empirical tangent: pairwise avg joint excess MSE",
        "Avg joint excess MSE",
    )
    _plot_difference_hist(
        axes[0, 1],
        emp_full_vals - emp_bias_vals,
        "Empirical tangent: full - bias pairwise excess",
        "Full - bias avg joint excess MSE",
        stats=emp_stats,
    )
    _plot_distribution_overlay(
        axes[1, 0],
        inf_bias_vals,
        inf_full_vals,
        "Infinite width: pairwise avg joint excess MSE",
        "Avg joint excess MSE",
    )
    _plot_difference_hist(
        axes[1, 1],
        inf_full_vals - inf_bias_vals,
        "Infinite width: full - bias pairwise excess",
        "Full - bias avg joint excess MSE",
        stats=inf_stats,
    )

    fig.suptitle(title_prefix, fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


def plot_pairwise_joint_interference_summary(
    *,
    empirical_bias,
    empirical_full,
    inf_bias,
    inf_full,
    title_prefix,
    distance_label="Task angular distance (rad)",
    save_path=None,
):
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.0))

    use_pi_distance = "Task angular distance" in distance_label
    plot_distance_label = r"Task angular distance $/ \pi$" if use_pi_distance else distance_label
    emp_bias_dists = np.asarray(empirical_bias["dists"], dtype=float) / np.pi if use_pi_distance else empirical_bias["dists"]
    emp_full_dists = np.asarray(empirical_full["dists"], dtype=float) / np.pi if use_pi_distance else empirical_full["dists"]
    inf_bias_dists = np.asarray(inf_bias["dists"], dtype=float) / np.pi if use_pi_distance else inf_bias["dists"]
    inf_full_dists = np.asarray(inf_full["dists"], dtype=float) / np.pi if use_pi_distance else inf_full["dists"]

    _plot_interference_distance_summary(
        axes[0, 0],
        emp_bias_dists,
        empirical_bias["pair_drop_mean"],
        emp_full_dists,
        empirical_full["pair_drop_mean"],
        "Empirical tangent: joint excess error vs distance",
        plot_distance_label,
        "Avg joint excess MSE",
    )
    _plot_interference_scatter(
        axes[0, 1],
        empirical_bias["pair_similarity"],
        empirical_bias["pair_drop_mean"],
        empirical_full["pair_similarity"],
        empirical_full["pair_drop_mean"],
        "Empirical tangent: joint excess error vs update similarity",
        "Single-task update cosine similarity",
        "Avg joint excess MSE",
        show_fit=False,
    )
    _plot_interference_distance_summary(
        axes[1, 0],
        inf_bias_dists,
        inf_bias["pair_drop_mean"],
        inf_full_dists,
        inf_full["pair_drop_mean"],
        "Infinite width: joint excess error vs distance",
        plot_distance_label,
        "Avg joint excess MSE",
    )
    _plot_interference_scatter(
        axes[1, 1],
        inf_bias["pair_similarity"],
        inf_bias["pair_drop_mean"],
        inf_full["pair_similarity"],
        inf_full["pair_drop_mean"],
        "Infinite width: joint excess error vs correction similarity",
        "Single-task correction cosine similarity",
        "Avg joint excess MSE",
        show_fit=False,
    )

    fig.suptitle(title_prefix, fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


def _directionalize_pairwise_result(result):
    dists = np.asarray(result["dists"], dtype=float)
    pair_similarity = np.asarray(result["pair_similarity"], dtype=float)
    return {
        "dists": np.concatenate([dists, dists]),
        "pair_similarity": np.concatenate([pair_similarity, pair_similarity]),
        "drop": np.concatenate(
            [
                np.asarray(result["drop_i"], dtype=float),
                np.asarray(result["drop_j"], dtype=float),
            ]
        ),
    }


def plot_directional_joint_interference_summary(
    *,
    empirical_bias,
    empirical_full,
    inf_bias,
    inf_full,
    title_prefix,
    distance_label="Task angular distance (rad)",
    save_path=None,
):
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.0))

    use_pi_distance = "Task angular distance" in distance_label
    plot_distance_label = r"Task angular distance $/ \pi$" if use_pi_distance else distance_label

    emp_bias_dir = _directionalize_pairwise_result(empirical_bias)
    emp_full_dir = _directionalize_pairwise_result(empirical_full)
    inf_bias_dir = _directionalize_pairwise_result(inf_bias)
    inf_full_dir = _directionalize_pairwise_result(inf_full)

    if use_pi_distance:
        emp_bias_dir["dists"] = emp_bias_dir["dists"] / np.pi
        emp_full_dir["dists"] = emp_full_dir["dists"] / np.pi
        inf_bias_dir["dists"] = inf_bias_dir["dists"] / np.pi
        inf_full_dir["dists"] = inf_full_dir["dists"] / np.pi

    _plot_interference_distance_summary(
        axes[0, 0],
        emp_bias_dir["dists"],
        emp_bias_dir["drop"],
        emp_full_dir["dists"],
        emp_full_dir["drop"],
        "Empirical tangent: task-specific excess error vs distance",
        plot_distance_label,
        r"Task-specific $MSE^{joint} - MSE^{solo}$",
    )
    _plot_interference_scatter(
        axes[0, 1],
        emp_bias_dir["pair_similarity"],
        emp_bias_dir["drop"],
        emp_full_dir["pair_similarity"],
        emp_full_dir["drop"],
        "Empirical tangent: task-specific excess error vs update similarity",
        "Single-task update cosine similarity",
        r"Task-specific $MSE^{joint} - MSE^{solo}$",
        show_fit=False,
    )
    _plot_interference_distance_summary(
        axes[1, 0],
        inf_bias_dir["dists"],
        inf_bias_dir["drop"],
        inf_full_dir["dists"],
        inf_full_dir["drop"],
        "Infinite width: task-specific excess error vs distance",
        plot_distance_label,
        r"Task-specific $MSE^{joint} - MSE^{solo}$",
    )
    _plot_interference_scatter(
        axes[1, 1],
        inf_bias_dir["pair_similarity"],
        inf_bias_dir["drop"],
        inf_full_dir["pair_similarity"],
        inf_full_dir["drop"],
        "Infinite width: task-specific excess error vs correction similarity",
        "Single-task correction cosine similarity",
        r"Task-specific $MSE^{joint} - MSE^{solo}$",
        show_fit=False,
    )

    fig.suptitle(title_prefix, fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


def record_joint_interference_summaries(
    *,
    summary_rows,
    fit_rows,
    paired_test_rows,
    title_prefix,
    analysis_prefix,
    similarity_label,
    bias_result,
    full_result,
    seed,
):
    print_comparison_summary(
        f"\n{title_prefix}",
        "Bias-only",
        bias_result["pair_drop_mean"],
        "Full",
        full_result["pair_drop_mean"],
    )
    append_summary_row(summary_rows, analysis_prefix, "bias_only", bias_result["pair_drop_mean"])
    append_summary_row(summary_rows, analysis_prefix, "full", full_result["pair_drop_mean"])

    slope, intercept, r2 = print_distance_fit_summary(
        f"{title_prefix} vs distance (bias)",
        bias_result["dists"],
        bias_result["pair_drop_mean"],
    )
    fit_rows.append({"analysis": f"{analysis_prefix}_vs_distance", "group": "bias_only", "slope": slope, "intercept": intercept, "r2": r2})
    slope, intercept, r2 = print_distance_fit_summary(
        f"{title_prefix} vs distance (full)",
        full_result["dists"],
        full_result["pair_drop_mean"],
    )
    fit_rows.append({"analysis": f"{analysis_prefix}_vs_distance", "group": "full", "slope": slope, "intercept": intercept, "r2": r2})
    slope, intercept, r2 = print_distance_fit_summary(
        f"{title_prefix} vs {similarity_label} (bias)",
        bias_result["pair_similarity"],
        bias_result["pair_drop_mean"],
    )
    fit_rows.append({"analysis": f"{analysis_prefix}_vs_similarity", "group": "bias_only", "slope": slope, "intercept": intercept, "r2": r2})
    slope, intercept, r2 = print_distance_fit_summary(
        f"{title_prefix} vs {similarity_label} (full)",
        full_result["pair_similarity"],
        full_result["pair_drop_mean"],
    )
    fit_rows.append({"analysis": f"{analysis_prefix}_vs_similarity", "group": "full", "slope": slope, "intercept": intercept, "r2": r2})

    mean_test = print_paired_mean_test_summary(
        f"{title_prefix} paired mean test (full vs bias)",
        bias_result["pair_drop_mean"],
        full_result["pair_drop_mean"],
        seed=seed,
    )
    paired_test_rows.append(
        {
            "analysis": f"{analysis_prefix}_mean_test",
            "group": "full_minus_bias",
            **mean_test,
        }
    )


def infer_task_family_from_results_dir(results_dir):
    name = Path(results_dir).name
    if name.startswith("single_axis_"):
        return "single_axis"
    if name.startswith("sampled_"):
        return "sampled"
    raise ValueError(
        f"Could not infer task family from results dir name '{name}'. "
        "Expected it to start with 'single_axis_' or 'sampled_'."
    )


def parse_run_metadata_from_name(name):
    patterns = {
        "seed": r"_seed(-?\d+)_",
        "K_tasks": r"_K(\d+)_",
        "m": r"_m(\d+)_",
        "pts": r"_pts(\d+)_",
        "dim": r"_dim(\d+)_",
        "target_phase_tag": r"_tphase([mp0-9]+)$",
    }
    out = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, name)
        if not match:
            continue
        token = match.group(1)
        if key == "target_phase_tag":
            sign = -1.0 if token.startswith("m") else 1.0
            core = token[1:] if token.startswith("m") else token
            out["target_phase"] = sign * float(core.replace("p", "."))
        else:
            out[key] = int(token)
    return out


def load_npz_result_group(npz_data, prefix):
    out = {}
    for key in npz_data.files:
        if key.startswith(prefix + "__"):
            out[key.split("__", 1)[1]] = np.asarray(npz_data[key])
    return out if out else None


def replot_saved_results(results_dir, *, figures_dir=None):
    results_dir = Path(results_dir)
    if figures_dir is None:
        figures_dir = DEFAULT_FIGURES_ROOT / results_dir.name
    else:
        figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    task_family = infer_task_family_from_results_dir(results_dir)
    run_meta = parse_run_metadata_from_name(results_dir.name)
    plot_cfg = task_family_plot_config(task_family)
    metadata = load_task_metadata(results_dir / "task_metadata.csv")
    task_m = metadata.get("m") if metadata.get("m") is not None else run_meta.get("m")
    color_values = metadata["color_values"]
    normals = metadata["normals"] if plot_cfg["distance_mode"] == "normals" else None

    if task_m is not None:
        plot_task_family_geometry_overview(
            normals=metadata["normals"],
            phases=metadata["phases"],
            target_phase=metadata["target_phase"],
            target_mode=metadata["target_mode"],
            global_field_name=metadata["global_field_name"],
            m=int(task_m),
            task_family=task_family,
            color_values=color_values,
            title_prefix="Task family geometry overview",
            save_path=figures_dir / "task_family_geometry_overview.png",
        )

        plot_paper_intro_summary_figure(
            task_family=task_family,
            normals=metadata["normals"],
            phases=metadata["phases"],
            target_phase=metadata["target_phase"],
            target_mode=metadata["target_mode"],
            global_field_name=metadata["global_field_name"],
            m=int(task_m),
            color_values=color_values,
            finite_bias_sim=load_matrix_csv(results_dir / "empirical_update_similarity_bias.csv"),
            finite_full_sim=load_matrix_csv(results_dir / "empirical_update_similarity_full.csv"),
            finite_rkhs_bias_sim=load_matrix_csv(results_dir / "empirical_rkhs_similarity_bias.csv"),
            finite_rkhs_full_sim=load_matrix_csv(results_dir / "empirical_rkhs_similarity_full.csv"),
            infinite_bias_sim=load_matrix_csv(results_dir / "infinite_width_rkhs_similarity_bias.csv"),
            infinite_full_sim=load_matrix_csv(results_dir / "infinite_width_rkhs_similarity_full.csv"),
            distance_label=plot_cfg["distance_label"],
            title_prefix="Kernel symmetry shapes task-update geometry",
            finite_precomputed_bias=load_manifold_analysis_csvs(results_dir, "empirical_task_update_manifold")["bias"],
            finite_precomputed_full=load_manifold_analysis_csvs(results_dir, "empirical_task_update_manifold")["full"],
            save_path=figures_dir / "paper_intro_summary_figure.png",
            seed=run_meta.get("seed", 0),
        )

    emp_update_manifold = load_manifold_analysis_csvs(results_dir, "empirical_task_update_manifold")
    emp_rkhs_manifold = load_manifold_analysis_csvs(results_dir, "empirical_task_rkhs_correction_manifold")
    inf_rkhs_manifold = load_manifold_analysis_csvs(results_dir, "infinite_width_task_correction_manifold")

    DB_sim_bias = load_matrix_csv(results_dir / "empirical_update_similarity_bias.csv")
    DB_sim_full = load_matrix_csv(results_dir / "empirical_update_similarity_full.csv")
    RKHS_sim_bias = load_matrix_csv(results_dir / "empirical_rkhs_similarity_bias.csv")
    RKHS_sim_full = load_matrix_csv(results_dir / "empirical_rkhs_similarity_full.csv")
    G_inf_bias = load_matrix_csv(results_dir / "infinite_width_rkhs_similarity_bias.csv")
    G_inf_full = load_matrix_csv(results_dir / "infinite_width_rkhs_similarity_full.csv")

    plot_task_manifold_summary(
        angles=color_values,
        sim_bias=DB_sim_bias,
        sim_full=DB_sim_full,
        manifold_kind="vectors",
        title_prefix="Empirical finite-width tangent updates",
        normals=normals,
        vector_norms_bias=metadata["delta_norm_bias"],
        vector_norms_full=metadata["delta_norm_full"],
        color_label=plot_cfg["color_label"],
        color_is_cyclic=plot_cfg["color_is_cyclic"],
        align_manifold_to_colors=plot_cfg["align_manifold_to_colors"],
        connect_manifold=plot_cfg["connect_manifold"],
        manifold_cmap=plot_cfg["manifold_cmap"],
        distance_label=plot_cfg["distance_label"],
        export_dir=None,
        export_stem=None,
        save_path=figures_dir / "empirical_task_update_manifold_summary.png",
        precomputed_bias=emp_update_manifold["bias"],
        precomputed_full=emp_update_manifold["full"],
    )

    plot_task_manifold_summary(
        angles=color_values,
        sim_bias=RKHS_sim_bias,
        sim_full=RKHS_sim_full,
        manifold_kind="gram",
        title_prefix="Empirical finite-width RKHS correction manifold",
        normals=normals,
        color_label=plot_cfg["color_label"],
        color_is_cyclic=plot_cfg["color_is_cyclic"],
        align_manifold_to_colors=plot_cfg["align_manifold_to_colors"],
        connect_manifold=plot_cfg["connect_manifold"],
        manifold_cmap=plot_cfg["manifold_cmap"],
        distance_label=plot_cfg["distance_label"],
        export_dir=None,
        export_stem=None,
        save_path=figures_dir / "empirical_task_rkhs_correction_manifold_summary.png",
        precomputed_bias=emp_rkhs_manifold["bias"],
        precomputed_full=emp_rkhs_manifold["full"],
    )

    plot_task_manifold_summary(
        angles=color_values,
        sim_bias=G_inf_bias,
        sim_full=G_inf_full,
        manifold_kind="gram",
        title_prefix="Infinite-width RKHS task-correction geometry",
        normals=normals,
        color_label=plot_cfg["color_label"],
        color_is_cyclic=plot_cfg["color_is_cyclic"],
        align_manifold_to_colors=plot_cfg["align_manifold_to_colors"],
        connect_manifold=plot_cfg["connect_manifold"],
        manifold_cmap=plot_cfg["manifold_cmap"],
        distance_label=plot_cfg["distance_label"],
        export_dir=None,
        export_stem=None,
        save_path=figures_dir / "infinite_width_task_correction_manifold_summary.png",
        precomputed_bias=inf_rkhs_manifold["bias"],
        precomputed_full=inf_rkhs_manifold["full"],
    )

    npz_path = results_dir / "pairwise_joint_multitask_interference_results.npz"
    if npz_path.exists():
        npz_data = np.load(npz_path)
        split_name_to_label = {"train": "Train", "testA": "Test A", "testB": "Test B"}
        for split_name, split_label in split_name_to_label.items():
            emp_bias = load_npz_result_group(npz_data, f"{split_name}_empirical_bias")
            emp_full = load_npz_result_group(npz_data, f"{split_name}_empirical_full")
            inf_bias = load_npz_result_group(npz_data, f"{split_name}_inf_bias")
            inf_full = load_npz_result_group(npz_data, f"{split_name}_inf_full")
            if any(group is None for group in [emp_bias, emp_full, inf_bias, inf_full]):
                continue

            plot_pairwise_joint_interference_summary(
                empirical_bias=emp_bias,
                empirical_full=emp_full,
                inf_bias=inf_bias,
                inf_full=inf_full,
                title_prefix=f"Pairwise multitask interference from joint tangent/kernel solves ({split_label})",
                distance_label=plot_cfg["distance_label"],
                save_path=figures_dir / f"pairwise_joint_multitask_interference_{split_name}_summary.png",
            )
            plot_directional_joint_interference_summary(
                empirical_bias=emp_bias,
                empirical_full=emp_full,
                inf_bias=inf_bias,
                inf_full=inf_full,
                title_prefix=f"Directional pairwise multitask interaction ({split_label})",
                distance_label=plot_cfg["distance_label"],
                save_path=figures_dir / f"pairwise_directional_joint_interference_{split_name}_summary.png",
            )
            plot_joint_excess_distribution_summary(
                empirical_bias=emp_bias,
                empirical_full=emp_full,
                inf_bias=inf_bias,
                inf_full=inf_full,
                title_prefix=f"Marginal pairwise joint excess MSE distributions ({split_label})",
                seed=0,
                save_path=figures_dir / f"joint_excess_distribution_{split_name}_summary.png",
            )

        train_emp_full = load_npz_result_group(npz_data, "train_empirical_full")
        train_inf_full = load_npz_result_group(npz_data, "train_inf_full")
        if train_emp_full is not None and train_inf_full is not None:
            plot_phase_conditioned_pairwise_summary(
                pair_dists=np.asarray(npz_data["pair_dists"], dtype=float),
                phase_compat=np.asarray(npz_data["phase_compat"], dtype=float),
                empirical_update_diff=np.asarray(npz_data["db_sim_full_offdiag"], dtype=float) - np.asarray(npz_data["db_sim_bias_offdiag"], dtype=float),
                inf_update_diff=np.asarray(npz_data["g_inf_full_offdiag"], dtype=float) - np.asarray(npz_data["g_inf_bias_offdiag"], dtype=float),
                empirical_full_joint_drop=train_emp_full["pair_drop_mean"],
                inf_full_joint_drop=train_inf_full["pair_drop_mean"],
                distance_label=plot_cfg["distance_label"],
                title_prefix="Phase-conditioned pairwise structure",
                save_path=figures_dir / "phase_conditioned_pairwise_summary.png",
            )
        if metadata["target_mode"] == "global_field":
            if plot_cfg["distance_mode"] == "angles":
                pair_dists_teacher, pairs = pairwise_distances_from_angles(color_values)
                pair_dists_teacher = pair_dists_teacher / np.pi
                teacher_distance_label = r"Task angular distance $/ \pi$"
            else:
                pair_dists_teacher, pairs = pairwise_distances_from_normals(metadata["normals"])
                teacher_distance_label = plot_cfg["distance_label"]

            teacher_corr = pairwise_teacher_correlation(
                normals=metadata["normals"],
                phases=metadata["phases"],
                m=int(task_m),
                pairs=pairs,
                target_phase=metadata["target_phase"],
                target_mode=metadata["target_mode"],
                global_field_name=metadata["global_field_name"],
            )
            if all(load_npz_result_group(npz_data, f"testA_{name}") is not None for name in ["empirical_bias", "empirical_full", "inf_bias", "inf_full"]) and all(
                load_npz_result_group(npz_data, f"testB_{name}") is not None for name in ["empirical_bias", "empirical_full", "inf_bias", "inf_full"]
            ):
                testA_emp_bias = load_npz_result_group(npz_data, "testA_empirical_bias")
                testA_emp_full = load_npz_result_group(npz_data, "testA_empirical_full")
                testA_inf_bias = load_npz_result_group(npz_data, "testA_inf_bias")
                testA_inf_full = load_npz_result_group(npz_data, "testA_inf_full")
                testB_emp_bias = load_npz_result_group(npz_data, "testB_empirical_bias")
                testB_emp_full = load_npz_result_group(npz_data, "testB_empirical_full")
                testB_inf_bias = load_npz_result_group(npz_data, "testB_inf_bias")
                testB_inf_full = load_npz_result_group(npz_data, "testB_inf_full")
                plot_teacher_conditioned_pairwise_summary(
                    pair_dists=pair_dists_teacher,
                    teacher_corr=teacher_corr,
                    empirical_bias_similarity=upper_tri_values(RKHS_sim_bias, pairs),
                    empirical_full_similarity=upper_tri_values(RKHS_sim_full, pairs),
                    inf_bias_similarity=upper_tri_values(G_inf_bias, pairs),
                    inf_full_similarity=upper_tri_values(G_inf_full, pairs),
                    empirical_bias_joint_drop=0.5 * (
                        np.asarray(testA_emp_bias["pair_drop_mean"], dtype=float)
                        + np.asarray(testB_emp_bias["pair_drop_mean"], dtype=float)
                    ),
                    empirical_full_joint_drop=0.5 * (
                        np.asarray(testA_emp_full["pair_drop_mean"], dtype=float)
                        + np.asarray(testB_emp_full["pair_drop_mean"], dtype=float)
                    ),
                    inf_bias_joint_drop=0.5 * (
                        np.asarray(testA_inf_bias["pair_drop_mean"], dtype=float)
                        + np.asarray(testB_inf_bias["pair_drop_mean"], dtype=float)
                    ),
                    inf_full_joint_drop=0.5 * (
                        np.asarray(testA_inf_full["pair_drop_mean"], dtype=float)
                        + np.asarray(testB_inf_full["pair_drop_mean"], dtype=float)
                    ),
                    distance_label=teacher_distance_label,
                    title_prefix="Teacher-conditioned pairwise structure",
                    save_path=figures_dir / "teacher_conditioned_pairwise_summary.png",
                )

    kernel_decomp_path = results_dir / "kernel_difference_decomposition_metrics.csv"
    if kernel_decomp_path.exists():
        grouped = load_kernel_decomposition_csv(kernel_decomp_path)
        if grouped.get("empirical") and grouped.get("infinite_width"):
            plot_kernel_difference_decomposition_summary(
                empirical_rows=grouped["empirical"],
                infinite_rows=grouped["infinite_width"],
                distance_label=plot_cfg["distance_label"],
                title_prefix="Kernel-block decomposition of full minus bias coupling",
                save_path=figures_dir / "kernel_block_difference_decomposition_summary.png",
            )

    print(f"Replotted figures to: {figures_dir}")
    return {"results_dir": results_dir, "figures_dir": figures_dir}


# Suite for analysis
def run_full_ntk_analysis_update(
        dim=8192, 
        K_tasks=36, 
        m=3, 
        bias_std=0.0, 
        pts=300, 
        test_pts=None,
        task_plot_id=0,
        seed=0,
        task_family="sampled",
        normals_mode="fibonacci",
        random_phase=True,
        base_normal=(0.0, 0.0, 1.0),
        rotation_axis=(1.0, 0.0, 0.0),
        target_phase=0.0,
        target_mode="local_harmonic",
        global_field_name="xz",
        reg=1e-5,
        ridge_mode="trace",
        ridge_tuning_path=None,
        ridge_single_task_T=1,
        ridge_pairwise_T=2,
        analyze_pairwise_joint=True,
        results_dir=None,
        figures_dir=None,
        make_plots=True):
    
    #analytic infinite width kernel set up:
    W_std = 1.0
    b_std = 1.0
    kappa_params = KernelParams(sigma_w2=W_std**2, sigma_b2=b_std**2, sigma_v2=1.0, sigma_bout2=1.0)
    kappa_scale = 1.0  # keep 1.0 unless you decide to calibrate
    
    key = random.PRNGKey(seed)
    params0 = init_mlp_params(key, 3, dim, bias_std=bias_std)

    task_data = sample_task_family_data(
        K_tasks=K_tasks,
        pts_per_circle=pts,
        test_pts_per_circle=test_pts,
        m=m,
        seed=seed,
        task_family=task_family,
        normals_mode=normals_mode,
        random_phase=random_phase,
        base_normal=base_normal,
        rotation_axis=rotation_axis,
        target_phase=target_phase,
        target_mode=target_mode,
        global_field_name=global_field_name,
    )
    task_Xs = task_data["Xs"]
    task_ys = task_data["ys"]
    task_testA_Xs = task_data["testA_Xs"]
    task_testA_ys = task_data["testA_ys"]
    task_testB_Xs = task_data["testB_Xs"]
    task_testB_ys = task_data["testB_ys"]
    task_normals = np.asarray(task_data["normals"], dtype=float)
    task_frame_v1 = np.asarray(task_data["frame_v1"], dtype=float)
    task_frame_v2 = np.asarray(task_data["frame_v2"], dtype=float)
    task_rotations = np.asarray(task_data["task_rotations"], dtype=float)
    task_phases = np.asarray(task_data["phases"], dtype=float)
    task_color_values = np.asarray(task_data["color_values"], dtype=float)
    results_dir = ensure_results_dir(
        results_dir,
        task_family=task_family,
        seed=seed,
        K_tasks=K_tasks,
        m=m,
        pts=pts,
        dim=dim,
        target_phase=target_phase,
        target_mode=target_mode,
        global_field_name=global_field_name,
        base_normal=base_normal,
        rotation_axis=rotation_axis,
    )
    figures_dir = ensure_figures_dir(
        figures_dir,
        task_family=task_family,
        seed=seed,
        K_tasks=K_tasks,
        m=m,
        pts=pts,
        dim=dim,
        target_phase=target_phase,
        target_mode=target_mode,
        global_field_name=global_field_name,
        base_normal=base_normal,
        rotation_axis=rotation_axis,
    )
    plot_normals = None if task_data["distance_mode"] == "angles" else task_normals

    if task_data["distance_mode"] == "angles":
        pair_dists, pairs = pairwise_distances_from_angles(task_color_values)
    else:
        pair_dists, pairs = pairwise_distances_from_normals(task_normals)
    teacher_corr = pairwise_teacher_correlation(
        normals=task_normals,
        phases=task_phases,
        m=m,
        pairs=pairs,
        target_phase=target_phase,
        target_mode=target_mode,
        global_field_name=global_field_name,
    )

    effective_random_phase = random_phase if (task_family == "sampled" and target_mode == "local_harmonic") else False
    print(
        f"Task family: {task_family}, seed={seed}, "
        f"distance={task_data['distance_label']}, "
        f"random_phase={effective_random_phase}, "
        f"target_mode={target_mode}, "
        f"global_field={global_field_name if target_mode == 'global_field' else 'n/a'}, "
        f"target_phase={target_phase:.6f}"
    )
    if task_family == "sampled":
        print(f"Normals mode: {normals_mode}")
    if task_family == "single_axis":
        print(
            "Single-axis geometry: "
            f"base_normal={np.asarray(base_normal, dtype=float)}, "
            f"rotation_axis={np.asarray(rotation_axis, dtype=float)}"
        )
    print(f"Ridge mode: {ridge_mode}")
    if ridge_tuning_path is not None and ridge_mode != "trace":
        print("Warning: ridge_tuning_path was provided, but tuned scales were selected under trace normalization.")
    print(f"Results dir: {results_dir}")
    print(f"Figures dir: {figures_dir}")
    if make_plots and plt is None:
        print("Matplotlib is unavailable; continuing without plots.")
        make_plots = False

    if ridge_tuning_path is not None:
        effective_pairwise_T = ridge_pairwise_T if analyze_pairwise_joint else ridge_single_task_T
        ridge_scales = load_ridge_tuning_scales(
            ridge_tuning_path,
            single_task_T=ridge_single_task_T,
            pairwise_task_T=effective_pairwise_T,
        )
        solo_reg_emp_bias = ridge_scales["single_emp_bias"]
        solo_reg_emp_full = ridge_scales["single_emp_full"]
        solo_reg_inf_bias = ridge_scales["single_inf_bias"]
        solo_reg_inf_full = ridge_scales["single_inf_full"]
        pair_reg_emp_bias = ridge_scales["pair_emp_bias"]
        pair_reg_emp_full = ridge_scales["pair_emp_full"]
        pair_reg_inf_bias = ridge_scales["pair_inf_bias"]
        pair_reg_inf_full = ridge_scales["pair_inf_full"]
        if ridge_scales["format"] == "shared_bias_full":
            print(
                "Loaded shared bias/full ridge scales from "
                f"{ridge_scales['path']}: "
                f"solo(T={ridge_scales['single_task_T']}) bias={solo_reg_emp_bias:.6g}, full={solo_reg_emp_full:.6g}; "
                f"pair(T={ridge_scales['pairwise_task_T']}) bias={pair_reg_emp_bias:.6g}, full={pair_reg_emp_full:.6g}. "
                "Reusing the same scales for both empirical and infinite-width solves."
            )
        else:
            print(
                "Loaded tuned ridge scales from "
                f"{ridge_scales['path']}: "
                f"solo(T={ridge_scales['single_task_T']}) "
                f"emp[bias={solo_reg_emp_bias:.6g}, full={solo_reg_emp_full:.6g}] "
                f"inf[bias={solo_reg_inf_bias:.6g}, full={solo_reg_inf_full:.6g}]; "
                f"pair(T={ridge_scales['pairwise_task_T']}) "
                f"emp[bias={pair_reg_emp_bias:.6g}, full={pair_reg_emp_full:.6g}] "
                f"inf[bias={pair_reg_inf_bias:.6g}, full={pair_reg_inf_full:.6g}]"
            )
        if not analyze_pairwise_joint:
            print(
                "Pairwise joint analysis is disabled, so only the solo ridge scales are used in this run."
            )
    else:
        solo_reg_emp_bias = reg
        solo_reg_emp_full = reg
        solo_reg_inf_bias = reg
        solo_reg_inf_full = reg
        pair_reg_emp_bias = reg
        pair_reg_emp_full = reg
        pair_reg_inf_bias = reg
        pair_reg_inf_full = reg
        print(f"Using manual ridge scale reg={reg:.6g} for all solves.")
    
    # Tracking MSE across all tasks
    mse_history = {'lin': [], 'nonlin': []}
    test_results = {}
    Xs = []
    ys = []
    f0s = []
    Jbs = []
    Jfulls = []
    Kbs = []
    Kfulls = []
    Kinf_biases = []
    Kinf_fulls = []
    alphas_bs = []
    alpha_fulls = []
    alphas_inf_bs = []
    alphas_inf_fulls = []
    delta_bs = []
    delta_fulls = []
    normals = []
    solo_preds_bias = []
    solo_preds_full = []
    solo_preds_inf_bias = []
    solo_preds_inf_full = []
    Xs_testA = []
    ys_testA = []
    f0s_testA = []
    Xs_testB = []
    ys_testB = []
    f0s_testB = []
    Kb_testA_self = []
    Kfull_testA_self = []
    Kinf_bias_testA_self = []
    Kinf_full_testA_self = []
    Kb_testB_self = []
    Kfull_testB_self = []
    Kinf_bias_testB_self = []
    Kinf_full_testB_self = []
    Jb_testsA = []
    Jfull_testsA = []
    Jb_testsB = []
    Jfull_testsB = []
    solo_preds_bias_testA = []
    solo_preds_full_testA = []
    solo_preds_inf_bias_testA = []
    solo_preds_inf_full_testA = []
    solo_preds_bias_testB = []
    solo_preds_full_testB = []
    solo_preds_inf_bias_testB = []
    solo_preds_inf_full_testB = []
    
    #perform analysis for infinite width NTK
    kappa_kernel_fn_full = make_kappa_kernel_fn(kappa_params, which="full", kappa_scale=kappa_scale)
    kappa_kernel_fn_bias = make_kappa_kernel_fn(kappa_params, which="bias", kappa_scale=kappa_scale)
    # batch for memory
    if nt is None:
        raise ImportError("neural_tangents not available for nt.batch; install neural_tangents or remove batching.")
    kappa_kernel_fn_full = nt.batch(kappa_kernel_fn_full, device_count=-1)
    kappa_kernel_fn_bias = nt.batch(kappa_kernel_fn_bias, device_count=-1)

    for i, (X_np, y_np, X_testA_np, y_testA_np, X_testB_np, y_testB_np) in enumerate(
        zip(task_Xs, task_ys, task_testA_Xs, task_testA_ys, task_testB_Xs, task_testB_ys)
    ):
        X = jnp.asarray(X_np)
        y = jnp.asarray(y_np)
        X_testA = jnp.asarray(X_testA_np)
        y_testA = jnp.asarray(y_testA_np)
        X_testB = jnp.asarray(X_testB_np)
        y_testB = jnp.asarray(y_testB_np)
        n = task_normals[i]
        phase = task_phases[i]
        f0 = mlp_apply(params0, X)
        f0_testA = mlp_apply(params0, X_testA)
        f0_testB = mlp_apply(params0, X_testB)

        # Get bias terms of the Jacobian
        J_tree = vmap(jacrev(f_single), (None, 0))(params0, X)
        Jb = jnp.concatenate([J_tree["b1"], J_tree["b2"][:, None]], axis=1)
        Jfull = flatten_pytree_jac(J_tree)
        J_tree_testA = vmap(jacrev(f_single), (None, 0))(params0, X_testA)
        Jb_testA = jnp.concatenate([J_tree_testA["b1"], J_tree_testA["b2"][:, None]], axis=1)
        Jfull_testA = flatten_pytree_jac(J_tree_testA)
        J_tree_testB = vmap(jacrev(f_single), (None, 0))(params0, X_testB)
        Jb_testB = jnp.concatenate([J_tree_testB["b1"], J_tree_testB["b2"][:, None]], axis=1)
        Jfull_testB = flatten_pytree_jac(J_tree_testB)

        # Solve for delta_b
        alpha_b, db, Kb = solve_tangent_update(Jb, y, f0, solo_reg_emp_bias, ridge_mode=ridge_mode)
        alpha_full, d_full, K_full = solve_tangent_update(Jfull, y, f0, solo_reg_emp_full, ridge_mode=ridge_mode)

        # Evaluate performance
        f_lin = f0 + Jb @ db
        h_dim = params0["b1"].shape[0]
        params_updated = {**params0, 
                          "b1": params0["b1"] + db[:h_dim], 
                          "b2": params0["b2"] + db[h_dim]}
        f_nonlin = mlp_apply(params_updated, X)

        # Calculate MSEs
        y_centered = y - jnp.mean(y)
        mse_lin = jnp.mean((f_lin - jnp.mean(f_lin) - y_centered)**2)
        mse_nonlin = jnp.mean((f_nonlin - jnp.mean(f_nonlin) - y_centered)**2)
        
        mse_history['lin'].append(mse_lin)
        mse_history['nonlin'].append(mse_nonlin)

        if i == task_plot_id:
            test_results = {
                'phi': np.mod(np.linspace(0, 2 * np.pi, len(y), endpoint=False) + phase, 2 * np.pi),
                'y': y_centered,
                'f_lin': f_lin - jnp.mean(f_lin),
                'f_nonlin': f_nonlin - jnp.mean(f_nonlin),
                'mse_lin': mse_lin,
                'mse_nonlin': mse_nonlin
            }
        
        
        K_full_inf = jnp.array(kappa_kernel_fn_full(X, X, get="ntk"))
        K_bias_inf = jnp.array(kappa_kernel_fn_bias(X, X, get="ntk"))
        Kb_testA = Jb_testA @ Jb.T
        Kfull_testA = Jfull_testA @ Jfull.T
        Kb_testB = Jb_testB @ Jb.T
        Kfull_testB = Jfull_testB @ Jfull.T
        K_bias_inf_testA = jnp.array(kappa_kernel_fn_bias(X_testA, X, get="ntk"))
        K_full_inf_testA = jnp.array(kappa_kernel_fn_full(X_testA, X, get="ntk"))
        K_bias_inf_testB = jnp.array(kappa_kernel_fn_bias(X_testB, X, get="ntk"))
        K_full_inf_testB = jnp.array(kappa_kernel_fn_full(X_testB, X, get="ntk"))
        alphas_bs_inf_width = solve_tangent_update_kernel(K_bias_inf, y, f0, solo_reg_inf_bias, ridge_mode=ridge_mode)
        alphas_full_inf_width = solve_tangent_update_kernel(K_full_inf, y, f0, solo_reg_inf_full, ridge_mode=ridge_mode)
        alphas_inf_bs.append(np.array(alphas_bs_inf_width))
        alphas_inf_fulls.append(np.array(alphas_full_inf_width))

        Xs.append(np.array(X))
        ys.append(np.array(y))
        f0s.append(np.array(f0))
        Xs_testA.append(np.array(X_testA))
        ys_testA.append(np.array(y_testA))
        f0s_testA.append(np.array(f0_testA))
        Xs_testB.append(np.array(X_testB))
        ys_testB.append(np.array(y_testB))
        f0s_testB.append(np.array(f0_testB))
        Jbs.append(np.array(Jb))
        Jfulls.append(np.array(Jfull))
        Jb_testsA.append(np.array(Jb_testA))
        Jfull_testsA.append(np.array(Jfull_testA))
        Jb_testsB.append(np.array(Jb_testB))
        Jfull_testsB.append(np.array(Jfull_testB))
        Kbs.append(np.array(Kb))
        Kfulls.append(np.array(K_full))
        Kinf_biases.append(np.array(K_bias_inf))
        Kinf_fulls.append(np.array(K_full_inf))
        Kb_testA_self.append(np.array(Kb_testA))
        Kfull_testA_self.append(np.array(Kfull_testA))
        Kb_testB_self.append(np.array(Kb_testB))
        Kfull_testB_self.append(np.array(Kfull_testB))
        Kinf_bias_testA_self.append(np.array(K_bias_inf_testA))
        Kinf_full_testA_self.append(np.array(K_full_inf_testA))
        Kinf_bias_testB_self.append(np.array(K_bias_inf_testB))
        Kinf_full_testB_self.append(np.array(K_full_inf_testB))
        alphas_bs.append(np.array(alpha_b))
        delta_bs.append(np.array(db))
        alpha_fulls.append(np.array(alpha_full))
        delta_fulls.append(np.array(d_full))
        normals.append(np.array(n))
        solo_preds_bias.append(np.array(f0 + Kb @ alpha_b))
        solo_preds_full.append(np.array(f0 + K_full @ alpha_full))
        solo_preds_inf_bias.append(np.array(f0 + K_bias_inf @ alphas_bs_inf_width))
        solo_preds_inf_full.append(np.array(f0 + K_full_inf @ alphas_full_inf_width))
        solo_preds_bias_testA.append(np.array(f0_testA + Kb_testA @ alpha_b))
        solo_preds_full_testA.append(np.array(f0_testA + Kfull_testA @ alpha_full))
        solo_preds_inf_bias_testA.append(np.array(f0_testA + K_bias_inf_testA @ alphas_bs_inf_width))
        solo_preds_inf_full_testA.append(np.array(f0_testA + K_full_inf_testA @ alphas_full_inf_width))
        solo_preds_bias_testB.append(np.array(f0_testB + Kb_testB @ alpha_b))
        solo_preds_full_testB.append(np.array(f0_testB + Kfull_testB @ alpha_full))
        solo_preds_inf_bias_testB.append(np.array(f0_testB + K_bias_inf_testB @ alphas_bs_inf_width))
        solo_preds_inf_full_testB.append(np.array(f0_testB + K_full_inf_testB @ alphas_full_inf_width))
        

    # Logging summary
    print(f"Mean Empirical MSE across all tasks: {np.mean(mse_history['nonlin']):.6f}")
    print(f"Mean Linear MSE across all tasks: {np.mean(mse_history['lin']):.6f}")

    # Build delta_b similarity matrix
    DB_matrix = np.stack(delta_bs)
    DB_normed = DB_matrix / (np.linalg.norm(DB_matrix, axis=1, keepdims=True) + 1e-9)
    sim_mat = DB_normed @ DB_normed.T
    
    DB_sim_bias = cosine_matrix(delta_bs)
    DB_sim_full = cosine_matrix(delta_fulls)

    RKHS_sim_bias = empirical_rkhs_task_similarity(Jbs, alphas_bs)
    RKHS_sim_full = empirical_rkhs_task_similarity(Jfulls, alpha_fulls)
    
    G_inf_bias = rkhs_task_similarity_from_kernel(
                                                    Xs,
                                                    alphas_inf_bs,
                                                    kappa_kernel_fn_bias,
                                                )

    G_inf_full = rkhs_task_similarity_from_kernel(
                                                    Xs,
                                                    alphas_inf_fulls,
                                                    kappa_kernel_fn_full,
                                                )

    phase_metrics = pairwise_phase_metrics(task_phases, pairs, m)
    emp_update_bias_offdiag = offdiag_values(DB_sim_bias)
    emp_update_full_offdiag = offdiag_values(DB_sim_full)
    emp_rkhs_bias_offdiag = offdiag_values(RKHS_sim_bias)
    emp_rkhs_full_offdiag = offdiag_values(RKHS_sim_full)
    inf_rkhs_bias_offdiag = offdiag_values(G_inf_bias)
    inf_rkhs_full_offdiag = offdiag_values(G_inf_full)
    empirical_kernel_decomp_rows = pairwise_kernel_difference_decomposition(
        pairs=pairs,
        dists=pair_dists,
        within_kernels_bias=Kbs,
        within_kernels_full=Kfulls,
        alphas_bias=alphas_bs,
        alphas_full=alpha_fulls,
        cross_kernel_bias_fn=lambda i, j: Jbs[i] @ Jbs[j].T,
        cross_kernel_full_fn=lambda i, j: Jfulls[i] @ Jfulls[j].T,
    )
    infinite_kernel_decomp_rows = pairwise_kernel_difference_decomposition(
        pairs=pairs,
        dists=pair_dists,
        within_kernels_bias=Kinf_biases,
        within_kernels_full=Kinf_fulls,
        alphas_bias=alphas_inf_bs,
        alphas_full=alphas_inf_fulls,
        cross_kernel_bias_fn=lambda i, j: np.array(kappa_kernel_fn_bias(Xs[i], Xs[j], get="ntk")),
        cross_kernel_full_fn=lambda i, j: np.array(kappa_kernel_fn_full(Xs[i], Xs[j], get="ntk")),
    )

    delta_b_norms = np.linalg.norm(np.stack(delta_bs), axis=1)
    delta_full_norms = np.linalg.norm(np.stack(delta_fulls), axis=1)
    solo_mse_emp_bias = np.asarray([centered_mse(pred, y) for pred, y in zip(solo_preds_bias, ys)], dtype=float)
    solo_mse_emp_full = np.asarray([centered_mse(pred, y) for pred, y in zip(solo_preds_full, ys)], dtype=float)
    solo_mse_inf_bias = np.asarray([centered_mse(pred, y) for pred, y in zip(solo_preds_inf_bias, ys)], dtype=float)
    solo_mse_inf_full = np.asarray([centered_mse(pred, y) for pred, y in zip(solo_preds_inf_full, ys)], dtype=float)
    solo_mse_emp_bias_testA = np.asarray([centered_mse(pred, y) for pred, y in zip(solo_preds_bias_testA, ys_testA)], dtype=float)
    solo_mse_emp_full_testA = np.asarray([centered_mse(pred, y) for pred, y in zip(solo_preds_full_testA, ys_testA)], dtype=float)
    solo_mse_inf_bias_testA = np.asarray([centered_mse(pred, y) for pred, y in zip(solo_preds_inf_bias_testA, ys_testA)], dtype=float)
    solo_mse_inf_full_testA = np.asarray([centered_mse(pred, y) for pred, y in zip(solo_preds_inf_full_testA, ys_testA)], dtype=float)
    solo_mse_emp_bias_testB = np.asarray([centered_mse(pred, y) for pred, y in zip(solo_preds_bias_testB, ys_testB)], dtype=float)
    solo_mse_emp_full_testB = np.asarray([centered_mse(pred, y) for pred, y in zip(solo_preds_full_testB, ys_testB)], dtype=float)
    solo_mse_inf_bias_testB = np.asarray([centered_mse(pred, y) for pred, y in zip(solo_preds_inf_bias_testB, ys_testB)], dtype=float)
    solo_mse_inf_full_testB = np.asarray([centered_mse(pred, y) for pred, y in zip(solo_preds_inf_full_testB, ys_testB)], dtype=float)

    summary_rows = []
    fit_rows = []
    paired_test_rows = []

    print_comparison_summary(
        "\nEmpirical tangent update norms",
        "Bias-only",
        delta_b_norms,
        "Full",
        delta_full_norms,
    )
    append_summary_row(summary_rows, "empirical_tangent_update_norms", "bias_only", delta_b_norms)
    append_summary_row(summary_rows, "empirical_tangent_update_norms", "full", delta_full_norms)
    print_comparison_summary(
        "\nEmpirical tangent update cosine off-diagonals",
        "Bias-only",
        emp_update_bias_offdiag,
        "Full",
        emp_update_full_offdiag,
    )
    append_summary_row(summary_rows, "empirical_tangent_update_offdiag", "bias_only", emp_update_bias_offdiag)
    append_summary_row(summary_rows, "empirical_tangent_update_offdiag", "full", emp_update_full_offdiag)
    print_difference_summary(
        "Empirical tangent update paired similarity differences",
        emp_update_full_offdiag - emp_update_bias_offdiag,
    )
    append_summary_row(summary_rows, "empirical_tangent_update_diff", "full_minus_bias", emp_update_full_offdiag - emp_update_bias_offdiag)
    slope, intercept, r2 = print_distance_fit_summary(
        "Empirical tangent paired difference vs distance",
        pair_dists,
        emp_update_full_offdiag - emp_update_bias_offdiag,
    )
    fit_rows.append({"analysis": "empirical_tangent_update_diff_vs_distance", "group": "full_minus_bias", "slope": slope, "intercept": intercept, "r2": r2})
    print_comparison_summary(
        "\nEmpirical RKHS correction cosine off-diagonals",
        "Bias-only",
        emp_rkhs_bias_offdiag,
        "Full",
        emp_rkhs_full_offdiag,
    )
    append_summary_row(summary_rows, "empirical_rkhs_offdiag", "bias_only", emp_rkhs_bias_offdiag)
    append_summary_row(summary_rows, "empirical_rkhs_offdiag", "full", emp_rkhs_full_offdiag)
    print_difference_summary(
        "Empirical RKHS paired similarity differences",
        emp_rkhs_full_offdiag - emp_rkhs_bias_offdiag,
    )
    append_summary_row(summary_rows, "empirical_rkhs_diff", "full_minus_bias", emp_rkhs_full_offdiag - emp_rkhs_bias_offdiag)
    slope, intercept, r2 = print_distance_fit_summary(
        "Empirical RKHS paired difference vs distance",
        pair_dists,
        emp_rkhs_full_offdiag - emp_rkhs_bias_offdiag,
    )
    fit_rows.append({"analysis": "empirical_rkhs_diff_vs_distance", "group": "full_minus_bias", "slope": slope, "intercept": intercept, "r2": r2})
    print_comparison_summary(
        "\nInfinite-width RKHS correction cosine off-diagonals",
        "Bias-only",
        inf_rkhs_bias_offdiag,
        "Full",
        inf_rkhs_full_offdiag,
    )
    append_summary_row(summary_rows, "infinite_width_rkhs_offdiag", "bias_only", inf_rkhs_bias_offdiag)
    append_summary_row(summary_rows, "infinite_width_rkhs_offdiag", "full", inf_rkhs_full_offdiag)
    print_difference_summary(
        "Infinite-width RKHS paired similarity differences",
        inf_rkhs_full_offdiag - inf_rkhs_bias_offdiag,
    )
    append_summary_row(summary_rows, "infinite_width_rkhs_diff", "full_minus_bias", inf_rkhs_full_offdiag - inf_rkhs_bias_offdiag)
    slope, intercept, r2 = print_distance_fit_summary(
        "Infinite-width RKHS paired difference vs distance",
        pair_dists,
        inf_rkhs_full_offdiag - inf_rkhs_bias_offdiag,
    )
    fit_rows.append({"analysis": "infinite_width_rkhs_diff_vs_distance", "group": "full_minus_bias", "slope": slope, "intercept": intercept, "r2": r2})

    if analyze_pairwise_joint:
        eval_splits = {
            "train": {
                "eval_ys": ys,
                "eval_f0s": f0s,
                "emp_bias_self": Kbs,
                "emp_full_self": Kfulls,
                "inf_bias_self": Kinf_biases,
                "inf_full_self": Kinf_fulls,
                "solo_emp_bias": solo_preds_bias,
                "solo_emp_full": solo_preds_full,
                "solo_inf_bias": solo_preds_inf_bias,
                "solo_inf_full": solo_preds_inf_full,
                "emp_bias_eval_cross_fn": lambda i, j: Jbs[i] @ Jbs[j].T,
                "emp_full_eval_cross_fn": lambda i, j: Jfulls[i] @ Jfulls[j].T,
                "inf_bias_eval_cross_fn": lambda i, j: np.array(kappa_kernel_fn_bias(Xs[i], Xs[j], get="ntk")),
                "inf_full_eval_cross_fn": lambda i, j: np.array(kappa_kernel_fn_full(Xs[i], Xs[j], get="ntk")),
            },
            "testA": {
                "eval_ys": ys_testA,
                "eval_f0s": f0s_testA,
                "emp_bias_self": Kb_testA_self,
                "emp_full_self": Kfull_testA_self,
                "inf_bias_self": Kinf_bias_testA_self,
                "inf_full_self": Kinf_full_testA_self,
                "solo_emp_bias": solo_preds_bias_testA,
                "solo_emp_full": solo_preds_full_testA,
                "solo_inf_bias": solo_preds_inf_bias_testA,
                "solo_inf_full": solo_preds_inf_full_testA,
                "emp_bias_eval_cross_fn": lambda i, j: Jb_testsA[i] @ Jbs[j].T,
                "emp_full_eval_cross_fn": lambda i, j: Jfull_testsA[i] @ Jfulls[j].T,
                "inf_bias_eval_cross_fn": lambda i, j: np.array(kappa_kernel_fn_bias(Xs_testA[i], Xs[j], get="ntk")),
                "inf_full_eval_cross_fn": lambda i, j: np.array(kappa_kernel_fn_full(Xs_testA[i], Xs[j], get="ntk")),
            },
            "testB": {
                "eval_ys": ys_testB,
                "eval_f0s": f0s_testB,
                "emp_bias_self": Kb_testB_self,
                "emp_full_self": Kfull_testB_self,
                "inf_bias_self": Kinf_bias_testB_self,
                "inf_full_self": Kinf_full_testB_self,
                "solo_emp_bias": solo_preds_bias_testB,
                "solo_emp_full": solo_preds_full_testB,
                "solo_inf_bias": solo_preds_inf_bias_testB,
                "solo_inf_full": solo_preds_inf_full_testB,
                "emp_bias_eval_cross_fn": lambda i, j: Jb_testsB[i] @ Jbs[j].T,
                "emp_full_eval_cross_fn": lambda i, j: Jfull_testsB[i] @ Jfulls[j].T,
                "inf_bias_eval_cross_fn": lambda i, j: np.array(kappa_kernel_fn_bias(Xs_testB[i], Xs[j], get="ntk")),
                "inf_full_eval_cross_fn": lambda i, j: np.array(kappa_kernel_fn_full(Xs_testB[i], Xs[j], get="ntk")),
            },
        }

        joint_results = {}
        for split_name, eval_cfg in eval_splits.items():
            joint_results[split_name] = {
                "emp_bias": pairwise_joint_interference_analysis(
                    pairs=pairs,
                    dists=pair_dists,
                    train_ys=ys,
                    train_f0s=f0s,
                    train_within_kernels=Kbs,
                    eval_ys=eval_cfg["eval_ys"],
                    eval_f0s=eval_cfg["eval_f0s"],
                    eval_self_kernels=eval_cfg["emp_bias_self"],
                    solo_eval_preds=eval_cfg["solo_emp_bias"],
                    similarity_matrix=DB_sim_bias,
                    train_cross_kernel_fn=lambda i, j: Jbs[i] @ Jbs[j].T,
                    eval_cross_kernel_fn=eval_cfg["emp_bias_eval_cross_fn"],
                    reg=pair_reg_emp_bias,
                    ridge_mode=ridge_mode,
                ),
                "emp_full": pairwise_joint_interference_analysis(
                    pairs=pairs,
                    dists=pair_dists,
                    train_ys=ys,
                    train_f0s=f0s,
                    train_within_kernels=Kfulls,
                    eval_ys=eval_cfg["eval_ys"],
                    eval_f0s=eval_cfg["eval_f0s"],
                    eval_self_kernels=eval_cfg["emp_full_self"],
                    solo_eval_preds=eval_cfg["solo_emp_full"],
                    similarity_matrix=DB_sim_full,
                    train_cross_kernel_fn=lambda i, j: Jfulls[i] @ Jfulls[j].T,
                    eval_cross_kernel_fn=eval_cfg["emp_full_eval_cross_fn"],
                    reg=pair_reg_emp_full,
                    ridge_mode=ridge_mode,
                ),
                "inf_bias": pairwise_joint_interference_analysis(
                    pairs=pairs,
                    dists=pair_dists,
                    train_ys=ys,
                    train_f0s=f0s,
                    train_within_kernels=Kinf_biases,
                    eval_ys=eval_cfg["eval_ys"],
                    eval_f0s=eval_cfg["eval_f0s"],
                    eval_self_kernels=eval_cfg["inf_bias_self"],
                    solo_eval_preds=eval_cfg["solo_inf_bias"],
                    similarity_matrix=G_inf_bias,
                    train_cross_kernel_fn=lambda i, j: np.array(kappa_kernel_fn_bias(Xs[i], Xs[j], get="ntk")),
                    eval_cross_kernel_fn=eval_cfg["inf_bias_eval_cross_fn"],
                    reg=pair_reg_inf_bias,
                    ridge_mode=ridge_mode,
                ),
                "inf_full": pairwise_joint_interference_analysis(
                    pairs=pairs,
                    dists=pair_dists,
                    train_ys=ys,
                    train_f0s=f0s,
                    train_within_kernels=Kinf_fulls,
                    eval_ys=eval_cfg["eval_ys"],
                    eval_f0s=eval_cfg["eval_f0s"],
                    eval_self_kernels=eval_cfg["inf_full_self"],
                    solo_eval_preds=eval_cfg["solo_inf_full"],
                    similarity_matrix=G_inf_full,
                    train_cross_kernel_fn=lambda i, j: np.array(kappa_kernel_fn_full(Xs[i], Xs[j], get="ntk")),
                    eval_cross_kernel_fn=eval_cfg["inf_full_eval_cross_fn"],
                    reg=pair_reg_inf_full,
                    ridge_mode=ridge_mode,
                ),
            }

        empirical_joint_bias = joint_results["train"]["emp_bias"]
        empirical_joint_full = joint_results["train"]["emp_full"]
        inf_joint_bias = joint_results["train"]["inf_bias"]
        inf_joint_full = joint_results["train"]["inf_full"]

        split_configs = [
            ("train", "Train", "empirical_tangent_joint_excess_mse_train", "Solo update cosine similarity", "infinite_width_joint_excess_mse_train", "Solo correction cosine similarity"),
            ("testA", "Test A", "empirical_tangent_joint_excess_mse_testA", "Solo update cosine similarity", "infinite_width_joint_excess_mse_testA", "Solo correction cosine similarity"),
            ("testB", "Test B", "empirical_tangent_joint_excess_mse_testB", "Solo update cosine similarity", "infinite_width_joint_excess_mse_testB", "Solo correction cosine similarity"),
        ]
        for split_name, split_label, emp_analysis_prefix, emp_sim_label, inf_analysis_prefix, inf_sim_label in split_configs:
            split_results = joint_results[split_name]
            record_joint_interference_summaries(
                summary_rows=summary_rows,
                fit_rows=fit_rows,
                paired_test_rows=paired_test_rows,
                title_prefix=f"Empirical tangent {split_label} pairwise avg joint excess MSE",
                analysis_prefix=emp_analysis_prefix,
                similarity_label=emp_sim_label,
                bias_result=split_results["emp_bias"],
                full_result=split_results["emp_full"],
                seed=seed,
            )
            record_joint_interference_summaries(
                summary_rows=summary_rows,
                fit_rows=fit_rows,
                paired_test_rows=paired_test_rows,
                title_prefix=f"Infinite-width {split_label} pairwise avg joint excess MSE",
                analysis_prefix=inf_analysis_prefix,
                similarity_label=inf_sim_label,
                bias_result=split_results["inf_bias"],
                full_result=split_results["inf_full"],
                seed=seed,
            )

            if make_plots:
                plot_pairwise_joint_interference_summary(
                    empirical_bias=split_results["emp_bias"],
                    empirical_full=split_results["emp_full"],
                    inf_bias=split_results["inf_bias"],
                    inf_full=split_results["inf_full"],
                    title_prefix=f"Pairwise multitask interference from joint tangent/kernel solves ({split_label})",
                    distance_label=task_data["distance_label"],
                    save_path=figures_dir / f"pairwise_joint_multitask_interference_{split_name}_summary.png",
                )
                plot_directional_joint_interference_summary(
                    empirical_bias=split_results["emp_bias"],
                    empirical_full=split_results["emp_full"],
                    inf_bias=split_results["inf_bias"],
                    inf_full=split_results["inf_full"],
                    title_prefix=f"Directional pairwise multitask interaction ({split_label})",
                    distance_label=task_data["distance_label"],
                    save_path=figures_dir / f"pairwise_directional_joint_interference_{split_name}_summary.png",
                )
                plot_joint_excess_distribution_summary(
                    empirical_bias=split_results["emp_bias"],
                    empirical_full=split_results["emp_full"],
                    inf_bias=split_results["inf_bias"],
                    inf_full=split_results["inf_full"],
                    title_prefix=f"Marginal pairwise joint excess MSE distributions ({split_label})",
                    seed=seed,
                    save_path=figures_dir / f"joint_excess_distribution_{split_name}_summary.png",
                )

        save_pairwise_results_npz(
            results_dir / "pairwise_joint_multitask_interference_results.npz",
            {
                "train_empirical_bias": joint_results["train"]["emp_bias"],
                "train_empirical_full": joint_results["train"]["emp_full"],
                "train_inf_bias": joint_results["train"]["inf_bias"],
                "train_inf_full": joint_results["train"]["inf_full"],
                "testA_empirical_bias": joint_results["testA"]["emp_bias"],
                "testA_empirical_full": joint_results["testA"]["emp_full"],
                "testA_inf_bias": joint_results["testA"]["inf_bias"],
                "testA_inf_full": joint_results["testA"]["inf_full"],
                "testB_empirical_bias": joint_results["testB"]["emp_bias"],
                "testB_empirical_full": joint_results["testB"]["emp_full"],
                "testB_inf_bias": joint_results["testB"]["inf_bias"],
                "testB_inf_full": joint_results["testB"]["inf_full"],
            },
            extras={
                "task_color_values": task_color_values,
                "task_phases": task_phases,
                "target_phase": np.asarray([target_phase], dtype=float),
                "task_normals": task_normals,
                "pair_dists": pair_dists,
                "phase_delta_wrapped": phase_metrics["phase_delta_wrapped"],
                "phase_delta_abs": phase_metrics["phase_delta_abs"],
                "phase_compat": phase_metrics["phase_compat"],
                "teacher_corr": teacher_corr,
                "db_sim_bias_offdiag": emp_update_bias_offdiag,
                "db_sim_full_offdiag": emp_update_full_offdiag,
                "rkhs_sim_bias_offdiag": emp_rkhs_bias_offdiag,
                "rkhs_sim_full_offdiag": emp_rkhs_full_offdiag,
                "g_inf_bias_offdiag": inf_rkhs_bias_offdiag,
                "g_inf_full_offdiag": inf_rkhs_full_offdiag,
            },
        )
        if make_plots:
            plot_phase_conditioned_pairwise_summary(
                pair_dists=pair_dists,
                phase_compat=phase_metrics["phase_compat"],
                empirical_update_diff=emp_update_full_offdiag - emp_update_bias_offdiag,
                inf_update_diff=inf_rkhs_full_offdiag - inf_rkhs_bias_offdiag,
                empirical_full_joint_drop=empirical_joint_full["pair_drop_mean"],
                inf_full_joint_drop=inf_joint_full["pair_drop_mean"],
                distance_label=task_data["distance_label"],
                title_prefix="Phase-conditioned pairwise structure",
                save_path=figures_dir / "phase_conditioned_pairwise_summary.png",
            )
        if target_mode == "global_field" and make_plots:
            if task_data["distance_mode"] == "angles":
                teacher_pair_dists = pair_dists / np.pi
                teacher_distance_label = r"Task angular distance $/ \pi$"
            else:
                teacher_pair_dists = pair_dists
                teacher_distance_label = task_data["distance_label"]
            plot_teacher_conditioned_pairwise_summary(
                pair_dists=teacher_pair_dists,
                teacher_corr=teacher_corr,
                empirical_bias_similarity=emp_rkhs_bias_offdiag,
                empirical_full_similarity=emp_rkhs_full_offdiag,
                inf_bias_similarity=inf_rkhs_bias_offdiag,
                inf_full_similarity=inf_rkhs_full_offdiag,
                empirical_bias_joint_drop=0.5 * (
                    np.asarray(joint_results["testA"]["emp_bias"]["pair_drop_mean"], dtype=float)
                    + np.asarray(joint_results["testB"]["emp_bias"]["pair_drop_mean"], dtype=float)
                ),
                empirical_full_joint_drop=0.5 * (
                    np.asarray(joint_results["testA"]["emp_full"]["pair_drop_mean"], dtype=float)
                    + np.asarray(joint_results["testB"]["emp_full"]["pair_drop_mean"], dtype=float)
                ),
                inf_bias_joint_drop=0.5 * (
                    np.asarray(joint_results["testA"]["inf_bias"]["pair_drop_mean"], dtype=float)
                    + np.asarray(joint_results["testB"]["inf_bias"]["pair_drop_mean"], dtype=float)
                ),
                inf_full_joint_drop=0.5 * (
                    np.asarray(joint_results["testA"]["inf_full"]["pair_drop_mean"], dtype=float)
                    + np.asarray(joint_results["testB"]["inf_full"]["pair_drop_mean"], dtype=float)
                ),
                distance_label=teacher_distance_label,
                title_prefix="Teacher-conditioned pairwise structure",
                save_path=figures_dir / "teacher_conditioned_pairwise_summary.png",
            )
    else:
        empirical_joint_bias = empirical_joint_full = None
        inf_joint_bias = inf_joint_full = None
        joint_results = None

    save_task_metadata_csv(
        results_dir / "task_metadata.csv",
        normals=task_normals,
        phases=task_phases,
        target_phase=target_phase,
        target_mode=target_mode,
        global_field_name=global_field_name,
        m=m,
        task_family=task_family,
        color_values=task_color_values,
        delta_b_norms=delta_b_norms,
        delta_full_norms=delta_full_norms,
        solo_mse_emp_bias=solo_mse_emp_bias,
        solo_mse_emp_full=solo_mse_emp_full,
        solo_mse_inf_bias=solo_mse_inf_bias,
        solo_mse_inf_full=solo_mse_inf_full,
    )

    kernel_decomp_rows = (
        [{"regime": "empirical", **row} for row in empirical_kernel_decomp_rows]
        + [{"regime": "infinite_width", **row} for row in infinite_kernel_decomp_rows]
    )
    save_pairwise_metrics_csv(results_dir / "kernel_difference_decomposition_metrics.csv", kernel_decomp_rows)
    if make_plots:
        plot_kernel_difference_decomposition_summary(
            empirical_rows=empirical_kernel_decomp_rows,
            infinite_rows=infinite_kernel_decomp_rows,
            distance_label=task_data["distance_label"],
            title_prefix="Kernel-block decomposition of full minus bias coupling",
            save_path=figures_dir / "kernel_block_difference_decomposition_summary.png",
        )
        plot_task_family_geometry_overview(
            normals=task_normals,
            phases=task_phases,
            target_phase=target_phase,
            target_mode=target_mode,
            global_field_name=global_field_name,
            m=m,
            task_family=task_family,
            color_values=task_color_values,
            title_prefix="Task family geometry overview",
            save_path=figures_dir / "task_family_geometry_overview.png",
        )
        plot_paper_intro_summary_figure(
            task_family=task_family,
            normals=task_normals,
            phases=task_phases,
            target_phase=target_phase,
            target_mode=target_mode,
            global_field_name=global_field_name,
            m=m,
            color_values=task_color_values,
            finite_bias_sim=DB_sim_bias,
            finite_full_sim=DB_sim_full,
            finite_rkhs_bias_sim=RKHS_sim_bias,
            finite_rkhs_full_sim=RKHS_sim_full,
            infinite_bias_sim=G_inf_bias,
            infinite_full_sim=G_inf_full,
            distance_label=task_data["distance_label"],
            title_prefix="Kernel symmetry shapes task-update geometry",
            finite_manifold_bias=np.stack(delta_bs),
            finite_manifold_full=np.stack(delta_fulls),
            save_path=figures_dir / "paper_intro_summary_figure.png",
            seed=seed,
        )

    pairwise_rows = []
    for idx, (i, j) in enumerate(pairs):
        row = {
            "pair_index": idx,
            "task_i": i,
            "task_j": j,
            "distance": float(pair_dists[idx]),
            "phase_delta_wrapped": float(phase_metrics["phase_delta_wrapped"][idx]),
            "phase_delta_abs": float(phase_metrics["phase_delta_abs"][idx]),
            "phase_compat": float(phase_metrics["phase_compat"][idx]),
            "teacher_corr": float(teacher_corr[idx]),
            "emp_update_bias": float(emp_update_bias_offdiag[idx]),
            "emp_update_full": float(emp_update_full_offdiag[idx]),
            "emp_update_full_minus_bias": float(emp_update_full_offdiag[idx] - emp_update_bias_offdiag[idx]),
            "emp_rkhs_bias": float(emp_rkhs_bias_offdiag[idx]),
            "emp_rkhs_full": float(emp_rkhs_full_offdiag[idx]),
            "emp_rkhs_full_minus_bias": float(emp_rkhs_full_offdiag[idx] - emp_rkhs_bias_offdiag[idx]),
            "inf_rkhs_bias": float(inf_rkhs_bias_offdiag[idx]),
            "inf_rkhs_full": float(inf_rkhs_full_offdiag[idx]),
            "inf_rkhs_full_minus_bias": float(inf_rkhs_full_offdiag[idx] - inf_rkhs_bias_offdiag[idx]),
        }
        if analyze_pairwise_joint:
            row.update(
                {
                    "emp_joint_drop_mean_bias": float(empirical_joint_bias["pair_drop_mean"][idx]),
                    "emp_joint_drop_mean_full": float(empirical_joint_full["pair_drop_mean"][idx]),
                    "emp_joint_drop_max_bias": float(empirical_joint_bias["pair_drop_max"][idx]),
                    "emp_joint_drop_max_full": float(empirical_joint_full["pair_drop_max"][idx]),
                    "emp_joint_drop_i_bias": float(empirical_joint_bias["drop_i"][idx]),
                    "emp_joint_drop_j_bias": float(empirical_joint_bias["drop_j"][idx]),
                    "emp_joint_drop_i_full": float(empirical_joint_full["drop_i"][idx]),
                    "emp_joint_drop_j_full": float(empirical_joint_full["drop_j"][idx]),
                    "inf_joint_drop_mean_bias": float(inf_joint_bias["pair_drop_mean"][idx]),
                    "inf_joint_drop_mean_full": float(inf_joint_full["pair_drop_mean"][idx]),
                    "inf_joint_drop_max_bias": float(inf_joint_bias["pair_drop_max"][idx]),
                    "inf_joint_drop_max_full": float(inf_joint_full["pair_drop_max"][idx]),
                    "inf_joint_drop_i_bias": float(inf_joint_bias["drop_i"][idx]),
                    "inf_joint_drop_j_bias": float(inf_joint_bias["drop_j"][idx]),
                    "inf_joint_drop_i_full": float(inf_joint_full["drop_i"][idx]),
                    "inf_joint_drop_j_full": float(inf_joint_full["drop_j"][idx]),
                    "emp_joint_drop_mean_bias_testA": float(joint_results["testA"]["emp_bias"]["pair_drop_mean"][idx]),
                    "emp_joint_drop_mean_full_testA": float(joint_results["testA"]["emp_full"]["pair_drop_mean"][idx]),
                    "emp_joint_drop_mean_bias_testB": float(joint_results["testB"]["emp_bias"]["pair_drop_mean"][idx]),
                    "emp_joint_drop_mean_full_testB": float(joint_results["testB"]["emp_full"]["pair_drop_mean"][idx]),
                    "inf_joint_drop_mean_bias_testA": float(joint_results["testA"]["inf_bias"]["pair_drop_mean"][idx]),
                    "inf_joint_drop_mean_full_testA": float(joint_results["testA"]["inf_full"]["pair_drop_mean"][idx]),
                    "inf_joint_drop_mean_bias_testB": float(joint_results["testB"]["inf_bias"]["pair_drop_mean"][idx]),
                    "inf_joint_drop_mean_full_testB": float(joint_results["testB"]["inf_full"]["pair_drop_mean"][idx]),
                }
            )
        pairwise_rows.append(row)

    preserve_joint_csvs = not analyze_pairwise_joint
    save_csv_preserving_existing(
        results_dir / "pairwise_metrics.csv",
        save_pairwise_metrics_csv,
        pairwise_rows,
        preserve_existing=preserve_joint_csvs,
        label="pairwise metrics CSV",
    )
    save_csv_preserving_existing(
        results_dir / "summary_stats.csv",
        save_summary_stats_csv,
        summary_rows,
        preserve_existing=preserve_joint_csvs,
        label="summary stats CSV",
    )
    save_csv_preserving_existing(
        results_dir / "fit_summaries.csv",
        save_summary_stats_csv,
        fit_rows,
        preserve_existing=preserve_joint_csvs,
        label="fit summaries CSV",
    )
    save_csv_preserving_existing(
        results_dir / "paired_mean_tests.csv",
        save_summary_stats_csv,
        paired_test_rows,
        preserve_existing=preserve_joint_csvs,
        label="paired mean tests CSV",
    )

    save_matrix_csv(results_dir / "empirical_update_similarity_bias.csv", DB_sim_bias)
    save_matrix_csv(results_dir / "empirical_update_similarity_full.csv", DB_sim_full)
    save_matrix_csv(results_dir / "empirical_rkhs_similarity_bias.csv", RKHS_sim_bias)
    save_matrix_csv(results_dir / "empirical_rkhs_similarity_full.csv", RKHS_sim_full)
    save_matrix_csv(results_dir / "infinite_width_rkhs_similarity_bias.csv", G_inf_bias)
    save_matrix_csv(results_dir / "infinite_width_rkhs_similarity_full.csv", G_inf_full)
    save_task_update_vectors_npz(
        results_dir / "task_update_vectors.npz",
        normals=task_normals,
        phases=task_phases,
        color_values=task_color_values,
        frame_v1=task_frame_v1,
        frame_v2=task_frame_v2,
        task_rotations=task_rotations,
        delta_bias=np.stack(delta_bs),
        delta_full=np.stack(delta_fulls),
        target_phase=target_phase,
        target_mode=target_mode,
        global_field_name=global_field_name,
        m=m,
        task_family=task_family,
        base_normal=base_normal,
        rotation_axis=rotation_axis,
    )

    # ------------------------------------------------------------------
    # Requested organized figures:
    #   1) finite empirical update geometry
    #   2) infinite-width RKHS correction geometry
    # ------------------------------------------------------------------
    if make_plots:
        plot_task_manifold_summary(
            angles=task_color_values,
            sim_bias=DB_sim_bias,
            sim_full=DB_sim_full,
            manifold_bias=np.stack(delta_bs),
            manifold_full=np.stack(delta_fulls),
            manifold_kind="vectors",
            title_prefix="Empirical finite-width tangent updates",
            normals=plot_normals,
            vector_norms_bias=delta_b_norms,
            vector_norms_full=delta_full_norms,
            color_label=task_data["color_label"],
            color_is_cyclic=task_data["color_is_cyclic"],
            align_manifold_to_colors=task_data["align_manifold_to_colors"],
            connect_manifold=task_data["connect_manifold"],
            manifold_cmap=task_data["manifold_cmap"],
            distance_label=task_data["distance_label"],
            export_dir=results_dir,
            export_stem="empirical_task_update_manifold",
            save_path=figures_dir / "empirical_task_update_manifold_summary.png",
        )

        plot_task_manifold_summary(
            angles=task_color_values,
            sim_bias=RKHS_sim_bias,
            sim_full=RKHS_sim_full,
            manifold_bias=RKHS_sim_bias,
            manifold_full=RKHS_sim_full,
            manifold_kind="gram",
            title_prefix="Empirical finite-width RKHS correction manifold",
            normals=plot_normals,
            color_label=task_data["color_label"],
            color_is_cyclic=task_data["color_is_cyclic"],
            align_manifold_to_colors=task_data["align_manifold_to_colors"],
            connect_manifold=task_data["connect_manifold"],
            manifold_cmap=task_data["manifold_cmap"],
            distance_label=task_data["distance_label"],
            export_dir=results_dir,
            export_stem="empirical_task_rkhs_correction_manifold",
            save_path=figures_dir / "empirical_task_rkhs_correction_manifold_summary.png",
        )

        plot_task_manifold_summary(
            angles=task_color_values,
            sim_bias=G_inf_bias,
            sim_full=G_inf_full,
            manifold_bias=G_inf_bias,
            manifold_full=G_inf_full,
            manifold_kind="gram",
            title_prefix="Infinite-width RKHS task-correction geometry",
            normals=plot_normals,
            color_label=task_data["color_label"],
            color_is_cyclic=task_data["color_is_cyclic"],
            align_manifold_to_colors=task_data["align_manifold_to_colors"],
            connect_manifold=task_data["connect_manifold"],
            manifold_cmap=task_data["manifold_cmap"],
            distance_label=task_data["distance_label"],
            export_dir=results_dir,
            export_stem="infinite_width_task_correction_manifold",
            save_path=figures_dir / "infinite_width_task_correction_manifold_summary.png",
        )

    return {
        "results_dir": results_dir,
        "figures_dir": figures_dir,
        "task_data": task_data,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Analyze task manifolds for finite-width tangent updates and infinite-width kernels.",
    )
    parser.add_argument("--dim", type=int, default=16384, help="Hidden width of the MLP.")
    parser.add_argument("--K-tasks", type=int, default=30, help="Number of tasks to sample.")
    parser.add_argument("--m", type=int, default=3, help="Spherical harmonic frequency on each circle.")
    parser.add_argument("--bias-std", type=float, default=0.0, help="Initialization std for hidden/output biases.")
    parser.add_argument("--pts", type=int, default=300, help="Points sampled per task circle.")
    parser.add_argument("--test-pts", type=int, default=None, help="Held-out test points per task circle. Defaults to 2 * pts.")
    parser.add_argument("--task-plot-id", type=int, default=0, help="Task index used for the stored single-task diagnostic slice.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--task-family",
        choices=["sampled", "single_axis"],
        default="sampled",
        help="Task family to analyze.",
    )
    parser.add_argument(
        "--normals-mode",
        choices=["fibonacci", "random"],
        default="fibonacci",
        help="How to sample normals for sampled task families.",
    )
    parser.add_argument(
        "--random-phase",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use random task phases in sampled mode.",
    )
    parser.add_argument(
        "--base-normal",
        type=parse_vector_arg,
        default=(0.0, 0.0, 1.0),
        help="Base great-circle normal as a comma-separated vector, used in single-axis mode.",
    )
    parser.add_argument(
        "--rotation-axis",
        type=parse_vector_arg,
        default=(1.0, 0.0, 0.0),
        help="Rotation axis as a comma-separated vector, used in single-axis mode.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["local_harmonic", "global_field"],
        default="local_harmonic",
        help="Whether labels come from the local circle harmonic or from a shared global field on the sphere.",
    )
    parser.add_argument(
        "--global-field",
        choices=["xz", "x2_minus_y2", "z2_legendre", "yz"],
        default="xz",
        help="Shared global sphere field used when --target-mode global_field.",
    )
    parser.add_argument(
        "--target-phase",
        type=parse_angle_arg,
        default=0.0,
        help=(
            "Additional label-phase offset in y = sin(m * phi + target_phase). "
            "Only used when --target-mode local_harmonic; use pi/2 for the even-parity control experiment."
        ),
    )
    parser.add_argument(
        "--reg",
        type=float,
        default=1e-5,
        help="Manual ridge scale used when --ridge-tuning-path is not provided.",
    )
    parser.add_argument(
        "--ridge-mode",
        choices=["trace", "max_eig"],
        default="trace",
        help="How to scale ridge regularization. Default matches the trace-normalized multitask pipeline.",
    )
    parser.add_argument(
        "--ridge-tuning-path",
        type=Path,
        default=None,
        help=(
            "Optional NPZ file with tuned ridge scales. Supports either legacy shared bias/full tuning "
            "or separate empirical/infinite-width tuning. Uses T=1 for solo solves and T=2 for pairwise solves by default."
        ),
    )
    parser.add_argument(
        "--ridge-single-task-T",
        type=int,
        default=1,
        help="T index to use from the ridge tuning NPZ for single-task/update-geometry solves.",
    )
    parser.add_argument(
        "--ridge-pairwise-T",
        type=int,
        default=2,
        help="T index to use from the ridge tuning NPZ for pairwise joint-interference solves.",
    )
    parser.add_argument(
        "--analyze-pairwise-joint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run pairwise joint multitask solves and downstream diagnostics.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Directory for CSV/NPZ outputs. Defaults to "
            f"{DEFAULT_RESULTS_ROOT}/<run_name>."
        ),
    )
    parser.add_argument(
        "--load-results-dir",
        type=Path,
        default=None,
        help="Replot figures from an existing saved results directory without recomputing the analysis pipeline.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help=(
            "Directory for saved figure PNGs. Defaults to "
            f"{DEFAULT_FIGURES_ROOT}/<run_name>."
        ),
    )
    parser.add_argument(
        "--make-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to generate and save matplotlib figures.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.load_results_dir is not None:
        replot_saved_results(args.load_results_dir, figures_dir=args.figures_dir)
        return
    run_full_ntk_analysis_update(
        dim=args.dim,
        K_tasks=args.K_tasks,
        m=args.m,
        bias_std=args.bias_std,
        pts=args.pts,
        test_pts=args.test_pts,
        task_plot_id=args.task_plot_id,
        seed=args.seed,
        task_family=args.task_family,
        normals_mode=args.normals_mode,
        random_phase=args.random_phase,
        base_normal=args.base_normal,
        rotation_axis=args.rotation_axis,
        target_mode=args.target_mode,
        global_field_name=args.global_field,
        target_phase=args.target_phase,
        reg=args.reg,
        ridge_mode=args.ridge_mode,
        ridge_tuning_path=args.ridge_tuning_path,
        ridge_single_task_T=args.ridge_single_task_T,
        ridge_pairwise_T=args.ridge_pairwise_T,
        analyze_pairwise_joint=args.analyze_pairwise_joint,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        make_plots=args.make_plots,
    )


if __name__ == "__main__":
    main()
