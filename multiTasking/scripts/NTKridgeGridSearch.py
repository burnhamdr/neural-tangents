import os
import json
from tqdm import tqdm
import numpy as np
import jax.numpy as jnp
import jax
from scipy.stats import wilcoxon

try:
    import neural_tangents as nt
    from neural_tangents import stax
except Exception:
    nt = None
    stax = None

# ----------------------------
# Task sampling (same as yours)
# ----------------------------
def sample_random_normals_s2(K: int, rng: np.random.Generator) -> np.ndarray:
    n = rng.normal(size=(K, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return n

def normals_fibonacci_s2(K: int) -> np.ndarray:
    ga = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(K, dtype=float)
    z = 1.0 - 2.0*(i + 0.5)/K
    r = np.sqrt(np.maximum(0.0, 1.0 - z*z))
    phi = ga * i
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.stack([x, y, z], axis=1)

def generate_single_circle(normal, pts, m, phase_offset=0.0):
    n = normal / np.linalg.norm(normal)
    v1 = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
    v1 = v1 - n * np.dot(n, v1)
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(n, v1)

    phi = np.linspace(0, 2*np.pi, pts, endpoint=False) + phase_offset
    X = np.outer(np.cos(phi), v1) + np.outer(np.sin(phi), v2)
    y = np.sin(m * phi)
    return X, y

def generate_multi_tasks_from_normals_phases(
    normals: np.ndarray,
    phases: np.ndarray,
    pts_per_circle: int,
    m: int,
):
    X_list, y_list, task_slices = [], [], []
    start = 0
    for j in range(normals.shape[0]):
        X, y = generate_single_circle(normals[j], pts_per_circle, m, phase_offset=float(phases[j]))
        X_list.append(X)
        y_list.append(y)
        task_slices.append(slice(start, start + pts_per_circle))
        start += pts_per_circle
    return np.vstack(X_list), np.concatenate(y_list), task_slices


# ----------------------------
# Closed-form spherical kernels
# ----------------------------
class KernelParams:
    def __init__(self, sigma_w2=1.0, sigma_b2=1.0, sigma_v2=1.0):
        self.sigma_w2 = float(sigma_w2)
        self.sigma_b2 = float(sigma_b2)
        self.sigma_v2 = float(sigma_v2)

def _relu_kappa_components_sphere(t: jnp.ndarray, par: KernelParams):
    q = par.sigma_w2 + par.sigma_b2
    c = par.sigma_w2 * t + par.sigma_b2
    rho = jnp.clip(c / q, -1.0, 1.0)
    theta = jnp.arccos(rho)
    kappa1 = (jnp.pi - theta) / (2.0 * jnp.pi)
    kappa0 = (q / (2.0 * jnp.pi)) * (jnp.sin(theta) + (jnp.pi - theta) * jnp.cos(theta))
    return kappa0, kappa1

def gram_from_kappa(Xa: jnp.ndarray, Xb: jnp.ndarray, par: KernelParams, which: str, kappa_scale: float = 1.0):
    t = jnp.clip(Xa @ Xb.T, -1.0, 1.0)
    k0, k1 = _relu_kappa_components_sphere(t, par)
    if which == "full":
        K = k0 + par.sigma_v2 * (t + 1.0) * k1
    elif which == "bias":
        K = par.sigma_v2 * k1
    else:
        raise ValueError("which must be 'full' or 'bias'")
    return K * float(kappa_scale)

def make_kappa_kernel_fn(par: KernelParams, which: str = "full", kappa_scale: float = 1.0):
    """
    Returns a function kernel_fn(X1, X2, get=...) suitable for nt.batch.
    """
    def kernel_fn(X1, X2, get="ntk", **kwargs):
        K = gram_from_kappa(X1, X2, par=par, which=which, kappa_scale=kappa_scale)
        if get in ("ntk", "nngp"):
            return K
        if isinstance(get, (tuple, list)):
            return tuple(K for _ in get)
        raise ValueError(f"Unsupported get={get}")
    return kernel_fn


# ----------------------------
# neural-tangents backend
# ----------------------------
def gram_from_neural_tangents(Xa: jnp.ndarray, Xb: jnp.ndarray, which: str, W_std=1.0, b_std=1.0):
    if nt is None:
        raise ImportError("neural_tangents not available.")
    if which == "full":
        trainable = ("W", "b")
    elif which == "bias":
        trainable = ("b",)
    else:
        raise ValueError("which must be 'full' or 'bias'")

    _, _, kernel_fn = stax.serial(
        stax.Dense(1, W_std=W_std, b_std=b_std, parameterization="ntk", trainable=trainable),
        stax.Relu(),
        stax.Dense(1, W_std=W_std, b_std=b_std, parameterization="ntk", trainable=trainable),
    )
    kernel_fn = nt.batch(kernel_fn, device_count=-1)
    return jnp.array(kernel_fn(Xa, Xb, "ntk"))

def estimate_lambda_max(matvec, N, n_iter=20, key=0):
    rng = np.random.default_rng(key)
    v = jnp.array(rng.normal(size=(N,)), dtype=jnp.float32)
    v = v / jnp.linalg.norm(v)

    for _ in range(n_iter):
        Kv = matvec(v)
        v = Kv / jnp.linalg.norm(Kv)

    return float(jnp.dot(v, matvec(v)))

def top_eig_power_iteration(K: jnp.ndarray,
                            n_iter: int = 50,
                            tol: float = 1e-6,
                            seed: int = 0):
    """
    Estimate largest eigenvalue of symmetric PSD K via power iteration.
    Returns: (lambda_max_est, v)
    """
    assert K.ndim == 2 and K.shape[0] == K.shape[1]
    n = K.shape[0]

    key = jax.random.PRNGKey(seed)
    v = jax.random.normal(key, (n,), dtype=K.dtype)
    v = v / jnp.linalg.norm(v)

    def body(state, _):
        v, lam_old = state
        w = K @ v
        v_new = w / jnp.linalg.norm(w)
        lam_new = jnp.vdot(v_new, K @ v_new).real  # Rayleigh quotient
        return (v_new, lam_new), lam_new

    # initialize lam_old = 0
    (v_fin, lam_fin), lams = jax.lax.scan(body, (v, jnp.array(0., K.dtype)), None, length=n_iter)

    # Optional: crude convergence check using last few Rayleigh quotients
    # (keep it simple; you can also stop early with a while_loop)
    return lam_fin, v_fin


# ----------------------------
# Unified runner with BOTH switches:
#   - backend: "nt" vs "kappa"
#   - generalization: "within" vs "across"
# ----------------------------
def _circle_basis_from_normal(normal: np.ndarray):
    n = normal / np.linalg.norm(normal)
    v1 = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
    v1 = v1 - n * np.dot(n, v1)
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(n, v1)
    return n, v1, v2

def generate_single_circle_with_phi(normal: np.ndarray, phi: np.ndarray, m: int):
    """
    normal: (3,)
    phi: (pts,)
    returns X: (pts,3), y: (pts,)
    """
    _, v1, v2 = _circle_basis_from_normal(normal)
    X = np.outer(np.cos(phi), v1) + np.outer(np.sin(phi), v2)
    y = np.sin(m * phi)
    return X, y

def generate_multi_tasks_from_normals_phases_with_mode(
    normals: np.ndarray,
    phases: np.ndarray,
    pts_per_circle: int,
    m: int,
    rng: np.random.Generator,
    mode: str = "grid",   # "grid" or "random"
):
    X_list, y_list, task_slices = [], [], []
    start = 0
    for j in range(normals.shape[0]):
        phase = float(phases[j])
        if mode == "grid":
            phi = np.linspace(0, 2*np.pi, pts_per_circle, endpoint=False) + phase
        elif mode == "random":
            phi = rng.uniform(0.0, 2*np.pi, size=pts_per_circle) + phase
        else:
            raise ValueError("mode must be 'grid' or 'random'")

        X, y = generate_single_circle_with_phi(normals[j], phi, m)
        X_list.append(X)
        y_list.append(y)
        task_slices.append(slice(start, start + pts_per_circle))
        start += pts_per_circle
    return np.vstack(X_list), np.concatenate(y_list), task_slices

def run_kernel_multi_great_circle(
    K_tasks: int,
    pts_per_circle: int,
    m: int,
    seed: int,
    reg_scale: float = 1e-5,
    normals_mode: str = "random",
    random_phase: bool = True,
    generalization: str = "within",
    backend: str = "kappa",
    which: str = "full",
    W_std: float = 1.0,
    b_std: float = 1.0,
    kappa_params=None,
    kappa_scale: float = 1.0,
    test_pts_per_circle: int | None = None,
    # NEW: noise controls
    snr: float | None = None,          # e.g. 1, 2, 5, 10, 20, ...
    noise_std: float | None = None,    # overrides snr if provided
    noise_on: str = "both",            # "train", "test", "both"
):
    """
    Returns per-task MSEs on TWO independent test replicates (A,B),
    plus seed-level aggregates for baseline repeatability and model-baseline gap.

    Baseline repeatability is estimated by comparing baseline MSE on test A vs test B.
    Model-baseline gap is |MSE_model - MSE_base| / MSE_base on replicate A (and also B).
    """
    rng = np.random.default_rng(seed)
    if test_pts_per_circle is None:
        test_pts_per_circle = 2 * pts_per_circle

    def sample_tasks(rng_local):
        if normals_mode == "random":
            normals = sample_random_normals_s2(K_tasks, rng=rng_local)
        elif normals_mode == "fibonacci":
            normals = normals_fibonacci_s2(K_tasks)
        else:
            raise ValueError("normals_mode must be 'random' or 'fibonacci'")
        phases = rng_local.uniform(0, 2*np.pi, size=K_tasks) if random_phase else np.zeros(K_tasks)
        return normals, phases

    normals_tr, phases_tr = sample_tasks(rng)

    if generalization == "within":
        normals_te, phases_te = normals_tr, phases_tr
    elif generalization == "across":
        normals_te, phases_te = sample_tasks(rng)
    else:
        raise ValueError("generalization must be 'within' or 'across'")

    # ----------------------------
    # Train set: grid sampling
    # ----------------------------
    X_train_np, y_train_np, task_slices_train = generate_multi_tasks_from_normals_phases_with_mode(
        normals_tr, phases_tr, pts_per_circle=pts_per_circle, m=m, rng=rng, mode="grid"
    )

    # ----------------------------
    # Test replicate A/B: independent random sampling of phi
    # ----------------------------
    rngA = np.random.default_rng(seed + 10_000_001)

    X_testA_np, y_testA_np, task_slices_testA = generate_multi_tasks_from_normals_phases_with_mode(
        normals_te, phases_te, pts_per_circle=test_pts_per_circle, m=m, rng=rngA, mode="random"
    )
    
    # ----------------------------
    # Optional: additive Gaussian label noise (SNR control)
    # ----------------------------
    # Signal variance for sin(m phi) is ~ 1/2 (exact for uniform phi).
    # We estimate it empirically on the *noise-free* training labels.
    if noise_std is None and snr is not None:
        sig_var = float(np.var(y_train_np))  # empirical
        noise_var = sig_var / float(snr)
        noise_std = float(np.sqrt(max(noise_var, 0.0)))

    if noise_std is not None and noise_std > 0:
        rngN = np.random.default_rng(seed + 20_000_000)

        if noise_on in ("train", "both"):
            y_train_np = y_train_np + rngN.normal(0.0, noise_std, size=y_train_np.shape)

        if noise_on in ("test", "both"):
            # independent noise stream for test
            rngNt = np.random.default_rng(seed + 30_000_000)
            y_testA_np = y_testA_np + rngNt.normal(0.0, noise_std, size=y_testA_np.shape)

    X_train = jnp.array(X_train_np, dtype=jnp.float32)
    y_train = jnp.array(y_train_np, dtype=jnp.float32).reshape(-1, 1)

    X_testA = jnp.array(X_testA_np, dtype=jnp.float32)
    y_testA = jnp.array(y_testA_np, dtype=jnp.float32).reshape(-1, 1)

    # ----------------------------
    # Kernels
    # ----------------------------
    if backend == "nt":
        K_train = gram_from_neural_tangents(X_train, X_train, which=which, W_std=W_std, b_std=b_std)
        K_A_train = gram_from_neural_tangents(X_testA, X_train, which=which, W_std=W_std, b_std=b_std)

    elif backend == "kappa":
        if kappa_params is None:
            kappa_params = KernelParams(sigma_w2=W_std**2, sigma_b2=b_std**2, sigma_v2=1.0)

        kappa_kernel_fn = make_kappa_kernel_fn(kappa_params, which=which, kappa_scale=kappa_scale)
        # batch for memory
        if nt is None:
            raise ImportError("neural_tangents not available for nt.batch; install neural_tangents or remove batching.")
        kappa_kernel_fn = nt.batch(kappa_kernel_fn, device_count=-1)

        K_train = jnp.array(kappa_kernel_fn(X_train, X_train, get="ntk"))
        K_A_train = jnp.array(kappa_kernel_fn(X_testA, X_train, get="ntk"))
    else:
        raise ValueError("backend must be 'nt' or 'kappa'")

    # ----------------------------
    # Ridge solve with trace normalization (your existing convention)
    # ----------------------------
    # lam_max, _ = top_eig_power_iteration(jnp.asarray(K_train), n_iter=60, seed=0)
    n = K_train.shape[0]
    s_trace = jnp.trace(K_train) / n
    K_tilde = K_train / s_trace
    lam = reg_scale
    # solve (K + eps*I) alpha = y_train
    K_reg = K_tilde + (n*lam) * jnp.eye(n, dtype=K_train.dtype)
    alpha = jnp.linalg.solve(K_reg, y_train)

    # Predictions on A/B
    y_predA = K_A_train @ alpha

    y_predA_np = np.asarray(y_predA).reshape(-1)
    y_testA_np_flat = np.asarray(y_testA).reshape(-1)

    # ----------------------------
    # Per-task MSEs and baselines on both replicates
    # ----------------------------
    mse_model_A, mse_base_A = [], []
    mse_model_B, mse_base_B = [], []

    for sl_tr, sl_A in zip(task_slices_train, task_slices_testA):
        # baseline constant fitted on TRAIN for that task
        c = float(np.mean(y_train_np[sl_tr]))
        # replicate A
        ytA = y_testA_np_flat[sl_A]
        ypA = y_predA_np[sl_A]
        mse_model_A.append(float(np.mean((ytA - ypA) ** 2)))
        mse_base_A.append(float(np.mean((ytA - c) ** 2)))

    mse_model_A = np.array(mse_model_A)
    mse_base_A  = np.array(mse_base_A)

    # Seed-level aggregates (grand mean across tasks)
    B_A = float(np.mean(mse_base_A))
    M_A = float(np.mean(mse_model_A))


    return dict(
        mse_model_A=mse_model_A,
        mse_base_A=mse_base_A,
        # metadata
        generalization=generalization,
        backend=backend,
        which=which,
    )


# ============================================================
# Optional: one-shot calibration of kappa_scale to match NT
# ============================================================

def calibrate_kappa_scale(
    X: np.ndarray,
    backend_nt_kwargs: dict,
    kappa_params: KernelParams,
    which="full",
    eps=1e-12,
):
    """
    Finds a scalar s such that s*K_kappa ≈ K_nt in least squares sense.
    Useful because NTK parameterization conventions can differ by a constant factor.

    Returns: scale s (float)
    """
    Xj = jnp.array(X, dtype=jnp.float32)

    # NT Gram
    K_nt = np.asarray(gram_from_neural_tangents(Xj, Xj, **backend_nt_kwargs, kind="ntk"))

    # kappa Gram
    K_k = np.asarray(gram_from_kappa(Xj, Xj, kappa_params, which=which))

    # Solve for scalar s minimizing ||s K_k - K_nt||_F^2
    num = float(np.sum(K_k * K_nt))
    den = float(np.sum(K_k * K_k)) + eps
    return num / den

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


# ----------------------------
# Stats helpers (from earlier)
# ----------------------------
def paired_signflip_pvalue_one_sided(deltas: np.ndarray, n_perm: int = 5000, seed: int = 0) -> float:
    """
    Paired sign-flip test on deltas = (baseline - model).
    H0: E[delta] <= 0   vs   H1: E[delta] > 0
    """
    rng = np.random.default_rng(seed)
    deltas = np.asarray(deltas, dtype=float).reshape(-1)
    obs = float(np.mean(deltas))

    signs = rng.choice([-1.0, 1.0], size=(n_perm, deltas.size))
    perm_means = (signs * deltas[None, :]).mean(axis=1)

    p = (np.sum(perm_means >= obs) + 1.0) / (n_perm + 1.0)
    return float(p)

def wilcoxon_paired_improvement(deltas, alternative="greater"):
    """
    deltas: 1D array of paired differences (baseline - model).
    H1 'greater' means model improves: deltas > 0.
    """
    deltas = np.asarray(deltas, dtype=float).reshape(-1)

    # Wilcoxon cannot use all-zero input; handle gracefully.
    if np.allclose(deltas, 0.0):
        return 1.0, 0.0

    # zero_method='wilcox' drops exact zeros (common choice)
    stat, p = wilcoxon(deltas, alternative=alternative, zero_method="wilcox", method="auto")
    return float(p), float(stat)

def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    m = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    q_out = np.empty_like(q)
    q_out[order] = q
    return q_out


# ----------------------------
# Updated main sweep + plotting
# ----------------------------
def make_effective_capacity_panels(
    T_list,
    n_seeds: int = 20,
    pts_per_circle: int = 512,
    test_pts_per_circle: int | None = None,
    m: int = 4,
    normals_mode: str = "random",
    random_phase: bool = True,
    W_std: float = 1.0,
    b_std: float = 1.0,
    reg_scale: dict = {"full_ntk": 1e-5, "bias_only": 1e-5},
    backend: str = "kappa",
    generalization: str = "within",
    kappa_params=None,
    kappa_scale: float = 1.0,
    show_seed_progress: bool = False,
    plot: bool = True,
    # NEW: noise controls
    snr: float | None = None,
    noise_std: float | None = None,
    noise_on: str = "both",
    # NEW: stats controls
    alpha: float = 0.05,
    n_perm: int = 5000,
    bh_correct: bool = True,
):
    kernels = {
        "full_ntk": dict(which="full"),
        "bias_only": dict(which="bias"),
    }

    results = {}


    for kname, kcfg in kernels.items():
        if show_seed_progress:
            print(f"Running kernel variant: {kname}\n")
        nmse_means, nmse_sems = [], []
        mse_means, mse_sems = [], []
        improv_medians = []
        #check if reg_scale dict points to a list
        if isinstance(reg_scale[kname], list):
            #check that the length matches T_list
            if len(reg_scale[kname]) != len(T_list):
                raise ValueError(f"reg_scale list length {len(reg_scale[kname])} does not match T_list length {len(T_list)}")
            t_reg_vals = reg_scale[kname]
        else:
            t_reg_vals = [reg_scale[kname]] * len(T_list)

        pvals = []
        pvals_w = []
        T_arr = np.asarray(T_list, dtype=int)
        for T in T_list:
            seed_level_deltas = []   # for your existing sign-flip test (base - model)
            seed_level_nmse = []
            seed_level_mse = []
            seed_level_improv_med = []
            

            seed_iter = tqdm(range(n_seeds), desc=f"T={T}", disable=not show_seed_progress)
            all_task_deltas = []  # stacked across seeds, then tasks

            for s in seed_iter:
                out = run_kernel_multi_great_circle(
                    K_tasks=T,
                    pts_per_circle=pts_per_circle,
                    test_pts_per_circle=test_pts_per_circle,
                    m=m,
                    seed=1000 * T + s,
                    reg_scale=t_reg_vals[T_list.index(T)],
                    normals_mode=normals_mode,
                    random_phase=random_phase,
                    generalization=generalization,
                    backend=backend,
                    which=kcfg["which"],
                    W_std=W_std,
                    b_std=b_std,
                    kappa_params=kappa_params,
                    kappa_scale=kappa_scale,
                    # pass noise controls
                    snr=snr,
                    noise_std=noise_std,
                    noise_on=noise_on,
                )

                mse_model = out["mse_model_A"]
                mse_base  = out["mse_base_A"]

                # task-wise paired deltas
                deltas_tasks = np.asarray(mse_base) - np.asarray(mse_model)
                seed_level_deltas.append(deltas_tasks)
                
                seed_level_mse.append(float(np.mean(mse_model)))

                nmse_tasks = mse_model / np.maximum(mse_base, 1e-12)
                seed_level_nmse.append(float(np.mean(nmse_tasks)))

                rel_improv_tasks = 1.0 - nmse_tasks
                seed_level_improv_med.append(float(np.median(rel_improv_tasks)))
            
            all_task_deltas = np.concatenate(seed_level_deltas, axis=0)
            p_val = paired_signflip_pvalue_one_sided(all_task_deltas, n_perm=n_perm, seed=1234 + int(T))
            p_val_w, wstat = wilcoxon_paired_improvement(all_task_deltas, alternative="greater")
            pvals.append(p_val)
            pvals_w.append((p_val_w, wstat))
            
            seed_level_nmse = np.asarray(seed_level_nmse, dtype=float)
            seed_level_improv_med = np.asarray(seed_level_improv_med, dtype=float)
            seed_level_mse = np.asarray(seed_level_mse, dtype=float)

            nmse_means.append(float(np.mean(seed_level_nmse)))
            nmse_sems.append(float(np.std(seed_level_nmse, ddof=1) / np.sqrt(n_seeds)))
            mse_means.append(float(np.mean(seed_level_mse)))
            mse_sems.append(float(np.std(seed_level_mse, ddof=1) / np.sqrt(n_seeds)))
            improv_medians.append(float(np.median(seed_level_improv_med)))

        pvals = np.asarray(pvals, dtype=float)
        qvals = benjamini_hochberg(pvals) if bh_correct else pvals
        qvals_w = benjamini_hochberg(np.asarray([pv for pv, _ in pvals_w])) if bh_correct else np.asarray([pv for pv, _ in pvals_w])

        # "Distinguishable" means significant improvement over baseline
        sig = qvals <= alpha
        sig_w = qvals_w <= alpha
        # Capacity definition (pick one):
        # 1) First T where NOT distinguishable (fails to reject improvement)
        cap_first_not_sig = int(T_arr[np.where(~sig)[0][0]]) if np.any(~sig) else int(T_arr[-1])
        # Alternative capacity definition:
        cap_first_not_sig_w = int(T_arr[np.where(~sig_w)[0][0]]) if np.any(~sig_w) else int(T_arr[-1])

        results[kname] = dict(
            nmse_mean=np.asarray(nmse_means),
            nmse_sem=np.asarray(nmse_sems),
            mse_mean=np.asarray(mse_means),
            mse_sem=np.asarray(mse_sems),
            improv_median=np.asarray(improv_medians),
            pvals=pvals,
            pvals_w=np.asarray([pv for pv, _ in pvals_w]),
            stat_w=np.asarray([stat for _, stat in pvals_w]),
            qvals=qvals,
            qvals_w=qvals_w,
            sig=sig,
            sig_w=sig_w,
            cap_first_not_sig=cap_first_not_sig,
            cap_first_not_sig_w=cap_first_not_sig_w,
        )
    if plot:
        # ----------------------------
        # Plot panels: A) NMSE, B) -log10(q), C) repeatability vs gap
        # ----------------------------
        colors = plt.get_cmap("tab10")

        fig, axes = plt.subplots(1, 2, figsize=(16.5, 4.5))

        # Panel A: NMSE vs T
        ax = axes[0]
        for i, kname in enumerate(["full_ntk", "bias_only"]):
            r = results[kname]
            ax.errorbar(
                T_arr, r["nmse_mean"], yerr=r["nmse_sem"],
                marker="o", linestyle="-", capsize=3, label=kname, color=colors(i)
            )
        ax.axhline(1.0, linestyle="--", linewidth=1, color="k", alpha=0.6)
        ax.set_xlabel("Number of tasks T")
        ax.set_ylabel("NMSE = MSE(model) / MSE(baseline)")
        ax.set_title(f"NMSE vs T ({generalization}, backend={backend})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        ax = axes[1]
        ax.plot(T_list, results['full_ntk']['mse_mean'],label='full NTK')
        # ax.fill_between(
        #     T_list,
        #     results['full_ntk']['mse_mean'] - results['full_ntk']['mse_sem'],
        #     results['full_ntk']['mse_mean'] + results['full_ntk']['mse_sem'],
        #     alpha=0.3
        # )
        ax.plot(T_list, results['bias_only']['mse_mean'],label='bias-only NTK')
        # ax.fill_between(
        #     T_list,
        #     results['bias_only']['mse_mean'] - results['bias_only']['mse_sem'],
        #     results['bias_only']['mse_mean'] + results['bias_only']['mse_sem'],
        #     alpha=0.3
        # )

        ax.set_yscale("log")  # equivalent to semilogy, but works nicely with fill_between

        ax.legend()
        ax.set_xlabel("Number of tasks T")
        ax.set_ylabel("MSE")
        ax.set_title("MSE vs Number of tasks")
        ax.grid(True, alpha=0.3)
        ax.axhline(0.5, linestyle="--", color="r")

        plt.tight_layout()
        plt.show()

    return results

def main():
    addn_save_str = "cntrl_tot_pts"
    T_list = [1,2,3,4,5,10,15,20,25,30,35,40,50,60,70,80,90,100]
    reg_list = np.logspace(-10, -2, 40).astype(np.float64)
    kernels = ["full_ntk", "bias_only"]

    par = KernelParams(sigma_w2=1.0**2, sigma_b2=1.0**2, sigma_v2=1.0)

    nT = len(T_list)
    nL = len(reg_list)

    # Store grids as dense arrays: (T, lambda)
    mse_grid = {k: np.full((nT, nL), np.nan, dtype=np.float64) for k in kernels}
    mse_star = {k: np.full((nT,), np.nan, dtype=np.float64) for k in kernels}
    lam_star = {k: np.full((nT,), np.nan, dtype=np.float64) for k in kernels}
    N_tot = 80 * 128  # approx total number of data points
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
            )

            # Make sure these are scalars; .item() avoids the NumPy 1.25 warning.
            for k in kernels:
                mse_grid[k][ti, li] = np.asarray(out[k]["mse_mean"]).item()

        # Choose best lambda for each kernel at this T
        for k in kernels:
            j = int(np.nanargmin(mse_grid[k][ti, :]))
            lam_star[k][ti] = reg_list[j]
            mse_star[k][ti] = mse_grid[k][ti, j]

    # Output directory + save
    out_dir = "ridge_tuning_results"
    os.makedirs(out_dir, exist_ok=True)

    npz_path = os.path.join(out_dir, "ridge_tuning_by_T.npz")
    np.savez_compressed(
        npz_path,
        T_list=np.array(T_list, dtype=np.int32),
        reg_list=reg_list,
        mse_grid_full_ntk=mse_grid["full_ntk"],
        mse_grid_bias_only=mse_grid["bias_only"],
        mse_star_full_ntk=mse_star["full_ntk"],
        mse_star_bias_only=mse_star["bias_only"],
        lam_star_full_ntk=lam_star["full_ntk"],
        lam_star_bias_only=lam_star["bias_only"],
    )

    # Optional: also save a small JSON summary for quick inspection
    summary = {
        "T_list": T_list,
        "reg_list_min": float(reg_list.min()),
        "reg_list_max": float(reg_list.max()),
        "lam_star_full_ntk": {str(T): float(lam_star["full_ntk"][i]) for i, T in enumerate(T_list)},
        "lam_star_bias_only": {str(T): float(lam_star["bias_only"][i]) for i, T in enumerate(T_list)},
        "mse_star_full_ntk": {str(T): float(mse_star["full_ntk"][i]) for i, T in enumerate(T_list)},
        "mse_star_bias_only": {str(T): float(mse_star["bias_only"][i]) for i, T in enumerate(T_list)},
    }
    with open(os.path.join(out_dir, f"ridge_tuning_summary_{addn_save_str}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {npz_path}")
    print(f"Saved: {os.path.join(out_dir, f'ridge_tuning_summary_{addn_save_str}.json')}")

if __name__ == "__main__":
    main()
