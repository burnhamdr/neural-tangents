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


import numpy as np
import jax.numpy as jnp
import jax

try:
    import neural_tangents as nt
    from neural_tangents import stax
except Exception:
    nt = None
    stax = None
    
from typing import List, Optional

def pad_arrays_to_max_length_along_axis(
    arrays: List[jnp.ndarray],
    axis: int = 0,
    pad_value: Optional[float] = jnp.nan,
    promote_to_float_for_nan: bool = True,
) -> List[jnp.ndarray]:
    """
    Right-pad each JAX array along a specified axis so all arrays match the
    maximum length along that axis among the inputs.

    Padding is applied at the *end* of the specified axis only.

    Parameters
    ----------
    arrays : list of jnp.ndarray
        List of JAX arrays. Arrays may have different sizes along `axis`,
        but should have compatible shapes on all other axes if you plan
        to stack them later.
    axis : int, default=0
        Axis along which to compute max length and apply padding.
        Supports negative axes.
    pad_value : float-like, default=jnp.nan
        Constant value used for padding.
    promote_to_float_for_nan : bool, default=True
        If True and pad_value is NaN, promote non-floating inputs to float32
        so NaN is representable.

    Returns
    -------
    padded_arrays : list of jnp.ndarray
        List of padded arrays, all having the same size along `axis`.
    """
    if len(arrays) == 0:
        return []

    # Normalize axis against the first array's ndim (assumes consistent ndim).
    first = jnp.asarray(arrays[0])
    ndim = first.ndim
    if ndim == 0:
        raise ValueError("Cannot pad scalar (0-d) arrays.")

    axis_norm = axis % ndim

    # (Optional) sanity check: all arrays have same ndim
    for i, a in enumerate(arrays):
        if jnp.asarray(a).ndim != ndim:
            raise ValueError(
                f"All arrays must have the same number of dimensions. "
                f"arrays[0].ndim={ndim}, arrays[{i}].ndim={jnp.asarray(a).ndim}"
            )

    # Compute target length along the specified axis
    max_len = max(jnp.asarray(a).shape[axis_norm] for a in arrays)

    padded_arrays = []
    for a in arrays:
        a = jnp.asarray(a)

        # Ensure float dtype if we're padding with NaN
        if (
            promote_to_float_for_nan
            and (pad_value is jnp.nan or (isinstance(pad_value, float) and jnp.isnan(pad_value)))
            and not jnp.issubdtype(a.dtype, jnp.floating)
        ):
            a = a.astype(jnp.float32)

        pad_amt = max_len - a.shape[axis_norm]
        if pad_amt < 0:
            raise ValueError("Unexpected: pad_amt < 0. Check max_len computation.")

        pad_width = [(0, 0)] * ndim
        pad_width[axis_norm] = (0, pad_amt)

        padded = jnp.pad(
            a,
            pad_width=pad_width,
            mode="constant",
            constant_values=pad_value,
        )
        padded_arrays.append(padded)

    return padded_arrays


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
    def __init__(self, sigma_w2=1.0, sigma_b2=1.0, sigma_v2=1.0, sigma_bout2=1.0):
        self.sigma_w2 = float(sigma_w2)
        self.sigma_b2 = float(sigma_b2)
        self.sigma_v2 = float(sigma_v2)
        self.sigma_bout2 = float(sigma_bout2)

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
        K = k0 + par.sigma_v2 * (t + 1.0) * k1 + par.sigma_bout2
    elif which == "bias":
        K = par.sigma_v2 * k1 + par.sigma_bout2
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
    compute_spectrum: bool = False,
    compute_testA: bool = True,
    compute_testB: bool = True,
):
    """
    Returns per-task MSEs on the requested independent test replicates (A, B).

    Replicate A is intended for hyperparameter selection / validation.
    Replicate B is intended for final reporting.
    """
    if not (compute_testA or compute_testB):
        raise ValueError("At least one of compute_testA or compute_testB must be True.")

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
    y_train_np_clean = y_train_np.copy()

    # ----------------------------
    # Test replicate A/B: independent random sampling of phi
    # ----------------------------
    X_testA_np = y_testA_np = task_slices_testA = None
    X_testB_np = y_testB_np = task_slices_testB = None
    if compute_testA:
        rngA = np.random.default_rng(seed + 10_000_001)
        X_testA_np, y_testA_np, task_slices_testA = generate_multi_tasks_from_normals_phases_with_mode(
            normals_te, phases_te, pts_per_circle=test_pts_per_circle, m=m, rng=rngA, mode="random"
        )
    if compute_testB:
        rngB = np.random.default_rng(seed + 10_000_002)
        X_testB_np, y_testB_np, task_slices_testB = generate_multi_tasks_from_normals_phases_with_mode(
            normals_te, phases_te, pts_per_circle=test_pts_per_circle, m=m, rng=rngB, mode="random"
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
            if compute_testA:
                rngNtA = np.random.default_rng(seed + 30_000_001)
                y_testA_np = y_testA_np + rngNtA.normal(0.0, noise_std, size=y_testA_np.shape)
            if compute_testB:
                rngNtB = np.random.default_rng(seed + 30_000_002)
                y_testB_np = y_testB_np + rngNtB.normal(0.0, noise_std, size=y_testB_np.shape)

    X_train = jnp.array(X_train_np, dtype=jnp.float32)
    y_train = jnp.array(y_train_np, dtype=jnp.float32).reshape(-1, 1)

    X_testA = y_testA = None
    X_testB = y_testB = None
    if compute_testA:
        X_testA = jnp.array(X_testA_np, dtype=jnp.float32)
        y_testA = jnp.array(y_testA_np, dtype=jnp.float32).reshape(-1, 1)
    if compute_testB:
        X_testB = jnp.array(X_testB_np, dtype=jnp.float32)
        y_testB = jnp.array(y_testB_np, dtype=jnp.float32).reshape(-1, 1)

    # ----------------------------
    # Kernels
    # ----------------------------
    if backend == "nt":
        K_train = gram_from_neural_tangents(X_train, X_train, which=which, W_std=W_std, b_std=b_std)

        def cross_kernel_fn(X_chunk):
            return gram_from_neural_tangents(X_chunk, X_train, which=which, W_std=W_std, b_std=b_std)

    elif backend == "kappa":
        if kappa_params is None:
            kappa_params = KernelParams(sigma_w2=W_std**2, sigma_b2=b_std**2, sigma_v2=1.0)

        kappa_kernel_fn = make_kappa_kernel_fn(kappa_params, which=which, kappa_scale=kappa_scale)
        # batch for memory
        if nt is None:
            raise ImportError("neural_tangents not available for nt.batch; install neural_tangents or remove batching.")
        kappa_kernel_fn = nt.batch(kappa_kernel_fn, device_count=-1)

        K_train = jnp.array(kappa_kernel_fn(X_train, X_train, get="ntk"))

        def cross_kernel_fn(X_chunk):
            return jnp.array(kappa_kernel_fn(X_chunk, X_train, get="ntk"))
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
    
    if compute_spectrum:
        eigvals = jnp.linalg.eigvalsh(K_train)
        # print(eigvals.shape)

    def predict_in_chunks(X_test, max_kernel_mb: int = 256):
        """
        Evaluate K(X_test, X_train) @ alpha without materializing the full
        test-train kernel matrix, which can exceed device memory for small T.
        """
        bytes_per_entry = np.dtype(np.float32).itemsize
        target_entries = max(1, (max_kernel_mb * 1024 * 1024) // bytes_per_entry)
        chunk_size = max(1, int(target_entries // max(1, n)))
        preds = []
        for start in range(0, X_test.shape[0], chunk_size):
            stop = min(start + chunk_size, X_test.shape[0])
            X_chunk = X_test[start:stop]
            K_chunk = cross_kernel_fn(X_chunk) / s_trace
            y_chunk = K_chunk @ alpha
            preds.append(np.asarray(y_chunk).reshape(-1))
        return np.concatenate(preds, axis=0)

    # Predictions on A/B
    y_predA_np = y_testA_np_flat = None
    y_predB_np = y_testB_np_flat = None
    if compute_testA:
        y_predA_np = predict_in_chunks(X_testA)
        y_testA_np_flat = np.asarray(y_testA).reshape(-1)
    if compute_testB:
        y_predB_np = predict_in_chunks(X_testB)
        y_testB_np_flat = np.asarray(y_testB).reshape(-1)

    # ----------------------------
    # Per-task MSEs and baselines on both replicates
    # ----------------------------
    mse_model_A, mse_base_A = None, None
    mse_model_B, mse_base_B = None, None
    if compute_testA:
        mse_model_A, mse_base_A = [], []
    if compute_testB:
        mse_model_B, mse_base_B = [], []

    for task_idx, sl_tr in enumerate(task_slices_train):
        # baseline constant fitted on TRAIN for that task
        c = float(np.mean(y_train_np_clean[sl_tr]))
        if compute_testA:
            sl_A = task_slices_testA[task_idx]
            ytA = y_testA_np_flat[sl_A]
            ypA = y_predA_np[sl_A]
            mse_model_A.append(float(np.mean((ytA - ypA) ** 2)))
            mse_base_A.append(float(np.mean((ytA - c) ** 2)))
        if compute_testB:
            sl_B = task_slices_testB[task_idx]
            ytB = y_testB_np_flat[sl_B]
            ypB = y_predB_np[sl_B]
            mse_model_B.append(float(np.mean((ytB - ypB) ** 2)))
            mse_base_B.append(float(np.mean((ytB - c) ** 2)))

    if compute_testA:
        mse_model_A = np.array(mse_model_A)
        mse_base_A  = np.array(mse_base_A)
    if compute_testB:
        mse_model_B = np.array(mse_model_B)
        mse_base_B  = np.array(mse_base_B)


    return dict(
        mse_model_A=mse_model_A,
        mse_base_A=mse_base_A,
        mse_model_B=mse_model_B,
        mse_base_B=mse_base_B,
        # metadata
        generalization=generalization,
        backend=backend,
        which=which,
        noise_std_used=noise_std,
        eigvals=eigvals if compute_spectrum else None,
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
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
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
    cntrl_pts_per_task: bool = False,
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
    alpha: float | None = None,
    n_perm: int = 5000,
    bh_correct: bool = True,
    compute_spectrum: bool = False,
    selection_split: str = "A",
    report_split: str | None = "B",
):
    valid_splits = {"A", "B"}
    if selection_split not in valid_splits:
        raise ValueError("selection_split must be 'A' or 'B'.")
    if report_split is not None and report_split not in valid_splits:
        raise ValueError("report_split must be None, 'A', or 'B'.")

    need_testA = selection_split == "A" or report_split == "A"
    need_testB = selection_split == "B" or report_split == "B"
    summary_split = report_split if report_split is not None else selection_split

    def _empty_metric_array():
        return np.full(len(T_list), np.nan, dtype=float)

    kernels = {
        "full_ntk": dict(which="full"),
        "bias_only": dict(which="bias"),
    }

    results = {}


    for kname, kcfg in kernels.items():
        if show_seed_progress:
            print(f"Running kernel variant: {kname}\n")
        nmse_means_A, nmse_sems_A = [], []
        mse_means_A, mse_sems_A = [], []
        base_means_A, base_sems_A = [], []
        nmse_means_B, nmse_sems_B = [], []
        mse_means_B, mse_sems_B = [], []
        base_means_B, base_sems_B = [], []
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
        if compute_spectrum:
            eigs = []
        T_arr = np.asarray(T_list, dtype=int)
        max_T = int(np.max(T_arr))
        for T in T_list:
            seed_level_deltas = []   # evaluated on the summary split used for reporting
            seed_level_nmse_A = []
            seed_level_mse_A = []
            seed_level_base_A = []
            seed_level_nmse_B = []
            seed_level_mse_B = []
            seed_level_base_B = []
            seed_level_improv_med = []
            if compute_spectrum:    
                seed_level_eigs = []

            seed_iter = tqdm(range(n_seeds), desc=f"T={T}", disable=not show_seed_progress)
            all_task_deltas = []  # stacked across seeds, then tasks
            if cntrl_pts_per_task:
                N_tot = max_T * pts_per_circle  # approx total number of data points
                pts_per_circle_T = max(16, N_tot // T)
            else:
                pts_per_circle_T = pts_per_circle
                
            for s in seed_iter:
                out = run_kernel_multi_great_circle(
                    K_tasks=T,
                    pts_per_circle=pts_per_circle_T,
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
                    compute_spectrum=compute_spectrum,
                    compute_testA=need_testA,
                    compute_testB=need_testB,
                )
                if compute_spectrum:
                    seed_level_eigs.append(out["eigvals"])

                mse_model_A = out["mse_model_A"]
                mse_base_A  = out["mse_base_A"]
                mse_model_B = out["mse_model_B"]
                mse_base_B  = out["mse_base_B"]
                noise_std_used = out["noise_std_used"]
                # --- Option A: subtract test noise floor (excess MSE) ---
                # Only subtract if you actually injected noise into the TEST labels.
                if (noise_std is not None or snr is not None) and (noise_on in ("test", "both")):
                    # Recover the actual noise_std used in the run
                    # Best: have run_kernel_multi_great_circle return noise_std_used.
                    # If not, recompute it the same way as in run_kernel_multi_great_circle:
                    if noise_std_used is None and snr is not None:
                        # This is slightly risky to recompute here unless you return it.
                        # Prefer returning noise_std_used from run_kernel_multi_great_circle.
                        raise RuntimeError("Return noise_std_used from run_kernel_multi_great_circle for correctness.")
                    
                    noise_var = float(noise_std_used)**2

                    if mse_model_A is not None:
                        mse_model_A_excess = np.maximum(mse_model_A - noise_var, 0.0)
                        mse_base_A_excess  = np.maximum(mse_base_A  - noise_var, 0.0)
                    else:
                        mse_model_A_excess = None
                        mse_base_A_excess = None
                    if mse_model_B is not None:
                        mse_model_B_excess = np.maximum(mse_model_B - noise_var, 0.0)
                        mse_base_B_excess  = np.maximum(mse_base_B  - noise_var, 0.0)
                    else:
                        mse_model_B_excess = None
                        mse_base_B_excess = None
                else:
                    mse_model_A_excess = mse_model_A
                    mse_base_A_excess  = mse_base_A
                    mse_model_B_excess = mse_model_B
                    mse_base_B_excess  = mse_base_B

                if summary_split == "A":
                    deltas_tasks = np.asarray(mse_base_A_excess) - np.asarray(mse_model_A_excess)
                else:
                    deltas_tasks = np.asarray(mse_base_B_excess) - np.asarray(mse_model_B_excess)
                seed_level_deltas.append(deltas_tasks)
                
                if mse_model_A is not None:
                    seed_level_mse_A.append(float(np.mean(mse_model_A)))
                    seed_level_base_A.append(float(np.mean(mse_base_A)))
                if mse_model_B is not None:
                    seed_level_mse_B.append(float(np.mean(mse_model_B)))
                    seed_level_base_B.append(float(np.mean(mse_base_B)))

                if mse_model_A is not None:
                    nmse_tasks_A = mse_model_A / np.maximum(mse_base_A, 1e-12)
                    seed_level_nmse_A.append(float(np.mean(nmse_tasks_A)))
                else:
                    nmse_tasks_A = None
                if mse_model_B is not None:
                    nmse_tasks_B = mse_model_B / np.maximum(mse_base_B, 1e-12)
                    seed_level_nmse_B.append(float(np.mean(nmse_tasks_B)))
                else:
                    nmse_tasks_B = None

                rel_improv_tasks = 1.0 - (nmse_tasks_A if summary_split == "A" else nmse_tasks_B)
                seed_level_improv_med.append(float(np.median(rel_improv_tasks)))
            
            if compute_spectrum:
                #stack and convert to float, all seed level eigval array should have same shape
                #since the size of the data should be the same for each seed for given T
                seed_level_eigs = jnp.stack(seed_level_eigs, axis=0)#stacked over seed
                eigs.append(seed_level_eigs)
            
            all_task_deltas = np.concatenate(seed_level_deltas, axis=0)
            if alpha is not None:
                p_val = paired_signflip_pvalue_one_sided(all_task_deltas, n_perm=n_perm, seed=1234 + int(T))
                p_val_w, wstat = wilcoxon_paired_improvement(all_task_deltas, alternative="greater")
                pvals.append(p_val)
                pvals_w.append((p_val_w, wstat))
            
            if seed_level_nmse_A:
                seed_level_nmse_A = np.asarray(seed_level_nmse_A, dtype=float)
                seed_level_mse_A = np.asarray(seed_level_mse_A, dtype=float)
                seed_level_base_A = np.asarray(seed_level_base_A, dtype=float)
            else:
                seed_level_nmse_A = seed_level_mse_A = seed_level_base_A = None
            if seed_level_nmse_B:
                seed_level_nmse_B = np.asarray(seed_level_nmse_B, dtype=float)
                seed_level_mse_B = np.asarray(seed_level_mse_B, dtype=float)
                seed_level_base_B = np.asarray(seed_level_base_B, dtype=float)
            else:
                seed_level_nmse_B = seed_level_mse_B = seed_level_base_B = None
            seed_level_improv_med = np.asarray(seed_level_improv_med, dtype=float)

            if seed_level_nmse_A is not None:
                nmse_means_A.append(float(np.mean(seed_level_nmse_A)))
                nmse_sems_A.append(float(np.std(seed_level_nmse_A, ddof=1) / (np.sqrt(n_seeds) + 1e-12)))
                mse_means_A.append(float(np.mean(seed_level_mse_A)))
                mse_sems_A.append(float(np.std(seed_level_mse_A, ddof=1) / (np.sqrt(n_seeds) + 1e-12)))
                base_means_A.append(float(np.mean(seed_level_base_A)))
                base_sems_A.append(float(np.std(seed_level_base_A, ddof=1) / (np.sqrt(n_seeds) + 1e-12)))
            if seed_level_nmse_B is not None:
                nmse_means_B.append(float(np.mean(seed_level_nmse_B)))
                nmse_sems_B.append(float(np.std(seed_level_nmse_B, ddof=1) / (np.sqrt(n_seeds) + 1e-12)))
                mse_means_B.append(float(np.mean(seed_level_mse_B)))
                mse_sems_B.append(float(np.std(seed_level_mse_B, ddof=1) / (np.sqrt(n_seeds) + 1e-12)))
                base_means_B.append(float(np.mean(seed_level_base_B)))
                base_sems_B.append(float(np.std(seed_level_base_B, ddof=1) / (np.sqrt(n_seeds) + 1e-12)))
            improv_medians.append(float(np.median(seed_level_improv_med)))

        if compute_spectrum:
            #because controlling for total points means changing pts_per_circle with T,
            #the eigval arrays may have different lengths along the last axis because we sometimes
            #cannot cleanly divide the total number of points by T tasks and thus have different data sizes
            eigs = pad_arrays_to_max_length_along_axis(eigs, axis=-1)  # pad along eigval axis
            eigs = jnp.stack(eigs, axis=0)  # (T, n_seeds, n_eigvals)
            #convert to numpy
            eigs = np.asarray(eigs, dtype=float)
        
        
        nmse_mean_A_arr = np.asarray(nmse_means_A, dtype=float) if nmse_means_A else _empty_metric_array()
        nmse_sem_A_arr = np.asarray(nmse_sems_A, dtype=float) if nmse_sems_A else _empty_metric_array()
        mse_mean_A_arr = np.asarray(mse_means_A, dtype=float) if mse_means_A else _empty_metric_array()
        mse_sem_A_arr = np.asarray(mse_sems_A, dtype=float) if mse_sems_A else _empty_metric_array()
        base_mean_A_arr = np.asarray(base_means_A, dtype=float) if base_means_A else _empty_metric_array()
        base_sem_A_arr = np.asarray(base_sems_A, dtype=float) if base_sems_A else _empty_metric_array()
        nmse_mean_B_arr = np.asarray(nmse_means_B, dtype=float) if nmse_means_B else _empty_metric_array()
        nmse_sem_B_arr = np.asarray(nmse_sems_B, dtype=float) if nmse_sems_B else _empty_metric_array()
        mse_mean_B_arr = np.asarray(mse_means_B, dtype=float) if mse_means_B else _empty_metric_array()
        mse_sem_B_arr = np.asarray(mse_sems_B, dtype=float) if mse_sems_B else _empty_metric_array()
        base_mean_B_arr = np.asarray(base_means_B, dtype=float) if base_means_B else _empty_metric_array()
        base_sem_B_arr = np.asarray(base_sems_B, dtype=float) if base_sems_B else _empty_metric_array()

        if summary_split == "A":
            nmse_mean_arr, nmse_sem_arr = nmse_mean_A_arr, nmse_sem_A_arr
            mse_mean_arr, mse_sem_arr = mse_mean_A_arr, mse_sem_A_arr
            base_mean_arr, base_sem_arr = base_mean_A_arr, base_sem_A_arr
        else:
            nmse_mean_arr, nmse_sem_arr = nmse_mean_B_arr, nmse_sem_B_arr
            mse_mean_arr, mse_sem_arr = mse_mean_B_arr, mse_sem_B_arr
            base_mean_arr, base_sem_arr = base_mean_B_arr, base_sem_B_arr

        results[kname] = dict(
                nmse_mean=nmse_mean_arr,
                nmse_sem=nmse_sem_arr,
                mse_mean=mse_mean_arr,
                mse_sem=mse_sem_arr,
                base_mean=base_mean_arr,
                base_sem=base_sem_arr,
                nmse_mean_A=nmse_mean_A_arr,
                nmse_sem_A=nmse_sem_A_arr,
                mse_mean_A=mse_mean_A_arr,
                mse_sem_A=mse_sem_A_arr,
                base_mean_A=base_mean_A_arr,
                base_sem_A=base_sem_A_arr,
                nmse_mean_B=nmse_mean_B_arr,
                nmse_sem_B=nmse_sem_B_arr,
                mse_mean_B=mse_mean_B_arr,
                mse_sem_B=mse_sem_B_arr,
                base_mean_B=base_mean_B_arr,
                base_sem_B=base_sem_B_arr,
                improv_median=np.asarray(improv_medians),
                eigvals=eigs if compute_spectrum else None,
                T_list=T_arr,
            )
        
        if alpha is None:
            continue
        else:
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

            results[kname].update(
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

def run_snr_sweep_and_save(
    out_path: str,
    snr_list,
    T_list,
    reg_scale,
    n_seeds: int = 20,
    pts_per_circle: int = 512,
    cntrl_pts_per_task: bool = False,
    test_pts_per_circle: int | None = None,
    m: int = 4,
    backend: str = "kappa",
    generalization: str = "within",
    normals_mode: str = "random",
    random_phase: bool = True,
    W_std: float = 1.0,
    b_std: float = 1.0,
    kappa_params=None,
    kappa_scale: float = 1.0,
    alpha: float | None = None,
    n_perm: int = 5000,
    bh_correct: bool = True,
    show_snr_progress: bool = True,
    noise_on: str = "both",
    compute_spectrum: bool = False,
):
    snr_list = np.asarray(snr_list, dtype=float)
    T_list_arr = np.asarray(T_list, dtype=int)

    # Store results in arrays indexed by [snr_idx, T_idx]
    def alloc():
        return np.zeros((len(snr_list), len(T_list_arr)), dtype=float)

    out = {
        "snr_list": snr_list,
        "T_list": T_list_arr,
        "reg_list_placeholder": np.array([]),  # optional
        # metrics
        "nmse_full": alloc(),
        "nmse_bias": alloc(),
        "mse_full": alloc(),
        "mse_bias": alloc(),
        "p_full": alloc(),
        "p_bias": alloc(),
        "q_full": alloc(),
        "q_bias": alloc(),
        # capacities per snr
        "cap_full_first_not_sig": np.zeros(len(snr_list), dtype=int),
        "cap_bias_first_not_sig": np.zeros(len(snr_list), dtype=int),
    }
    snr_iter = tqdm(snr_list, desc="SNR sweep", disable=not show_snr_progress)

    for i, snr in enumerate(snr_iter):
        #for SNR below 1, put floor on bias_only reg at 1E-9
        if snr < 1.0:
            reg_scale_adjusted = {
                "full_ntk": reg_scale["full_ntk"],
                "bias_only": list(np.maximum(np.array(reg_scale["bias_only"]), 1e-9)),
            }
        else:
            reg_scale_adjusted = reg_scale
        res = make_effective_capacity_panels(
            T_list=T_list_arr.tolist(),
            n_seeds=n_seeds,
            pts_per_circle=pts_per_circle,
            cntrl_pts_per_task=cntrl_pts_per_task,
            test_pts_per_circle=test_pts_per_circle,
            m=m,
            normals_mode=normals_mode,
            random_phase=random_phase,
            W_std=W_std,
            b_std=b_std,
            reg_scale=reg_scale_adjusted,          # can be scalar per kernel or list aligned with T_list
            backend=backend,
            generalization=generalization,
            kappa_params=kappa_params,
            kappa_scale=kappa_scale,
            plot=False,
            # noise
            snr=float(snr),
            noise_std=None,
            noise_on=noise_on,
            # stats
            alpha=alpha,
            n_perm=n_perm,
            bh_correct=bh_correct,
            compute_spectrum=compute_spectrum
            
        )

        # pack
        out["nmse_full"][i, :] = res["full_ntk"]["nmse_mean"]
        out["nmse_bias"][i, :] = res["bias_only"]["nmse_mean"]
        out["mse_full"][i, :]  = res["full_ntk"]["mse_mean"]
        out["mse_bias"][i, :]  = res["bias_only"]["mse_mean"]

        out["p_full"][i, :] = res["full_ntk"]["pvals_w"]
        out["p_bias"][i, :] = res["bias_only"]["pvals_w"]
        out["q_full"][i, :] = res["full_ntk"]["qvals_w"]
        out["q_bias"][i, :] = res["bias_only"]["qvals_w"]

        out["cap_full_first_not_sig"][i] = res["full_ntk"]["cap_first_not_sig_w"]
        out["cap_bias_first_not_sig"][i] = res["bias_only"]["cap_first_not_sig_w"]

    np.savez_compressed(out_path, **out)
    return out


def interpolate_lambda_star(
    T_list,
    lam_star,
    T_new,
    clip=True,
):
    """
    Linearly interpolate lambda*(T).

    Parameters
    ----------
    T_list : array-like, shape (n_T,)
        Original task counts where lambda* was tuned.
        Must be sorted in increasing order.

    lam_star : array-like, shape (n_T,)
        Optimal lambda values corresponding to T_list.

    T_new : array-like or scalar
        New task counts at which to interpolate lambda*.

    clip : bool, default=True
        If True, values outside [min(T_list), max(T_list)]
        are clipped to the boundary lambda values.
        If False, numpy.interp will still clip implicitly,
        but this makes intent explicit.

    Returns
    -------
    lam_new : ndarray or scalar
        Interpolated lambda values at T_new.
    """

    T_list = np.asarray(T_list, dtype=float)
    lam_star = np.asarray(lam_star, dtype=float)

    # Safety checks
    if T_list.ndim != 1 or lam_star.ndim != 1:
        raise ValueError("T_list and lam_star must be 1D arrays.")
    if T_list.shape[0] != lam_star.shape[0]:
        raise ValueError("T_list and lam_star must have same length.")

    T_new_arr = np.asarray(T_new, dtype=float)

    if clip:
        T_new_arr = np.clip(T_new_arr, T_list.min(), T_list.max())

    lam_new = np.interp(T_new_arr, T_list, lam_star)

    # Preserve scalar input
    if np.isscalar(T_new):
        return float(lam_new)

    return lam_new
