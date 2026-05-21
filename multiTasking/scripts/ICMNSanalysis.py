import warnings
import json
import csv
import numpy as np
import jax.numpy as jnp
try:
    import neural_tangents as nt
    from neural_tangents import stax
except Exception:
    nt = None
    stax = None
from scipy.special import sph_harm
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib import colors as mcolors
    import matplotlib.cm as cm
except Exception:
    plt = None
    Axes3D = None
    inset_axes = None
    mcolors = None
    cm = None
from tqdm import tqdm
from colorsys import hls_to_rgb
from scipy.special import eval_legendre
from scipy.integrate import quad
from dataclasses import dataclass
from scipy import special
from matplotlib.lines import Line2D

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

from multiTasking.utils.utils import (
    KernelParams,
    _relu_kappa_components_sphere,
    gram_from_kappa,
    sample_random_normals_s2,
    normals_fibonacci_s2,
    generate_single_circle_with_phi,
    generate_multi_tasks_from_normals_phases_with_mode,
)


warnings.filterwarnings(
    "ignore",
    message="`scipy.special.sph_harm` is deprecated*",  # match SciPy's exact message
    category=DeprecationWarning
)


def _require_matplotlib():
    if plt is None:
        raise ImportError("matplotlib is required for plotting in ICMNSanalysis.py")


def _read_csv_rows(path: Path | str):
    path = Path(path)
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))

# -----------------------------
# Geometry helpers
# -----------------------------
def sphere_surface_area(d: int) -> float:
    # ω_d = 2 π^{(d+1)/2} / Γ((d+1)/2)
    return 2.0 * np.pi ** ((d + 1) / 2.0) / special.gamma((d + 1) / 2.0)

def omega_ratio(p: int) -> float:
    return sphere_surface_area(p - 1) / sphere_surface_area(p - 2)


# -----------------------------
# ReLU kernel pieces (biased) on sphere
# -----------------------------
def kappa_bias_only(t: np.ndarray, par: KernelParams) -> np.ndarray:
    _, k1 = _relu_kappa_components_sphere(t, par)
    return par.sigma_v2 * k1 + par.sigma_bout2

def kappa_full_ntk(t: np.ndarray, par: KernelParams) -> np.ndarray:
    k0, k1 = _relu_kappa_components_sphere(t, par)
    return k0 + par.sigma_v2 * (t + 1.0) * k1 + par.sigma_bout2

# -----------------------------
# Colleague-style kappa(t) probing from neural-tangents kernel_fn (S^2 only)
# -----------------------------
def xt_from_t_s2(t: float) -> np.ndarray:
    """
    Return a point on S^2 with dot(x0, xt) = t, where x0 = (0,0,1).
    """
    t = float(np.clip(t, -1.0, 1.0))
    return np.array([[np.sqrt(max(0.0, 1.0 - t*t)), 0.0, t]], dtype=np.float64)

def kappa_from_kernel_fn(t: float, kernel_fn) -> float:
    """
    Evaluate isotropic kernel kappa(t) by probing kernel_fn(x0, xt, 'ntk') on S^2.
    kernel_fn is assumed to accept numpy/jax arrays; we pass numpy.
    """
    x0 = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    xt = xt_from_t_s2(t)
    val = kernel_fn(x0, xt, "ntk")
    # kernel_fn might return numpy, jax DeviceArray, or pytree; handle scalar [0,0]
    val = np.asarray(val)
    return float(val[0, 0])

def mercer_lambda_legendre_quad(l: int, kernel_fn, epsabs=1e-8, epsrel=1e-8) -> float:
    """
    Colleague method for S^2: lambda_l = 2π ∫_{-1}^1 kappa(t) P_l(t) dt.
    """
    Pl = lambda t: special.eval_legendre(l, t)
    integrand = lambda t: kappa_from_kernel_fn(t, kernel_fn) * Pl(t)
    val, _ = quad(integrand, -1.0, 1.0, epsabs=epsabs, epsrel=epsrel, limit=200)
    return float(2.0 * np.pi * val)

def compute_spectrum_colleague(
    Lmax: int,
    kernel_fn,
    epsabs=1e-8,
    epsrel=1e-8,
) -> np.ndarray:
    """
    Degree spectrum via colleague method (S^2 only).
    Returns lambdas[0..Lmax].
    """
    lam = np.zeros(Lmax + 1, dtype=float)
    for l in range(Lmax + 1):
        lam[l] = mercer_lambda_legendre_quad(l, kernel_fn, epsabs=epsabs, epsrel=epsrel)
        print(f"l={l}: lambda_colleague={lam[l]:.6g}")
    return lam

# -----------------------------
# Kernel ratio diagnostic
# -----------------------------
def kernel_ratio_grid(
    kappa_analytic_fn,
    par: KernelParams,
    kernel_fn,
    n_grid: int = 401,
    clip: float = 1e-14,
):
    """
    Evaluate kappa_nt(t)/kappa_analytic(t) on a t-grid.
    """
    ts = np.linspace(-1.0, 1.0, n_grid)
    k_ana = np.asarray(kappa_analytic_fn(ts, par), dtype=float)
    k_nt = np.array([kappa_from_kernel_fn(t, kernel_fn) for t in ts], dtype=float)

    denom = np.where(np.abs(k_ana) < clip, np.sign(k_ana) * clip + (k_ana == 0) * clip, k_ana)
    ratio = k_nt / denom
    return ts, k_nt, k_ana, ratio

def best_global_scale(a: np.ndarray, b: np.ndarray, mask=None) -> float:
    """
    Find scalar s minimizing ||s*a - b||_2 over masked entries.
    Useful to align colleague eigenvalues to analytic ones when only a scale differs.
    """
    if mask is None:
        mask = np.ones_like(a, dtype=bool)
    aa = a[mask]
    bb = b[mask]
    denom = float(np.dot(aa, aa))
    if denom <= 0:
        return 1.0
    return float(np.dot(aa, bb) / denom)



# -----------------------------
# Funk–Hecke via Gauss–Jacobi quadrature
# -----------------------------
#check convergence
def mu_convergence(l, kappa_fn, par, n1=300, n2=600):
    m1 = mu_ell_via_jacobi(l, kappa_fn, par, n_quad=n1)
    m2 = mu_ell_via_jacobi(l, kappa_fn, par, n_quad=n2)
    rel = abs(m2 - m1) / max(abs(m2), 1e-14)
    return m1, m2, rel

def mu_ell_via_jacobi(l: int, kappa_fn, par: KernelParams, n_quad: int = 500, p: int = 3) -> float:
    alpha = (p - 2) / 2.0
    a = alpha - 0.5  # (p-3)/2
    b = alpha - 0.5

    # nodes/weights for ∫_{-1}^1 f(t) (1-t)^a (1+t)^b dt
    t_nodes, w_nodes = special.roots_jacobi(n_quad, a, b)

    C_l = special.eval_gegenbauer(l, alpha, t_nodes)
    C_l_1 = float(special.eval_gegenbauer(l, alpha, 1.0))

    kap = kappa_fn(t_nodes, par)
    integral = float(np.sum(w_nodes * kap * C_l))

    return float(omega_ratio(p) * (integral / C_l_1))

def compute_spectrum(Lmax: int, kappa_fn, par: KernelParams, n_quad: int = 500, p: int = 3) -> np.ndarray:
    mus = np.zeros(Lmax + 1, dtype=float)
    for l in range(Lmax + 1):
        mus[l] = mu_ell_via_jacobi(l, kappa_fn, par, n_quad=n_quad, p=p)
        #check convergence
        m1, m2, rel = mu_convergence(l, kappa_fn, par, n1=n_quad, n2=n_quad*2)
        print(f"l={l}: mu(n={n_quad})={m1:.6g}, mu(n={n_quad*2})={m2:.6g}, rel diff={rel:.3g}")
    return mus


def compute_spectrum_quiet(
    Lmax: int,
    kappa_fn,
    par: KernelParams,
    n_quad: int = 500,
    p: int = 3,
) -> np.ndarray:
    mus = np.zeros(Lmax + 1, dtype=float)
    for l in range(Lmax + 1):
        mus[l] = mu_ell_via_jacobi(l, kappa_fn, par, n_quad=n_quad, p=p)
    return mus

# -----------------------------
# Error-band computation
# -----------------------------
def spectrum_with_error_band(
    Lmax: int,
    kappa_fn,
    par: KernelParams,
    n_quad: int,
    use_center: str = "fine",   # "fine" uses 2n as center; "coarse" uses n as center
    eps: float = 1e-300,        # avoid zeros for semilogy fill clipping
    p: int = 3,                 # dimension for Funk–Hecke quadrature (S^{p-1} sphere
):
    """
    Returns:
      ells: degrees
      mu_center: chosen center curve (default: fine = n=2n)
      mu_lo, mu_hi: error band using |mu_fine - mu_coarse|
      mu_coarse, mu_fine: raw curves (n and 2n)
    """
    mu_coarse = compute_spectrum(Lmax, kappa_fn, par, n_quad=n_quad, p=p)
    mu_fine = compute_spectrum(Lmax, kappa_fn, par, n_quad=2 * n_quad, p=p)

    err = abs(mu_fine - mu_coarse) / np.maximum(abs(mu_fine), 1e-14)

    if use_center == "fine":
        mu_center = mu_fine
    elif use_center == "coarse":
        mu_center = mu_coarse
    else:
        raise ValueError("use_center must be 'fine' or 'coarse'.")

    mu_lo = np.maximum(mu_center - err, eps)
    mu_hi = np.maximum(mu_center + err, eps)

    ells = np.arange(Lmax + 1, dtype=float)
    return ells, mu_center, mu_lo, mu_hi, mu_coarse, mu_fine, err



# -----------------------------
# Fit models (log-space least squares)
# -----------------------------
def _log_r2(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    ss_res = np.sum((y_true_log - y_pred_log) ** 2)
    ss_tot = np.sum((y_true_log - np.mean(y_true_log)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

def fit_powerlaw_log(mus: np.ndarray, ell_min: int, ell_max: int):
    """
    Fit mu ≈ C * ell^{-s} via log(mu)=log C - s log ell   (ell>=1)
    Returns C, s, R2_log
    """
    ells = np.arange(len(mus), dtype=float)
    sel = (ells >= max(1, ell_min)) & (ells <= ell_max) & (mus > 0)
    x = np.log(ells[sel])
    y = np.log(mus[sel])

    slope, intercept = np.polyfit(x, y, 1)  # y = slope*x + intercept
    C = float(np.exp(intercept))
    s = float(-slope)

    yhat = intercept + slope * x
    r2 = float(_log_r2(y, yhat))
    return C, s, r2


# -----------------------------
# Bietti reference (power law with exponent p)
# -----------------------------
def bietti_reference_powerlaw(Lmax: int, p: int, mu_ref: float, ell_ref: int) -> np.ndarray:
    ells = np.arange(Lmax + 1, dtype=float)
    ref = np.full_like(ells, np.nan, dtype=float)  # NaN at ell=0 prevents spike/line

    mask = ells >= 1
    ref[mask] = ells[mask] ** (-p)

    # normalize at ell_ref
    if ell_ref < 1 or ell_ref > Lmax:
        raise ValueError("ell_ref must be between 1 and Lmax for Bietti reference.")
    scale = mu_ref / (ell_ref ** (-p))
    ref[mask] *= scale
    return ref


def spherical_harmonic_multiplicity(ell: int | np.ndarray, p: int) -> np.ndarray:
    ell = np.asarray(ell, dtype=int)
    if p < 2:
        raise ValueError("p must be at least 2 for S^{p-1}.")
    numer = (2 * ell + p - 2) * special.gamma(ell + p - 2)
    denom = special.gamma(ell + 1) * special.gamma(p - 1)
    mult = np.rint(numer / denom).astype(int)
    return np.maximum(mult, 1)


def expand_degree_spectrum_to_sorted_ranks(
    mu_ell: np.ndarray,
    p: int,
    max_rank: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mu_ell = np.asarray(mu_ell, dtype=float).reshape(-1)
    mult = spherical_harmonic_multiplicity(np.arange(len(mu_ell)), p=p)
    expanded = np.repeat(mu_ell, mult)
    expanded = clean_empirical_eigvals(expanded, clip_negative_to_zero=True, sort_descending=True)
    if max_rank is not None:
        expanded = expanded[: int(max_rank)]
    return expanded, mult


def analytic_budget_reference_from_degree_spectrum(
    T_list,
    mu_ell_raw: np.ndarray,
    kappa_fn,
    par: KernelParams,
    p: int = 3,
    fit_range: tuple[int, int] = (1, 60),
    normalize: str = "trace",
    floor: float = 1e-12,
) -> dict:
    T_arr = np.asarray(T_list, dtype=int)
    max_T = int(np.max(T_arr)) if T_arr.size else 0
    trace_scale = float(np.asarray(kappa_fn(np.array([1.0], dtype=float), par), dtype=float).reshape(-1)[0])
    if normalize == "trace":
        mu_ell = np.asarray(mu_ell_raw, dtype=float) / max(trace_scale, floor)
    elif normalize in ("none", None):
        mu_ell = np.asarray(mu_ell_raw, dtype=float)
    else:
        raise ValueError("normalize must be 'trace' or 'none'.")

    rank_spectrum, multiplicities = expand_degree_spectrum_to_sorted_ranks(mu_ell, p=p, max_rank=max_T)
    _, b_req_exact = b_req_curve_from_spectrum(rank_spectrum, max_rank=max_T, floor=floor)
    b_req_exact_T = np.array(
        [b_req_exact[min(int(T), len(b_req_exact)) - 1] if len(b_req_exact) >= int(T) and int(T) > 0 else np.nan for T in T_arr],
        dtype=float,
    )

    C_degree, s_degree, r2_degree = fit_powerlaw_log(mu_ell, *fit_range)
    beta_rank = float(s_degree / max(p - 1, 1))
    cumulative_coeff = float(2.0 / special.gamma(p))
    C_rank = float(C_degree * (cumulative_coeff ** beta_rank))
    with np.errstate(divide="ignore", invalid="ignore"):
        b_req_powerlaw_T = np.sqrt((np.asarray(T_arr, dtype=float) ** beta_rank) / (max(C_rank, floor) * (beta_rank + 1.0)))

    return {
        "T_list": T_arr,
        "trace_scale": trace_scale,
        "degree_powerlaw_C": C_degree,
        "degree_powerlaw_s": s_degree,
        "degree_powerlaw_r2": r2_degree,
        "rank_powerlaw_C_tilde": C_rank,
        "rank_powerlaw_beta": beta_rank,
        "exact_rank_spectrum": rank_spectrum,
        "multiplicities": multiplicities,
        "b_req_exact_T": b_req_exact_T,
        "b_req_powerlaw_T": np.asarray(b_req_powerlaw_T, dtype=float),
    }


# -----------------------------
# Empirical spectrum diagnostics from saved eigenspectra
# -----------------------------
def _sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return 0.0
    return float(np.std(x, ddof=1) / np.sqrt(x.size))


def _nanmean_or_nan(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.mean(x))


def clean_empirical_eigvals(
    eigvals: np.ndarray,
    clip_negative_to_zero: bool = True,
    sort_descending: bool = True,
) -> np.ndarray:
    eigvals = np.asarray(eigvals, dtype=float).reshape(-1)
    eigvals = eigvals[np.isfinite(eigvals)]
    if clip_negative_to_zero:
        eigvals = np.clip(eigvals, 0.0, None)
    if sort_descending:
        eigvals = np.sort(eigvals)[::-1]
    return eigvals


def normalize_empirical_eigvals(
    eigvals: np.ndarray,
    mode: str = "trace",
    eps: float = 1e-12,
) -> tuple[np.ndarray, float]:
    eigvals = np.asarray(eigvals, dtype=float)
    if eigvals.size == 0:
        return eigvals, 1.0

    if mode == "trace":
        scale = float(np.sum(eigvals) / max(len(eigvals), 1))
    elif mode in ("none", None):
        scale = 1.0
    else:
        raise ValueError("mode must be 'trace' or 'none'.")

    scale = max(scale, eps)
    return eigvals / scale, scale


def spectral_decay_exponent(
    eigvals: np.ndarray,
    rank_min: int = 10,
    rank_max: int = 400,
    floor: float = 1e-12,
) -> tuple[float, float]:
    C_tilde, beta, r2 = fit_rank_powerlaw_log(
        eigvals,
        rank_min=rank_min,
        rank_max=rank_max,
        floor=floor,
    )
    return beta, r2


def fit_rank_powerlaw_log(
    eigvals: np.ndarray,
    rank_min: int = 10,
    rank_max: int = 400,
    floor: float = 1e-12,
) -> tuple[float, float, float]:
    eigvals = np.asarray(eigvals, dtype=float)
    ranks = np.arange(1, len(eigvals) + 1, dtype=float)
    mask = (
        (ranks >= rank_min)
        & (ranks <= min(rank_max, len(eigvals)))
        & (eigvals > floor)
    )
    if np.sum(mask) < 2:
        return np.nan, np.nan

    x = np.log(ranks[mask])
    y = np.log(eigvals[mask])
    slope, intercept = np.polyfit(x, y, 1)
    yhat = intercept + slope * x
    return float(np.exp(intercept)), float(-slope), float(_log_r2(y, yhat))


def effective_dimension_threshold(eigvals: np.ndarray, epsilon: float) -> float:
    eigvals = np.asarray(eigvals, dtype=float)
    return float(np.sum(eigvals > float(epsilon)))


def effective_dimension_ridge(eigvals: np.ndarray, lam: float, eps: float = 1e-12) -> float:
    eigvals = np.asarray(eigvals, dtype=float)
    lam = max(float(lam), eps)
    return float(np.sum(eigvals / (eigvals + lam)))


def b_req_curve_from_spectrum(
    eigvals: np.ndarray,
    max_rank: int | None = None,
    floor: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    eigvals = np.asarray(eigvals, dtype=float)
    if eigvals.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    if max_rank is None:
        max_rank = len(eigvals)
    max_rank = int(min(max_rank, len(eigvals)))

    safe = np.clip(eigvals[:max_rank], floor, None)
    prefix_inv = np.cumsum(1.0 / safe)
    ranks = np.arange(1, max_rank + 1, dtype=int)
    b_req = np.sqrt(prefix_inv / ranks)
    return ranks, b_req


def rkhs_norm_from_eigenbasis(
    eigvals: np.ndarray,
    coeff_energy: np.ndarray,
    floor: float = 1e-12,
) -> float:
    """
    Exact RKHS norm if you know the target energy in the kernel eigenbasis:

        ||f||_H^2 = sum_j coeff_energy[j] / mu_j

    with coeff_energy[j] = |<f, phi_j>|^2.
    """
    eigvals = np.asarray(eigvals, dtype=float)
    coeff_energy = np.asarray(coeff_energy, dtype=float)
    n = min(len(eigvals), len(coeff_energy))
    if n == 0:
        return np.nan
    safe = np.clip(eigvals[:n], floor, None)
    return float(np.sum(coeff_energy[:n] / safe))


def required_rkhs_budget_at_rank(
    eigvals: np.ndarray,
    rank: int,
    floor: float = 1e-12,
) -> float:
    ranks, b_req = b_req_curve_from_spectrum(eigvals, max_rank=rank, floor=floor)
    if ranks.size == 0:
        return np.nan
    return float(b_req[-1])


def sample_multitask_circle_family(
    K_tasks: int,
    pts_per_circle: int,
    m: int,
    seed: int,
    normals_mode: str = "random",
    random_phase: bool = True,
) -> dict:
    """
    Match the multitask circle family used in the spectrum sweeps, but return
    per-task arrays so we can build a composite RKHS target f_multi = sum_a f_a.
    """
    rng = np.random.default_rng(seed)
    if normals_mode == "random":
        normals = sample_random_normals_s2(K_tasks, rng=rng)
    elif normals_mode == "fibonacci":
        normals = normals_fibonacci_s2(K_tasks)
    else:
        raise ValueError("normals_mode must be 'random' or 'fibonacci'")

    phases = rng.uniform(0.0, 2.0 * np.pi, size=K_tasks) if random_phase else np.zeros(K_tasks)
    X_union, y_union, task_slices = generate_multi_tasks_from_normals_phases_with_mode(
        normals=normals,
        phases=phases,
        pts_per_circle=pts_per_circle,
        m=m,
        rng=rng,
        mode="grid",
    )
    X_union = np.asarray(X_union, dtype=float)
    y_union = np.asarray(y_union, dtype=float)
    Xs = [X_union[sl].copy() for sl in task_slices]
    ys = [y_union[sl].copy() for sl in task_slices]
    return {
        "Xs": Xs,
        "ys": ys,
        "normals": np.asarray(normals, dtype=float),
        "phases": np.asarray(phases, dtype=float),
        "X_union": X_union,
        "y_union": y_union,
    }


def great_circle_cosine_distance_stats(normals: np.ndarray) -> dict:
    normals = np.asarray(normals, dtype=float)
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError("normals must have shape (K, 3).")
    K = normals.shape[0]
    if K <= 1:
        return {
            "mean_cosine_distance": np.nan,
            "min_cosine_distance": np.nan,
            "mean_acute_angle_deg": np.nan,
        }

    unit = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    cos_sim = np.clip(np.abs(unit @ unit.T), 0.0, 1.0)
    iu = np.triu_indices(K, k=1)
    pair_cos = cos_sim[iu]
    pair_dist = 1.0 - pair_cos
    pair_angle = np.degrees(np.arccos(pair_cos))
    return {
        "mean_cosine_distance": float(np.mean(pair_dist)),
        "min_cosine_distance": float(np.min(pair_dist)),
        "mean_acute_angle_deg": float(np.mean(pair_angle)),
    }


def solve_min_norm_kernel_system(
    K: np.ndarray,
    y: np.ndarray,
    method: str = "pinv",
    pinv_rtol: float = 1e-10,
    pinv_atol: float = 1e-12,
    ridge_scale: float = 1e-10,
) -> tuple[np.ndarray, float, dict]:
    """
    Solve for RKHS coefficients alpha for the minimum-norm interpolant on a
    discrete task grid.

    `pinv` is exact for the finite-grid problem; `ridge` is a faster estimate.
    """
    K = np.asarray(K, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    K = 0.5 * (K + K.T)

    if method == "pinv":
        evals, evecs = np.linalg.eigh(K)
        lam_max = float(max(np.max(evals), 0.0))
        cutoff = max(float(pinv_atol), float(pinv_rtol) * lam_max)
        keep = evals > cutoff
        coeffs = evecs.T @ y
        alpha = np.zeros_like(y)
        if np.any(keep):
            alpha = evecs[:, keep] @ (coeffs[keep] / evals[keep])
        norm_sq = max(float(y @ alpha), 0.0)
        residual = float(np.linalg.norm(K @ alpha - y) / max(np.linalg.norm(y), pinv_atol))
        info = {
            "method": method,
            "rank": int(np.sum(keep)),
            "cutoff": cutoff,
            "residual": residual,
        }
        return alpha, norm_sq, info

    if method == "ridge":
        n = K.shape[0]
        lam = float(ridge_scale * (np.trace(K) / max(n, 1)) + pinv_atol)
        alpha = np.linalg.solve(K + lam * np.eye(n), y)
        norm_sq = max(float(alpha @ K @ alpha), 0.0)
        residual = float(np.linalg.norm(K @ alpha - y) / max(np.linalg.norm(y), pinv_atol))
        info = {
            "method": method,
            "rank": int(np.linalg.matrix_rank(K)),
            "cutoff": lam,
            "residual": residual,
        }
        return alpha, norm_sq, info

    raise ValueError("method must be 'pinv' or 'ridge'.")


def estimate_composite_target_rkhs_norm(
    Xs: list[np.ndarray],
    ys: list[np.ndarray],
    par: KernelParams,
    which: str,
    kappa_scale: float = 1.0,
    solve_method: str = "pinv",
    pinv_rtol: float = 1e-10,
    pinv_atol: float = 1e-12,
    ridge_scale: float = 1e-10,
) -> dict:
    """
    Estimate the RKHS norm of

        f_multi = sum_a f_a

    where each f_a is the minimum-norm RKHS function that matches task-a labels
    on its own great-circle sample grid.

    This is exact for the discrete sample grid when solve_method='pinv', and a
    consistent finite-grid estimate of the continuum quantity as the circle grid
    is refined.
    """
    X_union = np.vstack(Xs)
    K_diag_ref = np.asarray(
        gram_from_kappa(
            jnp.asarray(X_union[:1], dtype=jnp.float32),
            jnp.asarray(X_union[:1], dtype=jnp.float32),
            par=par,
            which=which,
            kappa_scale=kappa_scale,
        ),
        dtype=float,
    )
    trace_scale = max(float(K_diag_ref[0, 0]), pinv_atol)

    alphas = []
    self_norm_sq = []
    residuals = []
    ranks = []

    for Xa, ya in zip(Xs, ys):
        Kaa = np.asarray(
            gram_from_kappa(
                jnp.asarray(Xa, dtype=jnp.float32),
                jnp.asarray(Xa, dtype=jnp.float32),
                par=par,
                which=which,
                kappa_scale=kappa_scale,
            ),
            dtype=float,
        )
        alpha_a, norm_sq_a, info = solve_min_norm_kernel_system(
            Kaa,
            ya,
            method=solve_method,
            pinv_rtol=pinv_rtol,
            pinv_atol=pinv_atol,
            ridge_scale=ridge_scale,
        )
        alphas.append(alpha_a)
        self_norm_sq.append(norm_sq_a)
        residuals.append(info["residual"])
        ranks.append(info["rank"])

    self_norm_sq = np.asarray(self_norm_sq, dtype=float)
    total_sq = float(np.sum(self_norm_sq))
    cross_terms = []

    for i in range(len(Xs)):
        for j in range(i + 1, len(Xs)):
            Kij = np.asarray(
                gram_from_kappa(
                    jnp.asarray(Xs[i], dtype=jnp.float32),
                    jnp.asarray(Xs[j], dtype=jnp.float32),
                    par=par,
                    which=which,
                    kappa_scale=kappa_scale,
                ),
                dtype=float,
            )
            cross_ij = float(alphas[i] @ Kij @ alphas[j])
            cross_terms.append(cross_ij)
            total_sq += 2.0 * cross_ij

    interaction_sq = float(2.0 * np.sum(cross_terms)) if cross_terms else 0.0
    total_sq = max(total_sq, 0.0)
    normalized_total_sq = trace_scale * total_sq
    normalized_self_sq_sum = trace_scale * float(np.sum(self_norm_sq))
    normalized_interaction_sq = trace_scale * interaction_sq
    normalized_mean_pairwise_inner = trace_scale * (float(np.mean(cross_terms)) if cross_terms else 0.0)
    return {
        "trace_scale": trace_scale,
        "total_norm_sq": total_sq,
        "total_norm": float(np.sqrt(total_sq)),
        "normalized_total_norm_sq": normalized_total_sq,
        "normalized_total_norm": float(np.sqrt(max(normalized_total_sq, 0.0))),
        "self_norm_sq_sum": float(np.sum(self_norm_sq)),
        "normalized_self_norm_sq_sum": normalized_self_sq_sum,
        "interaction_norm_sq": interaction_sq,
        "normalized_interaction_norm_sq": normalized_interaction_sq,
        "mean_single_task_norm_sq": float(np.mean(self_norm_sq)),
        "mean_pairwise_inner": float(np.mean(cross_terms)) if cross_terms else 0.0,
        "normalized_mean_pairwise_inner": normalized_mean_pairwise_inner,
        "max_solver_residual": float(np.max(residuals)) if residuals else 0.0,
        "min_solver_rank": int(np.min(ranks)) if ranks else 0,
    }


def _summarize_seed_matrix(seed_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seed_values = np.asarray(seed_values, dtype=float)
    means = np.array([_nanmean_or_nan(row) for row in seed_values], dtype=float)
    sems = np.array([_sem(row) for row in seed_values], dtype=float)
    return means, sems


def estimate_composite_target_rkhs_norms_by_T(
    T_list,
    n_seeds: int = 20,
    pts_per_circle: int = 128,
    m: int = 3,
    par: KernelParams | None = None,
    kappa_scale: float = 1.0,
    normals_mode: str = "random",
    random_phase: bool = True,
    solve_method: str = "pinv",
    pinv_rtol: float = 1e-10,
    pinv_atol: float = 1e-12,
    ridge_scale: float = 1e-10,
    seed_schedule_multiplier: int = 1000,
    show_progress: bool = True,
) -> dict:
    if par is None:
        par = KernelParams(sigma_w2=1.0, sigma_b2=1.0, sigma_v2=1.0, sigma_bout2=1.0)

    T_arr = np.asarray(T_list, dtype=int)
    kernels = {"bias_only": "bias", "full_ntk": "full"}

    summary = {
        "T_list": T_arr,
        "n_seeds": int(n_seeds),
        "pts_per_circle": int(pts_per_circle),
        "m": int(m),
        "normals_mode": str(normals_mode),
        "random_phase": bool(random_phase),
        "solve_method": str(solve_method),
        "seed_schedule_multiplier": int(seed_schedule_multiplier),
        "kernels": {},
    }

    for kernel_name in kernels:
        summary["kernels"][kernel_name] = {
            "trace_scale_seed": np.full((len(T_arr), n_seeds), np.nan, dtype=float),
            "total_norm_seed": np.full((len(T_arr), n_seeds), np.nan, dtype=float),
            "total_norm_sq_seed": np.full((len(T_arr), n_seeds), np.nan, dtype=float),
            "normalized_total_norm_seed": np.full((len(T_arr), n_seeds), np.nan, dtype=float),
            "normalized_total_norm_sq_seed": np.full((len(T_arr), n_seeds), np.nan, dtype=float),
            "self_norm_sq_sum_seed": np.full((len(T_arr), n_seeds), np.nan, dtype=float),
            "normalized_self_norm_sq_sum_seed": np.full((len(T_arr), n_seeds), np.nan, dtype=float),
            "interaction_norm_sq_seed": np.full((len(T_arr), n_seeds), np.nan, dtype=float),
            "normalized_interaction_norm_sq_seed": np.full((len(T_arr), n_seeds), np.nan, dtype=float),
            "mean_pairwise_inner_seed": np.full((len(T_arr), n_seeds), np.nan, dtype=float),
            "normalized_mean_pairwise_inner_seed": np.full((len(T_arr), n_seeds), np.nan, dtype=float),
            "max_solver_residual_seed": np.full((len(T_arr), n_seeds), np.nan, dtype=float),
        }

    for i, T in enumerate(T_arr):
        if show_progress:
            print(f"Computing composite RKHS norms for T={int(T)} ({i + 1}/{len(T_arr)})")
        for s in range(n_seeds):
            seed = seed_schedule_multiplier * int(T) + int(s)
            task_data = sample_multitask_circle_family(
                K_tasks=int(T),
                pts_per_circle=pts_per_circle,
                m=m,
                seed=seed,
                normals_mode=normals_mode,
                random_phase=random_phase,
            )
            for kernel_name, which in kernels.items():
                out = estimate_composite_target_rkhs_norm(
                    task_data["Xs"],
                    task_data["ys"],
                    par=par,
                    which=which,
                    kappa_scale=kappa_scale,
                    solve_method=solve_method,
                    pinv_rtol=pinv_rtol,
                    pinv_atol=pinv_atol,
                    ridge_scale=ridge_scale,
                )
                kernel_payload = summary["kernels"][kernel_name]
                kernel_payload["trace_scale_seed"][i, s] = out["trace_scale"]
                kernel_payload["total_norm_seed"][i, s] = out["total_norm"]
                kernel_payload["total_norm_sq_seed"][i, s] = out["total_norm_sq"]
                kernel_payload["normalized_total_norm_seed"][i, s] = out["normalized_total_norm"]
                kernel_payload["normalized_total_norm_sq_seed"][i, s] = out["normalized_total_norm_sq"]
                kernel_payload["self_norm_sq_sum_seed"][i, s] = out["self_norm_sq_sum"]
                kernel_payload["normalized_self_norm_sq_sum_seed"][i, s] = out["normalized_self_norm_sq_sum"]
                kernel_payload["interaction_norm_sq_seed"][i, s] = out["interaction_norm_sq"]
                kernel_payload["normalized_interaction_norm_sq_seed"][i, s] = out["normalized_interaction_norm_sq"]
                kernel_payload["mean_pairwise_inner_seed"][i, s] = out["mean_pairwise_inner"]
                kernel_payload["normalized_mean_pairwise_inner_seed"][i, s] = out["normalized_mean_pairwise_inner"]
                kernel_payload["max_solver_residual_seed"][i, s] = out["max_solver_residual"]

    for kernel_name, kernel_payload in summary["kernels"].items():
        for metric in [
            "trace_scale",
            "total_norm",
            "total_norm_sq",
            "normalized_total_norm",
            "normalized_total_norm_sq",
            "self_norm_sq_sum",
            "normalized_self_norm_sq_sum",
            "interaction_norm_sq",
            "normalized_interaction_norm_sq",
            "mean_pairwise_inner",
            "normalized_mean_pairwise_inner",
            "max_solver_residual",
        ]:
            mean, sem = _summarize_seed_matrix(kernel_payload[f"{metric}_seed"])
            kernel_payload[f"{metric}_mean"] = mean
            kernel_payload[f"{metric}_sem"] = sem

    return summary


def save_composite_target_rkhs_summary(summary: dict, out_path: Path | str):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "T_list": np.asarray(summary["T_list"], dtype=int),
        "n_seeds": np.asarray([summary["n_seeds"]], dtype=int),
        "pts_per_circle": np.asarray([summary["pts_per_circle"]], dtype=int),
        "m": np.asarray([summary["m"]], dtype=int),
        "normals_mode": np.asarray([summary["normals_mode"]]),
        "random_phase": np.asarray([summary["random_phase"]], dtype=bool),
        "solve_method": np.asarray([summary["solve_method"]]),
        "seed_schedule_multiplier": np.asarray([summary["seed_schedule_multiplier"]], dtype=int),
    }
    for kernel_name, kernel_payload in summary["kernels"].items():
        for key, value in kernel_payload.items():
            payload[f"{kernel_name}_{key}"] = np.asarray(value)
    np.savez_compressed(out_path, **payload)
    print(f"Wrote: {out_path}")


def plot_composite_target_rkhs_norm_summary(summary: dict, out_path: Path | str, logy: bool = True):
    _require_matplotlib()
    out_path = Path(out_path)
    colors = {"bias_only": "tab:blue", "full_ntk": "tab:orange"}
    labels = {"bias_only": "bias-only", "full_ntk": "full NTK"}

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4), sharex=True)
    ax_norm_raw, ax_sq_raw = axes[0]
    ax_norm_normed, ax_sq_normed = axes[1]

    for kernel_name, payload in summary["kernels"].items():
        T_list = summary["T_list"]
        color = colors.get(kernel_name)
        label = labels.get(kernel_name, kernel_name)

        _plot_mean_sem(
            ax_norm_raw,
            T_list,
            payload["total_norm_mean"],
            payload["total_norm_sem"],
            label=label,
            color=color,
        )
        _plot_mean_sem(
            ax_norm_normed,
            T_list,
            payload["normalized_total_norm_mean"],
            payload["normalized_total_norm_sem"],
            label=label,
            color=color,
        )

        _plot_mean_sem(
            ax_sq_raw,
            T_list,
            payload["total_norm_sq_mean"],
            payload["total_norm_sq_sem"],
            label=f"{label}: total",
            color=color,
        )
        ax_sq_raw.plot(
            T_list,
            payload["self_norm_sq_sum_mean"],
            linestyle="--",
            linewidth=2,
            color=color,
            label=f"{label}: sum of self norms",
        )
        _plot_mean_sem(
            ax_sq_normed,
            T_list,
            payload["normalized_total_norm_sq_mean"],
            payload["normalized_total_norm_sq_sem"],
            label=f"{label}: total",
            color=color,
        )
        ax_sq_normed.plot(
            T_list,
            payload["normalized_self_norm_sq_sum_mean"],
            linestyle="--",
            linewidth=2,
            color=color,
            label=f"{label}: sum of self norms",
        )

    ax_norm_raw.set_title(r"Raw composite norm $\|f_{\mathrm{multi}}\|_{\mathcal{H}_K}$")
    ax_norm_raw.set_ylabel(r"$\|f_{\mathrm{multi}}\|_{\mathcal{H}_K}$")
    if logy:
        ax_norm_raw.set_yscale("log")

    ax_sq_raw.set_title(r"Raw norm-squared decomposition")
    ax_sq_raw.set_ylabel(r"$\|f_{\mathrm{multi}}\|_{\mathcal{H}_K}^2$")
    if logy:
        ax_sq_raw.set_yscale("log")

    ax_norm_normed.set_title(r"Trace-normalized norm $\|f_{\mathrm{multi}}\|_{\mathcal{H}_{\tilde K}}$")
    ax_norm_normed.set_ylabel(r"$\|f_{\mathrm{multi}}\|_{\mathcal{H}_{\tilde K}}$")
    if logy:
        ax_norm_normed.set_yscale("log")

    ax_sq_normed.set_title(r"Trace-normalized norm-squared decomposition")
    ax_sq_normed.set_ylabel(r"$\|f_{\mathrm{multi}}\|_{\mathcal{H}_{\tilde K}}^2$")
    if logy:
        ax_sq_normed.set_yscale("log")

    for ax in axes.ravel():
        ax.set_xlabel("Number of tasks T")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}")


def make_composite_target_rkhs_plots(
    out_dir: Path | str,
    stem: str,
    T_list,
    n_seeds: int,
    pts_per_circle: int,
    m: int,
    par: KernelParams,
    kappa_scale: float = 1.0,
    normals_mode: str = "random",
    random_phase: bool = True,
    solve_method: str = "pinv",
    pinv_rtol: float = 1e-10,
    pinv_atol: float = 1e-12,
    ridge_scale: float = 1e-10,
    show_progress: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = estimate_composite_target_rkhs_norms_by_T(
        T_list=T_list,
        n_seeds=n_seeds,
        pts_per_circle=pts_per_circle,
        m=m,
        par=par,
        kappa_scale=kappa_scale,
        normals_mode=normals_mode,
        random_phase=random_phase,
        solve_method=solve_method,
        pinv_rtol=pinv_rtol,
        pinv_atol=pinv_atol,
        ridge_scale=ridge_scale,
        show_progress=show_progress,
    )

    npz_path = out_dir / f"composite_target_rkhs_norms_{stem}.npz"
    fig_path = out_dir / f"composite_target_rkhs_norms_{stem}.png"
    save_composite_target_rkhs_summary(summary, npz_path)
    if plt is not None:
        plot_composite_target_rkhs_norm_summary(summary, fig_path)
    return summary


def sample_multitask_circle_family_with_test(
    K_tasks: int,
    pts_per_circle: int,
    m: int,
    seed: int,
    test_pts_per_circle: int | None = None,
    normals_mode: str = "random",
    random_phase: bool = True,
) -> dict:
    train = sample_multitask_circle_family(
        K_tasks=K_tasks,
        pts_per_circle=pts_per_circle,
        m=m,
        seed=seed,
        normals_mode=normals_mode,
        random_phase=random_phase,
    )
    if test_pts_per_circle is None:
        test_pts_per_circle = 2 * pts_per_circle

    rng_test = np.random.default_rng(seed + 10_000_001)
    X_union_test, y_union_test, test_slices = generate_multi_tasks_from_normals_phases_with_mode(
        normals=train["normals"],
        phases=train["phases"],
        pts_per_circle=test_pts_per_circle,
        m=m,
        rng=rng_test,
        mode="random",
    )
    X_union_test = np.asarray(X_union_test, dtype=float)
    y_union_test = np.asarray(y_union_test, dtype=float)
    train["Xs_test"] = [X_union_test[sl].copy() for sl in test_slices]
    train["ys_test"] = [y_union_test[sl].copy() for sl in test_slices]
    return train


def solve_trace_normalized_ridge_system(
    K_train: np.ndarray,
    y_train: np.ndarray,
    reg_scale: float,
    eps: float = 1e-12,
) -> tuple[np.ndarray, float]:
    K_train = np.asarray(K_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    n = K_train.shape[0]
    s_trace = float(np.trace(K_train) / max(n, 1))
    s_trace = max(s_trace, eps)
    alpha = np.linalg.solve(K_train / s_trace + (n * reg_scale) * np.eye(n), y_train)
    return alpha, s_trace


def predict_trace_normalized_ridge(
    K_eval_train: np.ndarray,
    alpha: np.ndarray,
    s_trace: float,
) -> np.ndarray:
    return (np.asarray(K_eval_train, dtype=float) / float(s_trace)) @ np.asarray(alpha, dtype=float)


def _all_task_pairs(T: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(T) for j in range(i + 1, T)]


def summarize_tuned_task_coupling_for_family(
    Xs: list[np.ndarray],
    ys: list[np.ndarray],
    Xs_test: list[np.ndarray],
    ys_test: list[np.ndarray],
    par: KernelParams,
    which: str,
    reg_scale: float,
    kappa_scale: float = 1.0,
    pair_indices: list[tuple[int, int]] | None = None,
) -> dict:
    T = len(Xs)
    solo_alphas = []
    solo_s_trace = []
    solo_norm_sq = []
    solo_mse = []

    K_self = []
    K_test_self = []

    for Xa, ya, Xta, yta in zip(Xs, ys, Xs_test, ys_test):
        Kaa = np.asarray(
            gram_from_kappa(
                jnp.asarray(Xa, dtype=jnp.float32),
                jnp.asarray(Xa, dtype=jnp.float32),
                par=par,
                which=which,
                kappa_scale=kappa_scale,
            ),
            dtype=float,
        )
        Kta = np.asarray(
            gram_from_kappa(
                jnp.asarray(Xta, dtype=jnp.float32),
                jnp.asarray(Xa, dtype=jnp.float32),
                par=par,
                which=which,
                kappa_scale=kappa_scale,
            ),
            dtype=float,
        )
        alpha_a, s_trace_a = solve_trace_normalized_ridge_system(Kaa, ya, reg_scale)
        pred_a = predict_trace_normalized_ridge(Kta, alpha_a, s_trace_a)

        K_self.append(Kaa)
        K_test_self.append(Kta)
        solo_alphas.append(alpha_a)
        solo_s_trace.append(s_trace_a)
        solo_norm_sq.append(max(float(alpha_a @ Kaa @ alpha_a), 0.0))
        solo_mse.append(float(np.mean((pred_a - np.asarray(yta, dtype=float)) ** 2)))

    all_pairs = _all_task_pairs(T)
    cos_vals = []
    for i, j in all_pairs:
        Kij = np.asarray(
            gram_from_kappa(
                jnp.asarray(Xs[i], dtype=jnp.float32),
                jnp.asarray(Xs[j], dtype=jnp.float32),
                par=par,
                which=which,
                kappa_scale=kappa_scale,
            ),
            dtype=float,
        )
        denom = np.sqrt(max(solo_norm_sq[i] * solo_norm_sq[j], 1e-12))
        cos_vals.append(float((solo_alphas[i] @ Kij @ solo_alphas[j]) / denom))
    cos_vals = np.asarray(cos_vals, dtype=float) if cos_vals else np.array([], dtype=float)

    if pair_indices is None:
        pair_indices = all_pairs

    pair_penalties = []
    for i, j in pair_indices:
        X_joint = np.vstack([Xs[i], Xs[j]])
        y_joint = np.concatenate([ys[i], ys[j]])
        K_joint = np.asarray(
            gram_from_kappa(
                jnp.asarray(X_joint, dtype=jnp.float32),
                jnp.asarray(X_joint, dtype=jnp.float32),
                par=par,
                which=which,
                kappa_scale=kappa_scale,
            ),
            dtype=float,
        )
        alpha_joint, s_trace_joint = solve_trace_normalized_ridge_system(K_joint, y_joint, reg_scale)

        for task_idx in (i, j):
            K_eval_joint = np.asarray(
                gram_from_kappa(
                    jnp.asarray(Xs_test[task_idx], dtype=jnp.float32),
                    jnp.asarray(X_joint, dtype=jnp.float32),
                    par=par,
                    which=which,
                    kappa_scale=kappa_scale,
                ),
                dtype=float,
            )
            pred_joint = predict_trace_normalized_ridge(K_eval_joint, alpha_joint, s_trace_joint)
            joint_mse = float(np.mean((pred_joint - np.asarray(ys_test[task_idx], dtype=float)) ** 2))
            pair_penalties.append(joint_mse - solo_mse[task_idx])

    pair_penalties = np.asarray(pair_penalties, dtype=float) if pair_penalties else np.array([], dtype=float)
    return {
        "mean_abs_rkhs_cosine": float(np.mean(np.abs(cos_vals))) if cos_vals.size else 0.0,
        "rms_rkhs_cosine": float(np.sqrt(np.mean(cos_vals ** 2))) if cos_vals.size else 0.0,
        "max_abs_rkhs_cosine": float(np.max(np.abs(cos_vals))) if cos_vals.size else 0.0,
        "mean_pair_interference": float(np.mean(pair_penalties)) if pair_penalties.size else 0.0,
        "median_pair_interference": float(np.median(pair_penalties)) if pair_penalties.size else 0.0,
        "frac_positive_pair_interference": float(np.mean(pair_penalties > 0.0)) if pair_penalties.size else 0.0,
    }


def estimate_tuned_multitask_coupling_by_T(
    T_list,
    reg_schedule: dict,
    n_seeds: int,
    pts_per_circle: int,
    cntrl_pts_per_task: bool,
    m: int,
    par: KernelParams,
    kappa_scale: float = 1.0,
    normals_mode: str = "random",
    random_phase: bool = True,
    test_pts_per_circle: int | None = None,
    max_pairs_per_seed: int = 120,
    seed_schedule_multiplier: int = 1000,
    show_progress: bool = True,
) -> dict:
    T_arr = np.asarray(T_list, dtype=int)
    max_T = int(np.max(T_arr))
    kernels = {"bias_only": "bias", "full_ntk": "full"}

    summary = {
        "T_list": T_arr,
        "n_seeds": int(n_seeds),
        "pts_per_circle": int(pts_per_circle),
        "cntrl_pts_per_task": bool(cntrl_pts_per_task),
        "m": int(m),
        "normals_mode": str(normals_mode),
        "random_phase": bool(random_phase),
        "max_pairs_per_seed": int(max_pairs_per_seed),
        "kernels": {},
    }

    for kernel_name in kernels:
        summary["kernels"][kernel_name] = {
            "mean_abs_rkhs_cosine_seed": np.full((len(T_arr), n_seeds), np.nan, dtype=float),
            "rms_rkhs_cosine_seed": np.full((len(T_arr), n_seeds), np.nan, dtype=float),
            "max_abs_rkhs_cosine_seed": np.full((len(T_arr), n_seeds), np.nan, dtype=float),
            "mean_pair_interference_seed": np.full((len(T_arr), n_seeds), np.nan, dtype=float),
            "median_pair_interference_seed": np.full((len(T_arr), n_seeds), np.nan, dtype=float),
            "frac_positive_pair_interference_seed": np.full((len(T_arr), n_seeds), np.nan, dtype=float),
        }

    for i, T in enumerate(T_arr):
        if cntrl_pts_per_task:
            N_tot = max_T * pts_per_circle
            pts_per_circle_T = max(16, N_tot // int(T))
        else:
            pts_per_circle_T = pts_per_circle
        test_pts_per_circle_T = 2 * pts_per_circle_T if test_pts_per_circle is None else int(test_pts_per_circle)

        if show_progress:
            print(f"Computing tuned multitask coupling for T={int(T)} ({i + 1}/{len(T_arr)})")

        for s in range(n_seeds):
            seed = seed_schedule_multiplier * int(T) + int(s)
            task_data = sample_multitask_circle_family_with_test(
                K_tasks=int(T),
                pts_per_circle=pts_per_circle_T,
                test_pts_per_circle=test_pts_per_circle_T,
                m=m,
                seed=seed,
                normals_mode=normals_mode,
                random_phase=random_phase,
            )

            all_pairs = _all_task_pairs(int(T))
            if max_pairs_per_seed > 0 and len(all_pairs) > max_pairs_per_seed:
                rng_pairs = np.random.default_rng(seed + 40_000_000)
                pair_ids = rng_pairs.choice(len(all_pairs), size=max_pairs_per_seed, replace=False)
                pair_subset = [all_pairs[int(pid)] for pid in pair_ids]
            else:
                pair_subset = all_pairs

            for kernel_name, which in kernels.items():
                reg_vals = reg_schedule[kernel_name]
                reg_scale = float(reg_vals[i] if isinstance(reg_vals, list) else reg_vals)
                out = summarize_tuned_task_coupling_for_family(
                    Xs=task_data["Xs"],
                    ys=task_data["ys"],
                    Xs_test=task_data["Xs_test"],
                    ys_test=task_data["ys_test"],
                    par=par,
                    which=which,
                    reg_scale=reg_scale,
                    kappa_scale=kappa_scale,
                    pair_indices=pair_subset,
                )
                for key, value in out.items():
                    summary["kernels"][kernel_name][f"{key}_seed"][i, s] = value

    for kernel_name, kernel_payload in summary["kernels"].items():
        for metric in [
            "mean_abs_rkhs_cosine",
            "rms_rkhs_cosine",
            "max_abs_rkhs_cosine",
            "mean_pair_interference",
            "median_pair_interference",
            "frac_positive_pair_interference",
        ]:
            mean, sem = _summarize_seed_matrix(kernel_payload[f"{metric}_seed"])
            kernel_payload[f"{metric}_mean"] = mean
            kernel_payload[f"{metric}_sem"] = sem

    return summary


def save_tuned_multitask_coupling_summary(summary: dict, out_path: Path | str):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "T_list": np.asarray(summary["T_list"], dtype=int),
        "n_seeds": np.asarray([summary["n_seeds"]], dtype=int),
        "pts_per_circle": np.asarray([summary["pts_per_circle"]], dtype=int),
        "cntrl_pts_per_task": np.asarray([summary["cntrl_pts_per_task"]], dtype=bool),
        "m": np.asarray([summary["m"]], dtype=int),
        "normals_mode": np.asarray([summary["normals_mode"]]),
        "random_phase": np.asarray([summary["random_phase"]], dtype=bool),
        "max_pairs_per_seed": np.asarray([summary["max_pairs_per_seed"]], dtype=int),
    }
    for kernel_name, kernel_payload in summary["kernels"].items():
        for key, value in kernel_payload.items():
            payload[f"{kernel_name}_{key}"] = np.asarray(value)
    np.savez_compressed(out_path, **payload)
    print(f"Wrote: {out_path}")


def plot_tuned_multitask_coupling_summary(summary: dict, out_path: Path | str):
    _require_matplotlib()
    out_path = Path(out_path)
    colors = {"bias_only": "tab:blue", "full_ntk": "tab:orange"}
    labels = {"bias_only": "bias-only", "full_ntk": "full NTK"}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    ax_cos, ax_int = axes

    for kernel_name, payload in summary["kernels"].items():
        T_list = summary["T_list"]
        color = colors.get(kernel_name)
        label = labels.get(kernel_name, kernel_name)

        _plot_mean_sem(
            ax_cos,
            T_list,
            payload["mean_abs_rkhs_cosine_mean"],
            payload["mean_abs_rkhs_cosine_sem"],
            label=label,
            color=color,
        )
        _plot_mean_sem(
            ax_int,
            T_list,
            payload["mean_pair_interference_mean"],
            payload["mean_pair_interference_sem"],
            label=label,
            color=color,
        )

    ax_cos.set_title(r"Tuned single-task RKHS coupling: mean $|\cos(g_i, g_j)|$")
    ax_cos.set_xlabel("Number of tasks T")
    ax_cos.set_ylabel(r"Mean absolute RKHS cosine")

    ax_int.set_title(r"Joint-training interference: mean $\Delta \mathrm{MSE}$")
    ax_int.set_xlabel("Number of tasks T")
    ax_int.set_ylabel(r"$\mathrm{MSE}_{\mathrm{joint}} - \mathrm{MSE}_{\mathrm{solo}}$")

    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}")


def make_tuned_multitask_coupling_plots(
    out_dir: Path | str,
    stem: str,
    T_list,
    reg_schedule: dict,
    n_seeds: int,
    pts_per_circle: int,
    cntrl_pts_per_task: bool,
    m: int,
    par: KernelParams,
    kappa_scale: float = 1.0,
    normals_mode: str = "random",
    random_phase: bool = True,
    test_pts_per_circle: int | None = None,
    max_pairs_per_seed: int = 120,
    show_progress: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = estimate_tuned_multitask_coupling_by_T(
        T_list=T_list,
        reg_schedule=reg_schedule,
        n_seeds=n_seeds,
        pts_per_circle=pts_per_circle,
        cntrl_pts_per_task=cntrl_pts_per_task,
        m=m,
        par=par,
        kappa_scale=kappa_scale,
        normals_mode=normals_mode,
        random_phase=random_phase,
        test_pts_per_circle=test_pts_per_circle,
        max_pairs_per_seed=max_pairs_per_seed,
        show_progress=show_progress,
    )

    npz_path = out_dir / f"tuned_multitask_coupling_{stem}.npz"
    fig_path = out_dir / f"tuned_multitask_coupling_{stem}.png"
    save_tuned_multitask_coupling_summary(summary, npz_path)
    if plt is not None:
        plot_tuned_multitask_coupling_summary(summary, fig_path)
    else:
        print("Skipping tuned multitask coupling plot rendering because matplotlib is unavailable.")
    return summary


def _ridge_threshold_from_reg(
    reg_scale: float,
    n_samples: int,
    spectrum_scale: float,
    normalize: str,
) -> float:
    reg_scale = float(reg_scale)
    if normalize == "trace":
        return float(n_samples * reg_scale)
    if normalize in ("none", None):
        return float(n_samples * reg_scale * spectrum_scale)
    raise ValueError("normalize must be 'trace' or 'none'.")


def summarize_saved_eigenspectra(
    npz_path: Path | str,
    meta_path: Path | str | None = None,
    normalize: str = "trace",
    fit_rank_window: tuple[int, int] = (10, 400),
    epsilon_list: tuple[float, ...] = (10.0, 1.0, 0.1, 0.01),
    lambda_grid: tuple[float, ...] = (10.0, 1.0, 0.1, 0.01),
    include_analytic_budget_reference: bool = False,
    analytic_p: int = 3,
    analytic_Lmax: int = 60,
    analytic_n_quad: int = 500,
    analytic_fit_range: tuple[int, int] = (1, 60),
    head_mass_rank: int = 10,
    floor: float = 1e-12,
) -> dict:
    npz_path = Path(npz_path)
    meta = {}
    if meta_path is not None:
        meta_path = Path(meta_path)
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)

    D = np.load(npz_path, allow_pickle=True)
    summary = {
        "npz_path": str(npz_path),
        "meta_path": str(meta_path) if meta_path is not None else None,
        "normalize": normalize,
        "fit_rank_window": tuple(int(v) for v in fit_rank_window),
        "epsilon_list": np.asarray(epsilon_list, dtype=float),
        "lambda_grid": np.asarray(lambda_grid, dtype=float),
        "head_mass_rank": int(head_mass_rank),
        "analytic_budget_reference": {},
        "kernels": {},
    }

    if include_analytic_budget_reference:
        max_T = int(max((max(np.asarray(D[k].item()["T_list"], dtype=int)) for k in D.files), default=0))
        target_Lmax = int(
            max(
                analytic_Lmax,
                np.ceil((max(max_T, 1) / max(2.0 / special.gamma(analytic_p), 1e-12)) ** (1.0 / max(analytic_p - 1, 1))) + 5,
                analytic_fit_range[1],
            )
        )
        W_std = float(meta.get("W_std", 1.0))
        b_std = float(meta.get("b_std", 1.0))
        analytic_par = KernelParams(sigma_w2=W_std**2, sigma_b2=b_std**2, sigma_v2=1.0, sigma_bout2=1.0)
        analytic_mu_by_kernel = {
            "bias_only": compute_spectrum_quiet(
                Lmax=target_Lmax,
                kappa_fn=kappa_bias_only,
                par=analytic_par,
                n_quad=analytic_n_quad,
                p=analytic_p,
            ),
            "full_ntk": compute_spectrum_quiet(
                Lmax=target_Lmax,
                kappa_fn=kappa_full_ntk,
                par=analytic_par,
                n_quad=analytic_n_quad,
                p=analytic_p,
            ),
        }
    else:
        analytic_mu_by_kernel = {}

    for kernel_name in D.files:
        payload = D[kernel_name].item()
        T_list = np.asarray(payload["T_list"], dtype=int)
        eigvals = np.asarray(payload["eigvals"], dtype=float)
        reg_schedule = np.asarray(meta.get("reg_scale", {}).get(kernel_name, []), dtype=float)
        has_ridge_schedule = reg_schedule.size == len(T_list)

        decay_mean = np.full(len(T_list), np.nan, dtype=float)
        decay_sem = np.full(len(T_list), np.nan, dtype=float)
        decay_r2_mean = np.full(len(T_list), np.nan, dtype=float)
        rank_powerlaw_C_mean = np.full(len(T_list), np.nan, dtype=float)
        rank_powerlaw_C_sem = np.full(len(T_list), np.nan, dtype=float)
        powerlaw_budget_pred_mean = np.full(len(T_list), np.nan, dtype=float)
        powerlaw_budget_pred_sem = np.full(len(T_list), np.nan, dtype=float)
        powerlaw_budget_correction_mean = np.full(len(T_list), np.nan, dtype=float)
        powerlaw_budget_correction_sem = np.full(len(T_list), np.nan, dtype=float)
        head_mass_empirical_mean = np.full(len(T_list), np.nan, dtype=float)
        head_mass_empirical_sem = np.full(len(T_list), np.nan, dtype=float)
        head_mass_powerlaw_mean = np.full(len(T_list), np.nan, dtype=float)
        head_mass_powerlaw_sem = np.full(len(T_list), np.nan, dtype=float)
        rkhs_budget_mean = np.full(len(T_list), np.nan, dtype=float)
        rkhs_budget_sem = np.full(len(T_list), np.nan, dtype=float)
        rkhs_budget_raw_mean = np.full(len(T_list), np.nan, dtype=float)
        rkhs_budget_raw_sem = np.full(len(T_list), np.nan, dtype=float)
        rkhs_budget_norm_mean = np.full(len(T_list), np.nan, dtype=float)
        rkhs_budget_norm_sem = np.full(len(T_list), np.nan, dtype=float)
        s_trace_mean = np.full(len(T_list), np.nan, dtype=float)
        s_trace_sem = np.full(len(T_list), np.nan, dtype=float)
        ridge_lambda_mean = np.full(len(T_list), np.nan, dtype=float)
        ridge_lambda_sem = np.full(len(T_list), np.nan, dtype=float)
        ridge_lambda_raw_mean = np.full(len(T_list), np.nan, dtype=float)
        ridge_lambda_raw_sem = np.full(len(T_list), np.nan, dtype=float)
        ridge_deff_mean = np.full(len(T_list), np.nan, dtype=float)
        ridge_deff_sem = np.full(len(T_list), np.nan, dtype=float)
        ridge_count_mean = np.full(len(T_list), np.nan, dtype=float)
        ridge_count_sem = np.full(len(T_list), np.nan, dtype=float)
        fixed_deff_mean = np.full((len(lambda_grid), len(T_list)), np.nan, dtype=float)
        fixed_deff_sem = np.full((len(lambda_grid), len(T_list)), np.nan, dtype=float)
        fixed_count_mean = np.full((len(epsilon_list), len(T_list)), np.nan, dtype=float)
        fixed_count_sem = np.full((len(epsilon_list), len(T_list)), np.nan, dtype=float)

        for i, T in enumerate(T_list):
            per_seed_decay = []
            per_seed_r2 = []
            per_seed_C = []
            per_seed_powerlaw_budget_pred = []
            per_seed_powerlaw_budget_correction = []
            per_seed_head_mass_empirical = []
            per_seed_head_mass_powerlaw = []
            per_seed_budget = []
            per_seed_budget_raw = []
            per_seed_budget_norm = []
            per_seed_s_trace = []
            per_seed_ridge_lambda = []
            per_seed_ridge_lambda_raw = []
            per_seed_ridge_deff = []
            per_seed_ridge_count = []
            per_seed_fixed_deff = [[] for _ in lambda_grid]
            per_seed_fixed_count = [[] for _ in epsilon_list]

            for seed_idx in range(eigvals.shape[1]):
                lam_raw = clean_empirical_eigvals(eigvals[i, seed_idx])
                if lam_raw.size == 0:
                    continue

                raw_scale = max(float(np.mean(lam_raw)), floor)
                per_seed_s_trace.append(raw_scale)
                lam, scale = normalize_empirical_eigvals(lam_raw, mode=normalize)
                pred_budget_val = np.nan
                C_tilde, beta, r2 = fit_rank_powerlaw_log(
                    lam,
                    rank_min=fit_rank_window[0],
                    rank_max=fit_rank_window[1],
                    floor=floor,
                )
                if np.isfinite(C_tilde) and np.isfinite(beta):
                    rank = min(int(T), len(lam))
                    ranks = np.arange(1, rank + 1, dtype=float)
                    pred_budget = np.sqrt(np.sum(ranks ** beta) / (max(C_tilde, floor) * max(rank, 1)))
                    pred_budget_val = float(pred_budget)
                    per_seed_C.append(C_tilde)
                    per_seed_powerlaw_budget_pred.append(pred_budget_val)
                    head_k = min(int(head_mass_rank), len(lam))
                    if head_k > 0 and np.sum(lam) > floor:
                        per_seed_head_mass_empirical.append(float(np.sum(lam[:head_k]) / np.sum(lam)))
                        all_ranks = np.arange(1, len(lam) + 1, dtype=float)
                        pred_spec = max(C_tilde, floor) * (all_ranks ** (-beta))
                        per_seed_head_mass_powerlaw.append(float(np.sum(pred_spec[:head_k]) / np.sum(pred_spec)))
                per_seed_decay.append(beta)
                per_seed_r2.append(r2)
                per_seed_budget_raw.append(
                    required_rkhs_budget_at_rank(lam_raw, rank=min(int(T), len(lam_raw)), floor=floor)
                )
                per_seed_budget_norm.append(
                    required_rkhs_budget_at_rank(lam, rank=min(int(T), len(lam)), floor=floor)
                )
                per_seed_budget.append(per_seed_budget_norm[-1] if normalize == "trace" else per_seed_budget_raw[-1])
                if np.isfinite(pred_budget_val) and pred_budget_val > floor:
                    per_seed_powerlaw_budget_correction.append(float(per_seed_budget[-1] / pred_budget_val))

                for j, lam_grid in enumerate(lambda_grid):
                    per_seed_fixed_deff[j].append(effective_dimension_ridge(lam, lam_grid, eps=floor))
                for j, eps_grid in enumerate(epsilon_list):
                    per_seed_fixed_count[j].append(effective_dimension_threshold(lam, eps_grid))

                if has_ridge_schedule:
                    ridge_level = _ridge_threshold_from_reg(
                        reg_scale=reg_schedule[i],
                        n_samples=len(lam_raw),
                        spectrum_scale=scale,
                        normalize=normalize,
                    )
                    ridge_level_raw = _ridge_threshold_from_reg(
                        reg_scale=reg_schedule[i],
                        n_samples=len(lam_raw),
                        spectrum_scale=raw_scale,
                        normalize="none",
                    )
                    per_seed_ridge_lambda.append(ridge_level)
                    per_seed_ridge_lambda_raw.append(ridge_level_raw)
                    per_seed_ridge_deff.append(effective_dimension_ridge(lam, ridge_level, eps=floor))
                    per_seed_ridge_count.append(effective_dimension_threshold(lam, ridge_level))

            decay_mean[i] = _nanmean_or_nan(per_seed_decay)
            decay_sem[i] = _sem(per_seed_decay)
            decay_r2_mean[i] = _nanmean_or_nan(per_seed_r2)
            rank_powerlaw_C_mean[i] = _nanmean_or_nan(per_seed_C)
            rank_powerlaw_C_sem[i] = _sem(per_seed_C)
            powerlaw_budget_pred_mean[i] = _nanmean_or_nan(per_seed_powerlaw_budget_pred)
            powerlaw_budget_pred_sem[i] = _sem(per_seed_powerlaw_budget_pred)
            powerlaw_budget_correction_mean[i] = _nanmean_or_nan(per_seed_powerlaw_budget_correction)
            powerlaw_budget_correction_sem[i] = _sem(per_seed_powerlaw_budget_correction)
            head_mass_empirical_mean[i] = _nanmean_or_nan(per_seed_head_mass_empirical)
            head_mass_empirical_sem[i] = _sem(per_seed_head_mass_empirical)
            head_mass_powerlaw_mean[i] = _nanmean_or_nan(per_seed_head_mass_powerlaw)
            head_mass_powerlaw_sem[i] = _sem(per_seed_head_mass_powerlaw)
            rkhs_budget_mean[i] = _nanmean_or_nan(per_seed_budget)
            rkhs_budget_sem[i] = _sem(per_seed_budget)
            rkhs_budget_raw_mean[i] = _nanmean_or_nan(per_seed_budget_raw)
            rkhs_budget_raw_sem[i] = _sem(per_seed_budget_raw)
            rkhs_budget_norm_mean[i] = _nanmean_or_nan(per_seed_budget_norm)
            rkhs_budget_norm_sem[i] = _sem(per_seed_budget_norm)
            s_trace_mean[i] = _nanmean_or_nan(per_seed_s_trace)
            s_trace_sem[i] = _sem(per_seed_s_trace)

            for j in range(len(lambda_grid)):
                fixed_deff_mean[j, i] = _nanmean_or_nan(per_seed_fixed_deff[j])
                fixed_deff_sem[j, i] = _sem(per_seed_fixed_deff[j])
            for j in range(len(epsilon_list)):
                fixed_count_mean[j, i] = _nanmean_or_nan(per_seed_fixed_count[j])
                fixed_count_sem[j, i] = _sem(per_seed_fixed_count[j])

            if has_ridge_schedule:
                ridge_lambda_mean[i] = _nanmean_or_nan(per_seed_ridge_lambda)
                ridge_lambda_sem[i] = _sem(per_seed_ridge_lambda)
                ridge_lambda_raw_mean[i] = _nanmean_or_nan(per_seed_ridge_lambda_raw)
                ridge_lambda_raw_sem[i] = _sem(per_seed_ridge_lambda_raw)
                ridge_deff_mean[i] = _nanmean_or_nan(per_seed_ridge_deff)
                ridge_deff_sem[i] = _sem(per_seed_ridge_deff)
                ridge_count_mean[i] = _nanmean_or_nan(per_seed_ridge_count)
                ridge_count_sem[i] = _sem(per_seed_ridge_count)

        summary["kernels"][kernel_name] = {
            "T_list": T_list,
            "decay_exponent_mean": decay_mean,
            "decay_exponent_sem": decay_sem,
            "decay_r2_mean": decay_r2_mean,
            "rank_powerlaw_C_mean": rank_powerlaw_C_mean,
            "rank_powerlaw_C_sem": rank_powerlaw_C_sem,
            "powerlaw_budget_pred_mean": powerlaw_budget_pred_mean,
            "powerlaw_budget_pred_sem": powerlaw_budget_pred_sem,
            "powerlaw_budget_correction_mean": powerlaw_budget_correction_mean,
            "powerlaw_budget_correction_sem": powerlaw_budget_correction_sem,
            "head_mass_empirical_mean": head_mass_empirical_mean,
            "head_mass_empirical_sem": head_mass_empirical_sem,
            "head_mass_powerlaw_mean": head_mass_powerlaw_mean,
            "head_mass_powerlaw_sem": head_mass_powerlaw_sem,
            "rkhs_budget_mean": rkhs_budget_mean,
            "rkhs_budget_sem": rkhs_budget_sem,
            "rkhs_budget_raw_mean": rkhs_budget_raw_mean,
            "rkhs_budget_raw_sem": rkhs_budget_raw_sem,
            "rkhs_budget_norm_mean": rkhs_budget_norm_mean,
            "rkhs_budget_norm_sem": rkhs_budget_norm_sem,
            "s_trace_mean": s_trace_mean,
            "s_trace_sem": s_trace_sem,
            "ridge_lambda_mean": ridge_lambda_mean,
            "ridge_lambda_sem": ridge_lambda_sem,
            "ridge_lambda_raw_mean": ridge_lambda_raw_mean,
            "ridge_lambda_raw_sem": ridge_lambda_raw_sem,
            "ridge_deff_mean": ridge_deff_mean,
            "ridge_deff_sem": ridge_deff_sem,
            "ridge_count_mean": ridge_count_mean,
            "ridge_count_sem": ridge_count_sem,
            "fixed_deff_mean": fixed_deff_mean,
            "fixed_deff_sem": fixed_deff_sem,
            "fixed_count_mean": fixed_count_mean,
            "fixed_count_sem": fixed_count_sem,
        }

        if kernel_name in analytic_mu_by_kernel:
            kappa_fn = kappa_bias_only if kernel_name == "bias_only" else kappa_full_ntk
            summary["analytic_budget_reference"][kernel_name] = analytic_budget_reference_from_degree_spectrum(
                T_list=T_list,
                mu_ell_raw=analytic_mu_by_kernel[kernel_name],
                kappa_fn=kappa_fn,
                par=analytic_par,
                p=analytic_p,
                fit_range=analytic_fit_range,
                normalize=normalize,
                floor=floor,
            )

    D.close()
    return summary


def _plot_mean_sem(ax, x, mean, sem, label, color, linestyle="-", alpha_fill=0.15):
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    sem = np.asarray(sem, dtype=float)
    if not np.any(np.isfinite(mean)):
        return
    ax.plot(x, mean, marker="o", linewidth=2, markersize=4, color=color, linestyle=linestyle, label=label)
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=alpha_fill)


def plot_spectrum_companion_summary(summary: dict, out_path: Path | str):
    _require_matplotlib()
    out_path = Path(out_path)
    colors = {"bias_only": "tab:blue", "full_ntk": "tab:orange"}
    labels = {"bias_only": "bias-only", "full_ntk": "full NTK"}

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=True)
    ax_decay, ax_deff, ax_count, ax_budget = axes.ravel()

    for kernel_name, payload in summary["kernels"].items():
        T_list = payload["T_list"]
        color = colors.get(kernel_name, None)
        label = labels.get(kernel_name, kernel_name)

        _plot_mean_sem(
            ax_decay,
            T_list,
            payload["decay_exponent_mean"],
            payload["decay_exponent_sem"],
            label=f"{label} (empirical)",
            color=color,
        )
        _plot_mean_sem(
            ax_deff,
            T_list,
            payload["ridge_deff_mean"],
            payload["ridge_deff_sem"],
            label=label,
            color=color,
        )
        _plot_mean_sem(
            ax_count,
            T_list,
            payload["ridge_count_mean"],
            payload["ridge_count_sem"],
            label=label,
            color=color,
        )
        _plot_mean_sem(
            ax_budget,
            T_list,
            payload["rkhs_budget_mean"],
            payload["rkhs_budget_sem"],
            label=f"{label} (empirical)",
            color=color,
        )
        _plot_mean_sem(
            ax_budget,
            T_list,
            payload["powerlaw_budget_pred_mean"],
            payload["powerlaw_budget_pred_sem"],
            label=f"{label} fit-based power-law closure",
            color=color,
            linestyle="--",
            alpha_fill=0.08,
        )

    ax_decay.set_title(r"Spectral decay exponent $\beta$ from $\mu_j \sim j^{-\beta}$")
    ax_decay.set_ylabel(r"Decay exponent $\beta$")

    ax_deff.set_title(r"Statistical effective dimension $d_{\mathrm{eff}}(\lambda_*)$")
    ax_deff.set_ylabel(r"$d_{\mathrm{eff}}(\lambda_*)$")
    ax_deff.set_yscale("log")

    ax_count.set_title(r"Modes above tuned ridge threshold $\#\{\mu_j > \lambda_*\}$")
    ax_count.set_xlabel("Number of tasks T")
    ax_count.set_ylabel("Mode count")
    ax_count.set_yscale("log")

    if summary["normalize"] == "trace":
        ax_budget.set_title(r"Trace-normalized RKHS budget proxy $\tilde b_{\mathrm{req}}(T)$")
        ax_budget.set_ylabel(r"$\tilde b_{\mathrm{req}}(T)$")
    else:
        ax_budget.set_title(r"RKHS budget proxy $b_{\mathrm{req}}(T)$")
        ax_budget.set_ylabel(r"$b_{\mathrm{req}}(T)$")
    ax_budget.set_xlabel("Number of tasks T")
    ax_budget.set_yscale("log")

    for ax in axes.ravel():
        ax.grid(True, which="both", alpha=0.3)

    handles = []
    labels_used = []
    for ax in (ax_decay, ax_budget):
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels_used:
                handles.append(handle)
                labels_used.append(label)
    fig.legend(handles, labels_used, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(
        "Companion spectral diagnostics from saved eigenspectra"
        + f" ({summary['normalize']}-normalized)",
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_spectrum_raw_scale_summary(summary: dict, out_path: Path | str):
    _require_matplotlib()
    out_path = Path(out_path)
    colors = {"bias_only": "tab:blue", "full_ntk": "tab:orange"}
    labels = {"bias_only": "bias-only", "full_ntk": "full NTK"}

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4), sharex=True)
    ax_ridge_raw, ax_trace, ax_budget_norm = axes

    for kernel_name, payload in summary["kernels"].items():
        T_list = payload["T_list"]
        color = colors.get(kernel_name, None)
        label = labels.get(kernel_name, kernel_name)

        _plot_mean_sem(
            ax_ridge_raw,
            T_list,
            payload["ridge_lambda_raw_mean"],
            payload["ridge_lambda_raw_sem"],
            label=label,
            color=color,
        )
        _plot_mean_sem(
            ax_trace,
            T_list,
            payload["s_trace_mean"],
            payload["s_trace_sem"],
            label=label,
            color=color,
        )
        _plot_mean_sem(
            ax_budget_norm,
            T_list,
            payload["rkhs_budget_norm_mean"],
            payload["rkhs_budget_norm_sem"],
            label=label,
            color=color,
        )

    ax_ridge_raw.set_title(r"Raw effective ridge threshold $n\,\lambda_*\,s_{\mathrm{trace}}$")
    ax_ridge_raw.set_xlabel("Number of tasks T")
    ax_ridge_raw.set_ylabel(r"$n\,\lambda_*\,s_{\mathrm{trace}}$")
    ax_ridge_raw.set_yscale("log")

    ax_trace.set_title(r"Train-kernel scale $s_{\mathrm{trace}} = \mathrm{Tr}(K)/n$")
    ax_trace.set_xlabel("Number of tasks T")
    ax_trace.set_ylabel(r"$s_{\mathrm{trace}}$")
    ax_trace.set_yscale("log")

    ax_budget_norm.set_title(r"Trace-normalized RKHS budget proxy $\tilde b_{\mathrm{req}}(T)$")
    ax_budget_norm.set_xlabel("Number of tasks T")
    ax_budget_norm.set_ylabel(r"$\tilde b_{\mathrm{req}}(T)$")
    ax_budget_norm.set_yscale("log")

    for ax in axes.ravel():
        ax.grid(True, which="both", alpha=0.3)

    handles, labels_used = ax_ridge_raw.get_legend_handles_labels()
    fig.legend(handles, labels_used, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Raw-scale kernel diagnostics from saved eigenspectra", y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_spectrum_companion_grids(summary: dict, out_path: Path | str):
    _require_matplotlib()
    out_path = Path(out_path)
    kernel_order = ["bias_only", "full_ntk"]
    labels = {"bias_only": "bias-only", "full_ntk": "full NTK"}
    lambda_grid = np.asarray(summary["lambda_grid"], dtype=float)
    epsilon_list = np.asarray(summary["epsilon_list"], dtype=float)
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), sharex="col")
    for col, kernel_name in enumerate(kernel_order):
        if kernel_name not in summary["kernels"]:
            continue

        payload = summary["kernels"][kernel_name]
        T_list = payload["T_list"]
        colors = cmap(np.linspace(0.15, 0.9, max(len(lambda_grid), len(epsilon_list))))

        ax = axes[0, col]
        for j, lam in enumerate(lambda_grid):
            ax.plot(
                T_list,
                payload["fixed_deff_mean"][j],
                marker="o",
                linewidth=1.8,
                markersize=4,
                color=colors[j],
                label=rf"$\lambda={lam:g}$",
            )
        ax.set_title(rf"{labels[kernel_name]}: $d_{{\mathrm{{eff}}}}(\lambda)$")
        ax.set_ylabel(r"$d_{\mathrm{eff}}(\lambda)$")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False, fontsize=8)

        ax = axes[1, col]
        for j, eps_val in enumerate(epsilon_list):
            ax.plot(
                T_list,
                payload["fixed_count_mean"][j],
                marker="o",
                linewidth=1.8,
                markersize=4,
                color=colors[j],
                label=rf"$\epsilon={eps_val:g}$",
            )
        ax.set_title(rf"{labels[kernel_name]}: $\#\{{\mu_j > \epsilon\}}$")
        ax.set_xlabel("Number of tasks T")
        ax.set_ylabel(r"$\#\{\mu_j > \epsilon\}$")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Fixed-threshold spectral diagnostics"
        + f" ({summary['normalize']}-normalized)",
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_empirical_companion_plots(
    npz_path: Path | str,
    meta_path: Path | str | None = None,
    out_dir: Path | str | None = None,
    normalize: str = "trace",
    fit_rank_window: tuple[int, int] = (10, 400),
    epsilon_list: tuple[float, ...] = (10.0, 1.0, 0.1, 0.01),
    lambda_grid: tuple[float, ...] = (10.0, 1.0, 0.1, 0.01),
    include_analytic_budget_reference: bool = False,
    analytic_p: int = 3,
    analytic_Lmax: int = 60,
    analytic_n_quad: int = 500,
    analytic_fit_range: tuple[int, int] = (1, 60),
) -> dict:
    npz_path = Path(npz_path)
    if out_dir is None:
        out_dir = npz_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_saved_eigenspectra(
        npz_path=npz_path,
        meta_path=meta_path,
        normalize=normalize,
        fit_rank_window=fit_rank_window,
        epsilon_list=epsilon_list,
        lambda_grid=lambda_grid,
        include_analytic_budget_reference=include_analytic_budget_reference,
        analytic_p=analytic_p,
        analytic_Lmax=analytic_Lmax,
        analytic_n_quad=analytic_n_quad,
        analytic_fit_range=analytic_fit_range,
    )

    stem = npz_path.stem.replace("eigen_spectra_by_T_", "")
    summary_fig = out_dir / f"spectrum_companion_summary_{stem}.png"
    raw_scale_fig = out_dir / f"spectrum_companion_raw_scale_{stem}.png"
    grid_fig = out_dir / f"spectrum_companion_thresholds_{stem}.png"
    summary_npz = out_dir / f"spectrum_companion_summary_{stem}.npz"

    payload = {
        "normalize": np.asarray([summary["normalize"]]),
        "fit_rank_window": np.asarray(summary["fit_rank_window"], dtype=int),
        "epsilon_list": np.asarray(summary["epsilon_list"], dtype=float),
        "lambda_grid": np.asarray(summary["lambda_grid"], dtype=float),
    }
    for kernel_name, kernel_payload in summary["kernels"].items():
        payload[f"{kernel_name}_T_list"] = np.asarray(kernel_payload["T_list"], dtype=int)
        for key, value in kernel_payload.items():
            if key == "T_list":
                continue
            payload[f"{kernel_name}_{key}"] = np.asarray(value)
    for kernel_name, analytic_payload in summary.get("analytic_budget_reference", {}).items():
        for key, value in analytic_payload.items():
            payload[f"analytic_{kernel_name}_{key}"] = np.asarray(value)
    np.savez_compressed(summary_npz, **payload)
    print(f"Wrote: {summary_npz}")

    if plt is not None:
        plot_spectrum_companion_summary(summary, summary_fig)
        plot_spectrum_raw_scale_summary(summary, raw_scale_fig)
        plot_spectrum_companion_grids(summary, grid_fig)
        print(f"Wrote: {summary_fig}")
        print(f"Wrote: {raw_scale_fig}")
        print(f"Wrote: {grid_fig}")
    else:
        print("Skipping empirical companion plot rendering because matplotlib is unavailable.")
    return summary


def mean_empirical_spectrum_by_T(
    npz_path: Path | str,
    kernel_name: str,
    normalize: str = "trace",
    L_max: int | None = 500,
) -> tuple[np.ndarray, list[np.ndarray]]:
    npz_path = Path(npz_path)
    D = np.load(npz_path, allow_pickle=True)
    try:
        payload = D[kernel_name].item()
        T_list = np.asarray(payload["T_list"], dtype=int)
        eigvals = np.asarray(payload["eigvals"], dtype=float)
        mean_spectra = []

        for i in range(len(T_list)):
            per_seed = []
            max_len = 0
            for seed_idx in range(eigvals.shape[1]):
                lam_raw = clean_empirical_eigvals(eigvals[i, seed_idx])
                if lam_raw.size == 0:
                    continue
                lam, _ = normalize_empirical_eigvals(lam_raw, mode=normalize)
                if L_max is not None:
                    lam = lam[: int(L_max)]
                per_seed.append(lam)
                max_len = max(max_len, len(lam))

            if not per_seed:
                mean_spectra.append(np.array([], dtype=float))
                continue

            arr = np.full((len(per_seed), max_len), np.nan, dtype=float)
            for j, lam in enumerate(per_seed):
                arr[j, : len(lam)] = lam
            mean_spectra.append(np.nanmean(arr, axis=0))
        return T_list, mean_spectra
    finally:
        D.close()


def plot_talk_empirical_spectrum_shape(
    summary: dict,
    npz_path: Path | str,
    out_path: Path | str,
    kernel_name: str = "full_ntk",
    normalize: str = "trace",
    L_max: int = 500,
    selected_tasks: tuple[int, ...] = (1, 5, 10, 50),
    residual_task: int = 10,
    residual_rank_max: int = 50,
):
    _require_matplotlib()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    T_list, mean_spectra = mean_empirical_spectrum_by_T(
        npz_path=npz_path,
        kernel_name=kernel_name,
        normalize=normalize,
        L_max=L_max,
    )
    payload = summary["kernels"][kernel_name]
    label = {"full_ntk": "full NTK", "bias_only": "bias-only"}.get(kernel_name, kernel_name)
    fit_rank_window = tuple(int(v) for v in summary.get("fit_rank_window", (10, 200)))

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))
    ax_spec = axes[0, 0]
    ax_beta = axes[0, 1]
    ax_empty = axes[1, 0]
    ax_pref = axes[1, 1]
    ax_empty.axis("off")

    chosen = []
    seen_idx = set()
    for target_T in selected_tasks:
        if len(T_list) == 0:
            break
        idx = int(np.argmin(np.abs(T_list - int(target_T))))
        if idx not in seen_idx:
            chosen.append(idx)
            seen_idx.add(idx)
    if not chosen:
        chosen = list(range(min(3, len(T_list))))

    selected_colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.8, max(len(chosen), 1)))
    max_rank_seen = 1
    example_fit_drawn = False
    for color, idx in zip(selected_colors, chosen):
        T = int(T_list[idx])
        lam = mean_spectra[idx]
        if lam.size == 0:
            continue
        ranks = np.arange(1, len(lam) + 1, dtype=int)
        max_rank_seen = max(max_rank_seen, len(lam))
        ax_spec.loglog(ranks, lam, color=color, linewidth=2.1, label=rf"$T={T}$")

        if (T == 1) or (not example_fit_drawn and idx == chosen[0]):
            C_fit, beta_fit, _ = fit_rank_powerlaw_log(
                lam,
                rank_min=fit_rank_window[0],
                rank_max=fit_rank_window[1],
            )
            fit_lo = max(1, fit_rank_window[0])
            fit_hi = min(fit_rank_window[1], len(lam))
            if fit_hi >= fit_lo:
                fit_ranks = np.arange(fit_lo, fit_hi + 1, dtype=float)
                fit_vals = C_fit * (fit_ranks ** (-beta_fit))
                ax_spec.loglog(
                    fit_ranks,
                    fit_vals,
                    linestyle="--",
                    linewidth=2.2,
                    color="black",
                    zorder=10,
                )
                example_fit_drawn = True
    ax_spec.plot([], [], "k--", linewidth=1.8, label="power-law fit")

    ax_spec.set_xlim(1, max(max_rank_seen, 10))
    ax_spec.set_xlabel("Eigenvalue rank j")
    ax_spec.set_ylabel(r"Trace-normalized eigenvalue $\tilde{\mu}_j$")
    ax_spec.set_title(f"(A) {label}: empirical spectra and fits")
    ax_spec.grid(True, which="both", alpha=0.3)
    ax_spec.legend(frameon=False, fontsize=9, ncol=2, loc="best")

    colors = {"bias_only": "tab:blue", "full_ntk": "tab:orange"}
    labels = {"bias_only": "bias-only", "full_ntk": "full NTK"}
    for name in ("bias_only", "full_ntk"):
        if name not in summary["kernels"]:
            continue
        kernel_payload = summary["kernels"][name]
        _plot_mean_sem(
            ax_beta,
            kernel_payload["T_list"],
            kernel_payload["decay_exponent_mean"],
            kernel_payload["decay_exponent_sem"],
            label=labels.get(name, name),
            color=colors.get(name),
        )
    ax_beta.set_xlabel("Number of tasks T")
    ax_beta.set_ylabel(r"Decay exponent $\beta$")
    ax_beta.set_title(r"(B) Empirical decay exponent")
    ax_beta.grid(True, which="both", alpha=0.3)
    beta_handles = [
        Line2D([0], [0], color=colors["bias_only"], linewidth=2.0, linestyle="-", label=labels["bias_only"]),
        Line2D([0], [0], color=colors["full_ntk"], linewidth=2.0, linestyle="-", label=labels["full_ntk"]),
    ]
    ax_beta.legend(handles=beta_handles, frameon=False)

    for name in ("bias_only", "full_ntk"):
        if name not in summary["kernels"]:
            continue
        kernel_payload = summary["kernels"][name]
        _plot_mean_sem(
            ax_pref,
            kernel_payload["T_list"],
            kernel_payload["rank_powerlaw_C_mean"],
            kernel_payload["rank_powerlaw_C_sem"],
            label=labels.get(name, name),
            color=colors.get(name),
        )
    ax_pref.set_title(r"(C) Power-law prefactor $\tilde C(T)$")
    ax_pref.set_xlabel("Number of tasks T")
    ax_pref.set_ylabel(r"$\tilde C$")
    ax_pref.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}")


def plot_talk_budget_proxy(
    summary: dict,
    npz_path: Path | str,
    out_path: Path | str,
    normalize: str = "trace",
    residual_task: int = 10,
    residual_rank_max: int = 50,
):
    _require_matplotlib()
    npz_path = Path(npz_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6))
    ax_budget, ax_corr, ax_resid = axes
    colors = {"bias_only": "tab:blue", "full_ntk": "tab:orange"}
    labels = {"bias_only": "bias-only", "full_ntk": "full NTK"}

    for kernel_name in ("bias_only", "full_ntk"):
        if kernel_name not in summary["kernels"]:
            continue
        payload = summary["kernels"][kernel_name]
        color = colors.get(kernel_name)
        label = labels.get(kernel_name, kernel_name)
        _plot_mean_sem(
            ax_budget,
            payload["T_list"],
            payload["rkhs_budget_norm_mean"],
            payload["rkhs_budget_norm_sem"],
            label=f"{label} empirical",
            color=color,
        )
        ax_budget.plot(
            payload["T_list"],
            payload["powerlaw_budget_pred_mean"],
            linestyle="--",
            linewidth=2.0,
            color=color,
            label=f"{label} power-law closure",
        )
        _plot_mean_sem(
            ax_corr,
            payload["T_list"],
            payload["powerlaw_budget_correction_mean"],
            payload["powerlaw_budget_correction_sem"],
            label=label,
            color=color,
        )

    ax_budget.set_xlabel("Number of tasks T")
    ax_budget.set_ylabel(r"$\tilde b_{\mathrm{req}}(T)$")
    ax_budget.set_title(r"(A) Trace-normalized RKHS budget proxy")
    ax_budget.set_yscale("log")
    ax_budget.grid(True, which="both", alpha=0.3)
    ax_budget.legend(frameon=False, fontsize=8, ncol=2)

    ax_corr.axhline(1.0, color="black", linestyle="--", linewidth=1.4)
    ax_corr.set_xlabel("Number of tasks T")
    ax_corr.set_ylabel(r"$\tilde b_{\mathrm{emp}} / \tilde b_{\mathrm{pl}}$")
    ax_corr.set_title(r"(B) Closure correction factor")
    ax_corr.grid(True, which="both", alpha=0.3)
    corr_handles = [
        Line2D([0], [0], color=colors["bias_only"], linewidth=2.0, linestyle="-", label=labels["bias_only"]),
        Line2D([0], [0], color=colors["full_ntk"], linewidth=2.0, linestyle="-", label=labels["full_ntk"]),
    ]
    ax_corr.legend(handles=corr_handles, frameon=False, fontsize=8)

    fit_rank_window = tuple(int(v) for v in summary.get("fit_rank_window", (10, 200)))
    for kernel_name in ("bias_only", "full_ntk"):
        if kernel_name not in summary["kernels"]:
            continue
        T_vals, spectra = mean_empirical_spectrum_by_T(
            npz_path=npz_path,
            kernel_name=kernel_name,
            normalize=normalize,
            L_max=max(residual_rank_max, 500),
        )
        if len(T_vals) == 0:
            continue
        idx = int(np.argmin(np.abs(T_vals - int(residual_task))))
        T_sel = int(T_vals[idx])
        lam = np.asarray(spectra[idx], dtype=float)
        if lam.size == 0:
            continue
        C_fit, beta_fit, _ = fit_rank_powerlaw_log(
            lam,
            rank_min=fit_rank_window[0],
            rank_max=fit_rank_window[1],
        )
        rank_max = min(int(residual_rank_max), len(lam))
        ranks = np.arange(1, rank_max + 1, dtype=float)
        pred = C_fit * (ranks ** (-beta_fit))
        residual = lam[:rank_max] / np.clip(pred, 1e-12, None)
        weighted_num = np.cumsum((ranks ** beta_fit) / np.clip(residual, 1e-12, None))
        weighted_den = np.cumsum(ranks ** beta_fit)
        cumulative_correction = np.sqrt(weighted_num / np.clip(weighted_den, 1e-12, None))
        ax_resid.plot(
            ranks,
            cumulative_correction,
            linewidth=2.0,
            color=colors.get(kernel_name),
            label=labels.get(kernel_name, kernel_name),
        )

    ax_resid.axhline(1.0, color="black", linestyle="--", linewidth=1.4)
    ax_resid.set_xlabel("Rank r")
    ax_resid.set_ylabel(r"$\left[\frac{\sum_{j\leq r} j^\beta / r_j}{\sum_{j\leq r} j^\beta}\right]^{1/2}$")
    ax_resid.set_title(rf"(C) Cumulative correction at $T={int(residual_task)}$")
    ax_resid.grid(True, which="both", alpha=0.3)
    resid_handles = [
        Line2D([0], [0], color=colors["bias_only"], linewidth=2.0, linestyle="-", label=labels["bias_only"]),
        Line2D([0], [0], color=colors["full_ntk"], linewidth=2.0, linestyle="-", label=labels["full_ntk"]),
        Line2D([0], [0], color="black", linewidth=1.4, linestyle="--", label="unity"),
    ]
    ax_resid.legend(handles=resid_handles, frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}")


def _representative_dense_task_prediction(
    T: int,
    reg_scale: float,
    kernel_name: str,
    meta: dict,
    seed: int = 0,
    task_index: int = 0,
    eval_pts: int = 512,
):
    rng = np.random.default_rng(int(seed))
    normals_mode = str(meta.get("normals_mode", "fibonacci"))
    random_phase = bool(meta.get("random_phase", True))
    generalization = str(meta.get("generalization", "within"))
    pts_per_circle = int(meta.get("pts_per_circle", 128))
    m = int(meta.get("m", 4))
    W_std = float(meta.get("W_std", 1.0))
    b_std = float(meta.get("b_std", 1.0))
    kappa_scale = float(meta.get("kappa_scale", 1.0))
    par = KernelParams(sigma_w2=W_std**2, sigma_b2=b_std**2, sigma_v2=1.0, sigma_bout2=1.0)
    which = "full" if kernel_name == "full_ntk" else "bias"

    def sample_tasks(rng_local):
        if normals_mode == "random":
            normals = sample_random_normals_s2(int(T), rng=rng_local)
        elif normals_mode == "fibonacci":
            normals = normals_fibonacci_s2(int(T))
        else:
            raise ValueError("normals_mode must be 'random' or 'fibonacci'.")
        phases = rng_local.uniform(0.0, 2.0 * np.pi, size=int(T)) if random_phase else np.zeros(int(T))
        return normals, phases

    normals_tr, phases_tr = sample_tasks(rng)
    if generalization == "within":
        normals_te, phases_te = normals_tr, phases_tr
    elif generalization == "across":
        normals_te, phases_te = sample_tasks(rng)
    else:
        raise ValueError("generalization must be 'within' or 'across'.")

    X_train_np, y_train_np, _ = generate_multi_tasks_from_normals_phases_with_mode(
        normals_tr,
        phases_tr,
        pts_per_circle=pts_per_circle,
        m=m,
        rng=rng,
        mode="grid",
    )
    X_train = jnp.asarray(X_train_np, dtype=jnp.float32)
    y_train = jnp.asarray(y_train_np, dtype=jnp.float32).reshape(-1, 1)
    K_train = gram_from_kappa(X_train, X_train, par=par, which=which, kappa_scale=kappa_scale)
    n = int(K_train.shape[0])
    s_trace = jnp.trace(K_train) / max(n, 1)
    K_reg = K_train / s_trace + (n * float(reg_scale)) * jnp.eye(n, dtype=K_train.dtype)
    alpha = jnp.linalg.solve(K_reg, y_train)

    task_index = int(min(max(task_index, 0), int(T) - 1))
    theta = np.linspace(0.0, 2.0 * np.pi, int(eval_pts), endpoint=False)
    phi_eval = theta + float(phases_te[task_index])
    X_eval_np, y_eval_np = generate_single_circle_with_phi(normals_te[task_index], phi_eval, m)
    X_eval = jnp.asarray(X_eval_np, dtype=jnp.float32)
    K_eval_train = gram_from_kappa(X_eval, X_train, par=par, which=which, kappa_scale=kappa_scale)
    y_pred = np.asarray((K_eval_train / s_trace) @ alpha).reshape(-1)
    return theta, np.asarray(y_eval_np, dtype=float), y_pred


def plot_talk_frozen_eval_mse(
    ridge_npz_path: Path | str,
    eval_npz_path: Path | str,
    out_path: Path | str,
    example_Ts: tuple[int, ...] = (1, 10),
    representative_seed: int = 0,
    representative_task_index: int = 0,
    eval_pts: int = 512,
):
    _require_matplotlib()
    ridge_npz_path = Path(ridge_npz_path)
    eval_npz_path = Path(eval_npz_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eval_meta_path = eval_npz_path.with_name(eval_npz_path.stem + "_meta.json")

    D_ridge = np.load(ridge_npz_path)
    D_eval = np.load(eval_npz_path, allow_pickle=True)
    try:
        meta = {}
        if eval_meta_path.exists():
            with open(eval_meta_path, "r") as f:
                meta = json.load(f)
        T_list = np.asarray(D_ridge["T_list"], dtype=int)
        lam_star_full = np.asarray(D_ridge["lam_star_full_ntk"], dtype=float)
        lam_star_bias = np.asarray(D_ridge["lam_star_bias_only"], dtype=float)
        full_res = D_eval["full_ntk"].item()
        bias_res = D_eval["bias_only"].item()
        mse_star_full = np.asarray(full_res["mse_mean_B"], dtype=float)
        mse_star_bias = np.asarray(bias_res["mse_mean_B"], dtype=float)

        fig = plt.figure(figsize=(9.1, 9.0))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0], hspace=0.32, wspace=0.30)
        ax_top = fig.add_subplot(gs[0, :])
        pred_axes = [
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
            [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],
        ]

        ax_top.plot(T_list, mse_star_full, marker="o", linewidth=2.0, label="full NTK", color="tab:orange")
        ax_top.plot(T_list, mse_star_bias, marker="o", linewidth=2.0, label="bias-only", color="tab:blue")
        ax_top.axhline(0.5, linestyle="--", color="crimson", linewidth=1.6, label="baseline")

        chosen = []
        seen = set()
        for target_T in example_Ts:
            idx = int(np.argmin(np.abs(T_list - int(target_T))))
            if idx not in seen:
                chosen.append(idx)
                seen.add(idx)
        example_palette = {
            1: "tab:green",
            10: "purple",
        }
        fallback_colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.75, max(len(chosen), 1)))
        tick_color_map = {}
        example_colors = []
        for fallback_color, idx in zip(fallback_colors, chosen):
            T_val = int(T_list[idx])
            color = example_palette.get(T_val, fallback_color)
            example_colors.append(color)
            tick_color_map[T_val] = color
            ax_top.axvline(T_val, color=color, linestyle=":", linewidth=1.2, alpha=0.9, zorder=0)

        ax_top.set_xlabel("Number of tasks T")
        ax_top.set_ylabel("Oracle (no-noise) MSE")
        ax_top.set_title(r"Frozen-$\lambda(A)$ multitask performance on test set B")
        ax_top.set_yscale("log")
        ax_top.grid(True, which="both", alpha=0.3)
        ax_top.legend(frameon=False)
        ax_top.set_xticks(T_list)
        ax_top.set_xticklabels([str(int(t)) for t in T_list])
        for tick_label, t_val in zip(ax_top.get_xticklabels(), T_list):
            if int(t_val) in tick_color_map:
                tick_label.set_color(tick_color_map[int(t_val)])
                tick_label.set_fontweight("bold")

        target_linewidth = 6.0
        style_handles = [
            Line2D([0], [0], color="0.65", linewidth=target_linewidth, label="target"),
            Line2D([0], [0], color="tab:orange", linewidth=2.0, label="full NTK"),
        ]
        scatter_handles_full = [
            Line2D([0], [0], marker="o", linestyle="None", color="tab:orange", markersize=5, label="full NTK"),
            Line2D([0], [0], color="black", linewidth=1.3, linestyle="--", label=r"$y=x$"),
        ]
        scatter_handles_bias = [
            Line2D([0], [0], marker="o", linestyle="None", color="tab:blue", markersize=5, label="bias-only"),
            Line2D([0], [0], color="black", linewidth=1.3, linestyle="--", label=r"$y=x$"),
        ]
        for row_idx, (color, idx) in enumerate(zip(example_colors, chosen)):
            T_val = int(T_list[idx])
            theta, y_true, y_pred_full = _representative_dense_task_prediction(
                T=T_val,
                reg_scale=float(lam_star_full[idx]),
                kernel_name="full_ntk",
                meta=meta,
                seed=representative_seed,
                task_index=representative_task_index,
                eval_pts=eval_pts,
            )
            _, _, y_pred_bias = _representative_dense_task_prediction(
                T=T_val,
                reg_scale=float(lam_star_bias[idx]),
                kernel_name="bias_only",
                meta=meta,
                seed=representative_seed,
                task_index=representative_task_index,
                eval_pts=eval_pts,
            )
            theta_plot = theta / np.pi
            ax_full = pred_axes[row_idx][0]
            ax_full.plot(theta_plot, y_true, color="0.65", linewidth=target_linewidth, zorder=1)
            ax_full.plot(theta_plot, y_pred_full, color="tab:orange", linewidth=2.0, linestyle="-", zorder=3)
            ax_full.text(
                0.02,
                0.92,
                rf"$T={T_val}$",
                transform=ax_full.transAxes,
                ha="left",
                va="top",
                fontsize=10.5,
                fontweight="bold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=color, alpha=0.92),
            )

        pred_axes[0][0].set_title("Full NTK representative trace")
        scatter_T = 10
        scatter_idx = int(np.argmin(np.abs(T_list - int(scatter_T))))
        scatter_T_val = int(T_list[scatter_idx])
        scatter_full_ax = pred_axes[0][1]
        scatter_bias_ax = pred_axes[1][1]
        _, y_true_scatter, y_pred_full_scatter = _representative_dense_task_prediction(
            T=scatter_T_val,
            reg_scale=float(lam_star_full[scatter_idx]),
            kernel_name="full_ntk",
            meta=meta,
            seed=representative_seed,
            task_index=representative_task_index,
            eval_pts=eval_pts,
        )
        _, _, y_pred_bias_scatter = _representative_dense_task_prediction(
            T=scatter_T_val,
            reg_scale=float(lam_star_bias[scatter_idx]),
            kernel_name="bias_only",
            meta=meta,
            seed=representative_seed,
            task_index=representative_task_index,
            eval_pts=eval_pts,
        )
        diag_min = min(
            float(np.min(y_true_scatter)),
            float(np.min(y_pred_full_scatter)),
            float(np.min(y_pred_bias_scatter)),
        )
        diag_max = max(
            float(np.max(y_true_scatter)),
            float(np.max(y_pred_full_scatter)),
            float(np.max(y_pred_bias_scatter)),
        )
        diag_pad = 0.04 * max(diag_max - diag_min, 1.0)
        diag_lo = diag_min - diag_pad
        diag_hi = diag_max + diag_pad
        scatter_full_ax.scatter(y_true_scatter, y_pred_full_scatter, color="tab:orange", s=14, alpha=0.85, zorder=3)
        scatter_bias_ax.scatter(y_true_scatter, y_pred_bias_scatter, color="tab:blue", s=14, alpha=0.8, zorder=3)
        for ax_scatter, edgecolor in ((scatter_full_ax, "tab:orange"), (scatter_bias_ax, "tab:blue")):
            ax_scatter.plot([diag_lo, diag_hi], [diag_lo, diag_hi], color="black", linewidth=1.3, linestyle="--", zorder=1)
            ax_scatter.text(
                0.02,
                0.92,
                rf"$T={scatter_T_val}$",
                transform=ax_scatter.transAxes,
                ha="left",
                va="top",
                fontsize=10.5,
                fontweight="bold",
                color="purple",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=edgecolor, alpha=0.92),
            )
        pred_axes[0][1].set_title(rf"Full NTK: $y_{{\mathrm{{pred}}}}$ vs. $y_{{\mathrm{{true}}}}$")
        pred_axes[1][1].set_title(rf"Bias-only: $y_{{\mathrm{{pred}}}}$ vs. $y_{{\mathrm{{true}}}}$")
        for row_idx, row in enumerate(pred_axes):
            for col_idx, ax_pred in enumerate(row):
                ax_pred.grid(True, alpha=0.3)
                if col_idx == 0:
                    ax_pred.set_xlim(0.0, 2.0)
                    ax_pred.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
                    ax_pred.set_ylim(-1.15, 1.15)
                    ax_pred.axhline(0.0, color="crimson", linewidth=1.6, linestyle="--", alpha=0.95, zorder=0)
                    if row_idx == len(pred_axes) - 1:
                        ax_pred.set_xlabel(r"Circle angle $\theta/\pi$")
                    else:
                        ax_pred.tick_params(labelbottom=False)
                    ax_pred.set_ylabel("Target / prediction")
                else:
                    ax_pred.set_xlim(-1.1, 1.1)
                    ax_pred.set_ylim(-1.1, 1.1)
                    ax_pred.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
                    ax_pred.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
                    if row_idx == len(pred_axes) - 1:
                        ax_pred.set_xlabel(r"$y_{\mathrm{true}}$")
                    else:
                        ax_pred.tick_params(labelbottom=False)
                    ax_pred.set_ylabel(r"$y_{\mathrm{pred}}$")

        legend1 = pred_axes[0][0].legend(
            handles=style_handles,
            frameon=True,
            loc="lower left",
            facecolor="white",
            edgecolor="0.75",
            framealpha=0.94,
        )
        pred_axes[0][0].add_artist(legend1)
        pred_axes[0][1].legend(
            handles=scatter_handles_full,
            frameon=True,
            loc="lower right",
            facecolor="white",
            edgecolor="0.75",
            framealpha=0.94,
        )
        pred_axes[1][1].legend(
            handles=scatter_handles_bias,
            frameon=True,
            loc="lower right",
            facecolor="white",
            edgecolor="0.75",
            framealpha=0.94,
        )

        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote: {out_path}")
    finally:
        D_ridge.close()
        D_eval.close()


def _wrap_task_angles(angle_values):
    angle_values = np.asarray(angle_values, dtype=float)
    if angle_values.size == 0:
        return angle_values
    return np.mod(angle_values, 2.0 * np.pi)


def _pairwise_circular_distances(angles):
    angles = np.asarray(angles, dtype=float)
    dists = []
    pairs = []
    for i in range(len(angles)):
        for j in range(i + 1, len(angles)):
            diff = float(abs(angles[i] - angles[j]))
            dists.append(min(diff, 2.0 * np.pi - diff))
            pairs.append((i, j))
    return np.asarray(dists, dtype=float), pairs


def _upper_tri_values_from_pairs(M, pairs):
    M = np.asarray(M, dtype=float)
    return np.asarray([M[i, j] for i, j in pairs], dtype=float)


def _group_means_by_distance(dists, values, decimals: int = 10):
    dists = np.asarray(dists, dtype=float)
    values = np.asarray(values, dtype=float)
    keys = np.round(dists, decimals=decimals)
    uniq = np.unique(keys)
    x = []
    mean = []
    sem = []
    for key in uniq:
        mask = keys == key
        vals = values[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        x.append(float(np.mean(dists[mask])))
        mean.append(float(np.mean(vals)))
        sem.append(float(np.std(vals, ddof=0) / max(np.sqrt(vals.size), 1.0)))
    return np.asarray(x, dtype=float), np.asarray(mean, dtype=float), np.asarray(sem, dtype=float)


def _align_coords_to_angles(coords, angle_values, eps: float = 1e-12):
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[0] == 0:
        return coords
    aligned = coords - np.mean(coords, axis=0, keepdims=True)
    theta = _wrap_task_angles(angle_values)
    if theta.shape[0] != aligned.shape[0]:
        return aligned
    if aligned.shape[1] < 2:
        return aligned
    ref = np.column_stack([np.cos(theta), np.sin(theta)])
    cross = aligned[:, :2].T @ ref
    if np.linalg.norm(cross) <= eps:
        return aligned
    U, _, Vt = np.linalg.svd(cross, full_matrices=False)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    aligned[:, :2] = aligned[:, :2] @ R
    return aligned


def _set_equal_3d_limits(ax, coords, pad_frac: float = 0.1):
    coords = np.asarray(coords, dtype=float)
    if coords.size == 0:
        return
    mins = np.nanmin(coords, axis=0)
    maxs = np.nanmax(coords, axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    radius = max(radius, 1e-6)
    radius *= (1.0 + pad_frac)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _load_single_axis_update_run(results_dir: Path | str):
    results_dir = Path(results_dir)
    meta_rows = _read_csv_rows(results_dir / "task_metadata.csv")
    angles = np.asarray([float(row["color_value"]) for row in meta_rows], dtype=float)

    sim_bias = np.loadtxt(results_dir / "empirical_update_similarity_bias.csv", delimiter=",", skiprows=1)
    sim_full = np.loadtxt(results_dir / "empirical_update_similarity_full.csv", delimiter=",", skiprows=1)

    coord_rows = _read_csv_rows(results_dir / "empirical_task_update_manifold_coords.csv")
    spec_rows = _read_csv_rows(results_dir / "empirical_task_update_manifold_spectrum.csv")

    coords = {}
    for kind in ("bias", "full"):
        kind_rows = sorted((row for row in coord_rows if row["kind"] == kind), key=lambda row: int(row["task_index"]))
        coords[kind] = np.asarray(
            [[float(row["coord_1"]), float(row["coord_2"]), float(row["coord_3"])] for row in kind_rows],
            dtype=float,
        )

    var_ratio = {}
    for kind in ("bias", "full"):
        kind_rows = sorted((row for row in spec_rows if row["kind"] == kind), key=lambda row: int(row["component"]))
        var_ratio[kind] = np.asarray([float(row["variance_ratio"]) for row in kind_rows], dtype=float)

    return {
        "angles": angles,
        "sim_bias": np.asarray(sim_bias, dtype=float),
        "sim_full": np.asarray(sim_full, dtype=float),
        "coords_bias": coords["bias"],
        "coords_full": coords["full"],
        "var_ratio_bias": var_ratio["bias"],
        "var_ratio_full": var_ratio["full"],
    }


def _load_single_axis_update_vectors(results_dir: Path | str):
    results_dir = Path(results_dir)
    obj = np.load(results_dir / "task_update_vectors.npz", allow_pickle=True)
    try:
        return {
            "delta_bias": np.asarray(obj["delta_bias"], dtype=float),
            "delta_full": np.asarray(obj["delta_full"], dtype=float),
            "angles": np.asarray(obj["task_color_values"], dtype=float),
        }
    finally:
        obj.close()


def _pca_reconstruction_curve(updates, max_k: int | None = None, eps: float = 1e-12):
    updates = np.asarray(updates, dtype=float)
    if updates.ndim != 2:
        raise ValueError("Expected task-by-parameter update matrix.")
    mean_update = np.mean(updates, axis=0, keepdims=True)
    centered = updates - mean_update
    U, sing_vals, Vt = np.linalg.svd(centered, full_matrices=False)
    max_rank = min(centered.shape[0], centered.shape[1])
    if max_k is None:
        max_k = max_rank
    max_k = int(max(1, min(max_k, max_rank)))
    total_frob = float(np.linalg.norm(centered, ord="fro"))
    denom = max(centered.shape[0] - 1, 1)
    evals = (sing_vals ** 2) / denom
    total_eval = float(np.sum(evals))
    ks = np.arange(1, max_k + 1, dtype=int)
    rel_errors = []
    cumulative_var = []
    for k in ks:
        recon = (U[:, :k] * sing_vals[:k]) @ Vt[:k, :]
        err = centered - recon
        rel_errors.append(float(np.linalg.norm(err, ord="fro") / max(total_frob, eps)))
        cumulative_var.append(float(np.sum(evals[:k]) / max(total_eval, eps)))
    return {
        "k": ks,
        "relative_fro_error": np.asarray(rel_errors, dtype=float),
        "cumulative_var": np.asarray(cumulative_var, dtype=float),
    }


def _draw_unit_sphere_wireframe_simple(ax, *, color="0.84", alpha=0.32):
    u = np.linspace(0.0, 2.0 * np.pi, 60)
    v = np.linspace(0.0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, rstride=3, cstride=3, color=color, linewidth=0.45, alpha=alpha)


def _set_equal_sphere_axes_simple(ax, *, lim=1.18):
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


def _orthonormal_circle_frame_np(normal):
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    v1 = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v1 = v1 - n * np.dot(n, v1)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(n, v1)
    return n, v1, v2


def _evaluate_global_sphere_field_np(X, field_name="xz"):
    X = np.asarray(X, dtype=float)
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
    raise ValueError("Unsupported global_field_name")


def _generate_circle_samples_np(
    normal,
    pts,
    m,
    *,
    phase=0.0,
    target_phase=0.0,
    target_mode="local_harmonic",
    global_field_name="xz",
):
    _, v1, v2 = _orthonormal_circle_frame_np(normal)
    phi = np.linspace(0.0, 2.0 * np.pi, int(pts), endpoint=False) + float(phase)
    X = np.outer(np.cos(phi), v1) + np.outer(np.sin(phi), v2)
    if target_mode == "local_harmonic":
        y = np.sin(int(m) * phi + float(target_phase))
    elif target_mode == "global_field":
        y = _evaluate_global_sphere_field_np(X, field_name=global_field_name)
    else:
        raise ValueError("target_mode must be 'local_harmonic' or 'global_field'")
    return X, y, phi


def _select_single_axis_highlight_indices(angle_values):
    angle_values = np.asarray(angle_values, dtype=float)
    anchors = np.array([0.0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, 1.0 * np.pi], dtype=float)
    chosen = []
    for angle in anchors:
        idx = int(np.argmin(np.abs(angle_values - angle)))
        if idx not in chosen:
            chosen.append(idx)
    return chosen


def _format_single_axis_theta_label(angle_value):
    angle_deg = float(np.degrees(np.mod(angle_value, 2.0 * np.pi)))
    return rf"$\theta={angle_deg:.0f}^\circ$"


def plot_talk_single_axis_geometry_overview(results_dir: Path | str, out_path: Path | str):
    _require_matplotlib()
    if Axes3D is None:
        raise ImportError("3D matplotlib support is required for the single-axis geometry overview.")

    results_dir = Path(results_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _read_csv_rows(results_dir / "task_metadata.csv")
    normals = np.asarray(
        [[float(r["normal_x"]), float(r["normal_y"]), float(r["normal_z"])] for r in rows],
        dtype=float,
    )
    phases = np.asarray([float(r["phase"]) for r in rows], dtype=float)
    angle_values = np.asarray([float(r["color_value"]) for r in rows], dtype=float)
    target_phase = float(rows[0].get("target_phase", 0.0))
    target_mode = rows[0].get("target_mode", "local_harmonic") or "local_harmonic"
    global_field_name = rows[0].get("global_field_name", "xz") or "xz"
    m = int(rows[0].get("m", 4))
    highlight_indices = _select_single_axis_highlight_indices(angle_values)
    highlight_set = set(highlight_indices)
    highlight_rank = {idx: rank for rank, idx in enumerate(highlight_indices)}
    wrapped_angles = _wrap_task_angles(angle_values)
    angle_norm = mcolors.Normalize(vmin=0.0, vmax=2.0 * np.pi)
    angle_cmap = plt.get_cmap("twilight_shifted")

    fig = plt.figure(figsize=(15.0, 7.2))
    gs = fig.add_gridspec(1, 2, left=0.04, right=0.98, top=0.90, bottom=0.16, wspace=0.16)
    ax_tasks = fig.add_subplot(gs[0, 0], projection="3d")
    ax_normals = fig.add_subplot(gs[0, 1], projection="3d")

    _draw_unit_sphere_wireframe_simple(ax_tasks)
    _draw_unit_sphere_wireframe_simple(ax_normals, color="0.86", alpha=0.26)

    rotation_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    for ax in (ax_tasks, ax_normals):
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
    sphere_label_scales = [1.17, 1.11, 1.19, 1.12, 1.17]
    normal_label_scales = [1.16, 1.12, 1.18, 1.12, 1.16]
    for idx, (normal, phase) in enumerate(zip(normals, phases)):
        X, y, _ = _generate_circle_samples_np(
            normal,
            240,
            m,
            phase=float(phase),
            target_phase=target_phase,
            target_mode=target_mode,
            global_field_name=global_field_name,
        )
        task_color = angle_cmap(angle_norm(wrapped_angles[idx]))
        if idx in highlight_set:
            ax_tasks.plot(X[:, 0], X[:, 1], X[:, 2], color=task_color, lw=1.25, alpha=0.84)
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
            label_scale = sphere_label_scales[highlight_rank[idx] % len(sphere_label_scales)]
            label_pos = label_scale * X[label_idx]
            ax_tasks.text(
                label_pos[0],
                label_pos[1],
                label_pos[2],
                _format_single_axis_theta_label(angle_values[idx]),
                fontsize=8.8,
                color=task_color,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.82),
            )
        else:
            ax_tasks.plot(X[:, 0], X[:, 1], X[:, 2], color="0.58", lw=0.8, alpha=0.18)

    order = np.argsort(angle_values)
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
        task_color = angle_cmap(angle_norm(wrapped_angles[idx]))
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
        label_scale = normal_label_scales[highlight_rank[idx] % len(normal_label_scales)]
        label_pos = label_scale * normal
        ax_normals.text(
            label_pos[0],
            label_pos[1],
            label_pos[2],
            _format_single_axis_theta_label(angle_values[idx]),
            fontsize=8.8,
            color=task_color,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.82),
        )

    ax_tasks.set_title("Task supports on the sphere\nHighlighted circles colored by target value")
    ax_normals.set_title("Great-circle normals generating the family")
    for ax in (ax_tasks, ax_normals):
        ax.view_init(elev=20, azim=36)
        _set_equal_sphere_axes_simple(ax)

    if target_scatter is not None:
        cax = fig.add_axes([0.34, 0.08, 0.32, 0.035])
        cbar = fig.colorbar(target_scatter, cax=cax, orientation="horizontal")
        cbar.set_label("Target value on highlighted circles")

    fig.suptitle("Single-axis task family geometry overview", fontsize=16, y=0.96)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote: {out_path}")


def plot_talk_single_axis_update_similarity(results_dir: Path | str, out_path: Path | str):
    _require_matplotlib()
    results_dir = Path(results_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run = _load_single_axis_update_run(results_dir)
    angles = run["angles"]
    dists, pairs = _pairwise_circular_distances(angles)
    sim_bias = _upper_tri_values_from_pairs(run["sim_bias"], pairs)
    sim_full = _upper_tri_values_from_pairs(run["sim_full"], pairs)
    x_bias, mean_bias, sem_bias = _group_means_by_distance(dists, sim_bias)
    x_full, mean_full, sem_full = _group_means_by_distance(dists, sim_full)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(dists / np.pi, sim_bias, s=18, alpha=0.16, color="tab:blue", edgecolors="none")
    ax.scatter(dists / np.pi, sim_full, s=18, alpha=0.16, color="tab:orange", edgecolors="none")
    ax.plot(x_bias / np.pi, mean_bias, color="tab:blue", linewidth=2.4, marker="o", markersize=4.5, label="Bias-only mean")
    ax.plot(x_full / np.pi, mean_full, color="tab:orange", linewidth=2.4, marker="o", markersize=4.5, label="Full NTK mean")
    ax.fill_between(x_bias / np.pi, mean_bias - sem_bias, mean_bias + sem_bias, color="tab:blue", alpha=0.14)
    ax.fill_between(x_full / np.pi, mean_full - sem_full, mean_full + sem_full, color="tab:orange", alpha=0.14)
    ax.set_xlabel(r"Task angular distance $/ \pi$")
    ax.set_ylabel("Update cosine similarity")
    ax.set_title("Single-axis task update similarity")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(np.linspace(0.0, 1.0, 5))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}")


def plot_talk_single_axis_update_manifolds(results_dir: Path | str, out_path: Path | str):
    _require_matplotlib()
    results_dir = Path(results_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run = _load_single_axis_update_run(results_dir)
    angles = _wrap_task_angles(run["angles"])
    coords_bias = _align_coords_to_angles(run["coords_bias"], angles)
    coords_full = _align_coords_to_angles(run["coords_full"], angles)
    order = np.argsort(angles)
    cmap = plt.get_cmap("twilight_shifted")
    norm = mcolors.Normalize(vmin=0.0, vmax=2.0 * np.pi)

    fig = plt.figure(figsize=(11.2, 5.2))
    ax_bias = fig.add_subplot(1, 2, 1, projection="3d")
    ax_full = fig.add_subplot(1, 2, 2, projection="3d")

    for ax, coords, title, vr in [
        (ax_bias, coords_bias, "Bias-only update manifold", run["var_ratio_bias"]),
        (ax_full, coords_full, "Full NTK update manifold", run["var_ratio_full"]),
    ]:
        ax.plot(
            coords[order, 0],
            coords[order, 1],
            coords[order, 2],
            color="0.35",
            linewidth=1.1,
            alpha=0.4,
        )
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=angles,
            cmap=cmap,
            norm=norm,
            s=46,
            edgecolors="none",
            alpha=0.9,
            depthshade=False,
        )
        ax.scatter(
            coords[[0], 0],
            coords[[0], 1],
            coords[[0], 2],
            c=[angles[0]],
            cmap=cmap,
            norm=norm,
            s=84,
            edgecolors="k",
            linewidth=0.7,
            depthshade=False,
        )
        top3 = float(np.nansum(vr[:3])) if np.size(vr) else np.nan
        ax.set_title(f"{title}\nTop-3 explained variance = {top3:.2f}")
        pc1 = 100.0 * float(vr[0]) if np.size(vr) >= 1 else np.nan
        pc2 = 100.0 * float(vr[1]) if np.size(vr) >= 2 else np.nan
        pc3 = 100.0 * float(vr[2]) if np.size(vr) >= 3 else np.nan
        ax.set_xlabel(f"PC1 ({pc1:.1f}%)")
        ax.set_ylabel(f"PC2 ({pc2:.1f}%)")
        ax.set_zlabel(f"PC3 ({pc3:.1f}%)")
        ax.view_init(elev=20, azim=42)

    combined = np.vstack([coords_bias, coords_full])
    _set_equal_3d_limits(ax_bias, combined)
    _set_equal_3d_limits(ax_full, combined)

    cax = fig.add_axes([0.30, 0.07, 0.40, 0.035])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cbar.set_label(r"Task angle $\theta / \pi$")
    cbar.set_ticks([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi])
    cbar.set_ticklabels(["0", "0.5", "1.0", "1.5", "2.0"])

    fig.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.16, wspace=0.06)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}")


def plot_talk_single_axis_update_reconstruction(results_dir: Path | str, out_path: Path | str, max_k: int | None = None):
    _require_matplotlib()
    results_dir = Path(results_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run = _load_single_axis_update_vectors(results_dir)
    bias_curve = _pca_reconstruction_curve(run["delta_bias"], max_k=max_k)
    full_curve = _pca_reconstruction_curve(run["delta_full"], max_k=max_k)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    ax_err, ax_var = axes

    for curve, color, label in [
        (bias_curve, "tab:blue", "Bias-only"),
        (full_curve, "tab:orange", "Full NTK"),
    ]:
        ax_err.plot(
            curve["k"],
            curve["relative_fro_error"],
            color=color,
            linewidth=2.3,
            marker="o",
            markersize=4.5,
            label=label,
        )
        ax_var.plot(
            curve["k"],
            curve["cumulative_var"],
            color=color,
            linewidth=2.3,
            marker="o",
            markersize=4.5,
            label=label,
        )

    ax_err.set_xlabel("Top-k PCA coordinates")
    ax_err.set_ylabel(r"Relative reconstruction error $\|U-\hat U_k\|_F / \|U-\bar U\|_F$")
    ax_err.set_title("Single-axis update-family compressibility")
    ax_err.set_yscale("log")
    ax_err.grid(True, which="both", alpha=0.25)
    ax_err.legend(frameon=False)

    ax_var.set_xlabel("Top-k PCA coordinates")
    ax_var.set_ylabel("Cumulative explained variance")
    ax_var.set_title("Variance captured by the shared update subspace")
    ax_var.set_ylim(0.0, 1.02)
    ax_var.grid(True, alpha=0.25)
    ax_var.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}")


def plot_talk_multitask_setup(
    out_path: Path | str,
    T_setup: int,
    T_list,
    pts_per_circle: int = 256,
    m: int = 4,
    seed: int = 0,
    normals_mode: str = "fibonacci",
    random_phase: bool = True,
):
    _require_matplotlib()
    if Axes3D is None:
        raise ImportError("3D matplotlib support is required for the multitask setup plot.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    task_data = sample_multitask_circle_family(
        K_tasks=int(T_setup),
        pts_per_circle=int(pts_per_circle),
        m=int(m),
        seed=int(seed),
        normals_mode=normals_mode,
        random_phase=random_phase,
    )

    fig = plt.figure(figsize=(13.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.15, 1.0), wspace=0.25)
    ax_sphere = fig.add_subplot(gs[0, 0], projection="3d")
    ax_space = fig.add_subplot(gs[0, 1])

    u = np.linspace(0.0, 2.0 * np.pi, 120)
    v = np.linspace(0.0, np.pi, 60)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax_sphere.plot_surface(
        xs,
        ys,
        zs,
        color="#d7dce2",
        alpha=0.16,
        linewidth=0.0,
        antialiased=True,
        shade=True,
    )
    ax_sphere.plot_wireframe(
        xs,
        ys,
        zs,
        rstride=10,
        cstride=10,
        color="gray",
        linewidth=0.45,
        alpha=0.22,
    )

    scatter_ref = None
    start_points = []
    for Xa, ya, normal in zip(task_data["Xs"], task_data["ys"], task_data["normals"]):
        scatter_ref = ax_sphere.scatter(
            Xa[:, 0],
            Xa[:, 1],
            Xa[:, 2],
            c=ya,
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            s=11,
            depthshade=False,
        )
        ax_sphere.plot(
            [0.0, normal[0]],
            [0.0, normal[1]],
            [0.0, normal[2]],
            color="black",
            linewidth=1.1,
            alpha=0.55,
        )
        if Xa.shape[0] > 0:
            start_points.append(np.asarray(Xa[0], dtype=float))

    if start_points:
        starts = np.asarray(start_points, dtype=float)
        ax_sphere.scatter(
            starts[:, 0],
            starts[:, 1],
            starts[:, 2],
            color="black",
            s=38,
            depthshade=False,
            zorder=12,
        )

    if scatter_ref is not None:
        cbar = fig.colorbar(
            scatter_ref,
            ax=ax_sphere,
            orientation="horizontal",
            fraction=0.055,
            pad=0.09,
            shrink=0.82,
        )
        cbar.set_label("Target value")

    ax_sphere.set_title(rf"(A) Example multitask geometry on $\mathbb{{S}}^2$ ($T={int(T_setup)}$)")
    ax_sphere.set_box_aspect((1.0, 1.0, 1.0))
    ax_sphere.set_xlim(-1.05, 1.05)
    ax_sphere.set_ylim(-1.05, 1.05)
    ax_sphere.set_zlim(-1.05, 1.05)
    ax_sphere.set_xticks([-1.0, 0.0, 1.0])
    ax_sphere.set_yticks([-1.0, 0.0, 1.0])
    ax_sphere.set_zticks([-1.0, 0.0, 1.0])
    ax_sphere.set_xlabel("x", labelpad=-6)
    ax_sphere.set_ylabel("y", labelpad=-6)
    ax_sphere.set_zlabel("z", labelpad=-4)
    ax_sphere.tick_params(labelsize=8, pad=0)
    ax_sphere.view_init(elev=22, azim=35)
    sphere_handles = [
        Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=6, label="random start"),
    ]
    ax_sphere.legend(handles=sphere_handles, frameon=False, loc="upper left")
    T_arr = np.asarray(T_list, dtype=int)
    mean_cosine_distance = []
    for T in T_arr:
        normals = normals_fibonacci_s2(int(T))
        stats = great_circle_cosine_distance_stats(normals)
        mean_cosine_distance.append(stats["mean_cosine_distance"])

    mean_cosine_distance = np.asarray(mean_cosine_distance, dtype=float)
    ax_space.plot(T_arr, mean_cosine_distance, marker="o", linewidth=2.1, color="tab:blue")
    ax_space.set_title(r"(B) Mean pairwise great-circle spacing under Fibonacci normals")
    ax_space.set_xlabel("Number of tasks T")
    ax_space.set_ylabel(r"Cosine distance $1-|n_i^\top n_j|$")
    ax_space.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}")


def plot_talk_fibonacci_normals(
    out_path: Path | str,
    T_values: tuple[int, ...] = (5, 10, 50),
):
    _require_matplotlib()
    if Axes3D is None:
        raise ImportError("3D matplotlib support is required for the Fibonacci normals plot.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13.6, 4.8))
    gs = fig.add_gridspec(1, len(T_values), wspace=0.08)

    u = np.linspace(0.0, 2.0 * np.pi, 120)
    v = np.linspace(0.0, np.pi, 60)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    for idx, T in enumerate(T_values):
        ax = fig.add_subplot(gs[0, idx], projection="3d")
        normals = normals_fibonacci_s2(int(T))
        ax.plot_surface(
            xs,
            ys,
            zs,
            color="#d7dce2",
            alpha=0.14,
            linewidth=0.0,
            antialiased=True,
            shade=True,
        )
        ax.plot_wireframe(
            xs,
            ys,
            zs,
            rstride=10,
            cstride=10,
            color="gray",
            linewidth=0.4,
            alpha=0.18,
        )
        ax.scatter(
            normals[:, 0],
            normals[:, 1],
            normals[:, 2],
            c=np.linspace(0.0, 1.0, len(normals)),
            cmap="viridis",
            s=46 if T <= 10 else 24,
            depthshade=False,
        )
        for normal in normals:
            ax.plot(
                [0.0, normal[0]],
                [0.0, normal[1]],
                [0.0, normal[2]],
                color="black",
                linewidth=0.8,
                alpha=0.35,
            )

        ax.set_title(rf"$T={int(T)}$")
        ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_zlim(-1.05, 1.05)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=22, azim=35)

    fig.suptitle(r"Fibonacci task normals on $\mathbb{S}^2$", y=0.97)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}")


def plot_talk_composite_complexity(
    npz_path: Path | str,
    out_path: Path | str,
    normalized: bool = True,
):
    _require_matplotlib()
    npz_path = Path(npz_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    D = np.load(npz_path, allow_pickle=True)
    try:
        T_list = np.asarray(D["T_list"], dtype=int)
        colors = {"bias_only": "tab:blue", "full_ntk": "tab:orange"}
        labels = {"bias_only": "bias-only", "full_ntk": "full NTK"}
        prefix = "normalized_total_norm" if normalized else "total_norm"
        ylabel = r"$\|f_{\mathrm{multi}}\|_{\mathcal{H}_{\tilde K}}$" if normalized else r"$\|f_{\mathrm{multi}}\|_{\mathcal{H}_K}$"
        title = r"Trace-normalized composite target complexity" if normalized else r"Composite target RKHS norm"

        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for kernel_name in ("bias_only", "full_ntk"):
            mean_key = f"{kernel_name}_{prefix}_mean"
            sem_key = f"{kernel_name}_{prefix}_sem"
            if mean_key not in D or sem_key not in D:
                continue
            _plot_mean_sem(
                ax,
                T_list,
                np.asarray(D[mean_key], dtype=float),
                np.asarray(D[sem_key], dtype=float),
                label=labels.get(kernel_name, kernel_name),
                color=colors.get(kernel_name),
            )

        ax.set_xlabel("Number of tasks T")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote: {out_path}")
    finally:
        D.close()


def make_analytic_eigenspectrum_plot(
    out_path: Path | str,
    p: int = 3,
    par: KernelParams | None = None,
    Lmax: int = 60,
    n_quad: int = 700,
    fit_range: tuple[int, int] = (1, 60),
):
    _require_matplotlib()
    if par is None:
        par = KernelParams(sigma_w2=1.0, sigma_b2=0.2, sigma_v2=1.0, sigma_bout2=1.0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mu_bias = compute_spectrum(Lmax, kappa_bias_only, par, n_quad=n_quad, p=p)
    mu_full = compute_spectrum(Lmax, kappa_full_ntk, par, n_quad=n_quad, p=p)

    ells = np.arange(Lmax + 1, dtype=float)

    C_b, s_b, r2pow_b = fit_powerlaw_log(mu_bias, *fit_range)
    C_f, s_f, r2pow_f = fit_powerlaw_log(mu_full, *fit_range)

    pow_fit_bias = np.full_like(ells, np.nan)
    pow_fit_full = np.full_like(ells, np.nan)
    mask = ells >= 1
    pow_fit_bias[mask] = C_b * (ells[mask] ** (-s_b))
    pow_fit_full[mask] = C_f * (ells[mask] ** (-s_f))

    ell_ref = 10
    mu_ref = float(mu_full[ell_ref])
    bietti = bietti_reference_powerlaw(Lmax, p=p, mu_ref=mu_ref, ell_ref=ell_ref)

    fig, ax = plt.subplots()

    ax.semilogy(ells, mu_bias, "o", markersize=3, label="bias-only μ_ℓ (numeric)", c="tab:blue")
    ax.semilogy(ells, mu_full, "o", markersize=3, label="full NTK μ_ℓ (numeric)", c="tab:orange")
    ax.semilogy(
        ells,
        pow_fit_bias,
        linewidth=2,
        label=f"bias power fit: s={s_b:.3g}, R²log={r2pow_b:.3f}",
        color="tab:blue",
    )
    ax.semilogy(
        ells,
        pow_fit_full,
        linewidth=2,
        label=f"full power fit: s={s_f:.3g}, R²log={r2pow_f:.3f}",
        color="tab:orange",
    )
    ax.semilogy(
        ells,
        bietti,
        "k--",
        linewidth=2,
        label=f"Bietti ref: ℓ^(-p), p={p}, normalized at ℓ={ell_ref}",
    )

    ax.set_xlabel("harmonic degree ℓ")
    ax.set_ylabel("eigenvalue μ_ℓ")
    ax.set_title(f"Biased ReLU NTK eigenspectra on S^{p-1}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Fit range:", fit_range)
    print(f"Bias pow:  C={C_b:.3g}, s={s_b:.3g}, R2_log={r2pow_b:.3f}")
    print(f"Full pow:  C={C_f:.3g}, s={s_f:.3g}, R2_log={r2pow_f:.3f}")
    print(f"Wrote: {out_path}")


def main():
    repo_root = next(parent for parent in (p.parent, *p.parents) if (parent / "multiTasking").is_dir())
    results_dir = repo_root / "multiTasking" / "results" / "eigen_spectra_results"
    figures_dir = repo_root / "multiTasking" / "ICMNS_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    stem = "eigen_spectra_multitask_cntrl_tot_pts_out_bias_valset"
    ridge_stem = stem.replace("eigen_spectra_multitask_", "")
    single_axis_run_dir = (
        repo_root
        / "multiTasking"
        / "scripts"
        / "results"
        / "single_axis_seed0_K30_m3_pts300_dim16384_tphase1p571"
    )

    if plt is not None:
        make_analytic_eigenspectrum_plot(
            out_path=figures_dir / "analytic_eigenspectra_sphere.png",
            p=3,
            par=KernelParams(sigma_w2=1.0, sigma_b2=0.2, sigma_v2=1.0, sigma_bout2=1.0),
            Lmax=60,
            n_quad=700,
            fit_range=(1, 60),
        )
    else:
        print("Skipping analytic eigenspectrum plot because matplotlib is unavailable.")

    empirical_npz = results_dir / f"eigen_spectra_by_T_{stem}.npz"
    empirical_meta = results_dir / f"eigen_spectra_by_T_{stem}_meta.json"
    if empirical_npz.exists():
        meta = {}
        if empirical_meta.exists():
            with open(empirical_meta, "r") as f:
                meta = json.load(f)

        summary = summarize_saved_eigenspectra(
            npz_path=empirical_npz,
            meta_path=empirical_meta,
            normalize="trace",
            fit_rank_window=(10, 200),
            include_analytic_budget_reference=False,
        )
        talk_summary_npz = results_dir / f"talk_spectral_summary_{stem}.npz"
        payload = {
            "normalize": np.asarray([summary["normalize"]]),
            "fit_rank_window": np.asarray(summary["fit_rank_window"], dtype=int),
            "head_mass_rank": np.asarray([summary.get("head_mass_rank", 10)], dtype=int),
        }
        for kernel_name, kernel_payload in summary["kernels"].items():
            payload[f"{kernel_name}_T_list"] = np.asarray(kernel_payload["T_list"], dtype=int)
            for key, value in kernel_payload.items():
                if key == "T_list":
                    continue
                payload[f"{kernel_name}_{key}"] = np.asarray(value)
        np.savez_compressed(talk_summary_npz, **payload)
        print(f"Wrote: {talk_summary_npz}")

        if plt is not None:
            plot_talk_fibonacci_normals(
                out_path=figures_dir / f"talk_fibonacci_normals_{ridge_stem}.png",
                T_values=(5, 10, 50),
            )

            plot_talk_multitask_setup(
                out_path=figures_dir / f"talk_multitask_setup_{ridge_stem}.png",
                T_setup=min(5, int(np.max(meta.get("T_list", [5])))),
                T_list=meta.get("T_list", summary["kernels"]["full_ntk"]["T_list"]),
                pts_per_circle=max(192, int(meta.get("pts_per_circle", 64))),
                m=int(meta.get("m", 4)),
                seed=0,
                normals_mode=str(meta.get("normals_mode", "fibonacci")),
                random_phase=bool(meta.get("random_phase", True)),
            )

            plot_talk_empirical_spectrum_shape(
                summary=summary,
                npz_path=empirical_npz,
                out_path=figures_dir / f"talk_empirical_spectrum_shape_{stem}.png",
                kernel_name="full_ntk",
                normalize="trace",
                L_max=500,
            )

            plot_talk_budget_proxy(
                summary=summary,
                npz_path=empirical_npz,
                out_path=figures_dir / f"talk_budget_proxy_{stem}.png",
            )

            ridge_npz = repo_root / "multiTasking" / "results" / "ridge_tuning_results" / f"ridge_tuning_by_T_{ridge_stem}.npz"
            eval_npz = repo_root / "multiTasking" / "results" / "ridge_eval_results" / f"frozen_ridge_eval_by_T_{ridge_stem}.npz"
            if ridge_npz.exists() and eval_npz.exists():
                plot_talk_frozen_eval_mse(
                    ridge_npz_path=ridge_npz,
                    eval_npz_path=eval_npz,
                    out_path=figures_dir / f"talk_frozen_mse_{ridge_stem}.png",
                )
            else:
                print(f"Skipping frozen-eval MSE plot; missing {ridge_npz} or {eval_npz}")

            composite_npz = results_dir / f"composite_target_rkhs_norms_{stem}.npz"
            if composite_npz.exists():
                plot_talk_composite_complexity(
                    npz_path=composite_npz,
                    out_path=figures_dir / f"talk_composite_complexity_{ridge_stem}.png",
                    normalized=True,
                )
            else:
                print(f"Skipping composite complexity plot; missing {composite_npz}")
        else:
            print("Skipping talk plot rendering because matplotlib is unavailable.")
    else:
        print(f"Skipping talk empirical plots; missing {empirical_npz}")

    if plt is not None:
        if single_axis_run_dir.exists():
            run_stem = single_axis_run_dir.name
            plot_talk_single_axis_geometry_overview(
                results_dir=single_axis_run_dir,
                out_path=figures_dir / f"talk_single_axis_geometry_overview_{run_stem}.png",
            )
            plot_talk_single_axis_update_reconstruction(
                results_dir=single_axis_run_dir,
                out_path=figures_dir / f"talk_single_axis_update_reconstruction_{run_stem}.png",
            )
            plot_talk_single_axis_update_similarity(
                results_dir=single_axis_run_dir,
                out_path=figures_dir / f"talk_single_axis_update_similarity_{run_stem}.png",
            )
            plot_talk_single_axis_update_manifolds(
                results_dir=single_axis_run_dir,
                out_path=figures_dir / f"talk_single_axis_update_manifolds_{run_stem}.png",
            )
        else:
            print(f"Skipping single-axis manifold talk plots; missing {single_axis_run_dir}")


if __name__ == "__main__":
    main()
