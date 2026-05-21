
import jax.numpy as jnp
import numpy as np
import jax
from jax import grad, jit, random, jacrev, vmap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib.colors import Normalize
from multiTasking.utils.utils import KernelParams, make_kappa_kernel_fn
try:
    import neural_tangents as nt
except Exception:
    nt = None


def generate_single_circle(normal, pts, m, phase=0.0):
    
    # Normalize just in case
    n = normal / jnp.linalg.norm(normal)
    v1 = jnp.array([1., 0., 0.]) if abs(n[0]) < 0.9 else jnp.array([0., 1., 0.])
    v1 = v1 - n * jnp.dot(n, v1)
    v1 /= jnp.linalg.norm(v1)
    v2 = jnp.cross(n, v1)

    # Apply phase shift directly to phi
    phi = jnp.linspace(0, 2*jnp.pi, pts, endpoint=False) + phase
    phi = phi % (2*jnp.pi)

    # Geometry on the sphere
    X = jnp.outer(jnp.cos(phi), v1) + jnp.outer(jnp.sin(phi), v2)

    # Circular harmonic ground truth
    y = jnp.sin(m * phi)

    return X, y

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
    random_phase=False
):
    normals = generate_normals(K_tasks)

    X_list, y_list, slices = [], [], []
    start = 0

    for j, n in enumerate(normals):
        if random_phase:
            phase_j = jnp.random.rand() * 2*jnp.pi
        else:
            phase_j = phase

        X, _ = generate_single_circle(n, pts_per_circle, m, phase=phase_j)
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
    
    axis = axis / jnp.linalg.norm(axis)
    n = base_normal / jnp.linalg.norm(base_normal)
    n_rot = (n * jnp.cos(angle) + jnp.cross(axis, n) * jnp.sin(angle) + axis * jnp.dot(axis, n) * (1 - jnp.cos(angle)))

    return n_rot / jnp.linalg.norm(n_rot)

def cosine_matrix(V, eps=1e-12):
    V = np.stack(V)
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + eps)
    return Vn @ Vn.T

def great_circle_distance(n1, n2):
    c = np.abs(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2)))
    c = np.clip(c, -1.0, 1.0)
    return np.arccos(c)

def empirical_rkhs_task_similarity(Jbs, alphas, eps=1e-12):
    T = len(Jbs)
    G = np.zeros((T, T))

    for i in range(T):
        Ji = Jbs[i]
        ai = alphas[i]
        Kii = Ji @ Ji.T
        ni = ai @ Kii @ ai

        for j in range(T):
            Jj = Jbs[j]
            aj = alphas[j]
            Kjj = Jj @ Jj.T
            nj = aj @ Kjj @ aj

            Kij = Ji @ Jj.T
            num = ai @ Kij @ aj
            G[i, j] = num / (np.sqrt(ni * nj) + eps)

    return G

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
        Kii = np.array(kernel_fn(Xs[i], Xs[i]))
        ai = np.array(alphas[i])
        ni = ai @ Kii @ ai
        norms.append(ni)

    for i in range(T):
        ai = np.array(alphas[i])
        for j in range(T):
            aj = np.array(alphas[j])
            Kij = np.array(kernel_fn(Xs[i], Xs[j]))
            num = ai @ Kij @ aj
            G[i, j] = num / (np.sqrt(norms[i] * norms[j]) + eps)

    return G

def flatten_pytree_jac(J_tree):
    leaves, _ = jax.tree_util.tree_flatten(J_tree)
    return jnp.concatenate([leaf.reshape(leaf.shape[0], -1) for leaf in leaves], axis=1)


def solve_tangent_update(J, y, f0, reg):
    K = J @ J.T
    lam = reg * jnp.max(jnp.linalg.eigvalsh(K))
    alpha = jnp.linalg.solve(K + lam * jnp.eye(K.shape[0]), y - f0)
    delta = J.T @ alpha
    return alpha, delta, K

def solve_tangent_update_kernel(K, y, f0, reg):
    lam = reg * jnp.max(jnp.linalg.eigvalsh(K))
    alpha = jnp.linalg.solve(K + lam * jnp.eye(K.shape[0]), y - f0)
    return alpha

# Suite for analysis
def run_full_ntk_analysis(
        dim=8192, 
        K_tasks=36, 
        m=3, 
        bias_std=0.0, 
        pts=300, 
        task_plot_id=0,
        reg=1e-5):
    
    #analytic infinite width kernel set up:
    W_std = 1.0
    b_std = 1.0
    kappa_params = KernelParams(sigma_w2=W_std**2, sigma_b2=b_std**2, sigma_v2=1.0, sigma_bout2=1.0)
    kappa_scale = 1.0  # keep 1.0 unless you decide to calibrate
    
    key = random.PRNGKey(0)
    base_n = jnp.array([0., 0., 1.])    # Normal for task 0
    rot_ax = jnp.array([1., 0., 0.])    # Rotational axis
    angles = jnp.linspace(0, 2*jnp.pi, K_tasks, endpoint=False)
    
    params0 = init_mlp_params(key, 3, dim, bias_std=bias_std)
    delta_bs = []
    
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
    alphas_bs = []
    alpha_fulls = []
    alphas_inf_bs = []
    alphas_inf_fulls = []
    delta_bs = []
    delta_fulls = []
    normals = []
    
    #perform analysis for infinite width NTK
    kappa_kernel_fn_full = make_kappa_kernel_fn(kappa_params, which="full", kappa_scale=kappa_scale)
    kappa_kernel_fn_bias = make_kappa_kernel_fn(kappa_params, which="bias", kappa_scale=kappa_scale)
    # batch for memory
    if nt is None:
        raise ImportError("neural_tangents not available for nt.batch; install neural_tangents or remove batching.")
    kappa_kernel_fn_full = nt.batch(kappa_kernel_fn_full, device_count=-1)
    kappa_kernel_fn_bias = nt.batch(kappa_kernel_fn_bias, device_count=-1)

    for i, angle in enumerate(angles):
        # Get normal and generate X,y for sin(m phi) on that circle. 
        n = rotate_normal_around_axis(base_n, rot_ax, angle)
        X, y = generate_single_circle(n, pts, m)
        f0 = mlp_apply(params0, X)

        # Get bias terms of the Jacobian
        J_tree = vmap(jacrev(f_single), (None, 0))(params0, X)
        Jb = jnp.concatenate([J_tree["b1"], J_tree["b2"][:, None]], axis=1)
        Jfull = flatten_pytree_jac(J_tree)

        # Solve for delta_b
        alpha_b, db, Kb = solve_tangent_update(Jb, y, f0, reg)
        alpha_full, d_full, K_full = solve_tangent_update(Jfull, y, f0, reg)

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
                'phi': np.linspace(0, 2*jnp.pi, len(y)),
                'y': y_centered,
                'f_lin': f_lin - jnp.mean(f_lin),
                'f_nonlin': f_nonlin - jnp.mean(f_nonlin),
                'mse_lin': mse_lin,
                'mse_nonlin': mse_nonlin
            }
        
        
        K_full_inf = jnp.array(kappa_kernel_fn_full(X, X, get="ntk"))
        K_bias_inf = jnp.array(kappa_kernel_fn_bias(X, X, get="ntk"))
        alphas_bs_inf_width = solve_tangent_update_kernel(K_bias_inf, y, f0, reg)
        alphas_full_inf_width = solve_tangent_update_kernel(K_full_inf, y, f0, reg)
        alphas_inf_bs.append(np.array(alphas_bs_inf_width))
        alphas_inf_fulls.append(np.array(alphas_full_inf_width))

        Xs.append(np.array(X))
        ys.append(np.array(y))
        f0s.append(np.array(f0))
        Jbs.append(np.array(Jb))
        Jfulls.append(np.array(Jfull))
        Kbs.append(np.array(Kb))
        Kfulls.append(np.array(K_full))
        alphas_bs.append(np.array(alpha_b))
        delta_bs.append(np.array(db))
        alpha_fulls.append(np.array(alpha_full))
        delta_fulls.append(np.array(d_full))
        normals.append(np.array(n))
        

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

    fig = plt.figure(figsize=(10, 9))

    # 1. delta_b Cosine Similarity Heatmap
    ax1 = fig.add_subplot(221)
    sns.heatmap(sim_mat, cmap='viridis', ax=ax1, cbar_kws={'label': 'Cosine Similarity'})
    ax1.invert_yaxis()
    ax1.set_title(r"Cosine Similarity between $\Delta_b$s")
    ax1.set_xlabel("Task Index")
    ax1.set_ylabel("Task Index")

    # 2. Similarity vs Angular Distance + Linear Fit and R^2
    ax2 = fig.add_subplot(222)
    dist_list, sim_list = [], []
    for i in range(K_tasks):
        for j in range(i+1, K_tasks):
            # We are sampling the whole 2pi, so if we have N tasks,
            # task 1 is as similar to task 0 as task N. To reflect this
            # we check the min between the abs diff and 360 - abs diff
            diff = jnp.abs(angles[i] - angles[j])
            dist_val = jnp.minimum(diff, 2*jnp.pi - diff)
            dist_list.append(dist_val)
            sim_list.append(sim_mat[i, j])
    
    dists, sims = np.array(dist_list), np.array(sim_list)
    ax2.scatter(dists, sims, alpha=0.4, s=15, label='Task Pairs')
    
    # Do linear fit, because it looks linear
    slope, intercept = np.polyfit(dists, sims, 1)
    r_squared = np.corrcoef(dists, sims)[0, 1]**2
    x_range = np.linspace(dists.min(), dists.max(), 100)
    fit_line = slope * x_range + intercept
    
    sign = "+" if intercept >= 0 else "-"
    label_str = f'Fit: $y = {slope:.4f}x {sign} {abs(intercept):.3f}$\n$R^2 = {r_squared:.3f}$'
    
    ax2.plot(x_range, fit_line, color='red', lw=2, label=label_str)
    ax2.set_title("NTK bias similarity vs Task similarity")
    ax2.set_xlabel("Angular Distance (rad)")
    ax2.set_ylabel(r"$\Delta_b$ Cosine Similarity")
    ax2.legend(loc='upper right', fontsize='small', frameon=True)
    ax2.grid(True, alpha=0.2)

    # 3. PCA Manifold
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(DB_matrix)
    var_exp = pca.explained_variance_ratio_ * 100 

    ax3 = fig.add_subplot(223, projection='3d')
    sc = ax3.scatter(pcs[:, 0], pcs[:, 1], pcs[:, 2], 
                    c=np.degrees(angles), cmap='hsv', s=50, edgecolors='k', linewidth=0.5)
    ax3.plot(pcs[:, 0], pcs[:, 1], pcs[:, 2], 'k-', alpha=0.3)
    ax3.plot([pcs[-1, 0], pcs[0, 0]], [pcs[-1, 1], pcs[0, 1]], [pcs[-1, 2], pcs[0, 2]], 'k--', alpha=0.3)

    ax3.set_title(f"3D Bias Update Manifold\nTotal Var: {np.sum(var_exp):.1f}%")
    ax3.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
    ax3.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")
    ax3.set_zlabel(f"PC3 ({var_exp[2]:.1f}%)")
    ax3.view_init(elev=20, azim=45)

    # 4. Performance Check
    ax4 = fig.add_subplot(224)
    ax4.plot(test_results['phi'], test_results['y'], 'k', alpha=0.3, lw=4, label='Target')
    ax4.plot(test_results['phi'], test_results['f_lin'], 'r--', label=f"Lin (MSE: {test_results['mse_lin']:.2e})")
    ax4.plot(test_results['phi'], test_results['f_nonlin'], 'b:', label=f"Nonlin (MSE: {test_results['mse_nonlin']:.2e})")
    ax4.set_title(f"Task {task_plot_id} Performance Check")
    ax4.legend(loc='upper right', fontsize='x-small')
    ax4.set_xlabel(r"$\phi$ (rad)")

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # Load multitask MSE results
    # ------------------------------------------------------------
    addn_str = 'cntrl_tot_pts_out_bias'
    D = np.load(
        f"/home/dburnham/Documents/NTK/neural-tangents/multiTasking/results/"
        f"ridge_tuning_results/ridge_tuning_by_T_{addn_str}.npz"
    )

    T_list = D["T_list"]

    mse_star_full = D["mse_star_full_ntk"]
    mse_star_bias = D["mse_star_bias_only"]

    # ------------------------------------------------------------
    # Fellowship figure: 3 panels
    #   A) similarity plot + inset schematic
    #   B) PCA manifold
    #   C) multitask performance vs task load
    # ------------------------------------------------------------
    title_fs = 16
    label_fs = 14
    tick_fs = 12
    legend_fs = 12
    cbar_fs = 11
    point_size = 24
    pca_point_size = 65
    line_fs = 10
    panel_fs = 18

    fig2 = plt.figure(figsize=(18, 5.4))

    # ============================================================
    # Panel A: similarity vs distance
    # ============================================================
    axA = fig2.add_subplot(131)

    axA.scatter(dists, sims, alpha=0.45, s=point_size, label='Task pairs')
    axA.plot(
        x_range, fit_line,
        color='k', lw=2.5, ls='--',
        label=label_str
    )

    axA.set_title("Task geometry predicts bias-update similarity", fontsize=title_fs)
    axA.set_xlabel(r"Task distance, $\Delta \theta$ (rad)", fontsize=label_fs)
    axA.set_ylabel(r"$\Delta_b$ cosine similarity", fontsize=label_fs)
    axA.tick_params(axis='both', labelsize=tick_fs)
    axA.grid(True, alpha=0.25)

    leg = axA.legend(
        loc='upper right',
        fontsize=legend_fs,
        frameon=True,
        borderpad=0.8,
        labelspacing=0.7,
        handlelength=2.0
    )
    leg.get_frame().set_alpha(0.95)

    # ---------------- inset schematic ----------------
    axS = fig2.add_axes([-0.015, 0.11, 0.28, 0.49], projection='3d')

    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 60)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    axS.plot_surface(xs, ys, zs, alpha=0.10, linewidth=0, color='lightgray')

    def circle_basis(normal):
        n = np.asarray(normal, dtype=float)
        n = n / np.linalg.norm(n)
        trial = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(trial, n)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        u_vec = trial - np.dot(trial, n) * n
        u_vec = u_vec / np.linalg.norm(u_vec)
        v_vec = np.cross(n, u_vec)
        v_vec = v_vec / np.linalg.norm(v_vec)
        return u_vec, v_vec

    idx0 = 0
    idx1 = max(1, K_tasks // 6)

    n0 = np.array(rotate_normal_around_axis(base_n, rot_ax, angles[idx0]), dtype=float)
    n1 = np.array(rotate_normal_around_axis(base_n, rot_ax, angles[idx1]), dtype=float)

    phi_ref = np.linspace(0, 2*np.pi, 300, endpoint=False)
    u0, v0 = circle_basis(n0)
    circle_0 = np.outer(np.cos(phi_ref), u0) + np.outer(np.sin(phi_ref), v0)
    axS.plot(circle_0[:, 0], circle_0[:, 1], circle_0[:, 2], lw=1.5, alpha=0.55, color='black')

    phi_task = np.linspace(0, 2*np.pi, 300, endpoint=False)
    u1, v1 = circle_basis(n1)
    circle_1 = np.outer(np.cos(phi_task), u1) + np.outer(np.sin(phi_task), v1)
    y_task = np.sin(m * phi_task)

    axS.scatter(
        circle_1[:, 0], circle_1[:, 1], circle_1[:, 2],
        c=y_task, cmap='coolwarm', s=9, alpha=0.95, linewidths=0
    )
    axS.plot(circle_1[:, 0], circle_1[:, 1], circle_1[:, 2], color='k', alpha=0.10, lw=0.8)

    axS.quiver(0, 0, 0, n0[0], n0[1], n0[2], color='black', linewidth=1.5, arrow_length_ratio=0.10)
    axS.quiver(0, 0, 0, n1[0], n1[1], n1[2], color='red', linewidth=1.5, arrow_length_ratio=0.10)

    omega = np.arccos(np.clip(np.dot(n0, n1), -1.0, 1.0))
    if omega > 1e-6:
        ts = np.linspace(0, 1, 100)
        arc = np.array([
            (np.sin((1 - s) * omega) / np.sin(omega)) * n0
            + (np.sin(s * omega) / np.sin(omega)) * n1
            for s in ts
        ])
        arc = 0.75 * arc / np.linalg.norm(arc, axis=1, keepdims=True)
        axS.plot(arc[:, 0], arc[:, 1], arc[:, 2], 'r--', lw=1.1, alpha=0.9)
        
    mid = arc[len(arc) // 2]
    mid_label = 1.18 * mid
    axS.text(
        mid_label[0], mid_label[1]-0.1, mid_label[2]-0.1,
        r'$\Delta \theta$',
        color='red',
        fontsize=10
    )

    axS.set_box_aspect([1, 1, 1])
    axS.set_xlim([-1.1, 1.1]); axS.set_ylim([-1.1, 1.1]); axS.set_zlim([-1.1, 1.1])
    axS.set_xticks([]); axS.set_yticks([]); axS.set_zticks([])
    axS.view_init(elev=20, azim=40)
    axS.set_facecolor((1, 1, 1, 0))

    # ============================================================
    # Panel B: PCA manifold
    # ============================================================
    axB = fig2.add_subplot(132, projection='3d')

    angle_radians = np.array(angles)
    norm = Normalize(vmin=0, vmax=2*np.pi)

    sc = axB.scatter(
        pcs[:, 0], pcs[:, 1], pcs[:, 2],
        c=angle_radians,
        cmap='twilight_shifted',
        norm=norm,
        s=pca_point_size,
        edgecolors='k',
        linewidth=0.6
    )

    axB.plot(pcs[:, 0], pcs[:, 1], pcs[:, 2], 'k-', alpha=0.35, lw=1.2)
    axB.plot(
        [pcs[-1, 0], pcs[0, 0]],
        [pcs[-1, 1], pcs[0, 1]],
        [pcs[-1, 2], pcs[0, 2]],
        'k--', alpha=0.35, lw=1.0
    )

    axB.set_title("Low-dimensional manifold of bias updates", fontsize=title_fs, pad=14)
    axB.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)", fontsize=label_fs, labelpad=10)
    axB.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)", fontsize=label_fs, labelpad=10)
    axB.set_zlabel(f"PC3 ({var_exp[2]:.1f}%)", fontsize=label_fs, labelpad=8)
    axB.tick_params(axis='both', labelsize=tick_fs)
    axB.view_init(elev=20, azim=45)

    # Horizontal colorbar below panel B
    cbar = fig2.colorbar(
        sc, ax=axB,
        orientation='horizontal',
        fraction=0.05,   # thickness
        pad=0.10,        # distance below subplot
        shrink=0.72
    )
    cbar.set_label("Task angle (rad)", fontsize=label_fs, labelpad=6)
    cbar.set_ticks([0, np.pi, 2*np.pi])
    cbar.set_ticklabels(["0", r"$\pi$", r"$2\pi$"])
    cbar.ax.tick_params(labelsize=cbar_fs)
    cbar.ax.xaxis.set_label_position('bottom')

    # ============================================================
    # Panel C: MSE vs number of tasks
    # ============================================================
    axC = fig2.add_subplot(133)

    # softer, more consistent colors
    full_color = '0.20'       # dark gray
    bias_color = '#c44e52'    # muted red
    base_color = '0.65'       # light gray

    axC.plot(
        T_list, mse_star_full,
        marker="o", ms=5, lw=2.4,
        color=full_color,
        label="Full NTK"
    )
    axC.plot(
        T_list, mse_star_bias,
        marker="o", ms=5, lw=2.4,
        color=bias_color,
        label="Bias-only NTK"
    )
    axC.axhline(
        0.5,
        linestyle="--", lw=1.6,
        color=base_color,
        label="Baseline"
    )
    

    axC.set_yscale('log')
    axC.set_xlabel(r"Number of tasks, $T$", fontsize=label_fs)
    axC.set_ylabel("Best-tuned MSE", fontsize=label_fs)
    axC.set_title("Multitask performance under increasing task load", fontsize=title_fs)
    axC.tick_params(axis='both', labelsize=tick_fs)
    axC.grid(True, alpha=0.22)

    legC = axC.legend(
        loc='lower right',
        fontsize=legend_fs,
        frameon=True,
        borderpad=0.6,
        labelspacing=0.5,
        handlelength=2.0
    )
    legC.get_frame().set_alpha(0.92)
    legC.get_frame().set_linewidth(0.8)

    # Panel labels
    fig2.text(0.005, 0.88, "(A)", fontsize=panel_fs, fontweight='bold')
    fig2.text(0.355, 0.88, "(B)", fontsize=panel_fs, fontweight='bold')
    fig2.text(0.675, 0.88, "(C)", fontsize=panel_fs, fontweight='bold')

    plt.tight_layout()
    fig2.savefig(
        "fellowship_bias_learning_figure_3panel.png",
        dpi=400,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.show()