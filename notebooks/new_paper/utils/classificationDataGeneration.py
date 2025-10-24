# Required: numpy, scipy, matplotlib
import numpy as np
from itertools import product
from math import ceil
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib import cm

def create_centers_grid(N, K, M, random_state=None):
    """
    Place K centers approximately evenly in [-M/2, M/2]^N by forming an L^N grid
    with L = ceil(K^(1/N)) then selecting K evenly spaced grid points.
    Returns centers: (K, N)
    """
    rng = np.random.default_rng(random_state)
    L = int(ceil(K ** (1.0 / N)))
    
    # positions centered around zero
    # spacing between adjacent centers = M / L
    positions_1d = np.linspace(-M/2 + M/(2*L), M/2 - M/(2*L), L)
    
    # full grid
    grid = np.array(list(product(positions_1d, repeat=N)))
    
    # if we have more grid points than needed, sample evenly spaced indices
    if len(grid) == K:
        centers = grid
    else:
        idxs = np.linspace(0, len(grid) - 1, K, dtype=int)
        centers = grid[idxs]
    
    rng.shuffle(centers)
    return centers

def sigma_from_overlap(centers, overlap):
    """
    Compute isotropic sigma so a Gaussian evaluated at the midpoint between nearest
    centers equals `overlap` (0 < overlap < 1).
    Derivation:
      overlap = exp(-(d/2)^2 / (2 sigma^2)) = exp(-d^2 / (8 sigma^2))
      => sigma = d / sqrt(8 * -ln(overlap))
    where d is the minimal center-to-center distance.
    """
    if not (0 < overlap < 1):
        raise ValueError("overlap must be in (0,1)")
    from scipy.spatial.distance import pdist
    dists = pdist(centers)
    if len(dists) == 0:
        return 0.25 * np.min(np.ptp(centers, axis=0) + 1e-6)
    dmin = np.min(dists)
    sigma = dmin / np.sqrt(8.0 * (-np.log(overlap)))
    return sigma

def sample_gaussians(centers, sigma, samples_per_gaussian, rng=None):
    """
    Sample isotropic Gaussian points for each center.
    Returns X (K*samples_per_gaussian, N) and labels (indices of gaussian).
    """
    rng = np.random.default_rng(rng)
    K, N = centers.shape
    total = K * samples_per_gaussian
    X = np.empty((total, N))
    y = np.empty(total, dtype=int)
    for i, c in enumerate(centers):
        start = i * samples_per_gaussian
        end = start + samples_per_gaussian
        X[start:end] = rng.normal(loc=c, scale=sigma, size=(samples_per_gaussian, N))
        y[start:end] = i
    return X, y

def make_classification_dataset(N, K, M, C, P, overlap=0.5, random_state=None):
    """
    Create classification dataset:
      - Place K gaussians in N-d hypercube [0,M]^N
      - Compute sigma via overlap
      - Assign gaussians to C classes (as evenly as possible)
      - Sample P total points equally across classes, split evenly among the gaussians
    Returns dict: {'X': X, 'y': y, 'assignment': assignment, 'centers': centers, 'sigma': sigma}
      assignment: {gaussian_index: class_label}
    """
    rng = np.random.default_rng(random_state)
    centers = create_centers_grid(N, K, M, random_state=random_state)
    sigma = sigma_from_overlap(centers, overlap)
    base = K // C
    extra = K % C
    assignment = {}
    gaussian_indices = np.arange(K)
    rng.shuffle(gaussian_indices)
    ptr = 0
    class_to_gaussians = {}
    for cls in range(C):
        take = base + (1 if cls < extra else 0)
        assigned = gaussian_indices[ptr:ptr+take].tolist()
        for g in assigned:
            assignment[int(g)] = int(cls)
        class_to_gaussians[cls] = assigned
        ptr += take
    samples_per_class = [P // C + (1 if i < (P % C) else 0) for i in range(C)]
    X_list, y_list = [], []
    for cls in range(C):
        gauss_for_cls = class_to_gaussians[cls]
        if len(gauss_for_cls) == 0:
            continue
        pts = samples_per_class[cls]
        pts_per_g = [pts // len(gauss_for_cls) + (1 if j < (pts % len(gauss_for_cls)) else 0) for j in range(len(gauss_for_cls))]
        for g_idx, num in zip(gauss_for_cls, pts_per_g):
            samples, _ = sample_gaussians(centers[[g_idx]], sigma, num, rng=rng)
            X_list.append(samples)
            y_list.append(np.full(num, cls, dtype=int))
    if len(X_list) == 0:
        return {'X': np.empty((0, N)), 'y': np.empty((0,), dtype=int), 'assignment': assignment, 'centers': centers, 'sigma': sigma}
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]
    return {'X': X, 'y': y, 'assignment': assignment, 'centers': centers, 'sigma': sigma}

# 2D plotting helpers
def plot_2d_kde_and_samples(centers, sigma, samples_per_gaussian=300, overlap=None, show=True, ax=None):
    """
    2D-only: sample points and plot KDE contours + scatter; each gaussian gets a unique color.
    """
    if centers.shape[1] != 2:
        raise ValueError("plot_2d_kde_and_samples is for 2D only")
    K = centers.shape[0]
    X, y = sample_gaussians(centers, sigma, samples_per_gaussian, rng=0)
    pad = sigma * 3.0 + 0.1 * np.max(centers.max(axis=0) - centers.min(axis=0))
    xmin, ymin = centers.min(axis=0) - pad
    xmax, ymax = centers.max(axis=0) + pad
    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])
    cmap = cm.get_cmap('tab20' if K<=20 else 'viridis', K)
    colors = [cmap(i) for i in range(K)]
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))
    for i in range(K):
        pts = X[y==i].T
        if pts.shape[1] < 2:
            continue
        try:
            std_mean = np.std(pts, axis=1).mean()
            bw = sigma / std_mean if std_mean > 0 else None
            kde = gaussian_kde(pts, bw_method=bw)
            zi = kde(grid_coords).reshape(xx.shape)
            ax.contour(xx, yy, zi, levels=3, alpha=0.6, colors=[colors[i]])
        except Exception:
            pass
        ax.scatter(pts[0], pts[1], s=6, color=colors[i], alpha=0.6, label=f'g{i}', edgecolors='none')
        ax.scatter(centers[i,0], centers[i,1], marker='x', color='k', s=30)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    title = f"2D Gaussians: K={K}, sigma={sigma:.3g}" + (f", overlap={overlap}" if overlap is not None else "")
    ax.set_title(title); ax.set_aspect('equal', 'box')
    if show:
        ax.legend(ncol=min(4, K), fontsize='small', markerscale=2)
        plt.show()
    return ax

# 2D plotting helpers
def plot_2d_kde(centers, sigma, X, y, show=True, ax=None):
    """
    2D-only: sample points and plot KDE contours + scatter; each gaussian gets a unique color.
    """
    if centers.shape[1] != 2:
        raise ValueError("plot_2d_kde is for 2D only")
    K = centers.shape[0]
    pad = sigma * 3.0 + 0.1 * np.max(centers.max(axis=0) - centers.min(axis=0))
    xmin, ymin = centers.min(axis=0) - pad
    xmax, ymax = centers.max(axis=0) + pad
    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])
    cmap = cm.get_cmap('tab20' if K<=20 else 'viridis', K)
    colors = [cmap(i) for i in range(K)]
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))
    for i in range(K):
        pts = X[y==i].T
        if pts.shape[1] < 2:
            continue
        try:
            std_mean = np.std(pts, axis=1).mean()
            bw = sigma / std_mean if std_mean > 0 else None
            kde = gaussian_kde(pts, bw_method=bw)
            zi = kde(grid_coords).reshape(xx.shape)
            ax.contour(xx, yy, zi, levels=3, alpha=0.6, colors=[colors[i]])
        except Exception:
            pass
        ax.scatter(pts[0], pts[1], s=6, color=colors[i], alpha=0.6, label=f'g{i}', edgecolors='none')
    for i in range(centers.shape[0]):
        ax.scatter(centers[i,0], centers[i,1], marker='x', color='k', s=30)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    title = f"2D Gaussians: K={K}, sigma={sigma:.3g}"
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    if show:
        ax.legend(ncol=min(4, K), fontsize='small', markerscale=2)
        plt.show() 
    return ax
    
def plot_train_test_split(X_train, y_train, X_test, y_test, centers, title="Train/Test Split Visualization"):
    """
    Visualize 2D train/test data for arbitrary number of classes.
    Each (train/test, class) pair gets its own color.
    Train = filled circles; Test = hollow circles.
    """
    if X_train.shape[1] != 2:
        raise ValueError("This visualization only supports 2D data.")
    
    classes = np.unique(np.concatenate([y_train, y_test]))
    n_classes = len(classes)
    
    # Generate distinct colors for train/test sets of each class
    cmap = cm.get_cmap('tab20' if 2 * n_classes <= 20 else 'hsv', 2 * n_classes)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    for i, cls in enumerate(classes):
        # two unique colors for this class (train/test)
        color_train = cmap(2 * i)
        color_test  = cmap(2 * i + 1)
        
        # --- TRAIN ---
        train_mask = y_train == cls
        ax.scatter(X_train[train_mask, 0], X_train[train_mask, 1],
                   color=color_train, s=30, alpha=0.8,
                   marker='o', edgecolors='none',
                   label=f"train c{cls}")
        
        # --- TEST ---
        test_mask = y_test == cls
        ax.scatter(X_test[test_mask, 0], X_test[test_mask, 1],
                   facecolors='none', edgecolors=color_test,
                   s=50, alpha=0.9, marker='o', linewidth=1.2,
                   label=f"test c{cls}")
    for i in range(centers.shape[0]):
        ax.scatter(centers[i,0], centers[i,1], marker='x', color='k', s=30)
        
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(title)
    ax.legend(ncol=2, fontsize='small', frameon=True)
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    plt.show()

def demo_2d_examples(M=1.0, Ks=(9,16), overlaps=(0.2,0.6), samples_per_gaussian=300, random_state=0):
    for K in Ks:
        for overlap in overlaps:
            centers = create_centers_grid(2, K, M, random_state=random_state)
            sigma = sigma_from_overlap(centers, overlap)
            print(f"Demo: K={K}, overlap={overlap} -> sigma={sigma:.4g}")
            plot_2d_kde_and_samples(centers, sigma, samples_per_gaussian=samples_per_gaussian, overlap=overlap)

def make_multitask_classification_datasets(
    data_dim,
    num_centers,
    dim_hypercube,
    num_class,
    num_points_per_task,
    num_tasks,
    centers_per_task=None,
    overlap=0.5,
    random_state=None,
    reuse=True
):
    """
    Parameters
    ----------
    data_dim : int
        Data point dimensionality.
    num_centers : int
        Total number of Gaussian centers available.
    dim_hypercube : float
        Range of the hypercube [0, M]^N for center placement.
    num_class : int
        Number of classes.
    num_points_per_task : int
        Total number of samples to generate per task. Hopefully roughly divisible by centers_per_task
    num_tasks : int
        Number of tasks to generate data for.
    centers_per_task : int
        Number of centers for each task, should be divisibe by num_class.
    overlap : float, default=0.5
        Determines Gaussian width (smaller means less overlap).
    random_state : int or None
        RNG seed for reproducibility.
    reuse : bool, default=True
        Whether to reuse the centers for different tasks or not.

    This function:

    - Places K gaussians in N-d hypercube [0,M]^N
    - Computes sigma via overlap

    Then creates num_tasks classification datasets by:
      
      - Assigning centers_per_task gaussians to C classes (as evenly as possible, considering the value of reuse).
      - Sample P total points equally across classes, split evenly among the gaussians.

    Returns List[dict] where each dict is the dataset for a task: 
        {'X': X, 'y': y, 'assignment': assignment, 'centers': centers, 'sigma': sigma}
        assignment: {gaussian_index: class_label}
    """

    rng = np.random.default_rng(random_state)
    centers_global = create_centers_grid(data_dim, num_centers, dim_hypercube, random_state=random_state)
    sigma = sigma_from_overlap(centers_global, overlap)

    if centers_per_task is None:
        centers_per_task = num_centers // num_tasks

    assert centers_per_task % num_class == 0, "centers_per_task must be divisible by num_class."
    if not reuse:
        assert centers_per_task * num_tasks <= num_centers, (
            f"Without reuse need centers_per_task * num_tasks <= num_centers. "
            f"Have {num_centers}, need {centers_per_task * num_tasks}."
        )

    datasets = []

    # If not reusing, prepare a global shuffled pool and take as many centers as needed
    if not reuse:
        pool = rng.permutation(num_centers)[: centers_per_task * num_tasks]
        # Reshape into (num_tasks, centers_per_task)
        pool = pool.reshape(num_tasks, centers_per_task)

    for t in range(num_tasks):
        # Select center indices for this task
        if reuse:
            # Choose centers_per_task distinct centers for this task (but they may be reused across tasks)
            selected_idx = rng.choice(num_centers, size=centers_per_task, replace=False)
        else:
            # Take from the pool, they are different between tasks
            selected_idx = pool[t].copy()

        # Shuffle the selected indices to mix their order
        rng.shuffle(selected_idx)

        # Get the coordinates for these selected global indices
        centers_task = centers_global[selected_idx].copy()

        # Assign centers evenly to classes
        assignment = {}
        per_class = centers_per_task // num_class  # Earlier assert guarantees integer
        for c in range(num_class):
            start = c * per_class
            end = (c + 1) * per_class
            for g in selected_idx[start:end]:
                assignment[int(g)] = int(c)

        # Generate samples dividing as evenly as possible
        samples_per_class = [
            num_points_per_task // num_class + (1 if i < (num_points_per_task % num_class) else 0)
            for i in range(num_class)
        ]

        X_parts = []
        y_parts = []

        for cls in range(num_class):
            # Global indices assigned to this class
            g_subset = [g for g, cl in assignment.items() if cl == cls]
            pts = samples_per_class[cls]
            # Split points across gaussians for this class
            pts_per_g = [
                pts // len(g_subset) + (1 if i < (pts % len(g_subset)) else 0)
                for i in range(len(g_subset))
            ]
            for g, n in zip(g_subset, pts_per_g):
                mu = centers_global[int(g)]
                samples, _ = sample_gaussians(mu.reshape(1,data_dim), sigma, n, rng=rng)
                X_parts.append(samples)
                y_parts.append(np.full(n, cls, dtype=int))

        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)

        perm = rng.permutation(len(X))
        datasets.append({
            'X': X[perm],
            'y': y[perm],
            'assignment': assignment,
            'centers': centers_task,   # Centers used by this task (in the shuffled order)
            'sigma': sigma
        })

    return datasets
