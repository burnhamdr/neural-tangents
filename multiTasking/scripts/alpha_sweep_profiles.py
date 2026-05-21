import argparse
import csv
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
except Exception:
    plt = None
    Normalize = None
    Line3DCollection = None

import taskManifold_update as tmu


DEFAULT_RESPONSE_COLS = [
    "emp_update_bias",
    "emp_update_full",
    "emp_update_full_minus_bias",
]


def read_csv_rows(path):
    path = Path(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_csv_list(text):
    return [item.strip() for item in str(text).split(",") if item.strip()]


def parse_angle_list(text):
    return [float(tmu.parse_angle_arg(item.strip())) for item in str(text).split(",") if item.strip()]


def default_alpha_grid(n=7):
    return np.linspace(0.0, 0.5 * np.pi, int(n), endpoint=True)


def fit_ols(y, columns):
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(y.size, dtype=float)] + [np.asarray(col, dtype=float) for col in columns])
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    rss = float(np.sum((y - y_hat) ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if tss <= 0.0 else 1.0 - rss / tss
    n = y.size
    p = X.shape[1]
    if n > p and np.isfinite(r2):
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p)
    else:
        adj_r2 = np.nan
    return beta, rss, r2, adj_r2, int(rank)


def grouped_distance_profile(distances, values, tol=1e-10):
    distances = np.asarray(distances, dtype=float)
    values = np.asarray(values, dtype=float)
    rounded = np.round(distances / tol).astype(np.int64)
    unique_keys = np.unique(rounded)
    rows = []
    for key in np.sort(unique_keys):
        mask = rounded == key
        d = float(np.mean(distances[mask]))
        vals = values[mask]
        rows.append(
            {
                "distance": d,
                "distance_pi": d / np.pi,
                "count": int(np.sum(mask)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "sem": float(np.std(vals, ddof=1) / np.sqrt(np.sum(mask))) if np.sum(mask) > 1 else 0.0,
            }
        )
    return rows


def summarize_profile_shape(distances, values):
    grouped = grouped_distance_profile(distances, values)
    x_group = np.asarray([row["distance_pi"] for row in grouped], dtype=float)
    y_group = np.asarray([row["mean"] for row in grouped], dtype=float)
    xc_group = x_group - 0.5

    beta_group, rss_group, r2_group, adj_r2_group, rank_group = fit_ols(y_group, [xc_group, xc_group ** 2])

    x_pairs = np.asarray(distances, dtype=float) / np.pi
    y_pairs = np.asarray(values, dtype=float)
    xc_pairs = x_pairs - 0.5
    beta_lin, rss_lin, r2_lin, adj_r2_lin, rank_lin = fit_ols(y_pairs, [xc_pairs])
    beta_quad, rss_quad, r2_quad, adj_r2_quad, rank_quad = fit_ols(y_pairs, [xc_pairs, xc_pairs ** 2])

    first_mean = float(y_group[0]) if y_group.size else np.nan
    last_mean = float(y_group[-1]) if y_group.size else np.nan
    mid_idx = int(np.argmin(np.abs(x_group - 0.5))) if y_group.size else 0
    mid_mean = float(y_group[mid_idx]) if y_group.size else np.nan
    u_depth = 0.5 * (first_mean + last_mean) - mid_mean if y_group.size else np.nan

    return {
        "n_distance_levels": int(y_group.size),
        "distance_min_mean": first_mean,
        "distance_mid_mean": mid_mean,
        "distance_max_mean": last_mean,
        "u_depth": float(u_depth) if np.isfinite(u_depth) else np.nan,
        "group_intercept": float(beta_group[0]),
        "group_linear_centered": float(beta_group[1]),
        "group_quadratic_centered": float(beta_group[2]),
        "group_r2": float(r2_group) if np.isfinite(r2_group) else np.nan,
        "group_adj_r2": float(adj_r2_group) if np.isfinite(adj_r2_group) else np.nan,
        "pair_linear_centered": float(beta_lin[1]),
        "pair_linear_adj_r2": float(adj_r2_lin) if np.isfinite(adj_r2_lin) else np.nan,
        "pair_quadratic_centered": float(beta_quad[2]),
        "pair_quadratic_adj_r2": float(adj_r2_quad) if np.isfinite(adj_r2_quad) else np.nan,
        "pair_quad_minus_lin_adj_r2": float(adj_r2_quad - adj_r2_lin) if np.isfinite(adj_r2_quad) and np.isfinite(adj_r2_lin) else np.nan,
        "group_profile": grouped,
    }


def analyze_run(results_dir, response_cols):
    results_dir = Path(results_dir)
    metadata = tmu.load_task_metadata(results_dir / "task_metadata.csv")
    rows = read_csv_rows(results_dir / "pairwise_metrics.csv")
    if not rows:
        raise ValueError(f"No pairwise rows found in {results_dir}")

    run_rows = []
    profile_rows = []
    pair_rows = []

    alpha = float(metadata["target_phase"])
    run_name = results_dir.name
    regime = "cosine_like" if np.isclose(np.mod(alpha, 2 * np.pi), 0.5 * np.pi) else "sine_like" if np.isclose(np.mod(alpha, 2 * np.pi), 0.0) else "mixed"

    for response_col in response_cols:
        if response_col not in rows[0]:
            continue

        distances = np.asarray([float(row["distance"]) for row in rows], dtype=float)
        values = np.asarray([float(row[response_col]) for row in rows], dtype=float)
        summary = summarize_profile_shape(distances, values)

        for distance, value in zip(distances, values):
            pair_rows.append(
                {
                    "run_name": run_name,
                    "results_dir": str(results_dir.resolve()),
                    "response_col": response_col,
                    "alpha": alpha,
                    "alpha_over_pi": alpha / np.pi,
                    "invariant_weight_sin_alpha": float(np.sin(alpha)),
                    "rotating_weight_cos_alpha": float(np.cos(alpha)),
                    "distance": float(distance),
                    "distance_pi": float(distance / np.pi),
                    "similarity": float(value),
                }
            )

        run_rows.append(
            {
                "run_name": run_name,
                "results_dir": str(results_dir.resolve()),
                "response_col": response_col,
                "alpha": alpha,
                "alpha_over_pi": alpha / np.pi,
                "invariant_weight_sin_alpha": float(np.sin(alpha)),
                "rotating_weight_cos_alpha": float(np.cos(alpha)),
                "regime_label": regime,
                "task_family": metadata["task_family"],
                "m": int(metadata["m"]),
                **{key: value for key, value in summary.items() if key != "group_profile"},
            }
        )

        for row in summary["group_profile"]:
            profile_rows.append(
                {
                    "run_name": run_name,
                    "results_dir": str(results_dir.resolve()),
                    "response_col": response_col,
                    "alpha": alpha,
                    "alpha_over_pi": alpha / np.pi,
                    "invariant_weight_sin_alpha": float(np.sin(alpha)),
                    "rotating_weight_cos_alpha": float(np.cos(alpha)),
                    "distance": row["distance"],
                    "distance_pi": row["distance_pi"],
                    "count": row["count"],
                    "mean_similarity": row["mean"],
                    "std_similarity": row["std"],
                    "sem_similarity": row["sem"],
                }
            )

    return run_rows, profile_rows, pair_rows


def wrapped_angle_distance(a, b):
    return abs(np.angle(np.exp(1j * (float(a) - float(b)))))


def nearest_angle_index(angles, target_angle):
    angles = np.asarray(angles, dtype=float)
    dists = np.asarray([wrapped_angle_distance(angle, target_angle) for angle in angles], dtype=float)
    return int(np.argmin(dists))


def camera_view_vector(elev_deg, azim_deg):
    elev = np.deg2rad(float(elev_deg))
    azim = np.deg2rad(float(azim_deg))
    return np.asarray(
        [
            np.cos(elev) * np.cos(azim),
            np.cos(elev) * np.sin(azim),
            np.sin(elev),
        ],
        dtype=float,
    )


def add_front_back_split_circle(
    ax,
    X,
    y,
    *,
    scale=1.0,
    elev=20.0,
    azim=36.0,
    cmap_name="coolwarm",
    vmin=-1.0,
    vmax=1.0,
    lw_front=2.7,
    lw_back=0.95,
    alpha_front=0.98,
    alpha_back=0.22,
):
    if Line3DCollection is None or Normalize is None:
        return None

    X = np.asarray(X, dtype=float) * float(scale)
    y = np.asarray(y, dtype=float)
    X_closed = np.vstack([X, X[0]])
    y_closed = np.concatenate([y, y[:1]])
    segments = np.stack([X_closed[:-1], X_closed[1:]], axis=1)
    seg_midpoints = 0.5 * (segments[:, 0, :] + segments[:, 1, :])
    seg_values = 0.5 * (y_closed[:-1] + y_closed[1:])

    view_vec = camera_view_vector(elev, azim)
    is_front = (seg_midpoints @ view_vec) >= 0.0
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(seg_values))

    if np.any(~is_front):
        rgba_back = rgba[~is_front].copy()
        rgba_back[:, 3] = alpha_back
        back_collection = Line3DCollection(
            segments[~is_front],
            colors=rgba_back,
            linewidths=lw_back,
        )
        ax.add_collection3d(back_collection)

    if np.any(is_front):
        rgba_front = rgba[is_front].copy()
        rgba_front[:, 3] = alpha_front
        front_collection = Line3DCollection(
            segments[is_front],
            colors=rgba_front,
            linewidths=lw_front,
        )
        ax.add_collection3d(front_collection)

    return cmap, norm


def make_alpha_extreme_same_support_comparison(run_rows, output_dir):
    if plt is None or Line3DCollection is None or Normalize is None or not run_rows:
        return []

    run_rows = sorted(run_rows, key=lambda row: row["alpha"])
    metadatas = []
    for row in run_rows:
        metadata_path = Path(row["results_dir"]) / "task_metadata.csv"
        if metadata_path.exists():
            metadatas.append((row, tmu.load_task_metadata(metadata_path)))

    if not metadatas:
        return []

    first_meta = metadatas[0][1]
    task_family = first_meta["task_family"] or "single_axis"
    if task_family != "single_axis":
        return []

    alpha_targets = [0.0, 0.5 * np.pi]
    chosen = []
    used_rows = set()
    for target in alpha_targets:
        best_idx = min(
            range(len(metadatas)),
            key=lambda idx: abs(float(metadatas[idx][0]["alpha"]) - target),
        )
        row_id = metadatas[best_idx][0]["results_dir"]
        if row_id in used_rows:
            continue
        chosen.append(metadatas[best_idx])
        used_rows.add(row_id)

    if len(chosen) < 2:
        return []

    color_values = np.asarray(first_meta["color_values"], dtype=float)
    idx_theta0 = nearest_angle_index(color_values, 0.0)
    idx_thetapi = nearest_angle_index(color_values, np.pi)

    fig = plt.figure(figsize=(12.2, 5.4))
    axes = []
    colorbar_mappable = None
    elev = 20.0
    azim = 36.0

    for panel_idx, (row, metadata) in enumerate(chosen):
        ax = fig.add_subplot(1, 2, panel_idx + 1, projection="3d")
        axes.append(ax)
        tmu._draw_unit_sphere_wireframe(ax, color="0.86", alpha=0.20)
        ax.plot([-1.18, 1.18], [0.0, 0.0], [0.0, 0.0], ls="--", lw=1.1, color="0.15", alpha=0.72)
        ax.scatter([-1.0, 1.0], [0.0, 0.0], [0.0, 0.0], color="0.15", s=18, depthshade=False, alpha=0.9)

        normals = np.asarray(metadata["normals"], dtype=float)
        phases = np.asarray(metadata["phases"], dtype=float)
        target_phase = float(metadata["target_phase"])
        target_mode = metadata["target_mode"]
        global_field_name = metadata["global_field_name"]
        m = int(metadata["m"])

        support_specs = [
            (idx_theta0, 1.028, r"$\theta = 0$ (outer)"),
            (idx_thetapi, 0.972, r"$\theta = \pi$ (inner)"),
        ]
        for task_idx, scale, _ in support_specs:
            X, y, _ = tmu.generate_circle_samples(
                normals[task_idx],
                360,
                m,
                phase=float(phases[task_idx]),
                target_phase=target_phase,
                target_mode=target_mode,
                global_field_name=global_field_name,
                mode="grid",
            )
            colorbar_mappable = add_front_back_split_circle(
                ax,
                X,
                y,
                scale=scale,
                elev=elev,
                azim=azim,
            )

        ax.text2D(0.05, 0.93, r"$\theta = 0$ outer ring", transform=ax.transAxes, fontsize=9, color="0.15")
        ax.text2D(0.05, 0.87, r"$\theta = \pi$ inner ring", transform=ax.transAxes, fontsize=9, color="0.15")
        relation = "anti-match" if np.isclose(np.mod(target_phase, 2 * np.pi), 0.0) else "match" if np.isclose(np.mod(target_phase, 2 * np.pi), 0.5 * np.pi) else "mixed"
        ax.set_title(rf"$\alpha / \pi = {row['alpha_over_pi']:.2f}$" + f"\n{relation} on the same support")
        ax.view_init(elev=elev, azim=azim)
        tmu._set_equal_sphere_axes(ax)

    if colorbar_mappable is not None:
        cmap, norm = colorbar_mappable
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.tight_layout(rect=[0.0, 0.0, 0.92, 0.95])
        cax = fig.add_axes([0.935, 0.20, 0.016, 0.60])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Target value")
    else:
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

    fig.suptitle(
        "Same geometric support, opposite parameterization: "
        r"$\theta = 0$ vs $\theta = \pi$",
        fontsize=14,
        y=0.98,
    )
    save_path = Path(output_dir) / "alpha_extreme_same_support_comparison.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [save_path]


def make_alpha_task_sphere_panels(run_rows, output_dir):
    if plt is None or not run_rows:
        return []

    run_rows = sorted(run_rows, key=lambda row: row["alpha"])
    metadatas = []
    for row in run_rows:
        results_dir = Path(row["results_dir"])
        metadata_path = results_dir / "task_metadata.csv"
        if not metadata_path.exists():
            continue
        metadatas.append((row, tmu.load_task_metadata(metadata_path)))

    if not metadatas:
        return []

    first_meta = metadatas[0][1]
    task_family = first_meta["task_family"] or "single_axis"
    color_values = np.asarray(first_meta["color_values"], dtype=float)
    highlight_indices = tmu.select_task_geometry_highlights(task_family, color_values)
    highlight_set = set(int(idx) for idx in highlight_indices)

    n_alpha = len(metadatas)
    ncols = min(3, n_alpha)
    nrows = int(np.ceil(n_alpha / ncols))
    fig = plt.figure(figsize=(5.3 * ncols, 4.7 * nrows))
    axes = []
    target_scatter = None

    for panel_idx, (row, metadata) in enumerate(metadatas):
        ax = fig.add_subplot(nrows, ncols, panel_idx + 1, projection="3d")
        axes.append(ax)
        tmu._draw_unit_sphere_wireframe(ax)

        if task_family == "single_axis":
            ax.plot(
                [-1.18, 1.18],
                [0.0, 0.0],
                [0.0, 0.0],
                ls="--",
                lw=1.1,
                color="0.15",
                alpha=0.72,
            )

        normals = np.asarray(metadata["normals"], dtype=float)
        phases = np.asarray(metadata["phases"], dtype=float)
        target_phase = float(metadata["target_phase"])
        target_mode = metadata["target_mode"]
        global_field_name = metadata["global_field_name"]
        m = int(metadata["m"])

        for task_idx, (normal, phase) in enumerate(zip(normals, phases)):
            X, y, _ = tmu.generate_circle_samples(
                normal,
                240,
                m,
                phase=float(phase),
                target_phase=target_phase,
                target_mode=target_mode,
                global_field_name=global_field_name,
                mode="grid",
            )
            if task_idx in highlight_set:
                target_scatter = ax.scatter(
                    X[:, 0],
                    X[:, 1],
                    X[:, 2],
                    c=y,
                    cmap="coolwarm",
                    vmin=-1.0,
                    vmax=1.0,
                    s=13,
                    alpha=0.98,
                    edgecolors="none",
                    depthshade=False,
                )
            else:
                ax.plot(X[:, 0], X[:, 1], X[:, 2], color="0.62", lw=0.8, alpha=0.18)

        ax.set_title(rf"$\alpha / \pi = {row['alpha_over_pi']:.2f}$")
        ax.view_init(elev=20, azim=36)
        tmu._set_equal_sphere_axes(ax)

    for panel_idx in range(n_alpha, nrows * ncols):
        ax = fig.add_subplot(nrows, ncols, panel_idx + 1, projection="3d")
        ax.set_axis_off()

    fig.suptitle(
        "Task family on the sphere across target phase\n"
        "Same supports across panels; highlighted circles are colored by target value",
        fontsize=14,
        y=0.98,
    )
    if target_scatter is not None:
        fig.tight_layout(rect=[0.0, 0.0, 0.92, 0.95])
        cax = fig.add_axes([0.935, 0.22, 0.016, 0.56])
        cbar = fig.colorbar(target_scatter, cax=cax)
        cbar.set_label("Target value on highlighted circles")
    else:
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    save_path = Path(output_dir) / "alpha_task_family_sphere_panels.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [save_path]


def maybe_run_experiment(alpha, args):
    results_dir = tmu.ensure_results_dir(
        args.results_dir,
        task_family="single_axis",
        seed=args.seed,
        K_tasks=args.K_tasks,
        m=args.m,
        pts=args.pts,
        dim=args.dim,
        target_phase=float(alpha),
        target_mode="local_harmonic",
        global_field_name="xz",
    )
    if (
        not args.rerun_existing
        and (results_dir / "pairwise_metrics.csv").exists()
        and (results_dir / "task_metadata.csv").exists()
    ):
        return results_dir

    try:
        action = "Recomputing" if args.rerun_existing else "Running"
        print(f"{action} alpha={float(alpha):.6f} into {results_dir}")
        tmu.run_full_ntk_analysis_update(
            dim=args.dim,
            K_tasks=args.K_tasks,
            m=args.m,
            bias_std=args.bias_std,
            pts=args.pts,
            test_pts=args.test_pts,
            task_plot_id=args.task_plot_id,
            seed=args.seed,
            task_family="single_axis",
            normals_mode="fibonacci",
            random_phase=False,
            target_phase=float(alpha),
            target_mode="local_harmonic",
            global_field_name="xz",
            reg=args.reg,
            ridge_mode=args.ridge_mode,
            ridge_tuning_path=args.ridge_tuning_path,
            ridge_single_task_T=args.ridge_single_task_T,
            ridge_pairwise_T=args.ridge_pairwise_T,
            analyze_pairwise_joint=args.analyze_pairwise_joint,
            results_dir=results_dir,
            figures_dir=args.figures_dir,
            make_plots=args.make_plots,
        )
    except ImportError as exc:
        raise SystemExit(
            "Running missing alpha experiments requires the numerical stack used by "
            "taskManifold_update.py, including neural_tangents. "
            f"Original error: {exc}"
        ) from exc
    return results_dir


def make_midpoint_extreme_comparisons(run_rows, output_dir, response_cols):
    if plt is None or not run_rows:
        return []

    try:
        from plot_midpoint_distance_structure import build_pair_table, make_plot
        from plot_midpoint_residual_scatter import make_plot as make_residual_scatter_plot
        from sequential_pairwise_decomposition import (
            analyze_run as analyze_seq_run,
            make_model_figure as make_seq_figure,
            write_csv_rows as write_seq_csv_rows,
        )
    except Exception:
        return []

    run_rows = sorted(run_rows, key=lambda row: row["alpha"])
    alpha_targets = [0.0, 0.5 * np.pi]
    chosen = []
    used_dirs = set()
    for target in alpha_targets:
        best_row = min(run_rows, key=lambda row: abs(float(row["alpha"]) - target))
        if best_row["results_dir"] in used_dirs:
            continue
        results_dir = Path(best_row["results_dir"])
        if not (results_dir / "pairwise_metrics.csv").exists():
            continue
        chosen.append(best_row)
        used_dirs.add(best_row["results_dir"])

    if len(chosen) < 2:
        return []

    created = []
    seq_output_dir = Path(output_dir) / "sequential_decomposition_extremes"
    seq_output_dir.mkdir(parents=True, exist_ok=True)
    seq_summary_rows = []
    seq_pair_rows = []
    seq_coeff_rows = []
    seq_response_cols = []
    for response_col in list(response_cols) + ["emp_rkhs_full", "inf_rkhs_full"]:
        if response_col not in seq_response_cols:
            seq_response_cols.append(response_col)
    for response_col in response_cols:
        run_tables = [build_pair_table(row["results_dir"], response_col) for row in chosen]
        save_path = (
            Path(output_dir)
            / f"midpoint_structure_{response_col}_alpha0_vs_alpha05.png"
        )
        make_plot(run_tables, response_col, save_path)
        created.append(save_path)

        residual_save_path = (
            Path(output_dir)
            / f"midpoint_structure_demeaned_{response_col}_alpha0_vs_alpha05.png"
        )
        make_plot(
            run_tables,
            response_col,
            residual_save_path,
            demean_by_distance=True,
        )
        created.append(residual_save_path)

        residual_scatter_save_path = (
            Path(output_dir)
            / f"midpoint_residual_scatter_{response_col}_alpha0_vs_alpha05.png"
        )
        make_residual_scatter_plot(
            run_tables,
            response_col,
            residual_scatter_save_path,
        )
        created.append(residual_scatter_save_path)

    for response_col in seq_response_cols:
        try:
            seq_results = [
                analyze_seq_run(row["results_dir"], response_col)
                for row in chosen
            ]
        except Exception:
            continue
        seq_figure_path = seq_output_dir / f"sequential_decomposition_{response_col}.png"
        make_seq_figure(seq_results, response_col, seq_figure_path)
        created.append(seq_figure_path)
        for result in seq_results:
            seq_summary_rows.append(result["summary_row"])
            seq_pair_rows.extend(result["pair_rows"])
            seq_coeff_rows.extend(result["interaction_coeff_rows"])

    if seq_summary_rows:
        write_seq_csv_rows(
            seq_output_dir / "sequential_decomposition_summary.csv",
            list(seq_summary_rows[0].keys()),
            seq_summary_rows,
        )
    if seq_pair_rows:
        write_seq_csv_rows(
            seq_output_dir / "sequential_decomposition_pairwise.csv",
            list(seq_pair_rows[0].keys()),
            seq_pair_rows,
        )
    if seq_coeff_rows:
        write_seq_csv_rows(
            seq_output_dir / "sequential_decomposition_interaction_coefficients.csv",
            list(seq_coeff_rows[0].keys()),
            seq_coeff_rows,
        )

    return created


def make_summary_plots(summary_rows, profile_rows, pair_rows, output_dir, run_rows=None):
    if plt is None or not summary_rows:
        return []

    output_dir = Path(output_dir)
    created = []
    if run_rows:
        created.extend(make_alpha_extreme_same_support_comparison(run_rows, output_dir))
        created.extend(make_alpha_task_sphere_panels(run_rows, output_dir))
    response_cols = sorted({row["response_col"] for row in summary_rows})

    for response_col in response_cols:
        resp_summary = [row for row in summary_rows if row["response_col"] == response_col]
        resp_summary = sorted(resp_summary, key=lambda row: row["alpha"])
        alphas = np.asarray([row["alpha_over_pi"] for row in resp_summary], dtype=float)
        linear = np.asarray([row["group_linear_centered"] for row in resp_summary], dtype=float)
        quad = np.asarray([row["group_quadratic_centered"] for row in resp_summary], dtype=float)
        depth = np.asarray([row["u_depth"] for row in resp_summary], dtype=float)

        fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
        axes[0].plot(alphas, linear, marker="o", lw=2.0, color="#c44e52", label="Linear term")
        axes[0].plot(alphas, quad, marker="o", lw=2.0, color="#4c72b0", label="Quadratic term")
        axes[0].axhline(0.0, color="0.2", lw=1.0, ls=":")
        axes[0].set_title(f"Centered profile-shape coefficients\n{response_col}")
        axes[0].set_xlabel(r"Target phase $\alpha / \pi$")
        axes[0].set_ylabel("Coefficient")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(frameon=True, fontsize=9)

        axes[1].plot(alphas, depth, marker="o", lw=2.0, color="#55a868")
        axes[1].axhline(0.0, color="0.2", lw=1.0, ls=":")
        axes[1].set_title(f"U-depth across alpha\n{response_col}")
        axes[1].set_xlabel(r"Target phase $\alpha / \pi$")
        axes[1].set_ylabel("Endpoint avg - midpoint")
        axes[1].grid(True, alpha=0.25)

        fig.tight_layout()
        save_path = output_dir / f"alpha_shape_summary_{response_col}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        created.append(save_path)

        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        resp_profiles = [row for row in profile_rows if row["response_col"] == response_col]
        alpha_vals = sorted({row["alpha"] for row in resp_profiles})
        cmap = plt.get_cmap("viridis")
        for idx, alpha in enumerate(alpha_vals):
            rows = [row for row in resp_profiles if np.isclose(row["alpha"], alpha)]
            rows = sorted(rows, key=lambda row: row["distance"])
            xs = np.asarray([row["distance_pi"] for row in rows], dtype=float)
            ys = np.asarray([row["mean_similarity"] for row in rows], dtype=float)
            color = cmap(idx / max(len(alpha_vals) - 1, 1))
            ax.plot(xs, ys, marker="o", ms=4, lw=1.7, color=color, label=rf"$\alpha/\pi={alpha / np.pi:.2f}$")
        ax.set_title(f"Mean similarity profile by alpha\n{response_col}")
        ax.set_xlabel(r"Task distance $/ \pi$")
        ax.set_ylabel("Mean pairwise similarity")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=True, fontsize=8, ncol=2)
        fig.tight_layout()
        save_path = output_dir / f"alpha_profile_family_{response_col}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        created.append(save_path)

        alpha_vals = sorted({row["alpha"] for row in resp_profiles})
        raw_rows = [row for row in pair_rows if row["response_col"] == response_col]
        if alpha_vals and raw_rows:
            n_alpha = len(alpha_vals)
            ncols = min(3, n_alpha)
            nrows = int(np.ceil(n_alpha / ncols))
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(4.6 * ncols, 3.6 * nrows),
                sharex=True,
                sharey=True,
                squeeze=False,
            )
            y_all = np.asarray([row["similarity"] for row in raw_rows], dtype=float)
            y_pad = 0.04 * max(np.ptp(y_all), 1e-6)

            for idx, alpha in enumerate(alpha_vals):
                ax = axes[idx // ncols][idx % ncols]
                rows_alpha = [row for row in raw_rows if np.isclose(row["alpha"], alpha)]
                rows_alpha = sorted(rows_alpha, key=lambda row: (row["distance"], row["similarity"]))
                prof_alpha = [row for row in resp_profiles if np.isclose(row["alpha"], alpha)]
                prof_alpha = sorted(prof_alpha, key=lambda row: row["distance"])

                xs = np.asarray([row["distance_pi"] for row in rows_alpha], dtype=float)
                ys = np.asarray([row["similarity"] for row in rows_alpha], dtype=float)
                if xs.size:
                    # Small deterministic jitter within each discrete distance group.
                    x_plot = xs.copy()
                    for xval in np.unique(np.round(xs, 10)):
                        mask = np.isclose(xs, xval, atol=1e-10)
                        count = int(np.sum(mask))
                        if count > 1:
                            x_plot[mask] += np.linspace(-0.008, 0.008, count)
                    ax.scatter(x_plot, ys, s=18, alpha=0.42, color="#4c72b0", edgecolors="none")

                if prof_alpha:
                    x_prof = np.asarray([row["distance_pi"] for row in prof_alpha], dtype=float)
                    y_prof = np.asarray([row["mean_similarity"] for row in prof_alpha], dtype=float)
                    y_sem = np.asarray([row["sem_similarity"] for row in prof_alpha], dtype=float)
                    ax.plot(x_prof, y_prof, color="#c44e52", lw=2.0)
                    ax.fill_between(x_prof, y_prof - y_sem, y_prof + y_sem, color="#c44e52", alpha=0.18)
                    ax.set_xticks(x_prof)

                ax.set_title(rf"$\alpha/\pi = {alpha / np.pi:.2f}$")
                ax.grid(True, alpha=0.2)
                ax.tick_params(axis="x", rotation=45)

            for idx in range(n_alpha, nrows * ncols):
                axes[idx // ncols][idx % ncols].set_axis_off()

            for ax in axes[-1]:
                if ax.axison:
                    ax.set_xlabel(r"Task distance $/ \pi$")
            for row_axes in axes:
                if row_axes[0].axison:
                    row_axes[0].set_ylabel("Update similarity")

            for row_axes in axes:
                for ax in row_axes:
                    if ax.axison:
                        ax.set_ylim(np.min(y_all) - y_pad, np.max(y_all) + y_pad)

            fig.suptitle(f"Pairwise update similarity vs task distance by alpha\n{response_col}", fontsize=14, y=0.995)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            save_path = output_dir / f"alpha_pairwise_panels_{response_col}.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            created.append(save_path)

    if run_rows:
        created.extend(
            make_midpoint_extreme_comparisons(
                run_rows,
                output_dir,
                response_cols,
            )
        )

    return created


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the single-axis local-harmonic target phase alpha from sine-like to cosine-like, "
            "and summarize how the pairwise update-similarity profile shape changes."
        )
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default=None,
        help="Comma-separated alpha values, e.g. '0,pi/6,pi/4,pi/3,pi/2'. Defaults to a 7-point grid from 0 to pi/2.",
    )
    parser.add_argument(
        "--run-missing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run missing alpha experiments before analyzing them.",
    )
    parser.add_argument(
        "--rerun-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Recompute alpha experiments even when cached CSVs already exist. "
            "Writes into the same per-alpha results directories and overwrites the saved outputs."
        ),
    )
    parser.add_argument(
        "--response-cols",
        type=str,
        default=",".join(DEFAULT_RESPONSE_COLS),
        help="Comma-separated response columns from pairwise_metrics.csv to summarize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("neural-tangents/multiTasking/scripts/results/alpha_sweep_summary"),
        help="Directory for aggregated CSVs and optional plots.",
    )
    parser.add_argument("--dim", type=int, default=16384)
    parser.add_argument("--K-tasks", type=int, default=30)
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--bias-std", type=float, default=0.0)
    parser.add_argument("--pts", type=int, default=300)
    parser.add_argument("--test-pts", type=int, default=None)
    parser.add_argument("--task-plot-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reg", type=float, default=1e-5)
    parser.add_argument("--ridge-mode", choices=["trace", "max_eig"], default="trace")
    parser.add_argument("--ridge-tuning-path", type=Path, default=None)
    parser.add_argument("--ridge-single-task-T", type=int, default=1)
    parser.add_argument("--ridge-pairwise-T", type=int, default=2)
    parser.add_argument(
        "--analyze-pairwise-joint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run joint-interference / excess-MSE diagnostics when launching missing experiments.",
    )
    parser.add_argument(
        "--make-plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate matplotlib figures while running missing experiments and for the sweep summary if available.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Optional override passed into taskManifold_update for a single run directory. Normally leave unset.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Optional override passed into taskManifold_update when running missing experiments.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    alphas = parse_angle_list(args.alphas) if args.alphas is not None else list(default_alpha_grid())
    response_cols = parse_csv_list(args.response_cols)

    if args.results_dir is not None and len(alphas) > 1:
        raise SystemExit(
            "--results-dir points to a single run directory, so it can only be used with one alpha at a time. "
            "Leave it unset for a multi-alpha sweep."
        )
    if args.figures_dir is not None and (args.run_missing or args.rerun_existing) and len(alphas) > 1:
        raise SystemExit(
            "--figures-dir points to a single figure directory, so it can only be used with one alpha at a time "
            "when --run-missing or --rerun-existing is enabled. Leave it unset for a multi-alpha sweep."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    profile_rows = []
    pair_rows = []
    run_rows = []

    for alpha in alphas:
        results_dir = tmu.ensure_results_dir(
            args.results_dir,
            task_family="single_axis",
            seed=args.seed,
            K_tasks=args.K_tasks,
            m=args.m,
            pts=args.pts,
            dim=args.dim,
            target_phase=float(alpha),
            target_mode="local_harmonic",
            global_field_name="xz",
        )
        if args.run_missing or args.rerun_existing:
            results_dir = maybe_run_experiment(alpha, args)

        if not (results_dir / "pairwise_metrics.csv").exists():
            print(f"Skipping alpha={alpha:.6f}; results not found at {results_dir}")
            continue

        run_rows.append(
            {
                "alpha": float(alpha),
                "alpha_over_pi": float(alpha / np.pi),
                "run_name": results_dir.name,
                "results_dir": str(results_dir.resolve()),
            }
        )

        run_summary, run_profiles, run_pairs = analyze_run(results_dir, response_cols)
        summary_rows.extend(run_summary)
        profile_rows.extend(run_profiles)
        pair_rows.extend(run_pairs)

    if not summary_rows:
        raise SystemExit("No alpha runs were available to summarize.")

    write_csv_rows(output_dir / "alpha_runs.csv", list(run_rows[0].keys()), run_rows)
    write_csv_rows(output_dir / "alpha_profile_shape_summary.csv", list(summary_rows[0].keys()), summary_rows)
    write_csv_rows(output_dir / "alpha_distance_profiles.csv", list(profile_rows[0].keys()), profile_rows)
    write_csv_rows(output_dir / "alpha_pairwise_points.csv", list(pair_rows[0].keys()), pair_rows)

    for row in sorted(summary_rows, key=lambda item: (item["response_col"], item["alpha"])):
        print(
            f"alpha/pi={row['alpha_over_pi']:.3f} | response={row['response_col']} | "
            f"linear={row['group_linear_centered']:.4f} | quad={row['group_quadratic_centered']:.4f} | "
            f"u_depth={row['u_depth']:.4f}"
        )

    plot_paths = make_summary_plots(summary_rows, profile_rows, pair_rows, output_dir, run_rows=run_rows)
    if plot_paths:
        for path in plot_paths:
            print(f"Saved plot: {path}")
    elif plt is None:
        print("Summary plots skipped because matplotlib is not installed in this environment.")


if __name__ == "__main__":
    main()
