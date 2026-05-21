import argparse
import csv
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from plot_midpoint_distance_structure import build_pair_table


DEFAULT_RESPONSE_COLS = [
    "emp_update_bias",
    "emp_update_full",
]


def write_csv_rows(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def harmonic_design(midpoint_pi, q_max, *, include_intercept=True):
    midpoint_pi = np.asarray(midpoint_pi, dtype=float)
    cols = [np.ones_like(midpoint_pi)] if include_intercept else []
    for q in range(1, int(q_max) + 1):
        cols.append(np.cos(q * np.pi * midpoint_pi))
        cols.append(np.sin(q * np.pi * midpoint_pi))
    if not cols:
        raise ValueError("Harmonic design needs at least one column.")
    return np.column_stack(cols)


def fit_linear_predict(y, X):
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    return beta, y_hat


def unique_distance_levels(distance_pi, tol=1e-10):
    distance_pi = np.asarray(distance_pi, dtype=float)
    rounded = np.round(distance_pi / tol).astype(np.int64)
    unique_keys = np.unique(rounded)
    levels = []
    for key in np.sort(unique_keys):
        mask = rounded == key
        levels.append(
            {
                "key": int(key),
                "value": float(np.mean(distance_pi[mask])),
                "mask": mask,
            }
        )
    return levels


def rounded_distance_key(distance_pi, ndigits=10):
    return float(np.round(float(distance_pi), ndigits))


def stage1_distance_means(distance_pi, y):
    distance_pi = np.asarray(distance_pi, dtype=float)
    y = np.asarray(y, dtype=float)
    levels = unique_distance_levels(distance_pi)
    fit = np.zeros_like(y)
    rows = []
    for level in levels:
        mean_val = float(np.mean(y[level["mask"]]))
        fit[level["mask"]] = mean_val
        rows.append(
            {
                "distance_pi": level["value"],
                "count": int(np.sum(level["mask"])),
                "mean_similarity": mean_val,
            }
        )
    return fit, rows, levels


def stage2_midpoint_additive(midpoint_pi, residual, q_max):
    X = harmonic_design(midpoint_pi, q_max, include_intercept=True)
    beta, fit = fit_linear_predict(residual, X)
    return {
        "beta": beta,
        "fit": fit,
        "q_max": int(q_max),
    }


def stage3_distance_midpoint_interaction(distance_pi, midpoint_pi, residual, q_max):
    distance_pi = np.asarray(distance_pi, dtype=float)
    midpoint_pi = np.asarray(midpoint_pi, dtype=float)
    residual = np.asarray(residual, dtype=float)
    levels = unique_distance_levels(distance_pi)
    fit = np.zeros_like(residual)
    coeff_rows = []
    pred_rows = []

    for level in levels:
        mask = level["mask"]
        X = harmonic_design(midpoint_pi[mask], q_max, include_intercept=False)
        beta, fit_level = fit_linear_predict(residual[mask], X)
        fit[mask] = fit_level
        coeff_rows.append(
            {
                "distance_pi": level["value"],
                "n_pairs": int(np.sum(mask)),
                "q_max": int(q_max),
                "coefficients": " ".join(f"{coef:.10g}" for coef in beta),
            }
        )
        order = np.argsort(midpoint_pi[mask])
        pred_rows.append(
            {
                "distance_pi": level["value"],
                "midpoint_pi": midpoint_pi[mask][order],
                "fit": fit_level[order],
            }
        )

    return {
        "fit": fit,
        "coeff_rows": coeff_rows,
        "pred_rows": pred_rows,
        "q_max": int(q_max),
    }


def compute_r2(y, residual):
    y = np.asarray(y, dtype=float)
    residual = np.asarray(residual, dtype=float)
    tss = float(np.sum((y - np.mean(y)) ** 2))
    rss = float(np.sum(residual ** 2))
    return np.nan if tss <= 0.0 else 1.0 - rss / tss


def analyze_run(results_dir, response_col, *, q_midpoint=4, q_interaction=4):
    table = build_pair_table(results_dir, response_col)
    rows = table["rows"]
    distance_pi = np.asarray([row["distance_pi"] for row in rows], dtype=float)
    midpoint_pi = np.asarray([row["midpoint_pi"] for row in rows], dtype=float)
    y = np.asarray([row["value"] for row in rows], dtype=float)

    fit1, stage1_rows, levels = stage1_distance_means(distance_pi, y)
    resid1 = y - fit1
    stage2 = stage2_midpoint_additive(midpoint_pi, resid1, q_midpoint)
    fit2 = stage2["fit"]
    resid2 = resid1 - fit2
    stage3 = stage3_distance_midpoint_interaction(distance_pi, midpoint_pi, resid2, q_interaction)
    fit3 = stage3["fit"]
    resid3 = resid2 - fit3

    r2_1 = compute_r2(y, resid1)
    r2_2 = compute_r2(y, resid2)
    r2_3 = compute_r2(y, resid3)

    pair_rows = []
    for idx in range(y.size):
        pair_rows.append(
            {
                "run_name": table["run_name"],
                "results_dir": table["results_dir"],
                "alpha": table["alpha"],
                "alpha_over_pi": table["alpha_over_pi"],
                "response_col": response_col,
                "distance_pi": float(distance_pi[idx]),
                "midpoint_pi": float(midpoint_pi[idx]),
                "value": float(y[idx]),
                "fit_distance": float(fit1[idx]),
                "resid_after_distance": float(resid1[idx]),
                "fit_midpoint_additive": float(fit2[idx]),
                "resid_after_midpoint": float(resid2[idx]),
                "fit_interaction": float(fit3[idx]),
                "resid_final": float(resid3[idx]),
            }
        )

    summary_row = {
        "run_name": table["run_name"],
        "results_dir": table["results_dir"],
        "alpha": table["alpha"],
        "alpha_over_pi": table["alpha_over_pi"],
        "response_col": response_col,
        "n_pairs": int(y.size),
        "n_distance_levels": int(len(levels)),
        "q_midpoint": int(q_midpoint),
        "q_interaction": int(q_interaction),
        "r2_distance_only": float(r2_1),
        "r2_after_additive_midpoint": float(r2_2),
        "r2_after_interaction": float(r2_3),
        "delta_r2_additive_midpoint": float(r2_2 - r2_1),
        "delta_r2_interaction": float(r2_3 - r2_2),
        "final_resid_std": float(np.std(resid3)),
    }

    return {
        "table": table,
        "response_col": response_col,
        "distance_pi": distance_pi,
        "midpoint_pi": midpoint_pi,
        "value": y,
        "fit_distance": fit1,
        "resid_after_distance": resid1,
        "fit_midpoint_additive": fit2,
        "resid_after_midpoint": resid2,
        "fit_interaction": fit3,
        "resid_final": resid3,
        "stage1_rows": stage1_rows,
        "stage2_q_max": int(q_midpoint),
        "stage3_q_max": int(q_interaction),
        "stage3_pred_rows": stage3["pred_rows"],
        "summary_row": summary_row,
        "pair_rows": pair_rows,
        "interaction_coeff_rows": [
            {
                "run_name": table["run_name"],
                "results_dir": table["results_dir"],
                "alpha": table["alpha"],
                "alpha_over_pi": table["alpha_over_pi"],
                "response_col": response_col,
                **row,
            }
            for row in stage3["coeff_rows"]
        ],
    }


def make_discrete_distance_palette(all_distances):
    all_distances = np.asarray(all_distances, dtype=float)
    unique_dist = np.unique(np.round(all_distances, 10))
    base_cmap = plt.get_cmap("viridis")
    color_values = base_cmap(np.linspace(0.06, 0.94, unique_dist.size))
    discrete_cmap = ListedColormap(color_values)
    if unique_dist.size == 1:
        boundaries = np.asarray([unique_dist[0] - 0.5, unique_dist[0] + 0.5], dtype=float)
    else:
        mids = 0.5 * (unique_dist[:-1] + unique_dist[1:])
        first = unique_dist[0] - (mids[0] - unique_dist[0])
        last = unique_dist[-1] + (unique_dist[-1] - mids[-1])
        boundaries = np.concatenate([[first], mids, [last]])
    norm = BoundaryNorm(boundaries, discrete_cmap.N, clip=True)
    color_by_distance = {
        rounded_distance_key(dval): color_values[idx]
        for idx, dval in enumerate(unique_dist)
    }
    return unique_dist, discrete_cmap, norm, boundaries, color_by_distance


def make_distance_jittered_scatter(ax, x, y, *, color="#4c72b0"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_plot = x.copy()
    for xval in np.unique(np.round(x, 10)):
        mask = np.isclose(x, xval, atol=1e-10)
        count = int(np.sum(mask))
        if count > 1:
            x_plot[mask] += np.linspace(-0.008, 0.008, count)
    ax.scatter(x_plot, y, s=20, alpha=0.38, color=color, edgecolors="none")


def make_model_figure(run_results, response_col, save_path):
    if not run_results:
        raise ValueError("No run results supplied.")

    run_results = sorted(run_results, key=lambda item: item["summary_row"]["alpha"])
    all_distances = np.concatenate([result["distance_pi"] for result in run_results])
    unique_dist, discrete_cmap, norm, boundaries, color_by_distance = make_discrete_distance_palette(all_distances)

    n_rows = len(run_results)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(16.4, 4.4 * n_rows),
        sharex="col",
        squeeze=False,
    )

    scatter_for_cbar = None
    for row_idx, result in enumerate(run_results):
        alpha_over_pi = result["summary_row"]["alpha_over_pi"]
        distance_pi = result["distance_pi"]
        midpoint_pi = result["midpoint_pi"]
        y = result["value"]
        resid1 = result["resid_after_distance"]
        resid2 = result["resid_after_midpoint"]
        stage1_rows = result["stage1_rows"]
        summary = result["summary_row"]

        ax0, ax1, ax2 = axes[row_idx]

        make_distance_jittered_scatter(ax0, distance_pi, y)
        x_mean = np.asarray([row["distance_pi"] for row in stage1_rows], dtype=float)
        y_mean = np.asarray([row["mean_similarity"] for row in stage1_rows], dtype=float)
        ax0.plot(x_mean, y_mean, color="#c44e52", lw=2.2, marker="o", ms=4)
        ax0.set_title(
            rf"$\alpha/\pi = {alpha_over_pi:.2f}$"
            + "\n"
            + rf"Distance only $R^2 = {summary['r2_distance_only']:.3f}$"
        )
        ax0.set_ylabel(response_col.replace("_", " "))
        ax0.set_xlabel(r"Task distance $/ \pi$")
        ax0.grid(True, alpha=0.18)
        ax0.set_xticks(x_mean)
        ax0.tick_params(axis="x", rotation=45)

        u_grid = np.linspace(0.0, 2.0, 500)
        X_grid = harmonic_design(u_grid, result["stage2_q_max"], include_intercept=True)
        X_data = harmonic_design(midpoint_pi, result["stage2_q_max"], include_intercept=True)
        beta2, _ = fit_linear_predict(resid1, X_data)
        additive_grid = X_grid @ beta2

        for dval in unique_dist:
            mask = np.isclose(distance_pi, dval, atol=1e-10)
            if np.sum(mask) < 2:
                continue
            order = np.argsort(midpoint_pi[mask])
            ax1.plot(
                midpoint_pi[mask][order],
                resid1[mask][order],
                color=color_by_distance[rounded_distance_key(dval)],
                lw=0.9,
                alpha=0.22,
                zorder=1,
            )
        scatter1 = ax1.scatter(
            midpoint_pi,
            resid1,
            c=distance_pi,
            cmap=discrete_cmap,
            norm=norm,
            s=24,
            alpha=0.58,
            edgecolors="none",
            zorder=2,
        )
        scatter_for_cbar = scatter1
        ax1.plot(u_grid, additive_grid, color="0.1", lw=2.0, zorder=3)
        ax1.axhline(0.0, color="0.3", lw=1.0, ls=":")
        ax1.set_title(
            rf"Add midpoint $\Delta R^2 = {summary['delta_r2_additive_midpoint']:.3f}$"
            + "\n"
            + rf"Cumulative $R^2 = {summary['r2_after_additive_midpoint']:.3f}$"
        )
        ax1.set_ylabel("Residual after distance")
        ax1.set_xlabel(r"Pair midpoint $/ \pi$")
        ax1.set_xlim(0.0, 2.0)
        ax1.grid(True, alpha=0.18)

        for pred in result["stage3_pred_rows"]:
            dval = float(pred["distance_pi"])
            ax2.plot(
                pred["midpoint_pi"],
                pred["fit"],
                color=color_by_distance[rounded_distance_key(dval)],
                lw=1.7,
                alpha=0.86,
                zorder=3,
            )
        scatter2 = ax2.scatter(
            midpoint_pi,
            resid2,
            c=distance_pi,
            cmap=discrete_cmap,
            norm=norm,
            s=24,
            alpha=0.48,
            edgecolors="none",
            zorder=2,
        )
        scatter_for_cbar = scatter2
        ax2.axhline(0.0, color="0.3", lw=1.0, ls=":")
        ax2.set_title(
            rf"Add distance$\times$midpoint $\Delta R^2 = {summary['delta_r2_interaction']:.3f}$"
            + "\n"
            + rf"Cumulative $R^2 = {summary['r2_after_interaction']:.3f}$"
        )
        ax2.set_ylabel("Residual after midpoint")
        ax2.set_xlabel(r"Pair midpoint $/ \pi$")
        ax2.set_xlim(0.0, 2.0)
        ax2.grid(True, alpha=0.18)

    fig.tight_layout(rect=[0.0, 0.0, 0.92, 0.95])
    cax = fig.add_axes([0.935, 0.18, 0.016, 0.62])
    cbar = fig.colorbar(
        scatter_for_cbar,
        cax=cax,
        orientation="vertical",
        boundaries=boundaries,
        ticks=unique_dist,
        spacing="proportional",
    )
    cbar.set_label(r"Task distance $/ \pi$")
    cbar.ax.set_yticklabels([f"{dval:.2f}" for dval in unique_dist])
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Sequential pairwise similarity decomposition\n"
        "Antipodal pairs omitted because pair midpoint is undefined",
        fontsize=14,
        y=0.995,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return save_path


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Fit a sequential distance -> additive midpoint -> distance-by-midpoint interaction "
            "decomposition to pairwise similarity data, and visualize the stages."
        )
    )
    parser.add_argument(
        "--results-dirs",
        nargs="+",
        required=True,
        help="Saved run directories to compare, e.g. alpha/pi=0 and alpha/pi=0.5 extremes.",
    )
    parser.add_argument(
        "--response-cols",
        type=str,
        default=",".join(DEFAULT_RESPONSE_COLS),
        help="Comma-separated response columns from pairwise_metrics.csv.",
    )
    parser.add_argument(
        "--q-midpoint",
        type=int,
        default=4,
        help="Maximum harmonic order for the additive midpoint stage.",
    )
    parser.add_argument(
        "--q-interaction",
        type=int,
        default=4,
        help="Maximum harmonic order for the distance-specific midpoint interaction stage.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for figures and CSV outputs.",
    )
    return parser


def parse_csv_list(text):
    return [item.strip() for item in str(text).split(",") if item.strip()]


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    response_cols = parse_csv_list(args.response_cols)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    pair_rows = []
    coeff_rows = []

    for response_col in response_cols:
        run_results = [
            analyze_run(
                results_dir,
                response_col,
                q_midpoint=args.q_midpoint,
                q_interaction=args.q_interaction,
            )
            for results_dir in args.results_dirs
        ]
        figure_path = output_dir / f"sequential_decomposition_{response_col}.png"
        make_model_figure(run_results, response_col, figure_path)
        print(f"Saved plot: {figure_path}")

        for result in run_results:
            summary_rows.append(result["summary_row"])
            pair_rows.extend(result["pair_rows"])
            coeff_rows.extend(result["interaction_coeff_rows"])

    if summary_rows:
        write_csv_rows(
            output_dir / "sequential_decomposition_summary.csv",
            list(summary_rows[0].keys()),
            summary_rows,
        )
    if pair_rows:
        write_csv_rows(
            output_dir / "sequential_decomposition_pairwise.csv",
            list(pair_rows[0].keys()),
            pair_rows,
        )
    if coeff_rows:
        write_csv_rows(
            output_dir / "sequential_decomposition_interaction_coefficients.csv",
            list(coeff_rows[0].keys()),
            coeff_rows,
        )


if __name__ == "__main__":
    main()
