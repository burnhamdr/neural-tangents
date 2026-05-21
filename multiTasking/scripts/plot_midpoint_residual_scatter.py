import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from plot_midpoint_distance_structure import (
    build_pair_table,
    demean_rows_by_distance,
)


def make_plot(run_tables, response_col, save_path):
    if not run_tables:
        raise ValueError("No runs supplied.")

    processed_tables = []
    for table in run_tables:
        rows = demean_rows_by_distance(table["rows"])
        processed = dict(table)
        processed["rows"] = rows
        processed_tables.append(processed)

    n = len(processed_tables)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 4.8), sharex=True, sharey=True, squeeze=False)
    axes = axes[0]

    all_resid = np.asarray([row["value"] for table in processed_tables for row in table["rows"]], dtype=float)
    all_dist = np.asarray([row["distance_pi"] for table in processed_tables for row in table["rows"]], dtype=float)
    y_pad = 0.06 * max(float(np.ptp(all_resid)), 1e-6)
    unique_dist = np.unique(np.round(all_dist, 10))
    base_cmap = plt.get_cmap("viridis")
    color_values = base_cmap(np.linspace(0.06, 0.94, unique_dist.size))
    discrete_cmap = ListedColormap(color_values)

    if unique_dist.size == 1:
        boundaries = np.asarray([unique_dist[0] - 0.5, unique_dist[0] + 0.5], dtype=float)
    else:
        midpoints = 0.5 * (unique_dist[:-1] + unique_dist[1:])
        first = unique_dist[0] - (midpoints[0] - unique_dist[0])
        last = unique_dist[-1] + (unique_dist[-1] - midpoints[-1])
        boundaries = np.concatenate([[first], midpoints, [last]])
    norm = BoundaryNorm(boundaries, discrete_cmap.N, clip=True)
    color_by_distance = {
        float(dval): color_values[idx]
        for idx, dval in enumerate(unique_dist)
    }

    scatter_for_cbar = None
    for ax, table in zip(axes, processed_tables):
        rows = table["rows"]
        x = np.asarray([row["midpoint_pi"] for row in rows], dtype=float)
        y = np.asarray([row["value"] for row in rows], dtype=float)
        d = np.asarray([row["distance_pi"] for row in rows], dtype=float)

        for dval in np.unique(np.round(d, 10)):
            mask = np.isclose(d, dval, atol=1e-10)
            if np.sum(mask) < 2:
                continue
            order = np.argsort(x[mask])
            x_line = x[mask][order]
            y_line = y[mask][order]
            ax.plot(
                x_line,
                y_line,
                color=color_by_distance[float(dval)],
                lw=1.0,
                alpha=0.38,
                zorder=1,
            )

        scatter = ax.scatter(
            x,
            y,
            c=d,
            cmap=discrete_cmap,
            norm=norm,
            s=24,
            alpha=0.55,
            edgecolors="none",
            zorder=2,
        )
        scatter_for_cbar = scatter

        ax.axhline(0.0, color="0.25", lw=1.0, ls=":")
        ax.set_title(rf"$\alpha/\pi = {table['alpha_over_pi']:.2f}$")
        ax.set_xlabel(r"Pair midpoint $/ \pi$")
        ax.grid(True, alpha=0.18)

    axes[0].set_ylabel(response_col.replace("_", " ") + " residual")
    for ax in axes:
        ax.set_xlim(0.0, 2.0)
        ax.set_ylim(float(np.min(all_resid) - y_pad), float(np.max(all_resid) + y_pad))

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
        "Distance-demeaned update similarity residual vs pair midpoint",
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
            "Plot pair midpoint against the residual update similarity after subtracting "
            "the mean value at each task-distance column."
        )
    )
    parser.add_argument(
        "--results-dirs",
        nargs="+",
        required=True,
        help="One or more saved run directories.",
    )
    parser.add_argument(
        "--response-col",
        type=str,
        default="emp_update_full",
        help="Column from pairwise_metrics.csv to residualize and plot.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        required=True,
        help="Output PNG path.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    run_tables = [build_pair_table(results_dir, args.response_col) for results_dir in args.results_dirs]
    save_path = make_plot(run_tables, args.response_col, args.save_path)
    print(f"Saved plot: {save_path}")


if __name__ == "__main__":
    main()
