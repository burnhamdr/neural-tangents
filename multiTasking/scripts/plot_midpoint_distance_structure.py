import argparse
import csv
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv_rows(path):
    path = Path(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_task_angles(results_dir):
    rows = read_csv_rows(Path(results_dir) / "task_metadata.csv")
    rows = sorted(rows, key=lambda row: int(row["task_index"]))
    angles = np.asarray([float(row["color_value"]) for row in rows], dtype=float)
    alpha = float(rows[0].get("target_phase", 0.0) or 0.0)
    return angles, alpha


def pair_midpoint(theta_i, theta_j, *, eps=1e-9):
    delta = np.angle(np.exp(1j * (float(theta_j) - float(theta_i))))
    dist = abs(delta)
    if abs(dist - np.pi) < eps:
        return np.nan, dist
    midpoint = (float(theta_i) + 0.5 * delta) % (2.0 * np.pi)
    return midpoint, dist


def build_pair_table(results_dir, response_col, *, drop_antipodal=True):
    results_dir = Path(results_dir)
    angles, alpha = load_task_angles(results_dir)
    rows = read_csv_rows(results_dir / "pairwise_metrics.csv")

    pair_rows = []
    for row in rows:
        i = int(row["task_i"])
        j = int(row["task_j"])
        midpoint, dist = pair_midpoint(angles[i], angles[j])
        if drop_antipodal and not np.isfinite(midpoint):
            continue
        pair_rows.append(
            {
                "distance_pi": float(dist / np.pi),
                "midpoint_pi": float(midpoint / np.pi) if np.isfinite(midpoint) else np.nan,
                "value": float(row[response_col]),
            }
        )

    return {
        "results_dir": str(results_dir.resolve()),
        "run_name": results_dir.name,
        "alpha": float(alpha),
        "alpha_over_pi": float(alpha / np.pi),
        "rows": pair_rows,
    }


def demean_rows_by_distance(rows, *, distance_key="distance_pi", value_key="value", tol=1e-10):
    if not rows:
        return []

    distances = np.asarray([row[distance_key] for row in rows], dtype=float)
    values = np.asarray([row[value_key] for row in rows], dtype=float)
    rounded = np.round(distances / tol).astype(np.int64)

    mean_by_key = {}
    for key in np.unique(rounded):
        mask = rounded == key
        mean_by_key[int(key)] = float(np.mean(values[mask]))

    demeaned_rows = []
    for row, key in zip(rows, rounded):
        row_new = dict(row)
        row_new["value"] = float(row[value_key] - mean_by_key[int(key)])
        demeaned_rows.append(row_new)
    return demeaned_rows


def harmonic_midpoint_r2(x_mid_pi, y):
    x_mid_pi = np.asarray(x_mid_pi, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack(
        [
            np.ones_like(x_mid_pi),
            np.cos(np.pi * x_mid_pi),
            np.sin(np.pi * x_mid_pi),
            np.cos(2.0 * np.pi * x_mid_pi),
            np.sin(2.0 * np.pi * x_mid_pi),
        ]
    )
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss = float(np.sum((y - np.mean(y)) ** 2))
    rss = float(np.sum((y - y_hat) ** 2))
    return np.nan if ss <= 0 else 1.0 - rss / ss


def make_plot(run_tables, response_col, save_path, *, demean_by_distance=False):
    if not run_tables:
        raise ValueError("No runs supplied.")

    processed_tables = []
    for table in run_tables:
        rows = table["rows"]
        if demean_by_distance:
            rows = demean_rows_by_distance(rows)
        processed_table = dict(table)
        processed_table["rows"] = rows
        processed_tables.append(processed_table)

    all_vals = np.asarray([row["value"] for table in processed_tables for row in table["rows"]], dtype=float)
    vmax = float(np.max(np.abs(all_vals))) if all_vals.size else 1.0
    vmax = max(vmax, 1e-6)

    n = len(processed_tables)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 4.8), sharex=True, sharey=True, squeeze=False)
    axes = axes[0]

    scatter_for_cbar = None
    for ax, table in zip(axes, processed_tables):
        rows = table["rows"]
        x = np.asarray([row["distance_pi"] for row in rows], dtype=float)
        y = np.asarray([row["midpoint_pi"] for row in rows], dtype=float)
        z = np.asarray([row["value"] for row in rows], dtype=float)

        scatter = ax.scatter(
            x,
            y,
            c=z,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            s=36,
            alpha=0.85,
            edgecolors="none",
        )
        scatter_for_cbar = scatter

        # Show the exact task-distance columns.
        uniq_x = np.unique(np.round(x, 10))
        ax.set_xticks(uniq_x)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.18)

        title = rf"$\alpha/\pi = {table['alpha_over_pi']:.2f}$"

        # Quantify how much same-distance variability midpoint explains.
        r2_parts = []
        for d_target in [0.2, 1.0 / 3.0, 0.4666666667, 2.0 / 3.0, 0.8]:
            mask = np.isclose(x, d_target, atol=1e-6)
            if np.sum(mask) >= 6:
                r2 = harmonic_midpoint_r2(y[mask], z[mask])
                r2_parts.append(r2)

        if demean_by_distance:
            midpoint_r2 = harmonic_midpoint_r2(y, z)
            title_parts = []
            if np.isfinite(midpoint_r2):
                title_parts.append(rf"pooled residual $R^2 \approx {midpoint_r2:.2f}$")
            if r2_parts:
                title_parts.append(rf"slice residual $R^2 \approx {np.mean(r2_parts):.2f}$")
            if title_parts:
                title += "\n" + " | ".join(title_parts)
        else:
            if r2_parts:
                title += "\n" + rf"slice midpoint $R^2 \approx {np.mean(r2_parts):.2f}$"

        ax.set_title(title)
        ax.set_xlabel(r"Task distance $/ \pi$")

    axes[0].set_ylabel(r"Pair midpoint $/ \pi$")
    axes[0].set_ylim(0.0, 2.0)

    cbar = fig.colorbar(scatter_for_cbar, ax=axes, orientation="vertical", fraction=0.04, pad=0.04)
    cbar_label = response_col.replace("_", " ")
    if demean_by_distance:
        cbar_label += " residual"
    cbar.set_label(cbar_label)

    figure_title = f"Pair midpoint structure at fixed task distance\n{response_col}"
    if demean_by_distance:
        figure_title += " residual after subtracting mean at each distance"
    else:
        figure_title += " (antipodal pairs omitted)"
    fig.suptitle(
        figure_title,
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return save_path


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Plot pairwise response values against task distance and pair midpoint. "
            "This is useful for seeing hidden structure at fixed distance."
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
        default="emp_update_full_minus_bias",
        help="Column from pairwise_metrics.csv to color by.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        required=True,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--demean-by-distance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Subtract the mean response at each task-distance column before plotting and fitting midpoint structure.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    run_tables = [build_pair_table(results_dir, args.response_col) for results_dir in args.results_dirs]
    save_path = make_plot(
        run_tables,
        args.response_col,
        args.save_path,
        demean_by_distance=args.demean_by_distance,
    )
    print(f"Saved plot: {save_path}")


if __name__ == "__main__":
    main()
