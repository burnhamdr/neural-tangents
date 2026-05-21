import argparse
import csv
from pathlib import Path
import re

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

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
    if text is None:
        return None
    items = [item.strip() for item in str(text).split(",")]
    return [item for item in items if item]


def discover_results_dirs(explicit_dirs=None, results_root=None):
    discovered = []
    seen = set()

    if explicit_dirs:
        for item in explicit_dirs:
            path = Path(item).resolve()
            if path in seen:
                continue
            seen.add(path)
            discovered.append(path)

    if results_root is not None:
        root = Path(results_root).resolve()
        for pairwise_csv in sorted(root.rglob("pairwise_metrics.csv")):
            parent = pairwise_csv.parent.resolve()
            if parent in seen:
                continue
            seen.add(parent)
            discovered.append(parent)

    valid = []
    for path in discovered:
        if (path / "pairwise_metrics.csv").exists() and (path / "task_metadata.csv").exists():
            valid.append(path)
    return valid


def is_even_phase_control(target_phase, tol=1e-6):
    phase = float(target_phase)
    return abs(np.cos(phase)) < tol and abs(abs(np.sin(phase)) - 1.0) < tol


def infer_regime_label(metadata):
    task_family = metadata.get("task_family") or "unknown"
    target_mode = metadata.get("target_mode") or "local_harmonic"
    global_field = metadata.get("global_field_name") or "xz"
    target_phase = float(metadata.get("target_phase", 0.0))

    if task_family == "single_axis" and target_mode == "local_harmonic":
        if is_even_phase_control(target_phase):
            return "domain_only_even"
        return "domain_plus_target_local"
    if target_mode == "global_field":
        return f"global_field_{global_field}"
    return f"{task_family}_{target_mode}"


def load_task_metadata(path):
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    rows = sorted(rows, key=lambda row: int(row["task_index"]))
    return {
        "task_index": np.asarray([int(row["task_index"]) for row in rows], dtype=int),
        "normals": np.asarray(
            [[float(row["normal_x"]), float(row["normal_y"]), float(row["normal_z"])] for row in rows],
            dtype=float,
        ),
        "phases": np.asarray([float(row["phase"]) for row in rows], dtype=float),
        "target_phase": float(rows[0].get("target_phase", 0.0) or 0.0),
        "target_mode": rows[0].get("target_mode", "local_harmonic") or "local_harmonic",
        "global_field_name": rows[0].get("global_field_name", "xz") or "xz",
        "m": int(float(rows[0]["m"])) if rows[0].get("m", "") not in {"", None} else None,
        "task_family": rows[0].get("task_family", "") or None,
        "color_values": np.asarray([float(row["color_value"]) for row in rows], dtype=float),
    }


def load_run_metadata(results_dir):
    metadata = load_task_metadata(Path(results_dir) / "task_metadata.csv")
    run_name = Path(results_dir).resolve().name
    if metadata["m"] is None:
        match = re.search(r"_m(\d+)", run_name)
        if match is not None:
            metadata["m"] = int(match.group(1))
    metadata["results_dir"] = str(Path(results_dir).resolve())
    metadata["run_name"] = run_name
    metadata["regime_label"] = infer_regime_label(metadata)
    return metadata


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
        compat.append(np.cos(float(m) * delta))
    return {
        "phase_delta_wrapped": np.asarray(wrapped, dtype=float),
        "phase_delta_abs": np.asarray(abs_wrapped, dtype=float),
        "phase_compat": np.asarray(compat, dtype=float),
    }


def circle_points_from_phi(normal, phi):
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    v1 = np.array([1.0, 0.0, 0.0], dtype=float) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=float)
    v1 = v1 - n * np.dot(n, v1)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(n, v1)
    phi = np.asarray(phi, dtype=float)
    X = np.outer(np.cos(phi), v1) + np.outer(np.sin(phi), v2)
    return X


def evaluate_global_sphere_field(X, field_name="xz"):
    X = np.asarray(X, dtype=float)
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]

    if field_name == "xz":
        return 2.0 * x * z
    if field_name == "x2_minus_y2":
        return x ** 2 - y ** 2
    if field_name == "z2_legendre":
        return 0.5 * (3.0 * z ** 2 - 1.0)
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
        return np.sin(float(m) * np.asarray(phi, dtype=float) + float(target_phase))
    if target_mode == "global_field":
        return evaluate_global_sphere_field(X, field_name=global_field_name)
    raise ValueError("target_mode must be 'local_harmonic' or 'global_field'")


def generate_circle_samples(
    normal,
    pts,
    m,
    *,
    phase=0.0,
    target_phase=0.0,
    target_mode="local_harmonic",
    global_field_name="xz",
):
    phi = np.linspace(0.0, 2.0 * np.pi, int(pts), endpoint=False) + float(phase)
    phi = np.mod(phi, 2.0 * np.pi)
    X = circle_points_from_phi(normal, phi)
    y = evaluate_task_targets(
        X,
        phi,
        m,
        target_phase=target_phase,
        target_mode=target_mode,
        global_field_name=global_field_name,
    )
    return X, y, phi


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
    task_targets = []
    for normal, phase in zip(np.asarray(normals, dtype=float), np.asarray(phases, dtype=float)):
        _, y, _ = generate_circle_samples(
            normal,
            pts_per_circle,
            int(m),
            phase=float(phase),
            target_phase=float(target_phase),
            target_mode=target_mode,
            global_field_name=global_field_name,
        )
        task_targets.append(np.asarray(y, dtype=float))

    return np.asarray(
        [best_circular_target_correlation(task_targets[i], task_targets[j]) for i, j in pairs],
        dtype=float,
    )


def load_pairwise_table(results_dir, metadata):
    rows = read_csv_rows(Path(results_dir) / "pairwise_metrics.csv")
    if not rows:
        raise ValueError(f"No pairwise rows found in {results_dir}")

    pairs = [(int(row["task_i"]), int(row["task_j"])) for row in rows]
    table = {
        "pairs": pairs,
        "distance": np.asarray([float(row["distance"]) for row in rows], dtype=float),
    }

    pairwise_columns = set(rows[0].keys())
    for column in pairwise_columns:
        if column in {"pair_index", "task_i", "task_j", "distance"}:
            continue
        values = []
        missing = False
        for row in rows:
            raw = row.get(column, "")
            if raw in {"", None}:
                missing = True
                break
            values.append(float(raw))
        if not missing:
            table[column] = np.asarray(values, dtype=float)

    if "phase_compat" not in table:
        phase_metrics = pairwise_phase_metrics(metadata["phases"], pairs, metadata["m"])
        table["phase_delta_wrapped"] = np.asarray(phase_metrics["phase_delta_wrapped"], dtype=float)
        table["phase_delta_abs"] = np.asarray(phase_metrics["phase_delta_abs"], dtype=float)
        table["phase_compat"] = np.asarray(phase_metrics["phase_compat"], dtype=float)

    if "teacher_corr" not in table:
        table["teacher_corr"] = pairwise_teacher_correlation(
            normals=metadata["normals"],
            phases=metadata["phases"],
            m=metadata["m"],
            pairs=pairs,
            target_phase=metadata["target_phase"],
            target_mode=metadata["target_mode"],
            global_field_name=metadata["global_field_name"],
        )

    return table


def feature_is_variable(values, tol=1e-8):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return False
    return float(np.std(values)) > tol


def build_feature_table(pairwise_table):
    distance = np.asarray(pairwise_table["distance"], dtype=float)
    d_pi = distance / np.pi
    cos_distance = np.cos(distance)
    teacher_corr = np.asarray(pairwise_table["teacher_corr"], dtype=float)
    phase_compat = np.asarray(pairwise_table["phase_compat"], dtype=float)

    return {
        "distance_pi": d_pi,
        "distance_pi_sq": d_pi ** 2,
        "cos_distance": cos_distance,
        "legendre_p2": 0.5 * (3.0 * cos_distance ** 2 - 1.0),
        "phase_compat": phase_compat,
        "distance_x_phase": d_pi * phase_compat,
        "teacher_corr": teacher_corr,
        "distance_x_teacher": d_pi * teacher_corr,
    }


def candidate_model_specs(feature_table):
    specs = [
        ("distance_linear", ["distance_pi"]),
        ("distance_quadratic", ["distance_pi", "distance_pi_sq"]),
        ("legendre_deg2", ["cos_distance", "legendre_p2"]),
    ]

    if feature_is_variable(feature_table["phase_compat"]):
        specs.append(
            ("distance_plus_phase", ["distance_pi", "distance_pi_sq", "phase_compat", "distance_x_phase"])
        )

    if feature_is_variable(feature_table["teacher_corr"]):
        specs.append(
            ("distance_plus_teacher", ["distance_pi", "distance_pi_sq", "teacher_corr", "distance_x_teacher"])
        )
        specs.append(("teacher_only", ["teacher_corr"]))

    return specs


def fit_ols(y, feature_table, feature_names):
    y = np.asarray(y, dtype=float)
    cols = [np.asarray(feature_table[name], dtype=float) for name in feature_names]

    finite_mask = np.isfinite(y)
    for col in cols:
        finite_mask &= np.isfinite(col)

    y_fit = y[finite_mask]
    x_cols = [col[finite_mask] for col in cols]

    if y_fit.size == 0:
        raise ValueError("No finite observations available for fit.")

    X = np.column_stack([np.ones(y_fit.size, dtype=float)] + x_cols)
    beta, _, rank, _ = np.linalg.lstsq(X, y_fit, rcond=None)
    y_hat = X @ beta

    rss = float(np.sum((y_fit - y_hat) ** 2))
    tss = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r2 = np.nan if tss <= 0.0 else 1.0 - rss / tss

    n_obs = int(y_fit.size)
    n_features = int(len(feature_names))
    n_params = int(n_features + 1)

    if n_obs > n_params and np.isfinite(r2):
        adj_r2 = 1.0 - (1.0 - r2) * (n_obs - 1) / (n_obs - n_params)
    else:
        adj_r2 = np.nan

    if rss > 0.0 and n_obs > 0:
        aic = float(n_obs * np.log(rss / n_obs) + 2.0 * n_params)
    else:
        aic = np.nan

    return {
        "n_obs": n_obs,
        "n_features": n_features,
        "rank": int(rank),
        "rss": rss,
        "r2": float(r2) if np.isfinite(r2) else np.nan,
        "adj_r2": float(adj_r2) if np.isfinite(adj_r2) else np.nan,
        "aic": aic,
        "coefficients": {"intercept": float(beta[0]), **{name: float(beta[idx + 1]) for idx, name in enumerate(feature_names)}},
    }


def resolve_response_cols(pairwise_table, requested_cols):
    if requested_cols is None:
        requested_cols = list(DEFAULT_RESPONSE_COLS)
    elif len(requested_cols) == 1 and requested_cols[0].lower() == "all":
        requested_cols = sorted(
            column
            for column in pairwise_table.keys()
            if column.startswith(("emp_", "inf_")) and pairwise_table[column].ndim == 1
        )

    return [column for column in requested_cols if column in pairwise_table]


def analyze_results_dir(results_dir, response_cols):
    metadata = load_run_metadata(results_dir)
    pairwise_table = load_pairwise_table(results_dir, metadata)
    feature_table = build_feature_table(pairwise_table)
    model_specs = candidate_model_specs(feature_table)

    fit_rows = []
    coef_rows = []

    for response_col in resolve_response_cols(pairwise_table, response_cols):
        response = np.asarray(pairwise_table[response_col], dtype=float)
        for model_name, feature_names in model_specs:
            fit = fit_ols(response, feature_table, feature_names)
            fit_rows.append(
                {
                    "run_name": metadata["run_name"],
                    "results_dir": metadata["results_dir"],
                    "regime_label": metadata["regime_label"],
                    "task_family": metadata["task_family"],
                    "target_mode": metadata["target_mode"],
                    "global_field_name": metadata["global_field_name"],
                    "target_phase": float(metadata["target_phase"]),
                    "m": int(metadata["m"]),
                    "response_col": response_col,
                    "model_name": model_name,
                    "n_obs": fit["n_obs"],
                    "n_features": fit["n_features"],
                    "rank": fit["rank"],
                    "rss": fit["rss"],
                    "r2": fit["r2"],
                    "adj_r2": fit["adj_r2"],
                    "aic": fit["aic"],
                }
            )
            for feature_name, coef in fit["coefficients"].items():
                coef_rows.append(
                    {
                        "run_name": metadata["run_name"],
                        "results_dir": metadata["results_dir"],
                        "regime_label": metadata["regime_label"],
                        "task_family": metadata["task_family"],
                        "target_mode": metadata["target_mode"],
                        "global_field_name": metadata["global_field_name"],
                        "target_phase": float(metadata["target_phase"]),
                        "m": int(metadata["m"]),
                        "response_col": response_col,
                        "model_name": model_name,
                        "feature_name": feature_name,
                        "coefficient": coef,
                    }
                )

    return metadata, fit_rows, coef_rows


def summarize_best_models(fit_rows):
    best_rows = {}
    for row in fit_rows:
        key = (row["run_name"], row["response_col"])
        prev = best_rows.get(key)
        current_score = row["adj_r2"]
        prev_score = prev["adj_r2"] if prev is not None else -np.inf
        if prev is None or (np.isfinite(current_score) and current_score > prev_score):
            best_rows[key] = row
    return list(best_rows.values())


def print_console_summary(best_rows, coef_rows):
    coef_map = {}
    for row in coef_rows:
        key = (row["run_name"], row["response_col"], row["model_name"], row["feature_name"])
        coef_map[key] = row["coefficient"]

    for row in sorted(best_rows, key=lambda item: (item["run_name"], item["response_col"])):
        parts = [
            f"run={row['run_name']}",
            f"response={row['response_col']}",
            f"best_model={row['model_name']}",
            f"adj_R2={row['adj_r2']:.4f}" if np.isfinite(row["adj_r2"]) else "adj_R2=nan",
        ]
        if row["model_name"] == "distance_quadratic":
            quad_key = (row["run_name"], row["response_col"], row["model_name"], "distance_pi_sq")
            if quad_key in coef_map:
                parts.append(f"curvature={coef_map[quad_key]:.4f}")
        print(" | ".join(parts))


def make_m_sweep_plot(fit_rows, coef_rows, response_col, save_path):
    if plt is None:
        return None

    filtered_fits = [row for row in fit_rows if row["response_col"] == response_col]
    if not filtered_fits:
        return None

    model_order = [
        "distance_linear",
        "distance_quadratic",
        "legendre_deg2",
        "distance_plus_phase",
        "distance_plus_teacher",
        "teacher_only",
    ]

    coef_lookup = {
        (row["run_name"], row["response_col"], row["model_name"], row["feature_name"]): row["coefficient"]
        for row in coef_rows
    }

    regimes = sorted({row["regime_label"] for row in filtered_fits})
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(len(regimes), 1)))
    color_map = {regime: colors[idx] for idx, regime in enumerate(regimes)}

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))

    for regime in regimes:
        regime_rows = [row for row in filtered_fits if row["regime_label"] == regime]
        for model_name in model_order:
            rows = [row for row in regime_rows if row["model_name"] == model_name]
            if not rows:
                continue
            rows = sorted(rows, key=lambda item: item["m"])
            ms = np.asarray([row["m"] for row in rows], dtype=float)
            scores = np.asarray([row["adj_r2"] for row in rows], dtype=float)
            axes[0].plot(
                ms,
                scores,
                marker="o",
                ms=5,
                lw=1.8,
                alpha=0.9,
                color=color_map[regime],
                ls="-" if "teacher" not in model_name else "--",
                label=f"{regime}: {model_name}",
            )

        quad_rows = [row for row in regime_rows if row["model_name"] == "distance_quadratic"]
        if quad_rows:
            quad_rows = sorted(quad_rows, key=lambda item: item["m"])
            ms = np.asarray([row["m"] for row in quad_rows], dtype=float)
            curvature = np.asarray(
                [
                    coef_lookup.get((row["run_name"], row["response_col"], row["model_name"], "distance_pi_sq"), np.nan)
                    for row in quad_rows
                ],
                dtype=float,
            )
            axes[1].plot(
                ms,
                curvature,
                marker="o",
                ms=5,
                lw=2.0,
                color=color_map[regime],
                label=regime,
            )

    axes[0].set_title(f"Nested model fit vs m\n{response_col}")
    axes[0].set_xlabel("Local harmonic frequency m")
    axes[0].set_ylabel("Adjusted $R^2$")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=True, fontsize=8, ncol=1)

    axes[1].axhline(0.0, color="0.2", lw=1.0, ls=":")
    axes[1].set_title(f"Distance-profile curvature vs m\n{response_col}")
    axes[1].set_xlabel("Local harmonic frequency m")
    axes[1].set_ylabel("Quadratic coefficient on distance / pi")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=True, fontsize=9)

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return save_path


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Fit nested pairwise similarity-profile models to saved multitask runs. "
            "This is meant to test whether geometry alone explains the profile shape "
            "or whether target/teacher structure is needed."
        )
    )
    parser.add_argument(
        "--results-dirs",
        nargs="*",
        default=None,
        help="Explicit saved run directories to analyze.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Root folder to recursively scan for saved run directories.",
    )
    parser.add_argument(
        "--response-cols",
        type=str,
        default=",".join(DEFAULT_RESPONSE_COLS),
        help="Comma-separated pairwise response columns to model, or 'all'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for aggregate CSVs and sweep plots. Defaults to <results-root>/profile_model_summary.",
    )
    parser.add_argument(
        "--plot-response",
        type=str,
        default=DEFAULT_RESPONSE_COLS[0],
        help="Response column used for the sweep summary plot.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    results_dirs = discover_results_dirs(args.results_dirs, args.results_root)
    if not results_dirs:
        raise SystemExit("No valid saved result directories were found.")

    response_cols = parse_csv_list(args.response_cols)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    elif args.results_root is not None:
        output_dir = Path(args.results_root) / "profile_model_summary"
    else:
        output_dir = Path(results_dirs[0]) / "profile_model_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_fit_rows = []
    all_coef_rows = []
    run_rows = []

    for results_dir in results_dirs:
        metadata, fit_rows, coef_rows = analyze_results_dir(results_dir, response_cols)
        all_fit_rows.extend(fit_rows)
        all_coef_rows.extend(coef_rows)
        run_rows.append(
            {
                "run_name": metadata["run_name"],
                "results_dir": metadata["results_dir"],
                "regime_label": metadata["regime_label"],
                "task_family": metadata["task_family"],
                "target_mode": metadata["target_mode"],
                "global_field_name": metadata["global_field_name"],
                "target_phase": float(metadata["target_phase"]),
                "m": int(metadata["m"]),
            }
        )

    if not all_fit_rows:
        raise SystemExit("No fits were produced from the requested runs.")

    write_csv_rows(output_dir / "run_metadata_summary.csv", list(run_rows[0].keys()), run_rows)
    write_csv_rows(output_dir / "profile_model_fits.csv", list(all_fit_rows[0].keys()), all_fit_rows)
    write_csv_rows(output_dir / "profile_model_coefficients.csv", list(all_coef_rows[0].keys()), all_coef_rows)

    best_rows = summarize_best_models(all_fit_rows)
    write_csv_rows(output_dir / "profile_model_best_by_response.csv", list(best_rows[0].keys()), best_rows)
    print_console_summary(best_rows, all_coef_rows)

    plot_path = make_m_sweep_plot(
        all_fit_rows,
        all_coef_rows,
        response_col=args.plot_response,
        save_path=output_dir / f"m_sweep_{args.plot_response}.png",
    )
    if plot_path is not None:
        print(f"Saved sweep plot: {plot_path}")
    elif plt is None:
        print("Plot skipped because matplotlib is not installed in this environment.")


if __name__ == "__main__":
    main()
