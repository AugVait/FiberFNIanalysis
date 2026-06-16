from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fiber_names import FiberNameMap, read_fiber_name_map
from .paths import DEFAULT_FIBER_NAMES_CONFIG, DEFAULT_RESULTS_DIR, resolve_path
from .plot_style import COLORS, apply_axes_style, save_figure, set_publication_style
from .wavelength_cut_fit_results import DEFAULT_OUT_SUBDIR, interval_sort_key, position_sort_key


def numeric_values(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return one numeric column with non-finite values removed."""
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan)


def format_nm_tick(value: object) -> str:
    """Format a wavelength tick for a numeric nm axis."""
    numeric = float(value)
    return str(int(round(numeric))) if numeric.is_integer() else f"{numeric:g}"


def wavelength_xlim(frame: pd.DataFrame) -> tuple[float, float]:
    """Return plot limits spanning the wavelength interval edges when available."""
    if {"band_min_nm", "band_max_nm"}.issubset(frame.columns):
        lower = numeric_values(frame, "band_min_nm").dropna()
        upper = numeric_values(frame, "band_max_nm").dropna()
        if not lower.empty and not upper.empty:
            return float(lower.min()), float(upper.max())

    centers = sorted(float(value) for value in frame["band_center_nm"].dropna().unique())
    if len(centers) > 1:
        step = float(np.median(np.diff(centers)))
    else:
        step = 20.0
    margin = max(step / 2.0, 1.0)
    return centers[0] - margin, centers[-1] + margin


def is_r_scan_position(position: object) -> bool:
    """Return True for repeat-scan position labels such as 37cmR."""
    return str(position).strip().lower().endswith("r")


def selection_value_enabled(value: object) -> bool:
    """Return whether one manual selection matrix cell keeps a point."""
    text = str(value).strip().lower()
    if text in {"", "0", "false", "no", "n", "skip", "exclude"}:
        return False
    if text in {"true", "yes", "y", "keep", "include", "x"}:
        return True
    try:
        return float(text) > 0.0
    except ValueError:
        return False


def default_selection_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the editable include/exclude matrix from the current result rows."""
    positions = sorted(frame["position"].dropna().unique(), key=position_sort_key)
    intervals = sorted(frame["interval"].dropna().unique(), key=interval_sort_key)
    count_matrix = (
        frame.pivot_table(
            index="position",
            columns="interval",
            values="source_file",
            aggfunc="count",
            dropna=False,
        )
        .reindex(index=positions, columns=intervals)
        .fillna(0)
        .astype(int)
    )
    for position in count_matrix.index:
        if is_r_scan_position(position):
            count_matrix.loc[position, :] = 0
    return count_matrix.reset_index()


def ensure_selection_table(path: Path, frame: pd.DataFrame) -> None:
    """Create a manual selection matrix if it does not already exist."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    default_selection_matrix(frame).to_csv(path, sep="\t", index=False)


def read_selection_table(path: Path) -> pd.DataFrame:
    """Read selected position/interval cells from an editable matrix file."""
    matrix = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    rows: list[dict[str, str]] = []
    for _, row in matrix.iterrows():
        position = str(row.get("position", "")).strip()
        if not position:
            continue
        for interval in matrix.columns:
            if interval == "position":
                continue
            if selection_value_enabled(row.get(interval, "")):
                rows.append({"position": position, "interval": interval})
    return pd.DataFrame(rows, columns=["position", "interval"])


def apply_manual_selection(
    frame: pd.DataFrame,
    selection_path: Path,
    *,
    create_missing_selection: bool = True,
) -> pd.DataFrame:
    """Keep only rows enabled in the matching manual selection matrix."""
    if create_missing_selection:
        ensure_selection_table(selection_path, frame)
    elif not selection_path.exists():
        raise SystemExit(f"Missing manual selection matrix: {selection_path}")
    selected = read_selection_table(selection_path)
    if selected.empty:
        return frame.iloc[0:0].copy()
    return frame.merge(selected, on=["position", "interval"], how="inner")


def combined_standard_error(values: pd.Series) -> float:
    """Combine independent one-sigma errors for one averaged grid point."""
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return float("nan")
    return float(np.sqrt(np.sum(np.square(numeric))) / len(numeric))


def standard_deviation_limits(frame: pd.DataFrame, value_column: str) -> tuple[float | None, float | None]:
    """Return mean +/- 1 SD for one selected fit family."""
    values = numeric_values(frame, value_column).dropna()
    if values.empty:
        return None, None
    center = float(values.mean())
    spread = float(values.std(ddof=0))
    if not np.isfinite(spread):
        return max(center, 0.0), max(center, 0.0)
    return max(center - spread, 0.0), max(center + spread, 0.0)


def format_significant(value: float) -> str:
    """Format one numeric value with three significant figures."""
    return f"{value:.3g}"


def aggregate_points(frame: pd.DataFrame, error_column: str | None = None) -> pd.DataFrame:
    """Average repeated rows at the same position and wavelength center."""
    if frame.empty:
        return frame.copy()
    aggregations = {
        "plot_value": ("plot_value", "mean"),
        "replicate_count": ("plot_value", "size"),
        "band_min_nm": ("band_min_nm", "min"),
        "band_max_nm": ("band_max_nm", "max"),
    }
    if error_column is not None and "plot_error" in frame.columns:
        aggregations["plot_error"] = ("plot_error", combined_standard_error)
    return (
        frame.groupby(["position", "band_center_nm"], dropna=False)
        .agg(**aggregations)
        .reset_index()
    )


def plot_sample_grid(
    *,
    sample_name: str,
    frame: pd.DataFrame,
    value_column: str,
    value_label: str,
    error_column: str | None,
    title_prefix: str,
    out_base: Path,
    outlier_threshold: float | None = None,
    color_min: float | None = None,
    color_max: float | None = None,
) -> None:
    """Plot one sample as positions x wavelength intervals with fit values as colored points."""
    if frame.empty:
        return

    values = numeric_values(frame, value_column)
    centers = numeric_values(frame, "band_center_nm")
    errors = numeric_values(frame, error_column) if error_column is not None and error_column in frame else None
    plot_frame = frame.assign(
        plot_value=values,
        band_center_nm=centers,
        plot_error=errors if errors is not None else np.nan,
    ).dropna(subset=["plot_value", "band_center_nm"])
    if plot_frame.empty:
        return

    positions = sorted(plot_frame["position"].dropna().unique(), key=position_sort_key)
    plot_frame = aggregate_points(plot_frame, error_column=error_column)
    axis_frame = plot_frame.copy()
    centers = sorted(plot_frame["band_center_nm"].dropna().unique())
    y_index = {position: index for index, position in enumerate(positions)}
    x = plot_frame["band_center_nm"].to_numpy(dtype=float)
    y = plot_frame["position"].map(y_index).astype(float).to_numpy()
    color_values = plot_frame["plot_value"].to_numpy(dtype=float)
    error_values = numeric_values(plot_frame, "plot_error") if "plot_error" in plot_frame else pd.Series(dtype=float)

    width = max(8.2, 0.34 * len(centers) + 4.4)
    height = max(6.2, 0.78 * len(positions) + 2.7)
    set_publication_style(base_font_size=10.0)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    colormap = matplotlib.colormaps["managua"]
    lower = float(color_min) if color_min is not None else float(np.nanmin(color_values))
    upper = float(color_max) if color_max is not None else float(np.nanmax(color_values))
    if not np.isfinite(lower):
        lower = 0.0
    if not np.isfinite(upper) or upper <= lower:
        upper = lower + 1.0
    norm = mcolors.Normalize(vmin=lower, vmax=upper, clip=True)
    x_step = float(np.median(np.diff(centers))) if len(centers) > 1 else 20.0
    marker_width = min(max(0.30 * x_step, 4.0), 8.0)
    marker_height = 0.055
    for row in plot_frame.itertuples(index=False):
        error_text = ""
        if np.isfinite(getattr(row, "plot_error", np.nan)):
            error_text = "+/-" + format_significant(float(row.plot_error))
        y_center = float(y_index[row.position])
        x_center = float(row.band_center_nm)
        ax.add_patch(
            Rectangle(
                (x_center - marker_width / 2.0, y_center + 0.035),
                marker_width,
                marker_height,
                facecolor=colormap(norm(float(row.plot_value))),
                edgecolor="none",
                alpha=0.95,
                zorder=2.5,
            )
        )
        ax.text(
            x_center,
            y_center - 0.12,
            format_significant(float(row.plot_value)),
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="semibold",
            color=COLORS["black"],
        )
        if error_text:
            ax.text(
                x_center,
                y_center + 0.18,
                error_text,
                ha="center",
                va="center",
                fontsize=7.2,
                color="#27313B",
            )
    ax.set_xticks(centers)
    ax.set_xticklabels([format_nm_tick(center) for center in centers])
    ax.set_yticks(range(len(positions)))
    ax.set_yticklabels(positions)
    ax.set_xlim(*wavelength_xlim(axis_frame))
    ax.set_ylim(-0.65, len(positions) - 0.35)
    ax.invert_yaxis()
    ax.set_xlabel("Wavelength center (nm)")
    ax.set_ylabel("Position")
    ax.set_title(f"{title_prefix}: {sample_name}", pad=5)
    apply_axes_style(ax, grid=True, minor_ticks=False)
    ax.set_axisbelow(True)
    colorbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap),
        ax=ax,
        pad=0.015,
        fraction=0.035,
    )
    colorbar.set_label(value_label)
    save_figure(fig, out_base.with_suffix(".png"), dpi=260)
    save_figure(fig, out_base.with_suffix(".pdf"))
    plt.close(fig)


def write_summary_plots(
    *,
    table_path: Path,
    value_column: str,
    value_label: str,
    error_column: str | None,
    title_prefix: str,
    out_dir: Path,
    fiber_names: FiberNameMap,
    selection_dir: Path,
    selection_suffix: str,
    create_missing_selections: bool = True,
    outlier_threshold: float | None = None,
    color_min: float | None = None,
    color_max: float | None = None,
    use_standard_deviation_limits: bool = False,
) -> int:
    """Write one grid plot per sample from one fit-result table."""
    frame = pd.read_csv(table_path, sep="\t")
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_frames: dict[str, pd.DataFrame] = {}
    selected_values: list[pd.DataFrame] = []
    for sample in sorted(frame["sample"].dropna().unique(), key=fiber_names.real_name):
        sample_frame = frame[frame["sample"] == sample].copy()
        selection_path = selection_dir / f"{sample}_{selection_suffix}_selection.txt"
        sample_frame = apply_manual_selection(
            sample_frame,
            selection_path,
            create_missing_selection=create_missing_selections,
        )
        selected_frames[sample] = sample_frame
        if not sample_frame.empty:
            selected_values.append(sample_frame)

    effective_color_min = color_min
    effective_color_max = color_max
    effective_outlier_threshold = outlier_threshold
    if use_standard_deviation_limits and selected_values:
        combined = pd.concat(selected_values, ignore_index=True)
        effective_color_min, effective_color_max = standard_deviation_limits(combined, value_column)
        if effective_outlier_threshold is not None:
            effective_outlier_threshold = effective_color_max

    count = 0
    for sample in sorted(frame["sample"].dropna().unique(), key=fiber_names.real_name):
        sample_frame = selected_frames[sample]
        out_base = out_dir / f"{sample}_{title_prefix.lower().replace(' ', '_')}_grid"
        plot_sample_grid(
            sample_name=fiber_names.real_name(sample),
            frame=sample_frame,
            value_column=value_column,
            value_label=value_label,
            error_column=error_column,
            title_prefix=title_prefix,
            out_base=out_base,
            outlier_threshold=effective_outlier_threshold,
            color_min=effective_color_min,
            color_max=effective_color_max,
        )
        count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    """Create sample-wise point-grid summary plots for wavelength-cut fits."""
    parser = argparse.ArgumentParser(description="Plot wavelength-cut rise and decay fit grids by sample.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--fit-subdir", default=DEFAULT_OUT_SUBDIR)
    parser.add_argument("--out-subdir", default="summary grids")
    parser.add_argument("--selection-subdir", default="manual selections")
    parser.add_argument(
        "--selection-dir",
        type=Path,
        default=None,
        help="Read-only directory containing tracked decay-time selection matrices.",
    )
    parser.add_argument("--fiber-names-config", type=Path, default=DEFAULT_FIBER_NAMES_CONFIG)
    args = parser.parse_args(argv)

    results_dir = resolve_path(args.results_dir)
    fiber_names_config = resolve_path(args.fiber_names_config)
    fiber_names = read_fiber_name_map(fiber_names_config)
    fit_dir = results_dir / args.fit_subdir
    out_root = fit_dir / args.out_subdir
    selection_dir = (
        resolve_path(args.selection_dir)
        if args.selection_dir is not None
        else fit_dir / args.selection_subdir / "decay_time_10ns"
    )
    create_missing_selections = args.selection_dir is None
    rise_count = write_summary_plots(
        table_path=fit_dir / "rise_time_2ns" / "rise_time_fits_2ns.txt",
        value_column="fitted_rise_time_10_90_ns",
        value_label="10-90% rise time (ns)",
        error_column="fitted_rise_time_10_90_se_ns",
        title_prefix="Rise time",
        out_dir=out_root / "rise_time_2ns",
        fiber_names=fiber_names,
        selection_dir=selection_dir,
        selection_suffix="decay_time_10ns_by_position_interval",
        create_missing_selections=create_missing_selections,
        use_standard_deviation_limits=True,
    )
    decay_count = write_summary_plots(
        table_path=fit_dir / "decay_time_10ns" / "decay_time_fits_10ns.txt",
        value_column="tau_ns",
        value_label="Decay time tau (ns)",
        error_column="tau_se_ns",
        title_prefix="Decay time",
        out_dir=out_root / "decay_time_10ns",
        fiber_names=fiber_names,
        selection_dir=selection_dir,
        selection_suffix="decay_time_10ns_by_position_interval",
        create_missing_selections=create_missing_selections,
        outlier_threshold=3.5,
        use_standard_deviation_limits=True,
    )
    print(f"rise summary plots: {rise_count}")
    print(f"decay summary plots: {decay_count}")
    print(f"output: {out_root}")
    print(f"manual selections: {selection_dir}")
    print(f"fiber names: {fiber_names_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
