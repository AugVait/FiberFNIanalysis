from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

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


def apply_manual_selection(frame: pd.DataFrame, selection_path: Path) -> pd.DataFrame:
    """Keep only rows enabled in the matching manual selection matrix."""
    ensure_selection_table(selection_path, frame)
    selected = read_selection_table(selection_path)
    if selected.empty:
        return frame.iloc[0:0].copy()
    return frame.merge(selected, on=["position", "interval"], how="inner")


def aggregate_points(frame: pd.DataFrame) -> pd.DataFrame:
    """Average repeated rows at the same position and wavelength center."""
    if frame.empty:
        return frame.copy()
    return (
        frame.groupby(["position", "band_center_nm"], dropna=False)
        .agg(
            plot_value=("plot_value", "mean"),
            replicate_count=("plot_value", "size"),
            band_min_nm=("band_min_nm", "min"),
            band_max_nm=("band_max_nm", "max"),
        )
        .reset_index()
    )


def plot_sample_grid(
    *,
    sample_name: str,
    frame: pd.DataFrame,
    value_column: str,
    value_label: str,
    title_prefix: str,
    out_base: Path,
    outlier_threshold: float | None = None,
    color_min: float | None = None,
) -> None:
    """Plot one sample as positions x wavelength intervals with fit values as colored points."""
    if frame.empty:
        return

    values = numeric_values(frame, value_column)
    centers = numeric_values(frame, "band_center_nm")
    plot_frame = frame.assign(plot_value=values, band_center_nm=centers).dropna(
        subset=["plot_value", "band_center_nm"]
    )
    if plot_frame.empty:
        return

    outlier_frame = pd.DataFrame(columns=plot_frame.columns)
    if outlier_threshold is not None:
        outlier_mask = plot_frame["plot_value"] > outlier_threshold
        outlier_frame = plot_frame[outlier_mask].copy()
        plot_frame = plot_frame[~outlier_mask].copy()

    positions = sorted(plot_frame["position"].dropna().unique(), key=position_sort_key)
    if not outlier_frame.empty:
        positions = sorted(
            set(positions).union(outlier_frame["position"].dropna().unique()),
            key=position_sort_key,
        )
    plot_frame = aggregate_points(plot_frame)
    outlier_frame = aggregate_points(outlier_frame)
    axis_frame = pd.concat(
        [plot_frame, outlier_frame],
        ignore_index=True,
    )
    centers = sorted(plot_frame["band_center_nm"].dropna().unique())
    if not outlier_frame.empty:
        centers = sorted(set(centers).union(outlier_frame["band_center_nm"].dropna().unique()))
    y_index = {position: index for index, position in enumerate(positions)}
    x = plot_frame["band_center_nm"].to_numpy(dtype=float)
    y = plot_frame["position"].map(y_index).astype(float).to_numpy()
    color_values = plot_frame["plot_value"].to_numpy(dtype=float)

    width = max(6.0, 0.18 * len(centers) + 3.6)
    height = max(5.2, 0.50 * len(positions) + 2.2)
    set_publication_style(base_font_size=8.0)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    scatter = None
    if not plot_frame.empty:
        scatter_kwargs = {}
        if outlier_threshold is not None:
            scatter_kwargs = {"vmin": color_min, "vmax": outlier_threshold}
        scatter = ax.scatter(
            x,
            y,
            c=color_values,
            cmap="viridis",
            s=48,
            marker="o",
            linewidths=0.45,
            edgecolors=COLORS["black"],
            alpha=0.92,
            **scatter_kwargs,
        )
    if not outlier_frame.empty:
        ax.scatter(
            outlier_frame["band_center_nm"].to_numpy(dtype=float),
            outlier_frame["position"].map(y_index).astype(float).to_numpy(),
            s=56,
            marker="x",
            linewidths=1.0,
            color=COLORS["vermillion"],
            label=f">{outlier_threshold:g} ns outlier" if outlier_threshold is not None else "outlier",
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
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.015, fraction=0.035)
        colorbar.set_label(value_label)
    if not outlier_frame.empty:
        ax.legend(loc="upper right", frameon=True, framealpha=0.88, facecolor="white", edgecolor="#C9CED6")
    save_figure(fig, out_base.with_suffix(".png"), dpi=260)
    save_figure(fig, out_base.with_suffix(".pdf"))
    plt.close(fig)


def write_summary_plots(
    *,
    table_path: Path,
    value_column: str,
    value_label: str,
    title_prefix: str,
    out_dir: Path,
    fiber_names: FiberNameMap,
    selection_dir: Path,
    selection_suffix: str,
    outlier_threshold: float | None = None,
    color_min: float | None = None,
) -> int:
    """Write one grid plot per sample from one fit-result table."""
    frame = pd.read_csv(table_path, sep="\t")
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for sample in sorted(frame["sample"].dropna().unique(), key=fiber_names.real_name):
        sample_frame = frame[frame["sample"] == sample].copy()
        selection_path = selection_dir / f"{sample}_{selection_suffix}_selection.txt"
        sample_frame = apply_manual_selection(sample_frame, selection_path)
        out_base = out_dir / f"{sample}_{title_prefix.lower().replace(' ', '_')}_grid"
        plot_sample_grid(
            sample_name=fiber_names.real_name(sample),
            frame=sample_frame,
            value_column=value_column,
            value_label=value_label,
            title_prefix=title_prefix,
            out_base=out_base,
            outlier_threshold=outlier_threshold,
            color_min=color_min,
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
    parser.add_argument("--fiber-names-config", type=Path, default=DEFAULT_FIBER_NAMES_CONFIG)
    args = parser.parse_args(argv)

    results_dir = resolve_path(args.results_dir)
    fiber_names_config = resolve_path(args.fiber_names_config)
    fiber_names = read_fiber_name_map(fiber_names_config)
    fit_dir = results_dir / args.fit_subdir
    out_root = fit_dir / args.out_subdir
    selection_root = fit_dir / args.selection_subdir
    rise_count = write_summary_plots(
        table_path=fit_dir / "rise_time_2ns" / "rise_time_fits_2ns.txt",
        value_column="fitted_rise_time_10_90_ns",
        value_label="10-90% rise time (ns)",
        title_prefix="Rise time",
        out_dir=out_root / "rise_time_2ns",
        fiber_names=fiber_names,
        selection_dir=selection_root / "rise_time_2ns",
        selection_suffix="rise_time_2ns_by_position_interval",
    )
    decay_count = write_summary_plots(
        table_path=fit_dir / "decay_time_10ns" / "decay_time_fits_10ns.txt",
        value_column="tau_ns",
        value_label="Decay time tau (ns)",
        title_prefix="Decay time",
        out_dir=out_root / "decay_time_10ns",
        fiber_names=fiber_names,
        selection_dir=selection_root / "decay_time_10ns",
        selection_suffix="decay_time_10ns_by_position_interval",
        outlier_threshold=3.0,
        color_min=1.0,
    )
    print(f"rise summary plots: {rise_count}")
    print(f"decay summary plots: {decay_count}")
    print(f"output: {out_root}")
    print(f"manual selections: {selection_root}")
    print(f"fiber names: {fiber_names_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
