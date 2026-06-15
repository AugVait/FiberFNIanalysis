from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import DEFAULT_RESULTS_DIR, resolve_path
from .plot_style import COLORS, apply_axes_style, save_figure, set_publication_style
from .wavelength_cut_fit_results import DEFAULT_OUT_SUBDIR, interval_sort_key, position_sort_key


def numeric_values(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return one numeric column with non-finite values removed."""
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan)


def add_duplicate_offsets(frame: pd.DataFrame) -> pd.DataFrame:
    """Add small offsets so repeated files at the same grid cell remain visible."""
    keyed = frame.sort_values(["sample", "position", "interval", "source_file"]).copy()
    groups = keyed.groupby(["sample", "position", "interval"], dropna=False)
    keyed["replicate_index"] = groups.cumcount()
    keyed["replicate_count"] = groups["source_file"].transform("count")
    centered = keyed["replicate_index"] - (keyed["replicate_count"] - 1) / 2.0
    keyed["x_offset"] = centered * 0.16
    return keyed


def plot_sample_grid(
    *,
    sample: str,
    frame: pd.DataFrame,
    value_column: str,
    value_label: str,
    title_prefix: str,
    out_base: Path,
) -> None:
    """Plot one sample as positions x wavelength intervals with fit values as colored points."""
    values = numeric_values(frame, value_column)
    plot_frame = frame.assign(plot_value=values).dropna(subset=["plot_value"]).copy()
    if plot_frame.empty:
        return

    intervals = sorted(plot_frame["interval"].dropna().unique(), key=interval_sort_key)
    positions = sorted(plot_frame["position"].dropna().unique(), key=position_sort_key)
    x_index = {interval: index for index, interval in enumerate(intervals)}
    y_index = {position: index for index, position in enumerate(positions)}
    plot_frame = add_duplicate_offsets(plot_frame)
    x = plot_frame["interval"].map(x_index).astype(float).to_numpy() + plot_frame["x_offset"].to_numpy(dtype=float)
    y = plot_frame["position"].map(y_index).astype(float).to_numpy()
    color_values = plot_frame["plot_value"].to_numpy(dtype=float)

    width = max(7.2, 0.46 * len(intervals) + 2.5)
    height = max(3.8, 0.42 * len(positions) + 1.5)
    set_publication_style(base_font_size=8.0)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
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
    )
    ax.set_xticks(range(len(intervals)))
    ax.set_xticklabels(intervals, rotation=45, ha="right")
    ax.set_yticks(range(len(positions)))
    ax.set_yticklabels(positions)
    ax.set_xlim(-0.65, len(intervals) - 0.35)
    ax.set_ylim(-0.65, len(positions) - 0.35)
    ax.invert_yaxis()
    ax.set_xlabel("Wavelength interval")
    ax.set_ylabel("Position")
    ax.set_title(f"{title_prefix}: {sample}", pad=5)
    apply_axes_style(ax, grid=True, minor_ticks=False)
    ax.set_axisbelow(True)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.015, fraction=0.035)
    colorbar.set_label(value_label)
    if (plot_frame["replicate_count"] > 1).any():
        ax.text(
            0.995,
            0.01,
            "jittered points indicate replicate files",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.0,
            color=COLORS["gray"],
        )
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
) -> int:
    """Write one grid plot per sample from one fit-result table."""
    frame = pd.read_csv(table_path, sep="\t")
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for sample in sorted(frame["sample"].dropna().unique()):
        sample_frame = frame[frame["sample"] == sample].copy()
        out_base = out_dir / f"{sample}_{title_prefix.lower().replace(' ', '_')}_grid"
        plot_sample_grid(
            sample=sample,
            frame=sample_frame,
            value_column=value_column,
            value_label=value_label,
            title_prefix=title_prefix,
            out_base=out_base,
        )
        count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    """Create sample-wise point-grid summary plots for wavelength-cut fits."""
    parser = argparse.ArgumentParser(description="Plot wavelength-cut rise and decay fit grids by sample.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--fit-subdir", default=DEFAULT_OUT_SUBDIR)
    parser.add_argument("--out-subdir", default="summary_point_grids")
    args = parser.parse_args(argv)

    results_dir = resolve_path(args.results_dir)
    fit_dir = results_dir / args.fit_subdir
    out_root = fit_dir / args.out_subdir
    rise_count = write_summary_plots(
        table_path=fit_dir / "rise_time_2ns" / "rise_time_fits_2ns.txt",
        value_column="fitted_rise_time_10_90_ns",
        value_label="10-90% rise time (ns)",
        title_prefix="Rise time",
        out_dir=out_root / "rise_time_2ns",
    )
    decay_count = write_summary_plots(
        table_path=fit_dir / "decay_time_10ns" / "decay_time_fits_10ns.txt",
        value_column="tau_ns",
        value_label="Decay time tau (ns)",
        title_prefix="Decay time",
        out_dir=out_root / "decay_time_10ns",
    )
    print(f"rise summary plots: {rise_count}")
    print(f"decay summary plots: {decay_count}")
    print(f"output: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
