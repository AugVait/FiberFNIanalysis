from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from .fiber_names import read_fiber_name_map
from .paths import DEFAULT_FIBER_NAMES_CONFIG, DEFAULT_RESULTS_DIR, resolve_path
from .plot_style import COLORS, apply_axes_style, save_figure, set_publication_style


DEFAULT_CUTS_SUBDIR = "carpet_wavelength_cuts_20nm_txt"
DEFAULT_OUT_SUBDIR = "decay_tau_slice_position_visuals"

LINE_COLORS = [
    COLORS["blue"],
    COLORS["vermillion"],
    COLORS["teal"],
    COLORS["purple"],
    "#CC79A7",
    "#E69F00",
]
MARKERS = ["o", "s", "^", "D", "v", "P"]


@dataclass(frozen=True)
class MeasurementMeta:
    sample: str
    irradiation: str
    group: str
    position: str
    distance_cm: int | None
    position_suffix: str
    time_window: str


def parse_measurement(source_file: str) -> MeasurementMeta:
    """Extract sample, position, and time-window metadata from one carpet filename."""
    stem = Path(source_file).stem.lower()
    sample_match = re.search(r"(?:^|[^a-z0-9])((?:bcf\d+g?)|(?:scsf\d+))(?=[^a-z0-9]|$)", stem)
    irradiation_match = re.search(r"_(noir|ir)_", stem)
    position_match = re.search(r"_(endcm|\d+cm[a-z0-9]*)(?:_|$)", stem)
    window_match = re.search(r"_(\d+(?:ns|us|ms))(?:_|$)", stem)
    sample = sample_match.group(1) if sample_match else "unknown"
    irradiation = irradiation_match.group(1) if irradiation_match else "unknown"
    position = position_match.group(1) if position_match else ""
    distance_cm = None
    position_suffix = ""
    if position and position != "endcm":
        distance_match = re.match(r"(?P<distance>\d+)cm(?P<suffix>.*)$", position)
        if distance_match:
            distance_cm = int(distance_match.group("distance"))
            position_suffix = distance_match.group("suffix")
    return MeasurementMeta(
        sample=sample,
        irradiation=irradiation,
        group=f"{sample}_{irradiation}",
        position=position,
        distance_cm=distance_cm,
        position_suffix=position_suffix,
        time_window=window_match.group(1) if window_match else "",
    )


def time_window_sort_key(window: str) -> tuple[int, float, str]:
    """Sort acquisition windows by duration."""
    match = re.fullmatch(r"(?P<value>\d+(?:\.\d+)?)(?P<unit>ns|us|ms)", str(window), flags=re.IGNORECASE)
    if not match:
        return (1, math.inf, str(window))
    scale = {"ns": 1.0, "us": 1000.0, "ms": 1_000_000.0}
    return (0, float(match.group("value")) * scale[match.group("unit").lower()], str(window))


def spaced_units(text: str) -> str:
    """Insert a space between numeric values and unit suffixes."""
    return re.sub(r"(?<=\d)(?=[A-Za-z])", " ", str(text))


def format_nm(value: object) -> str:
    """Format a wavelength edge for labels."""
    numeric = float(value)
    return str(int(round(numeric))) if numeric.is_integer() else f"{numeric:g}"


def finite_float(value: object) -> float | None:
    """Convert one value to a finite float when possible."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def csv_values(values: list[str] | None) -> set[str]:
    """Parse repeated or comma-separated scalar filters."""
    if not values:
        return set()
    return {
        item.strip().lower()
        for value in values
        for item in value.split(",")
        if item.strip()
    }


def read_decay_tau_rows(cuts_dir: Path, fiber_names_config: Path) -> pd.DataFrame:
    """Read all successful decay tau rows from batch wavelength-cut outputs."""
    inventory_path = cuts_dir / "inventory.txt"
    if not inventory_path.exists():
        raise FileNotFoundError(f"Missing wavelength-cut inventory: {inventory_path}")
    fiber_names = read_fiber_name_map(fiber_names_config)
    rows: list[dict[str, object]] = []
    with inventory_path.open(encoding="utf-8") as handle:
        for entry in csv.DictReader(handle, delimiter="\t"):
            if entry.get("status") != "ok":
                continue
            source_file = entry.get("source_file", "")
            output_folder = entry.get("output_folder", "")
            meta = parse_measurement(source_file)
            fit_path = cuts_dir / output_folder / "fit_summary.txt"
            if not fit_path.exists():
                continue
            with fit_path.open(encoding="utf-8") as fit_handle:
                for fit_row in csv.DictReader(fit_handle, delimiter="\t"):
                    if fit_row.get("status") != "fit":
                        continue
                    tau = finite_float(fit_row.get("tau_ns"))
                    band_center = finite_float(fit_row.get("band_center_nm"))
                    distance = meta.distance_cm
                    if tau is None or band_center is None or distance is None:
                        continue
                    tau_se = finite_float(fit_row.get("tau_se_ns"))
                    rows.append(
                        {
                            "source_file": source_file,
                            "output_folder": output_folder,
                            "sample": meta.sample,
                            "irradiation": meta.irradiation,
                            "group": meta.group,
                            "fiber_name": fiber_names.real_name(meta.group),
                            "position": meta.position,
                            "distance_cm": distance,
                            "position_suffix": meta.position_suffix,
                            "time_window": meta.time_window,
                            "band_min_nm": float(fit_row["band_min_nm"]),
                            "band_max_nm": float(fit_row["band_max_nm"]),
                            "band_center_nm": band_center,
                            "tau_ns": tau,
                            "tau_se_ns": tau_se if tau_se is not None else np.nan,
                            "r_squared": finite_float(fit_row.get("r_squared")) or np.nan,
                        }
                    )
    if not rows:
        raise ValueError(f"No successful decay fits found under {cuts_dir}")
    return pd.DataFrame(rows)


def aggregate_decay_tau(rows: pd.DataFrame) -> pd.DataFrame:
    """Average replicate measurements at the same group/window/position/wavelength."""
    group_cols = [
        "group",
        "fiber_name",
        "time_window",
        "distance_cm",
        "band_min_nm",
        "band_max_nm",
        "band_center_nm",
    ]
    return (
        rows.groupby(group_cols, dropna=False)
        .agg(
            tau_mean_ns=("tau_ns", "mean"),
            tau_std_ns=("tau_ns", "std"),
            tau_count=("tau_ns", "size"),
            r_squared_median=("r_squared", "median"),
        )
        .reset_index()
        .sort_values(["time_window", "group", "band_center_nm", "distance_cm"], key=lambda col: col)
    )


def write_table(path: Path, frame: pd.DataFrame) -> None:
    """Write a dataframe as tab-delimited text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, float_format="%.10g")


def style_for_groups(groups: list[str]) -> dict[str, tuple[str, str]]:
    """Return stable color/marker choices for plotted fiber groups."""
    return {
        group: (LINE_COLORS[index % len(LINE_COLORS)], MARKERS[index % len(MARKERS)])
        for index, group in enumerate(groups)
    }


def plot_small_multiples_by_slice(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Create one small-multiples decay-tau figure per acquisition window."""
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for window in sorted(frame["time_window"].dropna().unique(), key=time_window_sort_key):
        subset = frame[frame["time_window"] == window]
        if subset.empty:
            continue
        bands = sorted(subset["band_center_nm"].unique())
        groups = sorted(subset["group"].unique(), key=lambda group: subset.loc[subset["group"] == group, "fiber_name"].iloc[0])
        styles = style_for_groups(groups)
        cols = min(4, max(1, math.ceil(math.sqrt(len(bands)))))
        rows = math.ceil(len(bands) / cols)
        set_publication_style(base_font_size=7.6)
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(cols * 3.05, rows * 2.25),
            sharex=True,
            constrained_layout=True,
        )
        axes_arr = np.atleast_1d(axes).ravel()
        handles: dict[str, object] = {}
        for ax, band_center in zip(axes_arr, bands):
            band = subset[subset["band_center_nm"] == band_center]
            band_min = band["band_min_nm"].iloc[0]
            band_max = band["band_max_nm"].iloc[0]
            for group in groups:
                group_rows = band[band["group"] == group].sort_values("distance_cm")
                if group_rows.empty:
                    continue
                color, marker = styles[group]
                label = str(group_rows["fiber_name"].iloc[0])
                line = ax.plot(
                    group_rows["distance_cm"],
                    group_rows["tau_mean_ns"],
                    marker=marker,
                    ms=3.2,
                    lw=0.9,
                    color=color,
                    markerfacecolor="white" if group.endswith("_ir") else color,
                    markeredgecolor=color,
                    markeredgewidth=0.75,
                    label=label,
                )[0]
                handles.setdefault(label, line)
            ax.set_title(f"{format_nm(band_min)}-{format_nm(band_max)} nm", pad=3)
            ax.set_xlabel("Position (cm)")
            ax.set_ylabel(r"$\tau$ (ns)")
            ax.set_ylim(bottom=0)
            apply_axes_style(ax, grid=True)
        for ax in axes_arr[len(bands) :]:
            ax.axis("off")
        fig.suptitle(f"Decay time by position for each wavelength slice, {spaced_units(window)}", fontsize=9.2)
        if handles:
            fig.legend(
                list(handles.values()),
                list(handles.keys()),
                loc="center left",
                bbox_to_anchor=(1.002, 0.5),
                frameon=False,
                title="filled: Non-IR\nopen: IR",
            )
        png = out_dir / f"small_multiples_decay_tau_{window}.png"
        pdf = out_dir / f"small_multiples_decay_tau_{window}.pdf"
        save_figure(fig, png, dpi=260)
        save_figure(fig, pdf)
        plt.close(fig)
        output_paths.extend([png, pdf])
    return output_paths


def interpolated_grid(group_rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Build an interpolated position/wavelength grid for contour plotting."""
    positions = np.array(sorted(group_rows["distance_cm"].unique()), dtype=float)
    bands = np.array(sorted(group_rows["band_center_nm"].unique()), dtype=float)
    if positions.size < 2 or bands.size < 2 or len(group_rows) < 4:
        return None
    x_grid, y_grid = np.meshgrid(
        np.linspace(float(positions.min()), float(positions.max()), max(80, positions.size * 12)),
        np.linspace(float(bands.min()), float(bands.max()), max(80, bands.size * 8)),
    )
    points = group_rows[["distance_cm", "band_center_nm"]].to_numpy(dtype=float)
    values = group_rows["tau_mean_ns"].to_numpy(dtype=float)
    method = "linear" if len(np.unique(points[:, 0])) >= 2 and len(np.unique(points[:, 1])) >= 2 else "nearest"
    z_grid = griddata(points, values, (x_grid, y_grid), method=method)
    if np.isnan(z_grid).all():
        z_grid = griddata(points, values, (x_grid, y_grid), method="nearest")
    else:
        nearest = griddata(points, values, (x_grid, y_grid), method="nearest")
        z_grid = np.where(np.isfinite(z_grid), z_grid, nearest)
    return x_grid, y_grid, z_grid


def plot_contour_maps(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Create one multi-panel contour map per acquisition window."""
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for window in sorted(frame["time_window"].dropna().unique(), key=time_window_sort_key):
        subset = frame[frame["time_window"] == window]
        if subset.empty:
            continue
        groups = sorted(subset["group"].unique(), key=lambda group: subset.loc[subset["group"] == group, "fiber_name"].iloc[0])
        cols = min(3, max(1, len(groups)))
        rows = math.ceil(len(groups) / cols)
        finite_tau = subset["tau_mean_ns"].replace([np.inf, -np.inf], np.nan).dropna()
        if finite_tau.empty:
            continue
        vmin = float(finite_tau.quantile(0.02))
        vmax = float(finite_tau.quantile(0.98))
        if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
            vmin = float(finite_tau.min())
            vmax = float(finite_tau.max())
        levels = np.linspace(vmin, vmax, 15) if vmax > vmin else 12
        set_publication_style(base_font_size=8.0)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 3.0), constrained_layout=True)
        axes_arr = np.atleast_1d(axes).ravel()
        contour = None
        for ax, group in zip(axes_arr, groups):
            group_rows = subset[subset["group"] == group]
            grid = interpolated_grid(group_rows)
            if grid is None:
                ax.text(0.5, 0.5, "not enough data", transform=ax.transAxes, ha="center", va="center")
                ax.set_axis_off()
                continue
            x_grid, y_grid, z_grid = grid
            z_plot = np.clip(z_grid, vmin, vmax)
            contour = ax.contourf(x_grid, y_grid, z_plot, levels=levels, cmap="viridis", extend="both")
            ax.scatter(
                group_rows["distance_cm"],
                group_rows["band_center_nm"],
                s=7,
                c="white",
                edgecolors=COLORS["black"],
                linewidths=0.25,
                alpha=0.65,
            )
            ax.set_title(str(group_rows["fiber_name"].iloc[0]), pad=4)
            ax.set_xlabel("Position (cm)")
            ax.set_ylabel("Wavelength center (nm)")
            apply_axes_style(ax, grid=False)
        for ax in axes_arr[len(groups) :]:
            ax.axis("off")
        fig.suptitle(f"Decay-time contour maps, {spaced_units(window)}", fontsize=9.5)
        if contour is not None:
            colorbar = fig.colorbar(contour, ax=axes_arr[: len(groups)], pad=0.015, fraction=0.035)
            colorbar.set_label(r"Decay time $\tau$ (ns)")
        png = out_dir / f"contour_decay_tau_{window}.png"
        pdf = out_dir / f"contour_decay_tau_{window}.pdf"
        save_figure(fig, png, dpi=260)
        save_figure(fig, pdf)
        plt.close(fig)
        output_paths.extend([png, pdf])
    return output_paths


def main(argv: list[str] | None = None) -> int:
    """Plot decay time versus wavelength slice and position."""
    parser = argparse.ArgumentParser(description="Plot decay tau by wavelength slice and fiber position.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--cuts-subdir", default=DEFAULT_CUTS_SUBDIR)
    parser.add_argument("--out-subdir", default=DEFAULT_OUT_SUBDIR)
    parser.add_argument("--fiber-names-config", type=Path, default=DEFAULT_FIBER_NAMES_CONFIG)
    parser.add_argument(
        "--time-window",
        "--time-windows",
        dest="time_windows",
        action="append",
        default=None,
        help="Keep only selected acquisition windows, e.g. --time-window 10ns or --time-windows 2ns,10ns.",
    )
    args = parser.parse_args(argv)

    results_dir = resolve_path(args.results_dir)
    cuts_dir = results_dir / args.cuts_subdir
    out_dir = results_dir / args.out_subdir
    fiber_names_config = resolve_path(args.fiber_names_config)

    raw_rows = read_decay_tau_rows(cuts_dir, fiber_names_config)
    selected_windows = csv_values(args.time_windows)
    if selected_windows:
        raw_rows = raw_rows[raw_rows["time_window"].str.lower().isin(selected_windows)].copy()
        if raw_rows.empty:
            raise SystemExit(f"No successful decay fits found for time window(s): {', '.join(sorted(selected_windows))}")
    aggregate = aggregate_decay_tau(raw_rows)
    write_table(out_dir / "compiled_decay_tau_by_slice.txt", raw_rows)
    write_table(out_dir / "aggregated_decay_tau_by_slice.txt", aggregate)
    small_paths = plot_small_multiples_by_slice(aggregate, out_dir / "small_multiples_by_slice")
    contour_paths = plot_contour_maps(aggregate, out_dir / "contour_maps")

    print(f"raw fit rows: {len(raw_rows)}")
    print(f"aggregated rows: {len(aggregate)}")
    print(f"small-multiple files: {len(small_paths)}")
    print(f"contour files: {len(contour_paths)}")
    print(f"output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
