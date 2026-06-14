from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .fiber_names import FiberNameMap, read_fiber_name_map
from .paths import (
    DEFAULT_FIBER_NAMES_CONFIG,
    DEFAULT_PEAK_SHIFT_CONFIG,
    DEFAULT_RAW_DIR,
    DEFAULT_RESULTS_DIR,
    resolve_path,
)
from .plot_pl_spectra import (
    PLSpectrum,
    load_records,
    read_plot_configs,
    selected_records_from_configs,
)
from .plot_style import (
    COLORS,
    DOUBLE_COLUMN_WIDE,
    apply_axes_style,
    save_figure,
    set_publication_style,
)
from .scan_config import relative_config_dir
from .yaml_config import float_value, read_yaml_mapping, string_value


DEFAULT_SMOOTH_SIGMA_NM = 1.0
DEFAULT_FIT_HALF_WIDTH_NM = 6.0

LINE_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
)
MARKERS = ("o", "s", "^", "D", "v", "P")
MARKER_SIZE = 5.6


@dataclass(frozen=True)
class PeakShiftConfig:
    out_subdir: str = "peak_position_shift"
    pl_config_dir: str = "pl_spectra"
    smooth_sigma_nm: float = DEFAULT_SMOOTH_SIGMA_NM
    fit_half_width_nm: float = DEFAULT_FIT_HALF_WIDTH_NM
    exclude_points: tuple[str, ...] = ()


def split_csv_values(value: str) -> tuple[str, ...]:
    """Split a comma-separated scalar config value into non-empty strings."""
    return tuple(item.strip() for item in value.split(",") if item.strip())


def read_peak_shift_config(path: Path) -> PeakShiftConfig:
    """Read peak-position-shift settings from YAML."""
    values = read_yaml_mapping(path)
    return PeakShiftConfig(
        out_subdir=string_value(values, "out_subdir", "peak_position_shift"),
        pl_config_dir=string_value(values, "pl_config_dir", "pl_spectra"),
        smooth_sigma_nm=float_value(values, "smooth_sigma_nm", DEFAULT_SMOOTH_SIGMA_NM),
        fit_half_width_nm=float_value(values, "fit_half_width_nm", DEFAULT_FIT_HALF_WIDTH_NM),
        exclude_points=split_csv_values(string_value(values, "exclude_points", "")),
    )


def is_non_ir_group(group: str) -> bool:
    """Return whether a configured group is a non-IR sample."""
    return group.lower().endswith("_noir")


def parse_exclude_point(text: str) -> tuple[str, int]:
    """Parse a group:distance_cm exclusion token."""
    group, separator, distance_text = text.partition(":")
    if not separator or not group.strip() or not distance_text.strip():
        raise ValueError(f"Expected exclusion in group:distance_cm form, got: {text}")
    return group.strip(), int(distance_text.strip())


@dataclass(frozen=True)
class PeakPoint:
    record: PLSpectrum
    peak_wavelength_nm: float
    reference_peak_wavelength_nm: float
    reference_distance_cm: int
    peak_shift_nm: float


def smoothed_intensity(intensity: np.ndarray, wavelength_nm: np.ndarray, sigma_nm: float) -> np.ndarray:
    """Return intensity smoothed by an approximately wavelength-scaled Gaussian."""
    if sigma_nm <= 0 or wavelength_nm.size < 3:
        return intensity
    spacing = float(np.nanmedian(np.diff(wavelength_nm)))
    if not np.isfinite(spacing) or spacing <= 0:
        return intensity
    sigma_points = sigma_nm / spacing
    if sigma_points <= 0:
        return intensity
    return gaussian_filter1d(intensity, sigma=sigma_points, mode="nearest")


def quadratic_peak_position(wavelength_nm: np.ndarray, intensity: np.ndarray) -> float | None:
    """Fit a local quadratic and return its vertex when it is inside the fit window."""
    if wavelength_nm.size < 3:
        return None
    centered = wavelength_nm - float(wavelength_nm.mean())
    a, b, _ = np.polyfit(centered, intensity, deg=2)
    if a >= 0:
        return None
    vertex = -b / (2.0 * a) + float(wavelength_nm.mean())
    if float(wavelength_nm.min()) <= vertex <= float(wavelength_nm.max()):
        return float(vertex)
    return None


def peak_position_nm(
    record: PLSpectrum,
    xlim_nm: tuple[float, float],
    *,
    smooth_sigma_nm: float,
    fit_half_width_nm: float,
) -> float:
    """Estimate the PL peak wavelength for one selected spectrum."""
    wavelength = record.wavelength_nm
    intensity = record.normalized_intensity
    finite = np.isfinite(wavelength) & np.isfinite(intensity)
    in_window = (wavelength >= xlim_nm[0]) & (wavelength <= xlim_nm[1])
    keep = finite & in_window
    if np.count_nonzero(keep) < 3:
        raise ValueError(f"Not enough points in peak window for {record.path}")

    x = wavelength[keep]
    y = intensity[keep]
    y_smooth = smoothed_intensity(y, x, smooth_sigma_nm)
    peak_index = int(np.nanargmax(y_smooth))
    peak_guess = float(x[peak_index])

    local = np.abs(x - peak_guess) <= fit_half_width_nm
    if np.count_nonzero(local) >= 3:
        refined = quadratic_peak_position(x[local], y_smooth[local])
        if refined is not None:
            return refined
    return peak_guess


def selected_peak_points(
    raw_dir: Path,
    out_dir: Path,
    config_dir: Path,
    *,
    smooth_sigma_nm: float,
    fit_half_width_nm: float,
    excluded_points: set[tuple[str, int]],
) -> list[PeakPoint]:
    """Load selected PL spectra and calculate relative peak-position shifts."""
    records = load_records(raw_dir, out_dir, include_background=True, include_reverse=True)
    records_by_path = {record.path.as_posix(): record for record in records}
    configs = read_plot_configs(config_dir)
    selected_by_group, _ = selected_records_from_configs(configs, records_by_path)
    xlim_by_group = {config.group: config.xlim_nm for config in configs}

    points: list[PeakPoint] = []
    for group in sorted(selected_by_group):
        group_records = [
            record
            for record in selected_by_group[group]
            if record.distance_cm is not None
            and (record.group, int(record.distance_cm)) not in excluded_points
        ]
        group_records = sorted(group_records, key=lambda record: (record.distance_cm or 0, record.position_suffix, record.path.as_posix()))
        if not group_records:
            continue

        peak_rows = [
            (
                record,
                peak_position_nm(
                    record,
                    xlim_by_group.get(group, (400.0, 720.0)),
                    smooth_sigma_nm=smooth_sigma_nm,
                    fit_half_width_nm=fit_half_width_nm,
                ),
            )
            for record in group_records
        ]
        reference_record, reference_peak = peak_rows[0]
        reference_distance = int(reference_record.distance_cm or 0)
        points.extend(
            PeakPoint(
                record=record,
                peak_wavelength_nm=peak,
                reference_peak_wavelength_nm=reference_peak,
                reference_distance_cm=reference_distance,
                peak_shift_nm=peak - reference_peak,
            )
            for record, peak in peak_rows
        )
    return points


def write_peak_table(points: list[PeakPoint], out_csv: Path, fiber_names: FiberNameMap) -> None:
    """Write calculated peak positions and shifts to CSV."""
    fields = [
        "group",
        "fiber_name",
        "relative_path",
        "position_token",
        "distance_cm",
        "peak_wavelength_nm",
        "reference_distance_cm",
        "reference_peak_wavelength_nm",
        "peak_shift_nm",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in points:
            record = point.record
            writer.writerow(
                {
                    "group": record.group,
                    "fiber_name": fiber_names.real_name(record.group),
                    "relative_path": record.path.as_posix(),
                    "position_token": record.position_token,
                    "distance_cm": record.distance_cm,
                    "peak_wavelength_nm": f"{point.peak_wavelength_nm:.6g}",
                    "reference_distance_cm": point.reference_distance_cm,
                    "reference_peak_wavelength_nm": f"{point.reference_peak_wavelength_nm:.6g}",
                    "peak_shift_nm": f"{point.peak_shift_nm:.6g}",
                }
            )


def write_method_note(
    out_md: Path,
    *,
    points: list[PeakPoint],
    raw_dir: Path,
    config_path: Path,
    pl_config_dir: Path,
    fiber_names_config: Path,
    smooth_sigma_nm: float,
    fit_half_width_nm: float,
    excluded_points: set[tuple[str, int]],
    out_csv: Path,
    out_png: Path,
    out_pdf: Path,
    fiber_names: FiberNameMap,
) -> None:
    """Write a short provenance note beside the generated peak-shift outputs."""
    groups = sorted({point.record.group for point in points}, key=lambda value: fiber_names.real_name(value))
    count_lines = [
        f"- {fiber_names.real_name(group)} (`{group}`): {sum(1 for point in points if point.record.group == group)}"
        for group in groups
    ]
    if excluded_points:
        exclusion_lines = [
            f"- {fiber_names.real_name(group)} (`{group}`), {distance_cm} cm"
            for group, distance_cm in sorted(excluded_points)
        ]
    else:
        exclusion_lines = ["- none"]

    lines = [
        "# Peak Position Shift",
        "",
        "This folder is generated by `python -m lhcb_fibers_analysis.peak_position_shift`.",
        "",
        "## Inputs",
        "",
        f"- Raw data folder: `{raw_dir}`",
        f"- Peak-shift config: `{config_path}`",
        f"- PL spectrum selection config folder: `{pl_config_dir}`",
        f"- Fiber display-name config: `{fiber_names_config}`",
        "",
        "## Method",
        "",
        "- Uses only PL spectra with `include: true` in the PL spectrum selection YAML files.",
        "- Uses each selected spectrum's normalized intensity array.",
        f"- Smooths intensity with a Gaussian sigma of {smooth_sigma_nm:g} nm before peak finding.",
        f"- Refines the peak wavelength with a local quadratic fit over +/- {fit_half_width_nm:g} nm.",
        "- Reports peak-position shift relative to each fiber's first remaining selected numeric position.",
        "",
        "## Explicit Exclusions",
        "",
        *exclusion_lines,
        "",
        "## Selected Point Counts",
        "",
        *count_lines,
        "",
        "## Outputs",
        "",
        f"- Table: `{out_csv.name}`",
        f"- Figure PNG: `{out_png.name}`",
        f"- Figure PDF: `{out_pdf.name}`",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def plot_peak_shifts(points: list[PeakPoint], out_png: Path, out_pdf: Path, fiber_names: FiberNameMap) -> None:
    """Plot peak-position shift versus scan position for all selected samples."""
    if not points:
        raise ValueError("No selected peak-position points to plot.")

    groups = sorted({point.record.group for point in points}, key=lambda group: fiber_names.real_name(group))
    fig, ax = plt.subplots(figsize=DOUBLE_COLUMN_WIDE, constrained_layout=True)
    ax.axhline(0.0, color=COLORS["light_gray"], linewidth=0.8, zorder=0)

    for index, group in enumerate(groups):
        group_points = sorted(
            [point for point in points if point.record.group == group],
            key=lambda point: (point.record.distance_cm or 0, point.record.position_suffix, point.record.path.as_posix()),
        )
        x = [float(point.record.distance_cm or 0) for point in group_points]
        y = [point.peak_shift_nm for point in group_points]
        color = LINE_COLORS[index % len(LINE_COLORS)]
        ax.plot(
            x,
            y,
            color=color,
            marker=MARKERS[index % len(MARKERS)],
            markersize=MARKER_SIZE,
            markerfacecolor=color if is_non_ir_group(group) else "white",
            markeredgecolor=color,
            markeredgewidth=0.8,
            linewidth=1.0,
            label=fiber_names.real_name(group),
        )

    ax.set_xlabel("Position (cm)")
    ax.set_ylabel(r"$\Delta\lambda_\mathrm{peak}$ (nm)")
    apply_axes_style(ax)
    ax.legend(loc="best", ncols=2, columnspacing=1.1, handlelength=1.8)
    save_figure(fig, out_png)
    save_figure(fig, out_pdf)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """Run peak-position shift analysis for selected PL spectra."""
    parser = argparse.ArgumentParser(description="Plot selected PL peak-position shift versus fiber position.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_PEAK_SHIFT_CONFIG)
    parser.add_argument("--config-dir", type=Path, default=None, help="Override the PL spectrum selection config folder.")
    parser.add_argument("--fiber-names-config", type=Path, default=DEFAULT_FIBER_NAMES_CONFIG)
    parser.add_argument("--out-subdir", type=Path, default=None)
    parser.add_argument("--smooth-sigma-nm", type=float, default=None)
    parser.add_argument("--fit-half-width-nm", type=float, default=None)
    parser.add_argument(
        "--exclude-point",
        action="append",
        default=None,
        help="Exclude one selected point from this analysis, in group:distance_cm form.",
    )
    args = parser.parse_args(argv)

    raw_dir = resolve_path(args.raw_dir)
    results_dir = resolve_path(args.results_dir)
    config_path = resolve_path(args.config)
    config = read_peak_shift_config(config_path)
    config_dir = (
        resolve_path(args.config_dir)
        if args.config_dir is not None
        else resolve_path(relative_config_dir(config_path, config.pl_config_dir))
    )
    fiber_names_config = resolve_path(args.fiber_names_config)
    fiber_names = read_fiber_name_map(fiber_names_config)
    out_subdir = args.out_subdir if args.out_subdir is not None else Path(config.out_subdir)
    out_dir = (results_dir / out_subdir).resolve()
    smooth_sigma_nm = args.smooth_sigma_nm if args.smooth_sigma_nm is not None else config.smooth_sigma_nm
    fit_half_width_nm = args.fit_half_width_nm if args.fit_half_width_nm is not None else config.fit_half_width_nm
    exclude_tokens = args.exclude_point if args.exclude_point is not None else list(config.exclude_points)
    excluded_points = {parse_exclude_point(token) for token in exclude_tokens}

    set_publication_style()
    points = selected_peak_points(
        raw_dir,
        out_dir,
        config_dir,
        smooth_sigma_nm=smooth_sigma_nm,
        fit_half_width_nm=fit_half_width_nm,
        excluded_points=excluded_points,
    )
    if not points:
        raise SystemExit("No selected spectra with numeric positions were found.")

    out_csv = out_dir / "peak_position_shift.csv"
    out_png = out_dir / "peak_position_shift_all_samples.png"
    out_pdf = out_dir / "peak_position_shift_all_samples.pdf"
    out_md = out_dir / "README.md"
    write_peak_table(points, out_csv, fiber_names)
    plot_peak_shifts(points, out_png, out_pdf, fiber_names)
    write_method_note(
        out_md,
        points=points,
        raw_dir=raw_dir,
        config_path=config_path,
        pl_config_dir=config_dir,
        fiber_names_config=fiber_names_config,
        smooth_sigma_nm=smooth_sigma_nm,
        fit_half_width_nm=fit_half_width_nm,
        excluded_points=excluded_points,
        out_csv=out_csv,
        out_png=out_png,
        out_pdf=out_pdf,
        fiber_names=fiber_names,
    )

    print(f"selected peak points: {len(points)}")
    print("groups:")
    for group in sorted({point.record.group for point in points}, key=lambda value: fiber_names.real_name(value)):
        count = sum(1 for point in points if point.record.group == group)
        print(f"  {fiber_names.real_name(group)} ({group}): {count}")
    if excluded_points:
        print("excluded points:")
        for group, distance_cm in sorted(excluded_points):
            print(f"  {fiber_names.real_name(group)} ({group}): {distance_cm} cm")
    print(f"table: {out_csv}")
    print(f"figure: {out_png}")
    print(f"pdf: {out_pdf}")
    print(f"method note: {out_md}")


if __name__ == "__main__":
    main()
