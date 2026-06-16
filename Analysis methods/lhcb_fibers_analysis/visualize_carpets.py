from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .fiber_names import FiberNameMap, read_fiber_name_map
from .hamamatsu_streak import (
    TOP_EDGE_CROP_ROWS,
    StreakCarpet,
    crop_top_edge,
    image_extent,
    load_img,
    spectrograph_center_nm,
    time_axis_ns,
    time_range_ns,
    visible_time_range_ns,
    wavelength_axis_nm,
    x_scale_nm_per_pixel,
)
from .paths import (
    DEFAULT_CARPET_CONFIG,
    DEFAULT_FIBER_NAMES_CONFIG,
    DEFAULT_RAW_DIR,
    DEFAULT_RESULTS_DIR,
    resolve_path,
)
from .plot_style import (
    CARPET_CMAP,
    COLORS,
    DIAGNOSTIC_PANEL,
    apply_axes_style,
    save_figure,
    set_publication_style,
    style_colorbar,
)
from .scan_config import (
    SectionConfig,
    configs_exist,
    normalize_config_path,
    read_section_configs,
    relative_config_dir,
    yaml_scalar,
)
from .yaml_config import float_value, int_value, read_yaml_mapping, string_value


@dataclass(frozen=True)
class CarpetConfig:
    out_subdir: str = "carpets"
    scan_config_dir: str = "carpets"
    top_edge_crop_rows: int = TOP_EDGE_CROP_ROWS
    background_percentile: float = 5.0
    signal_scale_percentile: float = 99.4
    asinh_scale_divisor: float = 8.0
    contact_sheet_max_columns: int = 4


def read_carpet_config(path: Path) -> CarpetConfig:
    """Read Hamamatsu carpet visualization settings from YAML."""
    values = read_yaml_mapping(path)
    return CarpetConfig(
        out_subdir=string_value(values, "out_subdir", "carpets"),
        scan_config_dir=string_value(values, "scan_config_dir", "carpets"),
        top_edge_crop_rows=max(0, int_value(values, "top_edge_crop_rows", TOP_EDGE_CROP_ROWS)),
        background_percentile=float_value(values, "background_percentile", 5.0),
        signal_scale_percentile=float_value(values, "signal_scale_percentile", 99.4),
        asinh_scale_divisor=float_value(values, "asinh_scale_divisor", 8.0),
        contact_sheet_max_columns=max(1, int_value(values, "contact_sheet_max_columns", 4)),
    )


@dataclass(frozen=True)
class CarpetRecord:
    path: Path
    carpet: StreakCarpet
    sample: str
    irradiation: str
    position_token: str
    distance_cm: int | None
    replicate: str
    pulse_energy_nj: int | None
    time_window: str
    measurement_date: str
    acquisition_date: str
    acquisition_time: str
    out_png: Path

    @property
    def group(self) -> str:
        """Return the config group name."""
        sample = self.sample or "unknown"
        irradiation = self.irradiation or "unknown"
        return f"{sample}_{irradiation}"


def parse_filename(path: Path) -> dict[str, object]:
    """Extract carpet measurement metadata from a raw-data filename."""
    stem = path.stem.lower()
    sample = _match(stem, r"(?:^|[^a-z0-9])((?:bcf\d+g?)|(?:scsf\d+))(?=[^a-z0-9]|$)")
    irradiation = _match(stem, r"_(noir|ir)_")
    position_token = _match(stem, r"_(endcm|\d+cm[a-z0-9]*)(?:_|$)")
    distance_cm = None
    replicate = ""
    if position_token and position_token != "endcm":
        distance_cm = int(re.match(r"(\d+)", position_token).group(1))
        replicate = _match(position_token, r"cm([a-z0-9]+)$")
    return {
        "sample": sample,
        "irradiation": irradiation,
        "position_token": position_token,
        "distance_cm": distance_cm,
        "replicate": replicate,
        "pulse_energy_nj": _match_int(stem, r"_(\d+)nj"),
        "time_window": _match(stem, r"_(\d+(?:ns|us|ms))(?:_|$)"),
    }


def _match(text: str, pattern: str) -> str:
    """Return the first regex capture group from text, if present."""
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _match_int(text: str, pattern: str) -> int | None:
    """Return the first regex capture group converted to an integer."""
    value = _match(text, pattern)
    return int(value) if value else None


def sort_key(record: CarpetRecord) -> tuple:
    """Return a stable sort key for measurement records."""
    window_rank = {"1ns": 1, "2ns": 2, "5ns": 5, "10ns": 10, "20ns": 20, "50ns": 50, "100ns": 100}
    position = -1 if record.position_token == "endcm" else record.distance_cm or 9999
    return (
        record.group,
        record.measurement_date,
        position,
        record.replicate or "0",
        window_rank.get(record.time_window, 9999),
        str(record.path),
    )


def parse_time_window_filters(values: list[str] | None) -> set[str] | None:
    """Parse repeated or comma-separated acquisition-window filters."""
    if not values:
        return None
    windows = {
        item.strip().lower()
        for value in values
        for item in value.split(",")
        if item.strip()
    }
    return windows or None


def filter_by_time_windows(records: list[CarpetRecord], time_windows: set[str] | None) -> list[CarpetRecord]:
    """Keep only carpet records whose acquisition time window is selected."""
    if time_windows is None:
        return records
    return [record for record in records if record.time_window.lower() in time_windows]


def scaled_core_image(data: np.ndarray, config: CarpetConfig) -> np.ndarray:
    """Return an asinh-scaled carpet image after background subtraction."""
    arr = data.astype(np.float32)
    core = crop_top_edge(arr, config.top_edge_crop_rows)
    background = np.percentile(core, config.background_percentile)
    signal = np.clip(core - background, 0, None)
    positive = signal[signal > 0]
    if positive.size:
        divisor = config.asinh_scale_divisor if config.asinh_scale_divisor > 0 else 8.0
        scale = np.percentile(positive, config.signal_scale_percentile) / divisor
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
    else:
        scale = 1.0
    return np.arcsinh(signal / scale)


def _format_measurement(value: float | None, unit: str) -> str:
    """Format a numeric measurement with units or unknown."""
    if value is None:
        return "unknown"
    return f"{value:.4g} {unit}"


def spaced_units(text: str) -> str:
    """Insert a space between numeric values and unit suffixes."""
    return re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)


def label(record: CarpetRecord, fiber_names: FiberNameMap | None = None) -> str:
    """Build a display label for a measurement record."""
    group = fiber_names.real_name(record.group) if fiber_names is not None else record.group
    center_nm = spectrograph_center_nm(record.carpet)
    full_time_ns = time_range_ns(record.carpet)
    bits = [
        f"{group}, {spaced_units(record.position_token) or 'unknown'}, {spaced_units(record.time_window) or 'unknown'}",
        f"max {int(record.carpet.data.max())} counts; x {_format_measurement(center_nm, 'nm')}; y {_format_measurement(full_time_ns, 'ns')}",
    ]
    return "\n".join(bits)


def config_label(record: CarpetRecord) -> str:
    """Build a metadata label for a carpet scan config entry."""
    parts = [record.group]
    if record.position_token:
        parts.append(record.position_token)
    if record.time_window:
        parts.append(record.time_window)
    if record.measurement_date:
        parts.append(record.measurement_date)
    return ", ".join(parts)


def scan_note(record: CarpetRecord) -> str:
    """Classify a carpet scan for selection-config notes."""
    if record.position_token == "endcm":
        return "endpoint_condition"
    if record.replicate:
        return "replicate_or_reverse_orientation"
    return "default_on"


def write_scan_configs(records: list[CarpetRecord], config_dir: Path, config: CarpetConfig) -> None:
    """Write per-group YAML configs for discovered carpet scans."""
    config_dir.mkdir(parents=True, exist_ok=True)
    for group in sorted({record.group for record in records}):
        group_records = [record for record in records if record.group == group]
        lines = [
            "# Edit include values, then rerun python -m lhcb_fibers_analysis.visualize_carpets.",
            "# include: true writes an individual quicklook and adds the scan to the contact sheets.",
            f"group: {yaml_scalar(group)}",
            f"title: {yaml_scalar(group)}",
            f"top_edge_crop_rows: {yaml_scalar(config.top_edge_crop_rows)}",
            "scans:",
        ]
        for record in group_records:
            wavelengths_nm = wavelength_axis_nm(record.carpet)
            full_time_ns = time_range_ns(record.carpet)
            visible_time_ns = visible_time_range_ns(
                record.carpet,
                cropped=True,
                crop_rows=config.top_edge_crop_rows,
            )
            lines.extend(
                [
                    f"  - include: {yaml_scalar(True)}",
                    f"    path: {yaml_scalar(record.path.as_posix())}",
                    f"    label: {yaml_scalar(config_label(record))}",
                    f"    sample: {yaml_scalar(record.sample)}",
                    f"    irradiation: {yaml_scalar(record.irradiation)}",
                    f"    position_token: {yaml_scalar(record.position_token)}",
                    f"    distance_cm: {yaml_scalar(record.distance_cm if record.distance_cm is not None else '')}",
                    f"    replicate: {yaml_scalar(record.replicate)}",
                    f"    pulse_energy_nj: {yaml_scalar(record.pulse_energy_nj if record.pulse_energy_nj is not None else '')}",
                    f"    time_window: {yaml_scalar(record.time_window)}",
                    f"    measurement_date: {yaml_scalar(record.measurement_date)}",
                    f"    acquisition_date: {yaml_scalar(record.acquisition_date)}",
                    f"    acquisition_time: {yaml_scalar(record.acquisition_time)}",
                    f"    rows: {yaml_scalar(record.carpet.data.shape[0])}",
                    f"    cols: {yaml_scalar(record.carpet.data.shape[1])}",
                    f"    spectrograph_center_nm: {yaml_scalar(spectrograph_center_nm(record.carpet) or '')}",
                    f"    x_scale_nm_per_pixel: {yaml_scalar(x_scale_nm_per_pixel(record.carpet) or '')}",
                    f"    x_start_nm: {yaml_scalar(float(wavelengths_nm[0]) if wavelengths_nm is not None else '')}",
                    f"    x_end_nm: {yaml_scalar(float(wavelengths_nm[-1]) if wavelengths_nm is not None else '')}",
                    f"    time_range_ns: {yaml_scalar(full_time_ns or '')}",
                    f"    visible_time_range_ns: {yaml_scalar(visible_time_ns or '')}",
                    f"    note: {yaml_scalar(scan_note(record))}",
                ]
            )
        (config_dir / f"{group}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def selected_records_from_configs(
    configs: list[SectionConfig],
    records_by_path: dict[str, CarpetRecord],
) -> tuple[list[CarpetRecord], set[Path], set[Path]]:
    """Resolve selected records from YAML config entries."""
    configured_paths: set[Path] = set()
    selected_paths: set[Path] = set()
    selected: list[CarpetRecord] = []

    for config in configs:
        for entry in config.entries:
            rel_path = normalize_config_path(str(entry.get("path", "")))
            record = records_by_path.get(rel_path)
            if record is None:
                print(f"warning: carpet config path not found, skipping: {rel_path}")
                continue
            configured_paths.add(record.path)
            if bool(entry.get("include", False)):
                selected.append(record)
                selected_paths.add(record.path)

    return sorted(selected, key=sort_key), selected_paths, configured_paths


def save_individual(record: CarpetRecord, config: CarpetConfig, fiber_names: FiberNameMap) -> None:
    """Save an individual carpet quicklook figure."""
    set_publication_style(base_font_size=8.0)
    raw_data = record.carpet.data
    core_data = crop_top_edge(raw_data, config.top_edge_crop_rows)
    shown = scaled_core_image(raw_data, config)
    row_profile = shown.mean(axis=1)
    col_profile = shown.mean(axis=0)
    p999 = np.percentile(core_data, 99.9)
    wavelengths_nm = wavelength_axis_nm(record.carpet)
    times_ns = time_axis_ns(record.carpet, cropped=True, crop_rows=config.top_edge_crop_rows)
    extent = image_extent(record.carpet, cropped=True, crop_rows=config.top_edge_crop_rows)
    visible_time_ns = visible_time_range_ns(
        record.carpet,
        cropped=True,
        crop_rows=config.top_edge_crop_rows,
    )

    fig = plt.figure(figsize=DIAGNOSTIC_PANEL, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.25], height_ratios=[4, 1.25])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_row = fig.add_subplot(gs[0, 1])
    ax_col = fig.add_subplot(gs[1, 0])
    ax_info = fig.add_subplot(gs[1, 1])

    imshow_kwargs = {"extent": extent} if extent is not None else {}
    im = ax_img.imshow(shown, origin="lower", aspect="auto", cmap=CARPET_CMAP, **imshow_kwargs)
    ax_img.set_title(
        f"{fiber_names.real_name(record.group)}, {spaced_units(record.position_token) or 'unknown'}, {spaced_units(record.time_window) or 'unknown'}",
        pad=5,
    )
    ax_img.set_xlabel("Wavelength (nm)" if wavelengths_nm is not None else "x pixel")
    ax_img.set_ylabel("Time (ns)" if times_ns is not None else "y pixel")
    colorbar = fig.colorbar(im, ax=ax_img, fraction=0.035, pad=0.02, label="PL signal (a.u.)")
    style_colorbar(colorbar)

    row_axis = times_ns if times_ns is not None else np.arange(shown.shape[0])
    ax_row.plot(row_profile, row_axis, color=COLORS["teal"], linewidth=0.9)
    ax_row.set_title("Row mean", pad=4)
    ax_row.set_xlabel("Normalized signal")
    ax_row.set_ylabel("Time (ns)" if times_ns is not None else "y pixel")
    if visible_time_ns is not None:
        ax_row.set_ylim(0, visible_time_ns)
    apply_axes_style(ax_row, grid=True)

    col_axis = wavelengths_nm if wavelengths_nm is not None else np.arange(shown.shape[1])
    ax_col.plot(col_axis, col_profile, color=COLORS["blue"], linewidth=0.9)
    ax_col.set_title("Column mean", pad=4)
    ax_col.set_xlabel("Wavelength (nm)" if wavelengths_nm is not None else "x pixel")
    ax_col.set_ylabel("Normalized signal")
    if extent is not None:
        ax_col.set_xlim(extent[0], extent[1])
    apply_axes_style(ax_col, grid=True)

    info = [
        f"fiber: {fiber_names.real_name(record.group)}",
        f"experimental: {record.group}",
        f"position: {spaced_units(record.position_token) or 'unknown'}",
        f"pulse: {record.pulse_energy_nj or ''} nJ",
        f"window: {spaced_units(record.time_window)}",
        f"display: top {config.top_edge_crop_rows} rows cropped",
        f"x scale: {_format_measurement(spectrograph_center_nm(record.carpet), 'nm')} center, {_format_measurement(x_scale_nm_per_pixel(record.carpet), 'nm/pixel')}",
        f"y scale: {_format_measurement(visible_time_ns, 'ns')} visible of {_format_measurement(time_range_ns(record.carpet), 'ns')}",
        f"acquired: {record.acquisition_date} {record.acquisition_time}".strip(),
        f"core min: {int(core_data.min())}",
        f"core p99.9: {p999:.1f}",
        f"core max: {int(core_data.max())}",
        f"raw max: {int(raw_data.max())}",
    ]
    ax_info.axis("off")
    ax_info.text(0, 1, "\n".join(info), va="top", ha="left", fontsize=7.1, linespacing=1.18)

    save_figure(fig, record.out_png, dpi=260)
    plt.close(fig)


def save_contact_sheet(
    records: list[CarpetRecord],
    out_path: Path,
    title: str,
    config: CarpetConfig,
    fiber_names: FiberNameMap,
) -> None:
    """Save a contact sheet for selected carpet records."""
    if not records:
        return
    set_publication_style(base_font_size=7.2)
    cols = min(config.contact_sheet_max_columns, max(1, math.ceil(math.sqrt(len(records)))))
    rows = math.ceil(len(records) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.35, rows * 2.55), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()

    for ax, record in zip(axes_arr, records):
        extent = image_extent(record.carpet, cropped=True, crop_rows=config.top_edge_crop_rows)
        imshow_kwargs = {"extent": extent} if extent is not None else {}
        ax.imshow(
            scaled_core_image(record.carpet.data, config),
            origin="lower",
            aspect="auto",
            cmap=CARPET_CMAP,
            **imshow_kwargs,
        )
        ax.set_title(label(record, fiber_names), fontsize=6.5, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_arr[len(records):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=9.0, fontweight="normal")
    save_figure(fig, out_path, dpi=220)
    plt.close(fig)


def write_inventory(
    records: list[CarpetRecord],
    selected_record_paths: set[Path],
    out_csv: Path,
    out_dir: Path,
    config: CarpetConfig,
    fiber_names: FiberNameMap,
) -> None:
    """Write a CSV inventory of configured analysis records."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "relative_path",
        "fiber_name",
        "sample",
        "irradiation",
        "position_token",
        "distance_cm",
        "replicate",
        "pulse_energy_nj",
        "time_window",
        "measurement_date",
        "acquisition_date",
        "acquisition_time",
        "rows",
        "cols",
        "dtype",
        "top_edge_crop_rows",
        "core_rows",
        "core_cols",
        "x_unit",
        "spectrograph_center_nm",
        "x_scale_nm_per_pixel",
        "x_start_nm",
        "x_end_nm",
        "y_unit",
        "time_range_ns",
        "visible_time_range_ns",
        "time_ns_per_pixel",
        "core_min",
        "core_p50",
        "core_p99",
        "core_p999",
        "core_max",
        "core_mean_counts",
        "scaled_mean",
        "raw_min",
        "raw_max",
        "selected",
        "individual_png",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            data = record.carpet.data
            core_data = crop_top_edge(data, config.top_edge_crop_rows)
            scaled = scaled_core_image(data, config)
            wavelengths_nm = wavelength_axis_nm(record.carpet)
            full_time_ns = time_range_ns(record.carpet)
            visible_time_ns = visible_time_range_ns(
                record.carpet,
                cropped=True,
                crop_rows=config.top_edge_crop_rows,
            )
            time_ns_per_pixel = full_time_ns / data.shape[0] if full_time_ns is not None else None
            writer.writerow(
                {
                    "relative_path": record.path.as_posix(),
                    "fiber_name": fiber_names.real_name(record.group),
                    "sample": record.sample,
                    "irradiation": record.irradiation,
                    "position_token": record.position_token,
                    "distance_cm": record.distance_cm if record.distance_cm is not None else "",
                    "replicate": record.replicate,
                    "pulse_energy_nj": record.pulse_energy_nj if record.pulse_energy_nj is not None else "",
                    "time_window": record.time_window,
                    "measurement_date": record.measurement_date,
                    "acquisition_date": record.acquisition_date,
                    "acquisition_time": record.acquisition_time,
                    "rows": data.shape[0],
                    "cols": data.shape[1],
                    "dtype": str(data.dtype),
                    "top_edge_crop_rows": config.top_edge_crop_rows,
                    "core_rows": core_data.shape[0],
                    "core_cols": core_data.shape[1],
                    "x_unit": "nm" if wavelengths_nm is not None else "",
                    "spectrograph_center_nm": spectrograph_center_nm(record.carpet)
                    if wavelengths_nm is not None
                    else "",
                    "x_scale_nm_per_pixel": x_scale_nm_per_pixel(record.carpet) if wavelengths_nm is not None else "",
                    "x_start_nm": float(wavelengths_nm[0]) if wavelengths_nm is not None else "",
                    "x_end_nm": float(wavelengths_nm[-1]) if wavelengths_nm is not None else "",
                    "y_unit": "ns" if full_time_ns is not None else "",
                    "time_range_ns": full_time_ns if full_time_ns is not None else "",
                    "visible_time_range_ns": visible_time_ns if visible_time_ns is not None else "",
                    "time_ns_per_pixel": time_ns_per_pixel if time_ns_per_pixel is not None else "",
                    "core_min": int(core_data.min()),
                    "core_p50": float(np.percentile(core_data, 50)),
                    "core_p99": float(np.percentile(core_data, 99)),
                    "core_p999": float(np.percentile(core_data, 99.9)),
                    "core_max": int(core_data.max()),
                    "core_mean_counts": float(core_data.mean()),
                    "scaled_mean": float(scaled.mean()),
                    "raw_min": int(data.min()),
                    "raw_max": int(data.max()),
                    "selected": "yes" if record.path in selected_record_paths else "no",
                    "individual_png": Path(os.path.relpath(record.out_png, out_dir)).as_posix()
                    if record.path in selected_record_paths
                    else "",
                }
            )


def write_html(
    records: list[CarpetRecord],
    selected_records: list[CarpetRecord],
    contact_sheets: list[Path],
    out_html: Path,
    config: CarpetConfig,
    fiber_names: FiberNameMap,
) -> None:
    """Write an HTML index for generated analysis outputs."""
    index_dir = out_html.parent.resolve()

    def link_to(path: Path) -> str:
        """Return a relative link from the HTML index to an output path."""
        return Path(os.path.relpath(path.resolve(), index_dir)).as_posix()

    rows = []
    selected_record_paths = {record.path for record in selected_records}
    for record in records:
        full_time_ns = time_range_ns(record.carpet)
        quicklook_link = (
            f"<a href=\"{link_to(record.out_png)}\">quicklook</a>"
            if record.path in selected_record_paths
            else ""
        )
        rows.append(
            "<tr>"
            f"<td>{fiber_names.real_name(record.group)}</td>"
            f"<td>{record.group}</td>"
            f"<td>{record.position_token}</td>"
            f"<td>{record.time_window}</td>"
            f"<td>{_format_measurement(spectrograph_center_nm(record.carpet), 'nm')}</td>"
            f"<td>{_format_measurement(full_time_ns, 'ns')}</td>"
            f"<td>{'yes' if record.path in selected_record_paths else 'no'}</td>"
            f"<td>{record.path.as_posix()}</td>"
            f"<td>{quicklook_link}</td>"
            "</tr>"
        )
    sheet_links = "\n".join(
        f"<li><a href=\"{link_to(sheet)}\">{sheet.name}</a></li>" for sheet in contact_sheets
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Hamamatsu Carpet Visualizations</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #172033; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #d7dde8; padding: 6px 8px; text-align: left; }}
    th {{ background: #edf2f7; }}
  </style>
</head>
<body>
  <h1>Hamamatsu Carpet Visualizations</h1>
  <p>Loaded {len(records)} configured streak-camera carpets; {len(selected_records)} are selected for figure output. Individual quicklooks crop the top {config.top_edge_crop_rows} rows, use normalized signal, and compute row/column mean profiles from that same cropped/scaled view.</p>
  <p>X axes use the spectrograph wavelength calibration in nm. Y axes use the streak-camera time range in ns, scaled to the visible rows after the top-edge crop.</p>
  <h2>Contact Sheets</h2>
  <ul>{sheet_links}</ul>
  <h2>Individual Files</h2>
  <table>
    <thead><tr><th>fiber</th><th>experimental</th><th>position</th><th>window</th><th>x center</th><th>y range</th><th>selected</th><th>source</th><th>view</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def load_records(raw_dir: Path, out_dir: Path) -> list[CarpetRecord]:
    """Load discoverable PL spectrum records from raw data."""
    records: list[CarpetRecord] = []
    for path in sorted(raw_dir.rglob("*.img")):
        if ".venv" in path.parts or "analysis" in path.parts:
            continue
        carpet = load_img(path)
        parsed = parse_filename(path)
        rel_path = path.relative_to(raw_dir)
        parts = rel_path.parts
        measurement_date = ""
        for part in parts:
            if re.fullmatch(r"20\d{2} \d{2} \d{2}", part):
                measurement_date = part.replace(" ", "-", 1).replace(" ", "-", 1)
                break
        app = carpet.metadata.get("Application", {})
        group = f"{parsed['sample'] or 'unknown'}_{parsed['irradiation'] or 'unknown'}"
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)
        out_png = out_dir / "individual" / group / f"{safe_stem}.png"
        records.append(
            CarpetRecord(
                path=rel_path,
                carpet=carpet,
                sample=str(parsed["sample"]),
                irradiation=str(parsed["irradiation"]),
                position_token=str(parsed["position_token"]),
                distance_cm=parsed["distance_cm"],
                replicate=str(parsed["replicate"]),
                pulse_energy_nj=parsed["pulse_energy_nj"],
                time_window=str(parsed["time_window"]),
                measurement_date=measurement_date,
                acquisition_date=app.get("Date", ""),
                acquisition_time=app.get("Time", ""),
                out_png=out_png,
            )
        )
    return sorted(records, key=sort_key)


def main(argv: list[str] | None = None) -> None:
    """Run the carpet visualization command-line interface."""
    parser = argparse.ArgumentParser(description="Load all Hamamatsu .img carpets and create quicklook visualizations.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CARPET_CONFIG)
    parser.add_argument("--scan-config-dir", type=Path, default=None)
    parser.add_argument("--fiber-names-config", type=Path, default=DEFAULT_FIBER_NAMES_CONFIG)
    parser.add_argument("--out-subdir", type=Path, default=None)
    parser.add_argument(
        "--time-window",
        "--time-windows",
        dest="time_windows",
        action="append",
        default=None,
        help="Keep only selected carpet acquisition windows. Repeat or use commas, e.g. --time-window 10ns or --time-windows 2ns,10ns.",
    )
    parser.add_argument("--refresh-configs", action="store_true", help="Rewrite per-sample carpet scan configs.")
    args = parser.parse_args(argv)

    raw_dir = resolve_path(args.raw_dir)
    results_dir = resolve_path(args.results_dir)
    config_path = resolve_path(args.config)
    fiber_names_config = resolve_path(args.fiber_names_config)
    fiber_names = read_fiber_name_map(fiber_names_config)
    config = read_carpet_config(config_path)
    scan_config_dir = (
        resolve_path(args.scan_config_dir)
        if args.scan_config_dir is not None
        else relative_config_dir(config_path, config.scan_config_dir)
    )
    out_subdir = args.out_subdir or Path(config.out_subdir)
    out_dir = (results_dir / out_subdir).resolve()
    time_window_filters = parse_time_window_filters(args.time_windows)
    discovered_records = load_records(raw_dir, out_dir)
    if not discovered_records:
        raise SystemExit("No .img carpets found.")

    if args.refresh_configs or not configs_exist(scan_config_dir):
        write_scan_configs(filter_by_time_windows(discovered_records, time_window_filters), scan_config_dir, config)

    scan_configs = read_section_configs(scan_config_dir, "scans")
    if not scan_configs:
        raise SystemExit(f"No carpet scan YAML configs found in {scan_config_dir}")

    records_by_path = {record.path.as_posix(): record for record in discovered_records}
    selected_records, selected_record_paths, configured_record_paths = selected_records_from_configs(
        scan_configs, records_by_path
    )
    records = filter_by_time_windows(
        [record for record in discovered_records if record.path in configured_record_paths],
        time_window_filters,
    )
    selected_records = filter_by_time_windows(selected_records, time_window_filters)
    selected_record_paths = {record.path for record in selected_records}
    if not selected_records:
        filter_note = (
            ""
            if time_window_filters is None
            else f" for time window(s): {', '.join(sorted(time_window_filters))}"
        )
        raise SystemExit(f"No carpet scans are selected in {scan_config_dir}{filter_note}")

    for record in selected_records:
        save_individual(record, config, fiber_names)

    contact_sheets: list[Path] = []
    all_sheet = out_dir / "contact_sheets" / "all_carpets.png"
    save_contact_sheet(
        selected_records,
        all_sheet,
        f"All Hamamatsu Streak-Camera Carpets, Top {config.top_edge_crop_rows} Rows Cropped",
        config,
        fiber_names,
    )
    contact_sheets.append(all_sheet)

    for group in sorted({record.group for record in selected_records}):
        subset = [record for record in selected_records if record.group == group]
        sheet = out_dir / "contact_sheets" / f"{group}.png"
        title = f"{fiber_names.real_name(group)}, Top {config.top_edge_crop_rows} Rows Cropped"
        save_contact_sheet(subset, sheet, title, config, fiber_names)
        contact_sheets.append(sheet)

    inventory_csv = out_dir / "carpet_inventory.csv"
    write_inventory(records, selected_record_paths, inventory_csv, out_dir, config, fiber_names)
    write_html(records, selected_records, contact_sheets, out_dir / "index.html", config, fiber_names)

    groups = {}
    for record in records:
        groups.setdefault(record.group, 0)
        groups[record.group] += 1
    selected_groups = {}
    for record in selected_records:
        selected_groups.setdefault(record.group, 0)
        selected_groups[record.group] += 1
    print(f"loaded carpets: {len(records)} configured")
    print(f"selected carpets: {len(selected_records)}")
    if time_window_filters is not None:
        print(f"time-window filter: {', '.join(sorted(time_window_filters))}")
    print("groups:")
    for group, count in sorted(groups.items()):
        print(f"  {fiber_names.real_name(group)} ({group}): {selected_groups.get(group, 0)} selected / {count} listed")
    print(f"inventory: {inventory_csv}")
    print(f"config: {config_path}")
    print(f"fiber names: {fiber_names_config}")
    print(f"scan configs: {scan_config_dir}")
    print(f"contact sheets: {out_dir / 'contact_sheets'}")
    print(f"individual quicklooks: {out_dir / 'individual'}")
    print(f"html index: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
