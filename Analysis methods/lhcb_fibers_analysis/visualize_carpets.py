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
from .paths import DEFAULT_CARPET_CONFIG, DEFAULT_RAW_DIR, DEFAULT_RESULTS_DIR, resolve_path
from .yaml_config import float_value, int_value, read_yaml_mapping, string_value


@dataclass(frozen=True)
class CarpetConfig:
    out_subdir: str = "carpets"
    top_edge_crop_rows: int = TOP_EDGE_CROP_ROWS
    background_percentile: float = 5.0
    signal_scale_percentile: float = 99.4
    asinh_scale_divisor: float = 8.0
    contact_sheet_max_columns: int = 4


def read_carpet_config(path: Path) -> CarpetConfig:
    values = read_yaml_mapping(path)
    return CarpetConfig(
        out_subdir=string_value(values, "out_subdir", "carpets"),
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
        sample = self.sample or "unknown"
        irradiation = self.irradiation or "unknown"
        return f"{sample}_{irradiation}"


def parse_filename(path: Path) -> dict[str, object]:
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
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _match_int(text: str, pattern: str) -> int | None:
    value = _match(text, pattern)
    return int(value) if value else None


def sort_key(record: CarpetRecord) -> tuple:
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


def scaled_core_image(data: np.ndarray, config: CarpetConfig) -> np.ndarray:
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
    if value is None:
        return "unknown"
    return f"{value:.4g} {unit}"


def label(record: CarpetRecord) -> str:
    center_nm = spectrograph_center_nm(record.carpet)
    full_time_ns = time_range_ns(record.carpet)
    bits = [
        record.path.stem,
        f"{record.carpet.data.shape[0]}x{record.carpet.data.shape[1]}",
        f"max {int(record.carpet.data.max())}",
        f"x { _format_measurement(center_nm, 'nm') } | y { _format_measurement(full_time_ns, 'ns') }",
    ]
    return "\n".join(bits)


def save_individual(record: CarpetRecord, config: CarpetConfig) -> None:
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

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.25], height_ratios=[4, 1.25])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_row = fig.add_subplot(gs[0, 1])
    ax_col = fig.add_subplot(gs[1, 0])
    ax_info = fig.add_subplot(gs[1, 1])

    imshow_kwargs = {"extent": extent} if extent is not None else {}
    im = ax_img.imshow(shown, origin="lower", aspect="auto", cmap="magma", **imshow_kwargs)
    ax_img.set_title(record.path.name)
    ax_img.set_xlabel("wavelength (nm)" if wavelengths_nm is not None else "x pixel")
    ax_img.set_ylabel("time (ns)" if times_ns is not None else "y pixel")
    fig.colorbar(im, ax=ax_img, fraction=0.035, pad=0.02, label="asinh contrast")

    row_axis = times_ns if times_ns is not None else np.arange(shown.shape[0])
    ax_row.plot(row_profile, row_axis, color="#0F766E", linewidth=1.0)
    ax_row.set_title("row mean, cropped+scaled")
    ax_row.set_xlabel("asinh contrast")
    ax_row.set_ylabel("time (ns)" if times_ns is not None else "y pixel")
    if visible_time_ns is not None:
        ax_row.set_ylim(0, visible_time_ns)
    ax_row.grid(alpha=0.25)

    col_axis = wavelengths_nm if wavelengths_nm is not None else np.arange(shown.shape[1])
    ax_col.plot(col_axis, col_profile, color="#1D4ED8", linewidth=1.0)
    ax_col.set_title("column mean, cropped+scaled")
    ax_col.set_xlabel("wavelength (nm)" if wavelengths_nm is not None else "x pixel")
    ax_col.set_ylabel("asinh contrast")
    if extent is not None:
        ax_col.set_xlim(extent[0], extent[1])
    ax_col.grid(alpha=0.25)

    info = [
        f"sample: {record.sample}/{record.irradiation}",
        f"position: {record.position_token or 'unknown'}",
        f"pulse: {record.pulse_energy_nj or ''} nJ",
        f"window: {record.time_window or ''}",
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
    ax_info.text(0, 1, "\n".join(info), va="top", ha="left", fontsize=10)

    record.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(record.out_png, dpi=160)
    plt.close(fig)


def save_contact_sheet(records: list[CarpetRecord], out_path: Path, title: str, config: CarpetConfig) -> None:
    if not records:
        return
    cols = min(config.contact_sheet_max_columns, max(1, math.ceil(math.sqrt(len(records)))))
    rows = math.ceil(len(records) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.2), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()

    for ax, record in zip(axes_arr, records):
        extent = image_extent(record.carpet, cropped=True, crop_rows=config.top_edge_crop_rows)
        imshow_kwargs = {"extent": extent} if extent is not None else {}
        ax.imshow(
            scaled_core_image(record.carpet.data, config),
            origin="lower",
            aspect="auto",
            cmap="magma",
            **imshow_kwargs,
        )
        ax.set_title(label(record), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_arr[len(records):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_inventory(records: list[CarpetRecord], out_csv: Path, out_dir: Path, config: CarpetConfig) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "relative_path",
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
                    "individual_png": Path(os.path.relpath(record.out_png, out_dir)).as_posix(),
                }
            )


def write_html(
    records: list[CarpetRecord],
    contact_sheets: list[Path],
    out_html: Path,
    config: CarpetConfig,
) -> None:
    index_dir = out_html.parent.resolve()

    def link_to(path: Path) -> str:
        return Path(os.path.relpath(path.resolve(), index_dir)).as_posix()

    rows = []
    for record in records:
        full_time_ns = time_range_ns(record.carpet)
        rows.append(
            "<tr>"
            f"<td>{record.group}</td>"
            f"<td>{record.position_token}</td>"
            f"<td>{record.time_window}</td>"
            f"<td>{_format_measurement(spectrograph_center_nm(record.carpet), 'nm')}</td>"
            f"<td>{_format_measurement(full_time_ns, 'ns')}</td>"
            f"<td>{record.path.as_posix()}</td>"
            f"<td><a href=\"{link_to(record.out_png)}\">quicklook</a></td>"
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
  <p>Loaded {len(records)} streak-camera carpets. Individual quicklooks crop the top {config.top_edge_crop_rows} rows, use asinh contrast, and compute row/column mean profiles from that same cropped/scaled view.</p>
  <p>X axes use the spectrograph wavelength calibration in nm. Y axes use the streak-camera time range in ns, scaled to the visible rows after the top-edge crop.</p>
  <h2>Contact Sheets</h2>
  <ul>{sheet_links}</ul>
  <h2>Individual Files</h2>
  <table>
    <thead><tr><th>group</th><th>position</th><th>window</th><th>x center</th><th>y range</th><th>source</th><th>view</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def load_records(raw_dir: Path, out_dir: Path) -> list[CarpetRecord]:
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
    parser = argparse.ArgumentParser(description="Load all Hamamatsu .img carpets and create quicklook visualizations.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CARPET_CONFIG)
    parser.add_argument("--out-subdir", type=Path, default=None)
    args = parser.parse_args(argv)

    raw_dir = resolve_path(args.raw_dir)
    results_dir = resolve_path(args.results_dir)
    config = read_carpet_config(resolve_path(args.config))
    out_subdir = args.out_subdir or Path(config.out_subdir)
    out_dir = (results_dir / out_subdir).resolve()
    records = load_records(raw_dir, out_dir)
    if not records:
        raise SystemExit("No .img carpets found.")

    for record in records:
        save_individual(record, config)

    contact_sheets: list[Path] = []
    all_sheet = out_dir / "contact_sheets" / "all_carpets.png"
    save_contact_sheet(
        records,
        all_sheet,
        f"All Hamamatsu Streak-Camera Carpets, Top {config.top_edge_crop_rows} Rows Cropped",
        config,
    )
    contact_sheets.append(all_sheet)

    for group in sorted({record.group for record in records}):
        subset = [record for record in records if record.group == group]
        sheet = out_dir / "contact_sheets" / f"{group}.png"
        save_contact_sheet(subset, sheet, f"{group}, Top {config.top_edge_crop_rows} Rows Cropped", config)
        contact_sheets.append(sheet)

    inventory_csv = out_dir / "carpet_inventory.csv"
    write_inventory(records, inventory_csv, out_dir, config)
    write_html(records, contact_sheets, out_dir / "index.html", config)

    groups = {}
    for record in records:
        groups.setdefault(record.group, 0)
        groups[record.group] += 1
    print(f"loaded carpets: {len(records)}")
    print("groups:")
    for group, count in sorted(groups.items()):
        print(f"  {group}: {count}")
    print(f"inventory: {inventory_csv}")
    print(f"config: {resolve_path(args.config)}")
    print(f"contact sheets: {out_dir / 'contact_sheets'}")
    print(f"individual quicklooks: {out_dir / 'individual'}")
    print(f"html index: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
