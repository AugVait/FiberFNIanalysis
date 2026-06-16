from __future__ import annotations

import json
from pathlib import Path

from .batch_carpet_wavelength_cuts import (
    ScanOutput,
    clean_band_edges,
    common_wavelength_bounds,
    process_scan,
    safe_scan_folder,
    sample_folder,
    scan_time_window,
    write_tsv,
)


INVENTORY_FIELDS = [
    "source_file",
    "sample",
    "output_folder",
    "status",
    "band_count",
    "decay_fit_count",
    "decay_skipped_count",
    "rise_fit_count",
    "rise_skipped_count",
    "note",
]


def csv_values(values: object) -> set[str]:
    """Parse a scalar or sequence of comma-separated workflow values."""
    if values is None or values == "":
        return set()
    if isinstance(values, str):
        values = [values]
    return {
        item.strip().lower()
        for value in values
        for item in str(value).split(",")
        if item.strip()
    }


def selected_img_relative_paths(manifest_path: Path, time_windows: object = None) -> list[Path]:
    """Return selected carpet .img paths from the raw-data manifest."""
    selected_windows = csv_values(time_windows)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    paths: list[Path] = []
    for entry in manifest.get("files", []):
        relative = Path(str(entry.get("path", "")))
        if relative.suffix.lower() != ".img":
            continue
        if selected_windows and scan_time_window(relative).lower() not in selected_windows:
            continue
        paths.append(relative)
    return sorted(paths, key=lambda path: path.as_posix())


def scan_id(relative_path: str | Path) -> str:
    """Return the stable Snakemake id for a raw carpet scan."""
    return safe_scan_folder(Path(relative_path))


def scan_output_row(output: ScanOutput, output_root: Path) -> dict[str, object]:
    """Convert one process_scan result into the top-level inventory row shape."""
    return {
        "source_file": output.relative_path.as_posix(),
        "sample": sample_folder(output.relative_path),
        "output_folder": output.output_dir.relative_to(output_root).as_posix(),
        "status": output.status,
        "band_count": output.band_count,
        "decay_fit_count": output.decay_fit_count,
        "decay_skipped_count": output.decay_skipped_count,
        "rise_fit_count": output.rise_fit_count,
        "rise_skipped_count": output.rise_skipped_count,
        "note": output.note,
    }


def write_json(path: Path, payload: object) -> None:
    """Write a small workflow JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_common_band_edges(
    *,
    raw_dir: Path,
    relative_paths: list[Path],
    interval_nm: float,
    wavelength_min_nm: float | None,
    wavelength_max_nm: float | None,
    out_json: Path,
) -> None:
    """Build shared clean wavelength bands for the selected carpet scans."""
    paths = [raw_dir / relative_path for relative_path in relative_paths]
    if not paths:
        raise ValueError("No selected .img scans were found in the raw-data manifest.")
    common_bounds = common_wavelength_bounds(paths)
    band_edges = clean_band_edges(
        common_bounds[0],
        common_bounds[1],
        interval_nm,
        explicit_min_nm=wavelength_min_nm,
        explicit_max_nm=wavelength_max_nm,
    )
    if not band_edges:
        raise ValueError("No clean wavelength bands fit inside the common wavelength range.")
    write_json(
        out_json,
        {
            "scan_count": len(paths),
            "common_bounds_nm": list(common_bounds),
            "interval_nm": interval_nm,
            "band_edges": [list(edge) for edge in band_edges],
        },
    )


def read_band_edges(path: Path) -> list[tuple[float, float]]:
    """Read shared clean wavelength bands from workflow JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [(float(start), float(end)) for start, end in payload["band_edges"]]


def run_wavelength_cut_scan(
    *,
    raw_dir: Path,
    results_dir: Path,
    relative_path: Path,
    band_edges_json: Path,
    out_subdir: str,
    interval_nm: float,
    range_mode: str,
    wavelength_min_nm: float | None,
    wavelength_max_nm: float | None,
    top_edge_crop_rows: int,
    smooth_sigma: float,
    fit_start_ns: float | None,
    fit_start_offset_ns: float,
    fit_end_ns: float | None,
    end_fraction: float,
    min_fit_points: int,
    min_peak_sigma: float,
    secondary_peak_height_fraction: float,
    secondary_peak_prominence_fraction: float,
    secondary_peak_noise_sigma: float,
    secondary_peak_min_separation_ns: float,
    secondary_peak_exclusion_before_ns: float,
    tau_min_ns: float,
    tau_max_ns: float,
    write_fit_curves: bool,
    write_slice_plots: bool,
    status_json: Path,
) -> None:
    """Run one Snakemake-managed wavelength-cut scan job."""
    output_root = (results_dir / out_subdir).resolve()
    band_edges = read_band_edges(band_edges_json) if range_mode == "common" else None
    output = process_scan(
        raw_dir / relative_path,
        raw_dir=raw_dir,
        output_root=output_root,
        band_edges=band_edges,
        interval_nm=interval_nm,
        range_mode=range_mode,
        wavelength_min_nm=wavelength_min_nm,
        wavelength_max_nm=wavelength_max_nm,
        top_edge_crop_rows=top_edge_crop_rows,
        smooth_sigma=smooth_sigma,
        fit_start_ns=fit_start_ns,
        fit_start_offset_ns=fit_start_offset_ns,
        fit_end_ns=fit_end_ns,
        end_fraction=end_fraction,
        min_fit_points=min_fit_points,
        min_peak_sigma=min_peak_sigma,
        secondary_peak_height_fraction=secondary_peak_height_fraction,
        secondary_peak_prominence_fraction=secondary_peak_prominence_fraction,
        secondary_peak_noise_sigma=secondary_peak_noise_sigma,
        secondary_peak_min_separation_ns=secondary_peak_min_separation_ns,
        secondary_peak_exclusion_before_ns=secondary_peak_exclusion_before_ns,
        tau_min_ns=tau_min_ns,
        tau_max_ns=tau_max_ns,
        write_fit_curves=write_fit_curves,
        write_slice_plots=write_slice_plots,
    )
    row = scan_output_row(output, output_root)
    write_json(status_json, row)
    if output.status != "ok":
        raise RuntimeError(f"{relative_path.as_posix()} failed: {output.note}")


def aggregate_wavelength_cut_inventory(
    *,
    scan_status_jsons: list[Path],
    inventory_path: Path,
) -> None:
    """Aggregate per-scan status JSON files into the legacy inventory table."""
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(scan_status_jsons, key=lambda item: item.as_posix())
    ]
    rows.sort(key=lambda row: str(row.get("source_file", "")))
    write_tsv(inventory_path, rows, INVENTORY_FIELDS)
