from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .carpet_wavelength_cuts import CutProfile, FitResult, exp_decay, fit_cut_profile
from .hamamatsu_streak import (
    TOP_EDGE_CROP_ROWS,
    crop_top_edge,
    load_img,
    spectrograph_center_nm,
    time_axis_ns,
    time_range_ns,
    wavelength_axis_nm,
    x_scale_nm_per_pixel,
)
from .paths import DEFAULT_RAW_DIR, DEFAULT_RESULTS_DIR, resolve_path


DEFAULT_OUT_SUBDIR = "carpet_wavelength_cuts_20nm_txt"
DEFAULT_INTERVAL_NM = 20.0


@dataclass(frozen=True)
class BandProfile:
    requested_min_nm: float
    requested_max_nm: float
    cut: CutProfile


@dataclass(frozen=True)
class ScanOutput:
    source_path: Path
    relative_path: Path
    output_dir: Path
    band_count: int
    fit_count: int
    skipped_count: int
    status: str
    note: str


def safe_scan_folder(relative_path: Path) -> str:
    """Return a unique filesystem-safe folder name for one raw scan."""
    stem = relative_path.with_suffix("").as_posix()
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem.replace("/", "__")).strip("_")


def format_nm(value: float) -> str:
    """Format a wavelength edge for stable column labels."""
    if float(value).is_integer():
        return str(int(round(value)))
    return f"{value:g}".replace(".", "p")


def text_value(value: object) -> str:
    """Format a scalar for tab-delimited text output."""
    if value is None:
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.10g}"
    return str(value)


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    """Write rows as a tab-delimited text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: text_value(row.get(field, "")) for field in fields})


def clean_band_edges(
    lower_bound_nm: float,
    upper_bound_nm: float,
    interval_nm: float,
    *,
    explicit_min_nm: float | None = None,
    explicit_max_nm: float | None = None,
) -> list[tuple[float, float]]:
    """Return clean wavelength intervals inside the requested calibrated range."""
    if interval_nm <= 0:
        raise ValueError("interval_nm must be positive")
    lower = explicit_min_nm if explicit_min_nm is not None else math.ceil(lower_bound_nm / interval_nm) * interval_nm
    upper = explicit_max_nm if explicit_max_nm is not None else math.floor(upper_bound_nm / interval_nm) * interval_nm
    if upper <= lower:
        return []

    band_count = int(math.floor((upper - lower) / interval_nm + 1.0e-9))
    bands: list[tuple[float, float]] = []
    for index in range(band_count):
        start = lower + index * interval_nm
        end = start + interval_nm
        if end <= upper + 1.0e-9:
            bands.append((float(start), float(end)))
    return bands


def common_wavelength_bounds(paths: list[Path]) -> tuple[float, float]:
    """Return the wavelength range fully covered by all calibrated scans."""
    starts: list[float] = []
    ends: list[float] = []
    for path in paths:
        carpet = load_img(path)
        wavelengths = wavelength_axis_nm(carpet)
        if wavelengths is None:
            continue
        starts.append(float(np.nanmin(wavelengths)))
        ends.append(float(np.nanmax(wavelengths)))
    if not starts or not ends:
        raise ValueError("No scans have usable wavelength calibration metadata.")
    return max(starts), min(ends)


def make_band_profile(
    data: np.ndarray,
    wavelengths_nm: np.ndarray,
    start_nm: float,
    end_nm: float,
) -> BandProfile | None:
    """Average one clean wavelength interval into a time profile."""
    mask = (wavelengths_nm >= start_nm) & (wavelengths_nm < end_nm)
    if not np.any(mask):
        return None
    selected = wavelengths_nm[mask]
    profile = data[:, mask].mean(axis=1)
    cut = CutProfile(
        center_nm=float((start_nm + end_nm) / 2.0),
        wavelength_min_nm=float(selected.min()),
        wavelength_max_nm=float(selected.max()),
        column_count=int(np.count_nonzero(mask)),
        profile_counts=profile.astype(float),
    )
    return BandProfile(requested_min_nm=start_nm, requested_max_nm=end_nm, cut=cut)


def result_row(band: BandProfile, result: FitResult) -> dict[str, object]:
    """Convert one band fit result to a text-output row."""
    return {
        "band_min_nm": band.requested_min_nm,
        "band_max_nm": band.requested_max_nm,
        "band_center_nm": result.center_nm,
        "actual_wavelength_min_nm": result.wavelength_min_nm,
        "actual_wavelength_max_nm": result.wavelength_max_nm,
        "column_count": result.column_count,
        "status": result.status,
        "reason": result.reason,
        "tau_ns": result.tau_ns,
        "tau_se_ns": result.tau_se_ns,
        "amplitude_counts": result.amplitude_counts,
        "baseline_counts": result.baseline_counts,
        "r_squared": result.r_squared,
        "rmse_counts": result.rmse_counts,
        "peak_time_ns": result.peak_time_ns,
        "peak_counts": result.peak_counts,
        "detected_peak_count": result.detected_peak_count,
        "detected_peak_times_ns": result.detected_peak_times_ns,
        "secondary_peak_count": result.secondary_peak_count,
        "secondary_peak_times_ns": result.secondary_peak_times_ns,
        "fit_end_rule": result.fit_end_rule,
        "fit_start_ns": result.fit_start_ns,
        "fit_end_ns": result.fit_end_ns,
        "fit_points": result.fit_points,
    }


SUMMARY_FIELDS = [
    "band_min_nm",
    "band_max_nm",
    "band_center_nm",
    "actual_wavelength_min_nm",
    "actual_wavelength_max_nm",
    "column_count",
    "status",
    "reason",
    "tau_ns",
    "tau_se_ns",
    "amplitude_counts",
    "baseline_counts",
    "r_squared",
    "rmse_counts",
    "peak_time_ns",
    "peak_counts",
    "detected_peak_count",
    "detected_peak_times_ns",
    "secondary_peak_count",
    "secondary_peak_times_ns",
    "fit_end_rule",
    "fit_start_ns",
    "fit_end_ns",
    "fit_points",
]


def write_metadata(
    path: Path,
    *,
    source_path: Path,
    relative_path: Path,
    carpet_shape: tuple[int, int],
    top_edge_crop_rows: int,
    interval_nm: float,
    range_mode: str,
    wavelength_bounds: tuple[float, float],
    band_edges: list[tuple[float, float]],
    band_count: int,
    fit_count: int,
) -> None:
    """Write scan and extraction settings as key-value text."""
    rows = [
        {"key": "source_file", "value": relative_path.as_posix()},
        {"key": "source_full_path", "value": str(source_path)},
        {"key": "rows", "value": carpet_shape[0]},
        {"key": "columns", "value": carpet_shape[1]},
        {"key": "top_edge_crop_rows", "value": top_edge_crop_rows},
        {"key": "wavelength_interval_nm", "value": interval_nm},
        {"key": "range_mode", "value": range_mode},
        {"key": "wavelength_bound_min_nm", "value": wavelength_bounds[0]},
        {"key": "wavelength_bound_max_nm", "value": wavelength_bounds[1]},
        {"key": "first_band_min_nm", "value": band_edges[0][0] if band_edges else ""},
        {"key": "last_band_max_nm", "value": band_edges[-1][1] if band_edges else ""},
        {"key": "band_count", "value": band_count},
        {"key": "fit_count", "value": fit_count},
    ]
    write_tsv(path, rows, ["key", "value"])


def write_profiles(path: Path, time_ns: np.ndarray, bands: list[BandProfile]) -> None:
    """Write one wide text table with one profile column per wavelength band."""
    fields = ["time_ns"] + [
        f"mean_counts_{format_nm(band.requested_min_nm)}_{format_nm(band.requested_max_nm)}nm"
        for band in bands
    ]
    rows: list[dict[str, object]] = []
    for idx, time_value in enumerate(time_ns):
        row: dict[str, object] = {"time_ns": float(time_value)}
        for band, field in zip(bands, fields[1:]):
            row[field] = float(band.cut.profile_counts[idx])
        rows.append(row)
    write_tsv(path, rows, fields)


def write_fit_curve_points(
    path: Path,
    time_ns: np.ndarray,
    bands: list[BandProfile],
    results: list[FitResult],
) -> None:
    """Write raw and fitted sample points used by successful decay fits."""
    rows: list[dict[str, object]] = []
    for band, result in zip(bands, results):
        if result.status != "fit":
            continue
        mask = (time_ns >= result.fit_start_ns) & (time_ns <= result.fit_end_ns)
        for time_value, raw_counts in zip(time_ns[mask], band.cut.profile_counts[mask]):
            fitted_counts = exp_decay(
                np.array([time_value], dtype=float),
                result.amplitude_counts,
                result.tau_ns,
                result.baseline_counts,
                result.fit_start_ns,
            )[0]
            rows.append(
                {
                    "band_min_nm": band.requested_min_nm,
                    "band_max_nm": band.requested_max_nm,
                    "time_ns": float(time_value),
                    "raw_counts": float(raw_counts),
                    "fit_counts": float(fitted_counts),
                    "residual_counts": float(raw_counts - fitted_counts),
                }
            )
    write_tsv(
        path,
        rows,
        ["band_min_nm", "band_max_nm", "time_ns", "raw_counts", "fit_counts", "residual_counts"],
    )


def process_scan(
    path: Path,
    *,
    raw_dir: Path,
    output_root: Path,
    band_edges: list[tuple[float, float]] | None,
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
) -> ScanOutput:
    """Extract and fit wavelength bands for one carpet scan."""
    relative_path = path.relative_to(raw_dir)
    out_dir = output_root / safe_scan_folder(relative_path)
    try:
        carpet = load_img(path)
        wavelengths = wavelength_axis_nm(carpet)
        times = time_axis_ns(carpet, cropped=True, crop_rows=top_edge_crop_rows)
        if wavelengths is None:
            raise ValueError("missing wavelength calibration")
        if times is None:
            raise ValueError("missing streak-time calibration")
        if band_edges is None:
            local_bounds = (float(np.nanmin(wavelengths)), float(np.nanmax(wavelengths)))
            scan_band_edges = clean_band_edges(
                local_bounds[0],
                local_bounds[1],
                interval_nm,
                explicit_min_nm=wavelength_min_nm,
                explicit_max_nm=wavelength_max_nm,
            )
        else:
            local_bounds = (float(np.nanmin(wavelengths)), float(np.nanmax(wavelengths)))
            scan_band_edges = band_edges

        data = crop_top_edge(carpet.data.astype(float), top_edge_crop_rows)
        bands = [
            band
            for start_nm, end_nm in scan_band_edges
            if (band := make_band_profile(data, wavelengths, start_nm, end_nm)) is not None
        ]
        if not bands:
            raise ValueError("no clean wavelength bands overlap this scan")

        results = [
            fit_cut_profile(
                band.cut,
                times,
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
            )
            for band in bands
        ]
        fit_count = sum(1 for result in results if result.status == "fit")
        skipped_count = len(results) - fit_count

        write_metadata(
            out_dir / "metadata.txt",
            source_path=path,
            relative_path=relative_path,
            carpet_shape=carpet.data.shape,
            top_edge_crop_rows=top_edge_crop_rows,
            interval_nm=interval_nm,
            range_mode=range_mode,
            wavelength_bounds=local_bounds,
            band_edges=scan_band_edges,
            band_count=len(bands),
            fit_count=fit_count,
        )
        write_tsv(out_dir / "fit_summary.txt", [result_row(band, result) for band, result in zip(bands, results)], SUMMARY_FIELDS)
        write_profiles(out_dir / "profiles_by_band.txt", times, bands)
        if write_fit_curves:
            write_fit_curve_points(out_dir / "fit_curve_points.txt", times, bands, results)

        return ScanOutput(
            source_path=path,
            relative_path=relative_path,
            output_dir=out_dir,
            band_count=len(bands),
            fit_count=fit_count,
            skipped_count=skipped_count,
            status="ok",
            note="",
        )
    except Exception as exc:  # noqa: BLE001
        out_dir.mkdir(parents=True, exist_ok=True)
        write_tsv(out_dir / "error.txt", [{"key": "error", "value": str(exc)}], ["key", "value"])
        return ScanOutput(
            source_path=path,
            relative_path=relative_path,
            output_dir=out_dir,
            band_count=0,
            fit_count=0,
            skipped_count=0,
            status="error",
            note=str(exc),
        )


def write_inventory(path: Path, outputs: list[ScanOutput], raw_dir: Path, output_root: Path) -> None:
    """Write the top-level scan inventory."""
    rows: list[dict[str, object]] = []
    for item in outputs:
        rows.append(
            {
                "source_file": item.relative_path.as_posix(),
                "output_folder": item.output_dir.relative_to(output_root).as_posix(),
                "status": item.status,
                "band_count": item.band_count,
                "fit_count": item.fit_count,
                "skipped_count": item.skipped_count,
                "note": item.note,
            }
        )
    write_tsv(
        path,
        rows,
        ["source_file", "output_folder", "status", "band_count", "fit_count", "skipped_count", "note"],
    )


def main(argv: list[str] | None = None) -> int:
    """Run text-only batch wavelength cuts for all Hamamatsu carpet scans."""
    parser = argparse.ArgumentParser(description="Extract clean 20 nm wavelength cuts from all Hamamatsu .img carpets.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-subdir", default=DEFAULT_OUT_SUBDIR)
    parser.add_argument("--interval-nm", type=float, default=DEFAULT_INTERVAL_NM)
    parser.add_argument("--range-mode", choices=["common", "per-scan"], default="common")
    parser.add_argument("--wavelength-min-nm", type=float, default=None)
    parser.add_argument("--wavelength-max-nm", type=float, default=None)
    parser.add_argument("--top-edge-crop-rows", type=int, default=TOP_EDGE_CROP_ROWS)
    parser.add_argument("--smooth-sigma", type=float, default=2.0)
    parser.add_argument("--fit-start-ns", type=float, default=None)
    parser.add_argument("--fit-start-offset-ns", type=float, default=0.05)
    parser.add_argument("--fit-end-ns", type=float, default=None)
    parser.add_argument("--end-fraction", type=float, default=0.05)
    parser.add_argument("--min-fit-points", type=int, default=20)
    parser.add_argument("--min-peak-sigma", type=float, default=5.0)
    parser.add_argument("--secondary-peak-height-fraction", type=float, default=0.20)
    parser.add_argument("--secondary-peak-prominence-fraction", type=float, default=0.12)
    parser.add_argument("--secondary-peak-noise-sigma", type=float, default=3.0)
    parser.add_argument("--secondary-peak-min-separation-ns", type=float, default=0.25)
    parser.add_argument("--secondary-peak-exclusion-before-ns", type=float, default=0.05)
    parser.add_argument("--tau-min-ns", type=float, default=0.03)
    parser.add_argument("--tau-max-ns", type=float, default=200.0)
    parser.add_argument("--no-fit-curves", action="store_true")
    args = parser.parse_args(argv)

    raw_dir = resolve_path(args.raw_dir)
    results_dir = resolve_path(args.results_dir)
    output_root = (results_dir / args.out_subdir).resolve()
    paths = sorted(path for path in raw_dir.rglob("*.img") if ".venv" not in path.parts)
    if not paths:
        raise SystemExit(f"No .img scans found under {raw_dir}")

    band_edges: list[tuple[float, float]] | None
    common_bounds: tuple[float, float] | None = None
    if args.range_mode == "common":
        common_bounds = common_wavelength_bounds(paths)
        band_edges = clean_band_edges(
            common_bounds[0],
            common_bounds[1],
            args.interval_nm,
            explicit_min_nm=args.wavelength_min_nm,
            explicit_max_nm=args.wavelength_max_nm,
        )
        if not band_edges:
            raise SystemExit("No clean wavelength bands fit inside the common wavelength range.")
    else:
        band_edges = None

    outputs: list[ScanOutput] = []
    total = len(paths)
    for index, path in enumerate(paths, start=1):
        output = process_scan(
            path,
            raw_dir=raw_dir,
            output_root=output_root,
            band_edges=band_edges,
            interval_nm=args.interval_nm,
            range_mode=args.range_mode,
            wavelength_min_nm=args.wavelength_min_nm,
            wavelength_max_nm=args.wavelength_max_nm,
            top_edge_crop_rows=args.top_edge_crop_rows,
            smooth_sigma=args.smooth_sigma,
            fit_start_ns=args.fit_start_ns,
            fit_start_offset_ns=args.fit_start_offset_ns,
            fit_end_ns=args.fit_end_ns,
            end_fraction=args.end_fraction,
            min_fit_points=args.min_fit_points,
            min_peak_sigma=args.min_peak_sigma,
            secondary_peak_height_fraction=args.secondary_peak_height_fraction,
            secondary_peak_prominence_fraction=args.secondary_peak_prominence_fraction,
            secondary_peak_noise_sigma=args.secondary_peak_noise_sigma,
            secondary_peak_min_separation_ns=args.secondary_peak_min_separation_ns,
            secondary_peak_exclusion_before_ns=args.secondary_peak_exclusion_before_ns,
            tau_min_ns=args.tau_min_ns,
            tau_max_ns=args.tau_max_ns,
            write_fit_curves=not args.no_fit_curves,
        )
        outputs.append(output)
        print(f"[{index}/{total}] {output.status}: {output.relative_path.as_posix()} -> {output.output_dir.name}")

    write_inventory(output_root / "inventory.txt", outputs, raw_dir, output_root)
    ok_count = sum(1 for item in outputs if item.status == "ok")
    fit_count = sum(item.fit_count for item in outputs)
    skipped_count = sum(item.skipped_count for item in outputs)
    print()
    print(f"scans processed: {ok_count} ok / {len(outputs) - ok_count} error / {len(outputs)} total")
    if common_bounds is not None and band_edges is not None:
        print(f"common wavelength coverage: {common_bounds[0]:.6g}-{common_bounds[1]:.6g} nm")
        print(f"clean bands: {band_edges[0][0]:g}-{band_edges[-1][1]:g} nm in {args.interval_nm:g} nm intervals")
    print(f"fits: {fit_count} fit / {skipped_count} skipped")
    print(f"output: {output_root}")
    print(f"inventory: {output_root / 'inventory.txt'}")
    return 0 if ok_count == len(outputs) else 1


if __name__ == "__main__":
    sys.exit(main())
