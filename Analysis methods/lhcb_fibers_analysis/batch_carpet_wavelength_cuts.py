from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

from .carpet_wavelength_cuts import (
    CutProfile,
    FitResult,
    exp_decay,
    first_sustained_true,
    fit_cut_profile,
    robust_noise,
)
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
from .plot_style import DOUBLE_COLUMN_WIDE, apply_axes_style, save_figure, set_publication_style


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
    decay_fit_count: int
    decay_skipped_count: int
    rise_fit_count: int
    rise_skipped_count: int
    status: str
    note: str


@dataclass(frozen=True)
class RiseFitResult:
    center_nm: float
    wavelength_min_nm: float
    wavelength_max_nm: float
    column_count: int
    status: str
    reason: str
    rise_tau_ns: float
    rise_tau_se_ns: float
    rise_time_10_90_ns: float
    observed_rise_time_10_90_ns: float
    amplitude_counts: float
    baseline_counts: float
    midpoint_time_ns: float
    r_squared: float
    rmse_counts: float
    peak_time_ns: float
    peak_counts: float
    threshold_10_time_ns: float
    threshold_90_time_ns: float
    fit_start_ns: float
    fit_end_ns: float
    fit_points: int


def safe_scan_folder(relative_path: Path) -> str:
    """Return a unique filesystem-safe folder name for one raw scan."""
    stem = relative_path.with_suffix("").as_posix()
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem.replace("/", "__")).strip("_")


def sample_folder(relative_path: Path) -> str:
    """Return the sample/irradiation folder for one raw scan."""
    stem = relative_path.stem.lower()
    sample_match = re.search(r"(?:^|[^a-z0-9])((?:bcf\d+g?)|(?:scsf\d+))(?=[^a-z0-9]|$)", stem)
    irradiation_match = re.search(r"_(noir|ir)_", stem)
    if sample_match and irradiation_match:
        return f"{sample_match.group(1)}_{irradiation_match.group(1)}"
    if sample_match:
        return sample_match.group(1)
    return "unknown_sample"


def scan_time_window(relative_path: Path) -> str:
    """Return the acquisition time-window token from one scan path."""
    match = re.search(r"_(\d+(?:ns|us|ms))(?:_|$)", relative_path.stem.lower())
    return match.group(1) if match else ""


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


def sigmoid_rise(time_ns: np.ndarray, amplitude: float, k_ns: float, baseline: float, midpoint_ns: float) -> np.ndarray:
    """Evaluate a sigmoid rising edge."""
    exponent = np.clip(-(time_ns - midpoint_ns) / k_ns, -700, 700)
    return baseline + amplitude / (1.0 + np.exp(exponent))


def crossing_time(time_ns: np.ndarray, signal: np.ndarray, threshold: float, end_idx: int) -> float:
    """Return a linearly interpolated first rising threshold crossing before a peak."""
    for idx in range(1, end_idx + 1):
        previous = signal[idx - 1]
        current = signal[idx]
        if previous < threshold <= current:
            span = current - previous
            if span == 0:
                return float(time_ns[idx])
            fraction = (threshold - previous) / span
            return float(time_ns[idx - 1] + fraction * (time_ns[idx] - time_ns[idx - 1]))
    return float("nan")


def skipped_rise_result(
    cut: CutProfile,
    reason: str,
    peak_time_ns: float = float("nan"),
    peak_counts: float = float("nan"),
    threshold_10_time_ns: float = float("nan"),
    threshold_90_time_ns: float = float("nan"),
) -> RiseFitResult:
    """Create a rise-fit result describing why a band was skipped."""
    observed = (
        threshold_90_time_ns - threshold_10_time_ns
        if math.isfinite(threshold_10_time_ns) and math.isfinite(threshold_90_time_ns)
        else float("nan")
    )
    return RiseFitResult(
        center_nm=cut.center_nm,
        wavelength_min_nm=cut.wavelength_min_nm,
        wavelength_max_nm=cut.wavelength_max_nm,
        column_count=cut.column_count,
        status="skipped",
        reason=reason,
        rise_tau_ns=float("nan"),
        rise_tau_se_ns=float("nan"),
        rise_time_10_90_ns=float("nan"),
        observed_rise_time_10_90_ns=observed,
        amplitude_counts=float("nan"),
        baseline_counts=float("nan"),
        midpoint_time_ns=float("nan"),
        r_squared=float("nan"),
        rmse_counts=float("nan"),
        peak_time_ns=peak_time_ns,
        peak_counts=peak_counts,
        threshold_10_time_ns=threshold_10_time_ns,
        threshold_90_time_ns=threshold_90_time_ns,
        fit_start_ns=float("nan"),
        fit_end_ns=float("nan"),
        fit_points=0,
    )


def fit_rise_profile(
    cut: CutProfile,
    time_ns: np.ndarray,
    *,
    smooth_sigma: float,
    min_fit_points: int,
    min_peak_sigma: float,
    tau_min_ns: float,
    tau_max_ns: float,
    low_fraction: float = 0.10,
    high_fraction: float = 0.90,
) -> RiseFitResult:
    """Fit a sigmoid rise to the 10-90% rising edge before the dominant peak."""
    y = cut.profile_counts
    if y.size < min_fit_points:
        return skipped_rise_result(cut, "too_few_points")

    smooth = gaussian_filter1d(y, sigma=smooth_sigma) if smooth_sigma > 0 else y
    baseline_seed = float(np.percentile(y, 5))
    low_values = y[y <= np.percentile(y, 25)]
    noise = max(robust_noise(low_values), math.sqrt(max(baseline_seed, 0.0) + 1.0))
    signal = smooth - baseline_seed
    peak_idx = int(np.argmax(signal))
    peak_height = float(signal[peak_idx])
    peak_time = float(time_ns[peak_idx])
    peak_counts = float(y[peak_idx])

    if peak_height <= max(min_peak_sigma * noise, 1.0):
        return skipped_rise_result(cut, "low_signal", peak_time, peak_counts)
    if peak_idx < min_fit_points:
        return skipped_rise_result(cut, "too_few_pre_peak_points", peak_time, peak_counts)

    low_threshold = low_fraction * peak_height
    high_threshold = high_fraction * peak_height
    pre_peak = signal[: peak_idx + 1]
    low_cross_idx = first_sustained_true(pre_peak >= low_threshold, run_length=5)
    high_cross_idx = first_sustained_true(pre_peak >= high_threshold, run_length=3)
    if low_cross_idx is None:
        return skipped_rise_result(cut, "no_10_percent_crossing", peak_time, peak_counts)
    if high_cross_idx is None:
        return skipped_rise_result(cut, "no_90_percent_crossing", peak_time, peak_counts)
    if high_cross_idx <= low_cross_idx:
        return skipped_rise_result(cut, "invalid_threshold_order", peak_time, peak_counts)

    threshold_10_time = crossing_time(time_ns, signal, low_threshold, peak_idx)
    threshold_90_time = crossing_time(time_ns, signal, high_threshold, peak_idx)
    observed_rise_time = (
        threshold_90_time - threshold_10_time
        if math.isfinite(threshold_10_time) and math.isfinite(threshold_90_time)
        else float("nan")
    )

    margin = max(3, min_fit_points // 4)
    fit_start_idx = max(0, low_cross_idx - margin)
    fit_end_idx = min(peak_idx + 1, high_cross_idx + margin + 1)
    if fit_end_idx - fit_start_idx < min_fit_points:
        missing = min_fit_points - (fit_end_idx - fit_start_idx)
        fit_start_idx = max(0, fit_start_idx - math.ceil(missing / 2))
        fit_end_idx = min(peak_idx + 1, fit_end_idx + math.floor(missing / 2) + 1)
    if fit_end_idx - fit_start_idx < min_fit_points:
        return skipped_rise_result(
            cut,
            "too_few_rise_fit_points",
            peak_time,
            peak_counts,
            threshold_10_time,
            threshold_90_time,
        )

    x_fit = time_ns[fit_start_idx:fit_end_idx]
    y_fit = y[fit_start_idx:fit_end_idx]
    dt = float(np.median(np.diff(time_ns))) if time_ns.size > 1 else 1.0
    k_lower = max(abs(dt) / 100.0, 1.0e-6)
    baseline0 = max(0.0, float(np.percentile(y_fit[: max(3, min(8, len(y_fit)))], 20)))
    amplitude0 = max(float(y_fit[-1] - baseline0), 1.0)
    k0 = observed_rise_time / (2.0 * math.log(9.0)) if math.isfinite(observed_rise_time) and observed_rise_time > 0 else max(dt, tau_min_ns)
    k0 = min(max(k0, k_lower), tau_max_ns)
    midpoint0 = (
        float((threshold_10_time + threshold_90_time) / 2.0)
        if math.isfinite(threshold_10_time) and math.isfinite(threshold_90_time)
        else float(x_fit[len(x_fit) // 2])
    )
    midpoint0 = min(max(midpoint0, float(x_fit[0])), float(x_fit[-1]))

    try:
        popt, pcov = curve_fit(
            sigmoid_rise,
            x_fit,
            y_fit,
            p0=[amplitude0, k0, baseline0, midpoint0],
            bounds=(
                [0.0, k_lower, 0.0, float(x_fit[0])],
                [max(float(np.max(y_fit)) * 3.0, 1.0), tau_max_ns, max(float(np.max(y_fit)), 1.0e-9), float(x_fit[-1])],
            ),
            sigma=np.sqrt(np.maximum(y_fit, 1.0)),
            absolute_sigma=False,
            maxfev=20000,
        )
    except Exception as exc:  # noqa: BLE001
        return skipped_rise_result(
            cut,
            f"fit_failed:{exc}",
            peak_time,
            peak_counts,
            threshold_10_time,
            threshold_90_time,
        )

    y_pred = sigmoid_rise(x_fit, float(popt[0]), float(popt[1]), float(popt[2]), float(popt[3]))
    residual = y_fit - y_pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(residual**2)))
    perr = np.sqrt(np.diag(pcov)) if pcov.size else np.full(4, np.nan)
    sigmoid_k = float(popt[1])
    return RiseFitResult(
        center_nm=cut.center_nm,
        wavelength_min_nm=cut.wavelength_min_nm,
        wavelength_max_nm=cut.wavelength_max_nm,
        column_count=cut.column_count,
        status="fit",
        reason="",
        rise_tau_ns=sigmoid_k,
        rise_tau_se_ns=float(perr[1]) if len(perr) > 1 else float("nan"),
        rise_time_10_90_ns=2.0 * math.log(9.0) * sigmoid_k,
        observed_rise_time_10_90_ns=observed_rise_time,
        amplitude_counts=float(popt[0]),
        baseline_counts=float(popt[2]),
        midpoint_time_ns=float(popt[3]),
        r_squared=r_squared,
        rmse_counts=rmse,
        peak_time_ns=peak_time,
        peak_counts=peak_counts,
        threshold_10_time_ns=threshold_10_time,
        threshold_90_time_ns=threshold_90_time,
        fit_start_ns=float(x_fit[0]),
        fit_end_ns=float(x_fit[-1]),
        fit_points=len(x_fit),
    )


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


def rise_result_row(band: BandProfile, result: RiseFitResult) -> dict[str, object]:
    """Convert one rise-fit result to a text-output row."""
    return {
        "band_min_nm": band.requested_min_nm,
        "band_max_nm": band.requested_max_nm,
        "band_center_nm": result.center_nm,
        "actual_wavelength_min_nm": result.wavelength_min_nm,
        "actual_wavelength_max_nm": result.wavelength_max_nm,
        "column_count": result.column_count,
        "status": result.status,
        "reason": result.reason,
        "sigmoid_k_ns": result.rise_tau_ns,
        "sigmoid_k_se_ns": result.rise_tau_se_ns,
        "fitted_rise_time_10_90_ns": result.rise_time_10_90_ns,
        "observed_rise_time_10_90_ns": result.observed_rise_time_10_90_ns,
        "amplitude_counts": result.amplitude_counts,
        "baseline_counts": result.baseline_counts,
        "midpoint_time_ns": result.midpoint_time_ns,
        "r_squared": result.r_squared,
        "rmse_counts": result.rmse_counts,
        "peak_time_ns": result.peak_time_ns,
        "peak_counts": result.peak_counts,
        "threshold_10_time_ns": result.threshold_10_time_ns,
        "threshold_90_time_ns": result.threshold_90_time_ns,
        "fit_start_ns": result.fit_start_ns,
        "fit_end_ns": result.fit_end_ns,
        "fit_points": result.fit_points,
    }


RISE_SUMMARY_FIELDS = [
    "band_min_nm",
    "band_max_nm",
    "band_center_nm",
    "actual_wavelength_min_nm",
    "actual_wavelength_max_nm",
    "column_count",
    "status",
    "reason",
    "sigmoid_k_ns",
    "sigmoid_k_se_ns",
    "fitted_rise_time_10_90_ns",
    "observed_rise_time_10_90_ns",
    "amplitude_counts",
    "baseline_counts",
    "midpoint_time_ns",
    "r_squared",
    "rmse_counts",
    "peak_time_ns",
    "peak_counts",
    "threshold_10_time_ns",
    "threshold_90_time_ns",
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
    decay_fit_count: int,
    rise_fit_count: int,
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
        {"key": "decay_fit_count", "value": decay_fit_count},
        {"key": "rise_fit_count", "value": rise_fit_count},
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


def finite_profile_range(bands: list[BandProfile]) -> tuple[float, float] | None:
    """Return finite y limits for raw profile plots."""
    finite_values = np.concatenate(
        [band.cut.profile_counts[np.isfinite(band.cut.profile_counts)] for band in bands if band.cut.profile_counts.size]
    )
    if finite_values.size == 0:
        return None
    y_min = float(np.nanmin(finite_values))
    y_max = float(np.nanmax(finite_values))
    if not math.isfinite(y_min) or not math.isfinite(y_max):
        return None
    if y_min == y_max:
        padding = max(abs(y_min) * 0.05, 1.0)
    else:
        padding = 0.04 * (y_max - y_min)
    return y_min - padding, y_max + padding


def plot_raw_slices(
    out_base: Path,
    time_ns: np.ndarray,
    bands: list[BandProfile],
    results: list[FitResult],
    relative_path: Path,
) -> None:
    """Plot non-noise wavelength-cut kinetics as raw mean counts."""
    if not bands:
        return
    plotted_bands = [
        band
        for band, result in zip(bands, results)
        if not (result.status == "skipped" and result.reason == "low_signal")
    ]
    if not plotted_bands:
        return

    centers = np.array([band.cut.center_nm for band in plotted_bands], dtype=float)
    norm = Normalize(vmin=float(np.nanmin(centers)), vmax=float(np.nanmax(centers)))
    cmap = plt.get_cmap("viridis")
    raw_ylim = finite_profile_range(plotted_bands)

    for yscale, suffix, ylabel in [
        ("linear", "linear", "Mean counts"),
        ("log", "semilog", "Mean counts"),
    ]:
        set_publication_style(base_font_size=8.2)
        fig, ax = plt.subplots(figsize=DOUBLE_COLUMN_WIDE, constrained_layout=True)
        plotted = False
        for band in plotted_bands:
            label = f"{format_nm(band.requested_min_nm)}-{format_nm(band.requested_max_nm)} nm"
            values = band.cut.profile_counts.astype(float)
            y_values = np.where(values > 0, values, np.nan) if yscale == "log" else values
            if not np.isfinite(y_values).any():
                continue
            ax.plot(
                time_ns,
                y_values,
                lw=0.85,
                alpha=0.86,
                color=cmap(norm(band.cut.center_nm)),
                label=label,
            )
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_title(relative_path.stem.replace("_", " "), pad=5)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel(ylabel)
        if yscale == "linear" and raw_ylim is not None:
            ax.set_ylim(*raw_ylim)
        ax.set_yscale(yscale)
        apply_axes_style(ax, grid=True)
        scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar.set_array([])
        colorbar = fig.colorbar(scalar, ax=ax, pad=0.015, fraction=0.035)
        colorbar.set_label("Band center (nm)")
        save_figure(fig, out_base.with_name(f"{out_base.name}_{suffix}.png"), dpi=260)
        save_figure(fig, out_base.with_name(f"{out_base.name}_{suffix}.pdf"))
        plt.close(fig)


def plot_individual_cut_profiles(
    out_dir: Path,
    time_ns: np.ndarray,
    bands: list[BandProfile],
) -> None:
    """Write one simple raw-count plot for each wavelength cut."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for band in bands:
        set_publication_style(base_font_size=8.0)
        fig, ax = plt.subplots(figsize=(4.2, 2.8), constrained_layout=True)
        ax.plot(time_ns, band.cut.profile_counts, color="#0072B2", lw=1.0)
        ax.set_title(f"{format_nm(band.requested_min_nm)}-{format_nm(band.requested_max_nm)} nm", pad=4)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Mean counts")
        apply_axes_style(ax, grid=True)
        path = out_dir / f"cut_{format_nm(band.requested_min_nm)}_{format_nm(band.requested_max_nm)}nm.png"
        save_figure(fig, path, dpi=220)
        plt.close(fig)


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


def write_rise_fit_curve_points(
    path: Path,
    time_ns: np.ndarray,
    bands: list[BandProfile],
    results: list[RiseFitResult],
) -> None:
    """Write raw and fitted sample points used by successful rise fits."""
    rows: list[dict[str, object]] = []
    for band, result in zip(bands, results):
        if result.status != "fit":
            continue
        mask = (time_ns >= result.fit_start_ns) & (time_ns <= result.fit_end_ns)
        for time_value, raw_counts in zip(time_ns[mask], band.cut.profile_counts[mask]):
            fitted_counts = sigmoid_rise(
                np.array([time_value], dtype=float),
                result.amplitude_counts,
                result.rise_tau_ns,
                result.baseline_counts,
                result.midpoint_time_ns,
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
    write_slice_plots: bool,
) -> ScanOutput:
    """Extract and fit wavelength bands for one carpet scan."""
    relative_path = path.relative_to(raw_dir)
    out_dir = output_root / sample_folder(relative_path) / safe_scan_folder(relative_path)
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
        rise_results = [
            fit_rise_profile(
                band.cut,
                times,
                smooth_sigma=smooth_sigma,
                min_fit_points=min_fit_points,
                min_peak_sigma=min_peak_sigma,
                tau_min_ns=tau_min_ns,
                tau_max_ns=tau_max_ns,
            )
            for band in bands
        ]
        decay_fit_count = sum(1 for result in results if result.status == "fit")
        decay_skipped_count = len(results) - decay_fit_count
        rise_fit_count = sum(1 for result in rise_results if result.status == "fit")
        rise_skipped_count = len(rise_results) - rise_fit_count

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
            decay_fit_count=decay_fit_count,
            rise_fit_count=rise_fit_count,
        )
        write_tsv(out_dir / "fit_summary.txt", [result_row(band, result) for band, result in zip(bands, results)], SUMMARY_FIELDS)
        write_tsv(
            out_dir / "rise_fit_summary.txt",
            [rise_result_row(band, result) for band, result in zip(bands, rise_results)],
            RISE_SUMMARY_FIELDS,
        )
        write_profiles(out_dir / "profiles_by_band.txt", times, bands)
        if write_slice_plots:
            plot_raw_slices(out_dir / "raw_slices", times, bands, results, relative_path)
            plot_individual_cut_profiles(out_dir / "cut_profiles", times, bands)
        if write_fit_curves:
            write_fit_curve_points(out_dir / "fit_curve_points.txt", times, bands, results)
            write_rise_fit_curve_points(out_dir / "rise_fit_curve_points.txt", times, bands, rise_results)

        return ScanOutput(
            source_path=path,
            relative_path=relative_path,
            output_dir=out_dir,
            band_count=len(bands),
            decay_fit_count=decay_fit_count,
            decay_skipped_count=decay_skipped_count,
            rise_fit_count=rise_fit_count,
            rise_skipped_count=rise_skipped_count,
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
            decay_fit_count=0,
            decay_skipped_count=0,
            rise_fit_count=0,
            rise_skipped_count=0,
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
                "sample": sample_folder(item.relative_path),
                "output_folder": item.output_dir.relative_to(output_root).as_posix(),
                "status": item.status,
                "band_count": item.band_count,
                "decay_fit_count": item.decay_fit_count,
                "decay_skipped_count": item.decay_skipped_count,
                "rise_fit_count": item.rise_fit_count,
                "rise_skipped_count": item.rise_skipped_count,
                "note": item.note,
            }
        )
    write_tsv(
        path,
        rows,
        [
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
        ],
    )


def main(argv: list[str] | None = None) -> int:
    """Run text-only batch wavelength cuts for all Hamamatsu carpet scans."""
    parser = argparse.ArgumentParser(description="Extract clean 20 nm wavelength cuts from all Hamamatsu .img carpets.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-subdir", default=DEFAULT_OUT_SUBDIR)
    parser.add_argument("--interval-nm", type=float, default=DEFAULT_INTERVAL_NM)
    parser.add_argument("--range-mode", choices=["common", "per-scan"], default="common")
    parser.add_argument(
        "--time-window",
        "--time-windows",
        dest="time_windows",
        action="append",
        default=None,
        help="Keep only selected acquisition windows, e.g. --time-window 2ns --time-window 10ns.",
    )
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
    parser.add_argument("--no-slice-plots", action="store_true")
    args = parser.parse_args(argv)

    raw_dir = resolve_path(args.raw_dir)
    results_dir = resolve_path(args.results_dir)
    output_root = (results_dir / args.out_subdir).resolve()
    paths = sorted(path for path in raw_dir.rglob("*.img") if ".venv" not in path.parts)
    selected_windows = csv_values(args.time_windows)
    if selected_windows:
        paths = [path for path in paths if scan_time_window(path.relative_to(raw_dir)) in selected_windows]
    if not paths:
        if selected_windows:
            raise SystemExit(f"No .img scans found under {raw_dir} for time window(s): {', '.join(sorted(selected_windows))}")
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
            write_slice_plots=not args.no_slice_plots,
        )
        outputs.append(output)
        print(
            f"[{index}/{total}] {output.status}: "
            f"{output.relative_path.as_posix()} -> {output.output_dir.relative_to(output_root).as_posix()}"
        )

    write_inventory(output_root / "inventory.txt", outputs, raw_dir, output_root)
    ok_count = sum(1 for item in outputs if item.status == "ok")
    decay_fit_count = sum(item.decay_fit_count for item in outputs)
    decay_skipped_count = sum(item.decay_skipped_count for item in outputs)
    rise_fit_count = sum(item.rise_fit_count for item in outputs)
    rise_skipped_count = sum(item.rise_skipped_count for item in outputs)
    print()
    print(f"scans processed: {ok_count} ok / {len(outputs) - ok_count} error / {len(outputs)} total")
    if common_bounds is not None and band_edges is not None:
        print(f"common wavelength coverage: {common_bounds[0]:.6g}-{common_bounds[1]:.6g} nm")
        print(f"clean bands: {band_edges[0][0]:g}-{band_edges[-1][1]:g} nm in {args.interval_nm:g} nm intervals")
    print(f"decay fits: {decay_fit_count} fit / {decay_skipped_count} skipped")
    print(f"rise fits: {rise_fit_count} fit / {rise_skipped_count} skipped")
    print(f"output: {output_root}")
    print(f"inventory: {output_root / 'inventory.txt'}")
    return 0 if ok_count == len(outputs) else 1


if __name__ == "__main__":
    sys.exit(main())
