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
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from .hamamatsu_streak import (
    TOP_EDGE_CROP_ROWS,
    crop_top_edge,
    image_extent,
    load_img,
    time_axis_ns,
    wavelength_axis_nm,
)
from .paths import DEFAULT_RESULTS_DIR, resolve_path
from .plot_style import CARPET_CMAP, COLORS, apply_axes_style, save_figure, set_publication_style, style_colorbar


@dataclass(frozen=True)
class CutProfile:
    center_nm: float
    wavelength_min_nm: float
    wavelength_max_nm: float
    column_count: int
    profile_counts: np.ndarray


@dataclass(frozen=True)
class FitResult:
    center_nm: float
    wavelength_min_nm: float
    wavelength_max_nm: float
    column_count: int
    status: str
    reason: str
    tau_ns: float
    tau_se_ns: float
    amplitude_counts: float
    baseline_counts: float
    r_squared: float
    rmse_counts: float
    peak_time_ns: float
    peak_counts: float
    detected_peak_count: int
    detected_peak_times_ns: str
    secondary_peak_count: int
    secondary_peak_times_ns: str
    fit_end_rule: str
    fit_start_ns: float
    fit_end_ns: float
    fit_points: int
    plot_png: str


def safe_stem(path: Path) -> str:
    """Return a filesystem-safe stem for a generated output name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)


def parse_centers(text: str) -> list[float]:
    """Parse a comma-separated list of wavelength cut centers."""
    values = []
    for item in text.split(","):
        stripped = item.strip()
        if stripped:
            values.append(float(stripped))
    if not values:
        raise ValueError("--centers was provided but no numeric centers were found")
    return values


def centers_from_range(
    wavelengths_nm: np.ndarray,
    *,
    centers: str | None,
    wavelength_min_nm: float | None,
    wavelength_max_nm: float | None,
    step_nm: float,
) -> list[float]:
    """Build evenly spaced wavelength cut centers from range settings."""
    if centers:
        return parse_centers(centers)
    if step_nm <= 0:
        raise ValueError("--step-nm must be positive")

    min_nm = float(np.nanmin(wavelengths_nm)) if wavelength_min_nm is None else wavelength_min_nm
    max_nm = float(np.nanmax(wavelengths_nm)) if wavelength_max_nm is None else wavelength_max_nm
    if max_nm < min_nm:
        raise ValueError("wavelength maximum must be greater than wavelength minimum")

    count = int(math.floor((max_nm - min_nm) / step_nm)) + 1
    return [min_nm + i * step_nm for i in range(count)]


def exp_decay(time_ns: np.ndarray, amplitude: float, tau_ns: float, baseline: float, t0_ns: float) -> np.ndarray:
    """Evaluate a single-exponential decay with a constant baseline."""
    return amplitude * np.exp(-(time_ns - t0_ns) / tau_ns) + baseline


def robust_noise(values: np.ndarray) -> float:
    """Estimate noise using a median absolute deviation fallback."""
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad == 0:
        return float(np.std(values))
    return 1.4826 * mad


def first_sustained_true(mask: np.ndarray, run_length: int) -> int | None:
    """Return the first index where a boolean mask stays true for a run."""
    run = 0
    for idx, flag in enumerate(mask):
        run = run + 1 if flag else 0
        if run >= run_length:
            return idx - run_length + 1
    return None


def format_times(indices: list[int], time_ns: np.ndarray) -> str:
    """Format time values for compact CSV output."""
    return ";".join(f"{float(time_ns[index]):.6g}" for index in indices)


def parse_time_list(text: str) -> list[float]:
    """Parse semicolon-delimited time values from stored metadata."""
    values: list[float] = []
    for item in text.split(";"):
        if item.strip():
            values.append(float(item))
    return values


def significant_peak_indices(
    smooth: np.ndarray,
    baseline: float,
    peak_height: float,
    noise: float,
    dt_ns: float,
    *,
    primary_peak_idx: int,
    height_fraction: float,
    prominence_fraction: float,
    noise_sigma: float,
    min_separation_ns: float,
) -> list[int]:
    """Find signal peaks that pass height, prominence, and spacing cuts."""
    if not np.isfinite(dt_ns) or dt_ns <= 0:
        min_distance = 1
    else:
        min_distance = max(1, int(round(min_separation_ns / dt_ns)))
    height_threshold = baseline + max(height_fraction * peak_height, noise_sigma * noise, 1.0)
    prominence_threshold = max(prominence_fraction * peak_height, noise_sigma * noise, 1.0)
    peaks, _ = find_peaks(
        smooth,
        height=height_threshold,
        prominence=prominence_threshold,
        distance=min_distance,
    )
    peak_set = {int(peak) for peak in peaks}
    peak_set.add(primary_peak_idx)
    return sorted(peak_set)


def make_cut_profile(
    data: np.ndarray,
    wavelengths_nm: np.ndarray,
    center_nm: float,
    width_nm: float,
) -> CutProfile | None:
    """Average a wavelength band from a carpet into a time profile."""
    half_width = width_nm / 2.0
    mask = (wavelengths_nm >= center_nm - half_width) & (wavelengths_nm <= center_nm + half_width)
    if not np.any(mask):
        return None
    profile = data[:, mask].mean(axis=1)
    selected_wavelengths = wavelengths_nm[mask]
    return CutProfile(
        center_nm=center_nm,
        wavelength_min_nm=float(selected_wavelengths.min()),
        wavelength_max_nm=float(selected_wavelengths.max()),
        column_count=int(np.count_nonzero(mask)),
        profile_counts=profile.astype(float),
    )


def fit_cut_profile(
    cut: CutProfile,
    time_ns: np.ndarray,
    *,
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
) -> FitResult:
    """Fit a cut profile with a single-exponential decay when valid."""
    y = cut.profile_counts
    smooth = gaussian_filter1d(y, sigma=smooth_sigma) if smooth_sigma > 0 else y
    baseline_seed = float(np.percentile(y, 5))
    low_values = y[y <= np.percentile(y, 25)]
    noise = max(robust_noise(low_values), math.sqrt(max(baseline_seed, 0.0) + 1.0))
    signal = smooth - baseline_seed
    peak_idx = int(np.argmax(signal))
    peak_height = float(signal[peak_idx])
    peak_time = float(time_ns[peak_idx])
    peak_counts = float(y[peak_idx])
    dt_ns = float(np.median(np.abs(np.diff(time_ns)))) if time_ns.size > 1 else float("nan")
    peak_indices = significant_peak_indices(
        smooth,
        baseline_seed,
        peak_height,
        noise,
        dt_ns,
        primary_peak_idx=peak_idx,
        height_fraction=secondary_peak_height_fraction,
        prominence_fraction=secondary_peak_prominence_fraction,
        noise_sigma=secondary_peak_noise_sigma,
        min_separation_ns=secondary_peak_min_separation_ns,
    )
    secondary_peak_indices = [idx for idx in peak_indices if idx > peak_idx]
    detected_peak_times = format_times(peak_indices, time_ns)
    secondary_peak_times = format_times(secondary_peak_indices, time_ns)

    if peak_height <= max(min_peak_sigma * noise, 1.0):
        return skipped_result(
            cut,
            "low_signal",
            peak_time,
            peak_counts,
            peak_indices,
            secondary_peak_indices,
            time_ns,
        )

    start_time = fit_start_ns if fit_start_ns is not None else peak_time + fit_start_offset_ns
    start_candidates = np.where(time_ns >= start_time)[0]
    if start_candidates.size == 0:
        return skipped_result(
            cut,
            "fit_start_after_trace",
            peak_time,
            peak_counts,
            peak_indices,
            secondary_peak_indices,
            time_ns,
        )
    fit_start_idx = int(start_candidates[0])

    if fit_end_ns is not None:
        end_candidates = np.where(time_ns <= fit_end_ns)[0]
        if end_candidates.size == 0:
            return skipped_result(
                cut,
                "fit_end_before_trace",
                peak_time,
                peak_counts,
                peak_indices,
                secondary_peak_indices,
                time_ns,
            )
        fit_end_idx = int(end_candidates[-1]) + 1
        fit_end_rule = "manual_fit_end"
    elif end_fraction > 0:
        threshold = baseline_seed + end_fraction * peak_height
        below = smooth[fit_start_idx:] <= threshold
        sustained_idx = first_sustained_true(below, run_length=8)
        fit_end_idx = len(time_ns) if sustained_idx is None else fit_start_idx + sustained_idx
        fit_end_rule = "end_fraction" if sustained_idx is not None else "trace_end"
    else:
        fit_end_idx = len(time_ns)
        fit_end_rule = "trace_end"

    later_secondary_indices = [idx for idx in secondary_peak_indices if fit_start_idx < idx < fit_end_idx]
    if later_secondary_indices:
        first_secondary_idx = later_secondary_indices[0]
        secondary_cutoff_time = float(time_ns[first_secondary_idx]) - secondary_peak_exclusion_before_ns
        cutoff_candidates = np.where(time_ns <= secondary_cutoff_time)[0]
        if cutoff_candidates.size == 0:
            return skipped_result(
                cut,
                "secondary_peak_before_fit_start",
                peak_time,
                peak_counts,
                peak_indices,
                secondary_peak_indices,
                time_ns,
            )
        secondary_fit_end_idx = int(cutoff_candidates[-1]) + 1
        if secondary_fit_end_idx < fit_end_idx:
            fit_end_idx = secondary_fit_end_idx
            fit_end_rule = f"{fit_end_rule};secondary_peak_excluded"

    if fit_end_idx - fit_start_idx < min_fit_points and "secondary_peak_excluded" not in fit_end_rule:
        fit_end_idx = min(len(time_ns), fit_start_idx + min_fit_points)
    if fit_end_idx - fit_start_idx < min_fit_points:
        reason = "secondary_peak_too_close" if "secondary_peak_excluded" in fit_end_rule else "too_few_fit_points"
        return skipped_result(
            cut,
            reason,
            peak_time,
            peak_counts,
            peak_indices,
            secondary_peak_indices,
            time_ns,
        )

    x_fit = time_ns[fit_start_idx:fit_end_idx]
    y_fit = y[fit_start_idx:fit_end_idx]
    t0 = float(x_fit[0])
    tail = y_fit[max(0, int(0.8 * len(y_fit))) :]
    baseline0 = max(0.0, float(np.percentile(tail, 20)))
    amplitude0 = max(float(y_fit[0] - baseline0), 1.0)
    target = baseline0 + amplitude0 / math.e
    below_e = np.where(y_fit <= target)[0]
    if below_e.size and below_e[0] > 0:
        tau0 = max(float(x_fit[below_e[0]] - x_fit[0]), tau_min_ns)
    else:
        tau0 = max((float(x_fit[-1] - x_fit[0]) / 3.0), tau_min_ns)
    tau0 = min(tau0, tau_max_ns)

    try:
        popt, pcov = curve_fit(
            lambda x_data, amplitude, tau_ns, baseline: exp_decay(x_data, amplitude, tau_ns, baseline, t0),
            x_fit,
            y_fit,
            p0=[amplitude0, tau0, baseline0],
            bounds=(
                [0.0, tau_min_ns, 0.0],
                [np.inf, tau_max_ns, max(float(np.max(y_fit)), 1.0e-9)],
            ),
            sigma=np.sqrt(np.maximum(y_fit, 1.0)),
            absolute_sigma=False,
            maxfev=20000,
        )
    except Exception as exc:  # noqa: BLE001
        return skipped_result(
            cut,
            f"fit_failed:{exc}",
            peak_time,
            peak_counts,
            peak_indices,
            secondary_peak_indices,
            time_ns,
        )

    y_pred = exp_decay(x_fit, float(popt[0]), float(popt[1]), float(popt[2]), t0)
    residual = y_fit - y_pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(residual**2)))
    perr = np.sqrt(np.diag(pcov)) if pcov.size else np.full(3, np.nan)

    return FitResult(
        center_nm=cut.center_nm,
        wavelength_min_nm=cut.wavelength_min_nm,
        wavelength_max_nm=cut.wavelength_max_nm,
        column_count=cut.column_count,
        status="fit",
        reason="",
        tau_ns=float(popt[1]),
        tau_se_ns=float(perr[1]) if len(perr) > 1 else float("nan"),
        amplitude_counts=float(popt[0]),
        baseline_counts=float(popt[2]),
        r_squared=r_squared,
        rmse_counts=rmse,
        peak_time_ns=peak_time,
        peak_counts=peak_counts,
        detected_peak_count=len(peak_indices),
        detected_peak_times_ns=detected_peak_times,
        secondary_peak_count=len(secondary_peak_indices),
        secondary_peak_times_ns=secondary_peak_times,
        fit_end_rule=fit_end_rule,
        fit_start_ns=float(x_fit[0]),
        fit_end_ns=float(x_fit[-1]),
        fit_points=len(x_fit),
        plot_png="",
    )


def skipped_result(
    cut: CutProfile,
    reason: str,
    peak_time_ns: float,
    peak_counts: float,
    peak_indices: list[int],
    secondary_peak_indices: list[int],
    time_ns: np.ndarray,
) -> FitResult:
    """Create a fit result describing why a cut was skipped."""
    return FitResult(
        center_nm=cut.center_nm,
        wavelength_min_nm=cut.wavelength_min_nm,
        wavelength_max_nm=cut.wavelength_max_nm,
        column_count=cut.column_count,
        status="skipped",
        reason=reason,
        tau_ns=float("nan"),
        tau_se_ns=float("nan"),
        amplitude_counts=float("nan"),
        baseline_counts=float("nan"),
        r_squared=float("nan"),
        rmse_counts=float("nan"),
        peak_time_ns=peak_time_ns,
        peak_counts=peak_counts,
        detected_peak_count=len(peak_indices),
        detected_peak_times_ns=format_times(peak_indices, time_ns),
        secondary_peak_count=len(secondary_peak_indices),
        secondary_peak_times_ns=format_times(secondary_peak_indices, time_ns),
        fit_end_rule="",
        fit_start_ns=float("nan"),
        fit_end_ns=float("nan"),
        fit_points=0,
        plot_png="",
    )


def result_with_plot(result: FitResult, plot_png: str) -> FitResult:
    """Return a fit result updated with its diagnostic plot path."""
    return FitResult(
        center_nm=result.center_nm,
        wavelength_min_nm=result.wavelength_min_nm,
        wavelength_max_nm=result.wavelength_max_nm,
        column_count=result.column_count,
        status=result.status,
        reason=result.reason,
        tau_ns=result.tau_ns,
        tau_se_ns=result.tau_se_ns,
        amplitude_counts=result.amplitude_counts,
        baseline_counts=result.baseline_counts,
        r_squared=result.r_squared,
        rmse_counts=result.rmse_counts,
        peak_time_ns=result.peak_time_ns,
        peak_counts=result.peak_counts,
        detected_peak_count=result.detected_peak_count,
        detected_peak_times_ns=result.detected_peak_times_ns,
        secondary_peak_count=result.secondary_peak_count,
        secondary_peak_times_ns=result.secondary_peak_times_ns,
        fit_end_rule=result.fit_end_rule,
        fit_start_ns=result.fit_start_ns,
        fit_end_ns=result.fit_end_ns,
        fit_points=result.fit_points,
        plot_png=plot_png,
    )


def write_summary(path: Path, results: list[FitResult]) -> None:
    """Write wavelength-cut fit results to a CSV summary."""
    fields = [
        "center_nm",
        "wavelength_min_nm",
        "wavelength_max_nm",
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
        "plot_png",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: getattr(result, field) for field in fields})


def write_profiles(path: Path, time_ns: np.ndarray, cuts: list[CutProfile]) -> None:
    """Write extracted wavelength-cut profiles to a CSV file."""
    fields = ["center_nm", "wavelength_min_nm", "wavelength_max_nm", "time_ns", "mean_counts"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cut in cuts:
            for t, counts in zip(time_ns, cut.profile_counts):
                writer.writerow(
                    {
                        "center_nm": cut.center_nm,
                        "wavelength_min_nm": cut.wavelength_min_nm,
                        "wavelength_max_nm": cut.wavelength_max_nm,
                        "time_ns": float(t),
                        "mean_counts": float(counts),
                    }
                )


def plot_cut_fit(
    path: Path,
    cut: CutProfile,
    result: FitResult,
    time_ns: np.ndarray,
) -> None:
    """Plot one wavelength-cut profile and its fit diagnostics."""
    set_publication_style(base_font_size=8.0)
    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    ax.plot(time_ns, cut.profile_counts, "o", ms=2.1, color=COLORS["black"], alpha=0.65, label="cut profile")
    ax.axvline(result.peak_time_ns, color=COLORS["blue"], ls=":", lw=0.9, label="peak")
    secondary_label_used = False
    for secondary_time in parse_time_list(result.secondary_peak_times_ns):
        secondary_idx = int(np.argmin(np.abs(time_ns - secondary_time)))
        ax.axvline(
            secondary_time,
            color=COLORS["purple"],
            ls="--",
            lw=0.85,
            alpha=0.75,
            label="secondary peak" if not secondary_label_used else "_nolegend_",
        )
        ax.plot(
            time_ns[secondary_idx],
            cut.profile_counts[secondary_idx],
            "x",
            ms=5.0,
            mew=1.2,
            color=COLORS["purple"],
            label="_nolegend_",
        )
        secondary_label_used = True
    if result.status == "fit":
        fit_mask = (time_ns >= result.fit_start_ns) & (time_ns <= result.fit_end_ns)
        x_dense = np.linspace(result.fit_start_ns, result.fit_end_ns, 400)
        y_dense = exp_decay(x_dense, result.amplitude_counts, result.tau_ns, result.baseline_counts, result.fit_start_ns)
        ax.axvspan(result.fit_start_ns, result.fit_end_ns, color=COLORS["vermillion"], alpha=0.08, lw=0)
        ax.plot(x_dense, y_dense, "-", color=COLORS["vermillion"], lw=1.35, label="single-exponential fit")
        label = (
            f"tau = {result.tau_ns:.3g} ns\n"
            f"R^2 = {result.r_squared:.3f}\n"
            f"points = {result.fit_points}"
        )
        if result.secondary_peak_count:
            label += f"\nsecondary peaks = {result.secondary_peak_count}"
    else:
        fit_mask = np.zeros_like(time_ns, dtype=bool)
        label = f"Skipped: {result.reason}"
    if np.any(fit_mask):
        ax.plot(time_ns[fit_mask], cut.profile_counts[fit_mask], "o", ms=2.1, color=COLORS["vermillion"], alpha=0.75)
    ax.text(
        0.98,
        0.95,
        label,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=7.5,
        bbox={"boxstyle": "square,pad=0.25", "facecolor": "white", "edgecolor": "#C9CED6", "alpha": 0.92},
    )
    ax.set_title(f"{cut.center_nm:.1f} nm cut ({cut.wavelength_min_nm:.1f}-{cut.wavelength_max_nm:.1f} nm)", pad=4)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Mean counts")
    ax.legend(loc="lower left", frameon=True, framealpha=0.88, facecolor="white", edgecolor="#C9CED6")
    apply_axes_style(ax, grid=True)
    save_figure(fig, path, dpi=260)
    plt.close(fig)


def plot_profiles_overlay(path: Path, time_ns: np.ndarray, cuts: list[CutProfile]) -> None:
    """Plot all wavelength-cut profiles on one overlay."""
    set_publication_style(base_font_size=8.0)
    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    cmap = matplotlib.colormaps["viridis"]
    centers = np.array([cut.center_nm for cut in cuts], dtype=float)
    vmin = float(np.min(centers))
    vmax = float(np.max(centers))
    if vmin == vmax:
        vmax = vmin + 1.0
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    for cut in cuts:
        ax.plot(time_ns, cut.profile_counts, lw=0.9, alpha=0.82, color=cmap(norm(cut.center_nm)))
    scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.02)
    colorbar.set_label("Cut center (nm)")
    style_colorbar(colorbar)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Mean counts")
    ax.set_title("Wavelength-cut time profiles", pad=5)
    apply_axes_style(ax, grid=True)
    save_figure(fig, path, dpi=260)
    plt.close(fig)


def plot_tau_vs_wavelength(path: Path, results: list[FitResult]) -> None:
    """Plot fitted decay time versus wavelength cut center."""
    fit_results = [result for result in results if result.status == "fit" and np.isfinite(result.tau_ns)]
    if not fit_results:
        return
    set_publication_style(base_font_size=8.0)
    fig, ax = plt.subplots(figsize=(4.3, 3.0), constrained_layout=True)
    centers = np.array([result.center_nm for result in fit_results], dtype=float)
    tau = np.array([result.tau_ns for result in fit_results], dtype=float)
    tau_se = np.array([result.tau_se_ns for result in fit_results], dtype=float)
    tau_se[~np.isfinite(tau_se)] = 0.0
    ax.errorbar(centers, tau, yerr=tau_se, fmt="o-", ms=3.2, lw=1.0, color=COLORS["blue"], ecolor=COLORS["gray"])
    ax.set_xlabel("Cut center (nm)")
    ax.set_ylabel("Decay time tau (ns)")
    ax.set_title("Fitted decay time by wavelength cut", pad=5)
    apply_axes_style(ax, grid=True)
    save_figure(fig, path, dpi=260)
    plt.close(fig)


def plot_carpet_with_cuts(
    path: Path,
    carpet_path: Path,
    data: np.ndarray,
    extent: tuple[float, float, float, float] | None,
    cuts: list[CutProfile],
) -> None:
    """Plot a carpet image annotated with wavelength cut bands."""
    set_publication_style(base_font_size=8.0)
    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    vmin, vmax = np.percentile(data, (1, 99.7))
    imshow_kwargs = {"extent": extent} if extent is not None else {}
    im = ax.imshow(data, origin="lower", aspect="auto", cmap=CARPET_CMAP, vmin=vmin, vmax=vmax, **imshow_kwargs)
    for cut in cuts:
        ax.axvspan(cut.wavelength_min_nm, cut.wavelength_max_nm, color="white", alpha=0.10, lw=0)
        ax.axvline(cut.center_nm, color="white", alpha=0.35, lw=0.45)
    colorbar = fig.colorbar(im, ax=ax, pad=0.02)
    colorbar.set_label("Counts")
    style_colorbar(colorbar)
    ax.set_xlabel("Wavelength (nm)" if extent is not None else "x pixel")
    ax.set_ylabel("Time (ns)" if extent is not None else "y pixel")
    ax.set_title(carpet_path.stem.replace("_", " "), pad=5)
    save_figure(fig, path, dpi=260)
    plt.close(fig)


def default_out_dir(img_path: Path) -> Path:
    """Return the default output directory for a manual carpet cut run."""
    return DEFAULT_RESULTS_DIR / "manual_carpet_wavelength_cuts" / safe_stem(img_path)


def main(argv: list[str] | None = None) -> None:
    """Run the manual carpet wavelength-cut fitting command."""
    parser = argparse.ArgumentParser(
        description="Manually fit single-exponential decays from wavelength cuts through one Hamamatsu carpet."
    )
    parser.add_argument("img", type=Path, help="Hamamatsu .img carpet to analyze.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output folder. Defaults under Analysis results.")
    parser.add_argument("--top-edge-crop-rows", type=int, default=TOP_EDGE_CROP_ROWS)
    parser.add_argument("--centers", help="Comma-separated wavelength centers in nm. Overrides min/max/step.")
    parser.add_argument("--wavelength-min-nm", type=float, default=None)
    parser.add_argument("--wavelength-max-nm", type=float, default=None)
    parser.add_argument("--step-nm", type=float, default=10.0, help="Spacing between wavelength-cut centers.")
    parser.add_argument("--band-width-nm", type=float, default=10.0, help="Width of each wavelength cut.")
    parser.add_argument("--smooth-sigma", type=float, default=2.0)
    parser.add_argument("--fit-start-ns", type=float, default=None, help="Absolute fit start time. Default: peak + offset.")
    parser.add_argument("--fit-start-offset-ns", type=float, default=0.0, help="Offset from detected peak for auto fit start.")
    parser.add_argument("--fit-end-ns", type=float, default=None, help="Absolute fit end time. Default: auto or trace end.")
    parser.add_argument("--end-fraction", type=float, default=0.05, help="Auto fit end at this fraction of peak signal; 0 uses trace end.")
    parser.add_argument("--min-fit-points", type=int, default=20)
    parser.add_argument("--min-peak-sigma", type=float, default=5.0)
    parser.add_argument("--secondary-peak-height-fraction", type=float, default=0.20)
    parser.add_argument("--secondary-peak-prominence-fraction", type=float, default=0.12)
    parser.add_argument("--secondary-peak-noise-sigma", type=float, default=3.0)
    parser.add_argument("--secondary-peak-min-separation-ns", type=float, default=0.25)
    parser.add_argument("--secondary-peak-exclusion-before-ns", type=float, default=0.05)
    parser.add_argument("--tau-min-ns", type=float, default=0.03)
    parser.add_argument("--tau-max-ns", type=float, default=200.0)
    parser.add_argument("--no-individual-plots", action="store_true")
    args = parser.parse_args(argv)

    img_path = resolve_path(args.img)
    out_dir = resolve_path(args.out_dir) if args.out_dir is not None else default_out_dir(img_path).resolve()
    fits_dir = out_dir / "fits"

    carpet = load_img(img_path)
    wavelengths = wavelength_axis_nm(carpet)
    times = time_axis_ns(carpet, cropped=True, crop_rows=args.top_edge_crop_rows)
    if wavelengths is None:
        raise SystemExit("This carpet does not have usable wavelength calibration metadata.")
    if times is None:
        raise SystemExit("This carpet does not have usable streak-time calibration metadata.")

    data = crop_top_edge(carpet.data.astype(float), args.top_edge_crop_rows)
    centers = centers_from_range(
        wavelengths,
        centers=args.centers,
        wavelength_min_nm=args.wavelength_min_nm,
        wavelength_max_nm=args.wavelength_max_nm,
        step_nm=args.step_nm,
    )
    cuts = [
        cut
        for center in centers
        if (cut := make_cut_profile(data, wavelengths, center, args.band_width_nm)) is not None
    ]
    if not cuts:
        raise SystemExit("No wavelength cuts overlap the calibrated wavelength axis.")

    results: list[FitResult] = []
    for cut in cuts:
        result = fit_cut_profile(
            cut,
            times,
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
        )
        if args.no_individual_plots:
            results.append(result)
            continue
        plot_path = fits_dir / f"cut_{cut.center_nm:.1f}nm_fit.png"
        plot_cut_fit(plot_path, cut, result, times)
        results.append(result_with_plot(result, plot_path.relative_to(out_dir).as_posix()))

    write_summary(out_dir / "fit_summary.csv", results)
    write_profiles(out_dir / "cut_profiles.csv", times, cuts)
    plot_profiles_overlay(out_dir / "profiles_overlay.png", times, cuts)
    plot_tau_vs_wavelength(out_dir / "tau_vs_wavelength.png", results)
    plot_carpet_with_cuts(
        out_dir / "carpet_with_cuts.png",
        img_path,
        data,
        image_extent(carpet, cropped=True, crop_rows=args.top_edge_crop_rows),
        cuts,
    )

    fit_count = sum(1 for result in results if result.status == "fit")
    print(f"carpet: {img_path}")
    print(f"cuts: {len(cuts)}")
    print(f"fits: {fit_count} fit / {len(results) - fit_count} skipped")
    print(f"output: {out_dir}")
    print(f"summary: {out_dir / 'fit_summary.csv'}")


if __name__ == "__main__":
    main()
