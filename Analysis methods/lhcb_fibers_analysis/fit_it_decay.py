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
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from .fiber_names import FiberNameMap, read_fiber_name_map
from .paths import (
    DEFAULT_FIBER_NAMES_CONFIG,
    DEFAULT_IT_DECAY_CONFIG,
    DEFAULT_RAW_DIR,
    DEFAULT_RESULTS_DIR,
    resolve_path,
)
from .plot_style import (
    COLORS,
    DOUBLE_COLUMN_WIDE,
    apply_axes_style,
    save_figure,
    set_publication_style,
)
from .scan_config import (
    SectionConfig,
    configs_exist,
    normalize_config_path,
    read_section_configs,
    relative_config_dir,
    yaml_scalar,
)
from .yaml_config import bool_value, float_value, int_value, read_yaml_mapping, string_value


DEFAULT_TIME_WINDOW = "10ns"
LINE_COLORS = [
    COLORS["blue"],
    COLORS["vermillion"],
    COLORS["teal"],
    COLORS["purple"],
    "#CC79A7",
    "#E69F00",
]
MARKERS = ["o", "s", "^", "D", "v", "P"]
MARKER_SIZE = 4.2


@dataclass(frozen=True)
class FitConfig:
    out_subdir: str = "it_decay_fits_10ns"
    trace_config_dir: str = "it_decay_fits_10ns"
    time_window: str = DEFAULT_TIME_WINDOW
    fit_all_discovered_traces: bool = True
    fit_window_ns: float = 0.0
    skip_low_signal: bool = True
    multiple_peak_strategy: str = "skip"
    scatter_excluded_time_windows: tuple[str, ...] = ()
    scatter_guide: str = "linear"
    scatter_guide_alpha: float = 0.32
    smooth_sigma: float = 2.0
    min_points: int = 30
    min_fit_points: int = 25
    peak_height_fraction: float = 0.20
    peak_prominence_fraction: float = 0.12
    noise_sigma_threshold: float = 3.0
    low_signal_sigma_threshold: float = 5.0
    fit_tau_min_ns: float = 0.03
    fit_tau_max_ns: float = 50.0


def csv_values(value: object) -> tuple[str, ...]:
    """Parse a comma-separated scalar config value."""
    return tuple(part.strip() for part in str(value or "").split(",") if part.strip())


def read_fit_config(path: Path) -> FitConfig:
    """Read integrated-time decay fitting settings from YAML."""
    values = read_yaml_mapping(path)
    tau_min = float_value(values, "fit_tau_min_ns", 0.03)
    tau_max = float_value(values, "fit_tau_max_ns", 50.0)
    if tau_max <= tau_min:
        raise ValueError(f"fit_tau_max_ns must be greater than fit_tau_min_ns in {path}")
    multiple_peak_strategy = string_value(values, "multiple_peak_strategy", "skip").strip().lower()
    if multiple_peak_strategy not in {"skip", "first", "dominant"}:
        raise ValueError(
            f"multiple_peak_strategy must be one of skip, first, or dominant in {path}"
        )
    scatter_guide = string_value(values, "scatter_guide", "linear").strip().lower()
    if scatter_guide not in {"none", "linear", "point_to_point"}:
        raise ValueError(f"scatter_guide must be one of none, linear, or point_to_point in {path}")
    return FitConfig(
        out_subdir=string_value(values, "out_subdir", "it_decay_fits_10ns"),
        trace_config_dir=string_value(values, "trace_config_dir", "it_decay_fits_10ns"),
        time_window=string_value(values, "time_window", DEFAULT_TIME_WINDOW),
        fit_all_discovered_traces=bool_value(values, "fit_all_discovered_traces", True),
        fit_window_ns=max(0.0, float_value(values, "fit_window_ns", 0.0)),
        skip_low_signal=bool_value(values, "skip_low_signal", True),
        multiple_peak_strategy=multiple_peak_strategy,
        scatter_excluded_time_windows=csv_values(values.get("scatter_excluded_time_windows", "")),
        scatter_guide=scatter_guide,
        scatter_guide_alpha=max(0.0, min(1.0, float_value(values, "scatter_guide_alpha", 0.32))),
        smooth_sigma=float_value(values, "smooth_sigma", 2.0),
        min_points=max(1, int_value(values, "min_points", 30)),
        min_fit_points=max(1, int_value(values, "min_fit_points", 25)),
        peak_height_fraction=float_value(values, "peak_height_fraction", 0.20),
        peak_prominence_fraction=float_value(values, "peak_prominence_fraction", 0.12),
        noise_sigma_threshold=float_value(values, "noise_sigma_threshold", 3.0),
        low_signal_sigma_threshold=float_value(values, "low_signal_sigma_threshold", 5.0),
        fit_tau_min_ns=tau_min,
        fit_tau_max_ns=tau_max,
    )

FILENAME_RE = re.compile(
    r"^IT_(?P<sample>.+?)_"
    r"(?P<position>ENDcm|\d+cm[A-Za-z0-9]*)_"
    r"ex(?P<excitation>\d+nm)_"
    r"(?P<energy>\d+nJ)_"
    r"(?P<window>\d+(?:ns|us|ms))\.dat$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TraceInfo:
    path: Path
    relative_path: Path
    sample: str
    position: str
    distance_cm: int | None
    position_suffix: str
    excitation: str
    excitation_nm: int | None
    energy: str
    energy_nj: int | None
    window: str
    wavelength_label: str
    measurement_date: str


def position_parts(position: str) -> tuple[int | None, str]:
    """Split a position token into numeric distance and suffix."""
    match = re.match(r"(?P<distance>\d+)cm(?P<suffix>.*)$", position, flags=re.IGNORECASE)
    if not match:
        return None, ""
    return int(match.group("distance")), match.group("suffix")


def numeric_token(text: str) -> int | None:
    """Extract the first integer token from text."""
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else None


def numeric_float(text: str) -> float | None:
    """Convert text to a float when possible."""
    try:
        return float(text)
    except ValueError:
        return None


def measurement_date(path: Path, root: Path) -> str:
    """Extract the measurement date for an IT trace path."""
    for part in path.relative_to(root).parts:
        if re.fullmatch(r"20\d{2} \d{2} \d{2}", part):
            return part.replace(" ", "-", 1).replace(" ", "-", 1)
    return ""


def spaced_units(text: str) -> str:
    """Insert a space between numeric values and unit suffixes."""
    return re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)


def trace_title(info: TraceInfo, fiber_names: FiberNameMap) -> str:
    """Build the display title for an integrated-time decay trace."""
    wavelength = f"{info.wavelength_label} nm" if info.wavelength_label else "unknown wavelength"
    return (
        f"{fiber_names.real_name(info.sample)}, {spaced_units(info.position)}, IT {spaced_units(info.window)} "
        f"({wavelength}, ex {spaced_units(info.excitation)}, {spaced_units(info.energy)})"
    )


def natural_position_key(position: str) -> tuple[float, str]:
    """Return a sortable key for fiber position tokens."""
    if position == "ENDcm":
        return (math.inf, position)
    match = re.match(r"(?P<distance>\d+)cm(?P<suffix>.*)$", position)
    if not match:
        return (math.inf, position)
    return (float(match.group("distance")), match.group("suffix"))


def all_time_windows(time_window: str) -> bool:
    """Return true when IT trace discovery should include every acquisition window."""
    return time_window.strip().lower() in {"", "*", "all", "any"}


def it_discovery_pattern(time_window: str) -> str:
    """Return the filename pattern used for integrated-time trace discovery."""
    return "IT_*.dat" if all_time_windows(time_window) else f"IT_*_{time_window}.dat"


def time_window_sort_key(window: str) -> tuple[int, float, str]:
    """Sort acquisition-window labels by duration when possible."""
    match = re.fullmatch(r"(?P<value>\d+(?:\.\d+)?)(?P<unit>ns|us|ms)", window, flags=re.IGNORECASE)
    if not match:
        return (1, math.inf, window)
    unit_scale = {"ns": 1.0, "us": 1000.0, "ms": 1_000_000.0}
    duration_ns = float(match.group("value")) * unit_scale[match.group("unit").lower()]
    return (0, duration_ns, window)


def discover_traces(raw_dir: Path, time_window: str) -> list[TraceInfo]:
    """Find integrated-time traces matching the requested acquisition-window filter."""
    traces: list[TraceInfo] = []
    include_all_windows = all_time_windows(time_window)
    for path in sorted(raw_dir.rglob(it_discovery_pattern(time_window))):
        match = FILENAME_RE.match(path.name)
        if not match:
            continue
        parts = match.groupdict()
        if not include_all_windows and parts["window"].lower() != time_window.lower():
            continue
        first_two = path.read_text(encoding="utf-8", errors="replace").splitlines()[:2]
        wavelength_label = ""
        if len(first_two) > 1:
            fields = first_two[1].split("\t")
            if len(fields) > 1:
                wavelength_label = fields[1]
        distance_cm, position_suffix = position_parts(parts["position"])
        traces.append(
            TraceInfo(
                path=path,
                relative_path=path.relative_to(raw_dir),
                sample=parts["sample"],
                position=parts["position"],
                distance_cm=distance_cm,
                position_suffix=position_suffix,
                excitation=parts["excitation"],
                excitation_nm=numeric_token(parts["excitation"]),
                energy=parts["energy"],
                energy_nj=numeric_token(parts["energy"]),
                window=parts["window"],
                wavelength_label=wavelength_label,
                measurement_date=measurement_date(path, raw_dir),
            )
        )
    return traces


def read_trace(path: Path) -> pd.DataFrame:
    """Load an integrated-time trace data file into a DataFrame."""
    return pd.read_csv(
        path,
        sep="\t",
        skiprows=2,
        names=["delay_ns", "counts", "normalized"],
        dtype=float,
    ).dropna()


def trace_label(info: TraceInfo) -> str:
    """Build a metadata label for a trace config entry."""
    parts = [info.sample, info.position, info.window]
    if info.wavelength_label:
        parts.append(f"{info.wavelength_label} nm")
    if info.measurement_date:
        parts.append(info.measurement_date)
    return ", ".join(parts)


def trace_note(info: TraceInfo) -> str:
    """Classify a trace for selection-config notes."""
    if info.position.upper() == "ENDcm":
        return "endpoint_condition"
    if info.position_suffix:
        return "replicate_or_reverse_orientation"
    return "default_on"


def write_trace_configs(traces: list[TraceInfo], config_dir: Path) -> None:
    """Write per-sample YAML configs for discovered IT traces."""
    config_dir.mkdir(parents=True, exist_ok=True)
    for sample in sorted({trace.sample for trace in traces}):
        sample_traces = [trace for trace in traces if trace.sample == sample]
        lines = [
            "# Edit include values, then rerun python -m lhcb_fibers_analysis.fit_it_decay.",
            "# include: true fits the trace and writes its diagnostic PNG/PDF.",
            f"group: {yaml_scalar(sample)}",
            f"title: {yaml_scalar(sample)}",
            "traces:",
        ]
        for trace in sample_traces:
            df = read_trace(trace.path)
            delay = df["delay_ns"].to_numpy(dtype=float)
            counts = df["counts"].to_numpy(dtype=float)
            wavelength_nm = numeric_float(trace.wavelength_label)
            lines.extend(
                [
                    f"  - include: {yaml_scalar(True)}",
                    f"    path: {yaml_scalar(trace.relative_path.as_posix())}",
                    f"    label: {yaml_scalar(trace_label(trace))}",
                    f"    sample: {yaml_scalar(trace.sample)}",
                    f"    position: {yaml_scalar(trace.position)}",
                    f"    distance_cm: {yaml_scalar(trace.distance_cm if trace.distance_cm is not None else '')}",
                    f"    position_suffix: {yaml_scalar(trace.position_suffix)}",
                    f"    excitation_nm: {yaml_scalar(trace.excitation_nm if trace.excitation_nm is not None else '')}",
                    f"    pulse_energy_nj: {yaml_scalar(trace.energy_nj if trace.energy_nj is not None else '')}",
                    f"    time_window: {yaml_scalar(trace.window)}",
                    f"    wavelength_label_nm: {yaml_scalar(wavelength_nm if wavelength_nm is not None else trace.wavelength_label)}",
                    f"    measurement_date: {yaml_scalar(trace.measurement_date)}",
                    f"    points: {yaml_scalar(len(df))}",
                    f"    delay_min_ns: {yaml_scalar(float(np.nanmin(delay)) if delay.size else '')}",
                    f"    delay_max_ns: {yaml_scalar(float(np.nanmax(delay)) if delay.size else '')}",
                    f"    counts_max: {yaml_scalar(float(np.nanmax(counts)) if counts.size else '')}",
                    f"    note: {yaml_scalar(trace_note(trace))}",
                ]
            )
        (config_dir / f"{sample}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def selected_traces_from_configs(
    configs: list[SectionConfig],
    traces_by_path: dict[str, TraceInfo],
) -> tuple[list[TraceInfo], set[Path], set[Path]]:
    """Resolve selected IT traces from YAML config entries."""
    configured_paths: set[Path] = set()
    selected_paths: set[Path] = set()
    selected: list[TraceInfo] = []

    for config in configs:
        for entry in config.entries:
            rel_path = normalize_config_path(str(entry.get("path", "")))
            trace = traces_by_path.get(rel_path)
            if trace is None:
                print(f"warning: IT trace config path not found, skipping: {rel_path}")
                continue
            configured_paths.add(trace.relative_path)
            if bool(entry.get("include", False)):
                selected.append(trace)
                selected_paths.add(trace.relative_path)

    return sorted(selected, key=lambda trace: (trace.sample, natural_position_key(trace.position), trace.relative_path.as_posix())), selected_paths, configured_paths


def exp_decay(x: np.ndarray, amplitude: float, tau_ns: float, baseline: float, x0: float) -> np.ndarray:
    """Evaluate a single-exponential decay with a constant baseline."""
    return amplitude * np.exp(-(x - x0) / tau_ns) + baseline


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
    if mask.size < run_length:
        return None
    run = 0
    for idx, flag in enumerate(mask):
        run = run + 1 if flag else 0
        if run >= run_length:
            return idx - run_length + 1
    return None


def fit_trace(
    info: TraceInfo,
    df: pd.DataFrame,
    config: FitConfig,
    fiber_names: FiberNameMap,
) -> dict[str, object]:
    """Fit one integrated-time trace or return a skipped result."""
    x = df["delay_ns"].to_numpy(dtype=float)
    y = df["counts"].to_numpy(dtype=float)
    n = len(y)
    result: dict[str, object] = {
        "sample": info.sample,
        "fiber_name": fiber_names.real_name(info.sample),
        "position": info.position,
        "excitation": info.excitation,
        "energy": info.energy,
        "time_window": info.window,
        "wavelength_label": info.wavelength_label,
        "source_file": info.relative_path.as_posix(),
    }

    if n < config.min_points:
        result.update(status="skipped", skip_reason="too_few_points", n_points=n)
        return result

    dt = float(np.median(np.diff(x)))
    smooth = gaussian_filter1d(y, sigma=config.smooth_sigma)
    baseline_seed = float(np.percentile(y, 5))
    low_values = y[y <= np.percentile(y, 25)]
    noise = max(robust_noise(low_values), math.sqrt(max(baseline_seed, 0.0) + 1.0))
    signal = smooth - baseline_seed
    peak_idx = int(np.argmax(signal))
    peak_height = float(signal[peak_idx])
    fit_warnings: list[str] = []

    result.update(
        n_points=n,
        baseline_seed_counts=baseline_seed,
        noise_counts=noise,
        dominant_peak_delay_ns=float(x[peak_idx]),
        dominant_peak_counts=float(y[peak_idx]),
    )

    if peak_height <= max(config.low_signal_sigma_threshold * noise, config.low_signal_sigma_threshold):
        if config.skip_low_signal:
            result.update(status="skipped", skip_reason="low_signal")
            return result
        fit_warnings.append("low_signal_fit_forced")

    min_distance = max(8, int(round(0.25 / abs(dt)))) if dt else 8
    peaks, props = find_peaks(
        smooth,
        height=baseline_seed + max(config.peak_height_fraction * peak_height, config.noise_sigma_threshold * noise),
        prominence=max(config.peak_prominence_fraction * peak_height, config.noise_sigma_threshold * noise, 1.0),
        distance=min_distance,
    )
    significant_peaks = [
        int(p)
        for p in peaks
        if (smooth[p] - baseline_seed)
        >= max(config.peak_height_fraction * peak_height, config.noise_sigma_threshold * noise)
    ]
    if peak_idx not in significant_peaks:
        significant_peaks.append(peak_idx)
        significant_peaks = sorted(set(significant_peaks))

    result.update(
        detected_peak_count=len(significant_peaks),
        detected_peak_delays_ns=";".join(f"{x[p]:.6g}" for p in significant_peaks),
    )

    if len(significant_peaks) > 1:
        if config.multiple_peak_strategy == "skip":
            result.update(status="skipped", skip_reason="multiple_significant_peaks")
            return result
        if config.multiple_peak_strategy == "first":
            peak_idx = int(significant_peaks[0])
            fit_warnings.append("multiple_peaks_fit_first")
        else:
            fit_warnings.append("multiple_peaks_fit_dominant")

    result.update(
        primary_peak_delay_ns=float(x[peak_idx]),
        primary_peak_counts=float(y[peak_idx]),
    )

    fit_start_idx = peak_idx
    fit_peak_height = max(float(smooth[fit_start_idx] - baseline_seed), 0.0)
    fit_limit_idx = n
    if config.fit_window_ns > 0:
        fit_window_end_ns = float(x[fit_start_idx] + config.fit_window_ns)
        fit_limit_idx = int(np.searchsorted(x, fit_window_end_ns, side="right"))
        fit_limit_idx = min(n, max(fit_start_idx + 1, fit_limit_idx))

    threshold = baseline_seed + max(0.05 * fit_peak_height, config.noise_sigma_threshold * noise, 1.0)
    below_noise = smooth[fit_start_idx:] <= threshold
    sustained_idx = first_sustained_true(below_noise, run_length=10)
    if sustained_idx is None:
        fit_end_idx = fit_limit_idx
    else:
        fit_end_idx = min(fit_limit_idx, fit_start_idx + sustained_idx)

    min_fit_points = config.min_fit_points
    if fit_end_idx - fit_start_idx < min_fit_points:
        fit_end_idx = min(fit_limit_idx, fit_start_idx + min_fit_points)

    if fit_end_idx - fit_start_idx < min_fit_points:
        result.update(
            status="skipped",
            skip_reason="too_few_decay_points",
            fit_window_limit_ns=float(config.fit_window_ns) if config.fit_window_ns > 0 else "",
        )
        return result

    x_fit = x[fit_start_idx:fit_end_idx]
    y_fit = y[fit_start_idx:fit_end_idx]
    x0 = float(x_fit[0])

    tail = y_fit[max(0, int(0.8 * len(y_fit))) :]
    baseline0 = max(0.0, float(np.percentile(tail, 20)))
    amplitude0 = max(float(y_fit[0] - baseline0), 1.0)
    target = baseline0 + amplitude0 / math.e
    below_e = np.where(y_fit <= target)[0]
    if below_e.size and below_e[0] > 0:
        tau0 = max(float(x_fit[below_e[0]] - x_fit[0]), 0.05)
    else:
        tau0 = 1.0

    try:
        popt, pcov = curve_fit(
            lambda x_data, amplitude, tau_ns, baseline: exp_decay(
                x_data, amplitude, tau_ns, baseline, x0
            ),
            x_fit,
            y_fit,
            p0=[amplitude0, tau0, baseline0],
            bounds=(
                [0.0, config.fit_tau_min_ns, 0.0],
                [np.inf, config.fit_tau_max_ns, max(float(np.max(y_fit)), 1.0e-9)],
            ),
            sigma=np.sqrt(np.maximum(y_fit, 1.0)),
            absolute_sigma=False,
            maxfev=20000,
        )
    except Exception as exc:  # noqa: BLE001
        result.update(status="skipped", skip_reason=f"fit_failed:{exc}")
        return result

    y_pred = exp_decay(x_fit, popt[0], popt[1], popt[2], x0)
    residual = y_fit - y_pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(residual**2)))
    perr = np.sqrt(np.diag(pcov)) if pcov.size else np.full(3, np.nan)

    warnings: list[str] = list(fit_warnings)
    if popt[1] <= config.fit_tau_min_ns * 1.034 or popt[1] >= config.fit_tau_max_ns * 0.98:
        warnings.append("tau_near_bound")
    if r_squared < 0.90:
        warnings.append("low_r_squared")
    if popt[2] > 0.50 * max(float(np.max(y_fit)), 1.0):
        warnings.append("high_baseline_fraction")

    result.update(
        status="fit",
        skip_reason="",
        tau_ns=float(popt[1]),
        tau_se_ns=float(perr[1]) if len(perr) > 1 else float("nan"),
        amplitude_counts=float(popt[0]),
        amplitude_se_counts=float(perr[0]) if len(perr) > 0 else float("nan"),
        baseline_counts=float(popt[2]),
        baseline_se_counts=float(perr[2]) if len(perr) > 2 else float("nan"),
        r_squared=r_squared,
        rmse_counts=rmse,
        fit_window_limit_ns=float(config.fit_window_ns) if config.fit_window_ns > 0 else "",
        fit_start_ns=float(x_fit[0]),
        fit_end_ns=float(x_fit[-1]),
        fit_duration_ns=float(x_fit[-1] - x_fit[0]),
        fit_points=len(x_fit),
        fit_warning=";".join(warnings),
    )
    return result


def plot_trace(
    info: TraceInfo,
    df: pd.DataFrame,
    result: dict[str, object],
    plots_root: Path,
    out_root: Path,
    config: FitConfig,
    fiber_names: FiberNameMap,
) -> tuple[str, str]:
    """Write PNG and PDF diagnostics for one trace fit."""
    sample_dir = plots_root / info.sample
    sample_dir.mkdir(parents=True, exist_ok=True)
    stem = info.path.stem
    png_path = sample_dir / f"{stem}_fit.png"
    pdf_path = sample_dir / f"{stem}_fit.pdf"

    x = df["delay_ns"].to_numpy(dtype=float)
    y = df["counts"].to_numpy(dtype=float)
    smooth = gaussian_filter1d(y, sigma=config.smooth_sigma)
    y_scale = max(float(np.max(y)), 1.0)

    set_publication_style()
    fig, (ax, ax_res) = plt.subplots(
        2,
        1,
        figsize=DOUBLE_COLUMN_WIDE,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.08},
    )

    ax.plot(x, y / y_scale, "o", ms=2.2, color=COLORS["black"], alpha=0.70, label="IT data")
    ax.plot(x, smooth / y_scale, "-", lw=1.05, color=COLORS["gray"], label="smoothed")
    ax.axvline(
        float(result.get("primary_peak_delay_ns", x[int(np.argmax(y))])),
        lw=0.9,
        ls=":",
        color=COLORS["blue"],
        label="fit-start peak",
    )
    dominant_peak_delay = result.get("dominant_peak_delay_ns")
    if dominant_peak_delay not in {None, ""}:
        dominant_peak_x = float(dominant_peak_delay)
        fit_peak_x = float(result.get("primary_peak_delay_ns", dominant_peak_x))
        if abs(dominant_peak_x - fit_peak_x) > 0.5 * max(abs(float(np.median(np.diff(x)))), 1.0e-12):
            ax.axvline(
                dominant_peak_x,
                lw=0.85,
                ls="--",
                color=COLORS["gray"],
                label="dominant peak",
            )

    peak_delays = str(result.get("detected_peak_delays_ns", "")).split(";")
    peak_label_used = False
    for peak_delay in peak_delays:
        if not peak_delay:
            continue
        peak_x = float(peak_delay)
        peak_idx = int(np.argmin(np.abs(x - peak_x)))
        ax.plot(
            x[peak_idx],
            y[peak_idx] / y_scale,
            "x",
            ms=5.2,
            mew=1.2,
            color=COLORS["vermillion"],
            label="detected peak" if not peak_label_used else "_nolegend_",
        )
        peak_label_used = True

    if result.get("status") == "fit":
        fit_start = float(result["fit_start_ns"])
        fit_end = float(result["fit_end_ns"])
        fit_mask = (x >= fit_start) & (x <= fit_end)
        x_fit = x[fit_mask]
        y_fit = y[fit_mask]
        x_dense = np.linspace(fit_start, fit_end, 500)
        y_dense = exp_decay(
            x_dense,
            float(result["amplitude_counts"]),
            float(result["tau_ns"]),
            float(result["baseline_counts"]),
            fit_start,
        )
        y_pred = exp_decay(
            x_fit,
            float(result["amplitude_counts"]),
            float(result["tau_ns"]),
            float(result["baseline_counts"]),
            fit_start,
        )
        ax.axvspan(fit_start, fit_end, color=COLORS["vermillion"], alpha=0.08, lw=0, label="_nolegend_")
        ax.plot(x_dense, y_dense / y_scale, "-", lw=1.45, color=COLORS["vermillion"], label="single-exponential fit")
        ax_res.plot(x_fit, (y_fit - y_pred) / y_scale, "o", ms=2.2, color=COLORS["black"], alpha=0.70)
        annotation = (
            rf"$\tau$ = {float(result['tau_ns']):.3g} $\pm$ {float(result['tau_se_ns']):.2g} ns"
            "\n"
            rf"$R^2$ = {float(result['r_squared']):.3f}"
            "\n"
            f"fit points = {int(result['fit_points'])}"
        )
    else:
        ax_res.plot(x, np.zeros_like(x), "-", lw=0.8, color=COLORS["gray"])
        annotation = f"Skipped: {str(result.get('skip_reason', '')).replace('_', ' ')}"

    ax.text(
        0.98,
        0.95,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=8.0,
        bbox={"boxstyle": "square,pad=0.25", "facecolor": "white", "edgecolor": "#C9CED6", "alpha": 0.92},
    )
    ax.set_ylabel("Normalized intensity (a.u.)")
    ax.set_title(trace_title(info, fiber_names), pad=5)
    legend = ax.legend(
        loc="lower left",
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="#C9CED6",
        handlelength=1.7,
        borderpad=0.35,
        labelspacing=0.35,
    )
    legend.get_frame().set_linewidth(0.8)
    apply_axes_style(ax, grid=True)

    ax_res.axhline(0, color=COLORS["gray"], lw=0.8)
    ax_res.set_xlabel("Delay (ns)")
    ax_res.set_ylabel("Residual (a.u.)")
    apply_axes_style(ax_res, grid=True)

    save_figure(fig, png_path)
    save_figure(fig, pdf_path)
    plt.close(fig)
    return (
        Path(os.path.relpath(png_path, out_root)).as_posix(),
        Path(os.path.relpath(pdf_path, out_root)).as_posix(),
    )


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write dictionaries to a CSV file with a fixed field order."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_matrices(results: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build decay-time and status matrices from fit results."""
    samples = sorted({str(row.get("fiber_name", row["sample"])) for row in results})
    include_window_in_column = len({str(row.get("time_window", "")) for row in results}) > 1

    def column_label(row: dict[str, object]) -> str:
        position = str(row["position"])
        if not include_window_in_column:
            return position
        return f"{position} {row.get('time_window', '')}".strip()

    def column_sort_key(label: str) -> tuple[tuple[float, str], tuple[int, float, str], str]:
        if not include_window_in_column:
            return (natural_position_key(label), (0, 0.0, ""), label)
        position, _, window = label.rpartition(" ")
        return (natural_position_key(position), time_window_sort_key(window), label)

    columns = sorted({column_label(row) for row in results}, key=column_sort_key)
    tau_matrix = pd.DataFrame(index=samples, columns=columns, dtype=object)
    status_matrix = pd.DataFrame(index=samples, columns=columns, dtype=object)
    tau_matrix.index.name = "fiber_name"
    status_matrix.index.name = "fiber_name"

    for row in results:
        sample = str(row.get("fiber_name", row["sample"]))
        position = column_label(row)
        if row.get("status") == "fit":
            tau_matrix.loc[sample, position] = float(row["tau_ns"])
            se = float(row.get("tau_se_ns", float("nan")))
            warning = str(row.get("fit_warning", ""))
            if math.isfinite(se):
                label = f"{float(row['tau_ns']):.4g} +/- {se:.2g} ns"
            else:
                label = f"{float(row['tau_ns']):.4g} ns"
            if warning:
                label += f" [{warning}]"
            status_matrix.loc[sample, position] = label
        else:
            tau_matrix.loc[sample, position] = ""
            status_matrix.loc[sample, position] = f"SKIP: {row.get('skip_reason', '')}"

    return tau_matrix.fillna(""), status_matrix.fillna("")


def finite_float(value: object) -> float | None:
    """Convert a value to a finite float when possible."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def is_unirradiated_result(row: dict[str, object]) -> bool:
    """Return true for Non-IR/unirradiated samples."""
    sample = str(row.get("sample", "")).lower()
    fiber_name = str(row.get("fiber_name", "")).lower()
    return sample.endswith("_noir") or "non-ir" in fiber_name


def linear_guide_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a linear guide fit through per-position mean values."""
    unique_x = np.array(sorted(set(float(value) for value in x)), dtype=float)
    if unique_x.size == 0:
        return x, y
    mean_y = np.array([float(np.mean(y[np.isclose(x, value)])) for value in unique_x], dtype=float)
    if unique_x.size < 2:
        return unique_x, mean_y

    x_dense = np.linspace(float(unique_x[0]), float(unique_x[-1]), 120)
    slope, intercept = np.polyfit(unique_x, mean_y, deg=1)
    y_dense = slope * x_dense + intercept
    return x_dense, y_dense


def plot_decay_time_scatter(
    results: list[dict[str, object]],
    out_png: Path,
    out_pdf: Path,
    excluded_time_windows: tuple[str, ...] = (),
    guide: str = "linear",
    guide_alpha: float = 0.32,
) -> None:
    """Plot fitted decay time versus position with error bars and guide lines."""
    excluded_windows = {window.lower() for window in excluded_time_windows}
    fit_rows = [
        row
        for row in results
        if row.get("status") == "fit"
        and finite_float(row.get("tau_ns")) is not None
        and position_parts(str(row.get("position", "")))[0] is not None
        and str(row.get("time_window", "")).lower() not in excluded_windows
    ]
    if not fit_rows:
        return

    samples = sorted(
        {str(row.get("sample", "")) for row in fit_rows},
        key=lambda sample: str(next(row.get("fiber_name", sample) for row in fit_rows if row.get("sample") == sample)),
    )
    sample_style = {
        sample: (LINE_COLORS[index % len(LINE_COLORS)], MARKERS[index % len(MARKERS)])
        for index, sample in enumerate(samples)
    }

    set_publication_style()
    fig, ax = plt.subplots(figsize=DOUBLE_COLUMN_WIDE, constrained_layout=True)
    sample_handles: dict[str, object] = {}

    for sample in samples:
        group_rows = [
            row for row in fit_rows if str(row.get("sample", "")) == sample
        ]
        if not group_rows:
            continue
        group_rows = sorted(
            group_rows,
            key=lambda row: (
                position_parts(str(row.get("position", "")))[0] or math.inf,
                time_window_sort_key(str(row.get("time_window", ""))),
                str(row.get("source_file", "")),
            ),
        )
        x = np.array(
            [float(position_parts(str(row["position"]))[0] or 0) for row in group_rows],
            dtype=float,
        )
        y = np.array([float(row["tau_ns"]) for row in group_rows], dtype=float)
        yerr = np.array(
            [
                finite_float(row.get("tau_se_ns")) if finite_float(row.get("tau_se_ns")) is not None else 0.0
                for row in group_rows
            ],
            dtype=float,
        )
        color, marker = sample_style[sample]
        fill = color if is_unirradiated_result(group_rows[0]) else "white"
        label = str(group_rows[0].get("fiber_name", sample))
        if guide == "linear":
            x_guide, y_guide = linear_guide_xy(x, y)
            ax.plot(x_guide, y_guide, "-", color=color, linewidth=0.95, alpha=guide_alpha, zorder=1)
        elif guide == "point_to_point":
            ax.plot(x, y, "-", color=color, linewidth=0.85, alpha=guide_alpha, zorder=1)
        container = ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=marker,
            ms=MARKER_SIZE,
            linestyle="none",
            color=color,
            ecolor=color,
            elinewidth=0.75,
            capsize=2.0,
            markerfacecolor=fill,
            markeredgecolor=color,
            markeredgewidth=0.85,
            label=label,
            zorder=2,
        )
        sample_handles.setdefault(label, container)

    ax.set_xlabel("Position (cm)")
    ax.set_ylabel(r"Decay time $\tau$ (ns)")
    apply_axes_style(ax, grid=True)

    fig.legend(
        list(sample_handles.values()),
        list(sample_handles.keys()),
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncols=1,
        title="filled: Non-IR\nopen: IR",
        borderaxespad=0.0,
        columnspacing=1.0,
        handlelength=1.6,
    )

    save_figure(fig, out_png)
    save_figure(fig, out_pdf)
    plt.close(fig)


def write_method_note(
    path: Path,
    results: list[dict[str, object]],
    time_window: str,
    fit_all_discovered_traces: bool,
    fit_window_ns: float,
    skip_low_signal: bool,
    multiple_peak_strategy: str,
    scatter_excluded_time_windows: tuple[str, ...],
    scatter_guide: str,
    scatter_guide_alpha: float,
) -> None:
    """Write the method summary README for IT decay fits."""
    fit_count = sum(1 for row in results if row.get("status") == "fit")
    skip_count = len(results) - fit_count
    pattern = it_discovery_pattern(time_window)
    title = "IT All-Window Decay Fits" if all_time_windows(time_window) else "IT 10 ns Decay Fits"
    if fit_all_discovered_traces:
        scope = f"Analysis scope: every discovered `{pattern}` measurement under the raw-data folder."
    else:
        scope = "Analysis scope: traces selected by `include: true` in the trace YAML files."
    if all_time_windows(time_window):
        inputs = "`IT_*.dat` integrated-time traces. Streak-map `.img` files, `T_*.dat` band traces, and derived files are excluded."
        summary_window = "all acquisition-window"
    else:
        inputs = f"`IT_*_{time_window}.dat` only. Streak-map `.img` files, `T_*.dat` band traces, and other `IT_*.dat` time windows are excluded."
        summary_window = time_window
    if fit_window_ns > 0:
        fit_window_rule = f"Fit window: each fit starts at the selected peak and uses at most {fit_window_ns:g} ns of data after that peak."
    else:
        fit_window_rule = "Fit window: each fit starts at the selected peak and ends at the automatic low-signal cutoff."
    low_signal_rule = (
        "Low-signal traces are skipped before fitting."
        if skip_low_signal
        else "Low-signal traces are still fit and marked with a warning."
    )
    if multiple_peak_strategy == "skip":
        multiple_peak_rule = "Traces with more than one significant temporal peak are not fit."
    elif multiple_peak_strategy == "first":
        multiple_peak_rule = "When more than one significant temporal peak is detected, the fit starts at the first detected peak."
    else:
        multiple_peak_rule = "When more than one significant temporal peak is detected, the fit starts at the dominant peak."
    scatter_note = ""
    if scatter_excluded_time_windows:
        excluded = ", ".join(spaced_units(window) for window in scatter_excluded_time_windows)
        scatter_note = f" The scatter plot excludes {excluded} acquisition-window traces."
    if scatter_guide == "linear":
        scatter_note += f" Guide lines are per-sample linear fits drawn with alpha {scatter_guide_alpha:g}."
    elif scatter_guide == "point_to_point":
        scatter_note += f" Guide lines connect points in sample order with alpha {scatter_guide_alpha:g}."
    elif scatter_guide == "none":
        scatter_note += " Guide lines are disabled."
    method = f"""# {title}

Inputs: {inputs}

{scope}

{fit_window_rule}

{low_signal_rule}

{multiple_peak_rule}

Fit rule: each trace is smoothed only for peak detection, then a single exponential plus constant baseline is fit to the selected raw-count decay interval:

`I(t) = A exp(-(t - t_start) / tau) + B`

The peak detector requires a secondary peak to exceed both a relative threshold and a noise/prominence threshold, so isolated count noise is not treated as a separate decay component.

Outputs:

- `fit_inventory.csv`: one row per input trace, including fit status, tau, standard error, R^2, fit window, peak count, and plot paths.
- `decay_times_matrix.csv`: samples as rows, positions as columns, fitted decay times in ns.
- `decay_times_matrix_with_status.csv`: same layout with uncertainty or skip reason in each cell.
- `decay_times_scatter.png` and `.pdf`: fitted decay times versus position with standard-error bars and guide lines.{scatter_note}
- `plots/<sample>/`: PNG and PDF fit-quality plots for every input trace.

Summary: {fit_count} fitted, {skip_count} skipped, {len(results)} total {summary_window} IT traces.
"""
    path.write_text(method, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    """Run the integrated-time decay fitting command-line interface."""
    parser = argparse.ArgumentParser(description="Fit single-exponential decays for IT_*.dat traces.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_IT_DECAY_CONFIG)
    parser.add_argument("--trace-config-dir", type=Path, default=None)
    parser.add_argument("--fiber-names-config", type=Path, default=DEFAULT_FIBER_NAMES_CONFIG)
    parser.add_argument("--out-subdir", type=Path, default=None)
    parser.add_argument("--time-window", default=None)
    parser.add_argument("--refresh-configs", action="store_true", help="Rewrite per-sample IT trace configs.")
    args = parser.parse_args(argv)

    raw_dir = resolve_path(args.raw_dir)
    results_dir = resolve_path(args.results_dir)
    config_path = resolve_path(args.config)
    fiber_names_config = resolve_path(args.fiber_names_config)
    fiber_names = read_fiber_name_map(fiber_names_config)
    config = read_fit_config(config_path)
    trace_config_dir = (
        resolve_path(args.trace_config_dir)
        if args.trace_config_dir is not None
        else relative_config_dir(config_path, config.trace_config_dir)
    )
    out_subdir = args.out_subdir or Path(config.out_subdir)
    time_window = args.time_window or config.time_window
    out_root = (results_dir / out_subdir).resolve()
    plots_root = out_root / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    discovered_traces = discover_traces(raw_dir, time_window)
    if not discovered_traces:
        raise SystemExit(f"No {it_discovery_pattern(time_window)} files found under {raw_dir}")

    if args.refresh_configs or not configs_exist(trace_config_dir):
        write_trace_configs(discovered_traces, trace_config_dir)

    traces_by_path = {trace.relative_path.as_posix(): trace for trace in discovered_traces}
    configured_trace_paths: set[Path] = set()
    if configs_exist(trace_config_dir):
        trace_configs = read_section_configs(trace_config_dir, "traces")
        _, _, configured_trace_paths = selected_traces_from_configs(trace_configs, traces_by_path)
    elif not config.fit_all_discovered_traces:
        raise SystemExit(f"No IT trace YAML configs found in {trace_config_dir}")

    if config.fit_all_discovered_traces:
        traces = sorted(
            discovered_traces,
            key=lambda trace: (trace.sample, natural_position_key(trace.position), trace.relative_path.as_posix()),
        )
        selected_trace_paths = {trace.relative_path for trace in traces}
    else:
        trace_configs = read_section_configs(trace_config_dir, "traces")
        traces, selected_trace_paths, configured_trace_paths = selected_traces_from_configs(
            trace_configs,
            traces_by_path,
        )
        if not traces:
            raise SystemExit(f"No IT traces are selected in {trace_config_dir}")

    results: list[dict[str, object]] = []
    for info in traces:
        df = read_trace(info.path)
        result = fit_trace(info, df, config, fiber_names)
        png_path, pdf_path = plot_trace(info, df, result, plots_root, out_root, config, fiber_names)
        result["plot_png"] = png_path
        result["plot_pdf"] = pdf_path
        results.append(result)

    fields = [
        "sample",
        "fiber_name",
        "position",
        "excitation",
        "energy",
        "time_window",
        "wavelength_label",
        "status",
        "skip_reason",
        "tau_ns",
        "tau_se_ns",
        "amplitude_counts",
        "amplitude_se_counts",
        "baseline_counts",
        "baseline_se_counts",
        "r_squared",
        "rmse_counts",
        "fit_window_limit_ns",
        "fit_start_ns",
        "fit_end_ns",
        "fit_duration_ns",
        "fit_points",
        "fit_warning",
        "detected_peak_count",
        "detected_peak_delays_ns",
        "dominant_peak_delay_ns",
        "dominant_peak_counts",
        "primary_peak_delay_ns",
        "primary_peak_counts",
        "baseline_seed_counts",
        "noise_counts",
        "n_points",
        "source_file",
        "plot_png",
        "plot_pdf",
    ]
    write_csv(out_root / "fit_inventory.csv", results, fields)

    skipped = [row for row in results if row.get("status") != "fit"]
    write_csv(out_root / "skipped_traces.csv", skipped, fields)

    tau_matrix, status_matrix = build_matrices(results)
    tau_matrix.to_csv(out_root / "decay_times_matrix.csv", float_format="%.6g")
    status_matrix.to_csv(out_root / "decay_times_matrix_with_status.csv")
    plot_decay_time_scatter(
        results,
        out_root / "decay_times_scatter.png",
        out_root / "decay_times_scatter.pdf",
        config.scatter_excluded_time_windows,
        config.scatter_guide,
        config.scatter_guide_alpha,
    )
    write_method_note(
        out_root / "README.md",
        results,
        time_window,
        config.fit_all_discovered_traces,
        config.fit_window_ns,
        config.skip_low_signal,
        config.multiple_peak_strategy,
        config.scatter_excluded_time_windows,
        config.scatter_guide,
        config.scatter_guide_alpha,
    )

    fit_count = sum(1 for row in results if row.get("status") == "fit")
    discovery_label = "all acquisition-window" if all_time_windows(time_window) else time_window
    print(f"Discovered {len(discovered_traces)} {discovery_label} IT traces; fitting {len(selected_trace_paths)}.")
    if not config.fit_all_discovered_traces:
        print(f"Configured {len(configured_trace_paths)} {discovery_label} IT traces; selected {len(selected_trace_paths)}.")
    print(f"Processed {len(results)} {discovery_label} IT traces: {fit_count} fit, {len(results) - fit_count} skipped.")
    print(f"Output: {out_root}")
    print(f"config: {config_path}")
    print(f"fiber names: {fiber_names_config}")
    print(f"trace configs: {trace_config_dir}")


if __name__ == "__main__":
    main()
