from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from tqdm import tqdm

from .batch_carpet_wavelength_cuts import sigmoid_rise
from .carpet_wavelength_cuts import exp_decay
from .paths import DEFAULT_RESULTS_DIR, resolve_path
from .plot_style import COLORS, apply_axes_style, save_figure, set_publication_style


DEFAULT_CUTS_SUBDIR = "carpet_wavelength_cuts_20nm_txt"
DEFAULT_OUT_SUBDIR = "wavelength_cut_fit_results_2ns_rise_10ns_decay"
DECAY_BACKGROUND_PIXELS = 8
DECAY_MODEL_LABEL = "Model: y = A exp[-(t-t0)/tau] + B_bg"
DOUBLE_DECAY_MODEL_LABEL = "Model: y = A1 exp[-dt/tau1] + A2 exp[-dt/tau2], B = 0"
DECAY_SELECTION_SUFFIX = "_decay_time_10ns_by_position_interval_selection.txt"
DECAY_FIT_END_OVERRIDES_NS = {
    "bcf10_noir/2026_05_05__bcf10_noir_100cm_ex360nm_10nJ_10ns": 6.5,
    "bcf10_noir/2026_04_20__bcf_10__bcf10_noir_120cm_ex360nm_10nJ_10ns": 4.5,
    "bcf10_noir/2026_04_20__bcf_10__bcf10_noir_160cm_ex360nm_10nJ_10ns": 4.5,
    "bcf10_noir/2026_05_05__bcf10_noir_200cm_ex360nm_10nJ_10ns": 3.5,
    "bcf10_noir/2026_04_20__bcf_10__bcf10_noir_190cm_ex360nm_10nJ_10ns": 3.5,
}


def format_nm(value: object) -> str:
    """Format a wavelength edge for stable filenames and labels."""
    numeric = float(value)
    return str(int(round(numeric))) if numeric.is_integer() else f"{numeric:g}".replace(".", "p")


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


def write_frame_tsv(path: Path, frame: pd.DataFrame) -> None:
    """Write a dataframe as a tab-delimited text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, float_format="%.10g")


def safe_scan_folder(output_folder: str) -> Path:
    """Return a path that preserves sample/scan nesting from the cut inventory."""
    return Path(output_folder)


def source_time_window(source_file: str) -> str:
    """Extract the acquisition time-window token from one source filename."""
    match = re.search(r"_(\d+(?:ns|us|ms))(?:_|\.|$)", source_file.lower())
    return match.group(1) if match else ""


def source_position(source_file: str) -> str:
    """Extract the fiber-position token from one source filename."""
    match = re.search(r"_(endcm|\d+cm[a-z0-9]*)(?:_|\.|$)", Path(source_file).stem, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def interval_label(row: pd.Series) -> str:
    """Return a compact wavelength interval label."""
    return f"{format_nm(row['band_min_nm'])}-{format_nm(row['band_max_nm'])} nm"


def interval_sort_key(interval: str) -> tuple[float, float, str]:
    """Sort interval labels by numeric wavelength bounds."""
    match = re.match(r"\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", str(interval))
    if not match:
        return (math.inf, math.inf, str(interval))
    return (float(match.group(1)), float(match.group(2)), str(interval))


def position_sort_key(position: str) -> tuple[int, float, str]:
    """Sort positions numerically, leaving endpoint conditions last."""
    text = str(position)
    if text.lower() == "endcm":
        return (1, math.inf, text)
    match = re.match(r"(\d+)cm(.*)", text, flags=re.IGNORECASE)
    if match:
        return (0, float(match.group(1)), match.group(2))
    return (2, math.inf, text)


def profile_column(row: pd.Series) -> str:
    """Return the profile table column for one wavelength band."""
    return f"mean_counts_{format_nm(row['band_min_nm'])}_{format_nm(row['band_max_nm'])}nm"


def finite_float(row: pd.Series, key: str) -> float:
    """Read one finite float from a pandas row."""
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} is not finite")
    return value


def optional_float(row: pd.Series, key: str) -> float:
    """Read one float-like value, returning NaN when it is missing or non-finite."""
    try:
        value = float(row.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def plus_minus(value: float) -> str:
    """Format a one-sigma fit uncertainty for plot labels."""
    return f"+/- {value:.3g} ns" if math.isfinite(value) else "+/- n/a"


def initial_background_counts(counts: np.ndarray, pixel_count: int = DECAY_BACKGROUND_PIXELS) -> float:
    """Estimate fixed background from the first finite trace pixels."""
    first_pixels = counts[: max(1, pixel_count)]
    finite = first_pixels[np.isfinite(first_pixels)]
    if finite.size == 0:
        finite = counts[np.isfinite(counts)]
    if finite.size == 0:
        return 0.0
    return max(0.0, float(np.nanmedian(finite)))


def manual_fit_end_index(time_ns: np.ndarray, fit_start_idx: int, fit_end_ns: float) -> int | None:
    """Return an exclusive fit-end index for one manual time limit."""
    end_idx = int(np.searchsorted(time_ns, fit_end_ns, side="right"))
    return end_idx if end_idx > fit_start_idx + 2 else None


def progress_bar(iterable, **kwargs):
    """Return a Codex-visible progress bar for long manual runs."""
    return tqdm(iterable, file=sys.stdout, dynamic_ncols=True, ascii=True, **kwargs)


def double_exp_decay(
    time_ns: np.ndarray,
    amplitude_fast: float,
    tau_fast_ns: float,
    amplitude_slow: float,
    tau_slow_ns: float,
    t0_ns: float,
) -> np.ndarray:
    """Evaluate a zero-baseline double-exponential decay."""
    dt = time_ns - t0_ns
    return amplitude_fast * np.exp(-dt / tau_fast_ns) + amplitude_slow * np.exp(-dt / tau_slow_ns)


def plot_decay_fit(path: Path, time_ns: np.ndarray, counts: np.ndarray, row: pd.Series, title: str) -> None:
    """Plot one 10 ns decay fit for visual inspection."""
    fit_start = finite_float(row, "fit_start_ns")
    fit_end = finite_float(row, "fit_end_ns")
    tau = finite_float(row, "tau_ns")
    tau_se = optional_float(row, "tau_se_ns")
    amplitude = finite_float(row, "amplitude_counts")
    baseline = finite_float(row, "baseline_counts")
    x_dense = np.linspace(fit_start, fit_end, 400)
    y_dense = exp_decay(x_dense, amplitude, tau, baseline, fit_start)
    positive_values = np.concatenate(
        [
            counts[np.isfinite(counts) & (counts > 0)],
            y_dense[np.isfinite(y_dense) & (y_dense > 0)],
            np.array([baseline]) if math.isfinite(baseline) and baseline > 0 else np.array([], dtype=float),
        ]
    )

    set_publication_style(base_font_size=8.2)
    fig, ax = plt.subplots(figsize=(4.8, 3.25), constrained_layout=True)
    ax.plot(time_ns, counts, "-", color=COLORS["gray"], lw=0.8, alpha=0.75, label="raw cut")
    ax.plot(time_ns, counts, "o", color=COLORS["black"], ms=1.8, alpha=0.45, label="_nolegend_")
    ax.axvline(float(row["peak_time_ns"]), color=COLORS["blue"], ls=":", lw=0.9, label="peak")
    ax.axvspan(fit_start, fit_end, color=COLORS["vermillion"], alpha=0.08, lw=0)
    ax.plot(x_dense, y_dense, color=COLORS["vermillion"], lw=1.45, label=f"decay fit ({plus_minus(tau_se)})")
    if math.isfinite(baseline) and baseline > 0:
        ax.axhline(baseline, color=COLORS["teal"], ls="--", lw=1.0, label=f"background ({baseline:.3g})")
    ax.set_yscale("log", nonpositive="clip")
    if positive_values.size:
        ax.set_ylim(max(float(np.min(positive_values)) * 0.8, 1.0e-6), float(np.max(positive_values)) * 1.25)
    ax.text(
        0.98,
        0.95,
        f"{DECAY_MODEL_LABEL}\ntau = {tau:.3g} {plus_minus(tau_se)}\nR^2 = {float(row['r_squared']):.3f}\npoints = {int(row['fit_points'])}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=7.0,
        bbox={"boxstyle": "square,pad=0.25", "facecolor": "white", "edgecolor": "#C9CED6", "alpha": 0.92},
    )
    ax.set_title(title, pad=4)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Mean counts")
    ax.legend(loc="lower left", frameon=True, framealpha=0.88, facecolor="white", edgecolor="#C9CED6")
    apply_axes_style(ax, grid=True)
    save_figure(fig, path, dpi=240)
    plt.close(fig)


def plot_double_decay_fit(path: Path, time_ns: np.ndarray, counts: np.ndarray, row: pd.Series, title: str) -> None:
    """Plot one selected 10 ns double-exponential decay fit for visual inspection."""
    fit_start = finite_float(row, "fit_start_ns")
    fit_end = finite_float(row, "fit_end_ns")
    tau_fast = finite_float(row, "tau_fast_ns")
    tau_fast_se = optional_float(row, "tau_fast_se_ns")
    tau_slow = finite_float(row, "tau_slow_ns")
    tau_slow_se = optional_float(row, "tau_slow_se_ns")
    amplitude_fast = finite_float(row, "amplitude_fast_counts")
    amplitude_slow = finite_float(row, "amplitude_slow_counts")
    x_dense = np.linspace(fit_start, fit_end, 400)
    y_fast = exp_decay(x_dense, amplitude_fast, tau_fast, 0.0, fit_start)
    y_slow = exp_decay(x_dense, amplitude_slow, tau_slow, 0.0, fit_start)
    y_dense = y_fast + y_slow
    positive_values = np.concatenate(
        [
            counts[np.isfinite(counts) & (counts > 0)],
            y_dense[np.isfinite(y_dense) & (y_dense > 0)],
        ]
    )

    set_publication_style(base_font_size=8.2)
    fig, ax = plt.subplots(figsize=(4.8, 3.25), constrained_layout=True)
    ax.plot(time_ns, counts, "-", color=COLORS["gray"], lw=0.8, alpha=0.75, label="raw cut")
    ax.plot(time_ns, counts, "o", color=COLORS["black"], ms=1.8, alpha=0.45, label="_nolegend_")
    ax.axvline(float(row["peak_time_ns"]), color=COLORS["blue"], ls=":", lw=0.9, label="peak")
    ax.axvspan(fit_start, fit_end, color=COLORS["vermillion"], alpha=0.08, lw=0)
    ax.plot(x_dense, y_dense, color=COLORS["vermillion"], lw=1.55, label="2-exp fit")
    ax.plot(x_dense, y_fast, color=COLORS["blue"], lw=0.95, ls="--", alpha=0.9, label="fast component")
    ax.plot(x_dense, y_slow, color=COLORS["purple"], lw=0.95, ls="--", alpha=0.9, label="slow component")
    ax.set_yscale("log", nonpositive="clip")
    if positive_values.size:
        ax.set_ylim(max(float(np.min(positive_values)) * 0.8, 1.0e-6), float(np.max(positive_values)) * 1.25)
    ax.text(
        0.98,
        0.95,
        (
            f"{DOUBLE_DECAY_MODEL_LABEL}\n"
            f"tau1 = {tau_fast:.3g} {plus_minus(tau_fast_se)}\n"
            f"tau2 = {tau_slow:.3g} {plus_minus(tau_slow_se)}\n"
            f"R^2 = {float(row['r_squared']):.3f}, points = {int(row['fit_points'])}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=6.8,
        bbox={"boxstyle": "square,pad=0.25", "facecolor": "white", "edgecolor": "#C9CED6", "alpha": 0.92},
    )
    ax.set_title(title, pad=4)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Mean counts")
    ax.legend(loc="lower left", frameon=True, framealpha=0.88, facecolor="white", edgecolor="#C9CED6")
    apply_axes_style(ax, grid=True)
    save_figure(fig, path, dpi=240)
    plt.close(fig)


def plot_rise_fit(path: Path, time_ns: np.ndarray, counts: np.ndarray, row: pd.Series, title: str) -> None:
    """Plot one 2 ns rise-time fit for visual inspection."""
    fit_start = finite_float(row, "fit_start_ns")
    fit_end = finite_float(row, "fit_end_ns")
    sigmoid_k = finite_float(row, "sigmoid_k_ns")
    rise_time_se = optional_float(row, "fitted_rise_time_10_90_se_ns")
    amplitude = finite_float(row, "amplitude_counts")
    baseline = finite_float(row, "baseline_counts")
    midpoint = finite_float(row, "midpoint_time_ns")
    x_dense = np.linspace(fit_start, fit_end, 400)
    y_dense = sigmoid_rise(x_dense, amplitude, sigmoid_k, baseline, midpoint)

    set_publication_style(base_font_size=8.2)
    fig, ax = plt.subplots(figsize=(4.8, 3.25), constrained_layout=True)
    ax.plot(time_ns, counts, "-", color=COLORS["gray"], lw=0.8, alpha=0.75, label="raw cut")
    ax.plot(time_ns, counts, "o", color=COLORS["black"], ms=1.8, alpha=0.45, label="_nolegend_")
    ax.axvline(float(row["peak_time_ns"]), color=COLORS["blue"], ls=":", lw=0.9, label="peak")
    threshold_10 = float(row["threshold_10_time_ns"])
    threshold_90 = float(row["threshold_90_time_ns"])
    if math.isfinite(threshold_10):
        ax.axvline(threshold_10, color=COLORS["teal"], ls=":", lw=0.85, label="10/90%")
    if math.isfinite(threshold_90):
        ax.axvline(threshold_90, color=COLORS["teal"], ls=":", lw=0.85, label="_nolegend_")
    ax.axvspan(fit_start, fit_end, color=COLORS["vermillion"], alpha=0.08, lw=0)
    ax.plot(x_dense, y_dense, color=COLORS["vermillion"], lw=1.45, label=f"rise fit ({plus_minus(rise_time_se)})")
    rise_time = float(row["fitted_rise_time_10_90_ns"])
    ax.text(
        0.98,
        0.95,
        f"10-90% = {rise_time:.3g} {plus_minus(rise_time_se)}\nk = {sigmoid_k:.3g} ns\nR^2 = {float(row['r_squared']):.3f}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=7.5,
        bbox={"boxstyle": "square,pad=0.25", "facecolor": "white", "edgecolor": "#C9CED6", "alpha": 0.92},
    )
    ax.set_title(title, pad=4)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Mean counts")
    ax.legend(loc="lower left", frameon=True, framealpha=0.88, facecolor="white", edgecolor="#C9CED6")
    apply_axes_style(ax, grid=True)
    save_figure(fig, path, dpi=240)
    plt.close(fig)


def read_profiles(scan_dir: Path) -> pd.DataFrame:
    """Read raw cut profiles for one scan output."""
    return pd.read_csv(scan_dir / "profiles_by_band.txt", sep="\t")


def successful_rows(path: Path) -> pd.DataFrame:
    """Read successful fit rows from one summary table."""
    frame = pd.read_csv(path, sep="\t")
    return frame[frame["status"] == "fit"].copy()


def all_rows(path: Path) -> pd.DataFrame:
    """Read every fit-summary row from one scan table."""
    return pd.read_csv(path, sep="\t")


def selection_value_enabled(value: object) -> bool:
    """Return whether one manual selection matrix cell keeps a point."""
    text = str(value).strip().lower()
    if text in {"", "0", "false", "no", "n", "skip", "exclude"}:
        return False
    if text in {"true", "yes", "y", "keep", "include", "x"}:
        return True
    try:
        return float(text) > 0.0
    except ValueError:
        return False


def read_decay_manual_selection(selection_dir: Path) -> set[tuple[str, str, str]]:
    """Read selected sample/position/interval cells from decay manual-selection matrices."""
    if not selection_dir.exists():
        raise SystemExit(f"Missing decay manual-selection folder: {selection_dir}")
    selected: set[tuple[str, str, str]] = set()
    for path in sorted(selection_dir.glob(f"*{DECAY_SELECTION_SUFFIX}")):
        sample = path.name[: -len(DECAY_SELECTION_SUFFIX)]
        matrix = pd.read_csv(path, sep="\t", dtype=str).fillna("")
        for _, row in matrix.iterrows():
            position = str(row.get("position", "")).strip()
            if not position:
                continue
            for interval in matrix.columns:
                if interval == "position":
                    continue
                if selection_value_enabled(row.get(interval, "")):
                    selected.add((sample, position, interval))
    if not selected:
        raise SystemExit(f"No enabled decay manual-selection cells found in {selection_dir}")
    return selected


def first_sustained_true(mask: np.ndarray, run_length: int) -> int | None:
    """Return the first index where a boolean mask stays true for a run."""
    run = 0
    for idx, flag in enumerate(mask):
        run = run + 1 if bool(flag) else 0
        if run >= run_length:
            return idx - run_length + 1
    return None


def forced_decay_fit_row(
    row: pd.Series,
    time_ns: np.ndarray,
    counts: np.ndarray,
    output_folder: str,
) -> pd.Series:
    """Fit one decay profile even when the batch diagnostic marked it as low-signal."""
    y = counts.astype(float)
    smooth = gaussian_filter1d(y, sigma=2.0)
    background = initial_background_counts(y)
    signal = smooth - background
    peak_idx = int(np.nanargmax(signal))
    peak_height = float(signal[peak_idx])
    peak_time = float(time_ns[peak_idx])
    peak_counts = float(y[peak_idx])
    fit_start_time = peak_time + 0.05
    start_candidates = np.where(time_ns >= fit_start_time)[0]
    fit_start_idx = int(start_candidates[0]) if start_candidates.size else min(peak_idx + 1, len(time_ns) - 1)

    threshold = background + 0.05 * max(peak_height, 0.0)
    below = smooth[fit_start_idx:] <= threshold
    sustained_idx = first_sustained_true(below, run_length=8)
    fit_end_idx = len(time_ns) if sustained_idx is None else fit_start_idx + sustained_idx
    fit_end_rule = "forced_end_fraction" if sustained_idx is not None else "forced_trace_end"
    manual_fit_end_ns = DECAY_FIT_END_OVERRIDES_NS.get(output_folder)
    if manual_fit_end_ns is not None:
        manual_end_idx = manual_fit_end_index(time_ns, fit_start_idx, manual_fit_end_ns)
        if manual_end_idx is not None:
            fit_end_idx = min(fit_end_idx, manual_end_idx)
            fit_end_rule = f"manual_end_{format_nm(manual_fit_end_ns)}ns"
    min_fit_points = 20
    if fit_end_idx - fit_start_idx < min_fit_points:
        fit_end_idx = min(len(time_ns), fit_start_idx + min_fit_points)
        fit_end_rule += ";min_points_extended"
    if fit_end_idx - fit_start_idx < 3:
        fit_start_idx = max(0, min(fit_start_idx, len(time_ns) - 3))
        fit_end_idx = len(time_ns)
        fit_end_rule += ";trace_extended"

    x_fit = time_ns[fit_start_idx:fit_end_idx]
    y_fit = y[fit_start_idx:fit_end_idx]
    t0 = float(x_fit[0])
    amplitude0 = max(float(y_fit[0] - background), float(np.nanmax(y_fit) - background), 1.0e-6)
    duration = max(float(x_fit[-1] - x_fit[0]), 0.03)
    tau0 = min(max(duration / 3.0, 0.03), 200.0)

    fit_status = "fit"
    fit_note = ""
    try:
        popt, pcov = curve_fit(
            lambda x_data, amplitude, tau_ns: exp_decay(x_data, amplitude, tau_ns, background, t0),
            x_fit,
            y_fit,
            p0=[amplitude0, tau0],
            bounds=([0.0, 0.03], [np.inf, 200.0]),
            sigma=np.sqrt(np.maximum(y_fit, 1.0)),
            absolute_sigma=False,
            maxfev=20000,
        )
    except Exception as exc:  # noqa: BLE001
        fit_status = "fallback"
        fit_note = f"curve_fit_failed:{exc}"
        popt = np.array([amplitude0, tau0], dtype=float)
        pcov = np.full((2, 2), np.nan)

    y_pred = exp_decay(x_fit, float(popt[0]), float(popt[1]), background, t0)
    residual = y_fit - y_pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(residual**2)))
    perr = np.sqrt(np.diag(pcov)) if pcov.size else np.full(3, np.nan)

    forced = row.copy()
    forced["original_batch_status"] = row.get("status", "")
    forced["original_batch_reason"] = row.get("reason", "")
    forced["forced_decay_status"] = fit_status
    forced["forced_decay_note"] = fit_note
    forced["tau_ns"] = float(popt[1])
    forced["tau_se_ns"] = float(perr[1]) if len(perr) > 1 else float("nan")
    forced["amplitude_counts"] = float(popt[0])
    forced["baseline_counts"] = background
    forced["r_squared"] = r_squared
    forced["rmse_counts"] = rmse
    forced["peak_time_ns"] = peak_time
    forced["peak_counts"] = peak_counts
    forced["fit_start_ns"] = t0
    forced["fit_end_ns"] = float(x_fit[-1])
    forced["fit_points"] = len(x_fit)
    forced["fit_end_rule"] = fit_end_rule
    forced["detected_peak_count"] = row.get("detected_peak_count", "")
    forced["secondary_peak_count"] = row.get("secondary_peak_count", "")
    return forced


def forced_double_decay_fit_row(
    row: pd.Series,
    time_ns: np.ndarray,
    counts: np.ndarray,
    output_folder: str,
) -> pd.Series:
    """Fit one selected decay profile with a zero-baseline double exponential."""
    y = counts.astype(float)
    smooth = gaussian_filter1d(y, sigma=2.0)
    baseline_seed = max(0.0, float(np.nanpercentile(y, 5)))
    signal = smooth - baseline_seed
    peak_idx = int(np.nanargmax(signal))
    peak_height = float(signal[peak_idx])
    peak_time = float(time_ns[peak_idx])
    peak_counts = float(y[peak_idx])
    fit_start_time = peak_time + 0.05
    start_candidates = np.where(time_ns >= fit_start_time)[0]
    fit_start_idx = int(start_candidates[0]) if start_candidates.size else min(peak_idx + 1, len(time_ns) - 1)

    threshold = baseline_seed + 0.05 * max(peak_height, 0.0)
    below = smooth[fit_start_idx:] <= threshold
    sustained_idx = first_sustained_true(below, run_length=8)
    fit_end_idx = len(time_ns) if sustained_idx is None else fit_start_idx + sustained_idx
    fit_end_rule = "forced_end_fraction" if sustained_idx is not None else "forced_trace_end"
    manual_fit_end_ns = DECAY_FIT_END_OVERRIDES_NS.get(output_folder)
    if manual_fit_end_ns is not None:
        manual_end_idx = manual_fit_end_index(time_ns, fit_start_idx, manual_fit_end_ns)
        if manual_end_idx is not None:
            fit_end_idx = min(fit_end_idx, manual_end_idx)
            fit_end_rule = f"manual_end_{format_nm(manual_fit_end_ns)}ns"
    min_fit_points = 20
    if fit_end_idx - fit_start_idx < min_fit_points:
        fit_end_idx = min(len(time_ns), fit_start_idx + min_fit_points)
        fit_end_rule += ";min_points_extended"
    if fit_end_idx - fit_start_idx < 5:
        fit_start_idx = max(0, min(fit_start_idx, len(time_ns) - 5))
        fit_end_idx = len(time_ns)
        fit_end_rule += ";trace_extended"

    x_fit = time_ns[fit_start_idx:fit_end_idx]
    y_fit = y[fit_start_idx:fit_end_idx]
    t0 = float(x_fit[0])
    tail = y_fit[max(0, int(0.8 * len(y_fit))) :]
    baseline0 = max(0.0, float(np.nanpercentile(tail, 20)))
    amplitude0 = max(float(y_fit[0]), float(np.nanmax(y_fit) - baseline0), 1.0e-6)
    duration = max(float(x_fit[-1] - x_fit[0]), 0.03)
    tau_fast0 = min(max(duration / 8.0, 0.03), 200.0)
    tau_slow0 = min(max(duration / 2.0, tau_fast0 * 2.0), 200.0)
    if tau_slow0 <= tau_fast0:
        tau_slow0 = min(200.0, tau_fast0 * 3.0)

    fit_status = "fit"
    fit_note = ""
    try:
        popt, pcov = curve_fit(
            lambda x_data, amp_fast, tau_fast, amp_slow, tau_slow: double_exp_decay(
                x_data,
                amp_fast,
                tau_fast,
                amp_slow,
                tau_slow,
                t0,
            ),
            x_fit,
            y_fit,
            p0=[0.65 * amplitude0, tau_fast0, 0.35 * amplitude0, tau_slow0],
            bounds=([0.0, 0.03, 0.0, 0.03], [np.inf, 200.0, np.inf, 200.0]),
            sigma=np.sqrt(np.maximum(y_fit, 1.0)),
            absolute_sigma=False,
            maxfev=40000,
        )
    except Exception as exc:  # noqa: BLE001
        fit_status = "fallback"
        fit_note = f"curve_fit_failed:{exc}"
        popt = np.array([0.65 * amplitude0, tau_fast0, 0.35 * amplitude0, tau_slow0], dtype=float)
        pcov = np.full((4, 4), np.nan)

    if float(popt[1]) <= float(popt[3]):
        order = [0, 1, 2, 3]
    else:
        order = [2, 3, 0, 1]
    popt = np.asarray(popt, dtype=float)[order]
    pcov = np.asarray(pcov, dtype=float)[np.ix_(order, order)] if pcov.size else np.full((4, 4), np.nan)

    y_pred = double_exp_decay(x_fit, float(popt[0]), float(popt[1]), float(popt[2]), float(popt[3]), t0)
    residual = y_fit - y_pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(residual**2)))
    perr = np.sqrt(np.diag(pcov)) if pcov.size else np.full(4, np.nan)

    forced = row.copy()
    forced["original_batch_status"] = row.get("status", "")
    forced["original_batch_reason"] = row.get("reason", "")
    forced["forced_decay_status"] = fit_status
    forced["forced_decay_note"] = fit_note
    forced["amplitude_fast_counts"] = float(popt[0])
    forced["tau_fast_ns"] = float(popt[1])
    forced["amplitude_slow_counts"] = float(popt[2])
    forced["tau_slow_ns"] = float(popt[3])
    forced["amplitude_fast_se_counts"] = float(perr[0]) if len(perr) > 0 else float("nan")
    forced["tau_fast_se_ns"] = float(perr[1]) if len(perr) > 1 else float("nan")
    forced["amplitude_slow_se_counts"] = float(perr[2]) if len(perr) > 2 else float("nan")
    forced["tau_slow_se_ns"] = float(perr[3]) if len(perr) > 3 else float("nan")
    forced["amplitude_sum_counts"] = float(popt[0] + popt[2])
    forced["baseline_counts"] = 0.0
    forced["r_squared"] = r_squared
    forced["rmse_counts"] = rmse
    forced["peak_time_ns"] = peak_time
    forced["peak_counts"] = peak_counts
    forced["fit_start_ns"] = t0
    forced["fit_end_ns"] = float(x_fit[-1])
    forced["fit_points"] = len(x_fit)
    forced["fit_end_rule"] = fit_end_rule
    forced["detected_peak_count"] = row.get("detected_peak_count", "")
    forced["secondary_peak_count"] = row.get("secondary_peak_count", "")
    return forced


def crossing_time(time_ns: np.ndarray, signal: np.ndarray, threshold: float, end_idx: int) -> float:
    """Return a linearly interpolated first rising threshold crossing."""
    if not math.isfinite(threshold):
        return float("nan")
    for idx in range(1, end_idx + 1):
        previous = float(signal[idx - 1])
        current = float(signal[idx])
        if previous < threshold <= current:
            span = current - previous
            if span == 0:
                return float(time_ns[idx])
            fraction = (threshold - previous) / span
            return float(time_ns[idx - 1] + fraction * (time_ns[idx] - time_ns[idx - 1]))
    return float("nan")


def forced_rise_fit_row(row: pd.Series, time_ns: np.ndarray, counts: np.ndarray) -> pd.Series:
    """Fit one rise profile even when the batch diagnostic marked it as low-signal."""
    y = counts.astype(float)
    smooth = gaussian_filter1d(y, sigma=2.0)
    baseline_seed = max(0.0, float(np.nanpercentile(y, 5)))
    signal = smooth - baseline_seed
    peak_idx = int(np.nanargmax(signal))
    peak_height = float(signal[peak_idx])
    peak_time = float(time_ns[peak_idx])
    peak_counts = float(y[peak_idx])
    effective_height = max(peak_height, float(np.nanmax(signal) - np.nanmin(signal)), 1.0e-6)
    low_threshold = 0.10 * effective_height
    high_threshold = 0.90 * effective_height
    pre_peak = signal[: peak_idx + 1]
    low_cross_idx = first_sustained_true(pre_peak >= low_threshold, run_length=3)
    high_cross_idx = first_sustained_true(pre_peak >= high_threshold, run_length=2)
    if low_cross_idx is None:
        low_cross_idx = max(0, peak_idx - 20)
    if high_cross_idx is None or high_cross_idx <= low_cross_idx:
        high_cross_idx = peak_idx

    threshold_10_time = crossing_time(time_ns, signal, low_threshold, peak_idx)
    threshold_90_time = crossing_time(time_ns, signal, high_threshold, peak_idx)
    observed_rise_time = (
        threshold_90_time - threshold_10_time
        if math.isfinite(threshold_10_time) and math.isfinite(threshold_90_time)
        else float("nan")
    )

    min_fit_points = 20
    margin = max(3, min_fit_points // 4)
    fit_start_idx = max(0, low_cross_idx - margin)
    fit_end_idx = min(len(time_ns), high_cross_idx + margin + 1)
    if fit_end_idx - fit_start_idx < min_fit_points:
        missing = min_fit_points - (fit_end_idx - fit_start_idx)
        fit_start_idx = max(0, fit_start_idx - math.ceil(missing / 2))
        fit_end_idx = min(len(time_ns), fit_end_idx + math.floor(missing / 2) + 1)
    if fit_end_idx - fit_start_idx < 3:
        fit_start_idx = max(0, min(peak_idx, len(time_ns) - min_fit_points))
        fit_end_idx = min(len(time_ns), fit_start_idx + min_fit_points)

    x_fit = time_ns[fit_start_idx:fit_end_idx]
    y_fit = y[fit_start_idx:fit_end_idx]
    dt = float(np.median(np.diff(time_ns))) if time_ns.size > 1 else 0.01
    k_lower = max(abs(dt) / 100.0, 1.0e-6)
    baseline0 = max(0.0, float(np.nanpercentile(y_fit[: max(3, min(8, len(y_fit)))], 20)))
    amplitude0 = max(float(np.nanmax(y_fit) - baseline0), 1.0e-6)
    k0 = (
        observed_rise_time / (2.0 * math.log(9.0))
        if math.isfinite(observed_rise_time) and observed_rise_time > 0
        else max(abs(dt), 0.001)
    )
    k0 = min(max(k0, k_lower), 200.0)
    midpoint0 = (
        float((threshold_10_time + threshold_90_time) / 2.0)
        if math.isfinite(threshold_10_time) and math.isfinite(threshold_90_time)
        else float(time_ns[min(max(peak_idx, fit_start_idx), fit_end_idx - 1)])
    )
    midpoint0 = min(max(midpoint0, float(x_fit[0])), float(x_fit[-1]))
    upper_signal = max(float(np.nanmax(y_fit)), baseline0 + amplitude0, 1.0e-9)

    fit_status = "fit"
    fit_note = ""
    try:
        popt, pcov = curve_fit(
            sigmoid_rise,
            x_fit,
            y_fit,
            p0=[amplitude0, k0, baseline0, midpoint0],
            bounds=([0.0, k_lower, 0.0, float(x_fit[0])], [max(upper_signal * 3.0, 1.0), 200.0, upper_signal, float(x_fit[-1])]),
            sigma=np.sqrt(np.maximum(y_fit, 1.0)),
            absolute_sigma=False,
            maxfev=20000,
        )
    except Exception as exc:  # noqa: BLE001
        fit_status = "fallback"
        fit_note = f"curve_fit_failed:{exc}"
        popt = np.array([amplitude0, k0, baseline0, midpoint0], dtype=float)
        pcov = np.full((4, 4), np.nan)

    y_pred = sigmoid_rise(x_fit, float(popt[0]), float(popt[1]), float(popt[2]), float(popt[3]))
    residual = y_fit - y_pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(residual**2)))
    perr = np.sqrt(np.diag(pcov)) if pcov.size else np.full(4, np.nan)
    sigmoid_k = float(popt[1])

    forced = row.copy()
    forced["original_batch_status"] = row.get("status", "")
    forced["original_batch_reason"] = row.get("reason", "")
    forced["forced_rise_status"] = fit_status
    forced["forced_rise_note"] = fit_note
    forced["sigmoid_k_ns"] = sigmoid_k
    forced["sigmoid_k_se_ns"] = float(perr[1]) if len(perr) > 1 else float("nan")
    forced["fitted_rise_time_10_90_ns"] = 2.0 * math.log(9.0) * sigmoid_k
    forced["fitted_rise_time_10_90_se_ns"] = 2.0 * math.log(9.0) * forced["sigmoid_k_se_ns"]
    forced["observed_rise_time_10_90_ns"] = observed_rise_time
    forced["amplitude_counts"] = float(popt[0])
    forced["baseline_counts"] = float(popt[2])
    forced["midpoint_time_ns"] = float(popt[3])
    forced["r_squared"] = r_squared
    forced["rmse_counts"] = rmse
    forced["peak_time_ns"] = peak_time
    forced["peak_counts"] = peak_counts
    forced["threshold_10_time_ns"] = threshold_10_time
    forced["threshold_90_time_ns"] = threshold_90_time
    forced["fit_start_ns"] = float(x_fit[0])
    forced["fit_end_ns"] = float(x_fit[-1])
    forced["fit_points"] = len(x_fit)
    return forced


def result_row(
    *,
    entry: dict[str, str],
    fit_type: str,
    time_window: str,
    row: pd.Series,
    plot_path: Path,
    output_root: Path,
) -> dict[str, object]:
    """Build one combined output row."""
    base = {
        "fit_type": fit_type,
        "time_window": time_window,
        "sample": entry["sample"],
        "position": source_position(entry["source_file"]),
        "interval": interval_label(row),
        "source_file": entry["source_file"],
        "output_folder": entry["output_folder"],
        "band_min_nm": row["band_min_nm"],
        "band_max_nm": row["band_max_nm"],
        "band_center_nm": row["band_center_nm"],
        "r_squared": row["r_squared"],
        "rmse_counts": row["rmse_counts"],
        "peak_time_ns": row["peak_time_ns"],
        "peak_counts": row["peak_counts"],
        "fit_start_ns": row["fit_start_ns"],
        "fit_end_ns": row["fit_end_ns"],
        "fit_points": row["fit_points"],
        "qa_plot": plot_path.relative_to(output_root).as_posix(),
    }
    if fit_type == "rise":
        base.update(
            {
                "original_batch_status": row.get("original_batch_status", row.get("status", "")),
                "original_batch_reason": row.get("original_batch_reason", row.get("reason", "")),
                "forced_rise_status": row.get("forced_rise_status", ""),
                "forced_rise_note": row.get("forced_rise_note", ""),
                "sigmoid_k_ns": row["sigmoid_k_ns"],
                "sigmoid_k_se_ns": row["sigmoid_k_se_ns"],
                "fitted_rise_time_10_90_ns": row["fitted_rise_time_10_90_ns"],
                "fitted_rise_time_10_90_se_ns": row["fitted_rise_time_10_90_se_ns"],
                "observed_rise_time_10_90_ns": row["observed_rise_time_10_90_ns"],
            }
        )
    else:
        base.update(
            {
                "original_batch_status": row.get("original_batch_status", row.get("status", "")),
                "original_batch_reason": row.get("original_batch_reason", row.get("reason", "")),
                "forced_decay_status": row.get("forced_decay_status", ""),
                "forced_decay_note": row.get("forced_decay_note", ""),
                "tau_ns": row["tau_ns"],
                "tau_se_ns": row["tau_se_ns"],
                "amplitude_counts": row["amplitude_counts"],
                "baseline_counts": row["baseline_counts"],
                "detected_peak_count": row["detected_peak_count"],
                "secondary_peak_count": row["secondary_peak_count"],
                "fit_end_rule": row["fit_end_rule"],
            }
        )
    return base


def double_decay_result_row(
    *,
    entry: dict[str, str],
    time_window: str,
    row: pd.Series,
    plot_path: Path,
    output_root: Path,
) -> dict[str, object]:
    """Build one selected double-exponential decay output row."""
    return {
        "fit_type": "decay_double_exp",
        "time_window": time_window,
        "sample": entry["sample"],
        "position": source_position(entry["source_file"]),
        "interval": interval_label(row),
        "source_file": entry["source_file"],
        "output_folder": entry["output_folder"],
        "band_min_nm": row["band_min_nm"],
        "band_max_nm": row["band_max_nm"],
        "band_center_nm": row["band_center_nm"],
        "r_squared": row["r_squared"],
        "rmse_counts": row["rmse_counts"],
        "peak_time_ns": row["peak_time_ns"],
        "peak_counts": row["peak_counts"],
        "fit_start_ns": row["fit_start_ns"],
        "fit_end_ns": row["fit_end_ns"],
        "fit_points": row["fit_points"],
        "qa_plot": plot_path.relative_to(output_root).as_posix(),
        "original_batch_status": row.get("original_batch_status", row.get("status", "")),
        "original_batch_reason": row.get("original_batch_reason", row.get("reason", "")),
        "forced_decay_status": row.get("forced_decay_status", ""),
        "forced_decay_note": row.get("forced_decay_note", ""),
        "amplitude_fast_counts": row["amplitude_fast_counts"],
        "amplitude_fast_se_counts": row["amplitude_fast_se_counts"],
        "tau_fast_ns": row["tau_fast_ns"],
        "tau_fast_se_ns": row["tau_fast_se_ns"],
        "amplitude_slow_counts": row["amplitude_slow_counts"],
        "amplitude_slow_se_counts": row["amplitude_slow_se_counts"],
        "tau_slow_ns": row["tau_slow_ns"],
        "tau_slow_se_ns": row["tau_slow_se_ns"],
        "amplitude_sum_counts": row["amplitude_sum_counts"],
        "baseline_counts": row["baseline_counts"],
        "detected_peak_count": row["detected_peak_count"],
        "secondary_peak_count": row["secondary_peak_count"],
        "fit_end_rule": row["fit_end_rule"],
    }


def process_fit_family(
    *,
    entry: dict[str, str],
    cuts_dir: Path,
    output_root: Path,
    fit_type: str,
    time_window: str,
) -> list[dict[str, object]]:
    """Write QA plots and combined rows for one selected scan."""
    scan_dir = cuts_dir / entry["output_folder"]
    profiles = read_profiles(scan_dir)
    time_ns = profiles["time_ns"].to_numpy(dtype=float)
    if fit_type == "rise":
        summary = all_rows(scan_dir / "rise_fit_summary.txt")
        family_dir = output_root / "rise_time_2ns"
        plot_func = plot_rise_fit
    else:
        summary = all_rows(scan_dir / "fit_summary.txt")
        family_dir = output_root / "decay_time_10ns"
        plot_func = plot_decay_fit

    rows: list[dict[str, object]] = []
    plot_dir = family_dir / safe_scan_folder(entry["output_folder"]) / "1exp_fit_plots"
    for _, row in progress_bar(
        summary.iterrows(),
        total=len(summary),
        desc=f"{fit_type}: {Path(entry['source_file']).stem}",
        unit="fit",
        leave=False,
    ):
        column = profile_column(row)
        if column not in profiles:
            continue
        counts = profiles[column].to_numpy(dtype=float)
        if fit_type == "rise":
            row = forced_rise_fit_row(row, time_ns, counts)
        elif fit_type == "decay":
            row = forced_decay_fit_row(row, time_ns, counts, entry["output_folder"])
        label = interval_label(row)
        title = f"{Path(entry['source_file']).stem} | {label}"
        plot_path = plot_dir / f"{fit_type}_{format_nm(row['band_min_nm'])}_{format_nm(row['band_max_nm'])}nm.png"
        plot_func(plot_path, time_ns, counts, row, title)
        rows.append(
            result_row(
                entry=entry,
                fit_type=fit_type,
                time_window=time_window,
                row=row,
                plot_path=plot_path,
                output_root=output_root,
            )
        )
    return rows


def process_selected_double_decay_family(
    *,
    entry: dict[str, str],
    cuts_dir: Path,
    output_root: Path,
    time_window: str,
    selected: set[tuple[str, str, str]],
) -> list[dict[str, object]]:
    """Write double-exponential QA plots and rows for manually selected 10 ns decay cuts."""
    scan_dir = cuts_dir / entry["output_folder"]
    profiles = read_profiles(scan_dir)
    time_ns = profiles["time_ns"].to_numpy(dtype=float)
    summary = all_rows(scan_dir / "fit_summary.txt")
    family_dir = output_root / "decay_time_10ns_double_exp"
    plot_dir = family_dir / safe_scan_folder(entry["output_folder"]) / "2exp_fit_plots"
    sample = entry["sample"]
    position = source_position(entry["source_file"])

    selected_rows = []
    for _, row in summary.iterrows():
        label = interval_label(row)
        if (sample, position, label) in selected:
            selected_rows.append(row)

    rows: list[dict[str, object]] = []
    for row in progress_bar(
        selected_rows,
        total=len(selected_rows),
        desc=f"2-exp decay: {Path(entry['source_file']).stem}",
        unit="fit",
        leave=False,
    ):
        column = profile_column(row)
        if column not in profiles:
            continue
        counts = profiles[column].to_numpy(dtype=float)
        row = forced_double_decay_fit_row(row, time_ns, counts, entry["output_folder"])
        label = interval_label(row)
        title = f"{Path(entry['source_file']).stem} | {label}"
        plot_path = plot_dir / f"double_decay_{format_nm(row['band_min_nm'])}_{format_nm(row['band_max_nm'])}nm.png"
        plot_double_decay_fit(plot_path, time_ns, counts, row, title)
        rows.append(
            double_decay_result_row(
                entry=entry,
                time_window=time_window,
                row=row,
                plot_path=plot_path,
                output_root=output_root,
            )
        )
    return rows


COMMON_FIELDS = [
    "fit_type",
    "time_window",
    "sample",
    "position",
    "interval",
    "source_file",
    "output_folder",
    "band_min_nm",
    "band_max_nm",
    "band_center_nm",
    "r_squared",
    "rmse_counts",
    "peak_time_ns",
    "peak_counts",
    "fit_start_ns",
    "fit_end_ns",
    "fit_points",
    "qa_plot",
]

RISE_FIELDS = COMMON_FIELDS + [
    "original_batch_status",
    "original_batch_reason",
    "forced_rise_status",
    "forced_rise_note",
    "sigmoid_k_ns",
    "sigmoid_k_se_ns",
    "fitted_rise_time_10_90_ns",
    "fitted_rise_time_10_90_se_ns",
    "observed_rise_time_10_90_ns",
]

DECAY_FIELDS = COMMON_FIELDS + [
    "original_batch_status",
    "original_batch_reason",
    "forced_decay_status",
    "forced_decay_note",
    "tau_ns",
    "tau_se_ns",
    "amplitude_counts",
    "baseline_counts",
    "detected_peak_count",
    "secondary_peak_count",
    "fit_end_rule",
]

DOUBLE_DECAY_FIELDS = COMMON_FIELDS + [
    "original_batch_status",
    "original_batch_reason",
    "forced_decay_status",
    "forced_decay_note",
    "amplitude_fast_counts",
    "amplitude_fast_se_counts",
    "tau_fast_ns",
    "tau_fast_se_ns",
    "amplitude_slow_counts",
    "amplitude_slow_se_counts",
    "tau_slow_ns",
    "tau_slow_se_ns",
    "amplitude_sum_counts",
    "baseline_counts",
    "detected_peak_count",
    "secondary_peak_count",
    "fit_end_rule",
]


def write_sample_matrices(
    *,
    rows: list[dict[str, object]],
    value_field: str,
    out_dir: Path,
    file_suffix: str,
) -> None:
    """Write one sample-position by wavelength-interval matrix per sample."""
    if not rows:
        return
    frame = pd.DataFrame(rows)
    intervals = sorted(frame["interval"].dropna().unique(), key=interval_sort_key)
    for sample in sorted(frame["sample"].dropna().unique()):
        sample_frame = frame[frame["sample"] == sample].copy()
        positions = sorted(sample_frame["position"].dropna().unique(), key=position_sort_key)
        value_matrix = (
            sample_frame.pivot_table(
                index="position",
                columns="interval",
                values=value_field,
                aggfunc="mean",
                dropna=False,
            )
            .reindex(index=positions, columns=intervals)
            .reset_index()
        )
        count_matrix = (
            sample_frame.pivot_table(
                index="position",
                columns="interval",
                values=value_field,
                aggfunc="count",
                dropna=False,
            )
            .reindex(index=positions, columns=intervals)
            .fillna(0)
            .astype(int)
            .reset_index()
        )
        write_frame_tsv(out_dir / f"{sample}_{file_suffix}.txt", value_matrix)
        write_frame_tsv(out_dir / "replicate_counts" / f"{sample}_{file_suffix}_replicate_counts.txt", count_matrix)


def main(argv: list[str] | None = None) -> int:
    """Build 2 ns rise-time and 10 ns decay-time fit result folders."""
    parser = argparse.ArgumentParser(description="Build selected wavelength-cut fit result folders and QA plots.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--cuts-subdir", default=DEFAULT_CUTS_SUBDIR)
    parser.add_argument("--out-subdir", default=DEFAULT_OUT_SUBDIR)
    parser.add_argument("--rise-window", default="2ns")
    parser.add_argument("--decay-window", default="10ns")
    parser.add_argument("--decay-model", choices=["single", "double"], default="single")
    parser.add_argument("--selection-subdir", default="manual selections")
    parser.add_argument(
        "--selection-dir",
        type=Path,
        default=None,
        help="Read-only directory containing tracked decay-time selection matrices.",
    )
    args = parser.parse_args(argv)

    results_dir = resolve_path(args.results_dir)
    cuts_dir = results_dir / args.cuts_subdir
    output_root = results_dir / args.out_subdir
    inventory_path = cuts_dir / "inventory.txt"
    if not inventory_path.exists():
        raise SystemExit(f"Missing wavelength-cut inventory: {inventory_path}")

    with inventory_path.open(encoding="utf-8") as handle:
        entries = list(csv.DictReader(handle, delimiter="\t"))

    if args.decay_model == "double":
        selection_dir = (
            resolve_path(args.selection_dir)
            if args.selection_dir is not None
            else output_root / args.selection_subdir / "decay_time_10ns"
        )
        selected = read_decay_manual_selection(selection_dir)
        selected_scan_keys = {(sample, position) for sample, position, _ in selected}
        selected_entries = [
            entry
            for entry in entries
            if entry.get("status") == "ok"
            and source_time_window(entry.get("source_file", "")) == args.decay_window
            and (entry["sample"], source_position(entry.get("source_file", ""))) in selected_scan_keys
        ]
        double_decay_rows: list[dict[str, object]] = []
        for entry in progress_bar(selected_entries, desc="Selected 2-exp decay scans", unit="scan"):
            double_decay_rows.extend(
                process_selected_double_decay_family(
                    entry=entry,
                    cuts_dir=cuts_dir,
                    output_root=output_root,
                    time_window=args.decay_window,
                    selected=selected,
                )
            )

        write_tsv(
            output_root
            / "decay_time_10ns_double_exp"
            / "decay_time_double_exp_fits_10ns_manual_selection.txt",
            double_decay_rows,
            DOUBLE_DECAY_FIELDS,
        )
        write_sample_matrices(
            rows=double_decay_rows,
            value_field="tau_slow_ns",
            out_dir=output_root / "tabulated_by_sample" / "decay_time_10ns_double_exp",
            file_suffix="decay_time_double_exp_tau_slow_10ns_by_position_interval",
        )
        write_tsv(
            output_root / "double_exp_fit_result_inventory.txt",
            [
                {
                    "fit_type": "decay_double_exp",
                    "time_window": args.decay_window,
                    "fit_count": len(double_decay_rows),
                    "selection_count": len(selected),
                },
            ],
            ["fit_type", "time_window", "fit_count", "selection_count"],
        )
        print(f"selected manual cells: {len(selected)}")
        print(f"double-exp decay fits ({args.decay_window}): {len(double_decay_rows)}")
        print(f"output: {output_root / 'decay_time_10ns_double_exp'}")
        print(f"manual selections: {selection_dir}")
        return 0

    rise_rows: list[dict[str, object]] = []
    decay_rows: list[dict[str, object]] = []
    selected_entries = [
        entry
        for entry in entries
        if entry.get("status") == "ok"
        and source_time_window(entry.get("source_file", "")) in {args.rise_window, args.decay_window}
    ]
    for entry in progress_bar(selected_entries, desc="Fit result scans", unit="scan"):
        window = source_time_window(entry.get("source_file", ""))
        if window == args.rise_window:
            rise_rows.extend(
                process_fit_family(
                    entry=entry,
                    cuts_dir=cuts_dir,
                    output_root=output_root,
                    fit_type="rise",
                    time_window=window,
                )
            )
        elif window == args.decay_window:
            decay_rows.extend(
                process_fit_family(
                    entry=entry,
                    cuts_dir=cuts_dir,
                    output_root=output_root,
                    fit_type="decay",
                    time_window=window,
                )
            )

    write_tsv(output_root / "rise_time_2ns" / "rise_time_fits_2ns.txt", rise_rows, RISE_FIELDS)
    write_tsv(output_root / "decay_time_10ns" / "decay_time_fits_10ns.txt", decay_rows, DECAY_FIELDS)
    write_sample_matrices(
        rows=rise_rows,
        value_field="fitted_rise_time_10_90_ns",
        out_dir=output_root / "tabulated_by_sample" / "rise_time_2ns",
        file_suffix="rise_time_2ns_by_position_interval",
    )
    write_sample_matrices(
        rows=decay_rows,
        value_field="tau_ns",
        out_dir=output_root / "tabulated_by_sample" / "decay_time_10ns",
        file_suffix="decay_time_10ns_by_position_interval",
    )
    write_tsv(
        output_root / "fit_result_inventory.txt",
        [
            {"fit_type": "rise", "time_window": args.rise_window, "fit_count": len(rise_rows)},
            {"fit_type": "decay", "time_window": args.decay_window, "fit_count": len(decay_rows)},
        ],
        ["fit_type", "time_window", "fit_count"],
    )

    print(f"rise fits ({args.rise_window}): {len(rise_rows)}")
    print(f"decay fits ({args.decay_window}): {len(decay_rows)}")
    print(f"output: {output_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
