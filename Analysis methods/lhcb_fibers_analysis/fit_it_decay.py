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

from .paths import DEFAULT_IT_DECAY_CONFIG, DEFAULT_RAW_DIR, DEFAULT_RESULTS_DIR, resolve_path
from .yaml_config import float_value, int_value, read_yaml_mapping, string_value


DEFAULT_TIME_WINDOW = "10ns"


@dataclass(frozen=True)
class FitConfig:
    out_subdir: str = "it_decay_fits_10ns"
    time_window: str = DEFAULT_TIME_WINDOW
    smooth_sigma: float = 2.0
    min_points: int = 30
    min_fit_points: int = 25
    peak_height_fraction: float = 0.20
    peak_prominence_fraction: float = 0.12
    noise_sigma_threshold: float = 3.0
    low_signal_sigma_threshold: float = 5.0
    fit_tau_min_ns: float = 0.03
    fit_tau_max_ns: float = 50.0


def read_fit_config(path: Path) -> FitConfig:
    values = read_yaml_mapping(path)
    tau_min = float_value(values, "fit_tau_min_ns", 0.03)
    tau_max = float_value(values, "fit_tau_max_ns", 50.0)
    if tau_max <= tau_min:
        raise ValueError(f"fit_tau_max_ns must be greater than fit_tau_min_ns in {path}")
    return FitConfig(
        out_subdir=string_value(values, "out_subdir", "it_decay_fits_10ns"),
        time_window=string_value(values, "time_window", DEFAULT_TIME_WINDOW),
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
    excitation: str
    energy: str
    window: str
    wavelength_label: str


def natural_position_key(position: str) -> tuple[float, str]:
    if position == "ENDcm":
        return (math.inf, position)
    match = re.match(r"(?P<distance>\d+)cm(?P<suffix>.*)$", position)
    if not match:
        return (math.inf, position)
    return (float(match.group("distance")), match.group("suffix"))


def discover_traces(raw_dir: Path, time_window: str) -> list[TraceInfo]:
    traces: list[TraceInfo] = []
    for path in sorted(raw_dir.rglob(f"IT_*_{time_window}.dat")):
        match = FILENAME_RE.match(path.name)
        if not match:
            continue
        parts = match.groupdict()
        if parts["window"].lower() != time_window.lower():
            continue
        first_two = path.read_text(encoding="utf-8", errors="replace").splitlines()[:2]
        wavelength_label = ""
        if len(first_two) > 1:
            fields = first_two[1].split("\t")
            if len(fields) > 1:
                wavelength_label = fields[1]
        traces.append(
            TraceInfo(
                path=path,
                relative_path=path.relative_to(raw_dir),
                sample=parts["sample"],
                position=parts["position"],
                excitation=parts["excitation"],
                energy=parts["energy"],
                window=parts["window"],
                wavelength_label=wavelength_label,
            )
        )
    return traces


def read_trace(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        skiprows=2,
        names=["delay_ns", "counts", "normalized"],
        dtype=float,
    ).dropna()


def exp_decay(x: np.ndarray, amplitude: float, tau_ns: float, baseline: float, x0: float) -> np.ndarray:
    return amplitude * np.exp(-(x - x0) / tau_ns) + baseline


def robust_noise(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad == 0:
        return float(np.std(values))
    return 1.4826 * mad


def first_sustained_true(mask: np.ndarray, run_length: int) -> int | None:
    if mask.size < run_length:
        return None
    run = 0
    for idx, flag in enumerate(mask):
        run = run + 1 if flag else 0
        if run >= run_length:
            return idx - run_length + 1
    return None


def fit_trace(info: TraceInfo, df: pd.DataFrame, config: FitConfig) -> dict[str, object]:
    x = df["delay_ns"].to_numpy(dtype=float)
    y = df["counts"].to_numpy(dtype=float)
    n = len(y)
    result: dict[str, object] = {
        "sample": info.sample,
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

    result.update(
        n_points=n,
        baseline_seed_counts=baseline_seed,
        noise_counts=noise,
        primary_peak_delay_ns=float(x[peak_idx]),
        primary_peak_counts=float(y[peak_idx]),
    )

    if peak_height <= max(config.low_signal_sigma_threshold * noise, config.low_signal_sigma_threshold):
        result.update(status="skipped", skip_reason="low_signal")
        return result

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
        result.update(status="skipped", skip_reason="multiple_significant_peaks")
        return result

    post_peak = np.arange(peak_idx, n)
    below_85 = np.where(smooth[post_peak] <= baseline_seed + 0.85 * peak_height)[0]
    if below_85.size:
        fit_start_idx = int(post_peak[below_85[0]])
    else:
        fit_start_idx = min(n - 1, peak_idx + max(3, int(round(0.10 / abs(dt)))))
    fit_start_idx = max(fit_start_idx, peak_idx + 3)

    threshold = baseline_seed + max(0.05 * peak_height, config.noise_sigma_threshold * noise, 1.0)
    below_noise = smooth[fit_start_idx:] <= threshold
    sustained_idx = first_sustained_true(below_noise, run_length=10)
    if sustained_idx is None:
        fit_end_idx = n
    else:
        fit_end_idx = fit_start_idx + sustained_idx

    min_fit_points = config.min_fit_points
    if fit_end_idx - fit_start_idx < min_fit_points:
        fit_end_idx = min(n, fit_start_idx + min_fit_points)

    if fit_end_idx - fit_start_idx < min_fit_points:
        result.update(status="skipped", skip_reason="too_few_decay_points")
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

    warnings: list[str] = []
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
        fit_start_ns=float(x_fit[0]),
        fit_end_ns=float(x_fit[-1]),
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
) -> tuple[str, str]:
    sample_dir = plots_root / info.sample
    sample_dir.mkdir(parents=True, exist_ok=True)
    stem = info.path.stem
    png_path = sample_dir / f"{stem}_fit.png"
    pdf_path = sample_dir / f"{stem}_fit.pdf"

    x = df["delay_ns"].to_numpy(dtype=float)
    y = df["counts"].to_numpy(dtype=float)
    smooth = gaussian_filter1d(y, sigma=config.smooth_sigma)
    y_scale = max(float(np.max(y)), 1.0)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 0.8,
        }
    )
    fig, (ax, ax_res) = plt.subplots(
        2,
        1,
        figsize=(6.6, 4.8),
        dpi=220,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.08},
    )

    ax.plot(x, y / y_scale, "o", ms=2.6, color="#222222", alpha=0.80, label="IT data")
    ax.plot(x, smooth / y_scale, "-", lw=1.2, color="#6B7280", label="smoothed")
    ax.axvline(
        float(result.get("primary_peak_delay_ns", x[int(np.argmax(y))])),
        lw=0.9,
        ls=":",
        color="#1F77B4",
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
            ms=6,
            mew=1.2,
            color="#D62728",
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
        ax.axvspan(fit_start, fit_end, color="#D62728", alpha=0.08, lw=0, label="_nolegend_")
        ax.plot(x_dense, y_dense / y_scale, "-", lw=1.8, color="#D62728", label="single-exponential fit")
        ax_res.plot(x_fit, (y_fit - y_pred) / y_scale, "o", ms=2.6, color="#222222", alpha=0.80)
        annotation = (
            rf"$\tau$ = {float(result['tau_ns']):.3g} $\pm$ {float(result['tau_se_ns']):.2g} ns"
            "\n"
            rf"$R^2$ = {float(result['r_squared']):.3f}"
            "\n"
            f"fit points = {int(result['fit_points'])}"
        )
    else:
        ax_res.plot(x, np.zeros_like(x), "-", lw=0.8, color="#9CA3AF")
        annotation = f"Skipped: {result.get('skip_reason', '')}"

    ax.text(
        0.98,
        0.95,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="right",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.92},
    )
    ax.set_ylabel("Normalized intensity")
    ax.set_title(
        f"{info.sample}, {info.position}, IT {info.window}"
        f" ({info.wavelength_label} nm, ex {info.excitation}, {info.energy})",
        fontsize=12,
        pad=8,
    )
    ax.legend(frameon=True, loc="lower left", fontsize=8, framealpha=0.90, edgecolor="#D1D5DB")
    ax.grid(True, color="#E5E7EB", lw=0.6)

    ax_res.axhline(0, color="#6B7280", lw=0.8)
    ax_res.set_xlabel("Delay (ns)")
    ax_res.set_ylabel("Residual")
    ax_res.grid(True, color="#E5E7EB", lw=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
        ax_res.spines[spine].set_visible(False)

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return (
        Path(os.path.relpath(png_path, out_root)).as_posix(),
        Path(os.path.relpath(pdf_path, out_root)).as_posix(),
    )


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_matrices(results: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = sorted({str(row["sample"]) for row in results})
    positions = sorted({str(row["position"]) for row in results}, key=natural_position_key)
    tau_matrix = pd.DataFrame(index=samples, columns=positions, dtype=object)
    status_matrix = pd.DataFrame(index=samples, columns=positions, dtype=object)
    tau_matrix.index.name = "sample"
    status_matrix.index.name = "sample"

    for row in results:
        sample = str(row["sample"])
        position = str(row["position"])
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


def write_method_note(path: Path, results: list[dict[str, object]], time_window: str) -> None:
    fit_count = sum(1 for row in results if row.get("status") == "fit")
    skip_count = len(results) - fit_count
    method = f"""# IT 10 ns Decay Fits

Inputs: `IT_*_{time_window}.dat` only. Streak-map `.img` files, `T_*.dat` band traces, and other `IT_*.dat` time windows are excluded.

Fit rule: each trace is smoothed only for peak detection, then a single exponential plus constant baseline is fit to the raw-count falling edge after one dominant peak:

`I(t) = A exp(-(t - t_start) / tau) + B`

Traces with more than one significant temporal peak are not fit. The peak detector requires a secondary peak to exceed both a relative threshold and a noise/prominence threshold, so isolated count noise is not treated as a separate decay component.

Outputs:

- `fit_inventory.csv`: one row per input trace, including fit status, tau, standard error, R^2, fit window, peak count, and plot paths.
- `decay_times_matrix.csv`: samples as rows, positions as columns, fitted decay times in ns.
- `decay_times_matrix_with_status.csv`: same layout with uncertainty or skip reason in each cell.
- `plots/<sample>/`: PNG and PDF fit-quality plots for every input trace.

Summary: {fit_count} fitted, {skip_count} skipped, {len(results)} total {time_window} IT traces.
"""
    path.write_text(method, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit single-exponential decays for IT_*.dat traces.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_IT_DECAY_CONFIG)
    parser.add_argument("--out-subdir", type=Path, default=None)
    parser.add_argument("--time-window", default=None)
    args = parser.parse_args(argv)

    raw_dir = resolve_path(args.raw_dir)
    results_dir = resolve_path(args.results_dir)
    config = read_fit_config(resolve_path(args.config))
    out_subdir = args.out_subdir or Path(config.out_subdir)
    time_window = args.time_window or config.time_window
    out_root = (results_dir / out_subdir).resolve()
    plots_root = out_root / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    traces = discover_traces(raw_dir, time_window)
    if not traces:
        raise SystemExit(f"No IT_*_{time_window}.dat files found under {raw_dir}")

    results: list[dict[str, object]] = []
    for info in traces:
        df = read_trace(info.path)
        result = fit_trace(info, df, config)
        png_path, pdf_path = plot_trace(info, df, result, plots_root, out_root, config)
        result["plot_png"] = png_path
        result["plot_pdf"] = pdf_path
        results.append(result)

    fields = [
        "sample",
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
        "fit_start_ns",
        "fit_end_ns",
        "fit_points",
        "fit_warning",
        "detected_peak_count",
        "detected_peak_delays_ns",
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
    write_method_note(out_root / "README.md", results, time_window)

    fit_count = sum(1 for row in results if row.get("status") == "fit")
    print(f"Processed {len(results)} IT {time_window} traces: {fit_count} fit, {len(results) - fit_count} skipped.")
    print(f"Output: {out_root}")
    print(f"config: {resolve_path(args.config)}")


if __name__ == "__main__":
    main()
