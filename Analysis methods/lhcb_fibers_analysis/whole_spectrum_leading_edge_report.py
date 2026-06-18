from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .batch_carpet_wavelength_cuts import sigmoid_rise
from .fiber_names import read_fiber_name_map
from .leading_edge_position_report import observed_20_80_times, write_csv
from .paths import DEFAULT_FIBER_NAMES_CONFIG, DEFAULT_RESULTS_DIR, resolve_path
from .plot_style import COLORS, apply_axes_style, save_figure, set_publication_style
from .wavelength_cut_fit_results import (
    DEFAULT_CUTS_SUBDIR,
    DEFAULT_OUT_SUBDIR,
    forced_rise_fit_row,
    read_profiles,
    source_position,
    source_time_window,
    write_tsv,
)


REPORT_FIELDS = [
    "sample",
    "position",
    "distance_cm",
    "is_repeat_position",
    "source_file",
    "output_folder",
    "wavelength_min_nm",
    "wavelength_max_nm",
    "band_count",
    "leading_edge_midpoint_20_80_ns",
    "fitted_threshold_20_time_ns",
    "fitted_threshold_80_time_ns",
    "fitted_width_20_80_ns",
    "observed_midpoint_20_80_ns",
    "observed_threshold_20_time_ns",
    "observed_threshold_80_time_ns",
    "observed_width_20_80_ns",
    "sigmoid_k_ns",
    "sigmoid_k_se_ns",
    "fitted_rise_time_10_90_ns",
    "fitted_rise_time_10_90_se_ns",
    "observed_rise_time_10_90_ns",
    "r_squared",
    "rmse_counts",
    "peak_time_ns",
    "peak_counts",
    "fit_start_ns",
    "fit_end_ns",
    "fit_points",
    "whole_spectrum_counts_peak",
    "plot_outlier",
    "plot_exclusion_reason",
    "outlier_residual_ns",
    "outlier_robust_z",
]

SELECTED_TRACE_POSITIONS = ["200cm", "100cm", "37cm"]


def profile_band_columns(profiles: pd.DataFrame) -> list[str]:
    """Return wavelength-band profile columns in numeric order."""
    columns = []
    for column in profiles.columns:
        match = re.fullmatch(r"mean_counts_(\d+(?:p\d+)?|\d+(?:\.\d+)?)_(\d+(?:p\d+)?|\d+(?:\.\d+)?)nm", column)
        if match:
            lower = float(match.group(1).replace("p", "."))
            upper = float(match.group(2).replace("p", "."))
            columns.append((lower, upper, column))
    return [column for _, _, column in sorted(columns)]


def band_limits(columns: list[str]) -> tuple[float, float]:
    """Return the wavelength range covered by the selected band columns."""
    lower_values: list[float] = []
    upper_values: list[float] = []
    for column in columns:
        match = re.fullmatch(r"mean_counts_(\d+(?:p\d+)?|\d+(?:\.\d+)?)_(\d+(?:p\d+)?|\d+(?:\.\d+)?)nm", column)
        if match:
            lower_values.append(float(match.group(1).replace("p", ".")))
            upper_values.append(float(match.group(2).replace("p", ".")))
    if not lower_values or not upper_values:
        return float("nan"), float("nan")
    return min(lower_values), max(upper_values)


def position_distance_cm(position: str) -> float:
    """Extract numeric fiber distance from labels such as 100cmR."""
    match = re.match(r"(\d+(?:\.\d+)?)cm", str(position), flags=re.IGNORECASE)
    return float(match.group(1)) if match else float("nan")


def is_repeat_position(position: str) -> bool:
    """Return true for repeat-scan labels such as 37cmR."""
    return str(position).strip().lower().endswith("r")


def synthetic_rise_row() -> pd.Series:
    """Create the minimal row object required by the shared rise fitter."""
    return pd.Series({"status": "whole_spectrum", "reason": ""})


def report_row(
    *,
    entry: dict[str, str],
    fit_row: pd.Series,
    observed_times: dict[str, float],
    wavelength_min_nm: float,
    wavelength_max_nm: float,
    band_count: int,
    counts: np.ndarray,
) -> dict[str, object]:
    """Build one whole-spectrum leading-edge row."""
    position = source_position(entry["source_file"])
    sigmoid_k = float(fit_row["sigmoid_k_ns"])
    midpoint = float(fit_row["midpoint_time_ns"])
    half_width = math.log(4.0) * sigmoid_k
    threshold_20_time = midpoint - half_width
    threshold_80_time = midpoint + half_width
    return {
        "sample": entry["sample"],
        "position": position,
        "distance_cm": position_distance_cm(position),
        "is_repeat_position": is_repeat_position(position),
        "source_file": entry["source_file"],
        "output_folder": entry["output_folder"],
        "wavelength_min_nm": wavelength_min_nm,
        "wavelength_max_nm": wavelength_max_nm,
        "band_count": band_count,
        "leading_edge_midpoint_20_80_ns": midpoint,
        "fitted_threshold_20_time_ns": threshold_20_time,
        "fitted_threshold_80_time_ns": threshold_80_time,
        "fitted_width_20_80_ns": threshold_80_time - threshold_20_time,
        **observed_times,
        "sigmoid_k_ns": fit_row["sigmoid_k_ns"],
        "sigmoid_k_se_ns": fit_row["sigmoid_k_se_ns"],
        "fitted_rise_time_10_90_ns": fit_row["fitted_rise_time_10_90_ns"],
        "fitted_rise_time_10_90_se_ns": fit_row["fitted_rise_time_10_90_se_ns"],
        "observed_rise_time_10_90_ns": fit_row["observed_rise_time_10_90_ns"],
        "r_squared": fit_row["r_squared"],
        "rmse_counts": fit_row["rmse_counts"],
        "peak_time_ns": fit_row["peak_time_ns"],
        "peak_counts": fit_row["peak_counts"],
        "fit_start_ns": fit_row["fit_start_ns"],
        "fit_end_ns": fit_row["fit_end_ns"],
        "fit_points": fit_row["fit_points"],
        "whole_spectrum_counts_peak": float(np.nanmax(counts)) if counts.size else float("nan"),
        "plot_outlier": False,
        "plot_exclusion_reason": "",
        "outlier_residual_ns": float("nan"),
        "outlier_robust_z": float("nan"),
    }


def robust_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit a robust median-slope line to small position/timing samples."""
    slopes = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = float(x[j] - x[i])
            if dx != 0.0:
                slopes.append(float((y[j] - y[i]) / dx))
    slope = float(np.median(slopes)) if slopes else 0.0
    intercept = float(np.median(y - slope * x))
    return slope, intercept


def mark_plot_outliers(rows: list[dict[str, object]], *, robust_z_limit: float) -> list[dict[str, object]]:
    """Mark outliers relative to each sample's position trend for plot filtering."""
    frame = pd.DataFrame(rows)
    for sample in sorted(frame["sample"].dropna().unique()):
        sample_mask = frame["sample"] == sample
        candidate_mask = (
            sample_mask
            & pd.to_numeric(frame["distance_cm"], errors="coerce").notna()
            & pd.to_numeric(frame["leading_edge_midpoint_20_80_ns"], errors="coerce").notna()
            & ~frame["is_repeat_position"].astype(bool)
        )
        excluded_mask = sample_mask & ~candidate_mask
        frame.loc[excluded_mask, "plot_outlier"] = True
        frame.loc[excluded_mask, "plot_exclusion_reason"] = np.where(
            frame.loc[excluded_mask, "is_repeat_position"].astype(bool),
            "repeat_position",
            "non_numeric_position_or_missing_fit",
        )

        candidates = frame.loc[candidate_mask].copy()
        if len(candidates) < 4:
            continue

        x = pd.to_numeric(candidates["distance_cm"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(candidates["leading_edge_midpoint_20_80_ns"], errors="coerce").to_numpy(dtype=float)
        slope, intercept = robust_line(x, y)
        residual = y - (slope * x + intercept)
        residual_center = float(np.median(residual))
        mad = float(np.median(np.abs(residual - residual_center)))
        if not np.isfinite(mad) or mad <= 1.0e-12:
            robust_z = np.zeros_like(residual)
        else:
            robust_z = 0.6745 * (residual - residual_center) / mad
        outlier = np.abs(robust_z) > robust_z_limit
        frame.loc[candidates.index, "outlier_residual_ns"] = residual
        frame.loc[candidates.index, "outlier_robust_z"] = robust_z
        frame.loc[candidates.index[outlier], "plot_outlier"] = True
        frame.loc[candidates.index[outlier], "plot_exclusion_reason"] = "trend_residual_outlier"
    return frame.to_dict("records")


def build_rows(*, cuts_dir: Path, rise_window: str, robust_z_limit: float) -> list[dict[str, object]]:
    """Fit whole-spectrum leading edges from the selected 2 ns scans."""
    inventory_path = cuts_dir / "inventory.txt"
    if not inventory_path.exists():
        raise SystemExit(f"Missing wavelength-cut inventory: {inventory_path}")

    with inventory_path.open(encoding="utf-8") as handle:
        entries = [
            entry
            for entry in csv.DictReader(handle, delimiter="\t")
            if entry.get("status") == "ok" and source_time_window(entry.get("source_file", "")) == rise_window
        ]

    rows: list[dict[str, object]] = []
    for entry in tqdm(entries, desc="Whole-spectrum leading edges", unit="scan", dynamic_ncols=True, ascii=True):
        scan_dir = cuts_dir / entry["output_folder"]
        profiles = read_profiles(scan_dir)
        band_columns = profile_band_columns(profiles)
        if not band_columns:
            continue
        wavelength_min_nm, wavelength_max_nm = band_limits(band_columns)
        time_ns = profiles["time_ns"].to_numpy(dtype=float)
        counts = profiles[band_columns].sum(axis=1).to_numpy(dtype=float)
        fit_row = forced_rise_fit_row(synthetic_rise_row(), time_ns, counts)
        rows.append(
            report_row(
                entry=entry,
                fit_row=fit_row,
                observed_times=observed_20_80_times(time_ns, counts),
                wavelength_min_nm=wavelength_min_nm,
                wavelength_max_nm=wavelength_max_nm,
                band_count=len(band_columns),
                counts=counts,
            )
        )
    return mark_plot_outliers(rows, robust_z_limit=robust_z_limit)


def plot_sample_positions(
    *,
    sample: str,
    frame: pd.DataFrame,
    out_dir: Path,
    display_name: str,
) -> bool:
    """Plot whole-spectrum leading-edge timing versus position for one sample."""
    plot_frame = frame[(frame["sample"] == sample) & ~frame["plot_outlier"].astype(bool)].copy()
    plot_frame["distance_cm"] = pd.to_numeric(plot_frame["distance_cm"], errors="coerce")
    plot_frame["leading_edge_midpoint_20_80_ns"] = pd.to_numeric(
        plot_frame["leading_edge_midpoint_20_80_ns"], errors="coerce"
    )
    plot_frame = plot_frame.dropna(subset=["distance_cm", "leading_edge_midpoint_20_80_ns"])
    if plot_frame.empty:
        return False

    grouped = (
        plot_frame.groupby("distance_cm", dropna=False)
        .agg(
            midpoint_ns=("leading_edge_midpoint_20_80_ns", "mean"),
            midpoint_sd_ns=("leading_edge_midpoint_20_80_ns", "std"),
            replicate_count=("leading_edge_midpoint_20_80_ns", "size"),
        )
        .reset_index()
        .sort_values("distance_cm")
    )

    width = 6.9
    height = 4.35
    set_publication_style(base_font_size=9.4)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    x = grouped["distance_cm"].to_numpy(dtype=float)
    y = grouped["midpoint_ns"].to_numpy(dtype=float)
    yerr = grouped["midpoint_sd_ns"].fillna(0.0).to_numpy(dtype=float)
    ax.plot(x, y, "-o", color=COLORS["blue"], lw=1.45, ms=4.2, markeredgewidth=0.0)
    if np.any(yerr > 0):
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=COLORS["gray"], elinewidth=0.8, capsize=2.5)
    for row in grouped.itertuples(index=False):
        if int(row.replicate_count) > 1:
            ax.text(
                float(row.distance_cm),
                float(row.midpoint_ns),
                f" n={int(row.replicate_count)}",
                va="bottom",
                ha="left",
                fontsize=7.2,
                color=COLORS["gray"],
            )
    ax.set_xlabel("Position (cm)")
    ax.set_ylabel("Whole-spectrum leading-edge midpoint (ns)")
    ax.set_title(f"Whole-spectrum leading edge: {display_name}", pad=5)
    apply_axes_style(ax, grid=True)
    out_base = out_dir / f"{sample}_whole_spectrum_leading_edge_position"
    save_figure(fig, out_base.with_suffix(".png"), dpi=260)
    save_figure(fig, out_base.with_suffix(".pdf"))
    plt.close(fig)
    return True


def plot_all_samples(*, frame: pd.DataFrame, out_dir: Path, fiber_names_config: Path) -> int:
    """Write one outlier-filtered position plot per sample."""
    fiber_names = read_fiber_name_map(fiber_names_config)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for sample in sorted(frame["sample"].dropna().unique(), key=fiber_names.real_name):
        if plot_sample_positions(
            sample=sample,
            frame=frame,
            out_dir=out_dir,
            display_name=fiber_names.real_name(sample),
        ):
            count += 1
    return count


def normalized_trace(values: np.ndarray) -> np.ndarray:
    """Scale one trace to its finite peak value."""
    peak = float(np.nanmax(values)) if values.size else float("nan")
    if not np.isfinite(peak) or peak <= 0.0:
        return values
    return values / peak


def whole_spectrum_fit_data(row: pd.Series, cuts_dir: Path) -> dict[str, object] | None:
    """Load one whole-spectrum trace and refit it for plotting."""
    scan_dir = cuts_dir / str(row["output_folder"])
    profiles = read_profiles(scan_dir)
    band_columns = profile_band_columns(profiles)
    if not band_columns:
        return None
    time_ns = profiles["time_ns"].to_numpy(dtype=float)
    counts = profiles[band_columns].sum(axis=1).to_numpy(dtype=float)
    fit_row = forced_rise_fit_row(synthetic_rise_row(), time_ns, counts)
    fit_start = float(fit_row["fit_start_ns"])
    fit_end = float(fit_row["fit_end_ns"])
    x_dense = np.linspace(fit_start, fit_end, 400)
    y_dense = sigmoid_rise(
        x_dense,
        float(fit_row["amplitude_counts"]),
        float(fit_row["sigmoid_k_ns"]),
        float(fit_row["baseline_counts"]),
        float(fit_row["midpoint_time_ns"]),
    )
    return {
        "time_ns": time_ns,
        "counts": counts,
        "x_dense": x_dense,
        "y_dense": y_dense,
        "midpoint_time_ns": float(fit_row["midpoint_time_ns"]),
        "threshold_20_time_ns": float(row["fitted_threshold_20_time_ns"]),
        "threshold_80_time_ns": float(row["fitted_threshold_80_time_ns"]),
        "r_squared": float(fit_row["r_squared"]),
        "rmse_counts": float(fit_row["rmse_counts"]),
        "fit_points": int(fit_row["fit_points"]),
    }


def plot_individual_inspection(
    *,
    row: pd.Series,
    cuts_dir: Path,
    out_root: Path,
) -> dict[str, object] | None:
    """Write one individual whole-spectrum fit plot for inspection."""
    fit_data = whole_spectrum_fit_data(row, cuts_dir)
    if fit_data is None:
        return None

    sample = str(row["sample"])
    output_folder = Path(str(row["output_folder"]))
    out_dir = out_root / output_folder
    out_base = out_dir / "whole_spectrum_leading_edge_fit"
    y = normalized_trace(fit_data["counts"])
    y_fit = normalized_trace(fit_data["y_dense"])

    set_publication_style(base_font_size=8.7)
    fig, ax = plt.subplots(figsize=(5.25, 3.55), constrained_layout=True)
    ax.plot(fit_data["time_ns"], y, "-", color=COLORS["gray"], lw=0.8, alpha=0.72, label="whole spectrum")
    ax.plot(fit_data["time_ns"], y, "o", color=COLORS["black"], ms=1.35, alpha=0.22, label="_nolegend_")
    ax.plot(fit_data["x_dense"], y_fit, color=COLORS["vermillion"], lw=1.35, label="sigmoid fit")
    ax.axvspan(
        fit_data["threshold_20_time_ns"],
        fit_data["threshold_80_time_ns"],
        color=COLORS["teal"],
        alpha=0.10,
        lw=0,
        label="20-80%",
    )
    ax.axvline(
        fit_data["midpoint_time_ns"],
        color=COLORS["blue"],
        lw=0.95,
        label="20-80 midpoint",
    )
    exclusion = str(row.get("plot_exclusion_reason", "")).strip()
    status_text = f"\nplot excluded: {exclusion}" if exclusion else ""
    ax.text(
        0.985,
        0.94,
        (
            f"t = {fit_data['midpoint_time_ns']:.4g} ns\n"
            f"R^2 = {fit_data['r_squared']:.4f}\n"
            f"points = {fit_data['fit_points']}{status_text}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.6,
        bbox={"boxstyle": "square,pad=0.25", "facecolor": "white", "edgecolor": "#C9CED6", "alpha": 0.9},
    )
    ax.set_title(f"{sample} | {row['position']} | whole spectrum", pad=4)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Normalized intensity")
    ax.set_ylim(-0.04, 1.10)
    ax.legend(loc="lower left", frameon=True, framealpha=0.88, facecolor="white", edgecolor="#C9CED6")
    apply_axes_style(ax, grid=True)
    save_figure(fig, out_base.with_suffix(".png"), dpi=240)
    save_figure(fig, out_base.with_suffix(".pdf"))
    plt.close(fig)
    return {
        "sample": sample,
        "position": row["position"],
        "source_file": row["source_file"],
        "output_folder": row["output_folder"],
        "plot_outlier": row["plot_outlier"],
        "plot_exclusion_reason": row["plot_exclusion_reason"],
        "png_plot": out_base.with_suffix(".png").relative_to(out_root).as_posix(),
        "pdf_plot": out_base.with_suffix(".pdf").relative_to(out_root).as_posix(),
    }


def plot_individual_inspections(*, frame: pd.DataFrame, cuts_dir: Path, out_root: Path) -> int:
    """Write individual whole-spectrum inspection plots arranged by sample and measurement."""
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for _, row in tqdm(
        frame.iterrows(),
        total=len(frame),
        desc="Whole-spectrum QA plots",
        unit="plot",
        dynamic_ncols=True,
        ascii=True,
    ):
        result = plot_individual_inspection(row=row, cuts_dir=cuts_dir, out_root=out_root)
        if result is not None:
            rows.append(result)
    write_csv(
        out_root / "whole_spectrum_inspection_plot_inventory.csv",
        rows,
        [
            "sample",
            "position",
            "source_file",
            "output_folder",
            "plot_outlier",
            "plot_exclusion_reason",
            "png_plot",
            "pdf_plot",
        ],
    )
    return len(rows)


def plot_selected_trace_panels(
    *,
    frame: pd.DataFrame,
    cuts_dir: Path,
    out_dir: Path,
    fiber_names_config: Path,
) -> int:
    """Write one three-panel time trace plot per sample for 200, 100, and 37 cm."""
    fiber_names = read_fiber_name_map(fiber_names_config)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for sample in sorted(frame["sample"].dropna().unique(), key=fiber_names.real_name):
        sample_frame = frame[frame["sample"] == sample].copy()
        panel_data = []
        x_limits = []
        for position in SELECTED_TRACE_POSITIONS:
            position_rows = sample_frame[sample_frame["position"] == position]
            if position_rows.empty:
                panel_data.append({"position": position, "status": "missing"})
                continue
            row = position_rows.iloc[0]
            if bool(row.get("plot_outlier", False)):
                panel_data.append(
                    {
                        "position": position,
                        "status": "excluded",
                        "reason": str(row.get("plot_exclusion_reason", "plot_outlier")),
                    }
                )
                continue

            fit_data = whole_spectrum_fit_data(row, cuts_dir)
            if fit_data is None:
                panel_data.append({"position": position, "status": "missing"})
                continue
            panel_data.append(
                {
                    "position": position,
                    "status": "ok",
                    **fit_data,
                }
            )
            finite_time = fit_data["time_ns"][np.isfinite(fit_data["time_ns"])]
            if finite_time.size:
                x_limits.append((float(np.min(finite_time)), float(np.max(finite_time))))

        if not any(item["status"] == "ok" for item in panel_data):
            continue

        x_min = min(limit[0] for limit in x_limits) if x_limits else 0.0
        x_max = max(limit[1] for limit in x_limits) if x_limits else 2.0
        set_publication_style(base_font_size=9.4)
        fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(6.9, 6.35),
            sharex=True,
            constrained_layout=True,
        )
        legend_ax = None
        for ax, item in zip(axes, panel_data):
            position = item["position"]
            ax.set_ylabel(position)
            if item["status"] == "ok":
                if legend_ax is None:
                    legend_ax = ax
                y = normalized_trace(item["counts"])
                y_fit = normalized_trace(item["y_dense"])
                ax.plot(item["time_ns"], y, "-", color=COLORS["gray"], lw=0.8, alpha=0.72, label="whole spectrum")
                ax.plot(item["time_ns"], y, "o", color=COLORS["black"], ms=1.4, alpha=0.25, label="_nolegend_")
                ax.plot(item["x_dense"], y_fit, color=COLORS["vermillion"], lw=1.35, label="sigmoid fit")
                ax.axvline(
                    item["midpoint_time_ns"],
                    color=COLORS["blue"],
                    ls="-",
                    lw=0.95,
                    alpha=0.9,
                    label="20-80 midpoint",
                )
                ax.axvspan(
                    item["threshold_20_time_ns"],
                    item["threshold_80_time_ns"],
                    color=COLORS["teal"],
                    alpha=0.10,
                    lw=0,
                    label="20-80%",
                )
                ax.text(
                    0.985,
                    0.86,
                    f"t = {item['midpoint_time_ns']:.3g} ns",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8.2,
                    color=COLORS["black"],
                )
                ax.set_ylim(-0.04, 1.10)
            else:
                message = "missing" if item["status"] == "missing" else f"excluded: {item.get('reason', '')}"
                ax.text(
                    0.5,
                    0.5,
                    message,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9.0,
                    color=COLORS["gray"],
                )
                ax.set_ylim(0.0, 1.0)
            apply_axes_style(ax, grid=True, minor_ticks=False)
            ax.set_xlim(x_min, x_max)
        axes[0].set_title(f"Whole-spectrum leading edge traces: {fiber_names.real_name(sample)}", pad=5)
        axes[-1].set_xlabel("Time (ns)")
        fig.supylabel("Normalized intensity")
        if legend_ax is not None:
            legend_ax.legend(loc="upper left", frameon=True, framealpha=0.88, facecolor="white", edgecolor="#C9CED6")
        out_base = out_dir / f"{sample}_whole_spectrum_leading_edge_trace_panels"
        save_figure(fig, out_base.with_suffix(".png"), dpi=260)
        save_figure(fig, out_base.with_suffix(".pdf"))
        plt.close(fig)
        count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    """Write whole-spectrum leading-edge reports and outlier-excluded plots."""
    parser = argparse.ArgumentParser(description="Fit leading-edge timing for whole 2 ns streak spectra.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--cuts-subdir", default=DEFAULT_CUTS_SUBDIR)
    parser.add_argument("--fit-subdir", default=DEFAULT_OUT_SUBDIR)
    parser.add_argument("--rise-window", default="2ns")
    parser.add_argument("--fiber-names-config", type=Path, default=DEFAULT_FIBER_NAMES_CONFIG)
    parser.add_argument("--robust-z-limit", type=float, default=3.5)
    args = parser.parse_args(argv)

    results_dir = resolve_path(args.results_dir)
    cuts_dir = results_dir / args.cuts_subdir
    fit_dir = results_dir / args.fit_subdir
    out_dir = fit_dir / "whole_spectrum_leading_edge_2ns"
    inspection_plot_dir = out_dir / "individual_inspection_plots"
    plot_dir = fit_dir / "summary grids" / "whole_spectrum_leading_edge_2ns"
    trace_plot_dir = fit_dir / "summary grids" / "whole_spectrum_leading_edge_2ns_trace_panels"
    fiber_names_config = resolve_path(args.fiber_names_config)

    rows = build_rows(cuts_dir=cuts_dir, rise_window=args.rise_window, robust_z_limit=args.robust_z_limit)
    frame = pd.DataFrame(rows)
    csv_path = out_dir / "whole_spectrum_leading_edge_20_80_2ns.csv"
    table_path = out_dir / "whole_spectrum_leading_edge_20_80_2ns.txt"
    plot_csv_path = out_dir / "whole_spectrum_leading_edge_20_80_2ns_plot_included.csv"
    write_csv(csv_path, rows, REPORT_FIELDS)
    write_tsv(table_path, rows, REPORT_FIELDS)
    included = frame[~frame["plot_outlier"].astype(bool)].copy()
    write_csv(plot_csv_path, included.to_dict("records"), REPORT_FIELDS)
    inspection_plot_count = plot_individual_inspections(
        frame=frame,
        cuts_dir=cuts_dir,
        out_root=inspection_plot_dir,
    )
    plot_count = plot_all_samples(frame=frame, out_dir=plot_dir, fiber_names_config=fiber_names_config)
    trace_plot_count = plot_selected_trace_panels(
        frame=frame,
        cuts_dir=cuts_dir,
        out_dir=trace_plot_dir,
        fiber_names_config=fiber_names_config,
    )
    outlier_count = int(frame["plot_outlier"].astype(bool).sum()) if not frame.empty else 0
    write_tsv(
        fit_dir / "whole_spectrum_leading_edge_inventory.txt",
        [
            {
                "metric": "whole_spectrum_leading_edge_midpoint_20_80",
                "time_window": args.rise_window,
                "row_count": len(rows),
                "plot_included_count": len(included),
                "plot_excluded_count": outlier_count,
                "inspection_plot_count": inspection_plot_count,
                "summary_plot_count": plot_count,
                "trace_panel_plot_count": trace_plot_count,
                "robust_z_limit": args.robust_z_limit,
            }
        ],
        [
            "metric",
            "time_window",
            "row_count",
            "plot_included_count",
            "plot_excluded_count",
            "inspection_plot_count",
            "summary_plot_count",
            "trace_panel_plot_count",
            "robust_z_limit",
        ],
    )
    print(f"whole-spectrum rows ({args.rise_window}): {len(rows)}")
    print(f"plot-included rows: {len(included)}")
    print(f"plot-excluded rows: {outlier_count}")
    print(f"csv: {csv_path}")
    print(f"inspection plots: {inspection_plot_dir}")
    print(f"plots: {plot_dir}")
    print(f"trace panel plots: {trace_plot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
