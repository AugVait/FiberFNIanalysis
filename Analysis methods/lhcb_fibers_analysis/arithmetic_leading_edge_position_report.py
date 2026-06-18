from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .fiber_names import read_fiber_name_map
from .paths import DEFAULT_FIBER_NAMES_CONFIG, DEFAULT_MANUAL_SELECTIONS_DIR, DEFAULT_RESULTS_DIR, resolve_path, resolve_selection_root
from .plot_style import COLORS, apply_axes_style, save_figure, set_publication_style
from .wavelength_cut_fit_results import (
    DEFAULT_CUTS_SUBDIR,
    DEFAULT_OUT_SUBDIR,
    all_rows,
    format_nm,
    interval_label,
    profile_column,
    read_profiles,
    source_position,
    source_time_window,
    text_value,
    interval_sort_key,
    position_sort_key,
    write_sample_matrices,
    write_tsv,
)
from .wavelength_cut_summary_plots import write_summary_plots


DEFAULT_OUT_FAMILY = "leading_edge_position_2ns_arithmetic"
SUMMARY_SELECTION_SUFFIX = "leading_edge_position_arithmetic_2ns_by_position_interval"
SUMMARY_FIELDS = [
    "metric",
    "time_window",
    "sample",
    "position",
    "interval",
    "source_file",
    "output_folder",
    "band_min_nm",
    "band_max_nm",
    "band_center_nm",
    "leading_edge_midpoint_20_80_ns",
    "threshold_20_time_ns",
    "threshold_50_time_ns",
    "threshold_80_time_ns",
    "width_20_80_ns",
    "baseline_counts",
    "peak_time_ns",
    "peak_counts",
    "peak_signal_counts",
    "status",
    "reason",
    "qa_plot",
]


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    """Write rows as comma-separated values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: text_value(row.get(field, "")) for field in fields})


def is_repeat_position(position: object) -> bool:
    """Return True for repeat-scan position labels such as 37cmR."""
    return str(position).strip().lower().endswith("r")


def write_selection_tables(selection_dir: Path, rows: list[dict[str, object]], *, overwrite: bool = False) -> int:
    """Write editable include/exclude matrices for arithmetic leading-edge summary plots."""
    if not rows:
        return 0
    frame = pd.DataFrame(rows)
    selection_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for sample in sorted(frame["sample"].dropna().unique()):
        path = selection_dir / f"{sample}_{SUMMARY_SELECTION_SUFFIX}_selection.txt"
        if path.exists() and not overwrite:
            continue
        sample_frame = frame[frame["sample"] == sample].copy()
        positions = sorted(sample_frame["position"].dropna().unique(), key=position_sort_key)
        intervals = sorted(sample_frame["interval"].dropna().unique(), key=interval_sort_key)
        measured = sample_frame.assign(
            include=(
                (sample_frame["status"] == "measured")
                & pd.to_numeric(sample_frame["leading_edge_midpoint_20_80_ns"], errors="coerce").notna()
            ).astype(int)
        )
        matrix = (
            measured.pivot_table(
                index="position",
                columns="interval",
                values="include",
                aggfunc="max",
                fill_value=0,
                dropna=False,
            )
            .reindex(index=positions, columns=intervals)
            .fillna(0)
            .astype(int)
        )
        for position in matrix.index:
            if is_repeat_position(position):
                matrix.loc[position, :] = 0
        matrix.reset_index().to_csv(path, sep="\t", index=False)
        count += 1
    return count


def first_crossing_time(time_ns: np.ndarray, signal: np.ndarray, threshold: float, peak_idx: int) -> float:
    """Return the first linearly interpolated threshold crossing before the peak."""
    if not math.isfinite(threshold):
        return float("nan")
    if peak_idx <= 0:
        return float("nan")
    rising = np.maximum.accumulate(signal[: peak_idx + 1])
    for idx in range(1, len(rising)):
        previous = float(rising[idx - 1])
        current = float(rising[idx])
        if previous < threshold <= current:
            span = current - previous
            if span == 0.0:
                return float(time_ns[idx])
            fraction = (threshold - previous) / span
            return float(time_ns[idx - 1] + fraction * (time_ns[idx] - time_ns[idx - 1]))
    return float("nan")


def arithmetic_leading_edge(time_ns: np.ndarray, counts: np.ndarray) -> dict[str, float | str]:
    """Measure leading-edge timing by direct threshold arithmetic, with no fitting."""
    finite = np.isfinite(time_ns) & np.isfinite(counts)
    if np.count_nonzero(finite) < 3:
        return {"status": "skipped", "reason": "too_few_finite_points"}

    x = time_ns[finite].astype(float)
    y = counts[finite].astype(float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    baseline_count = max(3, min(20, int(math.ceil(0.05 * len(y)))))
    baseline = float(np.nanmedian(y[:baseline_count]))
    signal = y - baseline
    peak_idx = int(np.nanargmax(signal))
    peak_signal = float(signal[peak_idx])
    peak_counts = float(y[peak_idx])
    peak_time = float(x[peak_idx])
    if not math.isfinite(peak_signal) or peak_signal <= 0.0:
        return {
            "status": "skipped",
            "reason": "non_positive_peak_signal",
            "baseline_counts": baseline,
            "peak_time_ns": peak_time,
            "peak_counts": peak_counts,
            "peak_signal_counts": peak_signal,
        }

    threshold_20 = 0.20 * peak_signal
    threshold_50 = 0.50 * peak_signal
    threshold_80 = 0.80 * peak_signal
    threshold_20_time = first_crossing_time(x, signal, threshold_20, peak_idx)
    threshold_50_time = first_crossing_time(x, signal, threshold_50, peak_idx)
    threshold_80_time = first_crossing_time(x, signal, threshold_80, peak_idx)
    if not math.isfinite(threshold_20_time):
        status = "skipped"
        reason = "no_20_percent_crossing"
    elif not math.isfinite(threshold_80_time):
        status = "skipped"
        reason = "no_80_percent_crossing"
    elif threshold_80_time < threshold_20_time:
        status = "skipped"
        reason = "invalid_threshold_order"
    else:
        status = "measured"
        reason = ""

    midpoint = (
        0.5 * (threshold_20_time + threshold_80_time)
        if status == "measured"
        else float("nan")
    )
    width = threshold_80_time - threshold_20_time if status == "measured" else float("nan")
    return {
        "status": status,
        "reason": reason,
        "leading_edge_midpoint_20_80_ns": midpoint,
        "threshold_20_time_ns": threshold_20_time,
        "threshold_50_time_ns": threshold_50_time,
        "threshold_80_time_ns": threshold_80_time,
        "width_20_80_ns": width,
        "baseline_counts": baseline,
        "peak_time_ns": peak_time,
        "peak_counts": peak_counts,
        "peak_signal_counts": peak_signal,
    }


def plot_arithmetic_cut(
    path: Path,
    time_ns: np.ndarray,
    counts: np.ndarray,
    measurement: dict[str, float | str],
    title: str,
) -> None:
    """Plot one arithmetic leading-edge measurement for inspection."""
    baseline = float(measurement.get("baseline_counts", float("nan")))
    peak_signal = float(measurement.get("peak_signal_counts", float("nan")))
    normalized = (counts - baseline) / peak_signal if math.isfinite(peak_signal) and peak_signal > 0 else counts

    set_publication_style(base_font_size=8.3)
    fig, ax = plt.subplots(figsize=(4.85, 3.25), constrained_layout=True)
    ax.plot(time_ns, normalized, "-", color=COLORS["gray"], lw=0.85, alpha=0.78, label="cut profile")
    ax.plot(time_ns, normalized, "o", color=COLORS["black"], ms=1.5, alpha=0.28, label="_nolegend_")

    threshold_20_time = float(measurement.get("threshold_20_time_ns", float("nan")))
    threshold_50_time = float(measurement.get("threshold_50_time_ns", float("nan")))
    threshold_80_time = float(measurement.get("threshold_80_time_ns", float("nan")))
    midpoint = float(measurement.get("leading_edge_midpoint_20_80_ns", float("nan")))
    if math.isfinite(threshold_20_time) and math.isfinite(threshold_80_time):
        ax.axvspan(threshold_20_time, threshold_80_time, color=COLORS["teal"], alpha=0.10, lw=0, label="20-80%")
    if math.isfinite(midpoint):
        ax.axvline(midpoint, color=COLORS["blue"], lw=1.0, label="20-80 midpoint")
    if math.isfinite(threshold_50_time):
        ax.axvline(threshold_50_time, color=COLORS["purple"], lw=0.8, ls=":", label="50% crossing")
    ax.axhline(0.20, color=COLORS["teal"], lw=0.7, ls=":", alpha=0.75)
    ax.axhline(0.80, color=COLORS["teal"], lw=0.7, ls=":", alpha=0.75)

    status = str(measurement.get("status", ""))
    reason = str(measurement.get("reason", ""))
    status_line = f"{status}" if not reason else f"{status}: {reason}"
    ax.text(
        0.985,
        0.94,
        (
            f"t_mid = {midpoint:.4g} ns\n"
            f"t20 = {threshold_20_time:.4g} ns, t80 = {threshold_80_time:.4g} ns\n"
            f"{status_line}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.1,
        bbox={"boxstyle": "square,pad=0.25", "facecolor": "white", "edgecolor": "#C9CED6", "alpha": 0.92},
    )
    ax.set_title(title, pad=4)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Baseline-subtracted / peak")
    ax.set_ylim(-0.08, 1.16)
    ax.legend(loc="lower left", frameon=True, framealpha=0.88, facecolor="white", edgecolor="#C9CED6")
    apply_axes_style(ax, grid=True)
    save_figure(fig, path, dpi=240)
    save_figure(fig, path.with_suffix(".pdf"))
    plt.close(fig)


def build_rows(*, cuts_dir: Path, output_root: Path, rise_window: str) -> list[dict[str, object]]:
    """Measure arithmetic leading-edge timings for every 2 ns wavelength cut."""
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
    for entry in tqdm(entries, desc="Arithmetic leading-edge scans", unit="scan", dynamic_ncols=True, ascii=True):
        scan_dir = cuts_dir / entry["output_folder"]
        profiles = read_profiles(scan_dir)
        time_ns = profiles["time_ns"].to_numpy(dtype=float)
        summary = all_rows(scan_dir / "rise_fit_summary.txt")
        plot_dir = output_root / Path(entry["output_folder"]) / "arithmetic_plots"
        for _, band in tqdm(
            summary.iterrows(),
            total=len(summary),
            desc=Path(entry["source_file"]).stem,
            unit="cut",
            leave=False,
            dynamic_ncols=True,
            ascii=True,
        ):
            column = profile_column(band)
            if column not in profiles:
                continue
            counts = profiles[column].to_numpy(dtype=float)
            measurement = arithmetic_leading_edge(time_ns, counts)
            interval = interval_label(band)
            plot_path = plot_dir / f"leading_edge_{format_nm(band['band_min_nm'])}_{format_nm(band['band_max_nm'])}nm.png"
            plot_arithmetic_cut(
                plot_path,
                time_ns,
                counts,
                measurement,
                f"{Path(entry['source_file']).stem} | {interval}",
            )
            rows.append(
                {
                    "metric": "leading_edge_midpoint_20_80_arithmetic",
                    "time_window": rise_window,
                    "sample": entry["sample"],
                    "position": source_position(entry["source_file"]),
                    "interval": interval,
                    "source_file": entry["source_file"],
                    "output_folder": entry["output_folder"],
                    "band_min_nm": band["band_min_nm"],
                    "band_max_nm": band["band_max_nm"],
                    "band_center_nm": band["band_center_nm"],
                    **measurement,
                    "qa_plot": plot_path.relative_to(output_root).as_posix(),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    """Write arithmetic 2 ns leading-edge timing tables and cut-level QA plots."""
    parser = argparse.ArgumentParser(description="Measure 2 ns leading-edge timing by direct threshold arithmetic.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--cuts-subdir", default=DEFAULT_CUTS_SUBDIR)
    parser.add_argument("--fit-subdir", default=DEFAULT_OUT_SUBDIR)
    parser.add_argument("--out-family", default=DEFAULT_OUT_FAMILY)
    parser.add_argument("--rise-window", default="2ns")
    parser.add_argument("--selection-subdir", type=Path, default=DEFAULT_MANUAL_SELECTIONS_DIR)
    parser.add_argument("--fiber-names-config", type=Path, default=DEFAULT_FIBER_NAMES_CONFIG)
    parser.add_argument("--reset-selection-tables", action="store_true")
    args = parser.parse_args(argv)

    results_dir = resolve_path(args.results_dir)
    cuts_dir = results_dir / args.cuts_subdir
    fit_dir = results_dir / args.fit_subdir
    output_root = fit_dir / args.out_family
    rows = build_rows(cuts_dir=cuts_dir, output_root=output_root, rise_window=args.rise_window)
    table_path = output_root / "leading_edge_position_arithmetic_2ns.txt"
    csv_path = output_root / "leading_edge_position_arithmetic_2ns.csv"
    write_tsv(table_path, rows, SUMMARY_FIELDS)
    write_csv(csv_path, rows, SUMMARY_FIELDS)
    write_sample_matrices(
        rows=rows,
        value_field="leading_edge_midpoint_20_80_ns",
        out_dir=fit_dir / "tabulated_by_sample" / args.out_family,
        file_suffix="leading_edge_position_arithmetic_2ns_by_position_interval",
    )

    fiber_names = read_fiber_name_map(resolve_path(args.fiber_names_config))
    selection_dir = resolve_selection_root(args.selection_subdir, fit_dir) / args.out_family
    selection_count = write_selection_tables(
        selection_dir,
        rows,
        overwrite=args.reset_selection_tables,
    )
    plot_count = write_summary_plots(
        table_path=table_path,
        value_column="leading_edge_midpoint_20_80_ns",
        value_label="Leading-edge midpoint (ns)",
        error_column=None,
        title_prefix="Arithmetic leading edge position",
        out_dir=fit_dir / "summary grids" / args.out_family,
        fiber_names=fiber_names,
        selection_dir=selection_dir,
        selection_suffix=SUMMARY_SELECTION_SUFFIX,
        use_standard_deviation_limits=True,
    )
    measured_count = sum(1 for row in rows if row.get("status") == "measured")
    write_tsv(
        fit_dir / "arithmetic_leading_edge_position_inventory.txt",
        [
            {
                "metric": "leading_edge_midpoint_20_80_arithmetic",
                "time_window": args.rise_window,
                "row_count": len(rows),
                "measured_count": measured_count,
                "skipped_count": len(rows) - measured_count,
                "selection_table_count": selection_count,
                "summary_plot_count": plot_count,
            }
        ],
        [
            "metric",
            "time_window",
            "row_count",
            "measured_count",
            "skipped_count",
            "selection_table_count",
            "summary_plot_count",
        ],
    )
    print(f"arithmetic leading-edge rows ({args.rise_window}): {len(rows)}")
    print(f"measured: {measured_count}")
    print(f"skipped: {len(rows) - measured_count}")
    print(f"selection tables written: {selection_count}")
    print(f"output: {output_root}")
    print(f"summary plots: {fit_dir / 'summary grids' / args.out_family}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
