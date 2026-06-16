from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from .fiber_names import read_fiber_name_map
from .paths import DEFAULT_FIBER_NAMES_CONFIG, DEFAULT_RESULTS_DIR, resolve_path
from .wavelength_cut_fit_results import (
    DEFAULT_CUTS_SUBDIR,
    DEFAULT_OUT_SUBDIR,
    all_rows,
    crossing_time,
    forced_rise_fit_row,
    interval_label,
    profile_column,
    read_profiles,
    source_position,
    source_time_window,
    text_value,
    write_sample_matrices,
    write_tsv,
)
from .wavelength_cut_summary_plots import write_summary_plots


REPORT_FIELDS = [
    "sample",
    "position",
    "interval",
    "source_file",
    "output_folder",
    "band_min_nm",
    "band_max_nm",
    "band_center_nm",
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
    "original_batch_status",
    "original_batch_reason",
    "forced_rise_status",
    "forced_rise_note",
]


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    """Write report rows as comma-separated values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: text_value(row.get(field, "")) for field in fields})


def observed_20_80_times(time_ns: np.ndarray, counts: np.ndarray) -> dict[str, float]:
    """Measure the 20% and 80% leading-edge crossings from the smoothed profile."""
    y = counts.astype(float)
    smooth = gaussian_filter1d(y, sigma=2.0)
    baseline_seed = max(0.0, float(np.nanpercentile(y, 5)))
    signal = smooth - baseline_seed
    peak_idx = int(np.nanargmax(signal))
    peak_height = float(signal[peak_idx])
    effective_height = max(peak_height, float(np.nanmax(signal) - np.nanmin(signal)), 1.0e-6)
    threshold_20_time = crossing_time(time_ns, signal, 0.20 * effective_height, peak_idx)
    threshold_80_time = crossing_time(time_ns, signal, 0.80 * effective_height, peak_idx)
    if math.isfinite(threshold_20_time) and math.isfinite(threshold_80_time):
        midpoint = 0.5 * (threshold_20_time + threshold_80_time)
        width = threshold_80_time - threshold_20_time
    else:
        midpoint = float("nan")
        width = float("nan")
    return {
        "observed_threshold_20_time_ns": threshold_20_time,
        "observed_threshold_80_time_ns": threshold_80_time,
        "observed_midpoint_20_80_ns": midpoint,
        "observed_width_20_80_ns": width,
    }


def report_row(
    *,
    entry: dict[str, str],
    fit_row: pd.Series,
    observed_times: dict[str, float],
) -> dict[str, object]:
    """Build one leading-edge report row."""
    sigmoid_k = float(fit_row["sigmoid_k_ns"])
    midpoint = float(fit_row["midpoint_time_ns"])
    half_width = math.log(4.0) * sigmoid_k
    threshold_20_time = midpoint - half_width
    threshold_80_time = midpoint + half_width
    return {
        "sample": entry["sample"],
        "position": source_position(entry["source_file"]),
        "interval": interval_label(fit_row),
        "source_file": entry["source_file"],
        "output_folder": entry["output_folder"],
        "band_min_nm": fit_row["band_min_nm"],
        "band_max_nm": fit_row["band_max_nm"],
        "band_center_nm": fit_row["band_center_nm"],
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
        "original_batch_status": fit_row.get("original_batch_status", fit_row.get("status", "")),
        "original_batch_reason": fit_row.get("original_batch_reason", fit_row.get("reason", "")),
        "forced_rise_status": fit_row.get("forced_rise_status", ""),
        "forced_rise_note": fit_row.get("forced_rise_note", ""),
    }


def build_rows(*, cuts_dir: Path, rise_window: str) -> list[dict[str, object]]:
    """Compute leading-edge rows from the selected 2 ns wavelength cuts."""
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
    for entry in tqdm(entries, desc="Leading-edge scans", unit="scan", dynamic_ncols=True, ascii=True):
        scan_dir = cuts_dir / entry["output_folder"]
        profiles = read_profiles(scan_dir)
        time_ns = profiles["time_ns"].to_numpy(dtype=float)
        summary = all_rows(scan_dir / "rise_fit_summary.txt")
        for _, row in tqdm(
            summary.iterrows(),
            total=len(summary),
            desc=Path(entry["source_file"]).stem,
            unit="band",
            leave=False,
            dynamic_ncols=True,
            ascii=True,
        ):
            column = profile_column(row)
            if column not in profiles:
                continue
            counts = profiles[column].to_numpy(dtype=float)
            fit_row = forced_rise_fit_row(row, time_ns, counts)
            rows.append(
                report_row(
                    entry=entry,
                    fit_row=fit_row,
                    observed_times=observed_20_80_times(time_ns, counts),
                )
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    """Write leading-edge 20-80% midpoint reports and summary plots."""
    parser = argparse.ArgumentParser(description="Report 2 ns leading-edge 20-80% midpoint positions.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--cuts-subdir", default=DEFAULT_CUTS_SUBDIR)
    parser.add_argument("--fit-subdir", default=DEFAULT_OUT_SUBDIR)
    parser.add_argument("--rise-window", default="2ns")
    parser.add_argument("--selection-subdir", default="manual selections")
    parser.add_argument(
        "--selection-dir",
        type=Path,
        default=None,
        help="Read-only directory containing tracked decay-time selection matrices.",
    )
    parser.add_argument("--fiber-names-config", type=Path, default=DEFAULT_FIBER_NAMES_CONFIG)
    args = parser.parse_args(argv)

    results_dir = resolve_path(args.results_dir)
    cuts_dir = results_dir / args.cuts_subdir
    fit_dir = results_dir / args.fit_subdir
    out_dir = fit_dir / "leading_edge_position_2ns"
    selection_dir = (
        resolve_path(args.selection_dir)
        if args.selection_dir is not None
        else fit_dir / args.selection_subdir / "decay_time_10ns"
    )
    rows = build_rows(cuts_dir=cuts_dir, rise_window=args.rise_window)

    csv_path = out_dir / "leading_edge_position_20_80_2ns.csv"
    table_path = out_dir / "leading_edge_position_20_80_2ns.txt"
    write_csv(csv_path, rows, REPORT_FIELDS)
    write_tsv(table_path, rows, REPORT_FIELDS)
    write_sample_matrices(
        rows=rows,
        value_field="leading_edge_midpoint_20_80_ns",
        out_dir=fit_dir / "tabulated_by_sample" / "leading_edge_position_2ns",
        file_suffix="leading_edge_position_20_80_2ns_by_position_interval",
    )

    fiber_names = read_fiber_name_map(resolve_path(args.fiber_names_config))
    plot_count = write_summary_plots(
        table_path=table_path,
        value_column="leading_edge_midpoint_20_80_ns",
        value_label="Leading-edge midpoint (ns)",
        error_column=None,
        title_prefix="Leading edge position",
        out_dir=fit_dir / "summary grids" / "leading_edge_position_2ns",
        fiber_names=fiber_names,
        selection_dir=selection_dir,
        selection_suffix="decay_time_10ns_by_position_interval",
        create_missing_selections=args.selection_dir is None,
        use_standard_deviation_limits=True,
    )

    write_tsv(
        fit_dir / "leading_edge_position_inventory.txt",
        [
            {
                "metric": "leading_edge_midpoint_20_80",
                "time_window": args.rise_window,
                "row_count": len(rows),
                "summary_plot_count": plot_count,
            }
        ],
        ["metric", "time_window", "row_count", "summary_plot_count"],
    )
    print(f"leading-edge rows ({args.rise_window}): {len(rows)}")
    print(f"csv: {csv_path}")
    print(f"summary plots: {fit_dir / 'summary grids' / 'leading_edge_position_2ns'}")
    print(f"manual selections: {selection_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
