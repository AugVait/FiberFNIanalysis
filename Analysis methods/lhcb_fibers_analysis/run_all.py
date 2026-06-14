from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .paths import (
    DEFAULT_CONFIGS_DIR,
    DEFAULT_MANIFEST,
    DEFAULT_RAW_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_RUN_ALL_CONFIG,
    resolve_path,
)
from .raw_data_manifest import check_manifest
from .yaml_config import bool_value, read_yaml_mapping, string_value


def config_path(config_root: Path, value: str) -> Path:
    """Resolve a run-all config path relative to the config root."""
    path = Path(value)
    if path.is_absolute():
        return path
    return config_root / path


def split_csv_values(value: str) -> list[str]:
    """Split a comma-separated scalar config value into non-empty strings."""
    return [item.strip() for item in value.split(",") if item.strip()]


def optional_float(values: dict[str, object], key: str) -> float | None:
    """Read an optional float from a scalar config mapping."""
    value = values.get(key, "")
    if value in {"", None}:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def add_optional_arg(args: list[str], name: str, value: object | None) -> None:
    """Append one optional command argument when a value is present."""
    if value is not None and value != "":
        args.extend([name, str(value)])


def add_repeated_arg(args: list[str], name: str, values: list[str] | None) -> None:
    """Append repeated command arguments for a list of values."""
    for value in values or []:
        args.extend([name, value])


def base_analysis_args(raw_dir: Path, results_dir: Path) -> list[str]:
    """Return arguments shared by all analysis modules."""
    return ["--raw-dir", str(raw_dir), "--results-dir", str(results_dir)]


def run_pl_spectra(
    plot_module,
    common_args: list[str],
    pl_config_dir: Path,
    fiber_names_config: Path,
    intensity_mode: str,
    pl_x_min_nm: float | None,
    pl_x_max_nm: float | None,
) -> None:
    """Run one PL spectra mode with shared config and optional axis overrides."""
    args = common_args + [
        "--config-dir",
        str(pl_config_dir),
        "--fiber-names-config",
        str(fiber_names_config),
        "--intensity-mode",
        intensity_mode,
    ]
    add_optional_arg(args, "--x-min-nm", pl_x_min_nm)
    add_optional_arg(args, "--x-max-nm", pl_x_max_nm)
    plot_module.main(args)


def main(argv: list[str] | None = None) -> int:
    """Run the full reproducible analysis workflow."""
    parser = argparse.ArgumentParser(description="Reproduce all generated analysis results.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_RUN_ALL_CONFIG)
    parser.add_argument(
        "--carpet-time-window",
        "--carpet-time-windows",
        dest="carpet_time_windows",
        action="append",
        default=None,
        help="Filter carpet quicklooks to one or more acquisition windows, e.g. 10ns or 2ns,10ns.",
    )
    parser.add_argument("--it-time-window", default=None, help="Override the integrated-time trace window.")
    parser.add_argument("--pl-x-min-nm", type=float, default=None, help="Override PL plot x-axis minimum.")
    parser.add_argument("--pl-x-max-nm", type=float, default=None, help="Override PL plot x-axis maximum.")
    parser.add_argument("--skip-raw-check", action="store_true")
    args = parser.parse_args(argv)

    raw_dir = resolve_path(args.raw_dir)
    results_dir = resolve_path(args.results_dir)
    manifest = resolve_path(args.manifest)
    run_config_path = resolve_path(args.config)
    run_config = read_yaml_mapping(run_config_path)
    config_root = run_config_path.parent if run_config_path.exists() else DEFAULT_CONFIGS_DIR

    carpets_config = config_path(config_root, string_value(run_config, "carpets_config", "carpets.yaml"))
    it_decay_config = config_path(config_root, string_value(run_config, "it_decay_config", "it_decay_fits_10ns.yaml"))
    peak_position_shift_config = config_path(
        config_root,
        string_value(run_config, "peak_position_shift_config", "peak_position_shift.yaml"),
    )
    pl_config_dir = config_path(config_root, string_value(run_config, "pl_config_dir", "pl_spectra"))
    fiber_names_config = config_path(config_root, string_value(run_config, "fiber_names_config", "fiber_names.yaml"))
    carpet_time_windows = args.carpet_time_windows
    configured_carpet_windows = split_csv_values(string_value(run_config, "carpet_time_windows", ""))
    if carpet_time_windows is None and configured_carpet_windows:
        carpet_time_windows = configured_carpet_windows
    it_time_window = args.it_time_window or string_value(run_config, "it_time_window", "")
    pl_x_min_nm = args.pl_x_min_nm if args.pl_x_min_nm is not None else optional_float(run_config, "pl_x_min_nm")
    pl_x_max_nm = args.pl_x_max_nm if args.pl_x_max_nm is not None else optional_float(run_config, "pl_x_max_nm")

    if not args.skip_raw_check:
        print("Checking raw data before generating results...")
        result = check_manifest(raw_dir, manifest)
        if not result.ok:
            print("raw data check failed; run this for details:")
            print(f"  python -m lhcb_fibers_analysis.raw_data_manifest check --raw-dir \"{raw_dir}\"")
            return 1
        print("raw data check: OK")

    from . import fit_it_decay, peak_position_shift, plot_pl_spectra, visualize_carpets

    common = base_analysis_args(raw_dir, results_dir)

    if bool_value(run_config, "run_carpets", True):
        print()
        print("Generating Hamamatsu carpet visualizations...")
        carpet_args = common + ["--config", str(carpets_config), "--fiber-names-config", str(fiber_names_config)]
        add_repeated_arg(carpet_args, "--time-window", carpet_time_windows)
        visualize_carpets.main(carpet_args)

    if bool_value(run_config, "run_pl_normalized", True):
        print()
        print("Generating normalized PL spectra...")
        run_pl_spectra(
            plot_pl_spectra,
            common,
            pl_config_dir,
            fiber_names_config,
            "normalized",
            pl_x_min_nm,
            pl_x_max_nm,
        )

    if bool_value(run_config, "run_pl_raw", True):
        print()
        print("Generating raw PL spectra...")
        run_pl_spectra(
            plot_pl_spectra,
            common,
            pl_config_dir,
            fiber_names_config,
            "raw",
            pl_x_min_nm,
            pl_x_max_nm,
        )

    if bool_value(run_config, "run_peak_position_shift", True):
        print()
        print("Generating PL peak-position shift figure...")
        peak_args = common + [
            "--config",
            str(peak_position_shift_config),
            "--fiber-names-config",
            str(fiber_names_config),
        ]
        peak_position_shift.main(peak_args)

    if bool_value(run_config, "run_it_decay", True):
        print()
        print("Fitting IT decay traces...")
        fit_args = common + ["--config", str(it_decay_config), "--fiber-names-config", str(fiber_names_config)]
        add_optional_arg(fit_args, "--time-window", it_time_window)
        fit_it_decay.main(fit_args)

    print()
    print(f"All reproducible results written under: {results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
