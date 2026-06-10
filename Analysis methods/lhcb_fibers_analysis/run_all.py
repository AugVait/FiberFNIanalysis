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
    path = Path(value)
    if path.is_absolute():
        return path
    return config_root / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce all generated analysis results.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_RUN_ALL_CONFIG)
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
    pl_config_dir = config_path(config_root, string_value(run_config, "pl_config_dir", "pl_spectra"))

    if not args.skip_raw_check:
        print("Checking raw data before generating results...")
        result = check_manifest(raw_dir, manifest)
        if not result.ok:
            print("raw data check failed; run this for details:")
            print(f"  python -m lhcb_fibers_analysis.raw_data_manifest check --raw-dir \"{raw_dir}\"")
            return 1
        print("raw data check: OK")

    from . import fit_it_decay, plot_pl_spectra, visualize_carpets

    common = ["--raw-dir", str(raw_dir), "--results-dir", str(results_dir)]

    if bool_value(run_config, "run_carpets", True):
        print()
        print("Generating Hamamatsu carpet visualizations...")
        visualize_carpets.main(common + ["--config", str(carpets_config)])

    if bool_value(run_config, "run_pl_normalized", True):
        print()
        print("Generating normalized PL spectra...")
        plot_pl_spectra.main(common + ["--config-dir", str(pl_config_dir), "--intensity-mode", "normalized"])

    if bool_value(run_config, "run_pl_raw", True):
        print()
        print("Generating raw PL spectra...")
        plot_pl_spectra.main(common + ["--config-dir", str(pl_config_dir), "--intensity-mode", "raw"])

    if bool_value(run_config, "run_it_decay", True):
        print()
        print("Fitting IT decay traces...")
        fit_it_decay.main(common + ["--config", str(it_decay_config)])

    print()
    print(f"All reproducible results written under: {results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
