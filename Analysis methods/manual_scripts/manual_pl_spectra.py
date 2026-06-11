from __future__ import annotations

import sys
from pathlib import Path


# =========================
# Manual analysis preamble
# =========================
# Edit values in this block, then run this file directly from Python.
# No command-line arguments are needed.

METHODS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = METHODS_DIR.parent

RAW_DIR = PROJECT_ROOT / "raw data"
RESULTS_DIR = PROJECT_ROOT / "Analysis results"
CONFIG_DIR = METHODS_DIR / "configs" / "pl_spectra"
FIBER_NAMES_CONFIG = METHODS_DIR / "configs" / "fiber_names.yaml"

INTENSITY_MODES = ("normalized", "raw")
REFRESH_CONFIGS = False
NO_CLEAN = False

# Leave a mode value as None to use the standard output folder.
OUT_SUBDIR_BY_MODE = {
    "normalized": None,
    "raw": None,
}


# =========================
# Script body
# =========================

sys.path.insert(0, str(METHODS_DIR))

from lhcb_fibers_analysis import plot_pl_spectra  # noqa: E402


def run_mode(intensity_mode: str) -> None:
    args = [
        "--raw-dir",
        str(RAW_DIR),
        "--results-dir",
        str(RESULTS_DIR),
        "--config-dir",
        str(CONFIG_DIR),
        "--fiber-names-config",
        str(FIBER_NAMES_CONFIG),
        "--intensity-mode",
        intensity_mode,
    ]
    out_subdir = OUT_SUBDIR_BY_MODE.get(intensity_mode)
    if out_subdir is not None:
        args.extend(["--out-subdir", str(out_subdir)])
    if REFRESH_CONFIGS:
        args.append("--refresh-configs")
    if NO_CLEAN:
        args.append("--no-clean")
    plot_pl_spectra.main(args)


def main() -> None:
    for intensity_mode in INTENSITY_MODES:
        run_mode(intensity_mode)


if __name__ == "__main__":
    main()
