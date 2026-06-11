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
CONFIG = METHODS_DIR / "configs" / "carpets.yaml"
SCAN_CONFIG_DIR = None
FIBER_NAMES_CONFIG = METHODS_DIR / "configs" / "fiber_names.yaml"
OUT_SUBDIR = None

REFRESH_CONFIGS = False


# =========================
# Script body
# =========================

sys.path.insert(0, str(METHODS_DIR))

from lhcb_fibers_analysis import visualize_carpets  # noqa: E402


def main() -> None:
    args = [
        "--raw-dir",
        str(RAW_DIR),
        "--results-dir",
        str(RESULTS_DIR),
        "--config",
        str(CONFIG),
        "--fiber-names-config",
        str(FIBER_NAMES_CONFIG),
    ]
    if SCAN_CONFIG_DIR is not None:
        args.extend(["--scan-config-dir", str(SCAN_CONFIG_DIR)])
    if OUT_SUBDIR is not None:
        args.extend(["--out-subdir", str(OUT_SUBDIR)])
    if REFRESH_CONFIGS:
        args.append("--refresh-configs")
    visualize_carpets.main(args)


if __name__ == "__main__":
    main()
