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
MANIFEST = METHODS_DIR / "raw_data_manifest.json"
RUN_CONFIG = METHODS_DIR / "configs" / "run_all.yaml"

SKIP_RAW_CHECK = False


# =========================
# Script body
# =========================

sys.path.insert(0, str(METHODS_DIR))

from lhcb_fibers_analysis import run_all  # noqa: E402


def main() -> int:
    args = [
        "--raw-dir",
        str(RAW_DIR),
        "--results-dir",
        str(RESULTS_DIR),
        "--manifest",
        str(MANIFEST),
        "--config",
        str(RUN_CONFIG),
    ]
    if SKIP_RAW_CHECK:
        args.append("--skip-raw-check")
    return run_all.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
