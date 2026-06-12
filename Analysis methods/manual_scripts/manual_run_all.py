from __future__ import annotations

from manual_common import (
    CONFIGS_DIR,
    METHODS_DIR,
    add_flag,
    add_optional,
    add_repeated,
    common_analysis_args,
    show_settings,
    use_methods_package,
)


# =========================
# Manual analysis preamble
# =========================
# Edit values in this block, then run this file directly.
# Use None to keep the normal project defaults.

MANIFEST = METHODS_DIR / "raw_data_manifest.json"
RUN_CONFIG = CONFIGS_DIR / "run_all.yaml"

CARPET_TIME_WINDOWS = None
IT_TIME_WINDOW = None
PL_X_MIN_NM = None
PL_X_MAX_NM = None
SKIP_RAW_CHECK = False

# Examples:
# CARPET_TIME_WINDOWS = ("10ns",)
# CARPET_TIME_WINDOWS = ("2ns", "10ns")
# IT_TIME_WINDOW = "10ns"
# PL_X_MIN_NM = 400.0
# PL_X_MAX_NM = 650.0
# SKIP_RAW_CHECK = True


# =========================
# Script body
# =========================

use_methods_package()

from lhcb_fibers_analysis import run_all  # noqa: E402


def main() -> int:
    show_settings(
        "Manual full analysis",
        [
            ("raw data", "raw data"),
            ("results", "Analysis results"),
            ("carpet windows", CARPET_TIME_WINDOWS),
            ("IT window", IT_TIME_WINDOW),
            ("PL x min nm", PL_X_MIN_NM),
            ("PL x max nm", PL_X_MAX_NM),
            ("skip raw check", SKIP_RAW_CHECK),
        ],
    )
    args = common_analysis_args() + [
        "--manifest",
        str(MANIFEST),
        "--config",
        str(RUN_CONFIG),
    ]
    add_repeated(args, "--carpet-time-window", CARPET_TIME_WINDOWS)
    add_optional(args, "--it-time-window", IT_TIME_WINDOW)
    add_optional(args, "--pl-x-min-nm", PL_X_MIN_NM)
    add_optional(args, "--pl-x-max-nm", PL_X_MAX_NM)
    add_flag(args, "--skip-raw-check", SKIP_RAW_CHECK)
    return run_all.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
