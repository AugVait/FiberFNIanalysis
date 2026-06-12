from __future__ import annotations

from manual_common import (
    CONFIGS_DIR,
    FIBER_NAMES_CONFIG,
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

CONFIG = CONFIGS_DIR / "carpets.yaml"
SCAN_CONFIG_DIR = None
OUT_SUBDIR = None
TIME_WINDOWS = None

REFRESH_CONFIGS = False

# Examples:
# TIME_WINDOWS = ("10ns",)
# TIME_WINDOWS = ("2ns", "10ns")
# OUT_SUBDIR = "carpets_10ns_only"
# REFRESH_CONFIGS = True


# =========================
# Script body
# =========================

use_methods_package()

from lhcb_fibers_analysis import visualize_carpets  # noqa: E402


def main() -> None:
    show_settings(
        "Manual carpet quicklooks",
        [
            ("time windows", TIME_WINDOWS),
            ("output folder", OUT_SUBDIR),
            ("refresh configs", REFRESH_CONFIGS),
        ],
    )
    args = common_analysis_args() + [
        "--config",
        str(CONFIG),
        "--fiber-names-config",
        str(FIBER_NAMES_CONFIG),
    ]
    add_optional(args, "--scan-config-dir", SCAN_CONFIG_DIR)
    add_optional(args, "--out-subdir", OUT_SUBDIR)
    add_repeated(args, "--time-window", TIME_WINDOWS)
    add_flag(args, "--refresh-configs", REFRESH_CONFIGS)
    visualize_carpets.main(args)


if __name__ == "__main__":
    main()
