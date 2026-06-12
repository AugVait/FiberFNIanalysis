from __future__ import annotations

from manual_common import (
    CONFIGS_DIR,
    FIBER_NAMES_CONFIG,
    add_flag,
    add_optional,
    common_analysis_args,
    show_settings,
    use_methods_package,
)


# =========================
# Manual analysis preamble
# =========================
# Edit values in this block, then run this file directly.
# Use None to keep the normal project defaults.

CONFIG_DIR = CONFIGS_DIR / "pl_spectra"

INTENSITY_MODES = ("normalized", "raw")
X_MIN_NM = None
X_MAX_NM = None
REFRESH_CONFIGS = False
NO_CLEAN = False

# Leave a mode value as None to use the standard output folder.
OUT_SUBDIR_BY_MODE = {
    "normalized": None,
    "raw": None,
}

# Examples:
# INTENSITY_MODES = ("normalized",)
# X_MIN_NM = 400.0
# X_MAX_NM = 650.0
# OUT_SUBDIR_BY_MODE = {"normalized": "pl_spectra_400_650", "raw": None}
# NO_CLEAN = True


# =========================
# Script body
# =========================

use_methods_package()

from lhcb_fibers_analysis import plot_pl_spectra  # noqa: E402


def run_mode(intensity_mode: str) -> None:
    args = common_analysis_args() + [
        "--config-dir",
        str(CONFIG_DIR),
        "--fiber-names-config",
        str(FIBER_NAMES_CONFIG),
        "--intensity-mode",
        intensity_mode,
    ]
    add_optional(args, "--x-min-nm", X_MIN_NM)
    add_optional(args, "--x-max-nm", X_MAX_NM)
    out_subdir = OUT_SUBDIR_BY_MODE.get(intensity_mode)
    add_optional(args, "--out-subdir", out_subdir)
    add_flag(args, "--refresh-configs", REFRESH_CONFIGS)
    add_flag(args, "--no-clean", NO_CLEAN)
    plot_pl_spectra.main(args)


def main() -> None:
    show_settings(
        "Manual PL spectra",
        [
            ("modes", INTENSITY_MODES),
            ("x min nm", X_MIN_NM),
            ("x max nm", X_MAX_NM),
            ("refresh configs", REFRESH_CONFIGS),
            ("keep old outputs", NO_CLEAN),
        ],
    )
    for intensity_mode in INTENSITY_MODES:
        run_mode(intensity_mode)


if __name__ == "__main__":
    main()
