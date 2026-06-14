from __future__ import annotations

from manual_common import (
    CONFIGS_DIR,
    FIBER_NAMES_CONFIG,
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

CONFIG = CONFIGS_DIR / "peak_position_shift.yaml"
CONFIG_DIR_OVERRIDE = None
OUT_SUBDIR_OVERRIDE = None
SMOOTH_SIGMA_NM_OVERRIDE = None
FIT_HALF_WIDTH_NM_OVERRIDE = None
EXCLUDE_POINTS_OVERRIDE = None


# =========================
# Script body
# =========================

use_methods_package()

from lhcb_fibers_analysis import peak_position_shift  # noqa: E402


def main() -> None:
    show_settings(
        "Manual peak-position shift",
        [
            ("config", CONFIG),
            ("PL config dir override", CONFIG_DIR_OVERRIDE),
            ("out folder override", OUT_SUBDIR_OVERRIDE),
            ("smooth sigma nm override", SMOOTH_SIGMA_NM_OVERRIDE),
            ("fit half width nm override", FIT_HALF_WIDTH_NM_OVERRIDE),
            ("exclude points override", EXCLUDE_POINTS_OVERRIDE),
        ],
    )
    args = common_analysis_args() + [
        "--config",
        str(CONFIG),
        "--fiber-names-config",
        str(FIBER_NAMES_CONFIG),
    ]
    add_optional(args, "--config-dir", CONFIG_DIR_OVERRIDE)
    add_optional(args, "--out-subdir", OUT_SUBDIR_OVERRIDE)
    add_optional(args, "--smooth-sigma-nm", SMOOTH_SIGMA_NM_OVERRIDE)
    add_optional(args, "--fit-half-width-nm", FIT_HALF_WIDTH_NM_OVERRIDE)
    if EXCLUDE_POINTS_OVERRIDE is not None:
        for exclude_point in EXCLUDE_POINTS_OVERRIDE:
            args.extend(["--exclude-point", exclude_point])
    peak_position_shift.main(args)


if __name__ == "__main__":
    main()
