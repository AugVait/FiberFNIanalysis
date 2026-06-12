from __future__ import annotations

from manual_common import (
    CONFIGS_DIR,
    FIBER_NAMES_CONFIG,
    add_flag,
    add_optional,
    common_analysis_args,
    use_methods_package,
)


# =========================
# Manual analysis preamble
# =========================
# Edit values in this block, then run this file directly from Python.
# No command-line arguments are needed.

CONFIG = CONFIGS_DIR / "it_decay_fits_10ns.yaml"
TRACE_CONFIG_DIR = None
OUT_SUBDIR = None
TIME_WINDOW = None  # Firing/acquisition window, e.g. "10ns".

REFRESH_CONFIGS = False

# Examples:
# TIME_WINDOW = "10ns"
# OUT_SUBDIR = "it_decay_fits_10ns_manual"
# REFRESH_CONFIGS = True


# =========================
# Script body
# =========================

use_methods_package()

from lhcb_fibers_analysis import fit_it_decay  # noqa: E402


def main() -> None:
    args = common_analysis_args() + [
        "--config",
        str(CONFIG),
        "--fiber-names-config",
        str(FIBER_NAMES_CONFIG),
    ]
    add_optional(args, "--trace-config-dir", TRACE_CONFIG_DIR)
    add_optional(args, "--out-subdir", OUT_SUBDIR)
    add_optional(args, "--time-window", TIME_WINDOW)
    add_flag(args, "--refresh-configs", REFRESH_CONFIGS)
    fit_it_decay.main(args)


if __name__ == "__main__":
    main()
