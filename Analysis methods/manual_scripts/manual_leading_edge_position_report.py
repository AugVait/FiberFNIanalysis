from __future__ import annotations

from manual_common import FIBER_NAMES_CONFIG, RESULTS_DIR, show_settings, use_methods_package


# =========================
# Manual analysis preamble
# =========================

CUTS_SUBDIR = "carpet_wavelength_cuts_20nm_txt"
FIT_SUBDIR = "wavelength_cut_fit_results_2ns_rise_10ns_decay"
RISE_WINDOW = "2ns"
SELECTION_SUBDIR = "manual selections"


# =========================
# Script body
# =========================

use_methods_package()

from lhcb_fibers_analysis import leading_edge_position_report  # noqa: E402


def main() -> int:
    show_settings(
        "Manual leading-edge position report",
        [
            ("cuts", RESULTS_DIR / CUTS_SUBDIR),
            ("fit results", RESULTS_DIR / FIT_SUBDIR),
            ("rise window", RISE_WINDOW),
            ("manual selections", RESULTS_DIR / FIT_SUBDIR / SELECTION_SUBDIR),
            ("fiber names", FIBER_NAMES_CONFIG),
        ],
    )
    args = [
        "--results-dir",
        str(RESULTS_DIR),
        "--cuts-subdir",
        CUTS_SUBDIR,
        "--fit-subdir",
        FIT_SUBDIR,
        "--rise-window",
        RISE_WINDOW,
        "--selection-subdir",
        SELECTION_SUBDIR,
        "--fiber-names-config",
        str(FIBER_NAMES_CONFIG),
    ]
    return leading_edge_position_report.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
