from __future__ import annotations

from manual_common import FIBER_NAMES_CONFIG, MANUAL_SELECTIONS_DIR, RESULTS_DIR, show_settings, use_methods_package


# =========================
# Manual analysis preamble
# =========================

CUTS_SUBDIR = "carpet_wavelength_cuts_20nm_txt"
FIT_SUBDIR = "wavelength_cut_fit_results_2ns_rise_10ns_decay"
OUT_FAMILY = "leading_edge_position_2ns_arithmetic"
RISE_WINDOW = "2ns"
SELECTION_DIR = MANUAL_SELECTIONS_DIR


# =========================
# Script body
# =========================

use_methods_package()

from lhcb_fibers_analysis import arithmetic_leading_edge_position_report  # noqa: E402


def main() -> int:
    show_settings(
        "Manual arithmetic leading-edge position report",
        [
            ("cuts", RESULTS_DIR / CUTS_SUBDIR),
            ("fit results", RESULTS_DIR / FIT_SUBDIR),
            ("output family", OUT_FAMILY),
            ("rise window", RISE_WINDOW),
            ("manual selections", SELECTION_DIR),
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
        "--out-family",
        OUT_FAMILY,
        "--rise-window",
        RISE_WINDOW,
        "--selection-subdir",
        str(SELECTION_DIR),
        "--fiber-names-config",
        str(FIBER_NAMES_CONFIG),
    ]
    return arithmetic_leading_edge_position_report.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
