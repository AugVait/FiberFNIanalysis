from __future__ import annotations

from manual_common import FIBER_NAMES_CONFIG, RESULTS_DIR, show_settings, use_methods_package


# =========================
# Manual analysis preamble
# =========================

CUTS_SUBDIR = "carpet_wavelength_cuts_20nm_txt"
FIT_SUBDIR = "wavelength_cut_fit_results_2ns_rise_10ns_decay"
RISE_WINDOW = "2ns"
ROBUST_Z_LIMIT = 3.5


# =========================
# Script body
# =========================

use_methods_package()

from lhcb_fibers_analysis import whole_spectrum_leading_edge_report  # noqa: E402


def main() -> int:
    show_settings(
        "Manual whole-spectrum leading-edge report",
        [
            ("cuts", RESULTS_DIR / CUTS_SUBDIR),
            ("fit results", RESULTS_DIR / FIT_SUBDIR),
            ("rise window", RISE_WINDOW),
            ("robust z limit", ROBUST_Z_LIMIT),
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
        "--fiber-names-config",
        str(FIBER_NAMES_CONFIG),
        "--robust-z-limit",
        str(ROBUST_Z_LIMIT),
    ]
    return whole_spectrum_leading_edge_report.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
