from __future__ import annotations

from manual_common import FIBER_NAMES_CONFIG, RESULTS_DIR, show_settings, use_methods_package


# =========================
# Manual analysis preamble
# =========================

FIT_SUBDIR = "wavelength_cut_fit_results_2ns_rise_10ns_decay"
OUT_SUBDIR = "summary grids"
SELECTION_SUBDIR = "manual selections"


# =========================
# Script body
# =========================

use_methods_package()

from lhcb_fibers_analysis import wavelength_cut_summary_plots  # noqa: E402


def main() -> int:
    show_settings(
        "Manual wavelength-cut summary plots",
        [
            ("fit results", RESULTS_DIR / FIT_SUBDIR),
            ("output", RESULTS_DIR / FIT_SUBDIR / OUT_SUBDIR),
            ("manual selections", RESULTS_DIR / FIT_SUBDIR / SELECTION_SUBDIR),
            ("fiber names", FIBER_NAMES_CONFIG),
        ],
    )
    args = [
        "--results-dir",
        str(RESULTS_DIR),
        "--fit-subdir",
        FIT_SUBDIR,
        "--out-subdir",
        OUT_SUBDIR,
        "--selection-subdir",
        SELECTION_SUBDIR,
        "--fiber-names-config",
        str(FIBER_NAMES_CONFIG),
    ]
    return wavelength_cut_summary_plots.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
