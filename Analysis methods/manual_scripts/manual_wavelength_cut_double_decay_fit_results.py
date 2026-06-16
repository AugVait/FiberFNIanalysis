from __future__ import annotations

from manual_common import RESULTS_DIR, show_settings, use_methods_package


# =========================
# Manual analysis preamble
# =========================
# Fits only 10 ns decay cuts enabled in manual selections.

CUTS_SUBDIR = "carpet_wavelength_cuts_20nm_txt"
OUT_SUBDIR = "wavelength_cut_fit_results_2ns_rise_10ns_decay"
DECAY_WINDOW = "10ns"
SELECTION_SUBDIR = "manual selections"


# =========================
# Script body
# =========================

use_methods_package()

from lhcb_fibers_analysis import wavelength_cut_fit_results  # noqa: E402


def main() -> int:
    show_settings(
        "Manual selected double-exponential decay fit results",
        [
            ("cuts", RESULTS_DIR / CUTS_SUBDIR),
            ("output", RESULTS_DIR / OUT_SUBDIR / "decay_time_10ns_double_exp"),
            ("decay window", DECAY_WINDOW),
            ("manual selections", RESULTS_DIR / OUT_SUBDIR / SELECTION_SUBDIR / "decay_time_10ns"),
        ],
    )
    args = [
        "--results-dir",
        str(RESULTS_DIR),
        "--cuts-subdir",
        CUTS_SUBDIR,
        "--out-subdir",
        OUT_SUBDIR,
        "--decay-window",
        DECAY_WINDOW,
        "--decay-model",
        "double",
        "--selection-subdir",
        SELECTION_SUBDIR,
    ]
    return wavelength_cut_fit_results.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
