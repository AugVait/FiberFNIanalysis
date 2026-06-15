from __future__ import annotations

from manual_common import RESULTS_DIR, show_settings, use_methods_package


# =========================
# Manual analysis preamble
# =========================
# Edit values in this block, then run this file directly.

CUTS_SUBDIR = "carpet_wavelength_cuts_20nm_txt"
OUT_SUBDIR = "wavelength_cut_fit_results_2ns_rise_10ns_decay"
RISE_WINDOW = "2ns"
DECAY_WINDOW = "10ns"


# =========================
# Script body
# =========================

use_methods_package()

from lhcb_fibers_analysis import wavelength_cut_fit_results  # noqa: E402


def main() -> int:
    show_settings(
        "Manual wavelength-cut fit results",
        [
            ("cuts", RESULTS_DIR / CUTS_SUBDIR),
            ("output", RESULTS_DIR / OUT_SUBDIR),
            ("rise window", RISE_WINDOW),
            ("decay window", DECAY_WINDOW),
        ],
    )
    args = [
        "--results-dir",
        str(RESULTS_DIR),
        "--cuts-subdir",
        CUTS_SUBDIR,
        "--out-subdir",
        OUT_SUBDIR,
        "--rise-window",
        RISE_WINDOW,
        "--decay-window",
        DECAY_WINDOW,
    ]
    return wavelength_cut_fit_results.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
