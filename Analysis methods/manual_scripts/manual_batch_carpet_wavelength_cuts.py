from __future__ import annotations

from manual_common import (
    RESULTS_DIR,
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

OUT_SUBDIR = "carpet_wavelength_cuts_20nm_txt"

TIME_WINDOWS = ("2ns", "10ns")
INTERVAL_NM = 20.0
RANGE_MODE = "common"  # "common" keeps the same clean bands for every scan; "per-scan" uses each scan's full range.
WAVELENGTH_MIN_NM = 380.0
WAVELENGTH_MAX_NM = 720.0
TOP_EDGE_CROP_ROWS = 12

SMOOTH_SIGMA = 2.0
FIT_START_NS = None
FIT_START_OFFSET_NS = 0.05
FIT_END_NS = None
END_FRACTION = 0.05
MIN_FIT_POINTS = 20
MIN_PEAK_SIGMA = 5.0

SECONDARY_PEAK_HEIGHT_FRACTION = 0.20
SECONDARY_PEAK_PROMINENCE_FRACTION = 0.12
SECONDARY_PEAK_NOISE_SIGMA = 3.0
SECONDARY_PEAK_MIN_SEPARATION_NS = 0.25
SECONDARY_PEAK_EXCLUSION_BEFORE_NS = 0.05

TAU_MIN_NS = 0.03
TAU_MAX_NS = 200.0
WRITE_FIT_CURVES = True
WRITE_SLICE_PLOTS = True

# Examples:
# TIME_WINDOWS = ("2ns", "10ns")
# TIME_WINDOWS = None
# WAVELENGTH_MIN_NM = 400.0
# WAVELENGTH_MAX_NM = 720.0
# RANGE_MODE = "per-scan"
# WRITE_FIT_CURVES = False
# WRITE_SLICE_PLOTS = False


# =========================
# Script body
# =========================

use_methods_package()

from lhcb_fibers_analysis import batch_carpet_wavelength_cuts  # noqa: E402


def main() -> int:
    show_settings(
        "Manual batch carpet wavelength cuts",
        [
            ("output", RESULTS_DIR / OUT_SUBDIR),
            ("time windows", TIME_WINDOWS),
            ("interval nm", INTERVAL_NM),
            ("range mode", RANGE_MODE),
            ("wavelength min nm", WAVELENGTH_MIN_NM),
            ("wavelength max nm", WAVELENGTH_MAX_NM),
            ("fit start ns", FIT_START_NS),
            ("fit end ns", FIT_END_NS),
            ("write fit curves", WRITE_FIT_CURVES),
            ("write slice plots", WRITE_SLICE_PLOTS),
        ],
    )
    args = common_analysis_args() + [
        "--out-subdir",
        OUT_SUBDIR,
        "--interval-nm",
        str(INTERVAL_NM),
        "--range-mode",
        RANGE_MODE,
        "--top-edge-crop-rows",
        str(TOP_EDGE_CROP_ROWS),
        "--smooth-sigma",
        str(SMOOTH_SIGMA),
        "--fit-start-offset-ns",
        str(FIT_START_OFFSET_NS),
        "--end-fraction",
        str(END_FRACTION),
        "--min-fit-points",
        str(MIN_FIT_POINTS),
        "--min-peak-sigma",
        str(MIN_PEAK_SIGMA),
        "--secondary-peak-height-fraction",
        str(SECONDARY_PEAK_HEIGHT_FRACTION),
        "--secondary-peak-prominence-fraction",
        str(SECONDARY_PEAK_PROMINENCE_FRACTION),
        "--secondary-peak-noise-sigma",
        str(SECONDARY_PEAK_NOISE_SIGMA),
        "--secondary-peak-min-separation-ns",
        str(SECONDARY_PEAK_MIN_SEPARATION_NS),
        "--secondary-peak-exclusion-before-ns",
        str(SECONDARY_PEAK_EXCLUSION_BEFORE_NS),
        "--tau-min-ns",
        str(TAU_MIN_NS),
        "--tau-max-ns",
        str(TAU_MAX_NS),
    ]
    add_repeated(args, "--time-window", TIME_WINDOWS)
    add_optional(args, "--wavelength-min-nm", WAVELENGTH_MIN_NM)
    add_optional(args, "--wavelength-max-nm", WAVELENGTH_MAX_NM)
    add_optional(args, "--fit-start-ns", FIT_START_NS)
    add_optional(args, "--fit-end-ns", FIT_END_NS)
    add_flag(args, "--no-fit-curves", not WRITE_FIT_CURVES)
    add_flag(args, "--no-slice-plots", not WRITE_SLICE_PLOTS)
    return batch_carpet_wavelength_cuts.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
