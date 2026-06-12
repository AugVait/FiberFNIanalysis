from __future__ import annotations

from manual_common import PROJECT_ROOT, add_flag, add_optional, use_methods_package


# =========================
# Manual analysis preamble
# =========================
# Edit values in this block, then run this file directly from Python.
# No command-line arguments are needed.

IMG_PATH = (
    PROJECT_ROOT
    / "raw data"
    / "2026 04 17"
    / "bcf_6"
    / "bcf6_ir_100cm_ex360nm_10nJ_10ns.img"
)
OUT_DIR = None

TOP_EDGE_CROP_ROWS = 12

# Use CENTERS_NM for explicit cuts, or set it to None for min/max/step cuts.
CENTERS_NM = None
WAVELENGTH_MIN_NM = 400.0
WAVELENGTH_MAX_NM = 540.0
STEP_NM = 10.0
BAND_WIDTH_NM = 10.0

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

WRITE_INDIVIDUAL_PLOTS = True

# Examples:
# CENTERS_NM = (420, 440, 460, 480, 500)
# WAVELENGTH_MIN_NM = 380.0
# WAVELENGTH_MAX_NM = 620.0
# STEP_NM = 20.0
# BAND_WIDTH_NM = 20.0
# FIT_START_NS = 1.6
# FIT_END_NS = 7.5
# OUT_DIR = PROJECT_ROOT / "Analysis results" / "manual_carpet_wavelength_cuts" / "bcf6_100cm_selected"
# WRITE_INDIVIDUAL_PLOTS = False


# =========================
# Script body
# =========================

use_methods_package()

from lhcb_fibers_analysis import carpet_wavelength_cuts  # noqa: E402


def main() -> None:
    args = [
        str(IMG_PATH),
        "--top-edge-crop-rows",
        str(TOP_EDGE_CROP_ROWS),
        "--step-nm",
        str(STEP_NM),
        "--band-width-nm",
        str(BAND_WIDTH_NM),
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
    add_optional(args, "--out-dir", OUT_DIR)
    add_optional(args, "--wavelength-min-nm", WAVELENGTH_MIN_NM)
    add_optional(args, "--wavelength-max-nm", WAVELENGTH_MAX_NM)
    add_optional(args, "--fit-start-ns", FIT_START_NS)
    add_optional(args, "--fit-end-ns", FIT_END_NS)
    if CENTERS_NM is not None:
        centers_text = ",".join(str(center) for center in CENTERS_NM)
        args.extend(["--centers", centers_text])
    add_flag(args, "--no-individual-plots", not WRITE_INDIVIDUAL_PLOTS)
    carpet_wavelength_cuts.main(args)


if __name__ == "__main__":
    main()
