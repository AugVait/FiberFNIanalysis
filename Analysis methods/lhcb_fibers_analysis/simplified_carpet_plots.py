from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .fiber_names import read_fiber_name_map
from .hamamatsu_streak import (
    TOP_EDGE_CROP_ROWS,
    image_extent,
    load_img,
    time_axis_ns,
    visible_time_range_ns,
    wavelength_axis_nm,
)
from .paths import DEFAULT_FIBER_NAMES_CONFIG, DEFAULT_RAW_DIR, DEFAULT_RESULTS_DIR, resolve_path
from .plot_style import CARPET_CMAP, COLORS, apply_axes_style, save_figure, set_publication_style, style_colorbar
from .visualize_carpets import CarpetConfig, parse_filename, scaled_core_image, spaced_units


DEFAULT_X_MAX_NM = 660.0
DEFAULT_X_MIN_NM = 350.0
DEFAULT_SLICE_MIN_NM = 380.0
DEFAULT_SLICE_STEP_NM = 20.0
EXTRA_SLICE_BOUNDARIES_NM = (360.0,)


def safe_name(path: Path) -> str:
    """Return a compact safe filename from a raw-data relative path."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.with_suffix("").as_posix()).strip("_")


def normalize(values: np.ndarray) -> np.ndarray:
    """Normalize a non-negative profile to unit maximum."""
    maximum = float(np.nanmax(values)) if values.size else 0.0
    if not np.isfinite(maximum) or maximum <= 0:
        return np.zeros_like(values, dtype=float)
    return values.astype(float) / maximum


def title_for(path: Path, fiber_names_config: Path) -> str:
    """Build a short display title from the filename."""
    parsed = parse_filename(path)
    group = f"{parsed['sample'] or 'unknown'}_{parsed['irradiation'] or 'unknown'}"
    fiber_names = read_fiber_name_map(fiber_names_config)
    bits = [
        fiber_names.real_name(group),
        spaced_units(str(parsed["position_token"])) or "",
        spaced_units(str(parsed["time_window"])) or "",
    ]
    return ", ".join(bit for bit in bits if bit)


def draw_slice_lines(ax: plt.Axes, *, x_min_nm: float, x_max_nm: float, image: bool) -> None:
    """Draw regular wavelength-slice boundary lines."""
    color = "#D62728"
    alpha = 0.78 if image else 0.72
    first = DEFAULT_SLICE_MIN_NM
    while first < x_min_nm:
        first += DEFAULT_SLICE_STEP_NM
    regular = list(np.arange(first, x_max_nm + 0.01, DEFAULT_SLICE_STEP_NM))
    extra = [value for value in EXTRA_SLICE_BOUNDARIES_NM if x_min_nm <= value <= x_max_nm]
    for wavelength in sorted(set(regular + extra)):
        ax.axvline(wavelength, color=color, linestyle=(0, (1.0, 2.0)), linewidth=0.75, alpha=alpha)


def save_plot(
    img_path: Path,
    raw_dir: Path,
    out_path: Path,
    *,
    fiber_names_config: Path,
    with_slices: bool,
    x_max_nm: float,
    top_edge_crop_rows: int,
) -> None:
    """Save one simplified carpet plot."""
    carpet = load_img(img_path)
    config = CarpetConfig(top_edge_crop_rows=top_edge_crop_rows)
    shown = scaled_core_image(carpet.data, config)
    row_profile = normalize(shown.mean(axis=1))
    col_profile = normalize(shown.mean(axis=0))

    wavelengths_nm = wavelength_axis_nm(carpet)
    times_ns = time_axis_ns(carpet, cropped=True, crop_rows=top_edge_crop_rows)
    extent = image_extent(carpet, cropped=True, crop_rows=top_edge_crop_rows)
    visible_time_ns = visible_time_range_ns(carpet, cropped=True, crop_rows=top_edge_crop_rows)

    x_min_nm = DEFAULT_X_MIN_NM
    if extent is not None:
        x_min_nm = max(DEFAULT_X_MIN_NM, float(extent[0]))
    if wavelengths_nm is None:
        x_min_nm = 0.0
        x_max = shown.shape[1]
    else:
        x_max = x_max_nm

    set_publication_style(base_font_size=8.2)
    fig = plt.figure(figsize=(6.9, 4.75), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[4.0, 1.12], height_ratios=[4.0, 1.12])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_row = fig.add_subplot(gs[0, 1])
    ax_col = fig.add_subplot(gs[1, 0])
    ax_blank = fig.add_subplot(gs[1, 1])
    ax_blank.axis("off")

    imshow_kwargs = {"extent": extent} if extent is not None else {}
    im = ax_img.imshow(shown, origin="lower", aspect="auto", cmap=CARPET_CMAP, **imshow_kwargs)
    ax_img.set_title(title_for(img_path.relative_to(raw_dir), fiber_names_config), pad=5)
    ax_img.set_xlabel("Wavelength (nm)" if wavelengths_nm is not None else "x pixel")
    ax_img.set_ylabel("Time (ns)" if times_ns is not None else "y pixel")
    ax_img.set_xlim(x_min_nm, x_max)
    if visible_time_ns is not None:
        ax_img.set_ylim(0, visible_time_ns)
    if with_slices and wavelengths_nm is not None:
        draw_slice_lines(ax_img, x_min_nm=x_min_nm, x_max_nm=x_max_nm, image=True)
    apply_axes_style(ax_img, grid=False)

    colorbar = fig.colorbar(im, ax=ax_img, fraction=0.035, pad=0.02, label="PL signal (a.u.)")
    style_colorbar(colorbar)

    row_axis = times_ns if times_ns is not None else np.arange(shown.shape[0])
    ax_row.plot(row_profile, row_axis, color=COLORS["teal"], linewidth=0.95)
    ax_row.set_title("Row mean", pad=4)
    ax_row.set_xlabel("Normalized signal")
    ax_row.set_ylabel("Time (ns)" if times_ns is not None else "y pixel")
    ax_row.set_xlim(-0.03, 1.03)
    if visible_time_ns is not None:
        ax_row.set_ylim(0, visible_time_ns)
    apply_axes_style(ax_row, grid=True)

    col_axis = wavelengths_nm if wavelengths_nm is not None else np.arange(shown.shape[1])
    ax_col.plot(col_axis, col_profile, color=COLORS["blue"], linewidth=0.95)
    ax_col.set_title("Column mean", pad=4)
    ax_col.set_xlabel("Wavelength (nm)" if wavelengths_nm is not None else "x pixel")
    ax_col.set_ylabel("Normalized signal")
    ax_col.set_xlim(x_min_nm, x_max)
    ax_col.set_ylim(-0.05, 1.08)
    if with_slices and wavelengths_nm is not None:
        draw_slice_lines(ax_col, x_min_nm=x_min_nm, x_max_nm=x_max_nm, image=False)
    apply_axes_style(ax_col, grid=True)

    save_figure(fig, out_path, dpi=260)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """Create simplified carpet plots for every raw .img file."""
    parser = argparse.ArgumentParser(description="Create simplified streak-carpet overview plots.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-subdir", default="simplified_carpets_660nm")
    parser.add_argument("--fiber-names-config", type=Path, default=DEFAULT_FIBER_NAMES_CONFIG)
    parser.add_argument("--x-max-nm", type=float, default=DEFAULT_X_MAX_NM)
    parser.add_argument("--top-edge-crop-rows", type=int, default=TOP_EDGE_CROP_ROWS)
    args = parser.parse_args(argv)

    raw_dir = resolve_path(args.raw_dir)
    out_dir = resolve_path(args.results_dir) / args.out_subdir
    fiber_names_config = resolve_path(args.fiber_names_config)
    plain_dir = out_dir / "without_slice_lines"
    slice_dir = out_dir / "with_slice_lines"
    records = []

    img_paths = sorted(raw_dir.rglob("*.img"))
    if not img_paths:
        raise SystemExit(f"No .img files found in {raw_dir}")

    for index, img_path in enumerate(img_paths, start=1):
        rel_path = img_path.relative_to(raw_dir)
        name = f"{safe_name(rel_path)}.png"
        plain_path = plain_dir / name
        slice_path = slice_dir / name
        save_plot(
            img_path,
            raw_dir,
            plain_path,
            fiber_names_config=fiber_names_config,
            with_slices=False,
            x_max_nm=args.x_max_nm,
            top_edge_crop_rows=args.top_edge_crop_rows,
        )
        save_plot(
            img_path,
            raw_dir,
            slice_path,
            fiber_names_config=fiber_names_config,
            with_slices=True,
            x_max_nm=args.x_max_nm,
            top_edge_crop_rows=args.top_edge_crop_rows,
        )
        records.append(
            {
                "source_file": rel_path.as_posix(),
                "without_slice_lines": os.path.relpath(plain_path, out_dir).replace("\\", "/"),
                "with_slice_lines": os.path.relpath(slice_path, out_dir).replace("\\", "/"),
            }
        )
        print(f"{index}/{len(img_paths)} {rel_path.as_posix()}")

    index_path = out_dir / "index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_file", "without_slice_lines", "with_slice_lines"])
        writer.writeheader()
        writer.writerows(records)

    print(f"plots: {len(records)} files x 2 versions")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
