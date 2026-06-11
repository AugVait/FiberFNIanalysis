from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .plot_style import CARPET_CMAP, DIAGNOSTIC_PANEL, save_figure, set_publication_style, style_colorbar

TOP_EDGE_CROP_ROWS = 12


@dataclass(frozen=True)
class StreakCarpet:
    path: Path
    data: np.ndarray
    metadata: dict[str, dict[str, str]]
    header_text: str

    @property
    def shape(self) -> tuple[int, int]:
        """Return the image array shape."""
        return self.data.shape

    def to_xarray(self) -> xr.DataArray:
        """Convert the streak carpet to an xarray DataArray."""
        attrs: dict[str, Any] = {
            "source": str(self.path),
            "dtype": str(self.data.dtype),
        }
        for section, values in self.metadata.items():
            for key, value in values.items():
                attrs[f"{section}.{key}"] = value

        coords: dict[str, tuple[str, np.ndarray]] = {}
        wavelength_nm = wavelength_axis_nm(self)
        if wavelength_nm is not None:
            coords["wavelength_nm"] = ("x", wavelength_nm)
        streak_time_ns = time_axis_ns(self, cropped=False)
        if streak_time_ns is not None:
            coords["time_ns"] = ("y", streak_time_ns)
        return xr.DataArray(self.data, dims=("y", "x"), coords=coords, name="counts", attrs=attrs)


def _decode_header(raw: bytes, header_bytes: int) -> str:
    """Decode a raw IMG header block into text."""
    return raw[:header_bytes].decode("utf-8", errors="replace").replace("\x00", " ")


def _split_header_fields(text: str) -> list[str]:
    """Split a Hamamatsu header line into named fields."""
    fields: list[str] = []
    start = 0
    in_quotes = False
    for index, char in enumerate(text):
        if char == '"':
            in_quotes = not in_quotes
        elif char == "," and not in_quotes:
            fields.append(text[start:index])
            start = index + 1
    fields.append(text[start:])
    return fields


def _clean_metadata_value(value: str) -> str:
    """Normalize a metadata value from the IMG header."""
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == '"' and cleaned[-1] == '"':
        return cleaned[1:-1]
    return cleaned.strip('"')


def _split_metadata_fields(text: str) -> dict[str, dict[str, str]]:
    """Parse key-value metadata fields from a header line."""
    metadata: dict[str, dict[str, str]] = {}

    sections = list(re.finditer(r"\[([A-Za-z0-9 ._/-]+)\],", text))
    for index, match in enumerate(sections):
        section = match.group(1).strip()
        end = sections[index + 1].start() if index + 1 < len(sections) else len(text)
        rest = text[match.end() : end].strip()
        metadata.setdefault(section, {})
        for field in _split_header_fields(rest):
            if "=" not in field:
                continue
            key, value = field.split("=", 1)
            metadata[section][key.strip()] = _clean_metadata_value(value)
    return metadata


_SCALAR_NUMBER_RE = re.compile(
    r"\s*(?P<number>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(?P<unit>[A-Za-z/]+)?\s*"
)


def _parse_scalar_number(text: str) -> tuple[float, str] | None:
    """Parse a scalar numeric value from metadata text."""
    cleaned = _clean_metadata_value(text)
    match = _SCALAR_NUMBER_RE.fullmatch(cleaned)
    if not match:
        return None
    return float(match.group("number")), (match.group("unit") or "")


def _metadata_float(metadata: dict[str, dict[str, str]], section: str, key: str) -> float | None:
    """Read a floating-point metadata value by section and key."""
    value = metadata.get(section, {}).get(key)
    if value is None:
        return None
    parsed = _parse_scalar_number(value)
    if parsed is None:
        return None
    # In these HPD-TA exports, commas are field/list separators inside quoted
    # values. Scalar numeric metadata uses decimal points, so values with comma
    # notation are intentionally not normalized here.
    return parsed[0]


def _metadata_int(metadata: dict[str, dict[str, str]], section: str, key: str) -> int | None:
    """Read an integer metadata value by section and key."""
    value = _metadata_float(metadata, section, key)
    if value is None:
        return None
    return int(value)


def _find_frame_bytes(raw: bytes) -> int:
    """Locate the byte offset where IMG frame data begins."""
    match = re.search(rb"Prop_BufferFramebytes=(\d+)", raw[:20000])
    if not match:
        raise ValueError("Could not find Prop_BufferFramebytes in the Hamamatsu header.")
    return int(match.group(1))


def crop_top_edge(data: np.ndarray, crop_rows: int = TOP_EDGE_CROP_ROWS) -> np.ndarray:
    """Remove bright top-edge rows from an image array."""
    if crop_rows > 0 and data.shape[0] > crop_rows:
        return data[:-crop_rows, :]
    return data


def spectrograph_center_nm(carpet: StreakCarpet) -> float | None:
    """Return the spectrograph center wavelength from metadata."""
    return _metadata_float(carpet.metadata, "Spectrograph", "Wavelength")


def x_scale_nm_per_pixel(carpet: StreakCarpet) -> float | None:
    """Return the wavelength calibration scale from metadata."""
    unit = carpet.metadata.get("Scaling", {}).get("ScalingXUnit", "").lower()
    if unit and unit != "nm":
        return None
    scale = _metadata_float(carpet.metadata, "Scaling", "ScalingXScale")
    if scale is None or not np.isfinite(scale) or scale == 0:
        return None
    return scale


def time_range_ns(carpet: StreakCarpet) -> float | None:
    """Return the streak-camera time range in nanoseconds."""
    value = carpet.metadata.get("Streak camera", {}).get("Time Range")
    if value is None:
        return None
    parsed = _parse_scalar_number(value)
    if parsed is None:
        return None
    number, unit = parsed
    scale_to_ns = {
        "fs": 1e-6,
        "ps": 1e-3,
        "ns": 1.0,
        "us": 1e3,
        "ms": 1e6,
        "s": 1e9,
    }.get(unit.lower())
    if scale_to_ns is None:
        return None
    return number * scale_to_ns


def visible_time_range_ns(
    carpet: StreakCarpet,
    *,
    cropped: bool = True,
    crop_rows: int = TOP_EDGE_CROP_ROWS,
) -> float | None:
    """Return the visible time span after optional top-edge cropping."""
    full_range = time_range_ns(carpet)
    if full_range is None:
        return None
    rows = crop_top_edge(carpet.data, crop_rows).shape[0] if cropped else carpet.data.shape[0]
    return full_range * rows / carpet.data.shape[0]


def wavelength_axis_nm(carpet: StreakCarpet) -> np.ndarray | None:
    """Build the calibrated wavelength axis for a carpet image."""
    center_nm = spectrograph_center_nm(carpet)
    scale_nm = x_scale_nm_per_pixel(carpet)
    if center_nm is None or scale_nm is None:
        return None
    cols = carpet.data.shape[1]
    return center_nm + (np.arange(cols, dtype=float) - (cols - 1) / 2) * scale_nm


def time_axis_ns(
    carpet: StreakCarpet,
    *,
    cropped: bool = True,
    crop_rows: int = TOP_EDGE_CROP_ROWS,
) -> np.ndarray | None:
    """Build the calibrated time axis for a carpet image."""
    full_range = time_range_ns(carpet)
    if full_range is None:
        return None
    rows = crop_top_edge(carpet.data, crop_rows).shape[0] if cropped else carpet.data.shape[0]
    ns_per_pixel = full_range / carpet.data.shape[0]
    return (np.arange(rows, dtype=float) + 0.5) * ns_per_pixel


def image_extent(
    carpet: StreakCarpet,
    *,
    cropped: bool = True,
    crop_rows: int = TOP_EDGE_CROP_ROWS,
) -> tuple[float, float, float, float] | None:
    """Return an imshow extent from calibrated image axes."""
    wavelengths = wavelength_axis_nm(carpet)
    visible_time = visible_time_range_ns(carpet, cropped=cropped, crop_rows=crop_rows)
    if wavelengths is None or visible_time is None:
        return None
    step = float(np.median(np.diff(wavelengths))) if wavelengths.size > 1 else 1.0
    return (
        float(wavelengths[0] - step / 2),
        float(wavelengths[-1] + step / 2),
        0.0,
        float(visible_time),
    )


def load_img(path: str | Path) -> StreakCarpet:
    """Load a Hamamatsu HPD-TA `.img` streak-camera carpet.

    The files in this project store a readable metadata header followed by one
    contiguous binary frame. The frame length is encoded as Prop_BufferFramebytes.
    """

    img_path = Path(path)
    raw = img_path.read_bytes()
    frame_bytes = _find_frame_bytes(raw)
    header_bytes = len(raw) - frame_bytes
    if header_bytes <= 0:
        raise ValueError(f"Invalid frame length for {img_path}: {frame_bytes} bytes.")

    header_text = _decode_header(raw, header_bytes)
    metadata = _split_metadata_fields(header_text)

    bytes_per_pixel = _metadata_int(metadata, "Acquisition", "BytesPerPixel") or 2
    row_bytes = _metadata_int(metadata, "Camera", "Prop_BufferRowbytes")
    if row_bytes is None:
        width = _metadata_int(metadata, "Camera", "HWidth")
        if width is None:
            raise ValueError("Could not determine image width from header.")
        row_bytes = width * bytes_per_pixel

    width = row_bytes // bytes_per_pixel
    height = frame_bytes // row_bytes
    if width * height * bytes_per_pixel != frame_bytes:
        raise ValueError("Frame bytes do not divide cleanly into rows and pixels.")

    if bytes_per_pixel == 2:
        dtype = np.dtype("<u2")
    elif bytes_per_pixel == 4:
        dtype = np.dtype("<u4")
    else:
        raise ValueError(f"Unsupported BytesPerPixel={bytes_per_pixel}.")

    data = np.frombuffer(raw[header_bytes:], dtype=dtype).reshape(height, width)
    return StreakCarpet(path=img_path, data=data, metadata=metadata, header_text=header_text)


def save_quicklook(carpet: StreakCarpet, out_path: str | Path, *, percentile: tuple[float, float] = (1, 99.8)) -> None:
    """Save a quicklook image for a Hamamatsu streak carpet."""
    set_publication_style(base_font_size=8.0)
    data = crop_top_edge(carpet.data)
    low, high = np.percentile(data, percentile)
    fig, ax = plt.subplots(figsize=DIAGNOSTIC_PANEL, constrained_layout=True)
    extent = image_extent(carpet, cropped=True)
    imshow_kwargs: dict[str, Any] = {}
    if extent is not None:
        imshow_kwargs["extent"] = extent
    im = ax.imshow(data, origin="lower", aspect="auto", cmap=CARPET_CMAP, vmin=low, vmax=high, **imshow_kwargs)
    ax.set_title(carpet.path.stem.replace("_", " "), pad=5)
    ax.set_xlabel("Wavelength (nm)" if wavelength_axis_nm(carpet) is not None else "x pixel")
    ax.set_ylabel("Time (ns)" if time_axis_ns(carpet, cropped=True) is not None else "y pixel")
    colorbar = fig.colorbar(im, ax=ax, label="Counts")
    style_colorbar(colorbar)
    save_figure(fig, out_path, dpi=260)
    plt.close(fig)


def main() -> None:
    """Run the Hamamatsu IMG quicklook command-line interface."""
    parser = argparse.ArgumentParser(description="Load a Hamamatsu streak-camera .img carpet and optionally save a quicklook PNG.")
    parser.add_argument("img", type=Path)
    parser.add_argument("--out", type=Path, help="Optional quicklook PNG path.")
    args = parser.parse_args()

    carpet = load_img(args.img)
    print(f"path: {carpet.path}")
    print(f"shape: {carpet.shape[0]} rows x {carpet.shape[1]} columns")
    print(f"dtype: {carpet.data.dtype}")
    print(f"min/max: {carpet.data.min()} / {carpet.data.max()}")
    print(f"date: {carpet.metadata.get('Application', {}).get('Date', '')}")
    print(f"time: {carpet.metadata.get('Application', {}).get('Time', '')}")
    if args.out:
        save_quicklook(carpet, args.out)
        print(f"quicklook: {args.out}")


if __name__ == "__main__":
    main()
