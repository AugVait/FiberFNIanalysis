from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors

from .paths import DEFAULT_CONFIG_DIR, DEFAULT_RAW_DIR, DEFAULT_RESULTS_DIR, resolve_path

SPECTRAL_XLIM_NM = (400, 720)
INTENSITY_MODES = ("normalized", "raw")
DEFAULT_NORMALIZED_SUBDIR = Path("pl_spectra")
DEFAULT_RAW_SUBDIR = Path("pl_spectra_raw")


@dataclass(frozen=True)
class PlotConfig:
    group: str
    title: str
    xlim_nm: tuple[float, float]
    colormap: str
    spectra: list[dict[str, object]]


@dataclass(frozen=True)
class PLSpectrum:
    path: Path
    group: str
    sample: str
    irradiation: str
    position_token: str
    distance_cm: int | None
    position_suffix: str
    cwl_nm: int | None
    excitation_nm: int | None
    pulse_energy_nj: float | None
    slit_um: int | None
    iris: str
    measurement_date: str
    wavelength_nm: np.ndarray
    intensity: np.ndarray
    normalized_intensity: np.ndarray
    out_png: Path


def _match(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _match_int(text: str, pattern: str) -> int | None:
    value = _match(text, pattern)
    return int(value) if value else None


def _match_float(text: str, pattern: str) -> float | None:
    value = _match(text, pattern)
    return float(value) if value else None


def parse_filename(path: Path) -> dict[str, object]:
    stem = path.stem.lower()
    if stem.endswith("_calc"):
        stem = stem[:-5]

    sample = _match(stem, r"(?:^|[^a-z0-9])((?:bcf\d+g?)|(?:scsf\d+))(?=[^a-z0-9]|$)")
    irradiation = _match(stem, r"_(noir|ir)_")
    position_token = _match(stem, r"_(endcm|\d+cm[a-z0-9]*)(?:_|$)")
    distance_cm = None
    position_suffix = ""
    if position_token and position_token != "endcm":
        distance_cm = int(re.match(r"(\d+)", position_token).group(1))
        position_suffix = _match(position_token, r"cm([a-z0-9]+)$")

    return {
        "sample": sample,
        "irradiation": irradiation,
        "group": f"{sample}_{irradiation}" if sample and irradiation else "unknown_unknown",
        "position_token": position_token,
        "distance_cm": distance_cm,
        "position_suffix": position_suffix,
        "cwl_nm": _match_int(stem, r"_(\d+)cwl_"),
        "excitation_nm": _match_int(stem, r"_ex(\d+)nm"),
        "pulse_energy_nj": _match_float(stem, r"_(\d+(?:\.\d+)?)nj"),
        "slit_um": _match_int(stem, r"_(\d+)um"),
        "iris": _match(stem, r"_(i\d+)(?:_|$)"),
    }


def measurement_date(path: Path, root: Path) -> str:
    for part in path.relative_to(root).parts:
        if re.fullmatch(r"20\d{2} \d{2} \d{2}", part):
            return part.replace(" ", "-", 1).replace(" ", "-", 1)
    return ""


def safe_output_stem(path: Path, root: Path) -> str:
    stem = path.stem
    if stem.lower().endswith("_calc"):
        stem = stem[:-5]
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    digest = hashlib.sha1(path.relative_to(root).as_posix().encode("utf-8")).hexdigest()[:8]
    return f"{safe_stem}_{digest}"


def yaml_quote(value: object) -> str:
    text = "" if value is None else str(value)
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def yaml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return yaml_quote(value)


def parse_yaml_scalar(value: str) -> object:
    text = value.strip()
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    if text in {"", "null", "None"}:
        return ""
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    if len(text) >= 2 and text[0] == "'" and text[-1] == "'":
        return text[1:-1].replace("''", "'")
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def read_plot_config(path: Path) -> PlotConfig:
    top: dict[str, object] = {}
    spectra: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    in_spectra = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if not line.startswith(" "):
            if current is not None:
                spectra.append(current)
                current = None
            key, _, value = stripped.partition(":")
            if key == "spectra":
                in_spectra = True
                continue
            top[key] = parse_yaml_scalar(value)
            in_spectra = False
            continue

        if not in_spectra:
            continue

        if stripped.startswith("- "):
            if current is not None:
                spectra.append(current)
            current = {}
            remainder = stripped[2:].strip()
            if remainder:
                key, _, value = remainder.partition(":")
                current[key.strip()] = parse_yaml_scalar(value)
            continue

        if current is not None and ":" in stripped:
            key, _, value = stripped.partition(":")
            current[key.strip()] = parse_yaml_scalar(value)

    if current is not None:
        spectra.append(current)

    group = str(top.get("group", path.stem))
    x_min = float(top.get("x_min_nm", SPECTRAL_XLIM_NM[0]))
    x_max = float(top.get("x_max_nm", SPECTRAL_XLIM_NM[1]))
    return PlotConfig(
        group=group,
        title=str(top.get("title", group)),
        xlim_nm=(x_min, x_max),
        colormap=str(top.get("colormap", "brg")),
        spectra=spectra,
    )


def read_plot_configs(config_dir: Path) -> list[PlotConfig]:
    return [read_plot_config(path) for path in sorted(config_dir.glob("*.yaml"))]


def find_column(columns: list[str], predicate) -> str | None:
    for column in columns:
        if predicate(column.strip().lower()):
            return column
    return None


def read_calc_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(path, sep="\t", engine="python")
    columns = list(data.columns)
    wavelength_col = find_column(columns, lambda text: text.startswith("wavelength"))
    intensity_col = find_column(columns, lambda text: text.startswith("intensity (arb"))
    if wavelength_col is None or intensity_col is None:
        raise ValueError(f"Could not find wavelength/intensity columns in {path}")

    wavelength = pd.to_numeric(data[wavelength_col], errors="coerce").to_numpy(dtype=float)
    intensity = pd.to_numeric(data[intensity_col], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(wavelength) & np.isfinite(intensity)
    wavelength = wavelength[keep]
    intensity = intensity[keep]
    if wavelength.size == 0:
        raise ValueError(f"No numeric spectrum rows in {path}")

    order = np.argsort(wavelength)
    return wavelength[order], intensity[order]


def normalized(intensity: np.ndarray) -> np.ndarray:
    max_intensity = float(np.nanmax(intensity))
    if not np.isfinite(max_intensity) or max_intensity == 0:
        max_intensity = float(np.nanmax(np.abs(intensity)))
    if not np.isfinite(max_intensity) or max_intensity == 0:
        return np.zeros_like(intensity, dtype=float)
    return intensity / max_intensity


def sort_key(record: PLSpectrum) -> tuple:
    position = -1 if record.position_token == "endcm" else record.distance_cm if record.distance_cm is not None else 9999
    suffix = record.position_suffix or "0"
    return (
        record.group,
        record.measurement_date,
        position,
        suffix,
        record.cwl_nm or 0,
        record.pulse_energy_nj or 0,
        str(record.path),
    )


def nominal_position(record: PLSpectrum) -> str:
    if record.position_token == "endcm":
        return "endcm"
    if record.distance_cm is not None:
        return f"{record.distance_cm}cm"
    return record.position_token or "unknown"


def suffix_rank(position_suffix: str) -> tuple[int, int | str]:
    if not position_suffix:
        return (0, 0)
    if position_suffix.isdigit():
        return (2, int(position_suffix))
    return (1, position_suffix)


def representative_score(record: PLSpectrum) -> tuple:
    return (
        suffix_rank(record.position_suffix),
        record.measurement_date,
        record.cwl_nm or 0,
        record.pulse_energy_nj or 0,
        str(record.path),
    )


def select_overlay_records(records: list[PLSpectrum], *, deduplicate_positions: bool = True) -> list[PLSpectrum]:
    if not deduplicate_positions:
        return records

    selected: dict[tuple[str, str], PLSpectrum] = {}
    for record in records:
        key = (record.group, nominal_position(record))
        current = selected.get(key)
        if current is None or representative_score(record) > representative_score(current):
            selected[key] = record
    return sorted(selected.values(), key=sort_key)


def label(record: PLSpectrum) -> str:
    parts = [record.group]
    if record.position_token:
        parts.append(record.position_token)
    if record.cwl_nm is not None:
        parts.append(f"{record.cwl_nm} cwl")
    if record.measurement_date:
        parts.append(record.measurement_date)
    return ", ".join(parts)


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "font.family": "serif",
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def output_base_path(out_dir: Path) -> Path:
    return out_dir


def intensity_values(record: PLSpectrum, intensity_mode: str) -> np.ndarray:
    if intensity_mode == "normalized":
        return record.normalized_intensity
    if intensity_mode == "raw":
        return record.intensity
    raise ValueError(f"Unsupported intensity mode: {intensity_mode}")


def y_axis_label(intensity_mode: str) -> str:
    if intensity_mode == "normalized":
        return "Normalized PL intensity (a.u.)"
    return "PL intensity (arb. u.)"


def intensity_mode_title(intensity_mode: str) -> str:
    if intensity_mode == "normalized":
        return "Normalized PL Spectra"
    return "Raw PL Spectra"


def overlay_name_suffix(intensity_mode: str) -> str:
    if intensity_mode == "normalized":
        return "normalized"
    return "raw_intensity"


def apply_y_axis_style(ax, intensity_mode: str) -> None:
    if intensity_mode == "normalized":
        ax.set_ylim(-0.03, 1.06)
    else:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))


def plot_single(record: PLSpectrum, intensity_mode: str, xlim_nm: tuple[float, float]) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 3.35), constrained_layout=True)
    ax.plot(record.wavelength_nm, intensity_values(record, intensity_mode), color="black", linewidth=1.05)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(y_axis_label(intensity_mode))
    ax.set_xlim(*xlim_nm)
    apply_y_axis_style(ax, intensity_mode)
    ax.minorticks_on()
    ax.text(0.97, 0.96, label(record), transform=ax.transAxes, va="top", ha="right", fontsize=7.2)
    record.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(record.out_png)
    plt.close(fig)


def plot_overlay(
    records: list[PLSpectrum],
    out_png: Path,
    intensity_mode: str,
    *,
    title: str,
    xlim_nm: tuple[float, float],
    colormap: str,
) -> None:
    if not records:
        return

    numeric_distances = [record.distance_cm for record in records if record.distance_cm is not None]
    vmin = min(numeric_distances) if numeric_distances else 0
    vmax = max(numeric_distances) if numeric_distances else 1
    if vmin == vmax:
        vmax = vmin + 1
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps[colormap]

    fig, ax = plt.subplots(figsize=(3.75, 3.75), constrained_layout=True)
    for record in records:
        if record.distance_cm is None:
            color = "#6b7280"
        else:
            color = cmap(norm(record.distance_cm))
        ax.plot(
            record.wavelength_nm,
            intensity_values(record, intensity_mode),
            color=color,
            linestyle="-",
            linewidth=0.72,
            alpha=0.72,
        )

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(y_axis_label(intensity_mode))
    ax.set_xlim(*xlim_nm)
    apply_y_axis_style(ax, intensity_mode)
    ax.minorticks_on()
    ax.text(0.97, 0.96, f"{title}\nn = {len(records)}", transform=ax.transAxes, va="top", ha="right")

    if numeric_distances:
        scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar = fig.colorbar(scalar_mappable, ax=ax, fraction=0.046, pad=0.02)
        colorbar.set_label("Position (cm)")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def is_reverse_position(position_suffix: str) -> bool:
    return "r" in position_suffix.lower()


def is_background_record(record: PLSpectrum) -> bool:
    stem = record.path.stem.lower()
    return "background" in stem or "dark" in stem


def config_note(record: PLSpectrum, default_overlay_paths: set[Path], default_single_paths: set[Path]) -> str:
    if is_background_record(record):
        return "background_or_dark_default_off"
    if is_reverse_position(record.position_suffix):
        return "reverse_R_RA_default_off"
    if record.path in default_overlay_paths:
        return "included_in_overlay"
    if record.path in default_single_paths:
        return "standard_duplicate_or_older_replicate"
    return "default_off"


def write_plot_configs(records: list[PLSpectrum], config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    standard_records = [
        record
        for record in records
        if not is_background_record(record) and not is_reverse_position(record.position_suffix)
    ]
    default_overlay_records = select_overlay_records(standard_records, deduplicate_positions=True)
    default_overlay_paths = {record.path for record in default_overlay_records}
    default_single_paths = {record.path for record in standard_records}

    for group in sorted({record.group for record in records}):
        group_records = [record for record in records if record.group == group]
        lines = [
            "# Edit include values, then rerun python -m lhcb_fibers_analysis.plot_pl_spectra.",
            "# include: true writes an individual PNG and adds the spectrum to the combined sample PNG.",
            "# single is kept for older configs, but include is the output toggle in the current folder layout.",
            f"group: {yaml_scalar(group)}",
            f"title: {yaml_scalar(group)}",
            f"x_min_nm: {SPECTRAL_XLIM_NM[0]}",
            f"x_max_nm: {SPECTRAL_XLIM_NM[1]}",
            "colormap: \"brg\"",
            "spectra:",
        ]
        for record in group_records:
            include = record.path in default_overlay_paths
            single = record.path in default_single_paths
            lines.extend(
                [
                    f"  - include: {yaml_scalar(include)}",
                    f"    single: {yaml_scalar(single)}",
                    f"    path: {yaml_scalar(record.path.as_posix())}",
                    f"    label: {yaml_scalar(label(record))}",
                    f"    position_token: {yaml_scalar(record.position_token)}",
                    f"    distance_cm: {yaml_scalar(record.distance_cm if record.distance_cm is not None else '')}",
                    f"    position_suffix: {yaml_scalar(record.position_suffix)}",
                    f"    measurement_date: {yaml_scalar(record.measurement_date)}",
                    f"    note: {yaml_scalar(config_note(record, default_overlay_paths, default_single_paths))}",
                ]
            )
        (config_dir / f"{group}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def configs_exist(config_dir: Path) -> bool:
    return config_dir.exists() and any(config_dir.glob("*.yaml"))


def selected_records_from_configs(
    configs: list[PlotConfig],
    records_by_path: dict[str, PLSpectrum],
) -> tuple[dict[str, list[PLSpectrum]], set[Path]]:
    overlay_by_group: dict[str, list[PLSpectrum]] = {}
    overlay_paths: set[Path] = set()

    for config in configs:
        for entry in config.spectra:
            rel_path = normalize_config_path(str(entry.get("path", "")))
            record = records_by_path.get(rel_path)
            if record is None:
                print(f"warning: config path not found, skipping: {rel_path}")
                continue
            if bool(entry.get("include", False)):
                overlay_by_group.setdefault(config.group, []).append(record)
                overlay_paths.add(record.path)

    for group in list(overlay_by_group):
        overlay_by_group[group] = sorted(overlay_by_group[group], key=sort_key)
    return overlay_by_group, overlay_paths


def normalize_config_path(path_text: str) -> str:
    normalized = Path(path_text.replace("\\", "/")).as_posix()
    obsolete_prefix = "raw data 2026 04 014 LHCb Fibers/"
    if normalized.startswith(obsolete_prefix):
        return normalized[len(obsolete_prefix) :]
    return normalized


def clean_outputs(out_dir: Path) -> None:
    skipped: list[Path] = []
    if out_dir.exists():
        for path in sorted(out_dir.rglob("*"), reverse=True):
            if path.is_file() and path.suffix.lower() in {".png", ".pdf"}:
                try:
                    path.unlink()
                except PermissionError:
                    skipped.append(path)
        for path in sorted(out_dir.rglob("*"), reverse=True):
            if path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
    for child in (out_dir / "pl_spectra_inventory.csv", out_dir / "index.html"):
        if child.exists():
            try:
                child.unlink()
            except PermissionError:
                skipped.append(child)
    if skipped:
        print(f"warning: could not remove {len(skipped)} existing output files; matching outputs will be overwritten.")


def load_records(
    raw_dir: Path,
    out_dir: Path,
    *,
    include_background: bool = False,
    include_reverse: bool = False,
) -> list[PLSpectrum]:
    records: list[PLSpectrum] = []
    for path in sorted(raw_dir.rglob("*_calc.txt")):
        if ".venv" in path.parts or "analysis" in path.parts:
            continue
        stem = path.stem.lower()
        if not include_background and ("background" in stem or "dark" in stem):
            continue

        parsed = parse_filename(path)
        if parsed["group"] == "unknown_unknown":
            continue
        if not include_reverse and is_reverse_position(str(parsed["position_suffix"])):
            continue

        wavelength, intensity = read_calc_spectrum(path)
        rel_path = path.relative_to(raw_dir)
        rel_out = output_base_path(out_dir) / str(parsed["group"]) / safe_output_stem(path, raw_dir)
        records.append(
            PLSpectrum(
                path=rel_path,
                group=str(parsed["group"]),
                sample=str(parsed["sample"]),
                irradiation=str(parsed["irradiation"]),
                position_token=str(parsed["position_token"]),
                distance_cm=parsed["distance_cm"],
                position_suffix=str(parsed["position_suffix"]),
                cwl_nm=parsed["cwl_nm"],
                excitation_nm=parsed["excitation_nm"],
                pulse_energy_nj=parsed["pulse_energy_nj"],
                slit_um=parsed["slit_um"],
                iris=str(parsed["iris"]),
                measurement_date=measurement_date(path, raw_dir),
                wavelength_nm=wavelength,
                intensity=intensity,
                normalized_intensity=normalized(intensity),
                out_png=rel_out.with_suffix(".png"),
            )
        )
    return sorted(records, key=sort_key)


def write_inventory(
    records: list[PLSpectrum],
    overlay_paths: dict[str, Path],
    selected_record_paths: set[Path],
    intensity_mode: str,
    out_csv: Path,
    out_dir: Path,
) -> None:
    fields = [
        "intensity_mode",
        "relative_path",
        "group",
        "sample",
        "irradiation",
        "position_token",
        "distance_cm",
        "position_suffix",
        "cwl_nm",
        "excitation_nm",
        "pulse_energy_nj",
        "slit_um",
        "iris",
        "measurement_date",
        "points",
        "wavelength_min_nm",
        "wavelength_max_nm",
        "intensity_max",
        "selected",
        "individual_png",
        "combined_png",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            overlay_png = overlay_paths.get(record.group, "")
            individual_png = Path(os.path.relpath(record.out_png, out_dir)).as_posix()
            combined_png = (
                Path(os.path.relpath(overlay_png, out_dir)).as_posix()
                if isinstance(overlay_png, Path)
                else ""
            )
            writer.writerow(
                {
                    "intensity_mode": intensity_mode,
                    "relative_path": record.path.as_posix(),
                    "group": record.group,
                    "sample": record.sample,
                    "irradiation": record.irradiation,
                    "position_token": record.position_token,
                    "distance_cm": record.distance_cm if record.distance_cm is not None else "",
                    "position_suffix": record.position_suffix,
                    "cwl_nm": record.cwl_nm if record.cwl_nm is not None else "",
                    "excitation_nm": record.excitation_nm if record.excitation_nm is not None else "",
                    "pulse_energy_nj": record.pulse_energy_nj if record.pulse_energy_nj is not None else "",
                    "slit_um": record.slit_um if record.slit_um is not None else "",
                    "iris": record.iris,
                    "measurement_date": record.measurement_date,
                    "points": record.wavelength_nm.size,
                    "wavelength_min_nm": float(record.wavelength_nm.min()),
                    "wavelength_max_nm": float(record.wavelength_nm.max()),
                    "intensity_max": float(np.nanmax(record.intensity)),
                    "selected": "yes" if record.path in selected_record_paths else "no",
                    "individual_png": individual_png if record.path in selected_record_paths else "",
                    "combined_png": combined_png,
                }
            )


def write_html(
    records: list[PLSpectrum],
    selected_records: list[PLSpectrum],
    overlay_paths: dict[str, Path],
    intensity_mode: str,
    out_html: Path,
) -> None:
    index_dir = out_html.parent.resolve()

    def link_to(path: Path) -> str:
        return Path(os.path.relpath(path.resolve(), index_dir)).as_posix()

    groups = sorted({record.group for record in records})
    overlay_rows = []
    for group in groups:
        group_records = [record for record in records if record.group == group]
        group_selected_records = [record for record in selected_records if record.group == group]
        overlay_png = overlay_paths.get(group, "")
        overlay_png_link = f"<a href=\"{link_to(overlay_png)}\">PNG</a>" if isinstance(overlay_png, Path) else ""
        overlay_rows.append(
            "<tr>"
            f"<td>{group}</td>"
            f"<td>{len(group_selected_records)} / {len(group_records)}</td>"
            f"<td>{overlay_png_link}</td>"
            f"<td><a href=\"{group}/\">sample folder</a></td>"
            "</tr>"
        )

    single_rows = []
    for record in selected_records:
        single_rows.append(
            "<tr>"
            f"<td>{record.group}</td>"
            f"<td>{record.position_token}</td>"
            f"<td>{record.cwl_nm or ''}</td>"
            f"<td>{record.measurement_date}</td>"
            f"<td>{record.path.as_posix()}</td>"
            f"<td><a href=\"{link_to(record.out_png)}\">PNG</a></td>"
            "</tr>"
        )

    title = intensity_mode_title(intensity_mode)
    mode_note = (
        "Each spectrum is normalized to its own maximum intensity from the converted <code>_calc.txt</code> export."
        if intensity_mode == "normalized"
        else "Spectra are plotted using raw intensity values from the converted <code>_calc.txt</code> export."
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #172033; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #d7dde8; padding: 6px 8px; text-align: left; }}
    th {{ background: #edf2f7; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>{mode_note} Plot membership is controlled by YAML files under <code>Analysis methods/configs/pl_spectra</code>. In each spectrum entry, <code>include</code> controls both the individual PNG and the combined sample PNG.</p>
  <h2>Sample Folders</h2>
  <table>
    <thead><tr><th>sample</th><th>selected / listed</th><th>combined PNG</th><th>folder</th></tr></thead>
    <tbody>{''.join(overlay_rows)}</tbody>
  </table>
  <h2>Selected Individual Figures</h2>
  <table>
    <thead><tr><th>sample</th><th>position</th><th>cwl nm</th><th>date</th><th>source</th><th>PNG</th></tr></thead>
    <tbody>{''.join(single_rows)}</tbody>
  </table>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot time-integrated PL spectra from converted *_calc.txt files.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-subdir", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--intensity-mode", choices=INTENSITY_MODES, default="normalized")
    parser.add_argument("--refresh-configs", action="store_true", help="Rewrite YAML plot configs from the discovered *_calc.txt files.")
    parser.add_argument("--no-clean", action="store_true", help="Do not remove existing generated plots before writing.")
    args = parser.parse_args(argv)

    raw_dir = resolve_path(args.raw_dir)
    results_dir = resolve_path(args.results_dir)
    out_subdir = args.out_subdir
    if out_subdir is None:
        out_subdir = DEFAULT_RAW_SUBDIR if args.intensity_mode == "raw" else DEFAULT_NORMALIZED_SUBDIR
    out_dir = (results_dir / out_subdir).resolve()
    out_base = output_base_path(out_dir)
    config_dir = resolve_path(args.config_dir)
    set_publication_style()

    if not args.no_clean:
        clean_outputs(out_dir)

    discovered_records = load_records(raw_dir, out_dir, include_background=True, include_reverse=True)
    if not discovered_records:
        raise SystemExit("No converted PL spectra found.")

    if args.refresh_configs or not configs_exist(config_dir):
        write_plot_configs(discovered_records, config_dir)

    configs = read_plot_configs(config_dir)
    if not configs:
        raise SystemExit(f"No YAML plot configs found in {config_dir}")

    records_by_path = {record.path.as_posix(): record for record in discovered_records}
    selected_by_group, selected_record_paths = selected_records_from_configs(
        configs, records_by_path
    )

    configured_paths: list[Path] = []
    for config in configs:
        for entry in config.spectra:
            record = records_by_path.get(normalize_config_path(str(entry.get("path", ""))))
            if record is not None:
                configured_paths.append(record.path)
    configured_path_set = set(configured_paths)
    records = [record for record in discovered_records if record.path in configured_path_set]
    selected_records = [record for group_records in selected_by_group.values() for record in group_records]

    for config in configs:
        for record in selected_by_group.get(config.group, []):
            plot_single(record, args.intensity_mode, config.xlim_nm)

    overlay_paths: dict[str, Path] = {}
    for config in configs:
        group_records = selected_by_group.get(config.group, [])
        if not group_records:
            continue
        out_png = out_base / config.group / f"{config.group}_{overlay_name_suffix(args.intensity_mode)}_selected_spectra.png"
        plot_overlay(
            group_records,
            out_png,
            args.intensity_mode,
            title=config.title,
            xlim_nm=config.xlim_nm,
            colormap=config.colormap,
        )
        overlay_paths[config.group] = out_png

    inventory_csv = out_base / "pl_spectra_inventory.csv"
    write_inventory(records, overlay_paths, selected_record_paths, args.intensity_mode, inventory_csv, out_dir)
    write_html(records, selected_records, overlay_paths, args.intensity_mode, out_base / "index.html")

    groups: dict[str, int] = {}
    for record in records:
        groups.setdefault(record.group, 0)
        groups[record.group] += 1
    selected_groups: dict[str, int] = {}
    for record in selected_records:
        selected_groups.setdefault(record.group, 0)
        selected_groups[record.group] += 1
    print(f"loaded PL spectra: {len(records)}")
    print(f"intensity mode: {args.intensity_mode}")
    print(f"config dir: {config_dir}")
    print("groups:")
    for group, count in sorted(groups.items()):
        print(f"  {group}: {selected_groups.get(group, 0)} selected / {count} listed")
    print(f"sample folders: {out_dir}")
    print(f"inventory: {inventory_csv}")
    print(f"html index: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
