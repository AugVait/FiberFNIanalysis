from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path


METHODS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = METHODS_DIR.parent
RAW_DIR = PROJECT_ROOT / "raw data"
RESULTS_DIR = PROJECT_ROOT / "Analysis results"
CONFIGS_DIR = METHODS_DIR / "configs"
FIBER_NAMES_CONFIG = CONFIGS_DIR / "fiber_names.yaml"
MANUAL_SELECTIONS_DIR = CONFIGS_DIR / "manual selections"


def display_value(value: object) -> str:
    """Format an editable setting for a short run summary."""
    if value is None or value == "":
        return "default"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) or "default"
    return str(value)


def show_settings(title: str, settings: Iterable[tuple[str, object]]) -> None:
    """Print the important manual settings before a script starts work."""
    print()
    print(title)
    print("-" * len(title))
    for name, value in settings:
        print(f"{name}: {display_value(value)}")
    print()


def common_analysis_args() -> list[str]:
    """Return raw/results arguments shared by the manual analysis scripts."""
    return ["--raw-dir", str(RAW_DIR), "--results-dir", str(RESULTS_DIR)]


def use_methods_package() -> None:
    """Make the local analysis package importable when a manual script is run directly."""
    methods = str(METHODS_DIR)
    if methods not in sys.path:
        sys.path.insert(0, methods)


def add_optional(args: list[str], name: str, value: object) -> None:
    """Append a flag/value pair when the manual setting is not None."""
    if value is not None:
        args.extend([name, str(value)])


def add_flag(args: list[str], name: str, enabled: bool) -> None:
    """Append a boolean flag when enabled."""
    if enabled:
        args.append(name)


def add_repeated(args: list[str], name: str, values: str | Iterable[object] | None) -> None:
    """Append repeated flag/value pairs from a scalar or iterable manual setting."""
    if values is None:
        return
    if isinstance(values, str):
        items = [values]
    else:
        items = list(values)
    for value in items:
        args.extend([name, str(value)])
