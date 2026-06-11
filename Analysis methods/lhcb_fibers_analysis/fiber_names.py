from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .yaml_config import parse_yaml_scalar


FIBER_NAMES_SECTION = "fiber_names"


@dataclass(frozen=True)
class FiberNameMap:
    names: dict[str, str]

    def real_name(self, experimental_name: str) -> str:
        """Return the mapped real fiber name or the experimental fallback."""
        mapped = self.names.get(experimental_name, "").strip()
        return mapped or experimental_name


def read_fiber_name_map(path: Path) -> FiberNameMap:
    """Read experimental-name to real-fiber-name mappings from a small YAML file."""

    if not path.exists():
        return FiberNameMap({})

    names: dict[str, str] = {}
    in_section = False
    found_section = False
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if not line.startswith((" ", "\t")):
            key, separator, value = stripped.partition(":")
            if not separator:
                continue
            in_section = key.strip() == FIBER_NAMES_SECTION
            found_section = found_section or in_section
            if not found_section and value.strip():
                names[key.strip()] = str(parse_yaml_scalar(value))
            continue

        if not in_section or ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        names[key.strip()] = str(parse_yaml_scalar(value))

    return FiberNameMap(names)
