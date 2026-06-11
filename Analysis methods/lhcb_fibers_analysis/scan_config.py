from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SectionConfig:
    path: Path
    top: dict[str, object]
    entries: list[dict[str, object]]

    @property
    def group(self) -> str:
        """Return the config group name."""
        return str(self.top.get("group", self.path.stem))

    @property
    def title(self) -> str:
        """Return the config display title."""
        return str(self.top.get("title", self.group))


def yaml_quote(value: object) -> str:
    """Quote a value for the small YAML files used by the analyses."""
    text = "" if value is None else str(value)
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def yaml_scalar(value: object) -> str:
    """Format a scalar value for the small YAML config files."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return yaml_quote(value)


def parse_yaml_scalar(value: str) -> object:
    """Parse a scalar value from the small YAML config format."""
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


def read_section_config(path: Path, list_key: str) -> SectionConfig:
    """Read one section-style YAML config file."""
    top: dict[str, object] = {}
    entries: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    in_entries = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if not line.startswith(" "):
            if current is not None:
                entries.append(current)
                current = None
            key, _, value = stripped.partition(":")
            if key == list_key:
                in_entries = True
                continue
            top[key] = parse_yaml_scalar(value)
            in_entries = False
            continue

        if not in_entries:
            continue

        if stripped.startswith("- "):
            if current is not None:
                entries.append(current)
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
        entries.append(current)

    return SectionConfig(path=path, top=top, entries=entries)


def read_section_configs(config_dir: Path, list_key: str) -> list[SectionConfig]:
    """Read all section-style YAML configs in a directory."""
    return [read_section_config(path, list_key) for path in sorted(config_dir.glob("*.yaml"))]


def configs_exist(config_dir: Path) -> bool:
    """Return whether a config directory contains YAML files."""
    return config_dir.exists() and any(config_dir.glob("*.yaml"))


def normalize_config_path(path_text: str) -> str:
    """Normalize a path string from a YAML config entry."""
    return Path(path_text.replace("\\", "/")).as_posix()


def relative_config_dir(method_config_path: Path, configured: str | Path) -> Path:
    """Resolve a configured config directory relative to a method config."""
    path = Path(configured)
    if path.is_absolute():
        return path
    return method_config_path.parent / path
