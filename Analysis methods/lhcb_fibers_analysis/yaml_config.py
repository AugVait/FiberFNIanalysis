from __future__ import annotations

from pathlib import Path


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


def read_yaml_mapping(path: Path) -> dict[str, object]:
    """Read the small top-level scalar YAML files used by these analyses."""

    values: dict[str, object] = {}
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if raw_line.startswith((" ", "\t")):
            continue
        key, separator, value = stripped.partition(":")
        if not separator:
            continue
        values[key.strip()] = parse_yaml_scalar(value)
    return values


def string_value(values: dict[str, object], key: str, default: str) -> str:
    value = values.get(key, default)
    return str(value)


def int_value(values: dict[str, object], key: str, default: int) -> int:
    value = values.get(key, default)
    return int(value)


def float_value(values: dict[str, object], key: str, default: float) -> float:
    value = values.get(key, default)
    return float(value)


def bool_value(values: dict[str, object], key: str, default: bool) -> bool:
    value = values.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
