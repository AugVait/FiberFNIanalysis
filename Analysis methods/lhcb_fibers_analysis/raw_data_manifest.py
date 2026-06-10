from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .paths import DEFAULT_MANIFEST, DEFAULT_RAW_DIR, resolve_path


CHUNK_SIZE = 1024 * 1024
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class ManifestFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class CheckResult:
    missing: list[str]
    extra: list[str]
    size_mismatches: list[str]
    hash_mismatches: list[str]
    summary: dict[str, object]

    @property
    def ok(self) -> bool:
        return not (self.missing or self.extra or self.size_mismatches or self.hash_mismatches)


def relative_path(path: Path, raw_dir: Path) -> str:
    return path.relative_to(raw_dir).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_raw_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"Raw data path is not a directory: {raw_dir}")
    return sorted(path for path in raw_dir.rglob("*") if path.is_file())


def classify(path_text: str) -> list[str]:
    name = Path(path_text).name.lower()
    labels: list[str] = []
    if name.endswith(".img"):
        labels.append("img_carpets")
    if name.endswith("_calc.txt"):
        labels.append("pl_calc_spectra")
    if name.startswith("it_") and name.endswith("_10ns.dat"):
        labels.append("it_10ns_traces")
    if name.startswith("t_") and name.endswith(".dat"):
        labels.append("t_dat_traces")
    if name.startswith("dc_"):
        labels.append("dc_matrices")
    return labels


def extension_label(path_text: str) -> str:
    suffix = Path(path_text).suffix.lower()
    return suffix or "<none>"


def summarize(files: list[ManifestFile]) -> dict[str, object]:
    by_extension: Counter[str] = Counter()
    input_families: Counter[str] = Counter()
    total_bytes = 0
    for entry in files:
        by_extension[extension_label(entry.path)] += 1
        total_bytes += entry.size
        for label in classify(entry.path):
            input_families[label] += 1
    return {
        "total_files": len(files),
        "total_bytes": total_bytes,
        "by_extension": dict(sorted(by_extension.items())),
        "input_families": dict(sorted(input_families.items())),
    }


def build_manifest(raw_dir: Path) -> dict[str, object]:
    files = [
        ManifestFile(
            path=relative_path(path, raw_dir),
            size=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for path in iter_raw_files(raw_dir)
    ]
    return {
        "schema_version": MANIFEST_VERSION,
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "raw_dir_name": raw_dir.name,
        "summary": summarize(files),
        "files": [entry.__dict__ for entry in files],
    }


def load_manifest(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported manifest schema_version={manifest.get('schema_version')!r}; "
            f"expected {MANIFEST_VERSION}."
        )
    return manifest


def manifest_files(manifest: dict[str, object]) -> dict[str, ManifestFile]:
    entries = {}
    for raw_entry in manifest.get("files", []):
        entry = ManifestFile(
            path=str(raw_entry["path"]),
            size=int(raw_entry["size"]),
            sha256=str(raw_entry["sha256"]),
        )
        entries[entry.path] = entry
    return entries


def check_manifest(raw_dir: Path, manifest_path: Path) -> CheckResult:
    manifest = load_manifest(manifest_path)
    expected = manifest_files(manifest)
    current_paths = {relative_path(path, raw_dir): path for path in iter_raw_files(raw_dir)}

    missing = sorted(set(expected) - set(current_paths))
    extra = sorted(set(current_paths) - set(expected))
    size_mismatches: list[str] = []
    hash_mismatches: list[str] = []

    for path_text in sorted(set(expected) & set(current_paths)):
        expected_entry = expected[path_text]
        current_path = current_paths[path_text]
        current_size = current_path.stat().st_size
        if current_size != expected_entry.size:
            size_mismatches.append(
                f"{path_text}: expected {expected_entry.size} bytes, found {current_size} bytes"
            )
            continue
        current_hash = sha256_file(current_path)
        if current_hash != expected_entry.sha256:
            hash_mismatches.append(
                f"{path_text}: expected {expected_entry.sha256}, found {current_hash}"
            )

    current_files = [
        ManifestFile(
            path=path_text,
            size=current_paths[path_text].stat().st_size,
            sha256="",
        )
        for path_text in sorted(current_paths)
    ]
    return CheckResult(
        missing=missing,
        extra=extra,
        size_mismatches=size_mismatches,
        hash_mismatches=hash_mismatches,
        summary=summarize(current_files),
    )


def print_summary(summary: dict[str, object]) -> None:
    print(f"total files: {summary.get('total_files', 0)}")
    print(f"total bytes: {summary.get('total_bytes', 0)}")
    print("by extension:")
    for extension, count in dict(summary.get("by_extension", {})).items():
        print(f"  {extension}: {count}")
    print("input families:")
    for family, count in dict(summary.get("input_families", {})).items():
        print(f"  {family}: {count}")


def print_limited(title: str, values: list[str], limit: int = 20) -> None:
    if not values:
        return
    print(f"{title}: {len(values)}")
    for value in values[:limit]:
        print(f"  {value}")
    if len(values) > limit:
        print(f"  ... {len(values) - limit} more")


def create_command(args: argparse.Namespace) -> int:
    raw_dir = resolve_path(args.raw_dir)
    out_path = resolve_path(args.out)
    manifest = build_manifest(raw_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote manifest: {out_path}")
    print_summary(manifest["summary"])
    return 0


def check_command(args: argparse.Namespace) -> int:
    raw_dir = resolve_path(args.raw_dir)
    manifest_path = resolve_path(args.manifest)
    result = check_manifest(raw_dir, manifest_path)
    print(f"raw data: {raw_dir}")
    print(f"manifest: {manifest_path}")
    print_summary(result.summary)
    print_limited("missing files", result.missing)
    print_limited("extra files", result.extra)
    print_limited("size mismatches", result.size_mismatches)
    print_limited("hash mismatches", result.hash_mismatches)
    if result.ok:
        print("raw data check: OK")
        return 0
    print("raw data check: FAILED")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or verify the raw-data SHA-256 manifest.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create a raw-data manifest.")
    create_parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    create_parser.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    create_parser.set_defaults(func=create_command)

    check_parser = subparsers.add_parser("check", help="Verify raw data against a manifest.")
    check_parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    check_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    check_parser.set_defaults(func=check_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
