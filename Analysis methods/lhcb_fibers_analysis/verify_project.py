from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .paths import DEFAULT_MANIFEST, DEFAULT_RAW_DIR, resolve_path
from .raw_data_manifest import check_manifest, print_limited, print_summary
from .verify_environment import check_environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check Python dependencies and verify local raw data against the tracked manifest."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)

    print("Checking Python analysis environment...")
    missing = check_environment()
    environment_ok = not missing

    print()
    print("Checking raw data manifest...")
    result = check_manifest(resolve_path(args.raw_dir), resolve_path(args.manifest))
    print_summary(result.summary)
    print_limited("missing files", result.missing)
    print_limited("extra files", result.extra)
    print_limited("size mismatches", result.size_mismatches)
    print_limited("hash mismatches", result.hash_mismatches)

    if environment_ok and result.ok:
        print("project verification: OK")
        return 0

    if not environment_ok:
        print("project verification: FAILED because Python packages are missing")
    if not result.ok:
        print("project verification: FAILED because raw data differs from the manifest")
    return 1


if __name__ == "__main__":
    sys.exit(main())
