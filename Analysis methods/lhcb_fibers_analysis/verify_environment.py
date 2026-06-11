from __future__ import annotations

import argparse
import importlib
import sys


PACKAGES = [
    "numpy",
    "matplotlib",
    "scipy",
    "pandas",
    "PIL",
    "imageio",
    "tifffile",
    "h5py",
    "xarray",
    "skimage",
    "lmfit",
    "pybaselines",
    "tqdm",
    "ipykernel",
]


def check_environment() -> list[str]:
    """Check whether required Python packages can be imported."""
    missing: list[str] = []
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError:
            print(f"{name}: MISSING")
            missing.append(name)
            continue
        print(f"{name}: {getattr(module, '__version__', 'ok')}")
    return missing


def main(argv: list[str] | None = None) -> int:
    """Run the environment verification command."""
    parser = argparse.ArgumentParser(description="Check that the analysis Python packages can be imported.")
    parser.parse_args(argv)
    missing = check_environment()
    if missing:
        print("missing packages: " + ", ".join(missing))
        return 1
    print("analysis environment: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
