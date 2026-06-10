from __future__ import annotations

from pathlib import Path


METHODS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = METHODS_ROOT.parent

DEFAULT_RAW_DIR = PROJECT_ROOT / "raw data"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "Analysis results"
DEFAULT_CONFIGS_DIR = METHODS_ROOT / "configs"
DEFAULT_PL_CONFIG_DIR = DEFAULT_CONFIGS_DIR / "pl_spectra"
DEFAULT_CONFIG_DIR = DEFAULT_PL_CONFIG_DIR
DEFAULT_CARPET_CONFIG = DEFAULT_CONFIGS_DIR / "carpets.yaml"
DEFAULT_IT_DECAY_CONFIG = DEFAULT_CONFIGS_DIR / "it_decay_fits_10ns.yaml"
DEFAULT_RUN_ALL_CONFIG = DEFAULT_CONFIGS_DIR / "run_all.yaml"
DEFAULT_MANIFEST = METHODS_ROOT / "raw_data_manifest.json"


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()
