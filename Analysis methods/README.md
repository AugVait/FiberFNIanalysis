# LHCb Fibers Analysis Methods

This folder is the Git-synced part of the project. It contains Python analysis code, plot-selection configs, setup instructions, and a raw-data manifest used to verify that another PC has the same local data.

The sibling folders `../raw data`, `../Analysis results`, and `../Analysis Old` are intentionally ignored by Git.

## Python Setup

Use Python 3.12 and a local virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Check the environment and raw data:

```powershell
.\.venv\Scripts\python.exe -m lhcb_fibers_analysis.verify_project --raw-dir "..\raw data"
```

## Raw-Data Manifest

The manifest records every file under `../raw data` using relative path, byte size, and SHA-256 hash. It does not use timestamps.

Check a copied raw-data folder:

```powershell
.\.venv\Scripts\python.exe -m lhcb_fibers_analysis.raw_data_manifest check --raw-dir "..\raw data"
```

Regenerate the manifest only when the canonical raw data intentionally changes:

```powershell
.\.venv\Scripts\python.exe -m lhcb_fibers_analysis.raw_data_manifest create --raw-dir "..\raw data" --out raw_data_manifest.json
```

Commit the updated `raw_data_manifest.json` with the analysis-method changes that require the new data.

## Reproduce Results

Run all current analyses:

```powershell
.\.venv\Scripts\python.exe -m lhcb_fibers_analysis.run_all --raw-dir "..\raw data" --results-dir "..\Analysis results"
```

The run is controlled by YAML files under `configs`:

- `configs/run_all.yaml`: selects which analysis families run.
- `configs/carpets.yaml`: Hamamatsu carpet visualization settings.
- `configs/it_decay_fits_10ns.yaml`: IT decay fitting settings.
- `configs/pl_spectra/*.yaml`: PL spectrum plotting selections.

Outputs are written under `../Analysis results`:

- `carpets`: Hamamatsu `.img` quicklooks, contact sheets, inventory, and HTML index.
- `pl_spectra`: normalized PL plots and inventory.
- `pl_spectra_raw`: raw-intensity PL plots and inventory.
- `it_decay_fits_10ns`: `IT_*_10ns.dat` decay fits, matrices, plot PDFs/PNGs, and method note.

## Analysis Notes

- Hamamatsu carpet visualizations crop the bright top 12 rows before visual review or profile analysis.
- Numeric `cm` tokens are fiber positions in centimeters relative to the non-metalized end.
- `ENDcm` is an endpoint condition, not a numeric distance.
- `cm1` and `cm2` suffixes are replicate labels at the same distance.
- Preserve the distinction between raw `.TXT` spectra, converted `_calc.txt` spectra, `.img` carpets, and derived `.dat` or `DC_` files.
