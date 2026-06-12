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

For routine use, prefer the manual script. It has an editable preamble at the top and prints the selected settings before it starts:

```powershell
.\.venv\Scripts\python.exe .\manual_scripts\manual_run_all.py
```

The command-line entry point is still available for automation:

```powershell
.\.venv\Scripts\python.exe -m lhcb_fibers_analysis.run_all --raw-dir "..\raw data" --results-dir "..\Analysis results"
```

Optional run controls can be passed on the command line:

```powershell
.\.venv\Scripts\python.exe -m lhcb_fibers_analysis.run_all `
  --raw-dir "..\raw data" --results-dir "..\Analysis results" `
  --carpet-time-window 10ns `
  --it-time-window 10ns `
  --pl-x-min-nm 400 --pl-x-max-nm 720
```

The same controls are easier to edit in `manual_scripts/manual_run_all.py`:

```python
CARPET_TIME_WINDOWS = ("10ns",)
IT_TIME_WINDOW = "10ns"
PL_X_MIN_NM = 400.0
PL_X_MAX_NM = 720.0
```

The run is controlled by YAML files under `configs`:

- `configs/run_all.yaml`: selects which analysis families run.
- `configs/fiber_names.yaml`: maps experimental result names to real fiber names used in generated plot labels and tables.
- `configs/carpets.yaml`: Hamamatsu carpet visualization settings and pointer to detailed scan configs.
- `configs/carpets/*.yaml`: sample-wise streak-camera scan membership and metadata.
- `configs/it_decay_fits_10ns.yaml`: IT decay fitting settings and pointer to detailed trace configs.
- `configs/it_decay_fits_10ns/*.yaml`: sample-wise integrated-time trace membership and metadata.
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
- Real fiber names are display metadata only; raw-data paths and YAML selection keys continue to use experimental names.
- Preserve the distinction between raw `.TXT` spectra, converted `_calc.txt` spectra, `.img` carpets, and derived `.dat` or `DC_` files.

## Manual Carpet Wavelength Cuts

For one-off streak-carpet inspection, edit and run `manual_scripts/manual_carpet_wavelength_cuts.py`. It loads one `.img`, averages wavelength bands at regular centers, detects secondary peaks, fits a single exponential decay to the dominant peak while excluding later secondary peaks, and writes CSV/PNG diagnostics.

The underlying command-line helper is:

```powershell
.\.venv\Scripts\python.exe -m lhcb_fibers_analysis.carpet_wavelength_cuts `
  "..\raw data\2026 04 17\bcf_6\bcf6_ir_100cm_ex360nm_10nJ_10ns.img" `
  --wavelength-min-nm 400 --wavelength-max-nm 540 `
  --step-nm 10 --band-width-nm 10 `
  --fit-start-offset-ns 0.05
```

Main outputs are `fit_summary.csv`, `cut_profiles.csv`, `tau_vs_wavelength.png`, `profiles_overlay.png`, `carpet_with_cuts.png`, and optional per-cut fit PNGs under `fits/`.
