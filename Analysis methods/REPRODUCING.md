# Reproducing Results On Another Machine

## 1. Clone The Methods

Clone the project repository, then place the raw measurement folder beside `Analysis methods`:

```text
project root/
  Analysis methods/
  raw data/
```

The raw-data folder is not stored in Git.

## 2. Install Python Dependencies

```powershell
cd "Analysis methods"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Verify The Project

```powershell
.\.venv\Scripts\python.exe -m lhcb_fibers_analysis.verify_project --raw-dir "..\raw data"
```

The raw-data check must pass before reproducing results. For the current manifest, the key input-family counts are expected to include:

- `img_carpets`: 104
- `pl_calc_spectra`: 204
- `it_traces`: 36
- `it_10ns_traces`: 18

## 4. Rebuild All Results

The preferred route is Snakemake:

```powershell
.\.venv\Scripts\python.exe -m snakemake --dry-run --cores 1
.\.venv\Scripts\python.exe -m snakemake --cores 4
```

Useful named targets:

```powershell
.\.venv\Scripts\python.exe -m snakemake core_results --cores 4
.\.venv\Scripts\python.exe -m snakemake wavelength_cut_results --cores 4
.\.venv\Scripts\python.exe -m snakemake summary_grids --cores 1
.\.venv\Scripts\python.exe -m snakemake double_exp_results --cores 1
```

Snakemake reads `configs/snakemake.yaml`, checks `raw_data_manifest.json`, and writes generated outputs under `..\Analysis results`.

The manual wrapper script remains available:

```powershell
.\.venv\Scripts\python.exe .\manual_scripts\manual_run_all.py
```

It prints the selected paths and optional filters before running. Edit the preamble at the top of the script when you want to restrict carpet windows, change the IT window, or adjust PL wavelength limits.

The equivalent command-line route is:

```powershell
.\.venv\Scripts\python.exe -m lhcb_fibers_analysis.run_all --raw-dir "..\raw data" --results-dir "..\Analysis results"
```

`run_all` reads `configs/run_all.yaml`, which points to the YAML configs for each analysis family.
The same run config also points to `configs/fiber_names.yaml`, where experimental result names can be mapped to real fiber names for plot labels and output tables.

Expected generated result folders:

- `..\Analysis results\carpets`
- `..\Analysis results\pl_spectra`
- `..\Analysis results\pl_spectra_raw`
- `..\Analysis results\peak_position_shift`
- `..\Analysis results\it_decay_fits_all_it_10ns_window`

## Individual Manual Scripts

Use these when you only want one analysis family:

```powershell
.\.venv\Scripts\python.exe .\manual_scripts\manual_visualize_carpets.py
.\.venv\Scripts\python.exe .\manual_scripts\manual_pl_spectra.py
.\.venv\Scripts\python.exe .\manual_scripts\manual_peak_position_shift.py
.\.venv\Scripts\python.exe .\manual_scripts\manual_fit_it_decay.py
.\.venv\Scripts\python.exe .\manual_scripts\manual_carpet_wavelength_cuts.py
```

## Individual Commands

```powershell
.\.venv\Scripts\python.exe -m lhcb_fibers_analysis.visualize_carpets --raw-dir "..\raw data" --results-dir "..\Analysis results"
.\.venv\Scripts\python.exe -m lhcb_fibers_analysis.plot_pl_spectra --raw-dir "..\raw data" --results-dir "..\Analysis results" --intensity-mode normalized
.\.venv\Scripts\python.exe -m lhcb_fibers_analysis.plot_pl_spectra --raw-dir "..\raw data" --results-dir "..\Analysis results" --intensity-mode raw
.\.venv\Scripts\python.exe -m lhcb_fibers_analysis.peak_position_shift --raw-dir "..\raw data" --results-dir "..\Analysis results" --config configs\peak_position_shift.yaml
.\.venv\Scripts\python.exe -m lhcb_fibers_analysis.fit_it_decay --raw-dir "..\raw data" --results-dir "..\Analysis results"
```

Each individual command also accepts a config flag:

- `visualize_carpets --config configs\carpets.yaml`
- `visualize_carpets --fiber-names-config configs\fiber_names.yaml`
- `visualize_carpets --time-window 10ns`
- `plot_pl_spectra --config-dir configs\pl_spectra`
- `plot_pl_spectra --fiber-names-config configs\fiber_names.yaml`
- `plot_pl_spectra --x-min-nm 400 --x-max-nm 720`
- `peak_position_shift --config configs\peak_position_shift.yaml`
- `peak_position_shift --config-dir configs\pl_spectra`
- `peak_position_shift --fiber-names-config configs\fiber_names.yaml`
- `fit_it_decay --config configs\it_decay_fits_all_it_10ns_window.yaml`
- `fit_it_decay --config configs\it_decay_fits_10ns.yaml` for the legacy 10 ns filename-only analysis
- `fit_it_decay --fiber-names-config configs\fiber_names.yaml`

Use `--refresh-configs` only when intentionally rebuilding curated YAML selection files from discovered raw data. It is supported by `visualize_carpets`, `plot_pl_spectra`, and `fit_it_decay`.

## Manual Carpet Wavelength-Cut Fits

For exploratory work on a single streak-camera carpet:

```powershell
.\.venv\Scripts\python.exe -m lhcb_fibers_analysis.carpet_wavelength_cuts `
  "..\raw data\2026 04 17\bcf_6\bcf6_ir_100cm_ex360nm_10nJ_10ns.img" `
  --wavelength-min-nm 400 --wavelength-max-nm 540 `
  --step-nm 10 --band-width-nm 10 `
  --fit-start-offset-ns 0.05
```

This writes a manual output folder under `..\Analysis results\manual_carpet_wavelength_cuts\` unless `--out-dir` is supplied.
