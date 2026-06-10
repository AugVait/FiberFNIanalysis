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
- `it_10ns_traces`: 18

## 4. Rebuild All Results

```powershell
.\.venv\Scripts\python.exe -m lhcb_fibers_analysis.run_all --raw-dir "..\raw data" --results-dir "..\Analysis results"
```

`run_all` reads `configs/run_all.yaml`, which points to the YAML configs for each analysis family.

Expected generated result folders:

- `..\Analysis results\carpets`
- `..\Analysis results\pl_spectra`
- `..\Analysis results\pl_spectra_raw`
- `..\Analysis results\it_decay_fits_10ns`

## Individual Commands

```powershell
.\.venv\Scripts\python.exe -m lhcb_fibers_analysis.visualize_carpets --raw-dir "..\raw data" --results-dir "..\Analysis results"
.\.venv\Scripts\python.exe -m lhcb_fibers_analysis.plot_pl_spectra --raw-dir "..\raw data" --results-dir "..\Analysis results" --intensity-mode normalized
.\.venv\Scripts\python.exe -m lhcb_fibers_analysis.plot_pl_spectra --raw-dir "..\raw data" --results-dir "..\Analysis results" --intensity-mode raw
.\.venv\Scripts\python.exe -m lhcb_fibers_analysis.fit_it_decay --raw-dir "..\raw data" --results-dir "..\Analysis results" --time-window 10ns
```

Each individual command also accepts a config flag:

- `visualize_carpets --config configs\carpets.yaml`
- `plot_pl_spectra --config-dir configs\pl_spectra`
- `fit_it_decay --config configs\it_decay_fits_10ns.yaml`

Use `--refresh-configs` with `plot_pl_spectra` only when intentionally rebuilding the curated YAML selection files from discovered spectra.
