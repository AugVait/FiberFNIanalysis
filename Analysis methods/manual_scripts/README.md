# Manual Analysis Scripts

These are the easiest entry points for day-to-day analysis. Open one file, edit the small preamble near the top, and run it. No command-line flags are needed.

Start with the full workflow:

```powershell
cd "C:\WorkWork\LHCb Fibers\Analysis methods"
.\.venv\Scripts\python.exe .\manual_scripts\manual_run_all.py
```

Each script prints the important settings before it starts. If something looks wrong, stop the run, edit the preamble, and run it again.

## Pick A Script

- `manual_run_all.py`: run the full reproducible workflow.
- `manual_visualize_carpets.py`: generate Hamamatsu carpet quicklooks and contact sheets.
- `manual_pl_spectra.py`: generate normalized and/or raw PL spectra.
- `manual_peak_position_shift.py`: calculate PL peak-position shifts from `configs/peak_position_shift.yaml` and plot all selected samples together.
- `manual_fit_it_decay.py`: fit integrated-time decay traces.
- `manual_carpet_wavelength_cuts.py`: manually cut one carpet by wavelength and fit decay profiles.
- `manual_batch_carpet_wavelength_cuts.py`: cut every carpet into clean 20 nm wavelength bands and write one text-output folder per scan.

## Common Edits

- `TIME_WINDOWS` in `manual_visualize_carpets.py` filters carpet quicklooks by acquisition/firing window, for example `("10ns",)` or `("2ns", "10ns")`.
- `TIME_WINDOW` in `manual_fit_it_decay.py` filters integrated-time input files by filename window, for example `"10ns"`; leave it as `None` to use the default all-IT discovery with 10 ns fit windows.
- `X_MIN_NM` and `X_MAX_NM` in `manual_pl_spectra.py` override the PL plot wavelength axis without editing every spectrum YAML file.
- `configs/peak_position_shift.yaml` controls peak smoothing, local peak refinement, and explicitly omitted points for `manual_peak_position_shift.py`.
- `INTERVAL_NM`, `RANGE_MODE`, `WAVELENGTH_MIN_NM`, and `WAVELENGTH_MAX_NM` in `manual_batch_carpet_wavelength_cuts.py` control the clean wavelength bands used for all carpet text exports.
- `CARPET_TIME_WINDOWS`, `IT_TIME_WINDOW`, `PL_X_MIN_NM`, and `PL_X_MAX_NM` in `manual_run_all.py` pass the same controls into the full workflow.

Each script includes commented examples directly below its editable settings. Copy an example line, uncomment it, and adjust the value for the current run.

Use `None` when you want the normal project default.

The reusable analysis code remains under `lhcb_fibers_analysis`. These files only provide editable preambles for manual use.
