# Manual Analysis Scripts

These scripts are for interactive/manual runs where parameters are edited in the file instead of passed as command-line flags.

Open a script, edit the `Manual analysis preamble` block near the top, then run the file with the project virtual environment:

```powershell
cd "C:\WorkWork\LHCb Fibers\Analysis methods"
.\.venv\Scripts\python.exe .\manual_scripts\manual_run_all.py
```

Available scripts:

- `manual_run_all.py`: run the full reproducible workflow.
- `manual_visualize_carpets.py`: generate Hamamatsu carpet quicklooks and contact sheets.
- `manual_pl_spectra.py`: generate normalized and/or raw PL spectra.
- `manual_fit_it_decay.py`: fit integrated-time decay traces.
- `manual_carpet_wavelength_cuts.py`: manually cut one carpet by wavelength and fit decay profiles.

Common editable controls:

- `TIME_WINDOWS` in `manual_visualize_carpets.py` filters carpet quicklooks by acquisition/firing window, for example `("10ns",)` or `("2ns", "10ns")`.
- `TIME_WINDOW` in `manual_fit_it_decay.py` selects the integrated-time trace window, for example `"10ns"`.
- `X_MIN_NM` and `X_MAX_NM` in `manual_pl_spectra.py` override the PL plot wavelength axis without editing every spectrum YAML file.
- `CARPET_TIME_WINDOWS`, `IT_TIME_WINDOW`, `PL_X_MIN_NM`, and `PL_X_MAX_NM` in `manual_run_all.py` pass the same controls into the full workflow.

Each script includes commented examples directly below its editable settings. Copy an example line, uncomment it, and adjust the value for the current run.

The reusable analysis code remains under `lhcb_fibers_analysis`. These files only provide editable preambles for manual use.
