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

The reusable analysis code remains under `lhcb_fibers_analysis`. These files only provide editable preambles for manual use.
