# FiberFNIanalysis

LHCb fibers analysis workspace.

This workspace is arranged so Git can sync the reproducible analysis methods while large or generated material stays local.

## Sections

- `Analysis methods`: tracked analysis code, configs, setup notes, and the raw-data manifest.
- `raw data`: local measurement files. This folder is intentionally ignored by Git.
- `Analysis results`: generated outputs. This folder is intentionally ignored by Git.
- `Analysis Old`: legacy analysis archive. This folder is intentionally ignored by Git.

## Git Rule

Use the project root as the Git repository. The root `.gitignore` is allowlist-style: it tracks this README, `.gitignore`, and `Analysis methods/**`; everything else is ignored unless the ignore file is intentionally changed.

## First Check On A New PC

Copy the raw data into a sibling folder named `raw data`, then run:

```powershell
cd "Analysis methods"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m lhcb_fibers_analysis.verify_project --raw-dir "..\raw data"
```

If the raw-data check fails, the copied data does not match the manifest synced through Git.

After the check passes, the preferred way to rebuild tracked analysis outputs is Snakemake:

```powershell
cd "Analysis methods"
.\.venv\Scripts\python.exe -m snakemake --dry-run --cores 1
.\.venv\Scripts\python.exe -m snakemake --cores 4
```

Useful named targets include `core_results`, `wavelength_cut_results`, `summary_grids`, and `double_exp_results`.

The manual scripts still live in `Analysis methods\manual_scripts` for one-off runs. Edit the preamble at the top of a script when you want a narrower manual run.
