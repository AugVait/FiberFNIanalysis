# Manual Carpet Wavelength-Cut Fits

This guide is for exploratory analysis of one Hamamatsu streak-camera `.img` carpet at a time. The helper script loads a carpet, cuts it into wavelength bands, averages each band into a time profile, fits a single exponential decay to each profile, and writes CSV/PNG diagnostics.

The script is:

```powershell
python -m lhcb_fibers_analysis.carpet_wavelength_cuts
```

Run it from the `Analysis methods` folder with the local virtual environment.

## 1. Open PowerShell In The Methods Folder

```powershell
cd "C:\WorkWork\LHCb Fibers\Analysis methods"
```

If the environment is not installed yet:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Basic Example

This example analyzes one 10 ns BCF-6 carpet using 10 nm-wide wavelength cuts every 10 nm from 400 to 540 nm:

```powershell
.\.venv\Scripts\python.exe -m lhcb_fibers_analysis.carpet_wavelength_cuts `
  "..\raw data\2026 04 17\bcf_6\bcf6_ir_100cm_ex360nm_10nJ_10ns.img" `
  --wavelength-min-nm 400 `
  --wavelength-max-nm 540 `
  --step-nm 10 `
  --band-width-nm 10 `
  --fit-start-offset-ns 0.05
```

By default, output goes to:

```text
..\Analysis results\manual_carpet_wavelength_cuts\<input-file-name>\
```

## 3. What The Script Does

For each wavelength cut, the script:

1. Uses the `.img` wavelength calibration metadata to locate wavelength columns.
2. Selects columns inside the requested wavelength band.
3. Averages those columns to make one time profile.
4. Finds the profile peak.
5. Detects significant secondary peaks.
6. Fits the falling edge of the dominant peak with:

```text
I(t) = A exp(-(t - t_start) / tau) + B
```

The fit starts at either:

- `--fit-start-ns`, if you provide an absolute start time, or
- detected peak time plus `--fit-start-offset-ns`.

The fit ends at either:

- `--fit-end-ns`, if provided, or
- the point where the smoothed signal falls below `--end-fraction` of peak signal, or
- the end of the trace if `--end-fraction 0` is used.

If a significant secondary peak is detected after the dominant peak, the fit end is automatically moved earlier, just before that secondary peak. This prevents a delayed bump or reflected feature from being included in the single-exponential decay fit.

## 4. Output Files

Main files:

```text
fit_summary.csv
cut_profiles.csv
tau_vs_wavelength.png
profiles_overlay.png
carpet_with_cuts.png
fits\cut_<center>nm_fit.png
```

`fit_summary.csv` is usually the main result. It contains:

- wavelength cut center and actual wavelength bounds
- fit status
- `tau_ns`
- `tau_se_ns`
- fitted amplitude and baseline
- `R^2`
- peak time and peak counts
- detected peak count and detected peak times
- secondary peak count and secondary peak times
- fit end rule, including `secondary_peak_excluded` when relevant
- fit start/end times
- number of fit points
- path to the per-cut diagnostic plot

`cut_profiles.csv` contains the full time profile for every wavelength cut. Use it if you want to replot or refit manually.

`tau_vs_wavelength.png` shows the fitted decay time versus wavelength cut center.

`profiles_overlay.png` overlays all wavelength-cut time profiles.

`carpet_with_cuts.png` shows the original carpet with the wavelength cuts drawn on top.

## 5. Choosing Wavelength Cuts

Use regular cuts:

```powershell
--wavelength-min-nm 400 --wavelength-max-nm 540 --step-nm 10 --band-width-nm 10
```

This creates centers at:

```text
400, 410, 420, ..., 540 nm
```

Each center uses a band of width `--band-width-nm`. For example, a center of 460 nm and width 10 nm uses roughly 455-465 nm, depending on pixel calibration.

Use explicit centers:

```powershell
--centers "420,460,500" --band-width-nm 10
```

This ignores `--wavelength-min-nm`, `--wavelength-max-nm`, and `--step-nm`.

## 6. Choosing Fit Windows

Start slightly after the peak:

```powershell
--fit-start-offset-ns 0.05
```

Start at an exact time:

```powershell
--fit-start-ns 1.6
```

End at an exact time:

```powershell
--fit-end-ns 7.5
```

Fit until the end of the trace:

```powershell
--end-fraction 0
```

For noisy tails, an exact `--fit-end-ns` is often cleaner than automatic ending.

Secondary peaks are still excluded even when `--fit-end-ns` is supplied. If a secondary peak appears before the requested end time, the fit is shortened to stop before that secondary peak.

## 7. Useful Recipes

Broad overview scan:

```powershell
.\.venv\Scripts\python.exe -m lhcb_fibers_analysis.carpet_wavelength_cuts `
  "..\raw data\2026 04 17\bcf_6\bcf6_ir_100cm_ex360nm_10nJ_10ns.img" `
  --wavelength-min-nm 380 `
  --wavelength-max-nm 620 `
  --step-nm 20 `
  --band-width-nm 20 `
  --fit-start-offset-ns 0.05
```

Higher spectral detail:

```powershell
.\.venv\Scripts\python.exe -m lhcb_fibers_analysis.carpet_wavelength_cuts `
  "..\raw data\2026 04 17\bcf_6\bcf6_ir_100cm_ex360nm_10nJ_10ns.img" `
  --wavelength-min-nm 420 `
  --wavelength-max-nm 540 `
  --step-nm 5 `
  --band-width-nm 8 `
  --fit-start-offset-ns 0.05
```

Manual fixed fit window:

```powershell
.\.venv\Scripts\python.exe -m lhcb_fibers_analysis.carpet_wavelength_cuts `
  "..\raw data\2026 04 17\bcf_6\bcf6_ir_100cm_ex360nm_10nJ_10ns.img" `
  --centers "420,440,460,480,500" `
  --band-width-nm 10 `
  --fit-start-ns 1.6 `
  --fit-end-ns 7.5
```

Write to a named output folder:

```powershell
.\.venv\Scripts\python.exe -m lhcb_fibers_analysis.carpet_wavelength_cuts `
  "..\raw data\2026 04 17\bcf_6\bcf6_ir_100cm_ex360nm_10nJ_10ns.img" `
  --centers "420,460,500" `
  --band-width-nm 10 `
  --out-dir "..\Analysis results\manual_carpet_wavelength_cuts\bcf6_100cm_selected"
```

Skip per-cut PNGs if you only want CSV plus overview plots:

```powershell
--no-individual-plots
```

## 8. Important Parameters

```text
--top-edge-crop-rows     Crops bright top-edge rows before extracting profiles. Default: 12.
--centers                Comma-separated wavelength centers, e.g. "420,460,500".
--wavelength-min-nm      First wavelength center for regular cuts.
--wavelength-max-nm      Last wavelength center for regular cuts.
--step-nm                Spacing between regular cut centers. Default: 10.
--band-width-nm          Width of each wavelength band. Default: 10.
--smooth-sigma           Gaussian smoothing used for peak/end detection. Default: 2.
--fit-start-ns           Absolute fit start time.
--fit-start-offset-ns    Fit start relative to detected peak. Default: 0.
--fit-end-ns             Absolute fit end time.
--end-fraction           Automatic fit end threshold as fraction of peak signal. Default: 0.05.
--min-fit-points         Minimum points required for a fit. Default: 20.
--min-peak-sigma         Low-signal rejection threshold. Default: 5.
--secondary-peak-height-fraction
                         Secondary peak must exceed this fraction of dominant peak height. Default: 0.20.
--secondary-peak-prominence-fraction
                         Secondary peak prominence threshold relative to dominant peak height. Default: 0.12.
--secondary-peak-noise-sigma
                         Secondary peak must also exceed this noise threshold. Default: 3.
--secondary-peak-min-separation-ns
                         Minimum separation between detected peaks. Default: 0.25.
--secondary-peak-exclusion-before-ns
                         Fit ends this long before the first later secondary peak. Default: 0.05.
--tau-min-ns             Lower tau bound. Default: 0.03.
--tau-max-ns             Upper tau bound. Default: 200.
```

## 9. Interpreting Problems

If a cut is marked `skipped` in `fit_summary.csv`, check the `reason` column.

Common reasons:

```text
low_signal
fit_start_after_trace
fit_end_before_trace
too_few_fit_points
secondary_peak_too_close
fit_failed:<details>
```

If many cuts are `low_signal`, narrow the wavelength range to the emission region, increase `--band-width-nm`, or reduce `--min-peak-sigma`.

If fits look too long or too short, inspect `fits\cut_<center>nm_fit.png` and set `--fit-start-ns` and `--fit-end-ns` manually.

If a late-time tail dominates the fit, reduce `--fit-end-ns` or increase `--end-fraction`.

If secondary peaks are not being excluded, lower `--secondary-peak-height-fraction`, lower `--secondary-peak-prominence-fraction`, or lower `--secondary-peak-noise-sigma`.

If the script is excluding too aggressively because noise is being treated as a peak, increase those same secondary-peak thresholds or increase `--secondary-peak-min-separation-ns`.

If the fitted tau jumps sharply at low-signal wavelengths, check the per-cut PNGs before trusting those points.

## 10. Suggested Workflow

1. Run a broad overview with `--step-nm 20 --band-width-nm 20`.
2. Open `carpet_with_cuts.png` and `profiles_overlay.png`.
3. Check `tau_vs_wavelength.png` for obvious trends or bad points.
4. Inspect suspicious per-cut plots under `fits/`.
5. Rerun with narrower wavelength centers and an explicit fit window.
6. Use `fit_summary.csv` for the final table/plot.
