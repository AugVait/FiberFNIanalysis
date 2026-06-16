# Analysis Configs

These YAML files are part of the Git-synced method definition.

- `run_all.yaml`: chooses which analysis families run, points to the other config files, and can optionally override carpet time windows, IT trace discovery window, and PL wavelength-axis limits.
- `snakemake.yaml`: workflow orchestration settings for Snakemake targets, including output roots, wavelength-cut split-job settings, and downstream target options.
- `carpets.yaml`: Hamamatsu carpet quicklook settings, including output subfolder, top-edge crop, and the detailed scan-config directory.
- `fiber_names.yaml`: maps experimental result names such as `bcf1_noir` to real fiber names for generated plot labels and result tables.
- `peak_position_shift.yaml`: PL peak-position-shift settings, including output subfolder, selected PL config folder, smoothing/refinement settings, and explicit point exclusions.
- `carpets/*.yaml`: curated streak-camera `.img` scan membership and scan metadata by sample group.
- `it_decay_fits_all_it_10ns_window.yaml`: integrated-time decay fitting settings for all `IT_*.dat` traces, with each fit capped at 10 ns from the detected peak. It also records plot-only choices such as excluding 2 ns points from the scatter plot and using low-opacity per-sample linear guide fits.
- `it_decay_fits_all_it_10ns_window/*.yaml`: integrated-time trace metadata by sample group for the all-IT analysis. These files limit fitting only if `fit_all_discovered_traces` is set to `false`.
- `it_decay_fits_10ns.yaml`: legacy 10 ns filename-only integrated-time decay fitting settings.
- `pl_spectra/*.yaml`: curated PL spectrum membership and plotting metadata by sample group.
- `wavelength_cut_selections/decay_time_10ns/*.txt`: tracked manual selection matrices used by reproducible summary grids, leading-edge reports, and selected double-exponential decay fits.

Paths inside `run_all.yaml`, `carpets.yaml`, IT decay config YAML files, and `peak_position_shift.yaml` are relative to this `configs` folder unless absolute paths are used. Paths inside `carpets/*.yaml`, IT trace metadata YAML folders, and `pl_spectra/*.yaml` are relative to the raw-data directory passed with `--raw-dir`.

In `run_all.yaml`, leave `carpet_time_windows`, `it_time_window`, `pl_x_min_nm`, and `pl_x_max_nm` empty to use each method's normal config. The default IT decay config sets `time_window: "all"` for file discovery and `fit_window_ns: 10.0` for the fit interval. Set `carpet_time_windows` to a comma-separated list such as `"2ns,10ns"` to restrict carpet quicklooks. Set `it_time_window` only when you intentionally want to filter IT input files by filename window. Set `pl_x_min_nm` and/or `pl_x_max_nm` to override PL spectral plot limits globally.

In `fiber_names.yaml`, leave a real-name value empty until it is known. Empty or missing mappings fall back to the experimental name, so existing configs remain valid.

For all selected-data methods, `include: true` selects a scan, spectrum, or trace for figure generation. Entries with `include: false` remain documented in the YAML but are not plotted or fit.
