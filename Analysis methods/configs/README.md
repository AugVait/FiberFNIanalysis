# Analysis Configs

These YAML files are part of the Git-synced method definition.

- `run_all.yaml`: chooses which analysis families run, points to the other config files, and can optionally override carpet time windows, IT trace time window, and PL wavelength-axis limits.
- `carpets.yaml`: Hamamatsu carpet quicklook settings, including output subfolder, top-edge crop, and the detailed scan-config directory.
- `fiber_names.yaml`: maps experimental result names such as `bcf1_noir` to real fiber names for generated plot labels and result tables.
- `carpets/*.yaml`: curated streak-camera `.img` scan membership and scan metadata by sample group.
- `it_decay_fits_10ns.yaml`: integrated-time decay fitting settings for the 10 ns trace set and the detailed trace-config directory.
- `it_decay_fits_10ns/*.yaml`: curated integrated-time trace membership and trace metadata by sample group.
- `pl_spectra/*.yaml`: curated PL spectrum membership and plotting metadata by sample group.

Paths inside `run_all.yaml`, `carpets.yaml`, and `it_decay_fits_10ns.yaml` are relative to this `configs` folder unless absolute paths are used. Paths inside `carpets/*.yaml`, `it_decay_fits_10ns/*.yaml`, and `pl_spectra/*.yaml` are relative to the raw-data directory passed with `--raw-dir`.

In `run_all.yaml`, leave `carpet_time_windows`, `it_time_window`, `pl_x_min_nm`, and `pl_x_max_nm` empty to use each method's normal config. Set `carpet_time_windows` to a comma-separated list such as `"2ns,10ns"` to restrict carpet quicklooks. Set `it_time_window` to a value such as `"10ns"` to override the IT trace firing/acquisition window. Set `pl_x_min_nm` and/or `pl_x_max_nm` to override PL spectral plot limits globally.

In `fiber_names.yaml`, leave a real-name value empty until it is known. Empty or missing mappings fall back to the experimental name, so existing configs remain valid.

For all three methods, `include: true` selects a scan or trace for figure generation. Entries with `include: false` remain documented in the YAML but are not plotted or fit.
