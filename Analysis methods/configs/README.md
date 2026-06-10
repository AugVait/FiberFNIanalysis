# Analysis Configs

These YAML files are part of the Git-synced method definition.

- `run_all.yaml`: chooses which analysis families run and points to the other config files.
- `carpets.yaml`: Hamamatsu carpet quicklook settings, including output subfolder and top-edge crop.
- `it_decay_fits_10ns.yaml`: integrated-time decay fitting settings for the 10 ns trace set.
- `pl_spectra/*.yaml`: curated PL spectrum membership and plotting metadata by sample group.

Paths inside `run_all.yaml` are relative to this `configs` folder unless absolute paths are used. Paths inside `pl_spectra/*.yaml` are relative to the raw-data directory passed with `--raw-dir`.
