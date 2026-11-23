# Data

No raw data is committed. The raw files live outside this repository, in the
folder named by `raw_dir` in `config.yaml` (by default the parent folder), or in
`$LIGSEL_RAW_DIR`. Each file's source is listed in `config.yaml`.

- `raw_manifest.json` records the size and SHA-256 of each raw file. It is
  committed so every result is tied to a stated snapshot. `make verify` checks it.
- `interim/` and `processed/` are rebuilt by `make all` and are not committed,
  because tables extracted from the supporting information are covered by the
  publishers' terms (CC BY-NC-ND 4.0 for the ChemRxiv SI, ACS terms for the JACS SI).
