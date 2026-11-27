# ligand-selection

**Question.** Across the reactivity-cliffs datasets in Univariate Classification of Phosphine Ligation State and Reactivity in Cross-Coupling Catalysis, a single steric descriptor, %V_bur(min), classifies monodentate phosphines as active or inactive at a threshold near 32% (Newman-Stonebraker, 2021). Does a multivariate model over Kraken's 190 DFT descriptors classify this threshold better, and if so, which descriptors carry the extra signal

## Run it

```bash
brew install poppler   # provides pdftotext, used to read the SI tables
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
make verify   # checks the raw files against data/raw_manifest.json
make test
make all      # rebuilds every table and figure from the raw files
```

The raw files are not in this repository. Put them where `config.yaml` says, or
point `LIGSEL_RAW_DIR` at the folder holding them; their sources are listed there.

## Layout

```
config.yaml        raw-file registry and sources
src/ligsel/        all code that produces a result
tests/             checks on the pipeline
data/              manifest only; interim/ and processed/ are regenerated
results/           figures and tables produced by make all
notebooks/         exploration only, never the source of a result
```

## Credit

The descriptors and reaction data are the work of others: Kraken (Gensch et
al., *JACS* 2022), the reactivity-cliffs study (Newman-Stonebraker et al.,
*Science* 2021) and its follow-up (*JACS* 2022). This repository's contribution
is the comparative analysis. The code is MIT licensed and the data is not redistributed.
