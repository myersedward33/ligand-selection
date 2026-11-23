# ligand-selection

**Question.** Across eleven Ni- and Pd-catalysed cross-coupling datasets, one
steric descriptor, %V_bur(min), separates active from inactive monodentate
phosphines (Newman-Stonebraker et al., *Science* 2021). Does a multivariate
model over Kraken's full 190-descriptor set do better, and if so, which
descriptors carry the extra signal?

**Status.** In progress, December 2026. Results and the writeup land here when
the analysis is complete.

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

The descriptors and reaction data are other people's work: Kraken (Gensch et
al., *JACS* 2022), the reactivity-cliffs study (Newman-Stonebraker et al.,
*Science* 2021) and its follow-up (*JACS* 2022). This repository's contribution
is the comparative analysis. Code is MIT licensed; data is not redistributed.
