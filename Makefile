# Every table and figure is rebuilt from the raw files by `make all`.
PY ?= python

.PHONY: all register verify kraken extract join features characterise baseline models hypotheses splits selection controls freeze external tepid test clean

TEPID_PY ?= $(HOME)/venv-tepid/bin/python
TEPID_DIR ?= $(HOME)/ext/TEPid

all: verify kraken extract join features characterise baseline models hypotheses splits selection controls


register:   ## record checksums of the raw files (run once per snapshot)
	$(PY) -m ligsel.acquire register

verify:     ## check the raw files match data/raw_manifest.json
	$(PY) -m ligsel.acquire verify

kraken:     ## step 3: units, column families, 2021-2025 ID crosswalk
	$(PY) -m ligsel.parse_kraken

extract:    ## step 4: SI yield and threshold tables
	$(PY) -m ligsel.extract_si

join:       ## step 9: SI ligands to Kraken 2025 by structure; stops below min_match_rate
	$(PY) -m ligsel.join

features:   ## step 13: the five feature views and dropped columns
	$(PY) -m ligsel.features

tepid:      ## step 11 (optional, separate env): TEPid predictions, see scripts/run_tepid.py
	$(TEPID_PY) scripts/run_tepid.py $(TEPID_DIR) .

characterise:  ## steps 11-12: correlation figure, TEP figure (needs make tepid first)
	$(PY) -m ligsel.characterise

baseline:   ## step 14: reproduce the published thresholds (the gate)
	$(PY) -m ligsel.baseline

models:     ## step 17: the model ladder (resumable; about an hour on 4 cores)
	$(PY) -m ligsel.models

hypotheses: ## score H1-H3 from the ladder
	$(PY) -m ligsel.summarise

splits:     ## step 18: leave-class-out, pooled leave-dataset-out and grouped CV; ligand overlap
	$(PY) -m ligsel.splits overlap
	$(PY) -m ligsel.splits

selection:  ## step 20a: stability selection (500 bootstraps) and best-pair choices
	$(PY) -m ligsel.selection pairs
	$(PY) -m ligsel.selection

controls:   ## step 20b: negative controls, 500-permutation nulls (H4)
	$(PY) -m ligsel.controls

freeze:     ## step 22: fit and freeze the external-test model (models/frozen_ni_pool.json); not part of all
	$(PY) -m ligsel.freeze

external:   ## step 23: score the frozen model ONCE on JACS 2022 III; refuses without the frozen tag
	$(PY) -m ligsel.external

test:
	$(PY) -m pytest -q

clean:      ## remove regenerable outputs
	rm -rf data/interim/* data/processed/* results/figures/* results/tables/*
