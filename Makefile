# Every table and figure is rebuilt from the raw files by `make all`.
PY ?= python

.PHONY: all register verify kraken extract test clean

all: verify kraken extract


register:   ## record checksums of the raw files (run once per snapshot)
	$(PY) -m ligsel.acquire register

verify:     ## check the raw files match data/raw_manifest.json
	$(PY) -m ligsel.acquire verify

kraken:     ## step 3: units, column families, 2021-2025 ID crosswalk
	$(PY) -m ligsel.parse_kraken

extract:    ## step 4: SI yield and threshold tables
	$(PY) -m ligsel.extract_si

test:
	$(PY) -m pytest -q

clean:      ## remove regenerable outputs
	rm -rf data/interim/* data/processed/* results/figures/* results/tables/*
