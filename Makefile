# Every table and figure is rebuilt from the raw files by `make all`.
PY ?= python

.PHONY: all register verify test clean

all: verify

register:   ## record checksums of the raw files (run once per snapshot)
	$(PY) -m ligsel.acquire register

verify:     ## check the raw files match data/raw_manifest.json
	$(PY) -m ligsel.acquire verify

test:
	$(PY) -m pytest -q

clean:      ## remove regenerable outputs
	rm -rf data/interim/* data/processed/* results/figures/* results/tables/*
