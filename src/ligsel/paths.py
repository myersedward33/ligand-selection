"""Repository paths and the raw-file registry from config.yaml."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.yaml"
DATA_DIR = REPO_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = REPO_ROOT / "results"
MANIFEST_PATH = DATA_DIR / "raw_manifest.json"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def raw_dir(config: dict | None = None) -> Path:
    """Folder holding the raw files: LIGSEL_RAW_DIR if set, else config raw_dir."""
    config = config or load_config()
    base = Path(os.environ.get("LIGSEL_RAW_DIR", config["raw_dir"]))
    return base if base.is_absolute() else (REPO_ROOT / base).resolve()


def raw_path(key: str, config: dict | None = None) -> Path:
    """Absolute path of one registered raw file, by its key in config.yaml."""
    config = config or load_config()
    return raw_dir(config) / config["files"][key]["path"]
