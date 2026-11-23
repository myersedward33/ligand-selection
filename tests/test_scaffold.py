"""Step 1 checks: the config is complete and the raw files match the registered snapshot."""
from ligsel.acquire import verify
from ligsel.paths import load_config

REQUIRED_KEYS = {
    "kraken_2025", "kraken_2021_descriptors", "kraken_2021_model_scores",
    "cliffs_si", "jacs2022_si",
}


def test_config_lists_required_files():
    files = load_config()["files"]
    assert REQUIRED_KEYS <= set(files)
    assert all("source" in spec for spec in files.values())


def test_raw_files_match_manifest():
    assert verify() == []
