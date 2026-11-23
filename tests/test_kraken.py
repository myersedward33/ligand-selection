"""Step 3 checks: units, the descriptor anatomy and the ID crosswalk."""
import pandas as pd
import pytest

from ligsel.paths import PROCESSED_DIR, RESULTS_DIR


@pytest.fixture(scope="module")
def kraken():
    return pd.read_csv(PROCESSED_DIR / "kraken_2025.csv.gz").set_index("kraken_id")


@pytest.fixture(scope="module")
def crosswalk():
    return pd.read_csv(PROCESSED_DIR / "kraken_crosswalk.csv").set_index("id_2021")


def test_percent_vbur_matches_si(kraken):
    # SI: XPhos (ID 1) %Vbur(min) 31.4, (t-Bu)XPhos (ID 90) 56.7
    assert kraken.loc[1, "pct_vbur_min"] == pytest.approx(31.4, abs=0.05)
    assert kraken.loc[90, "pct_vbur_min"] == pytest.approx(56.7, abs=0.05)


def test_descriptor_anatomy():
    fam = pd.read_csv(RESULTS_DIR / "tables" / "descriptor_families.csv")
    per_prop = fam.groupby("property")["summary"].nunique()
    assert len(fam) == 190
    assert (per_prop == 5).sum() == 28 and (per_prop == 1).sum() == 50


def test_crosswalk_follows_structure_not_id(crosswalk):
    # 2021 ID 950 is PBn(t-Bu)2, which is ID 951 in 2025; 2025 ID 950 is another molecule
    assert crosswalk.loc[950, "id_2025"] == 951
    assert crosswalk.loc[950, "id_reused_for_other_structure"]
    assert crosswalk.loc[1, "id_2025"] == 1
