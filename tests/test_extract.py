"""Step 4 checks: every SI table has its expected rows, and values agree with Kraken."""
import pandas as pd

from ligsel.paths import INTERIM_DIR, PROCESSED_DIR, RESULTS_DIR


def test_every_table_has_expected_rows():
    rep = pd.read_csv(RESULTS_DIR / "tables" / "si_extraction_report.csv")
    assert rep["ok"].all(), rep[~rep["ok"]]


def test_si_vbur_agrees_with_kraken_2021():
    y = pd.read_csv(INTERIM_DIR / "si_yields.csv")
    k = pd.read_csv(PROCESSED_DIR / "kraken_2021.csv.gz")[["kraken_id", "pct_vbur_min", "pct_vbur_boltz"]]
    m = y.merge(k, left_on="kraken_id_si", right_on="kraken_id", how="left")
    assert m["kraken_id"].notna().all(), "an SI ID is missing from the 2021 release"
    assert ((m["vbur_min"] - m["pct_vbur_min"]).abs() <= 0.1).all()
    assert ((m["vbur_boltz"] - m["pct_vbur_boltz"]).abs() <= 0.1).all()


def test_published_thresholds_present():
    t = pd.read_csv(INTERIM_DIR / "si_thresholds.csv")
    s8 = t[t["si_table"] == "S8"]
    assert len(s8) == 12
    assert s8.set_index("dataset").loc["cliffs_I", "threshold"] == 31.60
