"""Steps 9, 13 and 14: the join is complete, the views are as specified, the baseline code is right."""
import pandas as pd

from ligsel.features import feature_columns
from ligsel.paths import RESULTS_DIR


def test_every_modelled_dataset_fully_matched():
    rep = pd.read_csv(RESULTS_DIR / "tables" / "join_report.csv")
    used = rep[rep["role"].isin(["primary", "control", "external"])]
    assert (used["match_rate"] == 1.0).all()


def test_feature_views():
    sizes = {v: len(feature_columns(v)) for v in ("boltz", "min", "max", "vburminconf", "all")}
    assert sizes == {"boltz": 78, "min": 76, "max": 78, "vburminconf": 75, "all": 185}
    assert "vbur_vbur_vburminconf" not in feature_columns("all")


def test_baseline_code_reproduces_published_thresholds():
    # Where the SI data and Table S8 agree, the threshold must reproduce to 0.01
    t = pd.read_csv(RESULTS_DIR / "tables" / "baseline_reproduction.csv")
    t = t[t["subset"] == "all ligands"].set_index("dataset")
    for ds in ["cliffs_I", "cliffs_II", "cliffs_III", "cliffs_IV", "cliffs_V", "cliffs_VIII", "cliffs_X",
               "cliffs_XI", "cliffs_XII", "cliffs_RS1"]:
        assert abs(t.loc[ds, "threshold_kraken"] - t.loc[ds, "threshold_published"]) <= 0.01, ds
