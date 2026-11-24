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


def test_ligand_classes():
    from ligsel.splits import ligand_class
    assert ligand_class("c1ccc(cc1)P(c1ccccc1)c1ccccc1") == "triaryl"                        # PPh3
    assert ligand_class("CC(C)(C)P(C(C)(C)C)C(C)(C)C") == "trialkyl"                          # PtBu3
    assert ligand_class("c1ccc(cc1)OP(Oc1ccccc1)Oc1ccccc1") == "P-heteroatom"                 # P(OPh)3
    assert ligand_class("CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1") == "biaryl"  # XPhos
    assert ligand_class("CP(c1ccccc1)c1ccccc1") == "alkyl-aryl"                              # PMePh2


def test_fast_stump_matches_tree_stump():
    from ligsel.controls import FastBestStump
    from ligsel.features import dropped_columns, feature_columns
    from ligsel.models import Stump, load_dataset
    cols = feature_columns("all", dropped_columns())
    for ds in ("cliffs_I", "cliffs_RS2"):
        X, y = load_dataset(ds)
        a, b = Stump(candidates=cols).fit(X, y), FastBestStump(candidates=cols).fit(X, y)
        assert a.selected_ == b.selected_ and (a.predict(X) == b.predict(X)).all()


def test_gradient_boost_actually_splits():
    # regression test: with the library's default min_samples_leaf (20), every tree was a single leaf
    from ligsel.features import dropped_columns, feature_columns
    from ligsel.models import gradient_boost, load_dataset
    X, y = load_dataset("cliffs_IX")
    m = gradient_boost(feature_columns("all", dropped_columns())).fit(X, y)
    trees = m.model_.best_estimator_._predictors
    assert max(p.get_n_leaf_nodes() for t in trees for p in t) > 1
