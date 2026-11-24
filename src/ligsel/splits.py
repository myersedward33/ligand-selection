"""Step 18: validation split strategies, pooled models and the ligand-overlap check.

Per dataset (the eleven primary datasets):
  loo, rep5x5         from step 17 (results/tables/ladder_per_dataset.csv)
  leave_class_out     hold out one ligand class at a time (classes below)

Pooled (pre-registration s. 6). Pools: all eleven (with a Ni/Pd indicator feature, as the
pre-registration states the pool), and, added in step 18 before any pooled result was seen, the Ni
pool (I-V) and the Pd pool (VII-XII) separately, because the published thresholds point in opposite
directions on the two metals and a single linear model cannot represent that:
  leave_dataset_out   train on the other datasets of the pool, test on the held-out one
  group_ligand_5fold  5-fold CV grouped by ligand identity: no ligand in both train and test
  leave_class_out     hold out one ligand class across the whole pool

Rungs: baseline_cv, pair_min_boltz, elastic_net ('all' view), random_forest ('all' view).
Ligand classes, from the structure at phosphorus:
  biaryl        Buchwald-type: P on an aryl whose ortho ring atom carries another aryl
  P-heteroatom  any P-O, P-N or other non-carbon substituent (phosphites, amidites, aminophosphines)
  trialkyl      three non-aromatic carbon substituents
  triaryl       three aromatic carbon substituents
  alkyl-aryl    the rest
"""
from __future__ import annotations

import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from sklearn.model_selection import GroupKFold

from ligsel.features import dropped_columns, feature_columns
from ligsel.models import (VBOLTZ, VMIN, PairLogistic, Stump, ColumnModel, elastic_net, load_dataset,
                           random_forest)
from ligsel.paths import PROCESSED_DIR, RESULTS_DIR, load_config

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
BIARYL = Chem.MolFromSmarts("[PX3]-[a;r5,r6]:[a]-!@[a]")
CHECKPOINTS = PROCESSED_DIR / "split_runs"
NI = ["cliffs_I", "cliffs_II", "cliffs_III", "cliffs_IV", "cliffs_V"]
PD = ["cliffs_VII", "cliffs_VIII", "cliffs_IX", "cliffs_X", "cliffs_XI", "cliffs_XII"]


def ligand_class(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    ps = [a for a in mol.GetAtoms() if a.GetSymbol() == "P" and a.GetDegree() == 3]
    if not ps:
        return "other"
    nbrs = ps[0].GetNeighbors()
    if any(n.GetSymbol() != "C" for n in nbrs):
        return "P-heteroatom"
    if mol.HasSubstructMatch(BIARYL):
        return "biaryl"
    n_ar = sum(n.GetIsAromatic() for n in nbrs)
    return {0: "trialkyl", 3: "triaryl"}.get(n_ar, "alkyl-aryl")


def rungs(drop):
    cols = feature_columns("all", drop)
    return {"baseline_cv": Stump(column=VMIN), "pair_min_boltz": PairLogistic(pair=(VMIN, VBOLTZ)),
            "elastic_net": elastic_net(cols), "random_forest": random_forest(cols)}


def pool(datasets, metal_feature=False):
    Xs, ys, meta = [], [], []
    for ds in datasets:
        X, y = load_dataset(ds)
        if metal_feature:
            X = X.assign(metal_is_ni=float(ds in NI))
        Xs.append(X); ys.append(y)
        meta.append(pd.DataFrame({"dataset": ds, "kraken_id": X["kraken_id"].values}))
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys), pd.concat(meta, ignore_index=True)


def _fit_predict(model, X, y, tr, te):
    m = clone(model).fit(X.iloc[tr].reset_index(drop=True), y[tr])
    return te, m.predict(X.iloc[te].reset_index(drop=True))


def evaluate(model, X, y, folds, n_jobs=4):
    res = Parallel(n_jobs=n_jobs)(delayed(_fit_predict)(model, X, y, tr, te) for tr, te in folds)
    pred = np.full(len(y), -1)
    for te, p in res:
        pred[te] = p
    return pred


def folds_by(labels):
    labels = np.asarray(labels)
    return [(np.where(labels != g)[0], np.where(labels == g)[0]) for g in pd.unique(labels)
            if (labels != g).sum() > 0]


def job_list():
    k = pd.read_csv(PROCESSED_DIR / "kraken_2025.csv.gz", usecols=["kraken_id", "smiles"]).set_index("kraken_id")
    drop = dropped_columns()
    out = []
    primary = NI + PD
    for ds in primary:
        X, y = load_dataset(ds)
        cls = X["kraken_id"].map(lambda i: ligand_class(k.loc[i, "smiles"])).values
        for rung, m in rungs(drop).items():
            if rung in ("elastic_net", "random_forest") and (len(y) < 30 or min(y.sum(), len(y) - y.sum()) < 10):
                continue
            out.append((f"{ds}__leave_class_out__{rung}", ds, "leave_class_out", rung, m, X, y, folds_by(cls)))
    for pname, members, metal in (("pool_all11", primary, True), ("pool_Ni", NI, False), ("pool_Pd", PD, False)):
        X, y, meta = pool(members, metal_feature=metal)
        cls = meta["kraken_id"].map(lambda i: ligand_class(k.loc[i, "smiles"])).values
        lig = meta["kraken_id"].values
        gkf = list(GroupKFold(n_splits=5).split(X, y, groups=lig))
        schemes = {"leave_dataset_out": folds_by(meta["dataset"].values), "group_ligand_5fold": gkf,
                   "leave_class_out": folds_by(cls)}
        for rung, m in rungs(drop).items():
            if metal and rung in ("elastic_net", "random_forest"):
                cols = feature_columns("all", drop) + ["metal_is_ni"]
                m = elastic_net(cols) if rung == "elastic_net" else random_forest(cols)
            for scheme, folds in schemes.items():
                out.append((f"{pname}__{scheme}__{rung}", pname, scheme, rung, m, X, y, folds))
    return out


def run(budget_s=None):
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    todo = [j for j in job_list() if not (CHECKPOINTS / f"{j[0]}.json").exists()]
    for key, scope, scheme, rung, model, X, y, folds in todo:
        if budget_s and time.time() - t0 > budget_s:
            break
        t1 = time.time()
        pred = evaluate(model, X, y, folds)
        rec = {"scope": scope, "scheme": scheme, "rung": rung, "n": len(y), "folds": len(folds),
               "mcc": matthews_corrcoef(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
               "seconds": round(time.time() - t1, 1)}
        (CHECKPOINTS / f"{key}.json").write_text(json.dumps(rec))
        print(f"{scope:12s} {scheme:18s} {rung:15s} mcc {rec['mcc']:.3f}  {rec['seconds']}s", flush=True)
    left = sum(1 for j in todo if not (CHECKPOINTS / f"{j[0]}.json").exists())
    if left:
        print(f"{left} runs still to do")
        return None
    tab = pd.DataFrame([json.loads(f.read_text()) for f in sorted(CHECKPOINTS.glob("*.json"))])
    tab.round(4).to_csv(RESULTS_DIR / "tables" / "split_strategies.csv", index=False)
    print(f"complete: {len(tab)} runs")
    return tab


def overlap_and_classes():
    """Ligands shared between primary datasets, and the class of every modelled ligand."""
    k = pd.read_csv(PROCESSED_DIR / "kraken_2025.csv.gz", usecols=["kraken_id", "smiles"]).set_index("kraken_id")
    j = pd.read_csv(PROCESSED_DIR / "joined.csv")
    cfg = load_config()["datasets"]
    j = j[j["included"] & j["dataset"].map(lambda d: cfg[d]["role"] == "primary")]
    per_lig = j.groupby("id_2025").agg(n_datasets=("dataset", "nunique"),
                                       datasets=("dataset", lambda s: " ".join(sorted(set(s)))),
                                       ligand=("ligand", "first")).reset_index()
    per_lig["class"] = per_lig["id_2025"].map(lambda i: ligand_class(k.loc[int(i), "smiles"]))
    per_lig.to_csv(RESULTS_DIR / "tables" / "ligand_classes_and_overlap.csv", index=False)
    print(per_lig["class"].value_counts().to_string())
    print(f"\n{len(per_lig)} distinct ligands; {int((per_lig.n_datasets > 1).sum())} appear in more than one "
          f"primary dataset; max {int(per_lig.n_datasets.max())}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "overlap":
        overlap_and_classes()
    else:
        run(float(sys.argv[1]) if len(sys.argv) > 1 else None)
