"""Which ligands do the models get wrong, and do the failures form a chemical class?

Per-ligand out-of-fold predictions are not stored by the ladder, so they are recomputed here for the
two models that matter: the primary elastic net ('all' view) and baseline_cv, leave-one-ligand-out,
same code and seed as step 17. Each dataset's recomputed MCC must equal ladder_per_dataset.csv
exactly, or the run stops. For the external set, the frozen model's per-ligand predictions are
recomputed from models/frozen_ni_pool.json and must reproduce external_test.csv's confusion matrix;
nothing is refitted and the frozen numbers are not rescored.

Outputs
  results/tables/ligand_errors.csv          every ligand: appearances, errors per model, class
  results/tables/error_by_class.csv         error rate by ligand class, model and error type
  results/tables/external_errors.csv        the 43 external ligands, each model's prediction
  results/figures/worst_ligands.png         structures of the most often misclassified ligands
  results/figures/external_actives.png      the 12 external actives, marked by whether the frozen model found them
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.model_selection import LeaveOneOut

from ligsel.baseline import is_buchwald
from ligsel.features import dropped_columns, feature_columns
from ligsel.freeze import FROZEN
from ligsel.models import VMIN, Stump, _fit_predict, elastic_net, load_dataset
from ligsel.paths import PROCESSED_DIR, RESULTS_DIR, load_config
from ligsel.splits import ligand_class

RUNS = PROCESSED_DIR / "error_runs"
MODELS = {"elastic_net": lambda cols: elastic_net(cols), "baseline_cv": lambda cols: Stump(column=VMIN)}


def eligible(y) -> bool:
    return len(y) >= 30 and min(y.sum(), len(y) - y.sum()) >= 10


def loo_predictions(budget_s: float | None = None) -> bool:
    """Checkpointed per dataset and model; returns True when every run exists."""
    import time
    t0 = time.time()
    RUNS.mkdir(parents=True, exist_ok=True)
    cols = feature_columns("all", dropped_columns())
    cfg = load_config()["datasets"]
    done = True
    for ds in [d for d, c in cfg.items() if c["role"] == "primary"]:
        X, y = load_dataset(ds)
        for name, make in MODELS.items():
            f = RUNS / f"{ds}__{name}.csv"
            if f.exists() or (name == "elastic_net" and not eligible(y)):
                continue
            if budget_s and time.time() - t0 > budget_s:
                done = False
                continue
            res = Parallel(n_jobs=4)(delayed(_fit_predict)(make(cols), X, y, tr, te)
                                     for tr, te in LeaveOneOut().split(X, y))
            pred = np.full(len(y), -1)
            for te, p, _ in res:
                pred[te] = p
            pd.DataFrame({"dataset": ds, "kraken_id": X["kraken_id"], "active": y, "pred": pred}).to_csv(f, index=False)
            print(f"{ds:12s} {name:12s} LOO MCC {matthews_corrcoef(y, pred):.4f}", flush=True)
    return done


def run(budget_s: float | None = None):
    if not loo_predictions(budget_s):
        print("runs still to do; rerun to continue")
        return
    ladder = pd.read_csv(RESULTS_DIR / "tables" / "ladder_per_dataset.csv")
    k = pd.read_csv(PROCESSED_DIR / "kraken_2025.csv.gz", usecols=["kraken_id", "smiles", "pct_vbur_min"]).set_index("kraken_id")
    names = pd.read_csv(PROCESSED_DIR / "joined.csv").dropna(subset=["id_2025"]).drop_duplicates("id_2025")
    names = names.set_index(names["id_2025"].astype(int))["ligand"]

    long = []
    for f in sorted(RUNS.glob("*.csv")):
        d = pd.read_csv(f)
        ds, model = f.stem.split("__")
        ref = ladder[(ladder.dataset == ds) & (ladder.view == "all") & (ladder.rung == model)]["mcc_loo"].iloc[0]
        got = round(matthews_corrcoef(d["active"], d["pred"]), 4)
        if abs(got - ref) > 1e-4:
            sys.exit(f"{ds} {model}: recomputed LOO MCC {got} differs from the ladder's {ref}")
        long.append(d.assign(model=model))
    long = pd.concat(long, ignore_index=True)
    long["error"] = np.where(long["pred"] == long["active"], "",
                             np.where(long["active"] == 1, "missed active", "false active"))
    long["class"] = long["kraken_id"].map(lambda i: ligand_class(k.loc[i, "smiles"]))
    long["buchwald_type"] = long["kraken_id"].map(lambda i: is_buchwald(k.loc[i, "smiles"]))

    # per ligand: only datasets where both models were fitted, so the two error rates are comparable
    both = long[long["dataset"].isin(long.loc[long.model == "elastic_net", "dataset"].unique())]
    per = (both.assign(wrong=both["error"] != "")
               .pivot_table(index=["kraken_id", "dataset"], columns="model", values="wrong", aggfunc="first")
               .reset_index())
    lig = per.groupby("kraken_id").agg(appearances=("dataset", "size"), datasets=("dataset", " ".join),
                                       en_errors=("elastic_net", "sum"), baseline_errors=("baseline_cv", "sum"))
    lig["en_error_rate"] = (lig["en_errors"] / lig["appearances"]).round(3)
    lig["baseline_error_rate"] = (lig["baseline_errors"] / lig["appearances"]).round(3)
    lig["ligand"] = names.reindex(lig.index).values
    lig["class"] = [ligand_class(k.loc[i, "smiles"]) for i in lig.index]
    lig["buchwald_type"] = [is_buchwald(k.loc[i, "smiles"]) for i in lig.index]
    lig["pct_vbur_min"] = k.loc[lig.index, "pct_vbur_min"].round(2).values
    lig = lig.reset_index().sort_values(["en_errors", "en_error_rate", "appearances"], ascending=False)
    lig.to_csv(RESULTS_DIR / "tables" / "ligand_errors.csv", index=False)

    cls = (both.groupby(["class", "model"])
               .apply(lambda g: pd.Series({"predictions": len(g), "error_rate": round((g.error != "").mean(), 3),
                                           "missed_active_rate": round((g.error == "missed active").sum() / max((g.active == 1).sum(), 1), 3),
                                           "false_active_rate": round((g.error == "false active").sum() / max((g.active == 0).sum(), 1), 3)}),
                      include_groups=False)
               .reset_index())
    cls.to_csv(RESULTS_DIR / "tables" / "error_by_class.csv", index=False)

    # external set: per-ligand predictions from the frozen numbers, checked against the committed result
    spec = json.loads(FROZEN.read_text())
    j = pd.read_csv(PROCESSED_DIR / "joined.csv")
    j = j[(j["dataset"] == "j22_III") & j["included"]]
    kk = pd.read_csv(PROCESSED_DIR / "kraken_2025.csv.gz").set_index("kraken_id")
    X = kk.loc[j["id_2025"].astype(int).values].reset_index()
    y = j["active"].astype(bool).astype(int).values
    en = spec["elastic_net"]
    z = (X[en["columns"]].values - np.array(en["scaler_mean"])) / np.array(en["scaler_scale"])
    score = z @ np.array(en["coef"]) + en["intercept"]
    ext = pd.DataFrame({"kraken_id": X["kraken_id"], "ligand": j["ligand"].values, "active": y,
                        "pct_vbur_min": X["pct_vbur_min"].round(2), "class": X["smiles"].map(ligand_class),
                        "buchwald_type": X["smiles"].map(is_buchwald),
                        "in_ni_pool": X["kraken_id"].isin(spec["training"]["ligand_ids"]),
                        "elastic_net_score": score.round(3), "elastic_net": (score > 0).astype(int),
                        "threshold_published": (X["pct_vbur_min"] <= 31.89).astype(int)})
    committed = pd.read_csv(RESULTS_DIR / "tables" / "external_test.csv").query("subset == 'all'").set_index("model")
    for m in ("elastic_net", "threshold_published"):
        tn, fp, fn, tp = confusion_matrix(ext["active"], ext[m], labels=[0, 1]).ravel()
        c = committed.loc["elastic_net_frozen" if m == "elastic_net" else m]
        assert (tn, fp, fn, tp) == (c.tn, c.fp, c.fn, c.tp), f"{m}: per-ligand predictions do not match external_test.csv"
    ext.sort_values(["active", "elastic_net_score"]).to_csv(RESULTS_DIR / "tables" / "external_errors.csv", index=False)

    worst = lig[(lig["appearances"] >= 3) & (lig["en_error_rate"] >= 0.5)].head(16)
    mols = [Chem.MolFromSmiles(k.loc[i, "smiles"]) for i in worst["kraken_id"]]
    legends = [f"{n} ({c})\nEN wrong {e}/{a}, threshold wrong {b}/{a}" for n, c, e, b, a in
               zip(worst["ligand"], worst["class"], worst["en_errors"], worst["baseline_errors"], worst["appearances"])]
    img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(320, 280), legends=legends)
    img.save(RESULTS_DIR / "figures" / "worst_ligands.png")

    act = ext[ext["active"] == 1].sort_values("elastic_net_score")
    img = Draw.MolsToGridImage(
        [Chem.MolFromSmiles(kk.loc[i, "smiles"]) for i in act["kraken_id"]], molsPerRow=4, subImgSize=(320, 280),
        legends=[f"{n} ({c}), %Vbur(min) {v:.1f}\nfrozen EN {'found' if p else 'MISSED'}; threshold {'found' if t else 'MISSED'}"
                 + ("" if pool else "; not in Ni pool")
                 for n, c, v, p, t, pool in zip(act["ligand"], act["class"], act["pct_vbur_min"], act["elastic_net"],
                                                 act["threshold_published"], act["in_ni_pool"])])
    img.save(RESULTS_DIR / "figures" / "external_actives.png")

    print(lig.head(20).to_string(index=False), "\n")
    print(cls.to_string(index=False), "\n")
    print(f"external: {len(ext)} ligands written; worst-ligand figure: {len(worst)} structures")


if __name__ == "__main__":
    run(float(sys.argv[1]) if len(sys.argv) > 1 else None)
