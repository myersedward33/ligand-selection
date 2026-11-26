"""Reproducing the published %V_bur(min) threshold classifier:

Method as in SigmanGroup/Threshold: a depth-1 DecisionTreeClassifier, class_weight='balanced',
on one descriptor, with active = yield >= the dataset's cutoff, fitted and scored on the whole
dataset (in sample, as published). It is run twice: on %V_bur(min) from Kraken (the pipeline's
own input) and on the rounded values printed in the SI (what the authors' tables show).

The external test set (role 'external') is never loaded here: pre-registration section 8.

Output: results/tables/baseline_reproduction.csv (committed; thresholds and metrics only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier

from ligsel.paths import INTERIM_DIR, PROCESSED_DIR, RESULTS_DIR, load_config


def fit_threshold(x: np.ndarray, active: np.ndarray, class_weight="balanced") -> dict:
    dt = DecisionTreeClassifier(max_depth=1, class_weight=class_weight, random_state=0)
    dt.fit(x.reshape(-1, 1), active)
    pred = dt.predict(x.reshape(-1, 1))
    thr = float(dt.tree_.threshold[0])
    left_class = int(np.argmax(dt.tree_.value[dt.tree_.children_left[0]]))
    return {"threshold": thr, "direction": "Left" if left_class == 1 else "Right",
            "accuracy": accuracy_score(active, pred), "f1": f1_score(active, pred),
            "mcc": matthews_corrcoef(active, pred)}


# Buchwald-type (dialkyl-ortho-biaryl and N-aryl azole) phosphines: P on an aromatic ring whose
# neighbouring ring atom carries another aromatic ring through a non-ring bond. Agrees with the
# JACS 2022 SI's own Buchwald-type labels on 109 of 112 ligands.
BUCHWALD = Chem.MolFromSmarts("[PX3]-[a;r5,r6]:[a]-!@[a]")


def is_buchwald(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return bool(mol is not None and mol.HasSubstructMatch(BUCHWALD))


def explain_mismatch(g: pd.DataFrame, pub: pd.Series, tol: float) -> str:
    """Search single-ligand removals and single label flips that reproduce the published row exactly."""
    def matches(x, a):
        r = fit_threshold(x, a)
        return (abs(r["threshold"] - pub["threshold"]) <= tol and round(r["accuracy"], 2) == pub["accuracy"]
                and round(r["f1"], 2) == pub["f1"] and round(r["mcc"], 2) == pub["mcc"])
    x, a = g["pct_vbur_min"].values, g["active"].astype(bool).values
    drops = [g["ligand"].iloc[i] for i in range(len(g)) if matches(np.delete(x, i), np.delete(a, i))]
    flips = [g["ligand"].iloc[i] for i in range(len(g)) if matches(x, np.where(np.arange(len(a)) == i, ~a, a))]
    if drops:
        return "reproduced exactly if one ligand is removed: " + " or ".join(drops)
    if flips:
        return "reproduced exactly if one ligand's label flips: " + " or ".join(flips)
    return "no single removal or label flip reproduces the published row"


def run() -> pd.DataFrame:
    cfg = load_config()
    gate = cfg["gate"]
    j = pd.read_csv(PROCESSED_DIR / "joined.csv")
    j = j[j["included"] & j["role"].isin(["primary", "supplementary"])]
    k25 = pd.read_csv(PROCESSED_DIR / "kraken_2025.csv.gz", usecols=["kraken_id", "smiles", "pct_vbur_min"])
    k25["buchwald_type"] = k25["smiles"].map(is_buchwald)
    j = j.merge(k25, left_on="id_2025", right_on="kraken_id", how="left")
    pub = pd.read_csv(INTERIM_DIR / "si_thresholds.csv")
    pub = pub[pub["si_table"] == "S8"].set_index("dataset")

    rows = []
    # The SI (p. S84) fits reaction VII's threshold with the Buchwald-type ligands excluded.
    subsets = [(ds, g, "all ligands") for ds, g in j.groupby("dataset")]
    vii = j[j["dataset"] == "cliffs_VII"]
    subsets.append(("cliffs_VII", vii[~vii["buchwald_type"]], "Buchwald-type excluded (SI p. S84)"))
    for ds, g, subset in subsets:
        active = g["active"].astype(bool).values
        kr = fit_threshold(g["pct_vbur_min"].values, active)
        si = fit_threshold(g["vbur_min"].values, active)
        p = pub.loc[ds]
        rows.append({
            "dataset": ds, "subset": subset, "role": g["role"].iloc[0], "n": len(g), "active": int(active.sum()),
            "threshold_published": p["threshold"], "threshold_kraken": round(kr["threshold"], 2),
            "threshold_si_values": round(si["threshold"], 2),
            "direction_published": p["direction"], "direction_kraken": kr["direction"],
            "mcc_published": p["mcc"], "mcc_kraken": round(kr["mcc"], 3), "mcc_si_values": round(si["mcc"], 3),
            "accuracy_published": p["accuracy"], "accuracy_kraken": round(kr["accuracy"], 3),
            "f1_published": p["f1"], "f1_kraken": round(kr["f1"], 3),
        })
    t = pd.DataFrame(rows)
    t["threshold_diff"] = (t["threshold_kraken"] - t["threshold_published"]).round(2)
    t["mcc_diff"] = (t["mcc_kraken"] - t["mcc_published"]).round(3)
    t["passes_gate"] = ((t["threshold_diff"].abs() <= gate["threshold_tol"])
                        & (t["mcc_diff"].abs() <= gate["mcc_tol"])
                        & (t["direction_kraken"] == t["direction_published"]))
    order = ["cliffs_" + r for r in ["I", "II", "III", "IV", "V", "VII", "VIII", "IX", "X", "XI", "XII", "RS1"]]
    t["_o"] = t["dataset"].map(order.index)
    t = t.sort_values(["_o", "subset"]).drop(columns="_o").reset_index(drop=True)
    t["explanation"] = ""
    for i, r in t[~t["passes_gate"] & (t["subset"] == "all ligands")].iterrows():
        g = j[j["dataset"] == r["dataset"]]
        t.loc[i, "explanation"] = explain_mismatch(g, pub.loc[r["dataset"]], gate["threshold_tol"])
    t.to_csv(RESULTS_DIR / "tables" / "baseline_reproduction.csv", index=False)

    show = ["dataset", "subset", "n", "threshold_published", "threshold_kraken", "threshold_si_values",
            "direction_kraken", "mcc_published", "mcc_kraken", "passes_gate"]
    print(t[show].to_string(index=False))
    print()
    for _, r in t[t["explanation"] != ""].iterrows():
        print(f"{r['dataset']}: {r['explanation']}")
    prim = t[(t["role"] == "primary") & (t["subset"] == "all ligands")]
    n_pass = int(prim["passes_gate"].sum())
    verdict = "REPRODUCES" if n_pass >= gate["min_datasets_passing"] else "DOES NOT REPRODUCE"
    print(f"\nprimary datasets within tolerance: {n_pass} of {len(prim)} -> baseline {verdict} "
          f"(needs {gate['min_datasets_passing']}; the gate decision is Ted's)")
    return t


if __name__ == "__main__":
    run()
