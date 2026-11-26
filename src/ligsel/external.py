"""Score the frozen models ONCE on JACS 2022 Reaction III (H5):

Refuses to run unless the `frozen` tag exists, this file and models/frozen_ni_pool.json are unchanged
since that tag, the data files match the checksums frozen with the model, and the result has not
already been written.

Scored: the frozen elastic net, the %V_bur(min) threshold refitted on the Ni pool, and the published
threshold (31.89, as JACS 2022 Table S4 applied it to Reaction III). Subsets: all 43 ligands (primary),
the ligands not in the Ni pool, and the Buchwald-type ligands. MCC with a bootstrap 95% interval over
ligands; differences between models with a paired bootstrap interval. H5 is descriptive.

Outputs (committed): results/tables/external_test.csv, results/tables/external_test_differences.csv
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, matthews_corrcoef

from ligsel.acquire import sha256
from ligsel.baseline import is_buchwald
from ligsel.freeze import FROZEN
from ligsel.paths import INTERIM_DIR, PROCESSED_DIR, REPO_ROOT, RESULTS_DIR

OUT = RESULTS_DIR / "tables" / "external_test.csv"
OUT_DIFF = RESULTS_DIR / "tables" / "external_test_differences.csv"
N_BOOT = 2000


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()


def guard(spec: dict) -> None:
    if _git("tag", "--list", "frozen") != "frozen":
        sys.exit("no `frozen` tag: the external test runs only after step 22")
    changed = _git("diff", "--name-only", "frozen", "--", "models/frozen_ni_pool.json", "src/ligsel/external.py",
                   "src/ligsel/freeze.py")
    if changed:
        sys.exit(f"changed since the `frozen` tag, refusing to score: {changed}")
    for name, digest in spec["inputs_sha256"].items():
        path = next(p for p in (PROCESSED_DIR / name, INTERIM_DIR / name) if p.exists())
        if sha256(path) != digest:
            sys.exit(f"{name} differs from the copy the model was frozen with")
    if OUT.exists():
        sys.exit(f"{OUT.relative_to(REPO_ROOT)} exists: the external test has been scored once already")


def _threshold(x, t, direction):
    return ((x <= t) if direction == "Left" else (x > t)).astype(int)


def _boot(y, preds: dict, rng) -> dict:
    idx = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
    return {m: np.array([matthews_corrcoef(y[i], p[i]) for i in idx]) for m, p in preds.items()}


def run() -> pd.DataFrame:
    spec = json.loads(FROZEN.read_text())
    guard(spec)
    j = pd.read_csv(PROCESSED_DIR / "joined.csv")
    j = j[(j["dataset"] == spec["external"]["dataset"]) & j["included"]]
    k = pd.read_csv(PROCESSED_DIR / "kraken_2025.csv.gz").set_index("kraken_id")
    X = k.loc[j["id_2025"].astype(int).values].reset_index()
    y = j["active"].astype(bool).astype(int).values
    assert len(y) == spec["external"]["expected_n"], f"expected {spec['external']['expected_n']} ligands, found {len(y)}"

    en = spec["elastic_net"]
    z = (X[en["columns"]].values - np.array(en["scaler_mean"])) / np.array(en["scaler_scale"])
    tn, tp = spec["threshold_ni_pool"], spec["threshold_published"]
    preds = {
        "elastic_net_frozen": (z @ np.array(en["coef"]) + en["intercept"] > 0).astype(int),
        "threshold_ni_pool": _threshold(X["pct_vbur_min"].values, tn["threshold"], tn["direction"]),
        "threshold_published": _threshold(X["pct_vbur_min"].values, tp["threshold"], tp["direction"]),
    }
    subsets = {
        "all": np.ones(len(y), bool),
        "not_in_ni_pool": ~X["kraken_id"].isin(spec["training"]["ligand_ids"]).values,
        "buchwald_type": X["smiles"].map(is_buchwald).values,
    }
    rng = np.random.default_rng(spec["seed"])
    rows, diffs = [], []
    for sname, mask in subsets.items():
        ys = y[mask]
        ps = {m: p[mask] for m, p in preds.items()}
        boot = _boot(ys, ps, rng)
        for m, p in ps.items():
            cm = confusion_matrix(ys, p, labels=[0, 1]).ravel()
            lo, hi = np.percentile(boot[m], [2.5, 97.5])
            rows.append({"subset": sname, "model": m, "n": int(mask.sum()), "active": int(ys.sum()),
                         "tn": cm[0], "fp": cm[1], "fn": cm[2], "tp": cm[3],
                         "mcc": round(matthews_corrcoef(ys, p), 4), "mcc_lo": round(lo, 4), "mcc_hi": round(hi, 4),
                         "balanced_accuracy": round(balanced_accuracy_score(ys, p), 4) if 0 < ys.sum() < len(ys) else None})
        for a, b in (("elastic_net_frozen", "threshold_published"), ("elastic_net_frozen", "threshold_ni_pool"),
                     ("threshold_ni_pool", "threshold_published")):
            d = boot[a] - boot[b]
            lo, hi = np.percentile(d, [2.5, 97.5])
            diffs.append({"subset": sname, "comparison": f"{a} - {b}",
                          "mcc_difference": round(matthews_corrcoef(ys, ps[a]) - matthews_corrcoef(ys, ps[b]), 4),
                          "lo": round(lo, 4), "hi": round(hi, 4), "interval_excludes_zero": bool(lo > 0 or hi < 0)})
    t, d = pd.DataFrame(rows), pd.DataFrame(diffs)
    t.to_csv(OUT, index=False)
    d.to_csv(OUT_DIFF, index=False)

    # label check: the published threshold should reproduce JACS 2022 Table S4's confusion matrix
    s4 = pd.read_csv(INTERIM_DIR / "si_thresholds.csv")
    s4 = s4[(s4["dataset"] == "j22_III") & (s4["subset"] == "All")].iloc[0]
    ours = t[(t["subset"] == "all") & (t["model"] == "threshold_published")].iloc[0]
    same = all(int(ours[c]) == int(s4[c]) for c in ("tp", "tn", "fp", "fn"))
    print(t.to_string(index=False), "\n")
    print(d.to_string(index=False), "\n")
    print("published threshold reproduces JACS 2022 Table S4 (tp/tn/fp/fn):", "yes" if same else
          f"NO: ours {[int(ours[c]) for c in ('tp', 'tn', 'fp', 'fn')]}, S4 {[int(s4[c]) for c in ('tp', 'tn', 'fp', 'fn')]}")
    return t


if __name__ == "__main__":
    run()
