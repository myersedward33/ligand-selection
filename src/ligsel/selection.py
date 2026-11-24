"""Step 20a: which descriptors the models rely on.

1. Stability selection (pre-registration s. 8): 500 class-stratified bootstrap refits of the full
   elastic-net procedure ('all' view, inner-CV tuning included) per eligible primary dataset. A
   descriptor is 'selected' with a frequency above 0.80 and a consistent sign (>= 90% of its
   selections the same sign). Because many descriptors are near-twins, each descriptor's cluster
   (average |rho| >= 0.9, results/tables/descriptor_clusters.csv) is scored too: the share of refits
   that keep any member of the cluster.
2. Pair choices: how often each descriptor pair was chosen by best_pair across the leave-one-out
   folds of step 17.

Outputs: results/tables/stability_selection.csv, stability_clusters.csv, best_pair_choices.csv
"""
from __future__ import annotations

import collections
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ligsel.features import dropped_columns, feature_columns
from ligsel.models import SEED, elastic_net, load_dataset
from ligsel.paths import PROCESSED_DIR, RESULTS_DIR

warnings.filterwarnings("ignore")
ELIGIBLE = ["cliffs_I", "cliffs_II", "cliffs_III", "cliffs_IV", "cliffs_V", "cliffs_VII", "cliffs_VIII",
            "cliffs_IX", "cliffs_XI", "cliffs_XII"]
N_BOOT, CHUNK = 500, 100
CHECKPOINTS = PROCESSED_DIR / "stability_runs"


def _one(cols, X, y, seed):
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(y == c)[0], (y == c).sum(), replace=True) for c in (0, 1)])
    m = elastic_net(cols).fit(X.iloc[idx].reset_index(drop=True), y[idx])
    return m.model_.best_estimator_[-1].coef_[0]


def run(budget_s=None):
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    cols = feature_columns("all", dropped_columns())
    t0 = time.time()
    for ds in ELIGIBLE:
        X, y = load_dataset(ds)
        for c in range(N_BOOT // CHUNK):
            f = CHECKPOINTS / f"{ds}__{c}.npy"
            if f.exists():
                continue
            if budget_s and time.time() - t0 > budget_s:
                print("budget reached; rerun to continue"); return None
            t1 = time.time()
            coefs = Parallel(n_jobs=4)(delayed(_one)(cols, X, y, SEED + 1000 * c + i) for i in range(CHUNK))
            np.save(f, np.array(coefs))
            print(f"{ds:12s} bootstraps {c * CHUNK}-{c * CHUNK + CHUNK - 1}  {time.time() - t1:.0f}s", flush=True)
    summarise(cols)
    return True


def summarise(cols):
    clusters = pd.read_csv(RESULTS_DIR / "tables" / "descriptor_clusters.csv").set_index("column")
    fam = clusters["family"]
    rows, crow = [], []
    for ds in ELIGIBLE:
        B = np.concatenate([np.load(CHECKPOINTS / f"{ds}__{c}.npy") for c in range(N_BOOT // CHUNK)])
        nz = np.abs(B) > 1e-8
        freq = nz.mean(0)
        pos = np.where(nz.sum(0) > 0, (B > 0).sum(0) / np.maximum(nz.sum(0), 1), np.nan)
        sign_cons = np.maximum(pos, 1 - pos)
        for j, c in enumerate(cols):
            rows.append({"dataset": ds, "descriptor": c, "family": fam.get(c), "cluster": clusters.loc[c, "cluster"],
                         "frequency": freq[j], "share_positive": pos[j],
                         "stable": bool(freq[j] > 0.80 and sign_cons[j] >= 0.90)})
        cl = clusters.loc[cols, "cluster"].values
        for k in np.unique(cl):
            members = [c for c, kk in zip(cols, cl) if kk == k]
            crow.append({"dataset": ds, "cluster": k, "members": " ".join(members),
                         "family": "/".join(sorted(set(fam.get(m) for m in members))),
                         "frequency_any_member": nz[:, cl == k].any(1).mean()})
    t = pd.DataFrame(rows)
    t.round(3).to_csv(RESULTS_DIR / "tables" / "stability_selection.csv", index=False)
    c = pd.DataFrame(crow)
    c.round(3).to_csv(RESULTS_DIR / "tables" / "stability_clusters.csv", index=False)
    stable = t[t["stable"]].sort_values(["dataset", "frequency"], ascending=[True, False])
    print(stable[["dataset", "descriptor", "family", "frequency", "share_positive"]].to_string(index=False))


def pair_choices():
    sel = json.load(open(PROCESSED_DIR / "ladder_selections.json"))
    rows = []
    for s in sel:
        if s["rung"] != "best_pair" or s["view"] != "all":
            continue
        cnt = collections.Counter(tuple(sorted(p)) for p in s["loo_selections"])
        n = sum(cnt.values())
        for pair, k in cnt.most_common(3):
            rows.append({"dataset": s["dataset"], "pair": " + ".join(pair), "share_of_folds": k / n})
    t = pd.DataFrame(rows)
    t.round(3).to_csv(RESULTS_DIR / "tables" / "best_pair_choices.csv", index=False)
    print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pairs":
        pair_choices()
    else:
        run(float(sys.argv[1]) if len(sys.argv) > 1 else None)
