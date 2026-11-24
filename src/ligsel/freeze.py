"""Step 22: fit the model for the external test and freeze it (pre-registration section 8a).

Trains on the Ni pool (cliffs I-V), every included ligand labelled at its own dataset's cutoff; a
ligand tested in several reactions contributes one row per reaction. The elastic net is the
ladder's (same pipeline, same grid, all 185 descriptors). Its settings are chosen by 5-fold CV
grouped by ligand, so no ligand sits on both sides of an inner split, then it is refitted on the
whole pool. A %V_bur(min) threshold refitted on the same pool is frozen as the like-for-like
comparator. The external set (role 'external') is not read here.

Output: models/frozen_ni_pool.json (committed): every number needed to make a prediction, as plain
JSON rather than a pickle, plus checksums of the inputs and the package versions.
"""
from __future__ import annotations

import datetime as dt
import json
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold

from ligsel.acquire import sha256
from ligsel.baseline import fit_threshold
from ligsel.features import dropped_columns, feature_columns
from ligsel.models import SEED, elastic_net
from ligsel.paths import INTERIM_DIR, PROCESSED_DIR, REPO_ROOT, load_config
from ligsel.splits import NI, pool

FROZEN = REPO_ROOT / "models" / "frozen_ni_pool.json"
PUBLISHED = {"descriptor": "pct_vbur_min", "threshold": 31.89, "direction": "Left",
             "rule": "active if %V_bur(min) <= 31.89", "source": "JACS 2022 SI Table S4, Reaction III row"}
INPUTS = [PROCESSED_DIR / "joined.csv", PROCESSED_DIR / "kraken_2025.csv.gz", INTERIM_DIR / "si_yields.csv"]


def run() -> dict:
    cfg = load_config()
    cols = feature_columns("all", dropped_columns())
    X, y, meta = pool(NI)
    gs = elastic_net(cols).estimator  # the ladder's GridSearchCV: same pipeline, same grid
    gs.set_params(cv=StratifiedGroupKFold(5, shuffle=True, random_state=SEED))
    gs.fit(X[cols].values, y, groups=meta["kraken_id"].values)
    pipe = gs.best_estimator_
    scaler, lr = pipe[0], pipe[-1]
    coef, intercept = lr.coef_[0], float(lr.intercept_[0])

    # the JSON form must reproduce the fitted pipeline exactly
    z = (X[cols].values - scaler.mean_) / scaler.scale_
    assert ((z @ coef + intercept > 0).astype(int) == pipe.predict(X[cols].values)).all()

    stump = fit_threshold(X["pct_vbur_min"].values, y.astype(bool))
    spec = {
        "spec": "docs/preregistration.md, section 8a",
        "frozen_on": dt.date.today().isoformat(),
        "seed": SEED,
        "training": {
            "datasets": NI, "rows": int(len(y)), "active": int(y.sum()),
            "unique_ligands": int(meta["kraken_id"].nunique()),
            "y_cuts": {d: float(cfg["datasets"][d]["y_cut"]) for d in NI},
            "ligand_ids": sorted(int(i) for i in meta["kraken_id"].unique()),
        },
        "elastic_net": {
            "view": "all", "columns": cols,
            "C": float(lr.C), "l1_ratio": float(lr.l1_ratio),
            "inner_cv": "StratifiedGroupKFold(5, shuffle, seed), grouped by ligand",
            "inner_cv_mcc": float(gs.best_score_),
            "training_mcc_in_sample": float(matthews_corrcoef(y, pipe.predict(X[cols].values))),
            "n_nonzero": int((np.abs(coef) > 1e-8).sum()),
            "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
            "coef": coef.tolist(), "intercept": intercept,
            "rule": "active if intercept + coef . ((x - scaler_mean) / scaler_scale) > 0",
        },
        "threshold_ni_pool": {"descriptor": "pct_vbur_min", "threshold": stump["threshold"],
                              "direction": stump["direction"], "training_mcc_in_sample": stump["mcc"]},
        "threshold_published": PUBLISHED,
        "external": {"dataset": "j22_III", "y_cut": float(cfg["datasets"]["j22_III"]["y_cut"]), "expected_n": 43},
        "inputs_sha256": {p.name: sha256(p) for p in INPUTS},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
    }
    FROZEN.parent.mkdir(exist_ok=True)
    FROZEN.write_text(json.dumps(spec, indent=1))
    en = spec["elastic_net"]
    print(f"Ni pool: {len(y)} rows, {spec['training']['unique_ligands']} ligands, {int(y.sum())} active")
    print(f"elastic net: C={en['C']}, l1_ratio={en['l1_ratio']}, {en['n_nonzero']} of {len(cols)} descriptors kept, "
          f"grouped inner-CV MCC {en['inner_cv_mcc']:.3f}")
    print(f"Ni-pool threshold: {stump['threshold']:.2f} %V_bur(min), {stump['direction']}  (published: 31.89, Left)")
    print(f"written {FROZEN.relative_to(REPO_ROOT)}")
    return spec


if __name__ == "__main__":
    run()
