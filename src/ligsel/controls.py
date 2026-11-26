"""Negative controls and hypothesis H4:

Controls (no published threshold): cliffs RS2, RS3, XIII, XIV and JACS 2022 IV. None meets the
pre-registered size rule for multivariate fits (n >= 30 and >= 10 in the smaller class), so, as in
step 17, they get the univariate and fixed-pair rungs: baseline_cv, best_single, single_min_lr and
pair_min_boltz.

H4: no model beats a label-permutation null (500 permutations, p < 0.05). Each permutation reruns the
whole procedure, threshold or descriptor choice included, under stratified 5-fold CV with a fixed
seed (leave-one-out inside 500 permutations is too slow for best_single; LOO is still reported for
the real labels). p = (1 + number of permutations scoring >= observed) / (1 + 500).
"""
from __future__ import annotations

import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

from ligsel.features import dropped_columns, feature_columns
from ligsel.models import SEED, VBOLTZ, VMIN, PairLogistic, Stump, load_dataset
from ligsel.paths import PROCESSED_DIR, RESULTS_DIR

warnings.filterwarnings("ignore")
CONTROLS = ["cliffs_RS2", "cliffs_RS3", "cliffs_XIII", "cliffs_XIV", "j22_IV"]
N_PERM = 500
CHECKPOINTS = PROCESSED_DIR / "control_runs"


class FastBestStump(Stump):
    """best_single with the stump search vectorised over all candidate columns at once.

    Same rule as Stump: for each column the class-balanced Gini-optimal split (as a depth-1
    DecisionTreeClassifier finds it, threshold midway between neighbouring values), then the column
    whose split has the highest training MCC. Checked against Stump in tests/test_pipeline.py.
    """
    def fit(self, X, y):
        cols = self.candidates
        V = X[cols].values.astype(float)
        n = len(y)
        w = np.where(y == 1, n / (2 * y.sum()), n / (2 * (n - y.sum())))
        order = np.argsort(V, axis=0, kind="mergesort")
        Vs = np.take_along_axis(V, order, 0)
        w1 = np.take_along_axis((w * (y == 1))[:, None].repeat(V.shape[1], 1), order, 0)
        w0 = np.take_along_axis((w * (y == 0))[:, None].repeat(V.shape[1], 1), order, 0)
        L1, L0 = np.cumsum(w1, 0)[:-1], np.cumsum(w0, 0)[:-1]
        T1, T0 = w1.sum(0), w0.sum(0)
        R1, R0 = T1 - L1, T0 - L0
        WL, WR = L1 + L0, R1 + R0
        gini = lambda a, b, t: 1 - (a / t) ** 2 - (b / t) ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            imp = WL * gini(L1, L0, WL) + WR * gini(R1, R0, WR)
        imp[Vs[1:] <= Vs[:-1] + 1e-12] = np.inf          # no split between equal values
        k = np.argmin(imp, axis=0)
        j = np.arange(V.shape[1])
        thr = (Vs[k, j] + Vs[k + 1, j]) / 2
        left_pos = L1[k, j] > L0[k, j]
        pred = np.where(V <= thr, left_pos, ~left_pos).astype(int)
        tp = ((pred == 1) & (y[:, None] == 1)).sum(0); tn = ((pred == 0) & (y[:, None] == 0)).sum(0)
        fp = ((pred == 1) & (y[:, None] == 0)).sum(0); fn = ((pred == 0) & (y[:, None] == 1)).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.nan_to_num(mcc, nan=0.0)
        mcc[~np.isfinite(thr)] = -2
        b = int(np.argmax(mcc))
        self.selected_, self.thr_, self.left_pos_ = cols[b], thr[b], bool(left_pos[b])
        return self

    def predict(self, X):
        below = X[self.selected_].values <= self.thr_
        return np.where(below, self.left_pos_, not self.left_pos_).astype(int)


def rungs():
    cols = feature_columns("all", dropped_columns())
    return {"baseline_cv": Stump(column=VMIN), "best_single": FastBestStump(candidates=cols),
            "single_min_lr": PairLogistic(pair=(VMIN, VMIN)), "pair_min_boltz": PairLogistic(pair=(VMIN, VBOLTZ))}


def cv_mcc(model, X, y, splitter):
    pred = np.full(len(y), -1)
    for tr, te in splitter.split(X, y):
        m = clone(model).fit(X.iloc[tr].reset_index(drop=True), y[tr])
        pred[te] = m.predict(X.iloc[te].reset_index(drop=True))
    return matthews_corrcoef(y, pred)


def _perm(model, X, y, seed):
    yp = np.random.default_rng(seed).permutation(y)
    return cv_mcc(model, X, yp, StratifiedKFold(5, shuffle=True, random_state=SEED))


def run(budget_s=None):
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    todo = [(ds, r, m) for ds in CONTROLS for r, m in rungs().items()
            if not (CHECKPOINTS / f"{ds}__{r}.json").exists()]
    for ds, rung, model in todo:
        if budget_s and time.time() - t0 > budget_s:
            break
        t1 = time.time()
        X, y = load_dataset(ds)
        obs5 = cv_mcc(model, X, y, StratifiedKFold(5, shuffle=True, random_state=SEED))
        obs_loo = cv_mcc(model, X, y, LeaveOneOut())
        null = Parallel(n_jobs=4)(delayed(_perm)(model, X, y, SEED + i) for i in range(N_PERM))
        p = (1 + sum(v >= obs5 for v in null)) / (1 + N_PERM)
        rec = {"dataset": ds, "rung": rung, "n": len(y), "active": int(y.sum()), "mcc_loo": obs_loo,
               "mcc_5fold": obs5, "null_mean": float(np.mean(null)), "null_q95": float(np.quantile(null, 0.95)),
               "p_value": p, "beats_null": p < 0.05, "seconds": round(time.time() - t1, 1)}
        (CHECKPOINTS / f"{ds}__{rung}.json").write_text(json.dumps(rec))
        print(f"{ds:12s} {rung:15s} 5-fold {obs5:.3f}  null95 {rec['null_q95']:.3f}  p {p:.3f}  {rec['seconds']}s", flush=True)
    left = sum(1 for ds, r, _ in todo if not (CHECKPOINTS / f"{ds}__{r}.json").exists())
    if left:
        print(f"{left} runs still to do")
        return None
    tab = pd.DataFrame([json.loads(f.read_text()) for f in sorted(CHECKPOINTS.glob("*.json"))])
    tab.round(4).to_csv(RESULTS_DIR / "tables" / "negative_controls_h4.csv", index=False)
    print(f"complete; H4 holds (no model beats its null): {not tab['beats_null'].any()}")
    return tab


if __name__ == "__main__":
    run(float(sys.argv[1]) if len(sys.argv) > 1 else None)
