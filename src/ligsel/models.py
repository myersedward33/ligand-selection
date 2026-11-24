"""Step 17: the model ladder, per dataset, under the pre-registered validation (docs/preregistration.md).

Rungs (every one reported):
  baseline_cv     depth-1 tree on %V_bur(min), threshold refitted inside each training fold
  best_single     depth-1 tree on whichever single descriptor fits the training fold best
  best_pair       logistic regression on whichever descriptor pair fits the training fold best
                  (all ~17,000 pairs searched inside every fold; search size reported)
  pair_min_boltz  logistic regression on %V_bur(min) + %V_bur(Boltz): hypothesis H2, fixed in advance
  single_min_lr   logistic regression on %V_bur(min) alone: H2's comparator
  elastic_net     elastic-net logistic regression, hyperparameters by inner 5-fold CV (primary model)
  random_forest   nonlinear reference, inner-CV tuned ('all' view only)
  gradient_boost  nonlinear reference, inner-CV tuned ('all' view only)

Feature views (ligsel.features) apply to best_single, best_pair and elastic_net. H1 uses the 'all'
view, the full descriptor set the research question names. Every rung is fitted with class-balanced
weights and scaled inside the training fold only.

Validation: leave-one-ligand-out (primary) and 5 x repeated stratified 5-fold (reported). MCC is
computed on the pooled out-of-fold predictions; LOO gets a bootstrap 95% interval over ligands.

The external test set is never loaded: load_dataset refuses role 'external' (pre-registration s. 8).
"""
from __future__ import annotations

import itertools
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, make_scorer, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, LeaveOneOut, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ligsel.features import VIEWS, dropped_columns, feature_columns
from ligsel.paths import PROCESSED_DIR, RESULTS_DIR, load_config

warnings.filterwarnings("ignore")
SEED = 20261125
MCC = make_scorer(matthews_corrcoef)
VMIN, VBOLTZ = "vbur_vbur_min", "vbur_vbur_boltz"


# ---------------------------------------------------------------- data
def load_dataset(name: str) -> tuple[pd.DataFrame, np.ndarray]:
    cfg = load_config()["datasets"][name]
    if cfg["role"] == "external":
        raise PermissionError("the external test set is not loaded before the `frozen` tag")
    j = pd.read_csv(PROCESSED_DIR / "joined.csv")
    j = j[(j["dataset"] == name) & j["included"]]
    k = pd.read_csv(PROCESSED_DIR / "kraken_2025.csv.gz").set_index("kraken_id")
    X = k.loc[j["id_2025"].astype(int).values].reset_index(drop=True)
    return X, j["active"].astype(bool).astype(int).values


# ---------------------------------------------------------------- rungs
def _balanced_weights(y):
    w = np.ones(len(y), float)
    for c in (0, 1):
        w[y == c] = len(y) / (2 * max((y == c).sum(), 1))
    return w


class Stump(BaseEstimator, ClassifierMixin):
    """Depth-1 tree on one named column, or on the best column of `candidates` (training MCC)."""
    def __init__(self, column=VMIN, candidates=None):
        self.column, self.candidates = column, candidates

    def fit(self, X, y):
        cols = self.candidates or [self.column]
        best = (-2, None, None)
        for c in cols:
            t = DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=0)
            t.fit(X[[c]].values, y)
            score = matthews_corrcoef(y, t.predict(X[[c]].values))
            if score > best[0]:
                best = (score, c, t)
        self.selected_, self.tree_ = best[1], best[2]
        return self

    def predict(self, X):
        return self.tree_.predict(X[[self.selected_]].values)


def batched_logistic(Z, y, w, lam=1e-2, iters=20):
    """Logistic regression for many 2-feature designs at once. Z: (P, n, 2) standardised."""
    P, n, _ = Z.shape
    A = np.concatenate([np.ones((P, n, 1)), Z], axis=2)                  # (P, n, 3)
    beta = np.zeros((P, 3))
    R = np.diag([0.0, lam, lam])
    for _ in range(iters):
        eta = np.einsum("pnk,pk->pn", A, beta)
        p = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
        g = np.einsum("pnk,pn->pk", A, w * (p - y)) + beta @ R
        W = w * p * (1 - p)
        H = np.einsum("pnk,pn,pnl->pkl", A, W, A) + R
        beta -= np.linalg.solve(H, g[..., None])[..., 0]
    eta = np.einsum("pnk,pk->pn", A, beta)
    p = np.clip(1 / (1 + np.exp(-np.clip(eta, -30, 30))), 1e-9, 1 - 1e-9)
    loss = -(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum(axis=1) / w.sum()
    return beta, loss


class PairLogistic(BaseEstimator, ClassifierMixin):
    """Logistic regression on a fixed pair, or on the best pair of `candidates` (training log-loss)."""
    def __init__(self, pair=(VMIN, VBOLTZ), candidates=None):
        self.pair, self.candidates = pair, candidates

    def fit(self, X, y):
        cols = list(self.candidates) if self.candidates else list(self.pair)
        mu, sd = X[cols].mean(), X[cols].std(ddof=0).replace(0, 1)
        Zall = ((X[cols] - mu) / sd).values
        idx = list(itertools.combinations(range(len(cols)), 2)) if self.candidates else [(0, 1)]
        w = _balanced_weights(y)
        best_loss, best = np.inf, None
        for start in range(0, len(idx), 4000):
            chunk = np.array(idx[start:start + 4000])
            Z = np.stack([Zall[:, chunk[:, 0]].T, Zall[:, chunk[:, 1]].T], axis=2)
            beta, loss = batched_logistic(Z, y.astype(float), w)
            i = int(np.argmin(loss))
            if loss[i] < best_loss:
                best_loss, best = loss[i], (chunk[i], beta[i])
        (a, b), self.beta_ = best
        self.selected_ = (cols[a], cols[b])
        self.mu_, self.sd_ = mu.values[[a, b]], sd.values[[a, b]]  # positional: the pair may repeat a column
        self.n_searched_ = len(idx)
        return self

    def predict(self, X):
        Z = (np.column_stack([X[c].values for c in self.selected_]) - self.mu_) / self.sd_
        return (self.beta_[0] + Z @ self.beta_[1:] > 0).astype(int)


class ColumnModel(BaseEstimator, ClassifierMixin):
    """Fit an sklearn estimator on a fixed column list of the descriptor frame."""
    def __init__(self, estimator=None, columns=None):
        self.estimator, self.columns = estimator, columns

    def fit(self, X, y):
        self.model_ = clone(self.estimator).fit(X[self.columns].values, y)
        best = getattr(self.model_, "best_estimator_", self.model_)
        lr = best[-1] if hasattr(best, "steps") else best
        coef = getattr(lr, "coef_", None)
        self.selected_ = [c for c, v in zip(self.columns, coef[0]) if abs(v) > 1e-8] if coef is not None else None
        self.best_params_ = getattr(self.model_, "best_params_", None)
        return self

    def predict(self, X):
        return self.model_.predict(X[self.columns].values)


def inner_cv():
    return StratifiedKFold(5, shuffle=True, random_state=SEED)


def elastic_net(columns):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(
        penalty="elasticnet", solver="saga", class_weight="balanced", max_iter=3000, tol=1e-3,
        random_state=SEED))
    grid = {"logisticregression__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
            "logisticregression__l1_ratio": [0.3, 0.7, 1.0]}
    return ColumnModel(GridSearchCV(pipe, grid, scoring=MCC, cv=inner_cv(), n_jobs=1), columns)


def random_forest(columns):
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=1)
    return ColumnModel(GridSearchCV(rf, {"min_samples_leaf": [1, 3, 5]}, scoring=MCC, cv=inner_cv()), columns)


def gradient_boost(columns):
    gb = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED, max_iter=200)
    grid = {"learning_rate": [0.05, 0.15], "max_depth": [2, 3]}
    return ColumnModel(GridSearchCV(gb, grid, scoring=MCC, cv=inner_cv()), columns)


def ladder(view: str, drop: dict) -> dict:
    cols = feature_columns(view, drop)
    rungs = {"best_single": Stump(candidates=cols), "best_pair": PairLogistic(candidates=cols),
             "elastic_net": elastic_net(cols)}
    if view == "all":
        rungs = {"baseline_cv": Stump(column=VMIN), "single_min_lr": PairLogistic(pair=(VMIN, VMIN)),  # one feature, entered twice: same fit
                 "pair_min_boltz": PairLogistic(pair=(VMIN, VBOLTZ)), **rungs,
                 "random_forest": random_forest(cols), "gradient_boost": gradient_boost(cols)}
    return rungs


# ---------------------------------------------------------------- validation
def _fit_predict(model, X, y, tr, te):
    m = clone(model).fit(X.iloc[tr].reset_index(drop=True), y[tr])
    sel = getattr(m, "selected_", None)
    sel = list(sel) if isinstance(sel, (list, tuple)) else ([sel] if sel else [])
    return te, m.predict(X.iloc[te].reset_index(drop=True)), sel


def bootstrap_ci(y, pred, n=1000):
    rng = np.random.default_rng(SEED)
    vals = [matthews_corrcoef(y[i], pred[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return np.percentile(vals, [2.5, 97.5])


def run_one(dataset, view, rung, model, X, y, n_jobs=4):
    """LOO and 5x repeated 5-fold for one model; the folds run in parallel."""
    t0 = time.time()
    loo = list(LeaveOneOut().split(X, y))
    rep = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(X, y))
    res = Parallel(n_jobs=n_jobs)(delayed(_fit_predict)(model, X, y, tr, te) for tr, te in loo + rep)
    pred = np.full(len(y), -1)
    for te, p, _ in res[:len(loo)]:
        pred[te] = p
    reps = []
    for r in range(5):
        p = np.full(len(y), -1)
        for te, pr, _ in res[len(loo) + 5 * r: len(loo) + 5 * r + 5]:
            p[te] = pr
        reps.append(matthews_corrcoef(y, p))
    lo, hi = bootstrap_ci(y, pred)
    out = {"dataset": dataset, "view": view, "rung": rung, "n": len(y), "active": int(y.sum()),
           "mcc_loo": matthews_corrcoef(y, pred), "mcc_loo_lo": lo, "mcc_loo_hi": hi,
           "balacc_loo": balanced_accuracy_score(y, pred),
           "mcc_rep5x5_mean": float(np.mean(reps)), "mcc_rep5x5_sd": float(np.std(reps, ddof=1)),
           "seconds": round(time.time() - t0, 1)}
    if isinstance(model, PairLogistic) and model.candidates:
        out["pairs_searched"] = len(model.candidates) * (len(model.candidates) - 1) // 2
    return out, {"dataset": dataset, "view": view, "rung": rung, "loo_selections": [r[2] for r in res[:len(loo)]]}


def jobs(datasets=None):
    cfg = load_config()
    datasets = datasets or [d for d, c in cfg["datasets"].items() if c["role"] == "primary"]
    drop = dropped_columns()
    out = []
    for ds in datasets:
        X, y = load_dataset(ds)
        multivariate = len(y) >= 30 and min(y.sum(), len(y) - y.sum()) >= 10  # pre-registration s. 6
        for view in VIEWS:
            for rung, model in ladder(view, drop).items():
                if multivariate or rung in ("baseline_cv", "best_single", "single_min_lr", "pair_min_boltz"):
                    out.append((ds, view, rung, model, X, y))
    return out


CHECKPOINTS = PROCESSED_DIR / "ladder_runs"


def run(datasets=None, budget_s: float | None = None) -> pd.DataFrame | None:
    """Run every job not yet checkpointed. With budget_s, stop starting new jobs after that long,
    so the work can be spread over several sessions; rerun until it reports 'complete'."""
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    t0, todo = time.time(), []
    for j in jobs(datasets):
        f = CHECKPOINTS / f"{j[0]}__{j[1]}__{j[2]}.json"
        if not f.exists():
            todo.append((j, f))
    for j, f in todo:
        if budget_s and time.time() - t0 > budget_s:
            break
        out, sel = run_one(*j)
        f.write_text(json.dumps({"result": out, "selections": sel}, default=str))
        print(f"{out['dataset']:12s} {out['view']:12s} {out['rung']:15s} mcc_loo {out['mcc_loo']:.3f}  {out['seconds']}s", flush=True)
    left = sum(1 for _, f in todo if not f.exists())
    if left:
        print(f"{left} runs still to do", flush=True)
        return None
    recs = [json.loads(f.read_text()) for f in sorted(CHECKPOINTS.glob("*.json"))]
    table = pd.DataFrame([r["result"] for r in recs])
    table.round(4).to_csv(RESULTS_DIR / "tables" / "ladder_per_dataset.csv", index=False)
    with open(PROCESSED_DIR / "ladder_selections.json", "w") as fh:
        json.dump([r["selections"] for r in recs], fh, default=str)
    print(f"complete: {len(table)} runs written to results/tables/ladder_per_dataset.csv")
    return table


if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else None
    run(budget_s=budget)
