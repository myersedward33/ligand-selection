"""Five feature views and checks behind them:

The 28 properties Kraken summarizes five ways are the only place the ensemble statistic can vary,
so the views differ only there; the 50 Boltzmann-only properties enter every view unchanged:

  boltz        28 Boltzmann averages            + 50 Boltzmann-only  (the population view)
  min          28 minima                        + 50                 (the most accommodating conformer)
  max          28 maxima                        + 50
  vburminconf  28 values in the min-V_bur conformer + 50             (the binding-conformer view)
  all          all five summaries of the 28     + 50

Columns dropped everywhere, decided before modelling:
  * vbur_vbur_vburminconf: identical to vbur_vbur_min by construction (the min-V_bur conformer's V_bur)
  * columns where one value covers more than 95% of the 1,614 ligands (no information)
"""
from __future__ import annotations

import pandas as pd

from ligsel.paths import PROCESSED_DIR, RESULTS_DIR

VIEWS = ("boltz", "min", "max", "vburminconf", "all")
NEAR_CONSTANT_SHARE = 0.95


def _families() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "tables" / "descriptor_families.csv")


def dropped_columns(kraken: pd.DataFrame | None = None) -> dict[str, str]:
    kraken = kraken if kraken is not None else pd.read_csv(PROCESSED_DIR / "kraken_2025.csv.gz")
    cols = _families()["column"]
    drop = {"vbur_vbur_vburminconf": "identical to vbur_vbur_min"}
    for c in cols:
        share = kraken[c].value_counts(normalize=True).iloc[0]
        if share > NEAR_CONSTANT_SHARE:
            drop[c] = f"near-constant: one value on {share:.0%} of ligands"
    return drop


def feature_columns(view: str, drop: dict | None = None) -> list[str]:
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}")
    fam = _families()
    drop = drop if drop is not None else dropped_columns()
    five = fam["summaries_available"] == 5
    keep = fam["summary"] == "boltz" if view == "boltz" else fam["summary"].eq(view)
    if view == "all":
        keep = five
    cols = fam.loc[(five & keep) | ~five, "column"]
    return [c for c in cols if c not in drop]


def single_conformer_check(kraken: pd.DataFrame) -> pd.DataFrame:
    """Ligands whose every delta is zero have one conformer: min = max = Boltzmann average."""
    deltas = [c for c in kraken.columns if c.endswith("_delta")]
    single = (kraken[deltas].abs() < 1e-9).all(axis=1)
    rows = []
    for label, mask in (("single conformer", single), ("several conformers", ~single)):
        sub = kraken[mask]
        rows.append({"group": label, "n": int(mask.sum()),
                     "median_volume": sub["volume_boltz"].median(),
                     "median_pct_vbur_boltz": sub["pct_vbur_boltz"].median()})
    return pd.DataFrame(rows)


def run() -> None:
    k = pd.read_csv(PROCESSED_DIR / "kraken_2025.csv.gz")
    drop = dropped_columns(k)
    views = pd.DataFrame([{"view": v, "n_features": len(feature_columns(v, drop))} for v in VIEWS])
    pd.DataFrame([{"column": c, "reason": r} for c, r in drop.items()]).to_csv(
        RESULTS_DIR / "tables" / "dropped_descriptors.csv", index=False)
    views.to_csv(RESULTS_DIR / "tables" / "feature_views.csv", index=False)
    sc = single_conformer_check(k)
    sc.round(2).to_csv(RESULTS_DIR / "tables" / "single_conformer_check.csv", index=False)
    print(views.to_string(index=False)); print(); print(sc.round(2).to_string(index=False))
    print(f"\ndropped: {list(drop)}")


if __name__ == "__main__":
    run()
