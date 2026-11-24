"""Step 9: join every SI ligand to its 2025 Kraken row, by structure, and verify it.

Each SI row carries a 2021 Kraken ID. The crosswalk (step 3) maps it to the 2025 row holding the same
structure; the SI's own %V_bur(min) and %V_bur(Boltz) must then agree with that row to 0.1. Rows that
fail either step are excluded with a reason, and the run stops if any dataset's match rate falls below
min_match_rate in config.yaml, because unmatched ligands are rarely a random sample.

Outputs
  data/processed/joined.csv            one row per (dataset, ligand) with labels and the 2025 ID
  results/tables/join_report.csv       per-dataset counts and match rate (committed)
  results/tables/join_exclusions.csv   every excluded row with its reason (committed, IDs only)
"""
from __future__ import annotations

import sys

import pandas as pd

from ligsel.paths import INTERIM_DIR, PROCESSED_DIR, RESULTS_DIR, load_config

TOL = 0.1  # %V_bur agreement between the SI and Kraken; the SI rounds to 0.1


def run() -> pd.DataFrame:
    config = load_config()
    ds_cfg = pd.DataFrame(config["datasets"]).T.rename_axis("dataset").reset_index()
    y = pd.read_csv(INTERIM_DIR / "si_yields.csv")
    cw = pd.read_csv(PROCESSED_DIR / "kraken_crosswalk.csv")
    k25 = pd.read_csv(PROCESSED_DIR / "kraken_2025.csv.gz",
                      usecols=["kraken_id", "canonical_smiles", "pct_vbur_min", "pct_vbur_boltz"])

    j = (y.merge(ds_cfg, on="dataset", how="left")
          .merge(cw[["id_2021", "id_2025", "status"]], left_on="kraken_id_si", right_on="id_2021", how="left")
          .merge(k25.rename(columns={"kraken_id": "id_2025"}), on="id_2025", how="left"))
    assert j["role"].notna().all(), "a dataset is missing from config.yaml"

    j["exclusion"] = ""
    j.loc[j["yield"].isna(), "exclusion"] = "no yield in SI"
    j.loc[(j["exclusion"] == "") & j["id_2021"].isna(), "exclusion"] = "SI ID not in Kraken 2021"
    j.loc[(j["exclusion"] == "") & j["id_2025"].isna(), "exclusion"] = "no unique 2025 structure match"
    off = ((j["vbur_min"] - j["pct_vbur_min"]).abs() > TOL) | ((j["vbur_boltz"] - j["pct_vbur_boltz"]).abs() > TOL)
    j.loc[(j["exclusion"] == "") & off, "exclusion"] = "SI %V_bur disagrees with Kraken"
    j["included"] = j["exclusion"] == ""
    j["active"] = (j["yield"] >= j["y_cut"].astype(float)).where(j["included"])
    j["id_2025"] = j["id_2025"].astype("Int64")

    j["has_yield"] = j["yield"].notna()
    rep = (j.groupby(["dataset", "role", "metal"])
            .agg(si_rows=("included", "size"), no_yield=("has_yield", lambda s: int((~s).sum())),
                 matched=("included", "sum"), active=("active", lambda s: int(s.sum())),
                 y_cut=("y_cut", "first"))
            .reset_index())
    rep["inactive"] = rep["matched"] - rep["active"]
    # match rate counts only rows the SI gives a yield for: a missing yield is not a failed match
    rep["match_rate"] = (rep["matched"] / (rep["si_rows"] - rep["no_yield"])).round(3)
    order = {"primary": 0, "supplementary": 1, "control": 2, "external": 3, "excluded": 4}
    rep = rep.sort_values(["role", "dataset"], key=lambda s: s.map(order) if s.name == "role" else s)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["dataset", "role", "metal", "si_table", "kraken_id_si", "id_2025", "ligand", "buchwald",
            "vbur_min", "vbur_boltz", "yield", "y_cut", "active", "included", "exclusion"]
    j[cols].to_csv(PROCESSED_DIR / "joined.csv", index=False)
    rep.to_csv(RESULTS_DIR / "tables" / "join_report.csv", index=False)
    j.loc[~j["included"], ["dataset", "kraken_id_si", "id_2025", "ligand", "exclusion"]].to_csv(
        RESULTS_DIR / "tables" / "join_exclusions.csv", index=False)

    print(rep.to_string(index=False))
    print(f"\nexcluded rows: {(~j['included']).sum()}  " + str(j.loc[~j['included'], 'exclusion'].value_counts().to_dict()))
    low = rep[(rep["role"] != "excluded") & (rep["match_rate"] < config["min_match_rate"])]
    if len(low):
        sys.exit(f"match rate below {config['min_match_rate']} for: {', '.join(low['dataset'])}")
    return j


if __name__ == "__main__":
    run()
