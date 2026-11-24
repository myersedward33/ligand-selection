"""Step 3: parse Kraken, convert buried volume to percent, tag columns, build the ID crosswalk.

Outputs
  data/processed/kraken_2025.csv.gz        2025 descriptors + identifiers + %V_bur columns
  data/processed/kraken_2021.csv.gz        2021 descriptors + identifiers (the SI's ID key)
  data/processed/kraken_crosswalk.csv      every 2021 ID mapped to its 2025 row by structure
  results/tables/descriptor_families.csv   each of the 190 columns: property, summary, family

Traps handled here (see the workflow doc):
  * vbur_vbur_* is buried volume in cubic angstroms inside a 3.5 A sphere, not percent.
  * 55 Kraken IDs point to different molecules in 2021 and 2025, so IDs are never
    joined across releases: structures are.
"""
from __future__ import annotations

import math

import pandas as pd
from rdkit import Chem, RDLogger

from ligsel.paths import PROCESSED_DIR, RESULTS_DIR, load_config, raw_path

RDLogger.DisableLog("rdApp.*")

SPHERE_RADIUS_A = 3.5
SPHERE_VOLUME_A3 = 4 / 3 * math.pi * SPHERE_RADIUS_A**3  # 179.59 A^3
SUMMARIES = ("boltz", "min", "max", "delta", "vburminconf")
ID_COLS = ("kraken_id", "smiles", "canonical_smiles", "inchikey")

# Family of each property, by prefix. Reviewed by Ted in step 15 (23 Nov 2026): pyramidalisation
# steric, vmin_r electronic, whole-molecule volume, surface area and sphericity steric.
FAMILY_RULES = [
    ("sterimol_", "steric"), ("vbur_", "steric"), ("pyr_", "steric"),
    ("volume", "steric"), ("surface_area", "steric"), ("sphericity", "steric"),
    ("Pint_", "dispersion"),
    ("dipolemoment", "electrostatic"), ("qpole", "electrostatic"),
    ("vmin_", "electronic"), ("fmo_", "electronic"), ("somo_", "electronic"),
    ("nbo_", "electronic"), ("spindens_", "electronic"), ("nmr", "electronic"),
    ("efg", "electronic"), ("nuesp_", "electronic"), ("fukui_", "electronic"),
    ("E_oxidation", "electronic"), ("E_reduction", "electronic"), ("E_solv", "solvation"),
]


def split_column(col: str) -> tuple[str, str]:
    """'vbur_vbur_min' -> ('vbur_vbur', 'min'). Every Kraken column ends in a summary."""
    for s in sorted(SUMMARIES, key=len, reverse=True):
        if col.endswith("_" + s):
            return col[: -len(s) - 1], s
    raise ValueError(f"column {col!r} has no known summary suffix")


def family(prop: str) -> str:
    for prefix, fam in FAMILY_RULES:
        if prop.startswith(prefix):
            return fam
    raise ValueError(f"no family rule for property {prop!r}")


def structure_keys(smiles: str) -> tuple[str | None, str | None]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    can = Chem.MolToSmiles(mol)
    try:
        key = Chem.MolToInchiKey(mol) or None
    except Exception:
        key = None
    return can, key


def load_release(key: str, sheet: str | int, id_col: str, smiles_col: str) -> pd.DataFrame:
    df = pd.read_excel(raw_path(key), sheet_name=sheet)
    df = df.rename(columns={id_col: "kraken_id", smiles_col: "smiles"})
    df = df.drop(columns=[c for c in ("PC1", "PC2", "PC3", "PC4") if c in df.columns])
    keys = df["smiles"].map(structure_keys)
    df.insert(2, "canonical_smiles", [k[0] for k in keys])
    df.insert(3, "inchikey", [k[1] for k in keys])
    # match key: InChIKey, falling back to canonical SMILES where InChI fails (e.g. ferrocenyls)
    df["match_key"] = df["inchikey"].fillna("SMI:" + df["canonical_smiles"].fillna(df["smiles"]))
    return df


def descriptor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS and c != "match_key" and not c.startswith("pct_")]


def add_percent_vbur(df: pd.DataFrame) -> pd.DataFrame:
    for s in SUMMARIES:
        col = f"vbur_vbur_{s}"
        if col in df.columns:
            df[f"pct_vbur_{s}"] = df[col] / SPHERE_VOLUME_A3 * 100
    return df


def family_table(cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        prop, summ = split_column(c)
        rows.append({"column": c, "property": prop, "summary": summ, "family": family(prop)})
    t = pd.DataFrame(rows)
    n_summ = t.groupby("property")["summary"].transform("nunique")
    t["summaries_available"] = n_summ
    return t


def crosswalk(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Map every 2021 ID to its 2025 row by structure.

    Order of preference: the same ID if it still holds the same canonical structure;
    else the one 2025 row with the same InChIKey; else, among several, the one with
    the same canonical SMILES. Anything left is 'ambiguous' and must not be joined.
    The 2025 release lists some structures more than once (e.g. PMe3 five times),
    which is why a plain InChIKey lookup is not enough.
    """
    new_by_id = new.set_index("kraken_id")
    groups = new.groupby("match_key")
    rows = []
    for r in old.itertuples(index=False):
        same = new_by_id["canonical_smiles"].get(r.kraken_id)
        cands = groups.get_group(r.match_key) if r.match_key in groups.groups else new.iloc[0:0]
        if same is not None and same == r.canonical_smiles:
            id_new, status = r.kraken_id, "same_id"
        elif len(cands) == 1:
            id_new = int(cands["kraken_id"].iloc[0])
            status = "same_id" if id_new == r.kraken_id else "renumbered"
        elif len(cands) > 1:
            exact = cands[cands["canonical_smiles"] == r.canonical_smiles]
            if len(exact) == 1:
                id_new = int(exact["kraken_id"].iloc[0])
                status = "same_id" if id_new == r.kraken_id else "renumbered"
            else:
                id_new, status = None, "ambiguous"
        else:
            id_new, status = None, "absent_in_2025"
        rows.append({
            "id_2021": r.kraken_id, "id_2025": id_new, "status": status,
            "id_reused_for_other_structure": same is not None and same != r.canonical_smiles,
            "n_2025_candidates": len(cands), "canonical_smiles": r.canonical_smiles,
            "inchikey": r.inchikey,
        })
    cw = pd.DataFrame(rows)
    cw["id_2025"] = cw["id_2025"].astype("Int64")
    return cw


def duplicate_structures(new: pd.DataFrame) -> pd.DataFrame:
    """2025 rows that share a structure key: a data-quality finding to report."""
    d = new[new["match_key"].duplicated(keep=False)]
    return (d.groupby("match_key")
             .agg(kraken_ids=("kraken_id", lambda x: " ".join(map(str, sorted(x)))),
                  canonical_smiles=("canonical_smiles", "first"), n=("kraken_id", "size"))
             .reset_index(drop=True).sort_values("n", ascending=False))


def run() -> dict:
    load_config()
    new = add_percent_vbur(load_release("kraken_2025", 0, "Kraken_ID", "SMILES"))
    old = add_percent_vbur(load_release("kraken_2021_descriptors", "DFT_data", "ID", "smiles"))
    cols = descriptor_columns(new)
    assert len(cols) == 190, f"expected 190 descriptors, found {len(cols)}"
    assert set(cols) == set(descriptor_columns(old)), "releases have different descriptor columns"

    fam = family_table(cols)
    cw = crosswalk(old, new)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    new.to_csv(PROCESSED_DIR / "kraken_2025.csv.gz", index=False)
    old.to_csv(PROCESSED_DIR / "kraken_2021.csv.gz", index=False)
    cw.to_csv(PROCESSED_DIR / "kraken_crosswalk.csv", index=False)
    fam.to_csv(RESULTS_DIR / "tables" / "descriptor_families.csv", index=False)
    dups = duplicate_structures(new)
    dups.to_csv(RESULTS_DIR / "tables" / "kraken_duplicate_structures.csv", index=False)

    summary = {
        "ligands_2025": len(new), "ligands_2021": len(old), "descriptors": len(cols),
        "properties": fam["property"].nunique(),
        "five_summary_properties": int((fam.groupby("property")["summary"].nunique() == 5).sum()),
        "boltz_only_properties": int((fam.groupby("property")["summary"].nunique() == 1).sum()),
        "inchi_failures_2025": int(new["inchikey"].isna().sum()),
        **{f"crosswalk_{k}": int(v) for k, v in cw["status"].value_counts().items()},
        "ids_reused_for_other_structure": int(cw["id_reused_for_other_structure"].sum()),
        "duplicate_structure_groups_2025": len(dups),
    }
    for k, v in summary.items():
        print(f"{k:34s} {v}")
    return summary


if __name__ == "__main__":
    run()
