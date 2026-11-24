"""Step 4: extract every yield and threshold table from the two SI PDFs.

Outputs (not committed: the SI tables are covered by the publishers' terms)
  data/interim/si_yields.csv        one row per (dataset, ligand)
  data/interim/si_thresholds.csv    the published %V_bur(min) thresholds and metrics
  data/interim/spotcheck.csv        20 random rows with PDF pages, for a hand check
Committed
  results/tables/si_extraction_report.csv   row counts per table against expectations

Text comes from `pdftotext -layout` (poppler: `brew install poppler` on a Mac).
Rows are split on runs of 2+ spaces: first field the Kraken ID, last k fields the values,
anything between them the ligand name. Names that wrap onto neighbouring lines leave the
ID row without a name; the name is rebuilt from the fragments above and below it.
"""
from __future__ import annotations

import re
import subprocess
import sys

import numpy as np
import pandas as pd

from ligsel.paths import INTERIM_DIR, RESULTS_DIR, raw_path

V3 = ["vbur_boltz", "vbur_min", "yield"]

# (source, table number, title start, value columns, {value column: dataset}, expected rows, how the
# expectation was established). Datasets named cliffs_<reaction> or j22_<reaction>.
TABLES = [
    ("cliffs_si", "S1", "Table S1. Compiled yields", ["cone_boltz", "cone_min", "vbur_boltz", "vbur_min", "yield"],
     {"yield": "cliffs_I"}, 34, "SI text: 19 ligands from Wu et al. plus 15 added"),
    ("cliffs_si", "S2", "Table S2. Compiled yields",
     ["cone_boltz", "cone_min", "vbur_boltz", "vbur_min", "y_II", "y_III", "y_IV", "y_V", "y_RS1"],
     {"y_II": "cliffs_II", "y_III": "cliffs_III", "y_IV": "cliffs_IV", "y_V": "cliffs_V", "y_RS1": "cliffs_RS1"},
     90, "main text: 90 ligands in the HTE set"),
    ("cliffs_si", "S9", "Table S9. Tabulated", V3, {"yield": "cliffs_VII"}, 55, "parsed; checked against Kraken in step 9"),
    ("cliffs_si", "S10", "Table S10. Tabulated", V3, {"yield": "cliffs_VIII"}, 53, "parsed; checked against Kraken in step 9"),
    ("cliffs_si", "S11", "Table S11. Tabulated", V3, {"yield": "cliffs_IX"}, 30, "parsed; checked against Kraken in step 9"),
    ("cliffs_si", "S12", "Table S12. Tabulated", V3, {"yield": "cliffs_X"}, 28, "parsed; checked against Kraken in step 9"),
    ("cliffs_si", "S13", "Table S13. Tabulated", V3, {"yield": "cliffs_RS2"}, 30, "parsed; checked against Kraken in step 9"),
    ("cliffs_si", "S14", "Table S14. Tabulated", V3, {"yield": "cliffs_RS3"}, 29, "parsed; checked against Kraken in step 9"),
    ("cliffs_si", "S15", "Table S15. Tabulated", V3, {"yield": "cliffs_XI"}, 30, "parsed; checked against Kraken in step 9"),
    ("cliffs_si", "S16", "Table S16. Tabulated", V3, {"yield": "cliffs_XII"}, 71, "parsed; checked against Kraken in step 9"),
    ("cliffs_si", "S17", "Table S17. Tabulated", V3, {"yield": "cliffs_XIII"}, 79, "parsed; checked against Kraken in step 9"),
    ("cliffs_si", "S18", "Table S18. Tabulated", V3, {"yield": "cliffs_XIV"}, 14, "parsed; checked against Kraken in step 9"),
    # JACS 2022 lists %Vbur (min) BEFORE %Vbur (Boltz): the opposite of the cliffs SI.
    ("jacs2022_si", "S1", "Table S1. Compiled yields", ["buchwald", "vbur_min", "vbur_boltz", "y_I", "y_Ib", "y_II"],
     {"y_I": "j22_I", "y_Ib": "j22_Ib", "y_II": "j22_II"}, 95, "Table S4: TP+TN+FP+FN = 95 for Reaction I"),
    ("jacs2022_si", "S2", "Table S2. Compiled yields", ["buchwald", "vbur_min", "vbur_boltz", "yield"],
     {"yield": "j22_III"}, 17, "first page only; the other 26 rows are hand-transcribed (page S12 has no text)"),
    ("jacs2022_si", "S3", "Table S3. Compiled yields", ["buchwald", "vbur_min", "vbur_boltz", "yield"],
     {"yield": "j22_IV"}, 16, "parsed; ligandless control row dropped"),
]
EXPECTED_DATASET_ROWS = {"j22_III": (43, "Table S4: TP+TN+FP+FN = 43")}

NULLS = {"--", "—", "-", "N/A", "NA", "n.d.", "nd"}
TITLE = re.compile(r"^\s*Table S\d+\.")
PAGE_LABEL = re.compile(r"^S\d+$")


def pdf_text(key: str) -> list[str]:
    """Layout text of a PDF, cached in data/interim, one string per page."""
    cache = INTERIM_DIR / f"{key}.layout.txt"
    if not cache.exists():
        try:
            subprocess.run(["pdftotext", "-layout", str(raw_path(key)), str(cache)], check=True)
        except FileNotFoundError:
            sys.exit("pdftotext not found: install poppler (`brew install poppler` on a Mac)")
    return cache.read_text().split("\f")


def value(tok: str):
    tok = tok.strip().rstrip("*")
    if tok in NULLS:
        return np.nan
    if tok in ("Yes", "No"):
        return tok == "Yes"
    return float(tok)


def is_value(tok: str) -> bool:
    try:
        value(tok)
        return True
    except ValueError:
        return False


def table_lines(pages: list[str], title: str) -> list[tuple[int, str]]:
    """(pdf page, line) from the table's title to the next table title."""
    lines = [(i + 1, l) for i, p in enumerate(pages) for l in p.split("\n")]
    start = next(k for k, (_, l) in enumerate(lines) if l.strip().startswith(title))
    end = next((k for k in range(start + 1, len(lines)) if TITLE.match(lines[k][1])), len(lines))
    return lines[start:end]


def parse_row(line: str, k: int):
    fields = re.split(r"\s{2,}", line.strip())
    if len(fields) < k + 1 or not re.fullmatch(r"\d{1,4}", fields[0]):
        return None
    vals, middle = fields[-k:], fields[1:-k]
    if not all(is_value(v) for v in vals) or any(is_value(m) for m in middle):
        return None
    return int(fields[0]), " ".join(middle), [value(v) for v in vals]


def fragment(line: str) -> bool:
    s = line.strip()
    return bool(s) and len(s) < 60 and not PAGE_LABEL.match(s) and not TITLE.match(s) \
        and "Vbur" not in s and "Ligand" not in s and parse_row(line, 1) is None


def rebuild_name(lines, k: int) -> str:
    parts = []
    for j in (k - 1, k + 1):
        if 0 <= j < len(lines) and fragment(lines[j][1]):
            parts.append(lines[j][1].strip())
    return " ".join(parts)


def extract_table(pages, source, table, title, cols, datasets):
    lines = table_lines(pages, title)
    rows = []
    for k, (page, line) in enumerate(lines):
        parsed = parse_row(line, len(cols))
        if not parsed:
            continue
        kid, name, vals = parsed
        rec = dict(zip(cols, vals))
        rows.append({"source": source, "si_table": table, "pdf_page": page, "kraken_id_si": kid,
                     "ligand": name or rebuild_name(lines, k), "name_rebuilt": not name, **rec})
    return pd.DataFrame(rows)


def to_long(wide: pd.DataFrame, datasets: dict) -> pd.DataFrame:
    keep = [c for c in ("source", "si_table", "pdf_page", "kraken_id_si", "ligand", "name_rebuilt",
                        "buchwald", "vbur_min", "vbur_boltz", "cone_min", "cone_boltz") if c in wide]
    out = []
    for col, ds in datasets.items():
        part = wide[keep].copy()
        part.insert(0, "dataset", ds)
        part["yield"] = wide[col]
        out.append(part)
    return pd.concat(out, ignore_index=True)


def manual_j22_s2() -> pd.DataFrame:
    m = pd.read_csv(raw_path("jacs2022_si_S2_p12_manual"), comment="#")
    return pd.DataFrame({
        "dataset": "j22_III", "source": "jacs2022_si", "si_table": "S2 (hand-transcribed)",
        "pdf_page": 12, "kraken_id_si": m["kraken_id"], "ligand": m["ligand"], "name_rebuilt": False,
        "buchwald": m["buchwald"] == "Yes", "vbur_min": m["vbur_min"], "vbur_boltz": m["vbur_boltz"],
        "yield": m["yield"]})


ROMAN = r"(I|Ib|II|III|IV|V|VII|VIII|IX|X|XI|XII|RS1)"


def thresholds(cliffs_pages, j22_pages) -> pd.DataFrame:
    rows = []
    pat = re.compile(ROMAN + r"\s+(\d+)\s+(?:\S+\s+)?([\d.]+)\s+(Left|Right)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
    for table, weight in (("S7", "{0:1, 1:20}"), ("S8", "balanced")):
        for _, line in table_lines(cliffs_pages, f"Table {table}. Single node"):
            m = pat.search(line)
            if m:
                r, ycut, thr, d, acc, f1, mcc = m.groups()
                rows.append({"dataset": f"cliffs_{r}", "si_table": table, "subset": "All",
                             "class_weight": weight, "y_cut": float(ycut), "threshold": float(thr),
                             "direction": d, "accuracy": float(acc), "f1": float(f1), "mcc": float(mcc)})
    order = [("I", 10), ("Ib", 10), ("II", 20), ("III", 10), ("IV", 24)]
    stats = [l for _, l in table_lines(j22_pages, "Table S4. Compiled threshold")
             if re.match(r"^\s*(All|Buchwald)\s", l)]
    assert len(stats) == 2 * len(order), f"JACS 2022 Table S4: expected 10 rows, found {len(stats)}"
    for k, line in enumerate(stats):
        f = line.split()
        r, ycut = order[k // 2]
        num = [np.nan if x == "NA" else float(x) for x in f[1:]]
        rows.append({"dataset": f"j22_{r}", "si_table": "S4", "subset": f[0], "class_weight": "balanced",
                     "y_cut": float(ycut), "threshold": num[0], "direction": "Left" if f[1] != "NA" else None,
                     "tp": num[1], "tn": num[2], "fp": num[3], "fn": num[4], "mcc": num[5],
                     "npv": num[6], "recall": num[7]})
    return pd.DataFrame(rows)


def run() -> pd.DataFrame:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    pages = {"cliffs_si": pdf_text("cliffs_si"), "jacs2022_si": pdf_text("jacs2022_si")}
    longs, report = [], []
    printed = INTERIM_DIR / "si_tables_as_printed"
    printed.mkdir(exist_ok=True)
    for source, table, title, cols, datasets, expected, how in TABLES:
        wide = extract_table(pages[source], source, table, title, cols, datasets)
        # the same table with the PDF's own column order and headers, for checking by eye
        if len(wide):
            heads = {"cone_boltz": "Cone angle (Boltz)", "cone_min": "Cone angle (min)",
                     "vbur_boltz": "%Vbur (Boltz)", "vbur_min": "%Vbur (min)", "buchwald": "Buchwald-type",
                     "yield": "Yield (%)", **{c: f"{c[2:]} yield (%)" for c in cols if c.startswith("y_")}}
            out = wide[["pdf_page", "kraken_id_si", "ligand"] + cols].rename(
                columns={"pdf_page": "PDF page", "kraken_id_si": "Kraken ID", "ligand": "Ligand", **heads})
            out.to_csv(printed / f"{source}_{table}.csv", index=False)
        longs.append(to_long(wide, datasets))
        report.append({"source": source, "si_table": table, "datasets": " ".join(datasets.values()),
                       "rows_found": len(wide), "rows_expected": expected, "expectation_from": how,
                       "names_rebuilt": int(wide["name_rebuilt"].sum()) if len(wide) else 0,
                       "ok": len(wide) == expected})
    longs.append(manual_j22_s2())
    yields = pd.concat(longs, ignore_index=True)
    for ds, (n, how) in EXPECTED_DATASET_ROWS.items():
        found = int((yields["dataset"] == ds).sum())
        report.append({"source": "combined", "si_table": ds, "datasets": ds, "rows_found": found,
                       "rows_expected": n, "expectation_from": how, "names_rebuilt": 0, "ok": found == n})
    thr = thresholds(pages["cliffs_si"], pages["jacs2022_si"])
    report.append({"source": "cliffs_si", "si_table": "S7+S8", "datasets": "thresholds",
                   "rows_found": int((thr["si_table"] != "S4").sum()), "rows_expected": 24,
                   "expectation_from": "12 datasets x 2 class weights", "names_rebuilt": 0,
                   "ok": int((thr["si_table"] != "S4").sum()) == 24})

    yields.to_csv(INTERIM_DIR / "si_yields.csv", index=False)
    thr.to_csv(INTERIM_DIR / "si_thresholds.csv", index=False)
    rep = pd.DataFrame(report)
    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    rep.to_csv(RESULTS_DIR / "tables" / "si_extraction_report.csv", index=False)
    spot = yields.dropna(subset=["yield"]).sample(20, random_state=20261125)
    spot[["dataset", "source", "si_table", "pdf_page", "kraken_id_si", "ligand", "vbur_boltz", "vbur_min", "yield"]] \
        .sort_values(["source", "pdf_page"]).to_csv(INTERIM_DIR / "spotcheck.csv", index=False)
    print(rep[["source", "si_table", "rows_found", "rows_expected", "names_rebuilt", "ok"]].to_string(index=False))
    if not rep["ok"].all():
        sys.exit("extraction does not match expected row counts")
    print(f"\n{len(yields)} yield rows across {yields['dataset'].nunique()} datasets; {len(thr)} threshold rows")
    return rep


if __name__ == "__main__":
    run()
