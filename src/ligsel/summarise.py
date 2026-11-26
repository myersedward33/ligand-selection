"""Score H1-H3 from results/tables/ladder_per_dataset.csv:

H1  elastic-net ('all' view) beats the cross-validated baseline by >= 0.10 MCC (LOO) on a majority
    of the eligible primary datasets; also reported without VII, VIII and IX (deviation log, 23 Nov).
H2  on Ni datasets I-V, logistic %V_bur(min) + %V_bur(Boltz) beats %V_bur(min) alone by >= 0.05 MCC.
H3  for the elastic net, the vburminconf view beats the boltz view.
H4 and H5 are scored in later steps (negative controls; external test after the `frozen` tag).
"""
from __future__ import annotations

import pandas as pd

from ligsel.paths import RESULTS_DIR

SENSITIVITY_DROP = {"cliffs_VII", "cliffs_VIII", "cliffs_IX"}
NI = ["cliffs_I", "cliffs_II", "cliffs_III", "cliffs_IV", "cliffs_V"]


def run() -> dict:
    t = pd.read_csv(RESULTS_DIR / "tables" / "ladder_per_dataset.csv")
    get = lambda view, rung: t[(t.view == view) & (t.rung == rung)].set_index("dataset")["mcc_loo"]
    en, base = get("all", "elastic_net"), get("all", "baseline_cv")
    h1 = pd.DataFrame({"baseline_cv": base.reindex(en.index), "elastic_net": en})
    h1["delta"] = h1["elastic_net"] - h1["baseline_cv"]
    h1["meets_0.10"] = h1["delta"] >= 0.10
    sens = h1.drop(index=[d for d in SENSITIVITY_DROP if d in h1.index])

    h2 = pd.DataFrame({"min_alone": get("all", "single_min_lr").reindex(NI),
                       "min_plus_boltz": get("all", "pair_min_boltz").reindex(NI)})
    h2["delta"] = h2["min_plus_boltz"] - h2["min_alone"]
    h2["meets_0.05"] = h2["delta"] >= 0.05

    h3 = pd.DataFrame({"boltz": get("boltz", "elastic_net"), "vburminconf": get("vburminconf", "elastic_net")})
    h3["delta"] = h3["vburminconf"] - h3["boltz"]

    out = {
        "H1_datasets_meeting": f"{int(h1['meets_0.10'].sum())} of {len(h1)}",
        "H1_supported": bool(h1["meets_0.10"].sum() > len(h1) / 2),
        "H1_sensitivity_meeting": f"{int(sens['meets_0.10'].sum())} of {len(sens)}",
        "H1_sensitivity_supported": bool(sens["meets_0.10"].sum() > len(sens) / 2),
        "H2_datasets_meeting": f"{int(h2['meets_0.05'].sum())} of {len(h2)}",
        "H3_vburminconf_better": f"{int((h3['delta'] > 0).sum())} of {len(h3)}",
    }
    with open(
            RESULTS_DIR / "tables" / "hypotheses_h1_h3.md", "w") as fh:
        fh.write("# Hypotheses H1-H3 (leave-one-ligand-out MCC)\n\n")
        for k, v in out.items():
            fh.write(f"- {k}: {v}\n")
        for name, df in (("H1", h1), ("H1 sensitivity (without VII, VIII, IX)", sens), ("H2", h2), ("H3", h3)):
            fh.write(f"\n## {name}\n\n{df.round(3).to_markdown()}\n")
    for k, v in out.items():
        print(f"{k:28s} {v}")
    print("\n", h1.round(3).to_string())
    return out


if __name__ == "__main__":
    run()
