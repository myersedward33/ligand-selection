"""Characterize the Kraken library before any modelling:

Outputs
  results/figures/descriptor_correlation.png   clustered Spearman correlation of the 190 descriptors
  results/tables/descriptor_clusters.csv       each descriptor's cluster at average |rho| >= 0.9
  results/figures/tep_skew.png                 TEP of the modelled ligands vs Kraken vs TEPid's reference
  results/tables/tep_summary.csv               the numbers behind that figure

TEP comes from data/processed/tepid.csv (scripts/run_tepid.py); the TEP figure is skipped without it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

from ligsel.paths import PROCESSED_DIR, RESULTS_DIR, load_config

# Reference palette (dataviz skill, light mode)
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#1c5cab", "#86b6ef", "#f0efec", "#ee9a99", "#b8302f"])
FAMILY_ORDER = ["steric", "electronic", "electrostatic", "dispersion", "solvation"]
CLUSTER_RHO = 0.9


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(colors=INK2, labelsize=9)


def correlation_figure() -> dict:
    k = pd.read_csv(PROCESSED_DIR / "kraken_2025.csv.gz")
    fam = pd.read_csv(RESULTS_DIR / "tables" / "descriptor_families.csv")
    cols = fam["column"].tolist()
    rho = k[cols].corr(method="spearman").fillna(0.0)
    dist = 1 - rho.abs().values
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method="average")
    clusters = fcluster(Z, t=1 - CLUSTER_RHO, criterion="distance")
    order = leaves_list(Z)
    pd.DataFrame({"column": cols, "cluster": clusters}).merge(fam, on="column").to_csv(
        RESULTS_DIR / "tables" / "descriptor_clusters.csv", index=False)
    n_clusters = len(set(clusters))
    n_props = fam["property"].nunique()

    fig = plt.figure(figsize=(8.2, 8.0), facecolor=SURFACE)
    ax = fig.add_axes([0.08, 0.06, 0.78, 0.78])
    strip = fig.add_axes([0.08, 0.845, 0.78, 0.018])
    cax = fig.add_axes([0.88, 0.06, 0.018, 0.30])
    m = rho.values[np.ix_(order, order)]
    im = ax.imshow(m, cmap=DIVERGING, vmin=-1, vmax=1, interpolation="nearest", aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fams = fam.set_index("column").loc[[cols[i] for i in order], "family"].map(FAMILY_ORDER.index).values
    from matplotlib.colors import ListedColormap
    strip.imshow(fams[None, :], cmap=ListedColormap(SERIES[: len(FAMILY_ORDER)]), aspect="auto",
                 vmin=-0.5, vmax=len(FAMILY_ORDER) - 0.5, interpolation="nearest")
    strip.set_axis_off()
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Spearman ρ", color=INK2, fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    fig.text(0.08, 0.965, f"The 190 Kraken descriptors carry about {n_clusters} distinct signals at |ρ| ≥ {CLUSTER_RHO}",
             fontsize=13, fontweight="bold", color=INK)
    fig.text(0.08, 0.935, f"{n_props} underlying properties; 1,614 DFT ligands; descriptors ordered by "
             "average-linkage clustering on 1 − |ρ|", fontsize=9.5, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[i]) for i in range(len(FAMILY_ORDER))]
    fig.legend(handles, FAMILY_ORDER, loc="upper left", bbox_to_anchor=(0.08, 0.915), ncol=6,
               frameon=False, fontsize=8.5, labelcolor=INK2, handlelength=1, columnspacing=1.2)
    fig.savefig(RESULTS_DIR / "figures" / "descriptor_correlation.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return {"descriptor_clusters_rho0.9": n_clusters}


def tep_figure() -> dict:
    tep_path = PROCESSED_DIR / "tepid.csv"
    if not tep_path.exists():
        print("data/processed/tepid.csv missing: run `make tepid` first; skipping the TEP figure")
        return {}
    tep = pd.read_csv(tep_path).dropna()
    ref_all = pd.read_csv(PROCESSED_DIR / "tepid_reference.csv")
    # like for like: TEPid's predictions on its own reference set, not that set's DFT values
    ref = ref_all["tep_predicted"].dropna()
    err = (ref_all["tep_predicted"] - ref_all["tep_dft_corrected"]).dropna()
    j = pd.read_csv(PROCESSED_DIR / "joined.csv")
    roles = load_config()["datasets"]
    used = j[j["included"] & j["dataset"].map(lambda d: roles[d]["role"] in ("primary", "external"))]["id_2025"]
    modelled = tep[tep["kraken_id"].isin(used.dropna().astype(int))]["tep_cm1"]
    groups = [("TEPid reference set, TEPid-predicted", ref, SERIES[2]),
              ("Kraken DFT library, TEPid-predicted", tep["tep_cm1"], SERIES[0]),
              ("Modelled ligands, TEPid-predicted", modelled, SERIES[1])]
    summary = pd.DataFrame([{"set": n, "n": len(v), "median": v.median(), "mean": v.mean(), "sd": v.std(),
                             "q10": v.quantile(0.1), "q90": v.quantile(0.9)} for n, v, _ in groups])
    summary = pd.concat([summary, pd.DataFrame([{
        "set": "TEPid reference: prediction minus DFT", "n": len(err), "median": err.median(),
        "mean": err.mean(), "sd": err.std(), "q10": err.quantile(0.1), "q90": err.quantile(0.9)}])])
    summary.round(2).to_csv(RESULTS_DIR / "tables" / "tep_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 4.4), facecolor=SURFACE)
    fig.subplots_adjust(top=0.80, left=0.08, right=0.97, bottom=0.14)
    bins = np.arange(2045, 2115, 1.5)
    for name, v, c in groups:
        h, e = np.histogram(v, bins=bins, density=True)
        ax.step(e[:-1], h, where="post", color=c, lw=2, label=f"{name} (n = {len(v):,})")
        ax.axvline(v.median(), color=c, lw=1, ls=(0, (3, 3)))
    style(ax)
    ax.set_xlabel("Tolman electronic parameter (cm⁻¹); lower = stronger donor", color=INK2, fontsize=9.5)
    ax.set_ylabel("Density", color=INK2, fontsize=9.5)
    ax.grid(axis="y", color="#ecebe7", lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
    s = summary.set_index("set")["median"]
    fig.text(0.08, 0.93, "The ligands being modelled sit on the donor side of the electronic range",
             fontsize=13, fontweight="bold", color=INK)
    fig.text(0.08, 0.87, f"Medians (dashed): modelled {s.iloc[2]:.1f}, Kraken {s.iloc[1]:.1f}, TEPid reference "
             f"{s.iloc[0]:.1f} cm⁻¹, all TEPid predictions", fontsize=9, color=INK2)
    fig.savefig(RESULTS_DIR / "figures" / "tep_skew.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return {f"tep_median_{i}": round(float(s.iloc[i]), 1) for i in range(3)}


def run() -> dict:
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    out = {**correlation_figure(), **tep_figure()}
    for k_, v in out.items():
        print(f"{k_:30s} {v}")
    return out


if __name__ == "__main__":
    run()
