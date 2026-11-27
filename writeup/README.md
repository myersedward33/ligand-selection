# Can 185 DFT Descriptors Beat One? Multivariate vs. %V<sub>bur</sub>(min) Classification of Phosphine Ligand Activity in Ni- and Pd-Catalysed Cross-Coupling

**Edward Myers** · 26 November 2026

Code: [`myersedward33/ligand-selection`](https://github.com/myersedward33/ligand-selection) (MIT licence). The raw data are not redistributed.

## Contents

- [Software](#software)
- [Research question and hypotheses](#research-question-and-hypotheses)
- [Methodology](#methodology)
- [Results](#results)
- [Error analysis](#error-analysis)
- [Discussion](#discussion)
- [Limitations and next steps](#limitations-and-next-steps)
- [References](#references)

## Software

All code was written in Python 3.10.12; relevant libraries include scikit-learn 1.7.2, NumPy 2.2.6, pandas 2.3.3, SciPy and RDKit. TEPid was run in a separate pinned environment (LightGBM 3.3.5, scikit-learn 1.2.2).

## Research question and hypotheses

Across the reactivity-cliffs datasets of Newman-Stonebraker et al., a single steric descriptor, %V<sub>bur</sub>(min), classifies monodentate phosphines as active or inactive at a threshold near 32% [1]. Does a multivariate model over Kraken's 190 DFT descriptors classify them better, and if so, which descriptors carry the extra signal?

- **H1 (primary).** The cross-validated elastic-net model beats the cross-validated baseline by at least 0.10 MCC (leave-one-ligand-out) on a majority of the eligible primary datasets.
- **H2.** On the Ni datasets (I–V), logistic regression on %V<sub>bur</sub>(min) + %V<sub>bur</sub>(Boltz) beats %V<sub>bur</sub>(min) alone by at least 0.05 MCC. This hypothesis is based on the JACS 2022 paper, where %V<sub>bur</sub>(Boltz) separated Buchwald-type ligands that %V<sub>bur</sub>(min) could not [4].
- **H3.** For the 28 properties Kraken summarises in five ways, the minimum-%V<sub>bur</sub>-conformer values (`_vburminconf`) outperform the Boltzmann averages in the elastic-net model.
- **H4.** On the negative controls, no model beats a label-permutation null (500 permutations, p < 0.05). A model that does is treated as overfitting until shown otherwise.
- **H5.** On the external test, the frozen primary model and the baseline at the published 31.89% threshold are each scored once, after the `frozen` tag.

## Methodology

The questions, hypotheses, models and success criterion were pre-registered (git tag `prereg`) before any models were fitted. Later changes are listed with dates and reasons in the deviations log (`docs/preregistration.md`, section 9).

I began by collecting open-source data on a large ligand set that I could then fit a model ladder to. Descriptors to classify the ligands came from the Kraken 2025 release, which contains 1,614 monodentate phosphines with 190 DFT descriptors each [3, 16]. Ligand reactivity was assessed using yields from the SI of the Newman-Stonebraker et al. reactivity-cliffs study [1, 2] and its JACS 2022 follow-up [4], following each dataset's published yield cutoff.

This gave 11 primary datasets (Ni: I–V; Pd: VII–XII), one supplementary dataset (RS1), five negative controls (RS2, RS3, XIII, XIV, JACS 2022 IV) and one external test set (JACS 2022 III, 43 ligands). Reactions I, Ib and II from the JACS 2022 paper were excluded because they re-screen chemistry already covered in cliffs II–V.

SI tables from each paper were converted with `pdftotext -layout` and parsed into one row per dataset and ligand. Row counts were checked against the totals stated in each paper, and 20 random rows were checked by hand against the SI PDFs. Each SI identifies ligands using the 2021 Kraken IDs, and 55 of those IDs refer to different molecules in the 2025 release. Each 2021 ID was therefore mapped to its 2025 entry by structure (InChIKey, falling back to canonical SMILES), never by ID. A row was kept only if it had a yield, a unique 2025 structure match, and SI %V<sub>bur</sub>(min) and %V<sub>bur</sub>(Boltz) values within 0.1 of Kraken's. The pipeline stops if any dataset matches below 95%. Every dataset matched 100%, giving 661 primary rows over 194 distinct ligands.

A ligand was labelled active if its yield was at or above the dataset's published cutoff (5–30%). Controls without a published cutoff used 10%, the most common value.

Buried volumes were converted from Å³ to percent by dividing by 179.59 Å³ and multiplying by 100. Five descriptors (one %V<sub>bur</sub>(min) clone and four near-constant ones) were dropped before modelling, leaving 185 in total. There are five feature views: Boltzmann (78 features), minimum (76), maximum (78), minimum-%V<sub>bur</sub> conformer (75) and all summaries (185). They differ only in which summary of the 28 multi-summary properties they include. H1 uses the all-summaries view. Each descriptor was assigned a family (steric, electronic, electrostatic, dispersion or solvation) by name prefix, with borderline cases decided by hand. Descriptors were clustered by average linkage on 1 − |Spearman ρ| across the 1,614 library ligands and cut at |ρ| = 0.9, giving 108 clusters. To describe the library, Tolman electronic parameters were predicted with the published TEPid model and compared with TEPid's own reference set, prediction against prediction [5].

Next, the published classifier was refitted exactly: a depth-1 decision tree on %V<sub>bur</sub>(min) with balanced class weights, fitted and scored on the whole dataset, as in the SigmanGroup/Threshold code. To pass my gate, at least 10 of the 11 primary datasets had to reproduce the published threshold within 0.11 and the MCC within 0.03. For each dataset that failed, every single-ligand removal and single-label flip was tested for one that reproduced the published row exactly.

Each model was then fitted per dataset, with balanced class weights, and all scaling was fitted on the training fold only. The rungs of my model ladder are as follows:

**Table 1.** Employed model ladder rungs.

| Rung | Model |
| --- | --- |
| Baseline (cross-validated) | The published depth-1 tree on %V<sub>bur</sub>(min), with its threshold refitted in every training fold |
| Best single descriptor | Depth-1 tree on whichever of the 185 descriptors gives the highest training MCC |
| Best pair | Logistic regression on whichever of the ~17,000 descriptor pairs gives the lowest training log-loss (ridge penalty 0.01) |
| %V<sub>bur</sub>(min) logistic | Logistic regression on %V<sub>bur</sub>(min) alone (the H2 comparator) |
| %V<sub>bur</sub>(min) + %V<sub>bur</sub>(Boltz) | Logistic regression on this fixed pair (H2) |
| Elastic net (primary) | Standardised elastic-net logistic regression [8]; C ∈ {0.01, 0.03, 0.1, 0.3, 1, 3}, L1 ratio ∈ {0.3, 0.7, 1.0} |
| Random forest | 300 trees [13]; minimum samples per leaf ∈ {1, 3, 5} |
| Gradient boosting | Histogram gradient boosting, 200 iterations; learning rate ∈ {0.05, 0.15}, depth ∈ {2, 3}, minimum samples per leaf ∈ {1, 3, 5} |

Hyperparameters were chosen by stratified 5-fold cross-validation scored by MCC [10], run inside every outer training fold (nested) [12]. Descriptor and pair searches were also repeated inside every fold. Multivariate models were fitted only where n ≥ 30 and the smaller class had at least 10 ligands, which excludes dataset X.

My primary method of validation was leave-one-ligand-out. Ligands were split into five classes by the substituents on phosphorus, assigned by SMARTS rules: Buchwald-type biaryl, P-heteroatom, trialkyl, triaryl and alkyl-aryl. Three pools (all 11 datasets, the Ni pool (I–V) and the Pd pool (VII–XII)) were each evaluated by leave-one-dataset-out, 5-fold CV grouped by ligand identity, and leave-one-class-out, for the baseline, the fixed pair, the elastic net and the random forest. Inner tuning folds were stratified, not grouped by ligand. MCC was computed on the pooled held-out predictions, with a 95% interval from 1,000 bootstrap resamples over ligands. Balanced accuracy was also reported, along with repeated stratified 5-fold (5 repeats, mean ± SD).

To determine stability, each eligible dataset's whole elastic-net procedure, inner tuning included, was refitted on 500 class-stratified bootstrap samples [9]. A descriptor counted as stable if it was nonzero in more than 80% of refits and had the same sign in at least 90% of those. Because many descriptors are near-duplicates, each cluster was also scored by the share of refits that kept any of its members. The pair the best-pair search chose in each leave-one-out fold was tallied.

No control dataset meets the size rule for multivariate fits. The four univariate and fixed-pair rungs were therefore tested against a label-permutation null [11]. The observed stratified 5-fold MCC was compared with 500 permuted-label runs, each repeating the whole procedure, including the threshold or descriptor choice. p = (1 + number of null runs scoring at or above the observed MCC) / 501. As pre-registered, H4 fails if any p < 0.05. A Bonferroni correction is reported post hoc.

To assess H5, an external frozen test was used. The model chosen was an elastic net trained on the Ni pool: 394 rows, 103 ligands and 169 actives, each row labelled at its own dataset's cutoff. It used all 185 descriptors and the ladder's grid, tuned by 5-fold stratified CV grouped by ligand, then refitted on the whole pool. The selected settings were C = 0.03 and L1 ratio = 0.3, keeping 44 nonzero descriptors. The published threshold (active if %V<sub>bur</sub>(min) ≤ 31.89 [4]) and a threshold refitted on the same Ni pool (32.74) were used as the comparators. All prediction parameters were written to plain JSON and git-tagged `frozen` before any code read the external set. MCC was reported with a 95% interval from 2,000 bootstrap resamples over ligands, alongside balanced accuracy and the confusion matrix. Differences between models used a paired bootstrap, and the pre-stated rule was that one model transfers better than another only if the 95% interval of their difference on all 43 ligands excludes zero.

## Results

All MCCs are leave-one-ligand-out with a 95% bootstrap interval unless stated otherwise. Δ is the elastic net minus the cross-validated baseline. Every number below can be regenerated from the files named in each caption (`make all`, then `make freeze` and `make external` for the external test).

**Table 2.** All 18 datasets joined to Kraken at a 100% match rate: 661 primary rows over 194 ligands, with 31–67% of ligands active.

| Dataset | Metal | Role | Ligands (n) | Cutoff (% yield) | Active | Inactive |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| I | Ni | Primary | 34 | 10 | 19 | 15 |
| II | Ni | Primary | 90 | 10 | 38 | 52 |
| III | Ni | Primary | 90 | 5 | 28 | 62 |
| IV | Ni | Primary | 90 | 5 | 32 | 58 |
| V | Ni | Primary | 90 | 20 | 52 | 38 |
| VII | Pd | Primary | 55 | 10 | 29 | 26 |
| VIII | Pd | Primary | 53 | 10 | 27 | 26 |
| IX | Pd | Primary | 30 | 10 | 20 | 10 |
| X | Pd | Primary (univariate only) | 28 | 5 | 15 | 13 |
| XI | Pd | Primary | 30 | 30 | 19 | 11 |
| XII | Pd | Primary | 71 | 10 | 39 | 32 |
| RS1 | Ni | Supplementary | 89 | 10 | 34 | 55 |
| RS2 | Pd | Control | 30 | 10\* | 23 | 7 |
| RS3 | Pd | Control | 29 | 10\* | 23 | 6 |
| XIII | Pd | Control | 79 | 10\* | 75 | 4 |
| XIV | Pd | Control | 14 | 10\* | 7 | 7 |
| JACS 2022 IV | Ni | Control | 16 | 24 | 10 | 6 |
| JACS 2022 III | Ni | External test | 43 | 10 | 12 | 31 |

\* No published cutoff; 10% assumed.
*Sources: `results/tables/join_report.csv`, `si_extraction_report.csv`.*

**Table 3.** Comparison of published and calculated %V<sub>bur</sub>(min) thresholds and in-sample MCC. The pipeline reproduces the published %V<sub>bur</sub>(min) thresholds exactly on 9 of 11 datasets; all three gate failures trace to the published table, not the extracted data.

| Dataset | Published threshold | My threshold | Published MCC | My MCC | Within gate? | Notes |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| I | 31.60 | 31.60 | 0.65 | 0.652 | Yes | |
| II | 32.74 | 32.74 | 0.53 | 0.545 | Yes | |
| III | 31.17 | 31.17 | 0.51 | 0.513 | Yes | |
| IV | 31.89 | 31.89 | 0.45 | 0.462 | Yes | |
| V | 32.95 | 32.95 | 0.51 | 0.525 | Yes | |
| VII | 32.43 | 32.43 | 0.43 | 0.398 | No | Shown with Buchwald-type ligands excluded, as SI p. S84 does (n = 39); on all 55 ligands the fit moves to 25.18 |
| VIII | 28.87 | 28.87 | 0.75 | 0.705 | No | Threshold matches; MCC gap unexplained |
| IX | 28.65 | 30.53 | 0.84 | 0.722 | No | Reproduced exactly when PCy<sub>3</sub> or PCyp<sub>3</sub> is removed |
| X | 30.53 | 30.53 | 0.86 | 0.864 | Yes | |
| XI | 28.82 | 28.82 | 0.79 | 0.793 | Yes | |
| XII | 29.58 | 29.58 | 0.38 | 0.376 | Yes | |

*Sources: `results/tables/baseline_reproduction.csv`; sign-off, `docs/signoffs.md`.*

![Figure 1: per-dataset leave-one-ligand-out MCC with 95% intervals for the cross-validated threshold and the elastic net](figures/fig1_h1_elastic_net_vs_threshold.png)

**Figure 1.** The elastic net clears the pre-registered +0.10 MCC bar on only 3 of 10 datasets, and its intervals overlap the threshold's on most.
*Source: `results/tables/ladder_per_dataset.csv` (view = all; rungs `baseline_cv` and `elastic_net`). Dataset X excluded (n = 28).*

**Table 4.** MCC medians across datasets for each model. No model beats the threshold consistently, and searching the descriptor set for the best single descriptor or pair does worse than the threshold.

| Model | Datasets | Median MCC, leave-one-out | Median MCC, repeated 5-fold | Datasets ≥ 0.10 better than baseline | Datasets ≥ 0.10 worse |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random forest | 10 | 0.59 | 0.58 | 5 | 0 |
| %V<sub>bur</sub>(min) + %V<sub>bur</sub>(Boltz), logistic | 11 | 0.57 | 0.48 | 2 | 4 |
| **Elastic net (primary)** | 10 | **0.49** | **0.48** | **3** | **2** |
| %V<sub>bur</sub>(min) alone, logistic | 11 | 0.48 | 0.47 | 1 | 3 |
| **Threshold on %V<sub>bur</sub>(min) (baseline)** | 11 | **0.47** | **0.46** | – | – |
| Gradient boosting | 10 | 0.46 | 0.47 | 4 | 1 |
| Best pair (17,020 searched) | 10 | 0.43 | 0.43 | 4 | 4 |
| Best single descriptor (185 searched) | 11 | 0.27 | 0.37 | 1 | 7 |

*Source: `results/tables/ladder_per_dataset.csv` (view = all).*

**Table 5.** Evaluation of H2 and H3. Neither adding %V<sub>bur</sub>(Boltz) (H2) nor switching to minimum-%V<sub>bur</sub>-conformer values (H3) helps consistently; which conformer summary works best depends on the dataset.

| Dataset | H2: %V<sub>bur</sub>(min) alone | H2: + %V<sub>bur</sub>(Boltz) | H2 Δ (bar +0.05) | H3: Boltzmann view | H3: min-%V<sub>bur</sub> conformer view | H3 Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| I | 0.52 | 0.58 | **+0.06** | 0.64 | 0.65 | +0.01 |
| II | 0.42 | 0.41 | −0.01 | 0.53 | 0.50 | −0.04 |
| III | 0.48 | 0.58 | **+0.10** | 0.58 | 0.58 | 0.00 |
| IV | 0.40 | 0.43 | +0.03 | 0.40 | 0.33 | −0.07 |
| V | 0.45 | 0.37 | −0.07 | 0.68 | 0.64 | −0.05 |
| VII | – | – | – | 0.05 | 0.13 | +0.07 |
| VIII | – | – | – | 0.59 | 0.51 | −0.08 |
| IX | – | – | – | 0.57 | 0.29 | −0.28 |
| XI | – | – | – | 0.43 | 0.71 | +0.28 |
| XII | – | – | – | 0.21 | 0.43 | +0.22 |
| **Count** | | | **2 of 5** | | | **4 of 10 favour min-%V<sub>bur</sub> conformer** |

*H2 is defined on the Ni datasets only. Δ values are computed from unrounded MCCs. Source: `results/tables/hypotheses_h1_h3.md`.*

![Figure 2: median MCC under repeated 5-fold vs leave-one-class-out for four models](figures/fig2_seen_vs_unseen_ligand_classes.png)

**Figure 2.** Model accuracy on ligand classes seen vs. not seen in training. The more flexible the model, the more it loses on ligand classes it has not seen. The threshold loses nothing, while the random forest loses 0.40 MCC.
*Source: `results/tables/split_summary_per_dataset.csv`.*

**Table 6.** MCC on pooled held-out predictions. Pooled across reactions, the threshold is the most robust model on unseen ligand classes. The multivariate models' high leave-dataset-out scores in the Ni pool come from shared ligands, not transfer.

| Pool | Validation | Threshold | %V<sub>bur</sub>(min) + %V<sub>bur</sub>(Boltz) | Elastic net | Random forest |
| --- | --- | ---: | ---: | ---: | ---: |
| Ni (I–V) | 5-fold, grouped by ligand | 0.42 | 0.41 | 0.45 | **0.56** |
| Ni (I–V) | Leave one class out | 0.39 | **0.44** | 0.20 | 0.34 |
| Ni (I–V) | Leave one dataset out† | 0.44 | 0.42 | 0.64 | **0.73** |
| Pd (VII–XII) | 5-fold, grouped by ligand | 0.49 | 0.34 | 0.39 | **0.56** |
| Pd (VII–XII) | Leave one class out | **0.46** | 0.07 | 0.20 | 0.27 |
| Pd (VII–XII) | Leave one dataset out | **0.49** | 0.39 | 0.41 | 0.47 |
| All 11 (+ metal flag) | 5-fold, grouped by ligand | 0.10 | 0.11 | 0.22 | **0.44** |
| All 11 (+ metal flag) | Leave one class out | 0.06 | −0.12 | −0.01 | **0.27** |
| All 11 (+ metal flag) | Leave one dataset out | 0.05 | 0.08 | 0.36 | **0.60** |

Bold = best in row. † II–V share the same 90 ligands, so holding out one of them still leaves every ligand in training.
*Sources: `results/tables/split_summary_pooled.csv`, `split_strategies.csv`.*

**Table 7.** Descriptor stability across datasets. A descriptor is considered stable if it is nonzero in more than 80% of 500 bootstrap refits, with the same sign in at least 90% of those. The most consistently selected descriptors are buried-volume variants, and their sign flips between Ni and Pd exactly as the threshold does. The model largely re-finds the steric cliff.

| Descriptor (Kraken column) | Family | Stable in: smaller value → active (−) | Stable in: larger value → active (+) |
| --- | --- | --- | --- |
| Buried volume, max octant (`vbur_ovbur_max_boltz`) | Steric | III, IV (Ni) | VIII, IX, XI (Pd) |
| Buried volume, near region (`vbur_near_vbur_boltz`) | Steric | I, II, IV (Ni) | VIII, IX (Pd) |
| Buried volume, min quadrant (`vbur_qvbur_min_boltz`) | Steric | II, IV, V (Ni) | – |
| Buried Sterimol B1, min (`sterimol_burB1_min`) | Steric | I, II, III (Ni) | – |
| Sterimol B1, conformer range (`sterimol_B1_delta`) | Steric | – | II, III, IV (Ni) |
| Pyramidalisation at P, min (`pyr_P_min`) | Steric | XI (Pd) | II, III, IV (Ni) |
| NBO P–C antibonding energy, avg (`nbo_bds_e_avg_boltz`) | Electronic | – | II, III, IV, V (Ni) |
| Quadrupole tensor, yy (`qpoletens_yy_boltz`) | Electrostatic | II, III, IV (Ni) | XII (Pd) |
| Dipole moment, conformer range (`dipolemoment_delta`) | Electrostatic | II, III (Ni), VII (Pd) | – |

*Sources: `results/tables/stability_selection.csv`, `stability_clusters.csv`, `best_pair_choices.csv`.*

**Table 8.** Comparison of model accuracy on the external test (JACS 2022 Reaction III, 43 ligands, 33 of them in the Ni training pool). On a reaction it never saw, the frozen elastic net performs at chance (MCC −0.00), while both %V<sub>bur</sub>(min) thresholds find all 12 actives (MCC 0.41–0.46).

| Model | MCC [95% CI] | Balanced accuracy | Actives found | Inactives called active |
| --- | --- | ---: | ---: | ---: |
| Published threshold | 0.46 [0.30, 0.62] | 0.74 | 12 of 12 | 16 of 31 |
| Threshold refitted on Ni pool | 0.41 [0.27, 0.56] | 0.71 | 12 of 12 | 18 of 31 |
| Frozen elastic net (44 of 185 descriptors) | −0.00 [−0.29, 0.31] | 0.50 | 5 of 12 | 13 of 31 |

| Paired comparison, all 43 ligands | MCC difference [95% CI] | Interval excludes zero? |
| --- | --- | --- |
| Elastic net − published threshold | −0.46 [−0.74, −0.14] | Yes: elastic net worse |
| Elastic net − Ni-pool threshold | −0.41 [−0.69, −0.10] | Yes: elastic net worse |
| Ni-pool threshold − published threshold | −0.05 [−0.12, 0.00] | No |

*Sources: `results/tables/external_test.csv`, `external_test_differences.csv`, `external_errors.csv`; model `models/frozen_ni_pool.json`, tag `frozen`.*

![Figure 3: structures of the 12 active ligands in JACS 2022 Reaction III, labelled with %V_bur(min) and whether each model found them](figures/fig3_external_test_actives.png)

**Figure 3.** The 12 actives in JACS 2022 Reaction III, each labelled with its %V<sub>bur</sub>(min) and whether each model found it. The elastic net misses 7 of the 11 Buchwald-type actives, all sitting just under the threshold at %V<sub>bur</sub>(min) 29.6–31.7, and the threshold finds every one.
*Source: `results/figures/external_actives.png`.*

**Table 9.** 3 of 20 control tests beat the permutation null at p < 0.05, about what chance alone would produce, and none survives a post hoc Bonferroni correction. Each cell gives the stratified 5-fold MCC and the permutation p-value (500 permutations, each rerunning the full procedure; smallest possible p ≈ 0.002). Bonferroni cut for 20 tests: p < 0.0025. Multivariate models were not fitted, because no control met the size rule.

| Control | n (active) | Threshold (CV) | Best single | %V<sub>bur</sub>(min) logistic | %V<sub>bur</sub>(min) + %V<sub>bur</sub>(Boltz) |
| --- | --- | --- | --- | --- | --- |
| RS2 (VII at 1:1 L:Pd) | 30 (23) | 0.33, p = 0.10 | 0.12, p = 0.24 | −0.12, p = 0.67 | −0.26, p = 0.87 |
| RS3 (VIII at 1:1 L:Pd) | 29 (23) | **0.64, p = 0.004** | 0.04, p = 0.42 | **0.44, p = 0.028** | 0.30, p = 0.14 |
| XIII | 79 (75) | 0.21, p = 0.064 | −0.08, p = 0.92 | 0.18, p = 0.086 | 0.06, p = 0.36 |
| XIV | 14 (7) | −0.45, p = 0.93 | 0.14, p = 0.40 | −0.29, p = 0.75 | 0.32, p = 0.17 |
| JACS 2022 IV | 16 (10) | 0.33, p = 0.20 | 0.10, p = 0.42 | −0.29, p = 0.84 | **0.68, p = 0.016** |

Bold = p < 0.05.
*Source: `results/tables/negative_controls_h4.csv`.*

**Table 10.** Leave-one-ligand-out predictions pooled over the ten eligible primary datasets; one prediction per ligand per dataset. Within a reaction, the elastic net's gain comes almost entirely from triaryls and biaryls, where the threshold calls too many inactive ligands active. Trialkyl and P-heteroatom ligands defeat both models.

| Ligand class | Predictions | Elastic net error | Threshold error | Elastic net: actives missed | Threshold: inactives called active | Elastic net: inactives called active |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Biaryl (Buchwald-type) | 170 | **13.5%** | 21.2% | 15.0% | 31.1% | 12.2% |
| Alkyl-aryl | 132 | 18.9% | **16.7%** | 11.0% | 39.0% | 36.6% |
| Triaryl | 136 | **31.6%** | 48.5% | 34.7% | 73.6% | 29.9% |
| Trialkyl | 164 | **31.7%** | 32.9% | 28.0% | 51.7% | 34.8% |
| P-heteroatom | 31 | **48.4%** | 51.6% | 87.5% | 60.9% | 34.8% |

Bold = the lower error of the two models.
*Sources: `results/tables/error_by_class.csv`, `ligand_errors.csv`, `results/worst_ligand_report.md`.*

![Figure 4: structures of the 16 most often misclassified ligands, with elastic-net and threshold error counts](figures/fig4_worst_predicted_ligands.png)

**Figure 4.** The 16 ligands with the most elastic-net errors across datasets, with the elastic-net and threshold error counts under each. The most often misclassified ligands are small, inactive non-biaryls with %V<sub>bur</sub>(min) 21–29 in the Ni reactions II–V, and both models get them wrong together.
*Source: `results/figures/worst_ligands.png`.*

**Table 11.** Hypothesis verdicts at a glance. None of the five pre-registered hypotheses was supported, and on the external test the single descriptor transfers better than the 185-descriptor model.

| Hypothesis | Pre-registered test | Result | Verdict |
| --- | --- | --- | --- |
| H1 (primary) | Elastic net beats the cross-validated baseline by ≥ 0.10 MCC on a majority of the 10 eligible datasets | 3 of 10 (3 of 7 without VII, VIII, IX) | Not supported |
| H2 | %V<sub>bur</sub>(min) + %V<sub>bur</sub>(Boltz) beats %V<sub>bur</sub>(min) alone by ≥ 0.05 MCC on Ni datasets I–V | 2 of 5 | Not supported |
| H3 | Minimum-%V<sub>bur</sub>-conformer values beat Boltzmann averages in the elastic net | 4 of 10 (Boltzmann 5, 1 tie) | Not supported |
| H4 | No model beats a label-permutation null on the negative controls | 3 of 20 tests p < 0.05 (none after post hoc Bonferroni) | Not supported as written |
| H5 | Frozen Ni-pool model and thresholds, scored once on JACS 2022 Reaction III | Elastic net −0.00 vs published threshold 0.46; difference −0.46 [−0.74, −0.14] | Elastic net transfers worse |

*Sources: `results/tables/hypotheses_h1_h3.md`, `negative_controls_h4.csv`, `external_test.csv`, `external_test_differences.csv`.*

## Error analysis

Per-ligand leave-one-out predictions were recomputed for the elastic net and the baseline with the same code and seed. Each dataset's recomputed MCC was required to equal the original run's exactly. Errors were then tabulated per ligand and per class. For the external set, per-ligand predictions were recomputed from the frozen JSON and checked against the stored confusion matrix; nothing was refitted or rescored.

## Discussion

All of the primary datasets were modest in size (28–90 ligands) and reasonably balanced, so MCC serves as a fair metric for them (Table 2). It is important to note that these datasets are not independent: II–V are the same 90 ligands screened under four sets of conditions, and 122 of the 194 ligands appear in more than one dataset (54 in five or more). The controls are small and, in XIII's case, almost all active, which limits what they can show (Table 9). Of the 190 Kraken descriptors, 5 were dropped as redundant or near-constant, and the full set collapses into 108 correlation clusters.

Next, I assessed whether each dataset reproduced the published baseline values; only 8 of the 11 sets passed the gate (Table 3). Regardless, I proceeded with all eleven datasets as extracted, and pre-declared a sensitivity analysis without VII, VIII and IX before any model was fitted. The thresholds also show the Ni/Pd split that shapes later results: on Ni (sets I–V) actives fall below the threshold, and on Pd (sets VII–XII) above it.

With my analysis of individual datasets complete, I turned to applying multivariate models to each. H1 postulates that an elastic-net model will beat the baseline by at least 0.10 MCC on the majority of datasets. However, Figure 1 shows that only 3 sets meet this bar, so H1 is not supported. In the sensitivity analysis that excludes sets VII, VIII and IX, the elastic net still wins only a minority (3 of 7). The median MCC is 0.49 for the elastic net and 0.47 for the threshold. The gains are concentrated in three datasets and offset by two clear losses (IX, VII), with the rest close to zero (Figure 1). The largest gain is on dataset XI, but a single descriptor, pyramidalisation at P (`pyr_P_min`), was chosen in 100% of folds by the best-single-descriptor model and matches the elastic net there on its own (MCC 0.86). XI is therefore better read as "a different single descriptor wins" than as a multivariate success.

The elastic net and the threshold have nearly the same median, so the multivariate model buys almost nothing on a typical dataset. Table 4 shows that searching all 185 descriptors for the best single one gives the worst result in my model ladder: with 30–90 ligands, the search mostly finds descriptors that fit the training folds by chance. The best-pair search is erratic, ranging from 0.94 on I to 0.05 on XI. The random forest tops the table, but Figure 2 shows that its lead disappears on ligand classes it has not seen. For the fixed pair, the large gap between leave-one-out (0.57) and repeated 5-fold (0.48) reflects how noisy leave-one-out estimates are on small datasets.

H2 is defined on the Ni datasets only, while H3 compares elastic-net models on the two feature views. H2 was inspired by the JACS 2022 study [4], where %V<sub>bur</sub>(Boltz) separated Buchwald-type ligands that %V<sub>bur</sub>(min) could not. In datasets I–V the extra term helps on I and III and hurts on V, likely because cliffs I–V contain fewer Buchwald-type biaryls than the JACS 2022 sets, leaving %V<sub>bur</sub>(Boltz) less to separate. The H3 differences are large only on the small Pd datasets (IX, XI, XII, n = 30–71), and they point in both directions, indicating noise more than any real preference (Table 5). Thus, neither H2 nor H3 is supported.

In my random splits, every ligand class appears in training, so a model can do well by recognising ligands similar to ones it has seen. Figure 2 shows that holding out a whole class (for example, all triaryls) removes that aid and worsens the multivariate models' accuracy. The threshold scores the same either way, however, because it encodes one physical cut-off rather than memorising ligand families. The random forest, which tops Table 4, falls furthest between its repeated 5-fold and leave-one-class-out performance. Its extra 5-fold accuracy came from interpolating between familiar ligands rather than learning anything that extrapolates.

Table 6 shows that, pooled across reactions, the threshold is the most robust model on unseen ligand classes. As anticipated by the deviation, the all-11 pool fails: the threshold collapses to about 0.05–0.10 because Ni actives sit below the cut-off and Pd actives above it. Only the random forest, which can split on the metal flag, partly recovers on the all-11 pool. The Ni-pool leave-dataset-out scores look strong, at 0.64 for the elastic net and 0.73 for the random forest. Though this might suggest the multivariate models carry over to new reactions, they don't: II–V share their ligands, and the genuinely new reaction in Table 8 shows the opposite.

Buried-volume variants are the descriptors most consistently selected as carrying any extra signal (Table 7), and their sign flips between Ni and Pd exactly where the published threshold changes direction. Across the ten datasets, 112 descriptor–dataset picks were stable. By family they mirror what Kraken contains: steric 59% of stable picks vs 62% of candidates, electronic 22% vs 21%, electrostatic 17% vs 14%. Thus, no family is over-represented, and "sterics dominate" would only describe the descriptor set. On II, III and IV the best pair was surface area + total buried volume in 80–100% of folds, suggesting that the amount of ligand bulk near the metal relative to its total size may matter for the Ni catalyst, in line with earlier work on remote steric effects in Ni catalysis [6]. As a reminder, II–IV share their ligands, so this could equally be a feature of that particular ligand set.

I then tested the frozen elastic net on a reaction type not contained within the training set. On this reaction, the frozen elastic net performs no better than chance (balanced accuracy 0.50, MCC −0.00; Table 8). The failure is not driven by unfamiliar ligands either: 4 of the 7 actives the model missed were in its training data. Ultimately, the frozen elastic net fails to carry any information beyond the steric cliff to a new reaction. Interestingly, each active that the frozen elastic net incorrectly labelled as inactive carries a substituent on the lower ring ortho to the biaryl bond: OMe, NMe<sub>2</sub>, Me, i-Pr or Oi-Pr (Figure 3). None of the four biaryl actives the model found has one there. With 11 ligands, this is a lead for follow-up rather than a finding.

Only 3 of 20 control tests beat the permutation null at p < 0.05 (Table 9). Two of the three hits are on RS3, which is the Pd reaction VIII run at a 1:1 ligand-to-Pd ratio, and the third is the fixed pair on JACS 2022 IV (n = 16). As written, H4 is not supported. However, three hits in 20 tests is roughly what chance alone would produce, none survives a Bonferroni correction, and a ligation-state descriptor working at a different L:Pd ratio (RS3) is chemically plausible [1, 7]. Nothing suggests the primary-dataset results are artefacts of overfitting.

Table 10 helps to explain where these models fail. The threshold's main weakness is small, inactive triaryls that sit below the steric cliff: it calls 74% of them active. The elastic net cuts that to 30%, but at the cost of missing 35% of active triaryls. Biaryls are the easiest class within a reaction. Set beside Figure 3, this suggests that what the model learns about biaryls is specific to the reaction and does not apply outside it. Figure 4 shows the 16 most frequently misclassified ligands, many of which are the small trialkyls and triaryls discussed above. Several of these probably fail for reasons a free-ligand DFT descriptor cannot capture. PMe<sub>3</sub>·HBF<sub>4</sub> is a phosphonium salt, while Kraken describes free PMe<sub>3</sub>. PTA has basic amine nitrogens and is water-soluble. P(allyl)<sub>3</sub> carries reactive alkenes, and the fluorous triaryl has unusual solubility.

## Limitations and next steps

The largest limitation I encountered was available data. I could only locate small, dependent datasets to train on, which caused wide intervals, noisy estimates and counts of "datasets won" that overstate the evidence. The ligands in these datasets were also restrictive, occupying a narrow electronic range. It is likely that if these models were trained on datasets that included more electron-poor ligands, electronic descriptors would have a greater impact.

I hope to get my hands on more extensive datasets and give this process another go!

## References

1. Newman-Stonebraker, S. H.; Smith, S. R.; Borowski, J. E.; Peters, E.; Gensch, T.; Johnson, H. C.; Sigman, M. S.; Doyle, A. G. Univariate Classification of Phosphine Ligation State and Reactivity in Cross-Coupling Catalysis. *Science* **2021**, *374* (6565), 301–308. https://doi.org/10.1126/science.abj4213
2. Newman-Stonebraker, S.; Smith, S.; Borowski, J.; Peters, E.; Gensch, T.; Johnson, H.; Sigman, M.; Doyle, A. Linking Mechanistic Analysis of Catalytic Reactivity Cliffs to Ligand Classification. *ChemRxiv* **2021**. https://doi.org/10.26434/chemrxiv.14388557.v1
3. Gensch, T.; dos Passos Gomes, G.; Friederich, P.; Peters, E.; Gaudin, T.; Pollice, R.; Jorner, K.; Nigam, A.; Lindner-D'Addario, M.; Sigman, M. S.; Aspuru-Guzik, A. A Comprehensive Discovery Platform for Organophosphorus Ligands for Catalysis. *J. Am. Chem. Soc.* **2022**, *144* (3), 1205–1217. https://doi.org/10.1021/jacs.1c09718
4. Newman-Stonebraker, S. H.; Wang, J. Y.; Jeffrey, P. D.; Doyle, A. G. Structure–Reactivity Relationships of Buchwald-Type Phosphines in Nickel-Catalyzed Cross-Couplings. *J. Am. Chem. Soc.* **2022**, *144* (42), 19635–19648. https://doi.org/10.1021/jacs.2c09840
5. Ess, D.; Kirkland, J.; Stevens, H.; Olsen, J. Tolman Electronic Parameter Predictions from a Fast, Accurate, and Robust Machine Learning Model Provide Insight into Phosphine Ligand Electronic Effects. *ChemRxiv* **2023**. https://doi.org/10.26434/chemrxiv-2023-wjnpj
6. Wu, K.; Doyle, A. G. Parameterization of Phosphine Ligands Demonstrates Enhancement of Nickel Catalysis via Remote Steric Effects. *Nat. Chem.* **2017**, *9* (8), 779–784. https://doi.org/10.1038/nchem.2741
7. Borowski, J. E.; Newman-Stonebraker, S. H.; Doyle, A. G. Comparison of Monophosphine and Bisphosphine Precatalysts for Ni-Catalyzed Suzuki–Miyaura Cross-Coupling: Understanding the Role of the Ligation State in Catalysis. *ACS Catal.* **2023**, *13* (12), 7966–7977. https://doi.org/10.1021/acscatal.3c01331
8. Zou, H.; Hastie, T. Regularization and Variable Selection via the Elastic Net. *J. R. Stat. Soc. Ser. B Stat. Methodol.* **2005**, *67* (2), 301–320. https://doi.org/10.1111/j.1467-9868.2005.00503.x
9. Meinshausen, N.; Bühlmann, P. Stability Selection. *J. R. Stat. Soc. Ser. B Stat. Methodol.* **2010**, *72* (4), 417–473. https://doi.org/10.1111/j.1467-9868.2010.00740.x
10. Chicco, D.; Jurman, G. The Advantages of the Matthews Correlation Coefficient (MCC) over F1 Score and Accuracy in Binary Classification Evaluation. *BMC Genomics* **2020**, *21*, 6. https://doi.org/10.1186/s12864-019-6413-7
11. Ojala, M.; Garriga, G. C. Permutation Tests for Studying Classifier Performance. In *2009 Ninth IEEE International Conference on Data Mining*; IEEE: Miami Beach, FL, USA, 2009; pp 908–913. https://doi.org/10.1109/ICDM.2009.108
12. Cawley, G. C.; Talbot, N. L. C. On Over-Fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. *J. Mach. Learn. Res.* **2010**, *11* (70), 2079–2107.
13. Breiman, L. Random Forests. *Mach. Learn.* **2001**, *45* (1), 5–32. https://doi.org/10.1023/A:1010933404324
14. Ahneman, D. T.; Estrada, J. G.; Lin, S.; Dreher, S. D.; Doyle, A. G. Predicting Reaction Performance in C–N Cross-Coupling Using Machine Learning. *Science* **2018**, *360* (6385), 186–190. https://doi.org/10.1126/science.aar5169
15. Chuang, K. V.; Keiser, M. J. Comment on "Predicting Reaction Performance in C–N Cross-Coupling Using Machine Learning." *Science* **2018**, *362* (6416), eaat8603. https://doi.org/10.1126/science.aat8603
16. Kraken v2 descriptor library. https://descriptor-libraries.molssi.org/kraken/ (accessed 2026-11-26)
