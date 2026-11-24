# Pre-registration

## 1. Question

Across the reactivity-cliffs datasets, a single steric descriptor, %V_bur(min), classifies monodentate
phosphines as active or inactive at a threshold near 32% (Newman-Stonebraker et al., *Science* 2021).
Does a multivariate model over Kraken's 190 DFT descriptors classify better, and if so, which descriptors
carry the extra signal?

## 2. Data

| Role | Datasets | Source |
| --- | --- | --- |
| Primary (the eleven) | cliffs I, II, III, IV, V, VII, VIII, IX, X, XI, XII | cliffs SI Tables S1, S2, S9–S12, S15, S16 |
| Supplementary | cliffs RS1 | cliffs SI Table S2 |
| Negative controls (no published threshold) | cliffs RS2, RS3 (VII and VIII at 1:1 L:Pd), cliffs XIII, XIV; JACS 2022 IV | cliffs SI S13, S14, S17, S18; JACS 2022 SI S3 |
| External test, scored once | JACS 2022 III (Ni C–N coupling, 43 ligands) | JACS 2022 SI S2 |

Descriptors: Kraken 2025 release, 190 DFT descriptors, joined to SI rows by structure via the 2021 IDs the
SI uses (step 4 of the workflow). Ligand counts, actives and inactives at each dataset's published yield
cutoff (active = yield >= cutoff):

| Dataset | n | Cutoff (% yield) | Active | Inactive |
| --- | --- | --- | --- | --- |
| I | 34 | 10 | 19 | 15 |
| II | 90 | 10 | 38 | 52 |
| III | 90 | 5 | 28 | 62 |
| IV | 90 | 5 | 32 | 58 |
| V | 90 | 20 | 52 | 38 |
| VII | 55 | 10 | 29 | 26 |
| VIII | 53 | 10 | 27 | 26 |
| IX | 30 | 10 | 20 | 10 |
| X | 28 | 5 | 15 | 13 |
| XI | 30 | 30 | 19 | 11 |
| XII | 71 | 10 | 39 | 32 |
| JACS 2022 III (external) | 43 | 10 | 12 | 31 |

JACS 2022 Reactions I, Ib and II re-screen Ni Suzuki–Miyaura chemistry already covered by cliffs II–V.
Thus, we exclude them from our study.

## 3. Outcome and metric

- Outcome: active or inactive at each dataset's published cutoff (SI Table S8; JACS 2022 Table S4).
  Yield regression is secondary and reported only per dataset.
- Primary metric: Matthews correlation coefficient (MCC), as in the source papers. Secondary: balanced
  accuracy. Every metric is reported with n and with the spread across resamples.

## 4. Baseline, in two forms

1. **Published, in sample.** A single-node decision tree on %V_bur(min), balanced class weight, fitted to
   all of each dataset (SigmanGroup/Threshold code). This is what Table S8 reports, and it is what the
   gate checks.
2. **Cross-validated.** The same classifier with its threshold refitted inside each training fold. The
   published MCCs are in-sample, so every multivariate model is compared with this form, never with
   the published numbers. Comparing a cross-validated model with an in-sample baseline would flatter
   the model.

## 5. Gate (29 November)

The baseline reproduces if, for at least 10 of the 11 primary datasets, the refitted threshold is within
0.11 %V_bur(min) of Table S8 and the MCC is within 0.03. Otherwise stop and follow the kill criteria.

## 6. Models and validation

- Ladder, every rung reported: published threshold; best single descriptor; best pair (all ~18,000,
  search size reported); elastic-net logistic regression (primary multivariate model); random forest
  and gradient boosting as nonlinear references.
- Features standardised inside each training fold. Hyperparameters by nested cross-validation.
- Per dataset: leave-one-ligand-out (primary) and repeated 5-fold (reported). Pooled across the eleven:
  leave-one-dataset-out, grouped by ligand identity, and leave-one-ligand-class-out.
- Multivariate fits only where n >= 30 and the smaller class has >= 10 ligands. That keeps I, II, III,
  IV, V, VII, VIII, IX, XI and XII and leaves X (n = 28) to the univariate comparison.

## 7. Hypotheses, stated before any result

- **H1 (primary).** The cross-validated elastic-net model beats the cross-validated baseline by at least
  0.10 MCC (leave-one-ligand-out) on a majority of the eligible primary datasets.
- **H2.** On the Ni datasets (I–V), logistic regression on %V_bur(min) + %V_bur(Boltz) beats %V_bur(min)
  alone by at least 0.05 MCC. Source: JACS 2022, where %V_bur(Boltz) separated Buchwald-type ligands
  that %V_bur(min) could not.
- **H3.** For the 28 properties Kraken summarises five ways, the minimum-V_bur-conformer values
  (`_vburminconf`) outperform the Boltzmann averages in the elastic-net model.
- **H4.** On the negative controls, no model beats a label-permutation null (500 permutations,
  p < 0.05). A model that does is treated as overfitting until shown otherwise.
- **H5.** On the external test, the frozen primary model and the baseline at the published 31.89%
  threshold are each scored once, after the `frozen` tag.

**Success criterion:** H1. H2–H5 are reported whatever they show. A result that "the single descriptor
is as good as anything" is a finding, not a failure.

## 8. Things that are fixed now

- Random seed 20261125 for every resampling step.
- Stability selection: 500 bootstrap refits of the elastic-net model; a descriptor counts as selected
  above 80% selection frequency with a consistent sign.
- The external test set is not loaded by any modelling code until after the `frozen` tag.

## 8a. Frozen model and external test (H5)

Added after the `prereg` tag and before any code loaded the external set (deviation log, 11/24/2026).
H5 named "the frozen primary model" without saying what it is trained on; this section fixes that.

- **Training data.** The Ni pool: cliffs I–V, every included ligand, each labelled at its own dataset's
  yield cutoff. A ligand tested in several reactions contributes one row per reaction: 394 rows,
  103 ligands, 169 active.
- **Model.** The elastic-net logistic regression of section 6, on all 185 descriptors, with the same grid.
  Its settings are chosen by 5-fold cross-validation grouped by ligand, so no ligand sits on both sides of
  an inner split, and it is then refitted on the whole pool. Seed 20261125.
- **Comparators.** (a) The published threshold: active if %V_bur(min) <= 31.89, as JACS 2022 Table S4
  applies it to Reaction III. (b) A %V_bur(min) threshold refitted on the same Ni pool, the like-for-like
  comparator for the frozen model.
- **What is frozen.** `models/frozen_ni_pool.json`, written by `make freeze`: every number needed to
  predict (descriptor list, scaling, coefficients, both thresholds), checksums of the data files and the
  package versions. The `frozen` tag fixes it together with the scoring code, `src/ligsel/external.py`.
- **External set.** JACS 2022 Reaction III, 43 ligands, active = yield >= 10% (Table S4). 33 of the 43
  ligands are in the Ni pool, so this tests transfer to a new reaction, mostly with familiar ligands.
- **Reported.** For each model: MCC with a bootstrap 95% interval over ligands (2,000 resamples), balanced
  accuracy and the confusion matrix, on (1) all 43 ligands (primary), (2) the 10 ligands not in the Ni
  pool and (3) the Buchwald-type ligands. Differences between models get a paired bootstrap 95% interval.
  Subsets (2) and (3) are descriptive only.
- **Reading the result.** H5 is a check on transfer, not a test of H1. The frozen model is said to
  transfer better than a comparator only if the 95% interval of their difference on all 43 ligands
  excludes zero; otherwise the result is reported as no detectable difference.
- **Once.** The scoring code refuses to run without the tag, if the model or the code changed after the
  tag, if the data files changed, or if a result already exists. Any second run is a deviation.

## 9. Deviations log

Any changes made after the `prereg` tag are provided here with accompanying date and reason.

| Date | Change | Reason |
| --- | --- | --- |
| 11/23/2026 | Gate not met by the letter (8 of 11); proceeding with all eleven datasets. H1 additionally reported excluding VII, VIII and IX as a sensitivity analysis. | Extracted data hand-checked and match the SI; the three mismatches trace to differences between SI tables and the published Table S8 (see docs/signoffs.md). Added before any model was fitted. |
| 11/24/2026 | Pooled analyses also run on a Ni-only pool (I–V) and a Pd-only pool (VII–XII), alongside the pre-registered pool of all eleven combined. The all-eleven pool includes a metal indicator feature (Ni = 1). | The published thresholds point in opposite directions on Ni and Pd, so a single pooled model mixes two opposite trends. Added in step 18 before any pooled result was seen. |
| 11/24/2026 | Leave-one-ligand-class-out (LOOCV) also run within each dataset, not only on the pools. | Tests whether per-dataset models extrapolate to ligand classes they have not seen. Added in step 18 before any result was seen; reported as secondary. |
| 11/24/2026 | H4 permutation tests use stratified 5-fold CV (5x5) inside each of the 500 permutations, not leave-one-out. Only the baseline_cv, best_single, single_min_lr and pair_min_boltz models are tested. | LOO across 500 permutations is too slow for the best_single search. LOO MCC on the real labels is still reported. No control dataset meets section 6's eligibility rule for multivariate fits (every control has fewer than 10 ligands in its minority class), so the elastic net and the nonlinear models are not fitted on the controls. |
| 11/24/2026 | H4 is scored as written (any p < 0.05 counts against it) and 3 of 20 tests have p < 0.05, failing. A Bonferroni correction across the 20 tests is reported alongside as a post hoc analysis, not as the H4 result. | Multiple-comparison correction was not pre-registered. Reporting it as the verdict after seeing the p-values would be moving the goalposts. |
| 11/24/2026 | Gradient-boosting reference refitted with min_samples_leaf searched over [1, 3, 5]. | The library default (20) blocked every split below 40 training ligands, so the model predicted a constant on 9 of 10 datasets. Nonlinear reference only; H1–H3 and the primary model are unaffected. |
| 11/24/2026 | Section 8a added: the external-test model is trained on the Ni pool (I–V) with settings chosen by ligand-grouped CV; a threshold refitted on the Ni pool is added as a second comparator; H5 is read as descriptive, with a pre-stated rule. | H5 did not say what "the frozen primary model" is trained on. Added before any code loaded the external set. |
