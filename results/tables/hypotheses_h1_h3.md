# Hypotheses H1-H3 (leave-one-ligand-out MCC)

- H1_datasets_meeting: 3 of 10
- H1_supported: False
- H1_sensitivity_meeting: 3 of 7
- H1_sensitivity_supported: False
- H2_datasets_meeting: 2 of 5
- H3_vburminconf_better: 4 of 10

## H1

| dataset     |   baseline_cv |   elastic_net |   delta | meets_0.10   |
|:------------|--------------:|--------------:|--------:|:-------------|
| cliffs_III  |         0.388 |         0.658 |   0.27  | True         |
| cliffs_II   |         0.462 |         0.469 |   0.007 | False        |
| cliffs_IV   |         0.413 |         0.436 |   0.023 | False        |
| cliffs_IX   |         0.55  |         0.342 |  -0.208 | False        |
| cliffs_I    |         0.466 |         0.523 |   0.056 | False        |
| cliffs_VIII |         0.552 |         0.51  |  -0.042 | False        |
| cliffs_VII  |         0.309 |         0.048 |  -0.261 | False        |
| cliffs_V    |         0.525 |         0.634 |   0.109 | True         |
| cliffs_XII  |         0.316 |         0.399 |   0.083 | False        |
| cliffs_XI   |         0.558 |         0.856 |   0.298 | True         |

## H1 sensitivity (without VII, VIII, IX)

| dataset    |   baseline_cv |   elastic_net |   delta | meets_0.10   |
|:-----------|--------------:|--------------:|--------:|:-------------|
| cliffs_III |         0.388 |         0.658 |   0.27  | True         |
| cliffs_II  |         0.462 |         0.469 |   0.007 | False        |
| cliffs_IV  |         0.413 |         0.436 |   0.023 | False        |
| cliffs_I   |         0.466 |         0.523 |   0.056 | False        |
| cliffs_V   |         0.525 |         0.634 |   0.109 | True         |
| cliffs_XII |         0.316 |         0.399 |   0.083 | False        |
| cliffs_XI  |         0.558 |         0.856 |   0.298 | True         |

## H2

| dataset    |   min_alone |   min_plus_boltz |   delta | meets_0.05   |
|:-----------|------------:|-----------------:|--------:|:-------------|
| cliffs_I   |       0.523 |            0.581 |   0.058 | True         |
| cliffs_II  |       0.421 |            0.412 |  -0.009 | False        |
| cliffs_III |       0.48  |            0.58  |   0.101 | True         |
| cliffs_IV  |       0.4   |            0.431 |   0.031 | False        |
| cliffs_V   |       0.446 |            0.373 |  -0.073 | False        |

## H3

| dataset     |   boltz |   vburminconf |   delta |
|:------------|--------:|--------------:|--------:|
| cliffs_III  |   0.578 |         0.578 |   0     |
| cliffs_II   |   0.533 |         0.495 |  -0.038 |
| cliffs_IV   |   0.401 |         0.334 |  -0.067 |
| cliffs_IX   |   0.566 |         0.289 |  -0.277 |
| cliffs_I    |   0.642 |         0.652 |   0.01  |
| cliffs_VIII |   0.586 |         0.509 |  -0.076 |
| cliffs_VII  |   0.052 |         0.125 |   0.073 |
| cliffs_V    |   0.681 |         0.636 |  -0.046 |
| cliffs_XII  |   0.208 |         0.429 |   0.22  |
| cliffs_XI   |   0.426 |         0.709 |   0.284 |
