# Checkpoint sign-offs

Each manual checkpoint from our to-do list.
Claude does not fill in this file.

## Step 8: spot-check of the extracted SI data

- Date: 11/23/2026
- Checked: `data/interim/spotcheck.csv` against the SI PDFs, and `manual/jacs2022_SI_tableS2_pageS12.csv` against JACS 2022 SI page S12.
- Decision: APPROVE W/ NOTES
    NOTES: Thorough Claude checking confirmed the datasets were transcribed correctly.

## Step 10: join exclusions and match rates

- Date: 11/23/2026
- Checked: `results/tables/join_report.csv` and `results/tables/join_exclusions.csv`.
- Decision: APPROVE W/ NOTES
    NOTES: There were often multiple Kraken entries for a single molecule, so only a single data 
             set could be utilized.


## Step 15: descriptor family table and characterisation figures

- Date: 11/23/2026
- Checked: `results/tables/descriptor_families.csv`, `results/figures/descriptor_correlation.png`,
  `results/figures/tep_skew.png`.
- Decision: APPROVE W/ NOTES
    NOTES: A few labels were boarderline between families.  Ultimately, I labeled pyramidalisation as steric, vmin_r as electronic, 
           and volume, surface area and sphericity as steric.

## Step 16: the gate

- Date: 11/23/2026
- Result: 8 of 11 primary datasets pass (threshold within 0.11, MCC within 0.03). Thresholds reproduce exactly on 9 of 11. VII 
    reproduces under the SI's documented Buchwald exclusion (MCC 0.40 vs 0.43); IX reproduces exactly with PCy₃ or PCyp₃ removed; 
    VIII matches on threshold but not MCC (0.705 vs 0.75), unexplained.
    Rows of VIII (Table S10) and IX (Table S11) hand-checked against the SI: all match.
- Decision: proceed with all eleven datasets as extracted. The mismatches lie in the published Table S8, not the extracted data. 
    H1 is also reported excluding VII, VIII and IX (see deviation log).