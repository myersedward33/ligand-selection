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

- Date:
- Result from `results/tables/baseline_reproduction.csv`:
- Decision, with reasons (if proceeding despite a failed rule, also add a deviation-log entry):
