# Who Would We Miss?

## A Bayesian Climate Shock-to-Action Targeting Analysis Using SIHBS 2022

This repository contains a reproducible mini-project using Somalia Integrated Household Budget Survey (SIHBS) 2022 microdata to ask a decision-facing question:

> If emergency support can reach only a limited share of households, which targeting rule misses the fewest high-risk unreached households?

The analysis defines **high-risk unreached households** as households that:

1. experienced a climate/livelihood shock,  
2. experienced moderate/severe food insecurity or erosive coping, and  
3. reported no formal support from government, local NGO, or international organization.

The model is a **Bayesian hierarchical logistic regression** with region-level random intercepts, fitted with variational Bayes.

## Headline findings

- Analysis sample: **7,212 households** across **17 regions**.
- Weighted climate/livelihood shock exposure: **71.9%**.
- Weighted moderate/severe food insecurity proxy: **44.1%**.
- Weighted formal support after shock: **3.8%**.
- Weighted high-risk unreached outcome: **40.1%**.
- At **20% support coverage**, Bayesian risk targeting reached **34.0%** of high-risk unreached households, compared with **27.8%** under climate/livelihood shock-only targeting and **19.5%** under poverty-only targeting.

## Repository contents

```text
scripts/                       Reproducible analysis scripts
figures/                       LinkedIn/GitHub-ready figures
outputs/                       Aggregated public-safe results
data/metadata/                 SIHBS module inventory
data/processed/                Schema only; no household microdata included
reports/                       Mini-project report
docs/                          Methodology and LinkedIn post drafts
```

## Data note

Raw SIHBS microdata are **not included** in this public repository. To reproduce the analysis, place the official `.dta` files in `data/raw/` with the original filenames, then run:

```bash
python scripts/00_run_all.py
```

Household-level predictions are intentionally excluded from the public package.
