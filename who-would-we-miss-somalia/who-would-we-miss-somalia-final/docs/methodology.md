# Methodology

## Outcome

```text
high_risk_unreached =
  climate_livelihood_shock == 1
  AND (moderate_severe_food_insecurity == 1 OR erosive_coping == 1)
  AND formal_support_received == 0
```

Climate/livelihood shock includes drought or severe water shortage, large rise in food prices, livestock death, floods, and crop/agricultural price shocks.

Moderate/severe food insecurity is approximated from the eight food-security experience items using a raw score threshold of at least 4 affirmative responses. This is a transparent analytical proxy, not an officially calibrated FIES prevalence estimate.

Formal support includes help from international organizations, local NGOs, or government. Religious institutions are kept separate in the broader institutional-support variable.

## Bayesian model

The model is a Bayesian hierarchical logistic regression:

```text
logit(P(high_risk_unreached_i)) = beta_0 + beta X_i + u_region[i]
```

where `u_region` is a region-level random intercept.

Predictors include shock type, residence type, household size, female-headed status, poverty, IDP status, WASH deprivation, livestock ownership, crop activity, business activity, and head literacy.

## Targeting simulation

The simulation compares Bayesian risk targeting with climate/livelihood shock-only, drought-only, poverty-only, region-burden, direct-needs benchmark, and random targeting across support coverage levels from 5% to 50%.

## Limitations

- The Bayesian model is fitted unweighted because the variational Bayesian mixed model implementation used here does not natively support survey weights. Descriptive and targeting summaries are weighted.
- The food insecurity measure is an analytical proxy, not an official calibrated FIES estimate.
- The targeting simulation is a proof-of-concept for decision analytics, not an operational beneficiary-selection tool.
- Raw microdata and household-level prediction files should not be committed publicly.
